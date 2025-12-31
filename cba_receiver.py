import os
import json
import hashlib
import base64
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, Request, Header, HTTPException, Query  # type: ignore
from fastapi.responses import JSONResponse, HTMLResponse  # type: ignore
from pydantic import BaseModel, Field  # type: ignore
import requests  # type: ignore

# Crypto (PyCryptodome)
from Crypto.PublicKey import RSA  # type: ignore
from Crypto.Hash import SHA256  # type: ignore
from Crypto.Signature import pkcs1_15  # type: ignore

app = FastAPI(
    title="Local CBA Receiver (Mock)",
    version="1.2.0",
    description="""
A tiny FastAPI app to simulate the Central Bank receiver.
- **POST /api/v1/data** — accept a signed summary from a "bank".
- **GET  /api/v1/data** — list received messageIds.
- **GET  /api/v1/data/{message_id}** — fetch the stored envelope for a message.
- **GET  /summary/{message_id}** — proxy to the bank's LIVE summary API.
- **GET  /details/{message_id}** — proxy to your existing details API (GET /details/{id}).
- **GET  /** — simple HTML index with links.
""",
)

RECEIVED_DIR = os.environ.get("RECEIVED_DIR", "received")
os.makedirs(RECEIVED_DIR, exist_ok=True)

# Where the "CBA" will call to fetch detailed rows after you "click"
BANK_DETAILS_BASE = os.environ.get("BANK_DETAILS_BASE", "http://localhost:8000")
# Where the "CBA" will call to fetch the LIVE summary
# (this should point to the bank summary service; change the default as needed)
BANK_SUMMARY_BASE = os.environ.get("BANK_SUMMARY_BASE", "http://localhost:8000")

# Signature verification
DISABLE_SIG = os.environ.get("DISABLE_SIG", "0") == "1"
PUB_KEY_PATH = os.environ.get("PUB_KEY_PATH", "my_public_key.pem")


# -----------------------------
# Pydantic models (for Swagger)
# -----------------------------
class Envelope(BaseModel):
    MessageId: str = Field(..., description="Use the summary's external_id.")
    MessageType: int = Field(1, description="Arbitrary code; 1 used for FX summary in this mock.")
    Payload: Dict[str, Any] = Field(..., description="Your summary JSON object.")


class ReceiveResponse(BaseModel):
    status: str = "ok"
    messageId: str


class StoredMessage(BaseModel):
    messageId: str
    path: str


# -----------------------------
# Helpers
# -----------------------------
def _hex_or_b64_to_bytes(s: str) -> bytes:
    s = s.strip()
    # try hex
    try:
        return bytes.fromhex(s)
    except Exception:
        pass
    # try base64 (std + urlsafe)
    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            return decoder(s + "===")  # tolerant padding
        except Exception:
            continue
    raise ValueError("Signature header is neither hex nor base64.")


def verify_headers_and_body(x_client_id: str, x_client_sign: str, body_bytes: bytes):
    if DISABLE_SIG:
        return

    # Load public key
    try:
        with open(PUB_KEY_PATH, "rb") as f:
            pub_pem = f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Public key not found for signature verification.")

    pub_key = RSA.import_key(pub_pem)
    pub_der = pub_key.export_key(format="DER")
    thumbprint = hashlib.sha256(pub_der).hexdigest()

    if thumbprint.lower() != (x_client_id or "").lower():
        raise HTTPException(status_code=401, detail="X-Client-Id does not match public key thumbprint.")

    sig_bytes = _hex_or_b64_to_bytes(x_client_sign or "")

    h = SHA256.new(body_bytes)
    try:
        pkcs1_15.new(pub_key).verify(h, sig_bytes)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid X-Client-Sign signature over request body.")


def store_message(envelope: Dict[str, Any], raw_body: bytes):
    msg_id = envelope.get("MessageId") or envelope.get("messageId")
    if not msg_id:
        raise HTTPException(status_code=400, detail="MessageId is required.")
    # Persist both envelope and payload for demo (overwrites if same MessageId is resent)
    with open(os.path.join(RECEIVED_DIR, f"{msg_id}.json"), "w", encoding="utf-8") as f:
        json.dump(envelope, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RECEIVED_DIR, f"{msg_id}.raw"), "wb") as f:
        f.write(raw_body)
    return msg_id


# -----------------------------
# API routes
# -----------------------------
@app.post("/api/v1/data", response_model=ReceiveResponse, tags=["Data intake"])
async def receive_data(
    envelope: Envelope,
    request: Request,
    x_client_id: Optional[str] = Header(None, alias="X-Client-Id"),
    x_client_sign: Optional[str] = Header(None, alias="X-Client-Sign"),
):
    # Verify signature over exact request body bytes (not the parsed envelope)
    body_bytes = await request.body()
    if not body_bytes:
        raise HTTPException(status_code=400, detail="Empty body.")

    try:
        verify_headers_and_body(x_client_id or "", x_client_sign or "", body_bytes)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Signature processing error: {e}")

    # Store
    msg_id = store_message(envelope.dict(), body_bytes)
    return ReceiveResponse(status="ok", messageId=msg_id)


@app.get("/api/v1/data", response_model=List[StoredMessage], tags=["Data intake"])
def list_received() -> List[StoredMessage]:
    files = [f for f in os.listdir(RECEIVED_DIR) if f.endswith(".json")]
    out: List[StoredMessage] = []
    for f in sorted(files):
        msg_id = f[:-5]
        out.append(StoredMessage(messageId=msg_id, path=f"/api/v1/data/{msg_id}"))
    return out


@app.get("/api/v1/data/{message_id}", response_model=Envelope, tags=["Data intake"])
def get_envelope(message_id: str) -> Envelope:
    p = os.path.join(RECEIVED_DIR, f"{message_id}.json")
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="Not found")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Envelope(**data)


@app.get("/", response_class=HTMLResponse, tags=["UI"])
def index():
    files = [f for f in os.listdir(RECEIVED_DIR) if f.endswith(".json")]
    rows: List[str] = []
    for f in sorted(files):
        msg_id = f[:-5]
        rows.append(
            f"<tr>"
            f"<td>{msg_id}</td>"
            f"<td><a href='/api/v1/data/{msg_id}'>envelope (stored)</a></td>"
            f"<td><a href='/summary/{msg_id}'>summary (live)</a></td>"
            f"<td><a href='/details/{msg_id}'>details (proxy to bank)</a></td>"
            f"</tr>"
        )
    html = f"""
    <html>
      <head><title>Local CBA Receiver</title></head>
      <body>
        <h2>Received Messages</h2>
        <table border="1" cellspacing="0" cellpadding="6">
          <tr>
            <th>MessageId</th>
            <th>Envelope (stored)</th>
            <th>Summary (live)</th>
            <th>Details (proxy to bank)</th>
          </tr>
          {''.join(rows) if rows else '<tr><td colspan="4">No messages yet</td></tr>'}
        </table>
        <p>Swagger UI: <a href="/docs">/docs</a> • ReDoc: <a href="/redoc">/redoc</a></p>
        <p>
          BANK_DETAILS_BASE = {BANK_DETAILS_BASE}<br/>
          BANK_SUMMARY_BASE = {BANK_SUMMARY_BASE}
        </p>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/summary/{message_id}", tags=["Proxy to Bank"])
def proxy_summary(message_id: str):
    """
    Proxy to the bank's LIVE summary API.

    Calls:
        GET {BANK_SUMMARY_BASE}/summary/{message_id}

    so BANK_SUMMARY_BASE should be the base URL of your bank summary service,
    e.g. http://localhost:8002 (then this calls http://localhost:8002/summary/{id}).
    """
    url = f"{BANK_SUMMARY_BASE.rstrip('/')}/summary/{message_id}"
    try:
        r = requests.get(url, timeout=15)
        if r.headers.get("content-type", "").startswith("application/json"):
            return JSONResponse(status_code=r.status_code, content=r.json())
        return JSONResponse(status_code=r.status_code, content={"text": r.text})
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error calling summary API at {url}: {e}")


@app.get("/details/{message_id}", tags=["Proxy to Bank"])
def proxy_details(message_id: str):
    """
    Proxy to the bank's details API.

    Calls:
        GET {BANK_DETAILS_BASE}/details/{message_id}
    """
    url = f"{BANK_DETAILS_BASE.rstrip('/')}/details/{message_id}"
    try:
        r = requests.get(url, timeout=15)
        if r.headers.get("content-type", "").startswith("application/json"):
            return JSONResponse(status_code=r.status_code, content=r.json())
        return JSONResponse(status_code=r.status_code, content={"text": r.text})
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error calling details API at {url}: {e}")


@app.delete("/summary/{message_id}", tags=["Proxy to Bank"])
def delete_summary_via_bank(message_id: str):
    """
    Ask the bank to delete a summary and detach its detailed rows.

    Calls:
        DELETE {BANK_SUMMARY_BASE}/summary/{message_id}
    """
    url = f"{BANK_SUMMARY_BASE.rstrip('/')}/summary/{message_id}"
    try:
        r = requests.delete(url, timeout=15)
        # Try JSON first, fall back to plain text
        try:
            content = r.json()
        except ValueError:
            content = {"text": r.text}

        return JSONResponse(status_code=r.status_code, content=content)
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"Error calling bank DELETE summary API at {url}: {e}",
        )
