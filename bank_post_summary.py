import argparse
import json
import hashlib
import base64
import requests # type: ignore

from Crypto.PublicKey import RSA # type: ignore
from Crypto.Signature import pkcs1_15 # type: ignore
from Crypto.Hash import SHA256 # type: ignore

def load_summary(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def body_bytes_from_envelope(envelope: dict) -> bytes:
    # Compact separators to ensure deterministic byte string before signing
    return json.dumps(envelope, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

def compute_thumbprint(pub_pem: bytes) -> str:
    pub = RSA.import_key(pub_pem)
    der = pub.export_key(format="DER")
    return hashlib.sha256(der).hexdigest()

def sign_rs256(priv_pem: bytes, body_bytes: bytes) -> bytes:
    key = RSA.import_key(priv_pem)
    h = SHA256.new(body_bytes)
    return pkcs1_15.new(key).sign(h)

def main():
    ap = argparse.ArgumentParser(description="Simulate bank pushing a summary to local CBA receiver.")
    ap.add_argument("--summary", default="/Users/apple/Downloads/cba_local_sim_swagger/out_summaries7/10001-42d67b34-35e5-45d9-8fd1-a856bd0fcb09.json",
                help="Path to summary JSON.")
    ap.add_argument("--url", default="http://localhost:8080/api/v1/data", help="Local CBA receiver endpoint.")
    ap.add_argument("--pubkey", default="my_public_key.pem", help="Bank public key (PEM).")
    ap.add_argument("--privkey", default="my_private_key.pem", help="Bank private key (PEM).")
    ap.add_argument("--message-type", type=int, default=1, help="MessageType code to send.")
    ap.add_argument("--sign-encoding", choices=["hex","base64"], default="hex", help="Header encoding for signature.")
    args = ap.parse_args()

    summary = load_summary(args.summary)
    message_id = summary.get("external_id") or summary.get("MessageId") or summary.get("id")
    if not message_id:
        raise SystemExit("Summary JSON must include 'external_id' (or 'MessageId'/'id').")

    envelope = {
        "MessageId": message_id,
        "MessageType": args.message_type,
        "Payload": summary
    }
    body_bytes = body_bytes_from_envelope(envelope)

    with open(args.pubkey, "rb") as f:
        pub_pem = f.read()
    with open(args.privkey, "rb") as f:
        priv_pem = f.read()

    thumb = compute_thumbprint(pub_pem)
    sig = sign_rs256(priv_pem, body_bytes)
    x_client_sign = sig.hex() if args.sign_encoding == "hex" else base64.b64encode(sig).decode("ascii")

    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(body_bytes)),
        "X-Client-Id": thumb,
        "X-Client-Sign": x_client_sign,
    }

    print(f"POST {args.url}")
    r = requests.post(args.url, data=body_bytes, headers=headers, timeout=20)
    print("Status:", r.status_code)
    print("Body:", r.text)

if __name__ == "__main__":
    main()
