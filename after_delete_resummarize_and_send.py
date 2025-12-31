import os
from typing import List, Set

import requests  # type: ignore

import sum_jsons2  # our modified summarizer
from bank_post_summary import (
    load_summary,
    body_bytes_from_envelope,
    compute_thumbprint,
    sign_rs256,
)

OUTPUT_DIR = sum_jsons2.OUTPUT_DIR  # "out_summaries7"

CBA_URL = "http://localhost:8080/api/v1/data"      # CHANGE if needed
PUBKEY_PATH = "/Users/apple/Downloads/cba_local_sim_swagger/my_public_key.pem"         # CHANGE
PRIVKEY_PATH = "/Users/apple/Downloads/cba_local_sim_swagger/my_private_key.pem"       # CHANGE
MESSAGE_TYPE = 1


def list_summary_files() -> Set[str]:
    return {
        f
        for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".json") and f != "summaries.jsonl"
    }


def find_new_summary_files(before: Set[str]) -> List[str]:
    after = list_summary_files()
    new_files = sorted(after - before)
    return new_files


def send_specific_summaries(files: List[str]) -> None:
    if not files:
        print("ℹ️ No new summaries to send.")
        return

    with open(PUBKEY_PATH, "rb") as f:
        pub_pem = f.read()
    with open(PRIVKEY_PATH, "rb") as f:
        priv_pem = f.read()

    # Use the *same* thumbprint logic as bank_post_summary
    thumb = compute_thumbprint(pub_pem)

    for fname in files:
        path = os.path.join(OUTPUT_DIR, fname)
        summary = load_summary(path)

        message_id = (
            summary.get("external_id")
            or summary.get("MessageId")
            or summary.get("id")
        )
        if not message_id:
            print(f"⚠️ {fname}: no external_id/MessageId/id – skipping.")
            continue

        envelope = {
            "MessageId": message_id,
            "MessageType": MESSAGE_TYPE,
            "Payload": summary,
        }

        body_bytes = body_bytes_from_envelope(envelope)

        # Use the same RS256 signing implementation as bank_post_summary
        sig_bytes = sign_rs256(priv_pem, body_bytes)
        x_client_sign = sig_bytes.hex()

        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_bytes)),
            "X-Client-Id": thumb,
            "X-Client-Sign": x_client_sign,
        }

        print(f"📤 POST {CBA_URL}  (summary={fname})")
        try:
            r = requests.post(CBA_URL, data=body_bytes, headers=headers, timeout=20)
            print(f"   → Status: {r.status_code}")
            print(f"   → Body: {r.text}")
        except requests.RequestException as e:
            print(f"   ❌ Error sending {fname}: {e}")

    print("✅ Done sending new summaries.")


def main() -> None:
    before = list_summary_files()

    print("⚙️ Running summarizer on orphan rows...")
    sum_jsons2.summarize_file()

    new_files = find_new_summary_files(before)
    print(f"🆕 New summaries: {new_files}")

    send_specific_summaries(new_files)


if __name__ == "__main__":
    main()
