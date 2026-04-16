"""Load and decrypt the MERRIN dataset from HuggingFace.

Usage:
    python load_dataset.py [--output data/questions/MERRIN.jsonl]
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path

from datasets import load_dataset

ENCRYPTED_FIELDS = ["question", "answer", "resources"]


def derive_key(password: str, length: int) -> bytes:
    key = hashlib.sha256(password.encode()).digest()
    return (key * (length // len(key) + 1))[:length]


def decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode("utf-8")


def decrypt_record(record: dict) -> dict:
    """Decrypt encrypted fields in a dataset record."""
    canary = record.get("canary")
    if not canary:
        return record
    out = {}
    for k, v in record.items():
        if k == "canary":
            continue
        if k in ENCRYPTED_FIELDS and isinstance(v, str):
            plaintext = decrypt(v, canary)
            out[k] = json.loads(plaintext) if plaintext.startswith(("[", "{")) else plaintext
        else:
            out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser(description="Load and decrypt the MERRIN dataset")
    parser.add_argument(
        "--output", default="data/questions/MERRIN.jsonl",
        help="Output path for decrypted JSONL file (default: data/questions/MERRIN.jsonl)",
    )
    args = parser.parse_args()

    print("Loading MERRIN dataset from HuggingFace...")
    dataset = load_dataset("HanNight/MERRIN", split="test")
    print(f"Loaded {len(dataset)} records")

    print("Decrypting fields...")
    data = [decrypt_record(record) for record in dataset]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(data)} questions to {output_path}")
    print(f"Example: {data[0]['question'][:100]}...")


if __name__ == "__main__":
    main()
