import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vectorstore import ensure_index  # noqa: E402
from ingest import ingest_emails      # noqa: E402

def main() -> None:
    ensure_index()

    emails_path = ROOT / "data" / "emails.json"
    with emails_path.open("r", encoding="utf-8") as f:
        emails = json.load(f)

    result = ingest_emails(emails)
    print(f"Ingested {result['chunks_written']} chunks from {len(emails)} emails.")

if __name__ == "__main__":
    main()
