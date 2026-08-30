"""
Posts each .txt file in ./docs_raw/ to the data-service /documents endpoint,
which chunks it, embeds each chunk via ai-service, and stores it.

Run this with the stack up (docker compose up) and docs_raw/ already
populated by fetch_postgres_docs.py:

    python scripts/ingest_docs.py
"""

import json
import urllib.request
from pathlib import Path

DATA_SERVICE_URL = "http://localhost:8001"
DOCS_DIR = Path(__file__).parent.parent / "docs_raw"


def ingest_file(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    payload = {
        "title": path.stem,
        "source": f"postgresql-docs/{path.name}",
        "content": content,
    }
    req = urllib.request.Request(
        f"{DATA_SERVICE_URL}/documents",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    print(f"{path.name}: {result}")


def main() -> None:
    files = sorted(DOCS_DIR.glob("*.txt"))
    if not files:
        print("No files found in docs_raw/ — run fetch_postgres_docs.py first")
        return

    for path in files:
        ingest_file(path)


if __name__ == "__main__":
    main()
