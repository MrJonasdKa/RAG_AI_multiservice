"""
Pulls a starter set of PostgreSQL reference doc source files (SGML) from
the official Postgres GitHub mirror, strips markup down to plain text,
and writes .txt files to ./docs_raw/ for later ingestion.

No dependencies beyond the standard library.

Run:
    python scripts/fetch_postgres_docs.py
"""

import re
import urllib.request
from pathlib import Path

RAW_BASE = "https://raw.githubusercontent.com/postgres/postgres/master/doc/src/sgml/"

# Starter set — expand this list as you go. Browse available files at
# https://github.com/postgres/postgres/tree/master/doc/src/sgml
PAGES = [
    "query.sgml",
    "ddl.sgml",
    "dml.sgml",
    "indices.sgml",
    "mvcc.sgml",
    "perform.sgml",
]

OUT_DIR = Path(__file__).parent.parent / "docs_raw"

TAG_RE = re.compile(r"<[^>]+>")
ENTITY_RE = re.compile(r"&[a-zA-Z]+;")
WHITESPACE_RE = re.compile(r"\s+")


def strip_sgml(raw: str) -> str:
    text = TAG_RE.sub(" ", raw)
    text = ENTITY_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for page in PAGES:
        url = RAW_BASE + page
        print(f"fetching {url}")
        with urllib.request.urlopen(url) as resp:
            raw = resp.read().decode("utf-8")

        text = strip_sgml(raw)
        out_path = OUT_DIR / page.replace(".sgml", ".txt")
        out_path.write_text(text, encoding="utf-8")
        print(f"  -> {out_path} ({len(text)} chars)")


if __name__ == "__main__":
    main()
