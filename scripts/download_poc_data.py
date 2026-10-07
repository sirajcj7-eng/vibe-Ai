"""Download a small public-domain corpus (Project Gutenberg) for PoC testing.

Used ONLY to validate the pipeline. It is not part of the 50M training mix.
Texts go to data/raw/gutenberg/ (git-ignored). Provenance is written to
data/manifest.yaml (committed).
"""
from __future__ import annotations

import datetime
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "gutenberg"
MANIFEST = ROOT / "data" / "manifest.yaml"
URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
HEADERS = {"User-Agent": "vibe-ai-poc/0.1 (open research project; pipeline test)"}

BOOK_IDS = [
    1342, 11, 84, 98, 1661, 2701, 74, 1260, 345, 1400, 2600, 174, 76, 215,
    120, 1232, 2554, 768, 158, 161, 1080, 5200, 35, 36, 43, 829, 730, 46,
    2591, 514, 55, 103, 164, 1952,
]


def fetch(book_id: int) -> str:
    req = urllib.request.Request(URL.format(id=book_id), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8-sig", errors="replace")


def strip_boilerplate(text: str) -> str | None:
    """Keep only the book body between the Gutenberg START/END markers."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("*** START OF")), None)
    end = next((i for i, l in enumerate(lines) if l.startswith("*** END OF")), None)
    if start is None or end is None or end <= start:
        return None
    return "\n".join(lines[start + 1 : end]).strip() + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    books = []
    for book_id in BOOK_IDS:
        try:
            raw = fetch(book_id)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"[skip] {book_id}: {e}")
            continue
        body = strip_boilerplate(raw)
        if body is None:
            print(f"[skip] {book_id}: start/end markers not found")
            continue
        m = re.search(r"^Title:\s*(.+?)\s*$", raw, re.M)
        title = m.group(1) if m else "unknown"
        path = OUT_DIR / f"{book_id}.txt"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        size = path.stat().st_size
        books.append(
            {"id": book_id, "title": title, "url": URL.format(id=book_id), "bytes": size}
        )
        print(f"[ok] {book_id}: {title} ({size / 1e6:.2f} MB)")
        time.sleep(1.0)  # be polite to the server

    manifest = {}
    if MANIFEST.exists():
        manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8")) or {}
    manifest["poc_corpus"] = {
        "purpose": "pipeline validation only; not part of the 50M training mix",
        "source": "Project Gutenberg",
        "license_note": (
            "Public domain in the USA; Gutenberg header/footer removed. "
            "Terms outside the USA may differ. Not legal advice."
        ),
        "retrieved": datetime.date.today().isoformat(),
        "books": books,
    }
    MANIFEST.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    total = sum(b["bytes"] for b in books)
    print(f"\nDownloaded {len(books)}/{len(BOOK_IDS)} books, {total / 1e6:.1f} MB total")
    print(f"Manifest written to {MANIFEST}")


if __name__ == "__main__":
    main()