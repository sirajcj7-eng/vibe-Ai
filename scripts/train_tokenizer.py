"""Train the PoC tokenizer and report quality metrics (see doc 003)."""
from __future__ import annotations

import argparse
import time
import unicodedata
from collections import Counter
from pathlib import Path

from vibe.tokenizer import N_SPECIAL, decode_ids, encode_text, train_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "gutenberg")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ROOT / "data" / "tokenizer" / f"poc-{args.vocab_size}.json"

    files = sorted(args.data_dir.glob("*.txt"), key=lambda p: int(p.stem))
    if not files:
        print("No text files found. Run scripts/download_poc_data.py first.")
        return 1

    # Hold out every 5th book (by document) so metrics are on unseen text.
    held_out = files[::5]
    train_files = [f for i, f in enumerate(files) if i % 5 != 0]
    print(f"{len(train_files)} training books, {len(held_out)} held-out books")

    t0 = time.time()
    tok = train_tokenizer(train_files, args.vocab_size, out)
    print(f"Trained in {time.time() - t0:.1f}s -> {out} ({out.stat().st_size / 1024:.0f} KB)")
    vocab = tok.get_vocab_size()
    print(f"Vocab size: {vocab}")

    # Held-out metrics
    text = "\n".join(f.read_text(encoding="utf-8") for f in held_out)
    n_bytes = len(text.encode("utf-8"))
    t0 = time.time()
    ids = encode_text(tok, text)
    dt = time.time() - t0
    n_words = len(text.split())
    print("\nHeld-out text:")
    print(f"  bytes per token : {n_bytes / len(ids):.3f}")
    print(f"  tokens per word : {len(ids) / n_words:.3f}")
    print(f"  encode speed    : {n_bytes / dt / 1e6:.2f} MB/s")

    ok = decode_ids(tok, ids) == unicodedata.normalize("NFC", text)
    print(f"  lossless round-trip: {'PASS' if ok else 'FAIL'}")

    # Wasted vocabulary on the training books
    counts: Counter = Counter()
    for f in train_files:
        counts.update(encode_text(tok, f.read_text(encoding="utf-8")))
    learnable = range(N_SPECIAL, vocab)
    never = sum(1 for i in learnable if counts[i] == 0)
    rare = sum(1 for i in learnable if 0 < counts[i] < 5)
    print(f"\nNon-special tokens never used in training books: {never}")
    print(f"Non-special tokens used fewer than 5 times     : {rare}")

    sample = "The year 2026 had 365 days."
    print(f"\nExample: {sample!r}")
    print("  tokens:", tok.encode(sample).tokens)
    print("  (the 'Ġ' symbol means a leading space)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())