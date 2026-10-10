"""Tokenize the PoC corpus into uint16 shards, split by document."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from vibe.data import tokenize_documents, write_shards
from vibe.tokenizer import load_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "data" / "tokenizer" / "poc-4096.json")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "gutenberg")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "poc")
    ap.add_argument("--shard-tokens", type=int, default=2_000_000)
    args = ap.parse_args()

    files = sorted(args.data_dir.glob("*.txt"), key=lambda p: int(p.stem))
    if not files:
        print("No text files found. Run scripts/download_poc_data.py first.")
        return 1

    # Same rule as the tokenizer report: every 5th book is validation.
    val_files = files[::5]
    train_files = [f for i, f in enumerate(files) if i % 5 != 0]

    tok = load_tokenizer(args.tokenizer)
    t0 = time.time()
    train_tokens = tokenize_documents(tok, [f.read_text(encoding="utf-8") for f in train_files])
    val_tokens = tokenize_documents(tok, [f.read_text(encoding="utf-8") for f in val_files])
    dt = time.time() - t0

    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*.bin"):
        old.unlink()
    train_names = write_shards(train_tokens, args.out, "train", args.shard_tokens)
    val_names = write_shards(val_tokens, args.out, "val", args.shard_tokens)

    meta = {
        "dataset_version": "poc-v1",
        "tokenizer_file": args.tokenizer.name,
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "vocab_size": tok.get_vocab_size(),
        "dtype": "uint16",
        "split_rule": "sorted by book id; every 5th book (index % 5 == 0) is validation",
        "train": {"documents": len(train_files), "tokens": int(len(train_tokens)), "shards": train_names},
        "val": {"documents": len(val_files), "tokens": int(len(val_tokens)), "shards": val_names},
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Tokenized in {dt:.1f}s")
    print(f"train: {len(train_files)} books, {len(train_tokens):,} tokens, {len(train_names)} shard(s)")
    print(f"val  : {len(val_files)} books, {len(val_tokens):,} tokens, {len(val_names)} shard(s)")
    seq_len, batch = 256, 16
    n_seq = (len(train_tokens) - 1) // seq_len
    print(f"At seq_len {seq_len}, batch {batch}: {n_seq // batch:,} batches per epoch "
          f"({batch * seq_len:,} tokens per batch)")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())