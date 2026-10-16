"""Train candidate tokenizers on the sample and compare compression (Phase 7b.5).

    python scripts/train_real_tokenizer.py --vocab-sizes 8192,16384,32768
"""
from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path

from vibe.sources import SOURCES
from vibe.tokenizer import decode_ids, encode_text, train_tokenizer

ROOT = Path(__file__).resolve().parents[1]
NON_EMBEDDING_PARAMS = 41_959_424  # 50M model, doc 001
D_MODEL = 512


def eval_tokenizer(tok, eval_docs: dict) -> dict:
    rows = {}
    for key, docs in eval_docs.items():
        if not docs:
            continue
        n_tok = sum(len(e.ids) for e in tok.encode_batch(docs))
        n_bytes = sum(len(d.encode("utf-8")) for d in docs)
        n_words = max(1, sum(len(d.split()) for d in docs))
        rows[key] = {"bytes_per_token": n_bytes / n_tok, "tokens_per_word": n_tok / n_words}
    return rows


def roundtrip_failures(tok, eval_docs: dict, per_source: int = 30) -> int:
    bad = 0
    for docs in eval_docs.values():
        for d in docs[:per_source]:
            if decode_ids(tok, encode_text(tok, d)) != unicodedata.normalize("NFC", d):
                bad += 1
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, default=ROOT / "data" / "tok_sample")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "tokenizer")
    ap.add_argument("--vocab-sizes", default="8192,16384,32768")
    ap.add_argument("--holdout", type=float, default=0.05)
    args = ap.parse_args()
    sizes = [int(x) for x in args.vocab_sizes.split(",")]

    work = args.sample_dir / "_train_text"
    work.mkdir(parents=True, exist_ok=True)
    train_files, eval_docs = [], {}
    for p in sorted(args.sample_dir.glob("*.jsonl")):
        docs = [json.loads(line)["text"] for line in open(p, encoding="utf-8")]
        if not docs:
            continue
        cut = len(docs) if len(docs) < 2 else min(len(docs) - 1, int(len(docs) * (1 - args.holdout)))
        cut = max(1, cut)
        (work / f"{p.stem}.txt").write_text("\n\n".join(docs[:cut]), encoding="utf-8")
        train_files.append(work / f"{p.stem}.txt")
        eval_docs[p.stem] = docs[cut:]
    mb = sum(f.stat().st_size for f in train_files) / 1e6
    print(f"Training text: {mb:.0f} MB from {len(train_files)} sources; "
          f"held-out documents: {sum(len(v) for v in eval_docs.values())}", flush=True)

    results = {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for v in sizes:
        t0 = time.time()
        tok = train_tokenizer(train_files, v, args.out_dir / f"commonpile-{v}.json")
        rows = eval_tokenizer(tok, eval_docs)
        bad = roundtrip_failures(tok, eval_docs)
        results[v] = {"seconds": time.time() - t0, "vocab": tok.get_vocab_size(),
                      "rows": rows, "roundtrip_failures": bad}
        print(f"[{v}] trained in {time.time() - t0:.0f}s | vocab {tok.get_vocab_size()} | "
              f"round-trip failures {bad}", flush=True)

    shares = {s.key: s.share for s in SOURCES}
    keys = sorted((k for k in eval_docs if eval_docs[k]), key=lambda k: -shares.get(k, 0))
    print(f"\n{'source':<22}" + "".join(f"{v:>9}" for v in sizes) + "   bytes per token (held-out)")
    for k in keys:
        cells = [f"{results[v]['rows'][k]['bytes_per_token']:>9.2f}" if k in results[v]["rows"]
                 else f"{'-':>9}" for v in sizes]
        print(f"{k:<22}" + "".join(cells))

    summary = {}
    for v in sizes:
        rows = results[v]["rows"]
        num = sum(shares.get(k, 0) * 1000 / r["bytes_per_token"] for k, r in rows.items())
        den = sum(shares.get(k, 0) for k in rows)
        summary[v] = {"tokens_per_1000_bytes": num / den,
                      "embedding_share": v * D_MODEL / (v * D_MODEL + NON_EMBEDDING_PARAMS)}
    base = summary.get(16384, summary[sizes[0]])["tokens_per_1000_bytes"]
    print("\nMix-weighted tokens per 1000 bytes (lower is better), embedding share of a 50M model:")
    for v in sizes:
        s = summary[v]
        print(f"  vocab {v:>6}: {s['tokens_per_1000_bytes']:7.1f} tokens "
              f"({(s['tokens_per_1000_bytes'] / base - 1) * 100:+5.1f}% vs 16k) | "
              f"embeddings {s['embedding_share'] * 100:4.1f}% of parameters")

    out = ROOT / "results" / "tokenizer-compression.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results, "summary": summary}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())