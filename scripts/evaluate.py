"""Evaluate a checkpoint: validation loss, perplexity, bits/byte, and baselines."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from vibe.baselines import BigramModel, UnigramModel, baseline_val_loss
from vibe.data import open_split
from vibe.generate import load_model
from vibe.train import environment_info, evaluate

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed" / "poc")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = whole validation set")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model = load_model(args.checkpoint, args.device)
    cfg = model.cfg
    val_stream, meta = open_split(args.data_dir, "val")
    train_stream, _ = open_split(args.data_dir, "train")
    tpb = meta["val"]["tokens"] / meta["val"]["bytes"] if meta["val"].get("bytes") else None
    cap = args.max_batches or None

    model_loss = evaluate(model, val_stream, args.seq_len, args.batch_size, cap,
                          torch.device(args.device), "fp32")
    train_tokens = train_stream.read(0, train_stream.total)
    uni = UnigramModel(train_tokens, cfg.vocab_size)
    bi, lam = BigramModel.fit(train_tokens, cfg.vocab_size)
    uni_loss = baseline_val_loss(uni, val_stream, args.seq_len, args.batch_size, cap)
    bi_loss = baseline_val_loss(bi, val_stream, args.seq_len, args.batch_size, cap)

    def row(name: str, loss: float) -> dict:
        r = {"name": name, "val_loss": loss, "perplexity": math.exp(loss)}
        if tpb:
            r["bits_per_byte"] = loss / math.log(2) * tpb
        return r

    rows = [row("uniform (knows nothing)", math.log(cfg.vocab_size)),
            row("unigram", uni_loss), row(f"bigram (lam={lam})", bi_loss),
            row("transformer", model_loss)]

    print(f"\n{'model':<26}{'loss':>8}{'perplexity':>12}{'bits/byte':>11}")
    for r in rows:
        bpb = f"{r['bits_per_byte']:.3f}" if "bits_per_byte" in r else "-"
        print(f"{r['name']:<26}{r['val_loss']:>8.4f}{r['perplexity']:>12.1f}{bpb:>11}")
    print(f"\nTransformer vs bigram: {(1 - model_loss / bi_loss) * 100:.1f}% lower loss")

    info = {"checkpoint": str(args.checkpoint), "n_params": model.num_parameters(),
            "seq_len": args.seq_len, "val_tokens": meta["val"]["tokens"],
            "dataset_version": meta.get("dataset_version"), "results": rows,
            "env": environment_info()}
    metrics = args.checkpoint.parent / "metrics.jsonl"
    if metrics.exists():
        trains = [json.loads(l) for l in metrics.read_text(encoding="utf-8").splitlines()
                  if '"type": "train"' in l]
        if trains:
            info["train_steps"] = trains[-1]["step"]
            info["train_tokens_seen"] = trains[-1]["tokens_seen"]

    out = args.out or ROOT / "results" / f"{args.checkpoint.resolve().parent.name}-eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())