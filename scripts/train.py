"""Train a model on tokenized shards.

Example (quick smoke run):
    python scripts/train.py --steps 100 --batch-size 8 --seq-len 128 --out runs/smoke
Resume a stopped run:
    python scripts/train.py --steps 100 --batch-size 8 --seq-len 128 --out runs/smoke --resume
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vibe.config import load_config
from vibe.data import open_split
from vibe.train import TrainConfig, train

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", type=Path, default=ROOT / "configs" / "tiny.yaml")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed" / "poc")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "tiny-poc")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip-norm", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-batches", type=int, default=20, help="0 = whole validation set")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--threads", type=int, default=0, help="0 = PyTorch default")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-at", type=int, default=None)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = load_config(args.model_config)
    train_stream, meta = open_split(args.data_dir, "train")
    val_stream, _ = open_split(args.data_dir, "val")

    val = meta["val"]
    tokens_per_byte = None
    if val.get("bytes"):
        tokens_per_byte = val["tokens"] / val["bytes"]
    else:
        print("Note: meta.json has no byte counts; re-run scripts/prepare_data.py "
              "to get bits/byte. Reporting loss only.")

    t = TrainConfig(
        steps=args.steps, batch_size=args.batch_size, grad_accum=args.grad_accum,
        seq_len=args.seq_len, lr=args.lr, min_lr_frac=args.min_lr_frac,
        warmup_steps=args.warmup_steps, decay_frac=args.decay_frac,
        weight_decay=args.weight_decay, clip_norm=args.clip_norm,
        eval_every=args.eval_every, eval_batches=args.eval_batches,
        log_every=args.log_every, ckpt_every=args.ckpt_every, seed=args.seed,
        precision=args.precision, resume=args.resume, stop_at=args.stop_at,
    )
    total = t.steps * t.batch_size * t.grad_accum * t.seq_len
    print(f"Planned training tokens: {total:,} "
          f"({total / train_stream.total:.2f} epochs of {train_stream.total:,})")

    train(model_cfg, t, train_stream, val_stream, args.out,
          device=device, val_tokens_per_byte=tokens_per_byte)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())