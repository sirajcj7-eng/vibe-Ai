"""GPU training-throughput benchmark (Phase 6.9). Run on Kaggle or Colab.

    python scripts/benchmark_gpu.py --config configs/50m.yaml

Uses random tokens (throughput does not depend on the data) and the same
optimizer / mixed-precision code path as scripts/train.py.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from vibe.config import load_config
from vibe.model import Transformer
from vibe.train import TrainConfig, autocast_ctx, build_optimizer, environment_info

ROOT = Path(__file__).resolve().parents[1]


def ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def run_case(model, optimizer, scaler, batch, seq, steps, warmup, precision, device) -> dict:
    g = torch.Generator().manual_seed(0)
    vocab = model.cfg.vocab_size
    x = torch.randint(0, vocab, (batch, seq), generator=g).to(device)
    y = torch.randint(0, vocab, (batch, seq), generator=g).to(device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device, precision):
            _, loss, _ = model(x, targets=y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        return loss

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        loss = step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {
        "tok_per_s": batch * seq * steps / dt,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "loss": float(loss),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "50m.yaml")
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--seq-lens", default="512,1024,2048")
    ap.add_argument("--batch-sizes", default="2,4,8,16")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA GPU found. In Kaggle: Settings -> Accelerator -> GPU.")
        return 1
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        print("This GPU has no native bf16 (T4/P100). Use --precision fp16.")
        return 1

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    cfg = load_config(args.config)
    torch.manual_seed(0)
    model = Transformer(cfg).to(device)
    optimizer = build_optimizer(model, TrainConfig())
    scaler = torch.amp.GradScaler("cuda") if args.precision == "fp16" else None

    print(f"GPU: {props.name} | {props.total_memory / 2**30:.1f} GiB | "
          f"{torch.cuda.device_count()} device(s), using #0")
    print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | {args.precision}")
    print(f"Model: {cfg.name}, {model.num_parameters():,} params\n")

    rows = []
    for seq in ints(args.seq_lens):
        if seq > cfg.max_seq_len:
            continue
        for batch in ints(args.batch_sizes):
            try:
                r = run_case(model, optimizer, scaler, batch, seq,
                             args.steps, args.warmup, args.precision, device)
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                print(f"  seq {seq:>5} | batch {batch:>3} | out of memory")
                rows.append({"seq": seq, "batch": batch, "oom": True})
                continue
            r.update(seq=seq, batch=batch)
            rows.append(r)
            print(f"  seq {seq:>5} | batch {batch:>3} | {r['tok_per_s']:>9,.0f} tok/s | "
                  f"peak {r['peak_gib']:.2f} GiB")

    ok = [r for r in rows if not r.get("oom")]
    summary = {}
    if ok:
        best = max(ok, key=lambda r: r["tok_per_s"])
        per_micro = best["batch"] * best["seq"]
        accum = max(1, round(262_144 / per_micro))
        print(f"\nBest: seq {best['seq']}, batch {best['batch']} -> "
              f"{best['tok_per_s']:,.0f} tok/s (doc 002 assumed 17,000)")
        print(f"For 262,144-token steps use --grad-accum {accum} "
              f"({per_micro:,} tokens per micro-batch)")
        for tokens in (1e9, 5e9):
            print(f"  {tokens / 1e9:.0f}B tokens: {tokens / best['tok_per_s'] / 3600:,.1f} GPU-hours")
        summary = {"best": best, "grad_accum_for_262k": accum}

    name = props.name.replace(" ", "-")
    out = ROOT / "results" / f"benchmark-gpu-{name}-{cfg.name}-{args.precision}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gpu": props.name, "gpu_memory_gib": props.total_memory / 2**30,
                               "precision": args.precision, "n_params": model.num_parameters(),
                               "cases": rows, "summary": summary, "env": environment_info()},
                              indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())