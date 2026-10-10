"""CPU speed and memory benchmark (docs 005 and 006), FP32 in PyTorch.

Trained checkpoint:  python scripts/benchmark_cpu.py --checkpoint runs/tiny-1500/model_final.pt
Any config (random weights; speed does not depend on weights):
                     python scripts/benchmark_cpu.py --config configs/50m.yaml
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import psutil
import torch

from vibe.bench import cache_bytes, param_bytes, peak_mb, rss_mb, time_decode, time_prefill
from vibe.config import kv_cache_bytes_per_token, load_config
from vibe.generate import load_model
from vibe.model import Transformer
from vibe.train import environment_info

ROOT = Path(__file__).resolve().parents[1]


def ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "tiny.yaml")
    ap.add_argument("--threads", type=str, default=None, help="e.g. 1,2,4")
    ap.add_argument("--prefill-lens", type=str, default=None)
    ap.add_argument("--decode-contexts", type=str, default=None)
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    default_threads = torch.get_num_threads()
    base_rss = rss_mb()
    if args.checkpoint:
        model = load_model(args.checkpoint, "cpu")
        label = args.checkpoint.resolve().parent.name
    else:
        cfg = load_config(args.config)
        torch.manual_seed(0)
        model = Transformer(cfg).eval()
        label = f"{cfg.name}-random-init"
    cfg = model.cfg
    after_load = rss_mb()
    M = cfg.max_seq_len

    prefill_lens = ints(args.prefill_lens) if args.prefill_lens else sorted({min(128, M), min(512, M)})
    decode_ctx = ints(args.decode_contexts) if args.decode_contexts else sorted({16, M // 4, M // 2})
    thread_list = ints(args.threads) if args.threads else sorted(
        {1, default_threads, os.cpu_count() or default_threads}
    )

    print(f"Benchmark: {label} | {model.num_parameters():,} params | FP32 | context {M}")
    print(f"Logical CPUs: {os.cpu_count()} | torch default threads: {default_threads}")
    print(f"Weights in memory: {param_bytes(model) / 1e6:.1f} MB "
          f"(process grew by {after_load - base_rss:.0f} MB on load)")

    results: dict = {}
    for n in thread_list:
        torch.set_num_threads(n)
        r = {"prefill": {}, "decode": {}}
        print(f"\n--- {n} thread(s) ---")
        for length in prefill_lens:
            secs, _ = time_prefill(model, length, reps=args.reps if length <= 256 else max(1, args.reps - 1))
            r["prefill"][length] = {"seconds": secs, "tokens_per_s": length / secs}
            print(f"  prefill {length:>5} tokens: {secs:7.3f} s  ({length / secs:8.1f} tok/s)")
        for ctx in decode_ctx:
            rate = time_decode(model, ctx, steps=args.decode_steps, reps=args.reps)
            r["decode"][ctx] = rate
            print(f"  decode at context {ctx:>5}: {rate:8.1f} tok/s")
        results[n] = r

    torch.set_num_threads(default_threads)
    secs, caches = time_prefill(model, M, reps=1)
    measured = cache_bytes(caches)
    formula = kv_cache_bytes_per_token(cfg, bytes_per_value=4) * M
    mem = {
        "baseline_process_mb": base_rss,
        "after_model_load_mb": after_load,
        "full_context_prefill_seconds": secs,
        "kv_cache_fp32_mib_measured": measured / 1048576,
        "kv_cache_fp32_mib_formula": formula / 1048576,
        "peak_process_mb": peak_mb(),
    }
    print(f"\n--- memory (full {M}-token context, {default_threads} threads) ---")
    print(f"  full-context prefill   : {secs:.2f} s")
    print(f"  KV cache (FP32)        : {measured / 1048576:.2f} MiB measured, "
          f"{formula / 1048576:.2f} MiB by formula "
          f"({'match' if measured == formula else 'MISMATCH'})")
    print(f"  process before / after load : {base_rss:.0f} MB / {after_load:.0f} MB")
    print(f"  peak process memory    : {mem['peak_process_mb']:.0f} MB")

    env = environment_info()
    env.update({
        "logical_cpus": os.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "total_ram_mb": psutil.virtual_memory().total / 1e6,
    })
    out = ROOT / "results" / f"benchmark-cpu-{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"label": label, "n_params": model.num_parameters(),
                               "precision": "fp32", "threads": results,
                               "memory": mem, "env": env}, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())