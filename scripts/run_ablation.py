"""Run a one-field hyperparameter sweep with several seeds (Phase 6.10).

Examples (Kaggle, GPU):
  python scripts/run_ablation.py --name lr --sweep lr=5e-4,1e-3,2e-3,4e-3,8e-3 --seeds 0,1,2 --device cuda --precision fp16
  python scripts/run_ablation.py --name kv --sweep n_kv_heads=1,2,4 --match-params --fixed lr=0.002 --seeds 0,1,2 --device cuda --precision fp16
Finished runs are stored in results/ablation-<name>.json and skipped on re-run.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from vibe.ablation import parse_values, run_one, summarize
from vibe.config import load_config, total_params
from vibe.data import open_split
from vibe.train import TrainConfig, environment_info

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--sweep", required=True, help="field=v1,v2,...")
    ap.add_argument("--fixed", action="append", default=[], help="field=value (repeatable)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--match-params", action="store_true",
                    help="adjust ff_dim so every variant has about the base parameter count")
    ap.add_argument("--model-config", type=Path, default=ROOT / "configs" / "tiny.yaml")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed" / "poc")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    field, values_text = args.sweep.split("=", 1)
    values = parse_values(values_text)
    fixed = {}
    for item in args.fixed:
        k, v = item.split("=", 1)
        fixed[k] = parse_values(v)[0]
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.match_params and field == "ff_dim":
        raise SystemExit("--match-params cannot be combined with sweeping ff_dim")

    base_model = load_config(args.model_config)
    base_train = TrainConfig(
        steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len,
        warmup_steps=args.warmup_steps, precision=args.precision,
        eval_every=10**9, eval_batches=20, log_every=10**9, ckpt_every=10**9,
    )
    train_stream, meta = open_split(args.data_dir, "train")
    val_stream, _ = open_split(args.data_dir, "val")
    v = meta["val"]
    tpb = v["tokens"] / v["bytes"] if v.get("bytes") else None
    target = total_params(base_model) if args.match_params else None

    out_path = ROOT / "results" / f"ablation-{args.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(out_path.read_text(encoding="utf-8"))["records"] if out_path.exists() else []
    done = {(r["variant"], r["seed"]) for r in records}

    def save() -> None:
        out_path.write_text(json.dumps({
            "name": args.name, "sweep": {field: values}, "fixed": fixed,
            "match_params": args.match_params, "model": asdict(base_model),
            "train_defaults": asdict(base_train), "records": records,
            "summary": summarize(records), "env": environment_info(),
        }, indent=2), encoding="utf-8")

    for value in values:
        variant = f"{field}={value}"
        for seed in seeds:
            if (variant, seed) in done:
                print(f"[skip] {variant} seed {seed} (already done)")
                continue
            run_dir = ROOT / "runs" / f"ablation-{args.name}" / f"{variant}-s{seed}"
            if run_dir.exists():
                shutil.rmtree(run_dir)  # unfinished earlier attempt
            rec = run_one(base_model, base_train, {**fixed, field: value}, seed,
                          train_stream, val_stream, run_dir, device=args.device,
                          tokens_per_byte=tpb, target_params=target)
            rec["variant"] = variant
            records.append(rec)
            save()
            result = "DIVERGED" if rec["diverged"] else f"bits/byte {rec['val_bpb']:.4f}"
            print(f"[done] {variant:<18} seed {seed} | {result} | {rec['seconds']:.0f}s", flush=True)

    rows = summarize(records)
    print(f"\n{'variant':<20}{'params':>11}{'n':>3}{'mean bpb':>10}{'std':>8}{'min':>8}{'max':>8}{'diverged':>9}")
    for r in rows:
        if r["mean"] is None:
            print(f"{r['variant']:<20}{r['params']:>11,}{r['n']:>3}{'-':>10}{'-':>8}{'-':>8}{'-':>8}{r['diverged']:>9}")
        else:
            print(f"{r['variant']:<20}{r['params']:>11,}{r['n']:>3}{r['mean']:>10.4f}"
                  f"{r['std']:>8.4f}{r['min']:>8.4f}{r['max']:>8.4f}{r['diverged']:>9}")
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())