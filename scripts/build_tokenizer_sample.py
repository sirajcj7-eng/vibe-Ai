"""Build the tokenizer training sample from the real mix (Phase 7b.5).

Run on Kaggle (Internet on, no accelerator):
    python scripts/build_tokenizer_sample.py --sample-mb 300
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from vibe.corpus import collect_text
from vibe.hfread import iter_shard, open_dataset, probe_strata
from vibe.readplan import allocate_with_rollover, load_plan, plan_reads
from vibe.sources import SOURCES

ROOT = Path(__file__).resolve().parents[1]
STRATA_SOURCES = ("stackexchange", "stackv2_edu")
BYTES_PER_TOKEN = 3.6
SAFETY = 0.5            # trust only half of the rough availability estimate
MIN_EST_TOKENS_B = 0.02  # estimates of 0.0 for tiny sources are rounding, not zero


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-mb", type=int, default=300)
    ap.add_argument("--plan", type=Path, default=ROOT / "data" / "commonpile_plan.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "tok_sample")
    ap.add_argument("--max-shards", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    plan = load_plan(args.plan)
    sample_bytes = args.sample_mb * 1_000_000
    targets = {s.key: s.share * sample_bytes for s in SOURCES}
    available = {k: max(plan[k]["est_tokens_b"], MIN_EST_TOKENS_B) * 1e9 * BYTES_PER_TOKEN * SAFETY
                 for k in targets}
    final = allocate_with_rollover(targets, available, {s.key: s.group for s in SOURCES})

    wanted = {k.strip() for k in args.only.split(",") if k.strip()}
    selected = [s for s in SOURCES if not wanted or s.key in wanted]
    args.out.mkdir(parents=True, exist_ok=True)
    report, failures = {}, {}

    print(f"{'source':<21}{'shards':>7}{'target MB':>10}{'got MB':>8}{'docs':>7}  main drops")
    for src in selected:
        t0 = time.time()
        try:
            info = plan[src.key]
            ds, n_shards = open_dataset(info["repo"], info["revision"])
            if n_shards != info["shards"]:
                print(f"  warning: {src.key} has {n_shards} shards, plan says {info['shards']}")
            strata = None
            if src.key in STRATA_SOURCES:
                strata, mixed, errors = probe_strata(ds, src.key, n_shards)
                if mixed or errors:
                    print(f"  warning: {src.key} mixed shards {mixed}, probe errors {errors}")
            items = plan_reads(src.key, int(final[src.key]), n_shards, strata,
                               args.max_shards, args.seed)
            kept = chars = 0
            drops: Counter = Counter()
            strata_chars: Counter = Counter()
            with open(args.out / f"{src.key}.jsonl", "w", encoding="utf-8") as f:
                for it in items:
                    k, c, d = collect_text(src.key, iter_shard(ds, n_shards, it.shard), it.quota, f)
                    kept += k
                    chars += c
                    drops.update(d)
                    strata_chars[it.stratum or "-"] += c
        except Exception as e:
            failures[src.key] = str(e)[:300]
            print(f"{src.key:<21}FAILED: {str(e)[:110]}")
            continue
        top = ", ".join(f"{k}={v}" for k, v in drops.most_common(3)) or "-"
        print(f"{src.key:<21}{len(items):>7}{final[src.key] / 1e6:>10.1f}{chars / 1e6:>8.1f}"
              f"{kept:>7}  {top}  ({time.time() - t0:.0f}s)", flush=True)
        if src.key in STRATA_SOURCES:
            top_strata = sorted(strata_chars.items(), key=lambda kv: -kv[1])[:8]
            print("    MB per stratum:", ", ".join(f"{k}:{v / 1e6:.1f}" for k, v in top_strata))
        report[src.key] = {"target_mb": final[src.key] / 1e6, "got_mb": chars / 1e6, "docs": kept,
                           "drops": dict(drops), "shards_used": len(items),
                           "strata_mb": {k: v / 1e6 for k, v in strata_chars.items()}}

    total = sum(r["got_mb"] for r in report.values())
    by_group: Counter = Counter()
    for s in selected:
        if s.key in report:
            by_group[s.group] += report[s.key]["got_mb"]
    print(f"\nSample total: {total:.0f} MB. By group: " +
          ", ".join(f"{g}={v:.0f}" for g, v in by_group.items()))
    short = [k for k, r in report.items() if r["got_mb"] < 0.9 * r["target_mb"]]
    print("Under-delivered (less than 90% of target):", short or "none",
          "| failures:", list(failures) or "none")

    out_json = ROOT / "results" / "tok-sample-report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"sample_mb": args.sample_mb, "sources": report,
                                    "failures": failures}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())