"""Look at the real data before building the corpus (Phase 7b.1).

Streams a small sample from each Common Pile source in the mix, pins the
dataset revision, prints statistics and license metadata, saves samples
and writes the 'common_pile' section of data/manifest.yaml.

Run on Kaggle (Internet on, no accelerator needed):
    python scripts/sample_sources.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

from vibe.sources import (
    RESERVE_SHARE, SOURCES, TOTAL_TOKENS, doc_stats, license_values,
    metadata_keys, target_tokens, update_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CHARS_KEPT = 20_000  # per document, in the saved sample files


def resolve(api, candidates: tuple[str, ...]) -> tuple[str, str]:
    last = None
    for repo in candidates:
        try:
            return repo, api.dataset_info(repo).sha
        except Exception as e:  # not found, network, etc.
            last = e
    raise RuntimeError(f"none of {list(candidates)} could be resolved: {last}")


def sample_one(repo: str, sha: str, max_docs: int, max_chars: int, buffer_size: int) -> list:
    from datasets import load_dataset

    ds = load_dataset(repo, split="train", streaming=True, revision=sha)
    ds = ds.shuffle(seed=0, buffer_size=buffer_size)  # shuffles shard order too
    docs, chars = [], 0
    for row in ds:
        docs.append(row)
        chars += len(row.get("text") or "")
        if len(docs) >= max_docs or chars >= max_chars:
            break
    return docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=300, help="max documents per source")
    ap.add_argument("--max-chars", type=int, default=5_000_000, help="max characters per source")
    ap.add_argument("--buffer", type=int, default=32, help="shuffle buffer (small: some books are huge)")
    ap.add_argument("--only", default="", help="comma-separated source keys")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "samples")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.yaml")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    wanted = {k.strip() for k in args.only.split(",") if k.strip()}
    selected = [s for s in SOURCES if not wanted or s.key in wanted]
    args.out.mkdir(parents=True, exist_ok=True)

    entries: dict = {}
    failures: dict = {}
    print(f"{'source':<22}{'dataset':<44}{'docs':>5}{'mean chars':>12}{'median':>9}{'p95':>10}  license metadata")
    for src in selected:
        try:
            repo, sha = resolve(api, src.candidates)
            docs = sample_one(repo, sha, args.docs, args.max_chars, args.buffer)
        except Exception as e:
            failures[src.key] = str(e)[:300]
            print(f"{src.key:<22}FAILED: {str(e)[:120]}")
            continue

        lengths = [len(d.get("text") or "") for d in docs]
        st = doc_stats(lengths)
        metas = [d.get("metadata") for d in docs]
        lic = license_values(metas)
        keys = metadata_keys(metas)

        with open(args.out / f"{src.key}.jsonl", "w", encoding="utf-8") as f:
            for d in docs:
                rec = {"id": d.get("id"), "text": (d.get("text") or "")[:SAMPLE_CHARS_KEPT],
                       "metadata": d.get("metadata")}
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

        entries[src.key] = {
            "group": src.group,
            "share": src.share,
            "target_tokens": target_tokens(src),
            "repo": repo,
            "revision": sha,
            "is_filtered": repo.endswith("_filtered"),
            "sampled_docs": st["docs"],
            "mean_chars": round(st["mean"]),
            "median_chars": st["median"],
            "metadata_keys": keys,
            "license_values_seen": lic,
        }
        lic_short = "; ".join(f"{k}={list(v)[:2]}" for k, v in list(lic.items())[:1]) or "(none found)"
        print(f"{src.key:<22}{repo:<44}{st['docs']:>5}{st['mean']:>12,.0f}{st['median']:>9,}"
              f"{st['p95']:>10,}  {lic_short[:60]}", flush=True)

    update_manifest(args.manifest, "common_pile", {
        "retrieved": datetime.date.today().isoformat(),
        "paper": "arXiv 2506.05209",
        "total_token_budget": TOTAL_TOKENS,
        "reserve_share": RESERVE_SHARE,
        "sources": entries,
        "failures": failures,
    })
    print(f"\nResolved {len(entries)} of {len(selected)} sources. Wrote {args.manifest}")
    if failures:
        print("Failed:", ", ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())