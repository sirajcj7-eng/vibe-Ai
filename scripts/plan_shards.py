"""Shard inventory and stratum probe (Phase 7b.4).

For every source in the mix: resolves the dataset, records the full revision
hash (and compares it with the revision seen in the earlier audit), counts
shards and file size, estimates how many tokens are available, and for
stackexchange and stackv2_edu reads the first documents of every shard to
learn which site or programming language each shard holds.

Run on Kaggle (Internet on, no accelerator):  python scripts/plan_shards.py
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from vibe.filters import CODE_LANGUAGE_TARGETS, stratum
from vibe.sources import SOURCES, target_tokens

ROOT = Path(__file__).resolve().parents[1]
STRATA_SOURCES = ("stackexchange", "stackv2_edu")
DATA_SUFFIXES = (".json.gz", ".jsonl.gz", ".json", ".jsonl", ".parquet", ".zst")
BYTES_PER_TOKEN = 3.6   # rough, for a 16k tokenizer
GZIP_RATIO = 3.0        # rough, for gzipped text

AUDIT_PREFIXES = {
    "cccc": "03a3de5713", "youtube": "dff8c8a54e", "libretexts": "70388bca52",
    "oercommons": "506b6159da", "pressbooks": "1a1d3b50d7", "doab": "defb24ca72",
    "news": "59aaa8f104", "foodista": "bf2c7aaa2d", "public_domain_review": "efc7f21259",
    "stackexchange": "c0ac737383", "github_archive": "52282fe966", "ubuntu_irc": "84f88c9865",
    "wikimedia": "0641bb84bd", "pes2o": "297747513b", "arxiv_papers": "033cf7f53f",
    "arxiv_abstracts": "dc1ceab475", "pubmed": "c156f0569a", "stackv2_edu": "c354dbe884",
    "pre_1929_books": "23f9d96dbb", "project_gutenberg": "3cdf6879c8",
    "library_of_congress": "56725c7aa1",
}


def probe_strata(ds, source_key: str, n_shards: int, probe_docs: int):
    strata: dict[str, list[int]] = {}
    mixed, errors = [], []
    for i in range(n_shards):
        try:
            rows = list(itertools.islice(ds.shard(num_shards=n_shards, index=i), probe_docs))
        except Exception:
            errors.append(i)
            continue
        labels = [stratum(source_key, r.get("metadata")) for r in rows]
        if not labels:
            errors.append(i)
            continue
        if len(set(labels)) > 1:
            mixed.append(i)
        strata.setdefault(str(labels[0]), []).append(i)
        if (i + 1) % 25 == 0:
            print(f"    probed {i + 1}/{n_shards} shards", flush=True)
    return strata, mixed, errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-docs", type=int, default=5)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "plan-common-pile.json")
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi

    api = HfApi()
    wanted = {k.strip() for k in args.only.split(",") if k.strip()}
    selected = [s for s in SOURCES if not wanted or s.key in wanted]
    plan, failures = {}, {}

    print(f"{'source':<21}{'shards':>7}{'files GB':>10}{'~tok B':>8}{'need B':>8}{'cover':>7}  revision")
    for src in selected:
        try:
            repo = sha = info = last = None
            for cand in src.candidates:
                try:
                    info = api.dataset_info(cand, files_metadata=True)
                    repo, sha = cand, info.sha
                    break
                except Exception as e:
                    last = e
            if repo is None:
                raise RuntimeError(f"not resolved: {last}")
            gb = sum((s.size or 0) for s in info.siblings
                     if s.rfilename.endswith(DATA_SUFFIXES)) / 1e9
            ds = load_dataset(repo, split="train", streaming=True, revision=sha)
            n_shards = ds.n_shards
            tok_b = gb * GZIP_RATIO / BYTES_PER_TOKEN
            need_b = target_tokens(src) / 1e9
            cover = tok_b / need_b if need_b else 0.0
            same = sha.startswith(AUDIT_PREFIXES.get(src.key, "?"))
            print(f"{src.key:<21}{n_shards:>7}{gb:>10.1f}{tok_b:>8.1f}{need_b:>8.2f}"
                  f"{cover:>6.0f}x  {sha[:10]} {'same as audit' if same else 'CHANGED since audit'}"
                  f"{'  LOW COVERAGE' if cover < 5 else ''}", flush=True)
            entry = {"repo": repo, "revision": sha, "same_as_audit": same, "n_shards": n_shards,
                     "files_gb": round(gb, 2), "est_tokens_b": round(tok_b, 1),
                     "need_tokens_b": round(need_b, 2)}
            if src.key in STRATA_SOURCES:
                strata, mixed, errors = probe_strata(ds, src.key, n_shards, args.probe_docs)
                entry.update(strata=strata, mixed_shards=mixed, probe_errors=errors)
            plan[src.key] = entry
        except Exception as e:
            failures[src.key] = str(e)[:300]
            print(f"{src.key:<21}FAILED: {str(e)[:110]}")

    if "stackv2_edu" in plan and "strata" in plan["stackv2_edu"]:
        s = plan["stackv2_edu"]["strata"]
        print("\n=== stackv2_edu: shards per language ===")
        print(", ".join(f"{k}:{len(v)}" for k, v in sorted(s.items(), key=lambda kv: -len(kv[1]))))
        print("targets not found in data:", [l for l in CODE_LANGUAGE_TARGETS if l not in s])
        print("found but not in targets:", [l for l in s if l not in CODE_LANGUAGE_TARGETS])
        print("mixed shards:", plan["stackv2_edu"]["mixed_shards"],
              "| probe errors:", plan["stackv2_edu"]["probe_errors"])
    if "stackexchange" in plan and "strata" in plan["stackexchange"]:
        s = plan["stackexchange"]["strata"]
        top = sorted(s.items(), key=lambda kv: -len(kv[1]))
        print(f"\n=== stackexchange: {len(s)} sites; shards per site (top 15) ===")
        print(", ".join(f"{k}:{len(v)}" for k, v in top[:15]))
        print("mixed shards:", plan["stackexchange"]["mixed_shards"],
              "| probe errors:", plan["stackexchange"]["probe_errors"])

    print("\n=== REVISIONS (paste to me) ===")
    for k, v in plan.items():
        print(f"{k}: {v['revision']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"sources": plan, "failures": failures}, indent=2),
                        encoding="utf-8")
    print(f"\nPlanned {len(plan)} of {len(selected)} sources. Failures: {list(failures) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())