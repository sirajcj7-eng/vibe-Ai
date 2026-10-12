"""Source inventory and license audit (Phase 7b.2).

Samples documents from many shards of each source, checks every sampled
document against the license allow-list, measures length and junk signals,
and shows low-cardinality metadata fields (for example the wiki project).

Run on Kaggle (Internet on, no accelerator):
    python scripts/audit_sources.py
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path

import yaml

from vibe.licenses import LICENSE_FIELD, SOURCE_LEVEL_LICENSE, doc_license_status
from vibe.sources import SOURCES, doc_stats

ROOT = Path(__file__).resolve().parents[1]


def resolve(api, candidates):
    last = None
    for repo in candidates:
        try:
            return repo, api.dataset_info(repo).sha
        except Exception as e:
            last = e
    raise RuntimeError(f"none of {list(candidates)} resolved: {last}")


def sample_across_shards(repo, sha, shards_wanted, docs_per_shard, seed,
                         max_skip_docs=500, max_skip_chars=3_000_000):
    from datasets import load_dataset

    ds = load_dataset(repo, split="train", streaming=True, revision=sha)
    total = ds.n_shards
    rng = random.Random(seed)
    picks = sorted(rng.sample(range(total), min(shards_wanted, total)))
    docs = []
    for i in picks:
        shard = ds.shard(num_shards=total, index=i)
        skip = rng.randint(0, max_skip_docs)  # start somewhere inside the shard
        skipped_docs = skipped_chars = taken = 0
        for row in shard:
            if skipped_docs < skip and skipped_chars < max_skip_chars:
                skipped_docs += 1
                skipped_chars += len(row.get("text") or "")
                continue
            docs.append(row)
            taken += 1
            if taken >= docs_per_shard:
                break
    return docs, total


def junk_flags(text: str) -> dict:
    head = text[:20000]
    n = len(head) or 1
    return {
        "lt200": len(text) < 200,
        "lt500": len(text) < 500,
        "repl": "\ufffd" in head,
        "low_alpha": sum(c.isalpha() for c in head) / n < 0.6,
    }


def categories(metas, lengths, max_distinct=12):
    """Low-cardinality metadata fields: value -> [count, median length]."""
    keys = set()
    for m in metas:
        if isinstance(m, dict):
            keys.update(m)
    out = {}
    for k in sorted(keys, key=str):
        if "licen" in str(k).lower():
            continue
        groups, ok = {}, True
        for m, n in zip(metas, lengths):
            if not isinstance(m, dict) or k not in m:
                continue
            v = m[k]
            if not isinstance(v, (str, int, bool)) or (isinstance(v, str) and len(v) > 60):
                ok = False
                break
            groups.setdefault(str(v), []).append(n)
            if len(groups) > max_distinct:
                ok = False
                break
        if ok and groups and len(groups) < max(2, len(metas) // 3):
            top = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:6]
            out[str(k)] = {v: [len(ls), int(statistics.median(ls))] for v, ls in top}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=10)
    ap.add_argument("--docs-per-shard", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "audit-common-pile.json")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    pinned = {}
    if args.manifest.exists():
        data = yaml.safe_load(args.manifest.read_text(encoding="utf-8")) or {}
        for k, v in (data.get("common_pile", {}).get("sources", {}) or {}).items():
            pinned[k] = (v["repo"], v["revision"])

    wanted = {k.strip() for k in args.only.split(",") if k.strip()}
    selected = [s for s in SOURCES if not wanted or s.key in wanted]
    report, failures, details = {}, {}, []
    bad_total = 0

    print(f"{'source':<21}{'docs':>5}{'shards':>7}{'median':>9}{'%<200':>7}{'%<500':>7}"
          f"{'%lowA':>7}{'%repl':>7}{'badLic':>7}  revision")
    for src in selected:
        try:
            repo, sha = pinned.get(src.key) or resolve(api, src.candidates)
            docs, n_shards = sample_across_shards(
                repo, sha, args.shards, args.docs_per_shard, args.seed)
        except Exception as e:
            failures[src.key] = str(e)[:300]
            print(f"{src.key:<21}FAILED: {str(e)[:100]}")
            continue

        texts = [d.get("text") or "" for d in docs]
        lengths = [len(t) for t in texts]
        metas = [d.get("metadata") for d in docs]
        field = LICENSE_FIELD.get(src.key, "license")
        src_level = src.key in SOURCE_LEVEL_LICENSE
        labels, bad_values = Counter(), Counter()
        for m in metas:
            label, ok = doc_license_status(m, field, src_level)
            labels[label] += 1
            if not ok:
                raw = m.get(field) if isinstance(m, dict) else None
                bad_values[str(raw)[:80]] += 1
        flags = [junk_flags(t) for t in texts]
        n = max(1, len(docs))
        pct = lambda key: 100 * sum(f[key] for f in flags) / n  # noqa: E731
        st = doc_stats(lengths)
        bad = sum(bad_values.values())
        bad_total += bad
        print(f"{src.key:<21}{st['docs']:>5}{n_shards:>7}{st['median']:>9,}{pct('lt200'):>7.0f}"
              f"{pct('lt500'):>7.0f}{pct('low_alpha'):>7.0f}{pct('repl'):>7.0f}{bad:>7}  {sha[:10]}",
              flush=True)

        cats = categories(metas, lengths)
        example = json.dumps(metas[0], default=str)[:220] if metas else ""
        details.append((src.key, field, labels, bad_values, cats, example))
        report[src.key] = {
            "repo": repo, "revision": sha, "n_shards": n_shards, "docs": st["docs"],
            "median_chars": st["median"], "mean_chars": round(st["mean"]), "p95_chars": st["p95"],
            "pct_lt200": pct("lt200"), "pct_lt500": pct("lt500"),
            "pct_low_alpha": pct("low_alpha"), "pct_replacement_char": pct("repl"),
            "license_field": field, "license_labels": dict(labels),
            "bad_license_values": dict(bad_values), "categories": cats,
        }

    print("\n=== DETAILS (license labels, metadata categories value:count:median_chars) ===")
    for key, field, labels, bad_values, cats, example in details:
        print(f"\n## {key}  [license field: {field}]")
        print("  licenses:", dict(labels.most_common(6)))
        if bad_values:
            print("  NOT ALLOWED:", dict(bad_values.most_common(4)))
        for ck, vals in cats.items():
            shown = ", ".join(f"{v}:{c}:{m}" for v, (c, m) in vals.items())
            print(f"  {ck}: {shown}"[:300])
        print("  example metadata:", example)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"sources": report, "failures": failures}, indent=2),
                        encoding="utf-8")
    verdict = ("every sampled document passes the allow-list" if bad_total == 0
               else f"{bad_total} sampled documents FAIL the allow-list")
    print(f"\nAudit done: {len(report)} sources, {verdict}. Failures: {list(failures) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())