"""Why is wikimedia slow and sparse? Reads whole shards and reports where the
usable documents are (Phase 7b.5 follow-up).

Run on Kaggle (Internet on, no accelerator):
    python scripts/diag_wikimedia.py --shards 2
"""
from __future__ import annotations

import argparse
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from vibe.filters import check_document
from vibe.hfread import iter_shard, open_dataset
from vibe.readplan import load_plan

ROOT = Path(__file__).resolve().parents[1]
BYTES_PER_TOKEN = 3.55  # wikimedia at 16k, from the tokenizer comparison


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--bin", type=int, default=25_000, help="documents per report row")
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plan", type=Path, default=ROOT / "data" / "commonpile_plan.yaml")
    args = ap.parse_args()

    info = load_plan(args.plan)["wikimedia"]
    ds, n_shards = open_dataset(info["repo"], info["revision"])
    picks = random.Random(args.seed).sample(range(n_shards), min(args.shards, n_shards))
    print(f"wikimedia: {n_shards} shards; reading whole shards {picks}", flush=True)

    kept_mb_per_shard: list[float] = []
    examples: list[tuple] = []
    for si in picks:
        t0 = time.time()
        bins: dict[int, Counter] = defaultdict(Counter)
        wikis: Counter = Counter()
        namespaces: Counter = Counter()
        total_docs = kept = kept_chars = 0
        for j, row in enumerate(iter_shard(ds, n_shards, si)):
            meta = row.get("metadata") or {}
            text = row.get("text") or ""
            wikis[str(meta.get("wiki"))] += 1
            namespaces[str(meta.get("namespace"))] += 1
            ok, reason = check_document("wikimedia", text, meta)
            c = bins[j // args.bin]
            c["docs"] += 1
            c[reason] += 1
            total_docs += 1
            if ok:
                kept += 1
                kept_chars += len(text)
                c["kept_chars"] += len(text)
            elif reason == "replacement_char" and len(examples) < args.examples:
                i = text.index("\ufffd")
                examples.append((meta.get("title"), text[max(0, i - 50): i + 50]))
        mb = kept_chars / 1e6
        kept_mb_per_shard.append(mb)
        print(f"\nshard {si}: {total_docs:,} docs in {time.time() - t0:.0f}s; "
              f"usable {kept:,} docs = {mb:.1f} MB", flush=True)
        print("  wiki:", dict(wikis.most_common(6)))
        print("  namespace:", dict(namespaces.most_common(6)))
        print(f"  {'from doc':>9}{'docs':>8}{'kept':>8}{'MB':>7}{'wiki_flt':>9}{'short':>8}{'repl':>8}{'alpha':>7}")
        for b in sorted(bins):
            c = bins[b]
            print(f"  {b * args.bin:>9,}{c['docs']:>8,}{c['ok']:>8,}{c['kept_chars'] / 1e6:>7.1f}"
                  f"{c['wiki_filter']:>9,}{c['short']:>8,}{c['replacement_char']:>8,}{c['low_alpha']:>7,}")

    avg = sum(kept_mb_per_shard) / len(kept_mb_per_shard)
    total_mb = avg * n_shards
    print(f"\nEstimate: {avg:.1f} MB usable per shard x {n_shards} shards = {total_mb:,.0f} MB "
          f"= about {total_mb / BYTES_PER_TOKEN / 1000:.2f}B tokens (we need 0.70B)")
    print("\nExamples of long documents with U+FFFD (title, context):")
    for title, ctx in examples:
        print(" ", repr(title), repr(ctx))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())