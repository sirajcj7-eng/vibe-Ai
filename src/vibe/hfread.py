"""Hugging Face streaming helpers (need the `datasets` library; used on Kaggle)."""
from __future__ import annotations

import itertools

from vibe.filters import stratum


def open_dataset(repo: str, revision: str):
    from datasets import load_dataset

    ds = load_dataset(repo, split="train", streaming=True, revision=revision)
    return ds, ds.n_shards


def iter_shard(ds, n_shards: int, index: int):
    return ds.shard(num_shards=n_shards, index=index)


def probe_strata(ds, source_key: str, n_shards: int, probe_docs: int = 5):
    """Which stratum (site or language) does each shard hold?"""
    strata: dict[str, list[int]] = {}
    mixed, errors = [], []
    for i in range(n_shards):
        try:
            rows = list(itertools.islice(iter_shard(ds, n_shards, i), probe_docs))
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
    return strata, mixed, errors