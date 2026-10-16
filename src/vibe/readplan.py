"""Read plans: how much to take from which shard (Phase 7b)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import yaml

from vibe.filters import CODE_LANGUAGE_TARGETS, SE_ALLOWED_SITES, SE_MAX_SHARE


@dataclass(frozen=True)
class ReadItem:
    shard: int
    quota: int
    stratum: str | None = None


def load_plan(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["sources"]


def allocate_with_rollover(targets: dict, available: dict, groups: dict) -> dict:
    """Cap each source at its availability and move the shortfall to other
    sources of the same group that still have room, in proportion to their
    original targets. Units (tokens, bytes) are the caller's choice."""
    final = {k: min(v, available.get(k, v)) for k, v in targets.items()}
    for group in set(groups[k] for k in targets):
        members = [k for k in targets if groups[k] == group]
        shortfall = sum(targets[k] - final[k] for k in members)
        while shortfall > 1e-9:
            spare = {k: available.get(k, float("inf")) - final[k] for k in members}
            takers = [k for k in members if spare[k] > 1e-9]
            if not takers:
                break  # the group cannot absorb the rest
            weight = sum(targets[k] for k in takers)
            given = 0.0
            for k in takers:
                add = min(spare[k], shortfall * targets[k] / weight)
                final[k] += add
                given += add
            if given <= 1e-12:
                break
            shortfall -= given
    return final


def plan_reads(source_key: str, total: int, n_shards: int,
               strata: dict | None = None, max_shards: int = 24,
               seed: int = 0) -> list[ReadItem]:
    """Split `total` units over a seeded random selection of shards.

    Strata sources first split the total by stratum: Stack Exchange (capped
    Stack Overflow share, English topical sites only) and Stack v2 (code
    language targets, renormalised over the languages that exist).
    """
    rng = random.Random(f"{seed}-{source_key}")

    def spread(shards: list[int], units: int, stratum, cap: int) -> list[ReadItem]:
        if units <= 0 or not shards:
            return []
        chosen = sorted(rng.sample(shards, min(cap, len(shards))))
        base, extra = divmod(units, len(chosen))
        return [ReadItem(s, base + (1 if i < extra else 0), stratum)
                for i, s in enumerate(chosen)]

    if source_key == "stackexchange" and strata:
        sites = {k: v for k, v in strata.items() if k in SE_ALLOWED_SITES}
        so = sites.get("stackoverflow.com", [])
        others = {k: v for k, v in sites.items() if k != "stackoverflow.com"}
        so_units = round(total * SE_MAX_SHARE["stackoverflow.com"]) if (so and others) else (total if so else 0)
        items = spread(so, so_units, "stackoverflow.com", max_shards)
        rest = total - so_units
        n_other = sum(len(v) for v in others.values())
        for site, shards in sorted(others.items()):
            items += spread(shards, round(rest * len(shards) / n_other), site, max_shards)
        return items

    if source_key == "stackv2_edu" and strata:
        present = {l: s for l, s in CODE_LANGUAGE_TARGETS.items() if l in strata}
        norm = sum(present.values())
        items = []
        for lang, share in sorted(present.items()):
            items += spread(strata[lang], round(total * share / norm), lang, 6)
        return items

    return spread(list(range(n_shards)), total, None, max_shards)