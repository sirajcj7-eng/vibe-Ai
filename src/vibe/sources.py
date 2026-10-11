"""Registry of Common Pile sources for the real training mix
(docs/decisions/008-data-sources.md) plus small helpers."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

TOTAL_TOKENS = 5_000_000_000
RESERVE_SHARE = 0.05


@dataclass(frozen=True)
class Source:
    key: str                      # short name used in our files
    group: str
    share: float                  # fraction of the total token budget
    candidates: tuple[str, ...]   # Hugging Face dataset ids to try, preferred first


def _filtered(name: str) -> tuple[str, ...]:
    return (f"common-pile/{name}_filtered", f"common-pile/{name}")


SOURCES: tuple[Source, ...] = (
    # General web, spoken, educational: 30%
    Source("cccc", "web_spoken_edu", 0.14, _filtered("cccc")),
    Source("youtube", "web_spoken_edu", 0.06, _filtered("youtube")),
    Source("libretexts", "web_spoken_edu", 0.03, _filtered("libretexts")),
    Source("oercommons", "web_spoken_edu", 0.02, _filtered("oercommons")),
    Source("pressbooks", "web_spoken_edu", 0.02, _filtered("pressbooks")),
    Source("doab", "web_spoken_edu", 0.01, _filtered("doab")),
    Source("news", "web_spoken_edu", 0.015, _filtered("news")),
    Source("foodista", "web_spoken_edu", 0.0025, _filtered("foodista")),
    Source("public_domain_review", "web_spoken_edu", 0.0025, _filtered("public_domain_review")),
    # Q&A and discussion: 16%
    Source("stackexchange", "qa_discussion", 0.14, _filtered("stackexchange")),
    Source("github_archive", "qa_discussion", 0.01, _filtered("github_archive")),
    Source("ubuntu_irc", "qa_discussion", 0.01, _filtered("ubuntu_irc")),
    # Reference: 14%
    Source("wikimedia", "reference", 0.14, _filtered("wikimedia")),
    # Science and scholarly: 11%
    Source("pes2o", "science", 0.06,
           ("common-pile/pes2o_filtered", "common-pile/peS2o_filtered", "common-pile/peS2o")),
    Source("arxiv_papers", "science", 0.03, _filtered("arxiv_papers")),
    Source("arxiv_abstracts", "science", 0.01, _filtered("arxiv_abstracts")),
    Source("pubmed", "science", 0.01, _filtered("pubmed")),
    # Code: 15%
    Source("stackv2_edu", "code", 0.15,
           ("common-pile/stackv2_edu_filtered", "common-pile/stackv2_filtered",
            "common-pile/stackv2")),
    # Books: 9%
    Source("pre_1929_books", "books", 0.05, _filtered("pre_1929_books")),
    Source("project_gutenberg", "books", 0.02, _filtered("project_gutenberg")),
    Source("library_of_congress", "books", 0.02, _filtered("library_of_congress")),
)


def target_tokens(source: Source, total: int = TOTAL_TOKENS) -> int:
    return round(source.share * total)


def group_shares() -> dict[str, float]:
    out: dict[str, float] = {}
    for s in SOURCES:
        out[s.group] = out.get(s.group, 0.0) + s.share
    return out


def doc_stats(lengths: list[int]) -> dict:
    """Document count and length statistics (in characters)."""
    if not lengths:
        return {"docs": 0, "chars": 0, "mean": 0.0, "median": 0, "p95": 0, "max": 0}
    s = sorted(lengths)
    n = len(s)
    return {
        "docs": n,
        "chars": sum(s),
        "mean": sum(s) / n,
        "median": s[n // 2],
        "p95": s[min(n - 1, int(0.95 * n))],
        "max": s[-1],
    }


def metadata_keys(metadatas: list, limit: int = 30) -> list[str]:
    keys: set[str] = set()
    for m in metadatas:
        if isinstance(m, dict):
            keys.update(str(k) for k in m)
    return sorted(keys)[:limit]


def license_values(metadatas: list) -> dict[str, dict[str, int]]:
    """For every metadata key containing 'licen', count the values seen."""
    out: dict[str, Counter] = {}
    for m in metadatas:
        if not isinstance(m, dict):
            continue
        for k, v in m.items():
            if "licen" in str(k).lower():
                out.setdefault(str(k), Counter())[str(v)[:80]] += 1
    return {k: dict(c.most_common(8)) for k, c in out.items()}


def update_manifest(path: str | Path, section: str, data: dict) -> None:
    """Set one top-level section of the manifest, keeping all other sections."""
    path = Path(path)
    manifest: dict = {}
    if path.exists():
        manifest = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    manifest[section] = data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")