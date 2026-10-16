"""Document filters for the real training corpus (doc 008, Phase 7b)."""
from __future__ import annotations

from vibe.licenses import LICENSE_FIELD, SOURCE_LEVEL_LICENSE, doc_license_status

DEFAULT_MIN_CHARS = 500
MIN_CHARS = {"arxiv_abstracts": 300, "stackv2_edu": 300}

# Prose sources get a letters-ratio check. Code, math and Q&A with code are exempt.
ALPHA_SOURCES = frozenset({
    "cccc", "doab", "pressbooks", "oercommons", "libretexts", "news", "foodista",
    "public_domain_review", "wikimedia", "pre_1929_books", "project_gutenberg",
    "library_of_congress",
})
MIN_ALPHA_RATIO = 0.5

# Maximum share of a source's token budget that one stratum may use.
STRATA_CAPS = {"stackexchange": {"stackoverflow.com": 0.35}}

# English Stack Exchange sites kept (the filtered dataset has 13 sites; the
# non-English Stack Overflow sites and the meta sites are dropped).
SE_ALLOWED_SITES = frozenset({
    "stackoverflow.com", "math.stackexchange.com", "superuser.com",
    "unix.stackexchange.com", "diy.stackexchange.com", "blender.stackexchange.com",
    "islam.stackexchange.com", "sitecore.stackexchange.com",
})
SE_MAX_SHARE = {"stackoverflow.com": 0.35}

# Target share of the code budget per language (Stack v2 names, confirmed by
# the shard inventory: 14 languages, no SQL).
CODE_LANGUAGE_TARGETS = {
    "Python": 0.32, "JavaScript": 0.14, "TypeScript": 0.08, "Java": 0.08,
    "C": 0.07, "C++": 0.07, "Go": 0.05, "Rust": 0.05, "PHP": 0.04,
    "C#": 0.03, "Ruby": 0.02, "Shell": 0.02, "Markdown": 0.02, "Swift": 0.01,
}


def alpha_ratio(text: str, head: int = 20_000) -> float:
    sample = text[:head]
    return sum(c.isalpha() for c in sample) / len(sample) if sample else 0.0


def _truthy(value) -> bool:
    return value is True or str(value).strip().lower() == "true"


def check_document(source: str, text: str | None, meta) -> tuple[bool, str]:
    """Return (keep, reason). Reason is 'ok' or why the document is dropped."""
    text = text or ""
    meta = meta if isinstance(meta, dict) else {}
    _, allowed = doc_license_status(
        meta, LICENSE_FIELD.get(source, "license"), source in SOURCE_LEVEL_LICENSE
    )
    if not allowed:
        return False, "license"
    if len(text) < MIN_CHARS.get(source, DEFAULT_MIN_CHARS):
        return False, "short"
    if "\ufffd" in text[:20_000]:
        return False, "replacement_char"
    if source in ALPHA_SOURCES and alpha_ratio(text) < MIN_ALPHA_RATIO:
        return False, "low_alpha"
    if source == "wikimedia":
        if meta.get("wiki") != "wikipedia.com" or str(meta.get("namespace")) != "0":
            return False, "wiki_filter"
    if source == "stackexchange" and meta.get("site") not in SE_ALLOWED_SITES:
        return False, "site_excluded"
    if source == "stackv2_edu" and (_truthy(meta.get("is_generated")) or _truthy(meta.get("is_vendor"))):
        return False, "generated_or_vendor"
    return True, "ok"


def stratum(source: str, meta) -> str | None:
    """Grouping key used for caps and language targets (None = no strata)."""
    if not isinstance(meta, dict):
        return None
    if source == "stackexchange":
        return meta.get("site")
    if source == "stackv2_edu":
        return meta.get("language")
    return None