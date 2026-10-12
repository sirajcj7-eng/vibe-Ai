"""License allow-list for the training corpus (doc 004 rule 1, doc 008)."""
from __future__ import annotations

import re

ALLOWED = frozenset({
    "public-domain", "cc0", "cc-by", "cc-by-sa",
    "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "unlicense",
})

# Sources whose license is verified per source or domain, not per document.
SOURCE_LEVEL_LICENSE = frozenset({"cccc"})

# Metadata field that holds the document license (default: "license").
LICENSE_FIELD = {"pes2o": "oa_license"}

_RESTRICTED = (
    "noncommercial", "non-commercial", "non commercial", "by-nc", "bync",
    "by-nd", "bynd", "no derivatives", "noderivatives", "non-exclusive",
    "nonexclusive", "all rights reserved", "gpl",
)


def canonical_license(value) -> str:
    """Map one license string to a canonical label.

    Returns an ALLOWED label, 'restricted', 'missing', or 'unknown:<text>'.
    """
    if value is None:
        return "missing"
    v = str(value).strip().strip("[]'\" ").lower()
    if v in ("", "none", "null", "nan"):
        return "missing"
    if any(m in v for m in _RESTRICTED):
        return "restricted"
    if ("publicdomain/zero" in v or "creative commons zero" in v
            or v in ("cc0", "cc0-1.0") or v.startswith("cc0")):
        return "cc0"
    if "public domain" in v or v in ("pd", "publicdomain"):
        return "public-domain"
    if ("attribution share-alike" in v or "/by-sa/" in v
            or v in ("ccbysa", "cc-by-sa") or v.startswith("cc-by-sa")):
        return "cc-by-sa"
    if ("creativecommons.org/licenses/by/" in v
            or v.startswith("creative commons - attribution")
            or v in ("ccby", "cc-by") or v.startswith("cc-by-")):
        return "cc-by"
    if v == "mit":
        return "mit"
    if v in ("apache-2.0", "apache 2.0", "apache-2"):
        return "apache-2.0"
    if v == "bsd-3-clause":
        return "bsd-3-clause"
    if v == "bsd-2-clause":
        return "bsd-2-clause"
    if v == "isc":
        return "isc"
    if v == "unlicense":
        return "unlicense"
    return f"unknown:{v[:40]}"


def doc_license_status(meta, field: str = "license", source_level: bool = False):
    """Return (label, allowed) for one document's metadata."""
    if source_level:
        return "source-verified", True
    value = meta.get(field) if isinstance(meta, dict) else None
    if value is None:
        return "missing", False
    parts = [p for p in re.split(r"[,;]", str(value).strip().strip("[]")) if p.strip()]
    labels = [canonical_license(p) for p in parts] or ["missing"]
    allowed = all(label in ALLOWED for label in labels)
    return "+".join(sorted(set(labels))), allowed