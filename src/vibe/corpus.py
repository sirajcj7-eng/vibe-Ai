"""Corpus builder core (Phase 7b): validation split, token budgets, shard
writer and the filter -> tokenize -> write pipeline."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from tokenizers import Tokenizer

from vibe.data import MAX_VOCAB, TOKEN_DTYPE
from vibe.filters import check_document, stratum as stratum_of
from vibe.tokenizer import EOS_TOKEN


def doc_key(doc: dict) -> str:
    """Stable identifier used for the train/validation split."""
    if doc.get("id"):
        return str(doc["id"])
    meta = doc.get("metadata")
    if isinstance(meta, dict) and meta.get("provenance"):
        return str(meta["provenance"])
    return hashlib.sha1((doc.get("text") or "")[:1000].encode("utf-8")).hexdigest()


def is_val_doc(key: str, val_permille: int = 5) -> bool:
    """True for about val_permille / 1000 of all documents, decided by hash."""
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 1000 < val_permille


class StratumBudget:
    """Token budget for one source, with optional per-stratum caps.

    caps: stratum -> maximum fraction of `total` that stratum may use.
    exclusive: only strata listed in caps are accepted (for code languages).
    A stratum can overshoot its cap by at most one document.
    """

    def __init__(self, total: int, caps: dict[str, float] | None = None,
                 exclusive: bool = False) -> None:
        self.total = total
        self.caps = {k: int(v * total) for k, v in (caps or {}).items()}
        self.exclusive = exclusive
        self.used: Counter = Counter()
        self.used_total = 0

    def wants(self, stratum: str | None) -> bool:
        if self.used_total >= self.total:
            return False
        cap = self.caps.get(stratum)
        if cap is None:
            return not self.exclusive
        return self.used[stratum] < cap

    def add(self, stratum: str | None, n_tokens: int) -> None:
        self.used[stratum] += n_tokens
        self.used_total += n_tokens

    @property
    def done(self) -> bool:
        if self.used_total >= self.total:
            return True
        if self.exclusive:
            return all(self.used[k] >= cap for k, cap in self.caps.items())
        return False


class TokenShardWriter:
    """Append token arrays; write fixed-size uint16 shards under root/subdir."""

    def __init__(self, root: str | Path, subdir: str, prefix: str,
                 shard_tokens: int = 50_000_000) -> None:
        self.dir = Path(root) / subdir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.subdir, self.prefix, self.shard_tokens = subdir, prefix, shard_tokens
        self.buf: list[np.ndarray] = []
        self.buffered = 0
        self.total = 0
        self.names: list[str] = []

    def add(self, tokens: np.ndarray) -> None:
        if len(tokens) == 0:
            return
        self.buf.append(np.asarray(tokens, dtype=TOKEN_DTYPE))
        self.buffered += len(tokens)
        self.total += len(tokens)
        while self.buffered >= self.shard_tokens:
            self._flush(self.shard_tokens)

    def _flush(self, n: int) -> None:
        data = np.concatenate(self.buf)
        shard, rest = data[:n], data[n:]
        name = f"{self.prefix}_{len(self.names):03d}.bin"
        shard.tofile(self.dir / name)
        self.names.append(f"{self.subdir}/{name}")
        self.buf = [rest] if len(rest) else []
        self.buffered = len(rest)

    def close(self) -> list[str]:
        if self.buffered:
            self._flush(self.buffered)
        return self.names


def encode_documents(tok: Tokenizer, texts: list[str]) -> list[np.ndarray]:
    """One uint16 array per document, each ending with the EOS token."""
    eos_id = tok.token_to_id(EOS_TOKEN)
    if eos_id is None:
        raise ValueError(f"tokenizer has no {EOS_TOKEN} token")
    if tok.get_vocab_size() > MAX_VOCAB:
        raise ValueError("vocabulary too large for uint16 storage")
    out = []
    for enc in tok.encode_batch(list(texts)):  # batch encoding uses all cores
        arr = np.empty(len(enc.ids) + 1, dtype=TOKEN_DTYPE)
        arr[:-1] = enc.ids
        arr[-1] = eos_id
        out.append(arr)
    return out


@dataclass
class SourceResult:
    docs_seen: int = 0
    docs_kept_train: int = 0
    docs_kept_val: int = 0
    train_tokens: int = 0
    val_tokens: int = 0
    val_bytes: int = 0
    drops: Counter = field(default_factory=Counter)
    strata_tokens: Counter = field(default_factory=Counter)
    reached_budget: bool = False

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["drops"] = dict(self.drops)
        d["strata_tokens"] = dict(self.strata_tokens)
        return d


def process_documents(
    source: str,
    docs: Iterable[dict],
    tok: Tokenizer,
    budget: StratumBudget,
    train_writer: TokenShardWriter,
    val_writer: TokenShardWriter,
    *,
    batch_chars: int = 4_000_000,
    blocklist: Callable[[str], bool] | None = None,
    val_permille: int = 5,
    max_docs: int | None = None,
) -> SourceResult:
    """Filter, tokenize and write documents until the budget is reached."""
    res = SourceResult()
    pending: list[tuple[str, str | None, bool]] = []
    pending_chars = 0

    def flush() -> None:
        nonlocal pending, pending_chars
        if not pending:
            return
        arrays = encode_documents(tok, [p[0] for p in pending])
        for (text, stratum, is_val), arr in zip(pending, arrays):
            if is_val:
                val_writer.add(arr)
                res.docs_kept_val += 1
                res.val_tokens += len(arr)
                res.val_bytes += len(text.encode("utf-8"))
            elif budget.wants(stratum):
                train_writer.add(arr)
                budget.add(stratum, len(arr))
                res.docs_kept_train += 1
                res.train_tokens += len(arr)
                res.strata_tokens[stratum or "-"] += len(arr)
            else:
                res.drops["stratum_full"] += 1
        pending, pending_chars = [], 0

    for doc in docs:
        if max_docs is not None and res.docs_seen >= max_docs:
            break
        res.docs_seen += 1
        text = doc.get("text") or ""
        meta = doc.get("metadata")
        keep, reason = check_document(source, text, meta)
        if not keep:
            res.drops[reason] += 1
            continue
        if blocklist is not None and blocklist(text):
            res.drops["contaminated"] += 1
            continue
        stratum = stratum_of(source, meta)
        is_val = is_val_doc(doc_key(doc), val_permille)
        if not is_val and not budget.wants(stratum):
            res.drops["stratum_full"] += 1
            continue
        pending.append((text, stratum, is_val))
        pending_chars += len(text)
        if pending_chars >= batch_chars:
            flush()
            if budget.done:
                break
    flush()
    res.reached_budget = budget.done
    return res


def collect_text(source: str, docs: Iterable[dict], quota_chars: int, out) -> tuple:
    """Write passing documents as JSON lines until quota_chars is reached.

    Documents that fall in the validation split are skipped, so no held-out
    text is used to build the tokenizer. Returns (documents, characters, drops).
    """
    drops: Counter = Counter()
    kept = chars = 0
    for doc in docs:
        text = doc.get("text") or ""
        ok, reason = check_document(source, text, doc.get("metadata"))
        if not ok:
            drops[reason] += 1
            continue
        if is_val_doc(doc_key(doc)):
            drops["val_holdout"] += 1
            continue
        out.write(json.dumps({"text": text.replace("\r\n", "\n")}, ensure_ascii=False) + "\n")
        kept += 1
        chars += len(text)
        if chars >= quota_chars:
            break
    return kept, chars, drops