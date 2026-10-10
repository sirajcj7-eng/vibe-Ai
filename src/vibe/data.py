"""Token shards, packing, and a deterministic, resumable batch loader.

Documents are tokenized, separated by an EOS token, and concatenated into
one flat stream stored as uint16 shards. Training sequences are
non-overlapping windows of seq_len + 1 tokens: inputs are the first
seq_len tokens, targets are the same window shifted by one. Sequences may
cross document boundaries (no cross-document masking at this stage).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from vibe.tokenizer import EOS_TOKEN

TOKEN_DTYPE = np.dtype("<u2")  # little-endian uint16, fits vocab <= 65,536
MAX_VOCAB = 2**16


def tokenize_documents(tok: Tokenizer, texts: list[str]) -> np.ndarray:
    """Encode documents and append an EOS token after each one."""
    eos_id = tok.token_to_id(EOS_TOKEN)
    if eos_id is None:
        raise ValueError(f"tokenizer has no {EOS_TOKEN} token")
    if tok.get_vocab_size() > MAX_VOCAB:
        raise ValueError("vocabulary too large for uint16 storage")

    parts: list[np.ndarray] = []
    for enc in tok.encode_batch(list(texts)):  # batch encoding uses all cores
        parts.append(np.asarray(enc.ids, dtype=TOKEN_DTYPE))
        parts.append(np.array([eos_id], dtype=TOKEN_DTYPE))
    if not parts:
        return np.empty(0, dtype=TOKEN_DTYPE)
    return np.concatenate(parts)


def write_shards(
    tokens: np.ndarray, out_dir: str | Path, prefix: str, shard_tokens: int
) -> list[str]:
    """Write tokens as raw uint16 files of at most shard_tokens each."""
    if shard_tokens <= 0:
        raise ValueError("shard_tokens must be positive")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for i, start in enumerate(range(0, len(tokens), shard_tokens)):
        name = f"{prefix}_{i:03d}.bin"
        tokens[start : start + shard_tokens].astype(TOKEN_DTYPE).tofile(out_dir / name)
        names.append(name)
    return names


class TokenStream:
    """Read-only view over several shards, addressed as one flat array."""

    def __init__(self, paths: list[str | Path]) -> None:
        self.maps = [np.memmap(p, dtype=TOKEN_DTYPE, mode="r") for p in paths]
        self.lengths = [len(m) for m in self.maps]
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)]).astype(np.int64)
        self.total = int(self.offsets[-1])

    def read(self, start: int, length: int) -> np.ndarray:
        if start < 0 or length < 0 or start + length > self.total:
            raise IndexError(f"read({start}, {length}) outside stream of {self.total}")
        out = np.empty(length, dtype=TOKEN_DTYPE)
        filled, pos = 0, start
        shard = int(np.searchsorted(self.offsets, pos, side="right") - 1)
        while filled < length:
            local = pos - int(self.offsets[shard])
            take = min(length - filled, self.lengths[shard] - local)
            out[filled : filled + take] = self.maps[shard][local : local + take]
            filled += take
            pos += take
            shard += 1
        return out


def open_split(data_dir: str | Path, split: str) -> tuple[TokenStream, dict]:
    """Open the 'train' or 'val' shards listed in meta.json."""
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    paths = [data_dir / name for name in meta[split]["shards"]]
    return TokenStream(paths), meta


def _to_xy(rows: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    arr = np.stack(rows).astype(np.int64)
    x = torch.from_numpy(np.ascontiguousarray(arr[:, :-1]))
    y = torch.from_numpy(np.ascontiguousarray(arr[:, 1:]))
    return x, y


class PackedLoader:
    """Deterministic shuffled batches with exact resume.

    Each epoch uses permutation(seed + epoch) over all windows. State is
    just (epoch, batch_in_epoch), so a resumed run sees identical batches.
    The last partial batch of an epoch is dropped.
    """

    def __init__(
        self,
        stream: TokenStream,
        seq_len: int,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        self.stream = stream
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.n_seq = (stream.total - 1) // seq_len
        if self.n_seq < batch_size:
            raise ValueError(
                f"only {self.n_seq} sequences available, need at least {batch_size}"
            )
        self.batches_per_epoch = self.n_seq // batch_size
        self.epoch = 0
        self.batch_in_epoch = 0
        self._perm: np.ndarray | None = None
        self._perm_epoch = -1

    def _order(self) -> np.ndarray:
        if self._perm is None or self._perm_epoch != self.epoch:
            if self.shuffle:
                rng = np.random.default_rng(self.seed + self.epoch)
                self._perm = rng.permutation(self.n_seq)
            else:
                self._perm = np.arange(self.n_seq)
            self._perm_epoch = self.epoch
        return self._perm

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        perm = self._order()
        lo = self.batch_in_epoch * self.batch_size
        idx = perm[lo : lo + self.batch_size]
        rows = [self.stream.read(int(i) * self.seq_len, self.seq_len + 1) for i in idx]
        self.batch_in_epoch += 1
        if self.batch_in_epoch == self.batches_per_epoch:
            self.epoch += 1
            self.batch_in_epoch = 0
        return _to_xy(rows)

    def state_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "seed": self.seed,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "n_seq": self.n_seq,
        }

    def load_state_dict(self, state: dict) -> None:
        for key in ("seed", "seq_len", "batch_size", "n_seq"):
            if state[key] != getattr(self, key):
                raise ValueError(
                    f"cannot resume: {key} differs "
                    f"(saved {state[key]}, current {getattr(self, key)})"
                )
        self.epoch = int(state["epoch"])
        self.batch_in_epoch = int(state["batch_in_epoch"])
        self._perm = None
        self._perm_epoch = -1


def iter_sequential(
    stream: TokenStream,
    seq_len: int,
    batch_size: int,
    max_batches: int | None = None,
):
    """Fixed-order batches (for validation). Does not touch any loader state."""
    n_batches = ((stream.total - 1) // seq_len) // batch_size
    if max_batches is not None:
        n_batches = min(n_batches, max_batches)
    for b in range(n_batches):
        rows = [
            stream.read((b * batch_size + j) * seq_len, seq_len + 1)
            for j in range(batch_size)
        ]
        yield _to_xy(rows)