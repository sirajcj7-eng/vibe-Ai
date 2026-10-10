"""Unigram and bigram baselines (doc 005): the cheap bars a model must beat."""
from __future__ import annotations

import numpy as np

from vibe.data import TokenStream, iter_sequential

MAX_BIGRAM_VOCAB = 8192  # dense count matrix; use a sparse version for larger vocabularies


class UnigramModel:
    def __init__(self, train_tokens: np.ndarray, vocab_size: int, add_k: float = 0.1) -> None:
        counts = np.bincount(
            np.asarray(train_tokens, dtype=np.int64), minlength=vocab_size
        ).astype(np.float64)
        self.probs = (counts + add_k) / (counts.sum() + add_k * vocab_size)

    def nll(self, x: np.ndarray, y: np.ndarray) -> float:
        """Mean negative log-likelihood (nats/token); x is ignored."""
        return float(-np.log(self.probs[y]).mean())


class BigramModel:
    """P(w | v) = lam * count(v, w) / count(v) + (1 - lam) * P_unigram(w)."""

    def __init__(
        self,
        train_tokens: np.ndarray,
        vocab_size: int,
        lam: float = 0.8,
        add_k: float = 0.1,
    ) -> None:
        if vocab_size > MAX_BIGRAM_VOCAB:
            raise ValueError("vocabulary too large for the dense bigram baseline")
        t = np.asarray(train_tokens, dtype=np.int64)
        self.vocab_size = vocab_size
        self.lam = lam
        self.uni = UnigramModel(t, vocab_size, add_k)
        flat = np.bincount(t[:-1] * vocab_size + t[1:], minlength=vocab_size * vocab_size)
        self.counts = flat.astype(np.float32).reshape(vocab_size, vocab_size)
        self.rowsum = self.counts.sum(axis=1)

    def _probs_for(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        c = self.counts[x, y]
        r = self.rowsum[x]
        seen = r > 0
        p_bi = np.where(seen, c / np.maximum(r, 1.0), 0.0)
        lam = np.where(seen, self.lam, 0.0)  # unseen context falls back to unigram
        return lam * p_bi + (1.0 - lam) * self.uni.probs[y]

    def probs_given(self, v: int) -> np.ndarray:
        """Full next-token distribution after token v."""
        if self.rowsum[v] <= 0:
            return self.uni.probs.copy()
        return self.lam * self.counts[v] / self.rowsum[v] + (1.0 - self.lam) * self.uni.probs

    def nll(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(-np.log(self._probs_for(x, y)).mean())

    @classmethod
    def fit(
        cls,
        train_tokens: np.ndarray,
        vocab_size: int,
        add_k: float = 0.1,
        candidates: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 0.95),
    ) -> tuple["BigramModel", float]:
        """Choose lam on the last 5% of the training data, then refit on all of it."""
        t = np.asarray(train_tokens, dtype=np.int64)
        cut = int(len(t) * 0.95)
        head, tail = t[:cut], t[cut:]
        trial = cls(head, vocab_size, lam=candidates[0], add_k=add_k)
        best_lam, best = candidates[0], float("inf")
        for lam in candidates:
            trial.lam = lam
            loss = trial.nll(tail[:-1], tail[1:])
            if loss < best:
                best, best_lam = loss, lam
        del trial
        return cls(t, vocab_size, lam=best_lam, add_k=add_k), best_lam


def baseline_val_loss(
    model, stream: TokenStream, seq_len: int, batch_size: int, max_batches: int | None = None
) -> float:
    """Same validation windows the transformer is scored on."""
    total, n = 0.0, 0
    for x, y in iter_sequential(stream, seq_len, batch_size, max_batches):
        total += model.nll(x.numpy().ravel(), y.numpy().ravel())
        n += 1
    if n == 0:
        raise ValueError("validation set too small")
    return total / n