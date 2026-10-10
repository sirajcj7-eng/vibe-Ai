import math

import numpy as np
import pytest

from vibe.baselines import BigramModel, UnigramModel, baseline_val_loss
from vibe.data import TOKEN_DTYPE, TokenStream, write_shards


def test_unigram_loss_by_hand():
    m = UnigramModel(np.array([0, 0, 0, 1]), vocab_size=2, add_k=0.0)
    expected = -(math.log(0.75) + math.log(0.25)) / 2
    assert m.nll(np.array([0, 0]), np.array([0, 1])) == pytest.approx(expected)


def test_bigram_perfect_on_deterministic_transitions():
    train = np.array([0, 1] * 50)
    m = BigramModel(train, vocab_size=2, lam=1.0, add_k=0.0)
    assert m.nll(np.array([0, 1, 0, 1]), np.array([1, 0, 1, 0])) == pytest.approx(0.0, abs=1e-6)


def test_bigram_unseen_context_falls_back_to_unigram():
    train = np.array([0, 1])
    m = BigramModel(train, vocab_size=3, lam=0.8)
    x, y = np.array([2, 2]), np.array([0, 1])
    assert m.nll(x, y) == pytest.approx(m.uni.nll(x, y))


def test_bigram_distribution_sums_to_one_and_matches_nll():
    rng = np.random.default_rng(0)
    train = rng.integers(0, 6, size=500)
    m = BigramModel(train, vocab_size=6, lam=0.7)
    for v in range(6):
        assert m.probs_given(v).sum() == pytest.approx(1.0, abs=1e-6)
    p = m.probs_given(2)
    assert m.nll(np.array([2]), np.array([4])) == pytest.approx(-math.log(p[4]), rel=1e-5)


def test_baselines_on_a_stream(tmp_path):
    tokens = np.tile(np.array([0, 1], dtype=TOKEN_DTYPE), 2000)
    names = write_shards(tokens, tmp_path, "t", 1500)
    stream = TokenStream([tmp_path / n for n in names])
    uni = UnigramModel(tokens, 2)
    bi = BigramModel(tokens, 2, lam=0.95)
    assert baseline_val_loss(uni, stream, 10, 4) == pytest.approx(math.log(2), abs=0.01)
    assert baseline_val_loss(bi, stream, 10, 4) < 0.1


def test_bigram_rejects_huge_vocab():
    with pytest.raises(ValueError):
        BigramModel(np.array([0, 1, 2]), vocab_size=20_000)