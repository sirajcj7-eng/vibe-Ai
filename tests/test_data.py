import numpy as np
import pytest
import torch

from vibe.data import (
    TOKEN_DTYPE,
    PackedLoader,
    TokenStream,
    iter_sequential,
    tokenize_documents,
    write_shards,
)
from vibe.tokenizer import decode_ids, train_tokenizer


def make_stream(tmp_path, tokens, shard_tokens):
    names = write_shards(tokens, tmp_path, "t", shard_tokens)
    return TokenStream([tmp_path / n for n in names]), names


def test_shards_roundtrip_across_boundaries(tmp_path):
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 4000, size=10_000).astype(TOKEN_DTYPE)
    stream, names = make_stream(tmp_path, tokens, 3_000)
    assert len(names) == 4
    assert stream.total == 10_000
    for start, length in [(0, 10), (2_995, 10), (2_990, 3_100), (0, 10_000), (9_990, 10)]:
        assert np.array_equal(stream.read(start, length), tokens[start : start + length])


def test_read_out_of_range_raises(tmp_path):
    stream, _ = make_stream(tmp_path, np.arange(100, dtype=TOKEN_DTYPE), 40)
    with pytest.raises(IndexError):
        stream.read(95, 10)


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok")
    corpus = d / "c.txt"
    corpus.write_text("\n".join(["hello world, this is text 123."] * 50), encoding="utf-8")
    return train_tokenizer([corpus], vocab_size=300)


def test_tokenize_documents_adds_eos(tok):
    arr = tokenize_documents(tok, ["hello world", "text 123"])
    assert arr.dtype == TOKEN_DTYPE
    eos = np.flatnonzero(arr == 0)
    assert len(eos) == 2 and eos[-1] == len(arr) - 1
    assert decode_ids(tok, arr[: eos[0]].tolist()) == "hello world"


def test_loader_shapes_and_target_shift(tmp_path):
    stream, _ = make_stream(tmp_path, np.arange(1, 1001, dtype=TOKEN_DTYPE), 400)
    loader = PackedLoader(stream, seq_len=10, batch_size=4, seed=0)
    x, y = loader.next_batch()
    assert x.shape == (4, 10) and y.shape == (4, 10)
    assert x.dtype == torch.int64
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(y[:, -1], x[:, -1] + 1)


def test_epoch_visits_every_sequence_once(tmp_path):
    stream, _ = make_stream(tmp_path, np.arange(1, 1001, dtype=TOKEN_DTYPE), 400)
    loader = PackedLoader(stream, seq_len=10, batch_size=3, seed=0)
    assert loader.n_seq == 99 and loader.batches_per_epoch == 33
    firsts = []
    for _ in range(33):
        x, _ = loader.next_batch()
        firsts += x[:, 0].tolist()
    assert sorted(firsts) == [10 * k + 1 for k in range(99)]
    assert loader.epoch == 1 and loader.batch_in_epoch == 0


def test_same_seed_same_batches(tmp_path):
    stream, _ = make_stream(tmp_path, np.arange(1, 1001, dtype=TOKEN_DTYPE), 400)
    a = PackedLoader(stream, 10, 4, seed=0)
    b = PackedLoader(stream, 10, 4, seed=0)
    c = PackedLoader(stream, 10, 4, seed=1)
    xa = [a.next_batch()[0] for _ in range(5)]
    xb = [b.next_batch()[0] for _ in range(5)]
    xc = [c.next_batch()[0] for _ in range(5)]
    assert all(torch.equal(p, q) for p, q in zip(xa, xb))
    assert not all(torch.equal(p, q) for p, q in zip(xa, xc))


def test_resume_reproduces_batches_across_epochs(tmp_path):
    stream, _ = make_stream(tmp_path, np.arange(1, 202, dtype=TOKEN_DTYPE), 70)
    a = PackedLoader(stream, 10, 4, seed=3)  # 20 sequences, 5 batches per epoch
    for _ in range(3):
        a.next_batch()
    state = a.state_dict()
    expected = [a.next_batch() for _ in range(12)]  # crosses two epoch boundaries

    b = PackedLoader(stream, 10, 4, seed=3)
    b.load_state_dict(state)
    for ex, ey in expected:
        x, y = b.next_batch()
        assert torch.equal(x, ex) and torch.equal(y, ey)
    assert (a.epoch, a.batch_in_epoch) == (b.epoch, b.batch_in_epoch)


def test_resume_with_different_settings_rejected(tmp_path):
    stream, _ = make_stream(tmp_path, np.arange(1, 1001, dtype=TOKEN_DTYPE), 400)
    a = PackedLoader(stream, 10, 4, seed=0)
    state = a.state_dict()
    state["seq_len"] = 20
    with pytest.raises(ValueError):
        a.load_state_dict(state)


def test_too_little_data_rejected(tmp_path):
    stream, _ = make_stream(tmp_path, np.arange(1, 22, dtype=TOKEN_DTYPE), 40)
    with pytest.raises(ValueError):
        PackedLoader(stream, seq_len=10, batch_size=4)


def test_sequential_batches_are_in_order(tmp_path):
    stream, _ = make_stream(tmp_path, np.arange(1, 1001, dtype=TOKEN_DTYPE), 400)
    batches = list(iter_sequential(stream, seq_len=10, batch_size=3))
    assert len(batches) == 33
    x0, _ = batches[0]
    assert x0[0, 0].item() == 1 and x0[1, 0].item() == 11
    assert len(list(iter_sequential(stream, 10, 3, max_batches=2))) == 2