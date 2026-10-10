from pathlib import Path

import numpy as np
import pytest

from vibe.config import load_config
from vibe.data import TOKEN_DTYPE, TokenStream, write_shards
from vibe.model import Transformer
from vibe.train import TrainConfig, build_optimizer, lr_at, train

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def tiny_cfg():
    return load_config(CONFIGS / "tiny.yaml")


@pytest.fixture()
def streams(tmp_path):
    """A learnable stream: the token sequence 1..50 repeated."""
    def make(n_repeats, name):
        arr = np.tile(np.arange(1, 51, dtype=TOKEN_DTYPE), n_repeats)
        names = write_shards(arr, tmp_path / name, "s", 12_000)
        return TokenStream([tmp_path / name / n for n in names])

    return make(600, "train"), make(100, "val")


def tcfg(**kw):
    base = dict(
        steps=6, batch_size=4, grad_accum=1, seq_len=32, lr=3e-3,
        warmup_steps=2, decay_frac=0.34, eval_every=1000, eval_batches=2,
        log_every=1000, ckpt_every=1000, seed=0,
    )
    base.update(kw)
    return TrainConfig(**base)


def test_lr_schedule_shape():
    t = TrainConfig(steps=100, lr=1e-3, warmup_steps=10, decay_frac=0.2, min_lr_frac=0.1)
    lrs = [lr_at(s, t) for s in range(100)]
    assert lrs[0] == pytest.approx(1e-4)
    assert lrs[9] == pytest.approx(1e-3)
    assert all(lr == pytest.approx(1e-3) for lr in lrs[10:80])
    assert lrs[-1] == pytest.approx(1e-4)
    assert all(a >= b for a, b in zip(lrs[79:], lrs[80:]))
    assert all(1e-4 * 0.999 <= lr <= 1e-3 * 1.001 for lr in lrs)


def test_weight_decay_groups(tiny_cfg):
    model = Transformer(tiny_cfg)
    decay, no_decay = build_optimizer(model, tcfg()).param_groups
    n_decay = sum(p.numel() for p in decay["params"])
    n_emb = tiny_cfg.vocab_size * tiny_cfg.d_model
    n_norm = sum(p.numel() for n, p in model.named_parameters() if "norm" in n)
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert n_decay == model.num_parameters() - n_emb - n_norm


def test_invalid_config_rejected(tiny_cfg):
    with pytest.raises(ValueError):
        tcfg(seq_len=tiny_cfg.max_seq_len + 1).validate(tiny_cfg)
    with pytest.raises(ValueError):
        tcfg(steps=0).validate(tiny_cfg)


def test_loss_decreases_on_learnable_data(tiny_cfg, streams, tmp_path):
    tr, va = streams
    out = train(tiny_cfg, tcfg(steps=40, warmup_steps=5), tr, va,
                tmp_path / "run", verbose=False)
    assert out["losses"][-1] < 0.5 * out["losses"][0]
    assert out["final"]["val_loss"] < out["val"][0]["val_loss"]


def test_resume_reproduces_losses_exactly(tiny_cfg, streams, tmp_path):
    tr, va = streams
    full = train(tiny_cfg, tcfg(), tr, va, tmp_path / "a", verbose=False)
    assert len(full["losses"]) == 6

    first = train(tiny_cfg, tcfg(stop_at=3), tr, va, tmp_path / "b", verbose=False)
    assert first["step"] == 3 and first["final"] is None
    second = train(tiny_cfg, tcfg(resume=True), tr, va, tmp_path / "b", verbose=False)
    assert second["step"] == 6
    np.testing.assert_allclose(second["losses"], full["losses"][3:], rtol=0, atol=1e-6)


def test_resume_with_changed_batch_size_rejected(tiny_cfg, streams, tmp_path):
    tr, va = streams
    train(tiny_cfg, tcfg(stop_at=3), tr, va, tmp_path / "r", verbose=False)
    with pytest.raises(ValueError):
        train(tiny_cfg, tcfg(resume=True, batch_size=2), tr, va, tmp_path / "r", verbose=False)


def test_existing_checkpoint_not_overwritten_without_resume(tiny_cfg, streams, tmp_path):
    tr, va = streams
    train(tiny_cfg, tcfg(stop_at=2), tr, va, tmp_path / "o", verbose=False)
    with pytest.raises(FileExistsError):
        train(tiny_cfg, tcfg(), tr, va, tmp_path / "o", verbose=False)


def test_gradient_accumulation_matches_big_batch(tiny_cfg, streams, tmp_path):
    tr, va = streams
    big = train(tiny_cfg, tcfg(steps=3, batch_size=4, grad_accum=1), tr, va,
                tmp_path / "big", verbose=False)
    acc = train(tiny_cfg, tcfg(steps=3, batch_size=2, grad_accum=2), tr, va,
                tmp_path / "acc", verbose=False)
    np.testing.assert_allclose(acc["losses"], big["losses"], rtol=0, atol=1e-4)