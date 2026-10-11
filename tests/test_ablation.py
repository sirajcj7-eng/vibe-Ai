import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vibe.ablation import apply_overrides, match_ff_dim, parse_values, run_one, summarize
from vibe.config import load_config, total_params
from vibe.data import TOKEN_DTYPE, TokenStream, write_shards
from vibe.train import TrainConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def tiny_cfg():
    return load_config(CONFIGS / "tiny.yaml")


@pytest.fixture()
def streams(tmp_path):
    def make(n, name):
        arr = np.tile(np.arange(1, 51, dtype=TOKEN_DTYPE), n)
        names = write_shards(arr, tmp_path / name, "s", 12_000)
        return TokenStream([tmp_path / name / x for x in names])

    return make(600, "train"), make(100, "val")


def test_parse_values():
    assert parse_values("1,2,4") == [1, 2, 4]
    assert parse_values("5e-4,1e-3") == [0.0005, 0.001]
    assert parse_values("fp16") == ["fp16"]
    assert parse_values("true") == [True]


def test_apply_overrides_splits_model_and_train_fields(tiny_cfg):
    m, t = apply_overrides(tiny_cfg, TrainConfig(), {"n_kv_heads": 1, "lr": 0.002})
    assert m.n_kv_heads == 1 and t.lr == 0.002
    assert tiny_cfg.n_kv_heads == 2  # original untouched


def test_apply_overrides_rejects_unknown_and_invalid(tiny_cfg):
    with pytest.raises(ValueError):
        apply_overrides(tiny_cfg, TrainConfig(), {"nope": 1})
    with pytest.raises(ValueError):
        apply_overrides(tiny_cfg, TrainConfig(), {"n_kv_heads": 3})  # 4 heads not divisible by 3


@pytest.mark.parametrize("kv", [1, 2, 4])
def test_match_ff_dim_keeps_parameter_count(tiny_cfg, kv):
    target = total_params(tiny_cfg)
    cand = match_ff_dim(replace(tiny_cfg, n_kv_heads=kv), target)
    assert cand.ff_dim % 32 == 0
    assert abs(total_params(cand) - target) / target < 0.02


def test_summarize_groups_and_handles_divergence():
    recs = [
        {"variant": "a", "val_bpb": 1.0, "n_params": 10},
        {"variant": "a", "val_bpb": 1.2, "n_params": 10},
        {"variant": "b", "val_bpb": 0.9, "n_params": 10},
        {"variant": "c", "val_bpb": None, "n_params": 10},
    ]
    rows = summarize(recs)
    assert [r["variant"] for r in rows] == ["b", "a", "c"]
    a = rows[1]
    assert a["n"] == 2 and a["mean"] == pytest.approx(1.1)
    assert a["std"] == pytest.approx(0.14142, abs=1e-4)
    assert rows[2]["mean"] is None and rows[2]["diverged"] == 1


def test_run_one_end_to_end(tiny_cfg, streams, tmp_path):
    tr, va = streams
    base = TrainConfig(steps=4, batch_size=4, seq_len=32, warmup_steps=1, decay_frac=0.5,
                       eval_every=10**9, eval_batches=2, log_every=10**9, ckpt_every=10**9)
    target = total_params(tiny_cfg)
    rec = run_one(tiny_cfg, base, {"n_kv_heads": 1}, 1, tr, va, tmp_path / "run",
                  tokens_per_byte=0.3, target_params=target)
    assert rec["diverged"] is False and rec["seed"] == 1
    assert math.isfinite(rec["val_loss"]) and rec["val_bpb"] > 0
    assert abs(rec["n_params"] - target) / target < 0.02