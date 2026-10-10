from pathlib import Path

import pytest
import torch

from vibe.bench import cache_bytes, param_bytes, peak_mb, rss_mb, time_decode, time_prefill
from vibe.config import kv_cache_bytes_per_token, load_config
from vibe.model import Transformer

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Transformer(load_config(CONFIGS / "tiny.yaml")).eval()


def test_param_bytes_fp32(model):
    assert param_bytes(model) == model.num_parameters() * 4


def test_measured_cache_matches_formula(model):
    secs, caches = time_prefill(model, 20, reps=1)
    assert secs > 0
    assert cache_bytes(caches) == kv_cache_bytes_per_token(model.cfg, bytes_per_value=4) * 20


def test_decode_speed_positive(model):
    assert time_decode(model, 16, steps=4, reps=1) > 0


def test_decode_needs_room_in_context(model):
    with pytest.raises(ValueError):
        time_decode(model, model.cfg.max_seq_len - 2, steps=16)


def test_prefill_longer_than_context_rejected(model):
    with pytest.raises(ValueError):
        time_prefill(model, model.cfg.max_seq_len + 1)


def test_memory_probes_positive():
    assert rss_mb() > 0 and peak_mb() > 0