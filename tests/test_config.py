from pathlib import Path

import pytest

from vibe.config import (
    ModelConfig,
    kv_cache_bytes_per_token,
    load_config,
    param_breakdown,
    total_params,
    weight_megabytes,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_tiny_total():
    assert total_params(load_config(CONFIGS / "tiny.yaml")) == 1_262_720


def test_50m_total():
    assert total_params(load_config(CONFIGS / "50m.yaml")) == 50_348_544


def test_50m_components_match_doc_001():
    b = param_breakdown(load_config(CONFIGS / "50m.yaml"))
    assert b["embedding"] == 8_388_608
    assert b["attention_per_layer"] == 655_360
    assert b["mlp_per_layer"] == 1_966_080
    assert b["norms_per_layer"] == 1_024
    assert b["per_layer"] == 2_622_464
    assert b["all_layers"] == 41_959_424
    assert b["final_norm"] == 512


def test_untied_adds_output_head():
    cfg = load_config(CONFIGS / "50m.yaml")
    untied = ModelConfig(**{**cfg.__dict__, "tie_embeddings": False})
    assert total_params(untied) - total_params(cfg) == 16_384 * 512


def test_kv_cache_50m():
    cfg = load_config(CONFIGS / "50m.yaml")
    assert kv_cache_bytes_per_token(cfg) == 8_192
    assert kv_cache_bytes_per_token(cfg, n_kv_heads=8) == 32_768
    assert kv_cache_bytes_per_token(cfg) * 2048 == 16 * 1024 * 1024


def test_weight_megabytes_50m():
    n = total_params(load_config(CONFIGS / "50m.yaml"))
    assert round(weight_megabytes(n, 32), 1) == 201.4
    assert round(weight_megabytes(n, 16), 1) == 100.7
    assert round(weight_megabytes(n, 8), 1) == 50.3
    assert round(weight_megabytes(n, 4.5), 1) == 28.3


def test_invalid_heads_rejected():
    with pytest.raises(ValueError):
        ModelConfig(
            name="bad", vocab_size=100, d_model=128, n_layers=2,
            n_heads=4, n_kv_heads=3, head_dim=32, ff_dim=256, max_seq_len=64,
        )


def test_invalid_dim_rejected():
    with pytest.raises(ValueError):
        ModelConfig(
            name="bad", vocab_size=100, d_model=100, n_layers=2,
            n_heads=4, n_kv_heads=2, head_dim=32, ff_dim=256, max_seq_len=64,
        )