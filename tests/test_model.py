import math
from pathlib import Path

import pytest
import torch

from vibe.config import kv_cache_bytes_per_token, load_config, total_params
from vibe.model import Transformer, apply_rope, build_rope_cache

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def tiny_cfg():
    return load_config(CONFIGS / "tiny.yaml")


@pytest.fixture()
def tiny_model(tiny_cfg):
    torch.manual_seed(0)
    return Transformer(tiny_cfg).eval()


def rand_ids(cfg, batch, length, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (batch, length), generator=g)


@pytest.mark.parametrize("name", ["tiny.yaml", "50m.yaml"])
def test_param_count_matches_formula(name):
    cfg = load_config(CONFIGS / name)
    assert Transformer(cfg).num_parameters() == total_params(cfg)


def test_forward_shapes(tiny_model, tiny_cfg):
    ids = rand_ids(tiny_cfg, 3, 20)
    logits, loss, caches = tiny_model(ids, targets=ids)
    assert logits.shape == (3, 20, tiny_cfg.vocab_size)
    assert loss.ndim == 0
    assert caches is None


def test_initial_loss_near_ln_vocab(tiny_model, tiny_cfg):
    ids = rand_ids(tiny_cfg, 8, 64)
    with torch.no_grad():
        _, loss, _ = tiny_model(ids[:, :-1], targets=ids[:, 1:])
    assert abs(loss.item() - math.log(tiny_cfg.vocab_size)) < 0.25


def test_causality(tiny_model, tiny_cfg):
    a = rand_ids(tiny_cfg, 1, 16)
    b = a.clone()
    b[:, 10:] = (b[:, 10:] + 1) % tiny_cfg.vocab_size
    with torch.no_grad():
        la, _, _ = tiny_model(a)
        lb, _, _ = tiny_model(b)
    assert torch.allclose(la[:, :10], lb[:, :10], atol=1e-5)
    assert not torch.allclose(la[:, 10:], lb[:, 10:], atol=1e-5)


def test_kv_cache_decoding_matches_full_forward(tiny_model, tiny_cfg):
    ids = rand_ids(tiny_cfg, 1, 20)
    with torch.no_grad():
        full, _, _ = tiny_model(ids)
        pre, _, caches = tiny_model(ids[:, :12], use_cache=True)
        assert torch.allclose(pre, full[:, :12], atol=1e-4, rtol=1e-4)
        for t in range(12, 20):
            step, _, caches = tiny_model(
                ids[:, t : t + 1], kv_caches=caches, start_pos=t, use_cache=True
            )
            assert torch.allclose(step[:, 0], full[:, t], atol=1e-4, rtol=1e-4)


def test_chunked_prefill_with_cache(tiny_model, tiny_cfg):
    ids = rand_ids(tiny_cfg, 1, 12)
    with torch.no_grad():
        full, _, _ = tiny_model(ids)
        _, _, caches = tiny_model(ids[:, :8], use_cache=True)
        chunk, _, _ = tiny_model(
            ids[:, 8:12], kv_caches=caches, start_pos=8, use_cache=True
        )
    assert torch.allclose(chunk, full[:, 8:12], atol=1e-4, rtol=1e-4)


def test_cache_size_matches_config_formula(tiny_model, tiny_cfg):
    T = 10
    ids = rand_ids(tiny_cfg, 1, T)
    with torch.no_grad():
        _, _, caches = tiny_model(ids, use_cache=True)
    assert len(caches) == tiny_cfg.n_layers
    total_bytes = sum((k.numel() + v.numel()) * 4 for k, v in caches)  # fp32
    assert total_bytes == kv_cache_bytes_per_token(tiny_cfg, bytes_per_value=4) * T


def test_overfit_one_batch(tiny_cfg):
    torch.manual_seed(0)
    model = Transformer(tiny_cfg)
    ids = rand_ids(tiny_cfg, 2, 33)
    x, y = ids[:, :-1], ids[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    _, first, _ = model(x, targets=y)
    first = first.item()
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        _, loss, _ = model(x, targets=y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first


def test_all_parameters_get_gradients(tiny_cfg):
    torch.manual_seed(0)
    model = Transformer(tiny_cfg)
    ids = rand_ids(tiny_cfg, 2, 17)
    _, loss, _ = model(ids[:, :-1], targets=ids[:, 1:])
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_rope_scores_depend_only_on_relative_position():
    head_dim = 32
    cos, sin = build_rope_cache(head_dim, 64, 10000.0)
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, 1, 1, head_dim, generator=g)
    k = torch.randn(1, 1, 1, head_dim, generator=g)

    def score(m, n):
        qm = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
        kn = apply_rope(k, cos[n : n + 1], sin[n : n + 1])
        return (qm * kn).sum()

    assert torch.allclose(score(5, 2), score(25, 22), atol=1e-4)
    assert not torch.allclose(score(5, 2), score(5, 3), atol=1e-4)


def test_sequence_longer_than_context_rejected(tiny_model, tiny_cfg):
    ids = rand_ids(tiny_cfg, 1, tiny_cfg.max_seq_len + 1)
    with pytest.raises(ValueError):
        tiny_model(ids)