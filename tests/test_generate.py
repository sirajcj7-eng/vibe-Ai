from pathlib import Path

import pytest
import torch

from vibe.config import load_config
from vibe.generate import generate, sample_next
from vibe.model import Transformer

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PROMPT = [5, 17, 99, 3, 42]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Transformer(load_config(CONFIGS / "tiny.yaml")).eval()


def test_greedy_matches_full_forward_decoding(model):
    out = generate(model, PROMPT, max_new_tokens=8, temperature=0)
    ids = list(PROMPT)
    with torch.no_grad():
        for _ in range(8):
            logits, _, _ = model(torch.tensor([ids]))
            ids.append(int(logits[0, -1].argmax()))
    assert out == ids[len(PROMPT):]


def test_seed_controls_sampling(model):
    kw = dict(temperature=1.0, top_k=0, top_p=1.0)
    a = generate(model, PROMPT, 20, seed=1, **kw)
    b = generate(model, PROMPT, 20, seed=1, **kw)
    c = generate(model, PROMPT, 20, seed=2, **kw)
    assert a == b and a != c


def test_top_k_one_is_greedy(model):
    greedy = generate(model, PROMPT, 10, temperature=0)
    k1 = generate(model, PROMPT, 10, temperature=1.0, top_k=1, seed=3)
    assert greedy == k1


def test_stops_at_eos(model):
    first = generate(model, PROMPT, 1, temperature=0)[0]
    assert generate(model, PROMPT, 10, temperature=0, eos_id=first) == []


def test_respects_context_limit(model):
    prompt = [7] * (model.cfg.max_seq_len - 3)
    assert len(generate(model, prompt, 100, temperature=0)) == 3


def test_invalid_prompts_rejected(model):
    with pytest.raises(ValueError):
        generate(model, [], 5)
    with pytest.raises(ValueError):
        generate(model, [1] * (model.cfg.max_seq_len + 1), 5)


def test_top_k_only_picks_top_tokens():
    g = torch.Generator().manual_seed(0)
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    picks = {sample_next(logits, 1.0, 2, 1.0, g) for _ in range(200)}
    assert picks <= {2, 3}


def test_top_p_keeps_dominant_token():
    g = torch.Generator().manual_seed(0)
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0])
    assert {sample_next(logits, 1.0, 0, 0.5, g) for _ in range(50)} == {0}