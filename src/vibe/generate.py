"""Text generation with a KV cache, plus checkpoint loading."""
from __future__ import annotations

from pathlib import Path

import torch

from vibe.config import ModelConfig
from vibe.model import Transformer


def load_model(path: str | Path, device: str | torch.device = "cpu") -> Transformer:
    """Load model_final.pt or last.pt (both store the weights and the config)."""
    state = torch.load(path, map_location=device, weights_only=True)
    model = Transformer(ModelConfig(**state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def sample_next(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator,
) -> int:
    """Pick one token id from a 1-D logits vector."""
    if temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits.float().cpu() / temperature
    if top_k > 0:
        k = min(top_k, logits.numel())
        kth = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = (cumulative - sorted_probs) > top_p  # keep the token that crosses p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        probs = torch.zeros_like(probs).scatter(0, sorted_idx, sorted_probs)
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.no_grad()
def generate(
    model: Transformer,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    eos_id: int | None = None,
    seed: int | None = None,
) -> list[int]:
    """Return only the newly generated token ids (temperature 0 = greedy)."""
    device = next(model.parameters()).device
    max_len = model.cfg.max_seq_len
    if not prompt_ids:
        raise ValueError("prompt must contain at least one token")
    if len(prompt_ids) > max_len:
        raise ValueError(f"prompt has {len(prompt_ids)} tokens, context is {max_len}")

    model.eval()
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    else:
        gen.seed()

    budget = min(max_new_tokens, max_len - len(prompt_ids))
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, _, caches = model(ids, use_cache=True)  # prefill the prompt once
    pos = ids.shape[1]

    out: list[int] = []
    while len(out) < budget:
        token = sample_next(logits[0, -1], temperature, top_k, top_p, gen)
        if eos_id is not None and token == eos_id:
            break
        out.append(token)
        if len(out) >= budget:
            break
        step = torch.tensor([[token]], dtype=torch.long, device=device)
        logits, _, caches = model(step, kv_caches=caches, start_pos=pos, use_cache=True)
        pos += 1
    return out