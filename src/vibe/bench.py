"""CPU speed and memory measurement helpers (docs 005 and 006)."""
from __future__ import annotations

import statistics
import time

import psutil
import torch

from vibe.model import Transformer


def rss_mb() -> float:
    """Current process memory in MB (1 MB = 1,000,000 bytes)."""
    return psutil.Process().memory_info().rss / 1e6


def peak_mb() -> float:
    """Peak process memory so far (exact on Windows, current value elsewhere)."""
    mem = psutil.Process().memory_info()
    return getattr(mem, "peak_wset", mem.rss) / 1e6


def param_bytes(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def cache_bytes(caches) -> int:
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in caches)


def _prompt(model: Transformer, length: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, model.cfg.vocab_size, (1, length), generator=g)


@torch.no_grad()
def time_prefill(model: Transformer, length: int, reps: int = 3):
    """Median seconds to process a prompt of `length` tokens, plus its KV caches."""
    if length > model.cfg.max_seq_len:
        raise ValueError(f"length {length} exceeds context {model.cfg.max_seq_len}")
    model.eval()
    ids = _prompt(model, length)
    _, _, caches = model(ids, use_cache=True)  # warm-up
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _, _, caches = model(ids, use_cache=True)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), caches


@torch.no_grad()
def time_decode(model: Transformer, context: int, steps: int = 16, reps: int = 3) -> float:
    """Median decode speed (tokens/s) when `context` tokens are already cached."""
    if context + steps > model.cfg.max_seq_len:
        raise ValueError("context + steps exceeds the model's context window")
    model.eval()
    _, _, start_caches = model(_prompt(model, context), use_cache=True)
    rates = []
    for _ in range(reps):
        caches = start_caches  # decoding never modifies the starting caches
        token = torch.zeros((1, 1), dtype=torch.long)
        t0 = time.perf_counter()
        for i in range(steps):
            logits, _, caches = model(token, kv_caches=caches, start_pos=context + i, use_cache=True)
            token = logits[:, -1:].argmax(-1)
        rates.append(steps / (time.perf_counter() - t0))
    return statistics.median(rates)