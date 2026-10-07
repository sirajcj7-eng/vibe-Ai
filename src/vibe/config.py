"""Model configuration and analytic parameter / memory formulas.

These formulas implement the math in docs/decisions/001-model-shape.md.
No tensors are created here: this file only counts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ff_dim: int
    max_seq_len: int
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        for field in (
            "vocab_size", "d_model", "n_layers", "n_heads",
            "n_kv_heads", "head_dim", "ff_dim", "max_seq_len",
        ):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError(
                f"n_heads * head_dim ({self.n_heads * self.head_dim}) "
                f"must equal d_model ({self.d_model})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even (RoPE rotates pairs)")


def load_config(path: str | Path) -> ModelConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ModelConfig(**data)


def param_breakdown(cfg: ModelConfig) -> dict[str, int]:
    """Parameter count per component (no biases anywhere)."""
    d = cfg.d_model
    q_out = cfg.n_heads * cfg.head_dim
    kv_out = cfg.n_kv_heads * cfg.head_dim

    embedding = cfg.vocab_size * d
    attention = (
        d * q_out        # W_Q
        + q_out * d      # W_O
        + d * kv_out     # W_K
        + d * kv_out     # W_V
    )
    mlp = 3 * d * cfg.ff_dim          # SwiGLU: gate, up, down
    norms = 2 * d                     # attention norm + MLP norm
    per_layer = attention + mlp + norms
    all_layers = cfg.n_layers * per_layer
    final_norm = d
    lm_head = 0 if cfg.tie_embeddings else cfg.vocab_size * d

    total = embedding + all_layers + final_norm + lm_head
    return {
        "embedding": embedding,
        "attention_per_layer": attention,
        "mlp_per_layer": mlp,
        "norms_per_layer": norms,
        "per_layer": per_layer,
        "all_layers": all_layers,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "total": total,
    }


def total_params(cfg: ModelConfig) -> int:
    return param_breakdown(cfg)["total"]


def weight_megabytes(n_params: int, bits_per_weight: float) -> float:
    """Weight storage in MB (1 MB = 1,000,000 bytes)."""
    return n_params * bits_per_weight / 8 / 1e6


def kv_cache_bytes_per_token(
    cfg: ModelConfig,
    bytes_per_value: int = 2,
    n_kv_heads: int | None = None,
) -> int:
    """KV cache per token: 2 (K and V) * layers * kv_heads * head_dim * bytes."""
    kv_heads = cfg.n_kv_heads if n_kv_heads is None else n_kv_heads
    return 2 * cfg.n_layers * kv_heads * cfg.head_dim * bytes_per_value