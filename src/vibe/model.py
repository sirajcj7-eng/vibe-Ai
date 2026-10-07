"""Decoder-only transformer (docs/decisions/001-model-shape.md).

Pre-norm blocks, RMSNorm, RoPE, grouped-query attention, SwiGLU MLP,
no biases, tied input/output embeddings.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vibe.config import ModelConfig

# One layer's cache: (K, V), each (batch, kv_heads, seq, head_dim).
KVCache = tuple[torch.Tensor, torch.Tensor]


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * g"""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


def build_rope_cache(
    head_dim: int, max_seq_len: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (max_seq_len, head_dim / 2)."""
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    inv_freq = 1.0 / (theta ** exponents)
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (i, i + head_dim/2) by position-dependent angles.

    x: (batch, heads, seq, head_dim); cos/sin: (seq, head_dim / 2).
    Uses the half-split layout of Llama-family checkpoints, which keeps
    a later GGUF export simple.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class Attention(nn.Module):
    """Causal grouped-query attention."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.group_size = cfg.n_heads // cfg.n_kv_heads
        d = cfg.d_model
        self.q_proj = nn.Linear(d, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(d, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(d, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, d, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        present = (k, v) if use_cache else None  # cache stores KV heads only

        past_len = k.shape[2] - T
        # Query head h uses KV head h // group_size.
        k_full = k.repeat_interleave(self.group_size, dim=1)
        v_full = v.repeat_interleave(self.group_size, dim=1)

        # softmax(Q K^T / sqrt(head_dim) + causal mask) V
        if past_len == 0:
            out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=True)
        else:
            mask = torch.ones(T, k.shape[2], dtype=torch.bool, device=x.device)
            mask = mask.tril(diagonal=past_len)
            out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask)

        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out), present


class SwiGLU(nn.Module):
    """y = W_down( SiLU(W_gate x) * (W_up x) )"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ff_dim, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ff_dim, bias=False)
        self.down_proj = nn.Linear(cfg.ff_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        h, present = self.attn(self.attn_norm(x), cos, sin, past, use_cache)
        x = x + h
        x = x + self.mlp(self.mlp_norm(x))
        return x, present


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = (
            None
            if cfg.tie_embeddings
            else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        )

        cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale residual output projections by 1/sqrt(2 * layers).
        scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        with torch.no_grad():
            for block in self.blocks:
                block.attn.o_proj.weight.mul_(scale)
                block.mlp.down_proj.weight.mul_(scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_caches: list[KVCache] | None = None,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[KVCache] | None]:
        B, T = input_ids.shape
        if start_pos + T > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence end {start_pos + T} exceeds max_seq_len {self.cfg.max_seq_len}"
            )
        if kv_caches is not None and start_pos != kv_caches[0][0].shape[2]:
            raise ValueError("start_pos must equal the number of cached tokens")

        x = self.tok_emb(input_ids)
        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]

        presents: list[KVCache] = []
        for i, block in enumerate(self.blocks):
            past = None if kv_caches is None else kv_caches[i]
            x, present = block(x, cos, sin, past, use_cache)
            if present is not None:
                presents.append(present)

        x = self.final_norm(x)
        if self.lm_head is None:
            logits = F.linear(x, self.tok_emb.weight)  # tied embeddings
        else:
            logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size).float(),
                targets.reshape(-1),
            )
        return logits, loss, (presents if use_cache else None)