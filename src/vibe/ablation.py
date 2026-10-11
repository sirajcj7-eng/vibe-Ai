"""Helpers for hyperparameter ablations (Phase 6.10)."""
from __future__ import annotations

import dataclasses
import statistics
import time
from dataclasses import replace
from pathlib import Path

from vibe.config import ModelConfig, total_params
from vibe.data import TokenStream
from vibe.train import TrainConfig, train

MODEL_FIELDS = {f.name for f in dataclasses.fields(ModelConfig)}
TRAIN_FIELDS = {f.name for f in dataclasses.fields(TrainConfig)}


def _parse_one(item: str):
    low = item.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(item)
        except ValueError:
            pass
    return item


def parse_values(text: str) -> list:
    """'1,2,4' -> [1, 2, 4]; '5e-4,1e-3' -> [0.0005, 0.001]; 'fp16' -> ['fp16']."""
    return [_parse_one(x.strip()) for x in text.split(",") if x.strip()]


def apply_overrides(model_cfg: ModelConfig, train_cfg: TrainConfig, overrides: dict):
    """Split overrides between model and training config (both are re-validated)."""
    m, t = {}, {}
    for key, value in overrides.items():
        if key in MODEL_FIELDS:
            m[key] = value
        elif key in TRAIN_FIELDS:
            t[key] = value
        else:
            raise ValueError(f"unknown field: {key}")
    return replace(model_cfg, **m), replace(train_cfg, **t)


def match_ff_dim(cfg: ModelConfig, target_params: int, multiple: int = 32) -> ModelConfig:
    """Choose ff_dim (a multiple of `multiple`) giving the closest parameter count."""
    best = None
    for ff in range(multiple, cfg.ff_dim * 4 + 1, multiple):
        cand = replace(cfg, ff_dim=ff)
        diff = abs(total_params(cand) - target_params)
        if best is None or diff < best[0]:
            best = (diff, cand)
    return best[1]


def run_one(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    overrides: dict,
    seed: int,
    train_stream: TokenStream,
    val_stream: TokenStream,
    out_dir: str | Path,
    device: str = "cpu",
    tokens_per_byte: float | None = None,
    target_params: int | None = None,
) -> dict:
    """Train one variant and return its result record."""
    m, t = apply_overrides(model_cfg, train_cfg, overrides)
    if target_params is not None:
        m = match_ff_dim(m, target_params)
    t = replace(t, seed=seed)
    rec = {"overrides": overrides, "seed": seed, "n_params": total_params(m), "ff_dim": m.ff_dim}
    t0 = time.time()
    try:
        out = train(m, t, train_stream, val_stream, out_dir, device=device,
                    val_tokens_per_byte=tokens_per_byte, verbose=False)
    except RuntimeError as e:
        if "non-finite" not in str(e):
            raise
        rec.update(diverged=True, val_loss=None, val_bpb=None, error=str(e),
                   seconds=time.time() - t0)
        return rec
    final = out["final"]
    tail = out["losses"][-50:]
    rec.update(diverged=False, val_loss=final["val_loss"], val_bpb=final.get("val_bpb"),
               train_loss=sum(tail) / len(tail), seconds=time.time() - t0)
    return rec


def summarize(records: list[dict], key: str = "val_bpb") -> list[dict]:
    """Group by variant: mean/std/min/max of `key`, plus the diverged count."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r["variant"], []).append(r)
    rows = []
    for variant, rs in groups.items():
        vals = [r[key] for r in rs if r.get(key) is not None]
        rows.append({
            "variant": variant,
            "n": len(vals),
            "diverged": len(rs) - len(vals),
            "mean": statistics.fmean(vals) if vals else None,
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
            "params": rs[0]["n_params"],
        })
    rows.sort(key=lambda r: (r["mean"] is None, r["mean"] or 0.0))
    return rows