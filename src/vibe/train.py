"""Training loop (docs/decisions/002-compute-and-token-budget.md).

AdamW, warmup-stable-decay (WSD) schedule, gradient accumulation and
clipping, checkpoints that resume exactly, validation loss and bits/byte.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from vibe.config import ModelConfig
from vibe.data import PackedLoader, TokenStream, iter_sequential
from vibe.model import Transformer


@dataclass
class TrainConfig:
    steps: int = 200              # total optimizer steps (defines the LR schedule)
    batch_size: int = 16          # sequences per micro-batch
    grad_accum: int = 1           # micro-batches per optimizer step
    seq_len: int = 256
    lr: float = 1e-3              # peak learning rate
    min_lr_frac: float = 0.1      # final LR = lr * min_lr_frac
    warmup_steps: int = 20
    decay_frac: float = 0.2       # last fraction of steps decays linearly
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    clip_norm: float = 1.0
    eval_every: int = 50
    eval_batches: int = 20        # 0 = whole validation set
    log_every: int = 10
    ckpt_every: int = 50
    seed: int = 0
    precision: str = "fp32"       # fp32 | bf16 | fp16 (fp16 needs CUDA)
    resume: bool = False
    stop_at: int | None = None    # stop and save here without changing the schedule

    def validate(self, model_cfg: ModelConfig) -> None:
        for name in (
            "steps", "batch_size", "grad_accum", "seq_len",
            "eval_every", "log_every", "ckpt_every",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if not 0 < self.decay_frac <= 1:
            raise ValueError("decay_frac must be in (0, 1]")
        if self.seq_len > model_cfg.max_seq_len:
            raise ValueError(
                f"seq_len {self.seq_len} exceeds model max_seq_len {model_cfg.max_seq_len}"
            )
        if self.precision not in ("fp32", "bf16", "fp16"):
            raise ValueError("precision must be fp32, bf16 or fp16")
        if self.stop_at is not None and self.stop_at <= 0:
            raise ValueError("stop_at must be positive")


def lr_at(step: int, t: TrainConfig) -> float:
    """Warmup -> constant -> linear decay to lr * min_lr_frac (step is 0-based)."""
    min_lr = t.lr * t.min_lr_frac
    decay_steps = max(1, round(t.steps * t.decay_frac))
    decay_start = t.steps - decay_steps
    if step < t.warmup_steps:
        return t.lr * (step + 1) / t.warmup_steps
    if step < decay_start:
        return t.lr
    progress = min(1.0, (step - decay_start + 1) / decay_steps)
    return t.lr + (min_lr - t.lr) * progress


def build_optimizer(model: torch.nn.Module, t: TrainConfig) -> torch.optim.AdamW:
    """Weight decay on matrices only; none on norms and embeddings (doc 002)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "tok_emb" not in name:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": t.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=t.lr, betas=(t.beta1, t.beta2), eps=t.eps)


def autocast_ctx(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def save_checkpoint(
    path: Path, model, optimizer, scaler, loader: PackedLoader,
    step: int, tokens_seen: int, model_cfg: ModelConfig, t: TrainConfig,
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "loader": loader.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "step": step,
        "tokens_seen": tokens_seen,
        "model_config": asdict(model_cfg),
        "train_config": asdict(t),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written checkpoint


def load_checkpoint(path: Path, model, optimizer, scaler, loader, device) -> tuple[int, int]:
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])
    loader.load_state_dict(state["loader"])
    torch.set_rng_state(state["rng_cpu"].cpu())
    return int(state["step"]), int(state["tokens_seen"])


@torch.no_grad()
def evaluate(
    model, stream: TokenStream, seq_len: int, batch_size: int,
    max_batches: int | None, device: torch.device, precision: str,
) -> float:
    """Mean cross-entropy (nats per token) over fixed validation batches."""
    model.eval()
    total, n = 0.0, 0
    for x, y in iter_sequential(stream, seq_len, batch_size, max_batches):
        with autocast_ctx(device, precision):
            _, loss, _ = model(x.to(device), targets=y.to(device))
        total += loss.item()
        n += 1
    model.train()
    if n == 0:
        raise ValueError("validation set too small for this batch size and seq_len")
    return total / n


def environment_info() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def train(
    model_cfg: ModelConfig,
    t: TrainConfig,
    train_stream: TokenStream,
    val_stream: TokenStream,
    out_dir: str | Path,
    device: str | torch.device = "cpu",
    val_tokens_per_byte: float | None = None,
    verbose: bool = True,
) -> dict:
    t.validate(model_cfg)
    device = torch.device(device)
    if t.precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 needs a CUDA device; use bf16 or fp32 on CPU")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "last.pt"
    if ckpt_path.exists() and not t.resume:
        raise FileExistsError(
            f"{ckpt_path} exists. Use --resume to continue or choose a new --out folder."
        )

    torch.manual_seed(t.seed)
    model = Transformer(model_cfg).to(device)
    optimizer = build_optimizer(model, t)
    scaler = torch.amp.GradScaler("cuda") if t.precision == "fp16" else None
    loader = PackedLoader(train_stream, t.seq_len, t.batch_size, seed=t.seed)

    step, tokens_seen, resumed = 0, 0, False
    if t.resume and ckpt_path.exists():
        step, tokens_seen = load_checkpoint(ckpt_path, model, optimizer, scaler, loader, device)
        resumed = True

    metrics_path = out_dir / "metrics.jsonl"

    def write(rec: dict) -> None:
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    config_path = out_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(
            json.dumps(
                {
                    "model": asdict(model_cfg),
                    "train": asdict(t),
                    "n_params": model.num_parameters(),
                    "device": str(device),
                    "env": environment_info(),
                    "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if resumed:
        write({"type": "resume", "step": step})

    tokens_per_step = t.batch_size * t.grad_accum * t.seq_len
    stop = t.steps if t.stop_at is None else min(t.stop_at, t.steps)
    eval_cap = t.eval_batches if t.eval_batches > 0 else None
    losses: list[float] = []
    val_history: list[dict] = []

    def do_eval(max_batches: int | None) -> dict:
        val_loss = evaluate(
            model, val_stream, t.seq_len, t.batch_size, max_batches, device, t.precision
        )
        rec = {"type": "val", "step": step, "val_loss": val_loss}
        if val_tokens_per_byte:
            rec["val_bpb"] = val_loss / math.log(2) * val_tokens_per_byte
        write(rec)
        val_history.append(rec)
        if verbose:
            extra = f" | bits/byte {rec['val_bpb']:.3f}" if "val_bpb" in rec else ""
            print(f"  [val @ step {step}] loss {val_loss:.4f}{extra}", flush=True)
        return rec

    if verbose:
        print(f"device {device} | {model.num_parameters():,} params | "
              f"{tokens_per_step:,} tokens/step | steps {step} -> {stop} of {t.steps}")
    if not resumed:
        do_eval(eval_cap)  # baseline before any training

    model.train()
    window_start, window_tokens = time.time(), 0
    while step < stop:
        lr = lr_at(step, t)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)

        step_loss = 0.0
        for _ in range(t.grad_accum):
            x, y = loader.next_batch()
            x, y = x.to(device), y.to(device)
            with autocast_ctx(device, t.precision):
                _, loss, _ = model(x, targets=y)
            scaled = loss / t.grad_accum
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            step_loss += loss.item() / t.grad_accum

        if scaler is not None:
            scaler.unscale_(optimizer)
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), t.clip_norm))
        if not math.isfinite(step_loss):
            raise RuntimeError(f"loss became non-finite at step {step + 1}")
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        step += 1
        tokens_seen += tokens_per_step
        window_tokens += tokens_per_step
        losses.append(step_loss)

        if step == 1 or step % t.log_every == 0:
            tps = window_tokens / max(time.time() - window_start, 1e-9)
            write({
                "type": "train", "step": step, "loss": step_loss, "lr": lr,
                "grad_norm": gnorm, "tokens_seen": tokens_seen, "tok_per_sec": tps,
            })
            if verbose:
                print(f"step {step}/{t.steps} | loss {step_loss:.4f} | lr {lr:.2e} | "
                      f"gnorm {gnorm:.2f} | {tps:,.0f} tok/s", flush=True)
            window_start, window_tokens = time.time(), 0

        if step % t.eval_every == 0 and step < t.steps:
            do_eval(eval_cap)
        if step % t.ckpt_every == 0 or step == stop:
            save_checkpoint(ckpt_path, model, optimizer, scaler, loader,
                            step, tokens_seen, model_cfg, t)

    final = None
    if step == t.steps:
        final = do_eval(None)  # whole validation set
        torch.save(
            {"model": model.state_dict(), "model_config": asdict(model_cfg)},
            out_dir / "model_final.pt",
        )
        if verbose:
            print(f"Done. Final model saved to {out_dir / 'model_final.pt'}")
    elif verbose:
        print(f"Stopped at step {step}. Continue with --resume.")

    return {"step": step, "losses": losses, "val": val_history, "final": final}