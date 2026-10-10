"""Generate text from a trained checkpoint.

Single prompt:
  python scripts/generate.py --checkpoint runs/tiny-1500/model_final.pt --prompt "Once upon a time"
Fixed prompt set (saved to results/):
  python scripts/generate.py --checkpoint runs/tiny-1500/model_final.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from vibe.generate import generate, load_model
from vibe.tokenizer import EOS_TOKEN, decode_ids, encode_text, load_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "data" / "tokenizer" / "poc-4096.json")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--prompts-file", type=Path, default=ROOT / "data" / "eval" / "prompts.txt")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    model = load_model(args.checkpoint, args.device)
    eos_id = tok.token_to_id(EOS_TOKEN)

    if args.prompt is not None:
        prompts, save = [args.prompt], False
    else:
        lines = args.prompts_file.read_text(encoding="utf-8").splitlines()
        prompts = [l for l in lines if l.strip() and not l.startswith("#")]
        save = True

    run_name = args.checkpoint.resolve().parent.name
    settings = (f"temperature={args.temperature}, top_k={args.top_k}, "
                f"top_p={args.top_p}, seed={args.seed}+index, max_new_tokens={args.max_new_tokens}")
    md = [f"# Samples: {run_name}", "", f"Checkpoint: `{args.checkpoint.name}`", f"Settings: {settings}", ""]
    new_total, time_total = 0, 0.0

    for i, prompt in enumerate(prompts):
        ids = encode_text(tok, prompt)
        t0 = time.time()
        new = generate(model, ids, args.max_new_tokens, args.temperature,
                       args.top_k, args.top_p, eos_id, seed=args.seed + i)
        time_total += time.time() - t0
        new_total += len(new)
        text = decode_ids(tok, ids + new)
        print(f"\n[{i + 1}/{len(prompts)}] {text}")
        md += [f"## {i + 1}. {prompt}", "", "```", text, "```", ""]

    print(f"\nGenerated {new_total} tokens in {time_total:.1f}s "
          f"({new_total / max(time_total, 1e-9):.1f} tokens/s on {args.device})")
    if save:
        out = args.out or ROOT / "results" / f"{run_name}-samples.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(md), encoding="utf-8")
        print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())