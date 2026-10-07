"""Print parameter breakdown, weight memory and KV-cache size per config."""
import sys
from pathlib import Path

from vibe.config import (
    kv_cache_bytes_per_token,
    load_config,
    param_breakdown,
    weight_megabytes,
)

# Verified by hand in docs/decisions/001-model-shape.md and the PoC plan.
EXPECTED_TOTALS = {"tiny": 1_262_720, "50m": 50_348_544}

PRECISIONS = {
    "FP32": 32,
    "FP16/BF16": 16,
    "INT8": 8,
    "INT4 (group 32, FP16 scale)": 4.5,
}


def report(path: Path) -> bool:
    cfg = load_config(path)
    b = param_breakdown(cfg)

    print(f"\n=== {cfg.name} ({path.name}) ===")
    print(f"embedding            : {b['embedding']:>14,}")
    print(f"attention / layer    : {b['attention_per_layer']:>14,}")
    print(f"MLP / layer          : {b['mlp_per_layer']:>14,}")
    print(f"norms / layer        : {b['norms_per_layer']:>14,}")
    print(f"per layer total      : {b['per_layer']:>14,}")
    print(f"all {cfg.n_layers} layers        : {b['all_layers']:>14,}")
    print(f"final norm           : {b['final_norm']:>14,}")
    if b["lm_head"]:
        print(f"separate output head : {b['lm_head']:>14,}")
    print(f"TOTAL                : {b['total']:>14,}")

    print("\nWeight memory (MB = 1,000,000 bytes):")
    for label, bits in PRECISIONS.items():
        print(f"  {label:<30} {weight_megabytes(b['total'], bits):>9.1f} MB")

    per_tok = kv_cache_bytes_per_token(cfg)
    per_tok_mha = kv_cache_bytes_per_token(cfg, n_kv_heads=cfg.n_heads)
    mib = 1024 * 1024
    print(f"\nKV cache, FP16 (MiB = 1,048,576 bytes), {cfg.n_kv_heads} KV heads:")
    print(f"  per token : {per_tok:,} bytes")
    print(f"  at {cfg.max_seq_len} tokens: {per_tok * cfg.max_seq_len / mib:.2f} MiB")
    print(f"  full MHA would be {per_tok_mha:,} bytes/token, "
          f"{per_tok_mha * cfg.max_seq_len / mib:.2f} MiB at {cfg.max_seq_len}")

    expected = EXPECTED_TOTALS.get(cfg.name)
    if expected is None:
        return True
    ok = b["total"] == expected
    print(f"\nExpected {expected:,} -> {'OK' if ok else 'MISMATCH'}")
    return ok


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = [Path(p) for p in sys.argv[1:]] or [
        root / "configs" / "tiny.yaml",
        root / "configs" / "50m.yaml",
    ]
    results = [report(p) for p in paths]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())