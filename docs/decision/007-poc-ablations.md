# 007: Proof-of-Concept Ablations (tiny model)

Status: **Accepted**. Phase 6.10 results. Raw data:
results/ablation-lr.json, results/ablation-kv.json

## Setup
Tiny model (1,262,720 params: d=128, 4 layers, 4 query heads), Gutenberg
PoC corpus (5.27M train / 0.93M val tokens), 1,500 steps, batch 16 x 256
tokens (about 6.1M tokens, 1.2 epochs), warmup 50, WSD decay over the last
20% to 10% of peak, AdamW weight decay 0.1, clip 1.0. fp16 on a Kaggle
T4. Three seeds per setting. Metric: validation bits per byte over the
whole validation set (lower is better).

## Learning rate
| LR | Mean bpb | Std | Min | Max |
|---|---|---|---|---|
| 4e-3 | 1.7106 | 0.0029 | 1.7073 | 1.7125 |
| 2e-3 | 1.7174 | 0.0011 | 1.7167 | 1.7187 |
| 8e-3 | 1.7181 | 0.0041 | 1.7151 | 1.7228 |
| 1e-3 | 1.7302 | 0.0030 | 1.7275 | 1.7335 |
| 5e-4 | 1.8101 | 0.0045 | 1.8068 | 1.8152 |

No run diverged, even at 8e-3.

## KV heads (ff_dim adjusted so parameters match within 1.3%)
| Variant | Params | Mean bpb | Std |
|---|---|---|---|
| 1 KV head (4:1 GQA) | 1,279,104 | 1.7138 | 0.0048 |
| 4 KV heads (MHA) | 1,279,104 | 1.7169 | 0.0020 |
| 2 KV heads (2:1) | 1,262,720 | 1.7174 | 0.0011 |

## Findings
- fp16 on the T4 reproduces the FP32 CPU result (lr 1e-3: 1.7302 vs
  1.732), so the GPU training path is verified.
- Learning rate has a broad plateau from 2e-3 to 8e-3 on this tiny model.
  1e-3 is about 1.1% worse, 5e-4 about 5.8% worse. The training recipe
  is stable.
- Sharing KV heads has no measurable cost: differences (at most 0.2%)
  are within seed noise. GQA is confirmed.
- Noise floor: seed-to-seed std is about 0.001-0.005 bpb. Smaller
  differences are not evidence.

## Decisions
- Keep 8 query / 2 KV heads (4:1 GQA) for the 50M model.
- Do NOT carry the tiny-model learning rate to 50M. Short runs favor
  high LRs, and the optimum falls with width and training length (tiny
  is d=128, 50M is d=512). Before the long run, test lr in
  {1e-3, 2e-3, 4e-3} on the real 50M model, about 20M tokens each
  (about 1 GPU-hour in total), using the real data.
- Defer vocabulary size (8k/16k/32k) until the real tokenizer and data
  mix exist (Phases 3-4). RoPE theta and MLP width wait for 50M scale.

## Limits
1.26M parameters, 5M tokens, 256-token context, old novels. Treat as a
sanity check, not proof for 50M at 2,048 context.