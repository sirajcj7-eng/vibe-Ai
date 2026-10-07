# 001: Model Shape (50M)

Status: **Accepted (provisional)**. To be tested in Phase 6 ablations.

## Decision summary

| Item | Choice |
|---|---|
| Architecture | Decoder-only, pre-norm |
| Vocabulary | 16,384 |
| Width d | 512 |
| Layers L | 16 |
| Query heads | 8 (head dim 64) |
| KV heads | 2 (GQA, 4 query heads per KV head) |
| MLP | SwiGLU, ff = 1280 |
| Position | RoPE (base theta = 10,000, provisional) |
| Norm | RMSNorm, no biases anywhere |
| Embeddings | Tied input/output |
| Context | 2,048 |
| Init | N(0, 0.02); residual output projections scaled by 1/sqrt(2L) |

## Parameter count

Notation: V = 16,384, d = 512, L = 16, ff = 1280, n_q = 8,
n_kv = 2, d_h = 64.

- Embedding (tied, counted once): V*d = 8,388,608
- Attention per layer:
  - Q: d * (n_q*d_h) = 262,144
  - O: (n_q*d_h) * d = 262,144
  - K: d * (n_kv*d_h) = 65,536
  - V: d * (n_kv*d_h) = 65,536
  - Total: 655,360
- SwiGLU MLP per layer (gate, up, down): 3*d*ff = 1,966,080
- RMSNorm per layer (2 norms): 2*d = 1,024
- Per-layer total: 2,622,464
- All layers: 16 * 2,622,464 = 41,959,424
- Final RMSNorm: 512

**Total = 8,388,608 + 41,959,424 + 512 = 50,348,544 (50.35M)**

Embeddings are 16.7% of parameters. Non-embedding: 41.96M.

## Decisions and reasoning

### Decoder-only
Next-token prediction on one stack is the simplest design, trains
on any text, and supports generation, instruction tuning and tool
calls. Encoder-decoder would split a tiny parameter budget for no
clear gain here.

### Vocabulary 16,384
Embedding cost is V*d. At V = 32,768 the embedding would be 16.8M
(about 33% of the model), leaving too little for the transformer
layers. At V = 8,192 the model is cheaper but text splits into more
tokens, which raises training and inference cost per character.
16k is the compromise. Risk: weaker multilingual and code coverage.
Tokenizer compression will be measured in Phase 3 and this can be
revised.

### Tied embeddings
Untying adds V*d = 8.39M parameters (about 3 layers' worth).
At this scale it is better to spend them on depth. Tied is standard.

### d = 512, L = 16
Shapes compared at about 50M:

| Shape | d | L | Params (approx) |
|---|---|---|---|
| Wide/shallow | 640 | 9 | ~49M |
| **Chosen** | 512 | 16 | 50.35M |
| Deep/thin | 384 | 28 | ~50M |

Research on sub-billion models (e.g. MobileLLM) finds deeper and
thinner beats wide and shallow at fixed size. Very thin models
are slower per token on CPU because layers run sequentially and
matrices become small. 512/16 is the middle option.

### Heads: 8 query, head dim 64
d = n_q * d_h = 8 * 64 = 512. Head dim 64 is standard and works
well with CPU vector units and RoPE.

### Grouped-query attention (2 KV heads)
Attention per head h:

    Q = X W_Q, K = X W_K, V = X W_V
    A_h = softmax( Q_h K_g(h)^T / sqrt(d_h) + M ) V_g(h)

M is the causal mask (0 on allowed positions, -inf above the
diagonal). g(h) maps each query head to its KV group. Scaling by
1/sqrt(d_h) keeps score variance near 1.

Saving vs full MHA (8 KV heads): 2 * d * (6*d_h) = 393,216 params
per layer, 6.29M total, which we spend on the MLP instead.

KV cache per token = 2 (K,V) * L * n_kv * d_h
= 2 * 16 * 2 * 64 = 4,096 values:
- FP16: 8 KB/token (MHA would be 32 KB)
- At 2,048 tokens: 16 MB (MHA: 64 MB)

The saving is modest now and grows with model size and context.
Ablation: 2 vs 4 vs 8 KV heads in Phase 6.

### SwiGLU, ff = 1280
SwiGLU: y = W_down( SiLU(W_gate x) * (W_up x) ). It uses 3 matrices
instead of 2, so ff is set to about (8/3)*d = 1365 for equal cost to
a 4d GELU MLP. We round to 1280 (a multiple of 256) for hardware
efficiency. SwiGLU consistently beats GELU/ReLU at equal parameters.

### RoPE
Rotates Q and K by position-dependent angles, so attention scores
depend on relative position. No learned position parameters, and it
allows later context extension. theta = 10,000 is provisional.

### RMSNorm, pre-norm, no biases
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * g. Cheaper than LayerNorm
(no mean, no bias) and standard in modern models. Pre-norm keeps
training stable without careful warmup tuning. No biases saves
parameters and simplifies quantization.

### Initialization
Weights ~ N(0, 0.02). Residual output projections (attention O and
MLP down) are scaled by 1/sqrt(2L) = 1/sqrt(32) ~ 0.177 so residual
stream variance does not grow with depth.

### Context 2,048
Attention compute is O(n^2 * d) per layer and the KV cache grows
linearly in n. 2k is enough for useful tasks and cheap on CPU.
Long-context methods are a separate study (see roadmap).

## Weight memory

N = 50,348,544

| Precision | Formula | Size |
|---|---|---|
| FP32 | 4 B/param | 201.4 MB |
| FP16/BF16 | 2 B/param | 100.7 MB |
| INT8 | 1 B/param | 50.3 MB |
| INT4 (group 32, FP16 scale) | ~4.5 bits/param | ~28.3 MB |

## Runtime memory (estimate, to be measured)

Weights are not total RAM. Components:

| Component | Estimate (INT8, 2k ctx) |
|---|---|
| Weights | ~50 MB |
| KV cache (FP16, full 2,048) | 16 MB |
| Decode activations | under 1 MB (hidden 512, MLP 1280, logits 64 KB in FP32) |
| Prefill activations | tens of MB; naive attention scores would be 8*2048*2048*4 B = 134 MB per layer if materialized, so use chunked/fused attention |
| Tokenizer | ~1-2 MB |
| Runtime overhead and buffers | tens to hundreds of MB, depends on runtime |

KV cache scales linearly:

| Context | KV FP16 (GQA 2) | KV FP16 (MHA, for comparison) |
|---|---|---|
| 512 | 4 MB | 16 MB |
| 2,048 | 16 MB | 64 MB |
| 8,192 | 64 MB | 256 MB |

Every row above must be replaced with measurements in Phase 8.

## Compute (preview)

Training FLOPs ~ 6 * N * D.
- D = 1B tokens: ~3.0e17 FLOPs
- D = 10B tokens: ~3.0e18 FLOPs

Rough feasibility on free GPUs (Colab/Kaggle T4-class, ~30% real
utilization of fp16 peak, about 7 TFLOPs effective): 1B tokens is
on the order of 12 GPU-hours; 10B tokens is on the order of 120.
These are estimates. Session limits and weekly quotas mean the
token target must be set in the training-system doc (Phase 5)
with measured throughput from the proof-of-concept.

## Open questions
- Token budget for the 50M run (D)
- Vocab size vs multilingual/code needs (Phase 3)
- KV heads: 2 vs 4 (Phase 6)
- RoPE theta (Phase 6)
- Whether ff = 1280 vs 1376 changes results (Phase 6)

## Alternatives rejected for now
- Untied embeddings (cost: 8.39M params)
- 32k vocab (embedding share too large)
- Encoder-decoder
- Learned absolute positions (no extension path)
- Parameter sharing across layers (possible later study, hurts
  simplicity now)