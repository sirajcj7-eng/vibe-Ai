# 006: Inference and Quantization

Status: **Proposed**. Two decisions need approval (see the end).
All speed numbers are estimates from first principles and are
replaced with measurements in Phase 8.

## Goal
Run the model on an ordinary i5-class laptop CPU with low memory
and low latency, with no GPU required.

## Where the time goes

**Decode (one token at a time) is memory-bandwidth bound.** Every
generated token must read all weights from RAM once. Compute is
tiny: about 2N = 100 MFLOPs per token.

    bytes read per token ~ weight bytes + KV cache bytes
    tokens/sec <= effective bandwidth / bytes read per token

Tied embeddings help here: the input lookup reads one row, and the
output head reads the full V*d matrix once, so total reads are about
the full 50.35M parameters.

**Prefill (reading the prompt) is compute bound.** At T = 2048:
- Matmul: 2 * 41.96M * 2048 ~ 172 GFLOP (logits only for the last
  token)
- Attention (causal): 2 * T^2 * d * L = 2 * 2048^2 * 512 * 16
  ~ 69 GFLOP
- Total ~ 241 GFLOP

## Decode speed bound (estimate)

Assumption: laptop with 20 GB/s effective bandwidth (typical
dual-channel DDR4 after real-world losses; the real figure varies
by machine, 10-40 GB/s).

| Precision | Weights | + KV at 2K (16 MB) | Upper bound tok/s |
|---|---|---|---|
| FP16 | 100.7 MB | 117 MB | ~170 |
| INT8 | 50.3 MB | 66 MB | ~300 |
| INT4 (g32) | 28.3 MB | 44 MB | ~450 |

At short context the KV term is near zero, so INT8 approaches
~400 tok/s. Real results are typically 30-60% of the bound, because
the matrices are small and each token runs many small operations
across 16 layers. Expected INT8 decode: roughly 100-250 tok/s.

## Prefill speed (estimate)
At 50-150 GFLOP/s effective on a multi-core AVX2 CPU, a full
2,048-token prompt takes about 1.6-4.8 s (roughly 400-1,200
tok/s). A 128-token prompt should be well under 1 s.

## Quantization

### Formulas
Symmetric quantization with a scale per group:

    s = max|w| / (2^(b-1) - 1)
    q = round(w / s),   w_hat = q * s

Rounding noise per weight has variance about s^2 / 12.

Bits per weight with group size g and an FP16 scale:

    bits = b + 16/g

- INT8 per-channel (one scale per output row): ~8 bits
- INT4, g = 32: 4 + 0.5 = 4.5 bits
- Check: 50.35M * 4.5 / 8 = 28.3 MB, matching 001.

### Method options

| Method | Idea | Verdict |
|---|---|---|
| Post-training, round-to-nearest (RTN) | Quantize trained weights directly | Start here |
| Post-training with calibration (GPTQ/AWQ-style) | Use sample data to reduce error | Use if INT4 RTN loses too much |
| Quantization-aware training (QAT) | Train with simulated quantization | Later; adds cost and complexity |

Small models are more sensitive to quantization than large ones.
INT8 should be nearly lossless; INT4 will lose more and must be
measured.

### What stays at higher precision
- Norm weights: FP16/FP32
- Embedding / output head: INT8, even when the rest is INT4
- Activations: FP16/FP32 (weight-only quantization)
- KV cache: FP16 by default; INT8 KV is an optional later
  experiment (halves KV memory)

### Acceptance criteria (provisional)
- INT8: BPB increase under 1% and benchmark changes within noise
- INT4: loss reported honestly; ship only if the quality-per-MB
  tradeoff is justified

## Runtime options

| Runtime | Pros | Cons |
|---|---|---|
| PyTorch CPU (eager) | Same code as training; easy | Slower, heavier install |
| ONNX Runtime | Portable, good CPU kernels | Export and kernel limits for custom attention |
| llama.cpp (GGUF) | Fast CPU kernels, block quantization, runs everywhere, easy for developers to download | Needs a compatible architecture and tokenizer |

### Key design point: keep the architecture Llama-compatible
The 001 design (RoPE, RMSNorm, SwiGLU, GQA, no biases, tied
embeddings) matches the Llama-family structure that llama.cpp
supports. Dimensions also divide cleanly into block-quantization
sizes: d = 512 and ff = 1280 are multiples of 256.

Recommendation:
- PyTorch is the **reference implementation** for training and
  correctness
- Export to **GGUF** and run with llama.cpp for fast CPU
  distribution
- Verify exports by comparing logits against the PyTorch FP32
  model on fixed inputs (tolerance documented)

Risks to verify against the current llama.cpp version:
1. Tokenizer compatibility: single-digit splitting (003) needs a
   supported pre-tokenizer. If it is unsupported, we ship a small
   standalone tokenizer or adjust the pre-tokenization rule.
2. RoPE layout and pairing must match exactly, or outputs will be
   wrong without any error.
3. Custom special tokens must survive conversion.

## Runtime memory model

    RSS ~ weights + KV(T) + scratch + runtime overhead

- KV(T) = 8,192 bytes * T (FP16): 16 MB at 2K
- Scratch per prefill batch of b tokens ~ b * (d + 2*ff + n_q*d_h)
  * 4 B = b * 3,584 * 4 B (b = 512 gives ~7 MB)
- Chunked attention scores: b * T * n_q * 4 B (b = 512, T = 2048
  gives ~34 MB, reused across layers)
- Weights loaded with mmap are counted by the OS as they are used
- Runtime overhead: tens of MB for llama.cpp-style runtimes;
  hundreds of MB for a full PyTorch process

Provisional targets (INT8, 2K context, i5-class):
- Peak RSS under 250 MB
- Decode at least 100 tok/s at short context
- Time to first token under 1 s for a 128-token prompt

## Scaling the file-size goal to 500M
Using 4.5 bits per weight (Q4-style) and 8.5 bits (Q8-style, with
scale):

| Params | ~4.5-bit file | ~8.5-bit file | Decode bound at 20 GB/s (4.5-bit) |
|---|---|---|---|
| 50M | 28 MB | 53 MB | ~450 tok/s |
| 100M | 56 MB | 106 MB | ~350 tok/s |
| 250M | 141 MB | 266 MB | ~140 tok/s |
| 500M | 281 MB | 531 MB | ~70 tok/s |

This confirms the 250-500 MB target is reachable for a 500M model
at 4-bit, and that 8-bit at 500M would slightly exceed it. These
bounds ignore KV cache and overheads; real numbers will be lower.

## Other speed techniques (later, not 50M)
- Speculative decoding (small draft model + larger model)
- INT8 KV cache
- Prompt caching
- Sliding-window or other long-context attention (separate study)

## Open questions
- Reference CPU for official benchmarks
- Real effective bandwidth on the target machines
- Whether INT4 RTN is acceptable at 50M
## Decision
Accepted: Llama-compatible architecture, PyTorch reference plus
GGUF/llama.cpp for distribution, INT8 default, INT4 experimental.