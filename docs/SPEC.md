# EVY 50M: Frozen Design Spec (v0.1)

Summary of decisions 001-006. Details and math live in
`docs/decisions/`. Everything is provisional until tested in
Phase 6, and any change gets a new decision doc.

## Purpose
EVY 50M is a general conversational prototype that proves the
full from-scratch pipeline. Success means: measurable learning
and useful behavior from random initialization, reproducibly.

## Hard rules
From scratch (random init), no LLM APIs, no pretrained weights,
500M parameter ceiling, CPU-first inference, only legally usable
data, build in public.

## Model (001)
| Item | Value |
|---|---|
| Architecture | Decoder-only, pre-norm, Llama-compatible |
| Parameters | 50,348,544 |
| Vocabulary | 16,384 |
| d / layers | 512 / 16 |
| Heads | 8 query, 2 KV (GQA), head dim 64 |
| MLP | SwiGLU, ff = 1280 |
| Position / norm | RoPE / RMSNorm, no biases |
| Embeddings | Tied |
| Context | 2,048 |
| KV cache | 8 KB per token (FP16), 16 MB at 2K |

## Weight size
FP32 201 MB, FP16 101 MB, INT8 50 MB, INT4 ~28 MB.

## Training (002)
- 5B-token target, 1B-token evaluated milestone first
- ~403 MFLOPs per token, 2.0e18 FLOPs total for 5B
- Batch 262,144 tokens, AdamW, WSD schedule, peak LR 1e-3
  (to be swept), weight decay 0.1, clip 1.0
- Free Kaggle/Colab GPUs: roughly 80 GPU-hours (estimate)
- Laptop CPU: debugging, data prep, tiny models, inference tests

## Tokenizer (003)
Byte-level BPE, 16,384 vocab, single-digit numbers, ~32 reserved
special tokens (chat, tool, fill-in-the-middle), frozen before
pretraining.

## Data (004)
~5B tokens: educational web 50%, code 15%, math 8%, reference
10%, public-domain books 7%, open-access science 5%, reserve 5%.
Human-written, openly licensed data only (no synthetic data at
50M). Full filtering, dedup, PII, safety and decontamination
pipeline. Provenance in `data/manifest.yaml`.

## Evaluation (005)
BPB and validation loss as primary metrics, benchmarks with signal
at small scale (HellaSwag, PIQA, ARC-Easy, SciQ, LAMBADA, BLiMP),
custom math, code and instruction tests, passkey context tests,
qualitative prompt set, CPU speed and memory. Baselines set before
training. Three go/no-go gates.

## Inference (006)
PyTorch reference implementation, GGUF export for llama.cpp, INT8
default, INT4 experimental. Targets (INT8, 2K context): under 250 MB
RSS, 100+ tok/s decode, under 1 s time-to-first-token for 128 tokens.

## Phase 6 (proof-of-concept) will answer
- Real tokens/sec on T4/P100
- Peak LR and batch size
- KV heads: 2 vs 4
- Vocab: 8k vs 16k vs 32k (by bits per byte)
- RoPE theta, ff width
- Checkpoint and resume correctness
- llama.cpp tokenizer and RoPE compatibility

## Roadmap
Design (done) -> Phase 6 tiny PoC + ablations -> 50M run
(1B then 5B tokens) -> post-training, quantization, CPU tuning
-> 100M (specialization begins) -> 250M -> 500M.

## Sign-off
- [ ] I have read and agree with the spec
- [ ] I approve starting Phase 6 (first code)