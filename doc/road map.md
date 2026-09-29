# Project Principles

## What this project is
An open, build-in-public effort to design and train a small language
model completely from scratch, optimized for local CPU inference.

## Hard rules
1. **From scratch.** Weights start randomly initialized and are trained
   by us. No pretrained weights. No fine-tuning an existing LLM as the
   foundation.
2. **No external LLM APIs** (OpenAI, Anthropic, Gemini, etc.) in the
   model or its training pipeline.
3. **Frameworks are allowed.** PyTorch, NumPy, tokenizer and dataset
   libraries are tools. The architecture, config, training process and
   weights are ours.
4. **Parameter ceiling: 500M.** The first model is ~50M. Scaling is
   50M -> 100M -> 250M -> 500M, only after each step is understood.
5. **CPU-first.** Inference must be practical on ordinary modern CPUs
   (e.g. Intel Core i5 class). GPU is optional, never required.

## Design before code
No implementation until the design is written down and the math is
verified: architecture, parameter count, memory, training plan, data
plan, evaluation. Every major decision gets a file in
`docs/decisions/` with the reasoning and alternatives considered.

## Honesty rules
- We do not claim a small model contains all human knowledge.
- Weight size is not runtime RAM. Weights, KV cache, activations,
  tokenizer and buffers are calculated separately.
- Numbers are derived or measured, not assumed.
- Failed experiments are documented, not hidden.

## Data rules
Only legally usable data. Track licensing and provenance. Handle
deduplication, PII, benchmark contamination and low-quality text
explicitly.

## Transparency
Decisions, derivations, training runs, hardware used, benchmarks and
failures are all published in this repo.