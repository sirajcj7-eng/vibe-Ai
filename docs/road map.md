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

## Model family direction (EVY)
- EVY 50M is a general conversational prototype. Its job is to
  prove we can build, train, evaluate, save, load and run a
  language model from random initialization.
- No routing, mixture-of-experts, agents or tool ecosystems at 50M.
- Specialization (Code, Math, Physics, Research, General) begins
  around 100M. Whether variants use separate weights, a shared
  base, or adapters is a research question for Phase 8-9.
- Domain routing and retrieval (EVY Research) come after the base
  models are validated.
- No automatic scaling: a model is scaled only after the previous
  one is evaluated and its problems are understood.
- Every run records: parameter count, config, tokenizer version,
  dataset version, training tokens, context length, training
  config, compute, duration, validation loss, eval results,
  inference speed, memory, qualitative examples, failure modes.