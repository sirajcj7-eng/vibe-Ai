# Roadmap

Each phase has an exit criterion. We do not move on until it is met
and documented.

## Phase 0: Research and mathematical design
Derive parameter counts, memory, KV cache, and compute budgets.
Exit: numbers written in `docs/` and reviewed.

## Phase 1: Architecture specification
Decision docs for layers, width, heads, MLP, vocab, position
encoding, normalization, activation, attention type, weight tying,
initialization.
Exit: a frozen 50M spec with a verified parameter count.

## Phase 2: Evaluation design (moved earlier)
Define benchmarks and metrics before training anything: perplexity,
knowledge, math, reasoning, coding, instruction following, context,
speed, memory. Baselines come first.
Exit: eval plan and validation sets chosen.

## Phase 3: Tokenizer design
Vocab size, algorithm, special tokens, tokenizer-training corpus.
Exit: tokenizer spec and compression-rate targets.

## Phase 4: Dataset strategy
Sources, licenses, provenance, dedup, filtering, PII, contamination
checks, mix ratios.
Exit: a documented data manifest with license status.

## Phase 5: Training-system design
Optimizer, LR schedule, batch size, precision, checkpointing,
token budget, compute plan.
Exit: a budget we can actually afford.

## Phase 6: Tiny proof-of-concept (new: ablations added)
Train a very small model (a few million params) to validate the
pipeline end to end. Run small ablations here (MHA vs GQA,
activation, position encoding, vocab size) instead of guessing.
Exit: stable loss curves, working eval, resume-from-checkpoint works.

## Phase 7: 50M model
Full pretraining run on the frozen spec.
Exit: perplexity and benchmark results published against baselines.

## Phase 8: Post-training and optimization (combined)
Instruction tuning, INT8/INT4 quantization, CPU inference
optimization. Measure quality loss, latency, tokens/sec, real RAM.
Exit: a 50M model that runs locally on an i5-class CPU with
published measurements.

## Phase 9: 100M model
## Phase 10: 250M model
## Phase 11: 500M maximum model
Scale only using lessons and scaling curves from earlier phases.

## Changes from the original plan
- Evaluation is designed before training, not after.
- A tiny proof-of-concept comes before the 50M run, and is where
  architecture choices are tested empirically.
- Quantization and CPU optimization are part of one phase, and will
  inform later architecture choices (e.g. KV-cache size).