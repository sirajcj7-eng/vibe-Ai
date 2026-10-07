# 005: Evaluation Plan

Status: **Proposed**. Benchmarks are chosen for signal at 50M
scale. Expected score ranges are estimates and must be checked
against our own measured baselines.

## Principles
1. Define baselines before training anything.
2. Compare across models and tokenizers with bits per byte (BPB),
   never loss per token.
3. Report uncertainty. On a 1,000-question multiple-choice set the
   standard error is about 1.5 points, so differences under ~3
   points are noise.
4. Every result is saved with: git commit, config, tokenizer
   version, dataset version, seed, hardware.
5. Benchmarks are on the decontamination blocklist (see 004)
   before any training.

## Why many popular benchmarks are useless at 50M
MMLU, GSM8K, HumanEval and similar tests sit near random or zero
for a model this small. A flat line cannot show improvement. We
use benchmarks with real signal at this size, and add our own
small tests for capabilities where public ones fail.

## Language modeling (primary metric)
- Validation loss and BPB on the held-out set (see 004), with
  per-domain slices: web, code, math, books
- Training and validation loss curves
- Sanity anchor: a randomly initialized model should have loss
  ln(16384) = 9.70 nats (14 bits per token). If step-0 loss differs
  much, something is wrong.

## Baselines (set before training)
| Baseline | Purpose |
|---|---|
| Random init | Loss ln(V) = 9.70 |
| Unigram / n-gram model on our data | Cheap lower bar the model must beat clearly |
| Random guessing on each benchmark | Chance floor |
| Public small models (e.g. Pythia-70M, GPT-2 small) | Reference only. Evaluation use only, never weights or data |

Public models use different tokenizers and data, so compare them
using BPB and benchmark accuracy, and state the caveat.

## Capability benchmarks (zero-shot, log-likelihood scoring)

| Area | Benchmark | Why |
|---|---|---|
| Commonsense | HellaSwag, PIQA, Winogrande | Standard small-model signal |
| Science / knowledge | ARC-Easy, SciQ | Gives signal where MMLU does not |
| Language | LAMBADA, BLiMP | Long-range word prediction, grammar |
| Knowledge (later) | MMLU | Tracked from 100M up; expected near chance at 50M |

Tooling: EleutherAI lm-evaluation-harness. Use the same version
and settings for every model.

Orientation only: models near 70M parameters usually score close
to chance on Winogrande and around 25-40% on HellaSwag/ARC-style
tasks. Our measured baselines replace these numbers.

## Math
Public math benchmarks give no signal at 50M. Custom tests:
- n-digit addition, subtraction, multiplication (1-4 digits),
  exact match
- Simple word problems with a one-step operation
- Number comparison and ordering
Generated programmatically with a fixed seed and kept out of
training data. GSM8K tracked from 250M up.

## Coding
- Held-out code BPB, per language
- Small custom set: complete a function body, closing brackets,
  syntax-valid next line, simple known snippets
- HumanEval/MBPP tracked from 250M up (near zero at 50M)

## Instruction following (after post-training only)
- Custom set of ~100 prompts with checkable constraints (answer in
  one sentence, include a given word, output a list of N items)
- Scored by simple code checks, not by a model judge
- IFEval tracked from 100M up

## Multi-turn and context handling
- Passkey/needle retrieval at 256, 512, 1024, 2048 tokens
- Loss vs token position (does loss keep improving across the
  context window?)
- Multi-turn consistency set: ~30 conversations where a later
  turn depends on an earlier fact

## Qualitative evaluation
- Fixed set of ~50 prompts, run at every milestone checkpoint
- Greedy and temperature 0.7, fixed seed
- Outputs saved verbatim in `results/` so changes can be compared
- A running failure-mode log: repetition, hallucinated facts,
  broken formatting, refusal to stop, and so on

## Performance and memory (CPU)
Measured on a specified machine (CPU model, cores, RAM, OS):
- Prefill speed (tokens/sec) at 128, 512, 2048 tokens
- Decode speed (tokens/sec)
- Time to first token
- Peak RSS (real process memory), with and without KV cache filled
- Model file size per precision (FP16, INT8, INT4)
- Threads used (1, 4, all)
- Quality change from quantization: BPB and benchmark deltas

Estimates in 001 are replaced with these measurements.

## When evaluation runs
| Moment | What |
|---|---|
| Every 500 steps | Validation loss and BPB |
| Every ~2,500 steps | Fast subset: LAMBADA, custom math, 10 qualitative prompts |
| Milestone checkpoints (1B, 5B tokens) | Full suite, qualitative set, CPU benchmarks |
| After post-training | Instruction following, multi-turn |
| After quantization | Quality and speed deltas |

## Go/no-go gates (provisional)
- **Gate 1, proof-of-concept:** loss falls well below the unigram
  baseline; resume from checkpoint reproduces the same loss.
- **Gate 2, 1B checkpoint:** BPB clearly beats the n-gram
  baseline; at least two benchmarks above chance by more than 2
  standard errors; coherent paragraph-length generation.
- **Gate 3, 5B final:** improvement over the 1B checkpoint on
  validation and benchmarks; documented failure modes; CPU
  speed and memory measured.
Numeric thresholds are set after we have baseline measurements.

## Reporting format
One folder per run in `results/`: config, metrics JSON, sample
outputs, hardware notes, and a short written summary including
failures. Nothing is deleted.

## Open questions
- Exact lm-evaluation-harness task versions to pin
- Custom math/coding set sizes
- Reference CPU for official speed numbers