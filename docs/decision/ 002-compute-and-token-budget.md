# 002: Compute and Token Budget (50M)

Status: **Proposed**. Every throughput number is an estimate and
must be replaced with measurements from the Phase 6 proof-of-concept.

## Hardware available
- Free GPUs: Kaggle and Colab (T4/P100 class)
- Laptop CPU
- Session limits and weekly quotas change over time and must be
  re-checked before each run. Training must survive interruption.

## FLOPs per token

Rule of thumb 6N covers the weight matmuls. It leaves out attention
over the context, so we add that term (halved for causal masking):

    F_token ~ 6N + 6 * L * d * T

With N = 50.35M, L = 16, d = 512, T = 2048:
- 6N = 302M
- 6*L*d*T = 6 * 16 * 512 * 2048 = 101M
- **F_token ~ 403M FLOPs per training token**

Correction to 001: its 6ND preview omitted the attention term.
At T = 2048 the real cost is about 1.3x higher.

## Total compute by token budget

| Tokens D | Tokens/param | Total FLOPs |
|---|---|---|
| 1B | ~20 | 4.0e17 |
| 5B | ~100 | 2.0e18 |
| 10B | ~200 | 4.0e18 |

## Throughput estimate (GPU)

Assume 7 TFLOPs effective (T4-class, fp16, roughly 10-15% of peak;
small models with d = 512 use GPUs poorly). Plausible range: 5-10.

    tokens/sec = 7e12 / 4.03e8 ~ 17,000

| Tokens | GPU-hours | At ~30 GPU-hours/week |
|---|---|---|
| 1B | ~16 h | ~0.5 week |
| 5B | ~80 h | ~2.7 weeks |
| 10B | ~160 h | ~5.3 weeks |

The 30 h/week figure is an assumption about Kaggle's quota and
must be verified. T4 has no bf16, so use fp16 with loss scaling
(GradScaler). Sessions can end without warning.

## Laptop CPU role

Rough guess: 0.2-0.4 TFLOPs effective, so ~500-1,000 tokens/sec.
That makes 1B tokens take 2+ weeks: too slow for pretraining.
Use the CPU for:
- Debugging and unit-level correctness checks
- Tokenizer training and data preparation
- Tiny-model experiments
- Inference benchmarks (the real target platform)

## Token budget: options

| Option | Tokens | Verdict |
|---|---|---|
| Chinchilla-optimal | ~1B | Compute-optimal, but leaves a small model under-trained for inference use |
| **Target** | **5B** | Best value: ~2.7 weeks of free GPU |
| Stretch | 10B | Extra gain is small; 2x the time |

Small models used for inference should be trained past the
Chinchilla point, because training cost is paid once and inference
cost is paid on every use.

**Recommendation:** 5B tokens as the target, with a 1B-token
checkpoint as the first milestone (evaluated and published). The
budget is revised after measured throughput.

## Data storage

Vocab 16,384 fits in uint16 (2 bytes/token):
- 1B tokens = 2 GB
- 5B tokens = 10 GB

Store shards of ~100M tokens so runs can stream from disk and
resume from a shard offset.

## Training hyperparameters (provisional)

| Item | Value | Note |
|---|---|---|
| Sequence length | 2048 | Documents packed, separated by an EOS token |
| Batch size | 262,144 tokens/step (2^18) | 128 sequences; use gradient accumulation |
| Steps for 5B | ~19,000 | 5e9 / 262,144 |
| Optimizer | AdamW, betas (0.9, 0.95), eps 1e-8 | |
| Weight decay | 0.1 | Not on norms or embeddings |
| Peak LR | 1e-3 (sweep 5e-4 to 3e-3 in PoC) | Small models tolerate high LR |
| Schedule | Warmup-Stable-Decay (WSD) | See below |
| Warmup | ~500 steps | |
| Decay | Last 10-20% of steps, to 10% of peak | |
| Gradient clipping | Global norm 1.0 | |
| Precision | fp16 + GradScaler on T4; bf16 on CPU/newer GPUs | |
| Validation | Fixed held-out ~10M tokens | Eval every 500 steps |
| Checkpoint | Every ~30 min or 1,000 steps | Model, optimizer, scheduler, RNG, data position |

### Why WSD instead of cosine
Cosine needs the total step count fixed up front. WSD holds the LR
constant, then decays at the end. That fits interrupted sessions
and lets us decay early to produce a usable 1B-token checkpoint,
then continue training from the pre-decay checkpoint.

### Why 262k-token batches
For a model this small, larger batches waste compute without better
loss per token. 2^18 gives ~19k optimizer steps in 5B tokens, enough
updates for stable training. Tested in Phase 6.

## Resume and reliability requirements
- Checkpoints saved to persistent storage (Drive, Kaggle output,
  or a hub), never only local session disk
- Resume must reproduce the exact data position and RNG state
- Test resume in the proof-of-concept before any long run
- Log tokens seen, loss, LR, grad norm, tokens/sec, GPU type

## Open questions
- Real tokens/sec on T4 vs P100 (measure in PoC)
- Actual Kaggle/Colab weekly limits at run time
- Whether 5B tokens of sufficient quality can be assembled and
  licensed (Phase 4)
- Peak LR and batch size sweeps

## Decision needed
Confirm the 5B target (1B first milestone).
## Decision
Accepted: 5B-token target, with a 1B-token evaluated checkpoint as
the first milestone. Revisit after measured PoC throughput.