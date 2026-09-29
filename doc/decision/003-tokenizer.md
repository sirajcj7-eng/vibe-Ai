# 003: Tokenizer

Status: **Accepted (provisional)**. Frozen before the 50M run;
changing it later invalidates the trained model.

## Decision summary

| Item | Choice |
|---|---|
| Algorithm | Byte-level BPE |
| Vocabulary | 16,384 total, including special tokens |
| Normalization | Unicode NFC, no lowercasing |
| Numbers | Split into single digits |
| Special tokens | ~32 reserved slots (see below) |
| Library | Hugging Face `tokenizers` (training tool only) |
| Storage | uint16 token IDs |

## Why byte-level BPE
- The base alphabet is the 256 bytes, so any text, code or emoji
  can be encoded. No unknown token, ever.
- Lossless: decode(encode(x)) == x.
- Simple, proven, and fast on CPU.

Alternatives:
- Unigram/SentencePiece: comparable quality, but byte fallback and
  code handling are less clean.
- WordPiece: needs an unknown token and is weaker on code.
- Character/byte-level models: sequences 4x longer or more, which
  is too costly at fixed context.

## Vocabulary size
16,384 = 2^14, a multiple of 128 (efficient matrix shapes).
Recap from 001: embeddings cost V*d, and 16k keeps them at 16.7%
of the model. Expected compression is about 3.6-4.0 bytes/token on
English (estimate, to be measured).

Token count implication at 3.8 bytes/token: 5B tokens is roughly
19 GB of text. The data plan (Phase 4) must be sized for this.

## Numbers
Single-digit splitting ("2026" -> "2","0","2","6") makes arithmetic
patterns consistent instead of arbitrary chunks. It costs extra
tokens on numeric text, but math ability is a stated goal.

## Special tokens
Reserved now so we never resize the embedding later. Cost is
32 * 512 = 16K parameters (negligible).

| Token | Purpose |
|---|---|
| <|endoftext|> | Document separator and EOS |
| <|pad|> | Padding (batching at inference) |
| <|system|> <|user|> <|assistant|> | Chat roles |
| <|end|> | End of a chat turn |
| <|tool_call|> <|tool_result|> | Tool/function calling |
| <|fim_prefix|> <|fim_middle|> <|fim_suffix|> | Code fill-in-the-middle |
| ~20 unused | Reserved for future use |

Special tokens get IDs 0-31. Base vocabulary fills the rest.
Untrained reserved embeddings stay near their init until used in
instruction tuning; that is expected.

## Training the tokenizer
- Trained on a sample of the same mix as the pretraining data
  (about 1-2 GB), never on eval sets.
- Do it after the data mix is drafted (Phase 4 draft), otherwise
  the vocabulary will not match the corpus.
- Deterministic: fixed seed, saved config, versioned files
  (`tokenizer-v1.json`).

## Evaluation of the tokenizer
- Bytes per token by domain: English prose, code, math, other
  languages
- Fertility (tokens per word)
- Round-trip test on a large sample (must be 100% lossless)
- Count of rarely used tokens (wasted vocabulary)
- Tokenizer memory and encode speed on CPU

Important: compare vocab sizes with **bits per byte** (loss divided
by bytes per token), not loss per token. Loss per token is not
comparable across tokenizers.

## PoC ablation
In Phase 6 train tiny models with 8k, 16k and 32k vocab at equal
compute, compared on bits per byte. If 16k does not win or tie,
revise this doc before the 50M run.

## Open questions
- Multilingual coverage: English-first for now. Adding languages
  later will require more vocabulary or a retrained tokenizer.
- Code share in the tokenizer sample (depends on the data mix)
- Whether to add a small number of whitespace tokens for code
  indentation

## Alternatives rejected for now
- 32k+ vocab (embedding share too large at 50M)
- Byte-level models with no BPE (context and compute cost)
- Lowercasing (loses information for code and names)