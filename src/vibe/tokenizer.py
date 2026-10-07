"""Byte-level BPE tokenizer (docs/decisions/003-tokenizer.md).

Uses the Hugging Face `tokenizers` library as a training tool. The
design (vocab, special tokens, digit splitting, normalization) is ours.
"""
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

# IDs 0-31 are reserved, in this exact order, so they never move.
SPECIAL_TOKENS = [
    "<|endoftext|>",   # 0: document separator / EOS
    "<|pad|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",         # end of a chat turn
    "<|tool_call|>",
    "<|tool_result|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
] + [f"<|reserved_{i}|>" for i in range(21)]

N_SPECIAL = len(SPECIAL_TOKENS)  # 32
EOS_TOKEN = SPECIAL_TOKENS[0]
BYTE_ALPHABET_SIZE = 256


def train_tokenizer(
    files: list[str | Path],
    vocab_size: int,
    out_path: str | Path | None = None,
) -> Tokenizer:
    if vocab_size < BYTE_ALPHABET_SIZE + N_SPECIAL:
        raise ValueError(f"vocab_size must be at least {BYTE_ALPHABET_SIZE + N_SPECIAL}")

    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=False,
    )
    tok.train([str(f) for f in files], trainer)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tok.save(str(out_path))
    return tok


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def encode_text(tok: Tokenizer, text: str) -> list[int]:
    return tok.encode(text).ids


def decode_ids(tok: Tokenizer, ids: list[int]) -> str:
    return tok.decode(ids, skip_special_tokens=False)