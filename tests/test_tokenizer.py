import pytest

from vibe.tokenizer import (
    N_SPECIAL,
    SPECIAL_TOKENS,
    decode_ids,
    encode_text,
    load_tokenizer,
    train_tokenizer,
)

SAMPLES = [
    "Hello, world! This is a small sample of English text.",
    "def add(a, b):\n    return a + b\n",
    "The year 2026 has 365 days; pi is about 3.14159.",
    "naïve café — 你好 🙂",
    "  leading and trailing spaces  \n",
    "tab\tseparated\tvalues",
]


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok")
    corpus = d / "corpus.txt"
    corpus.write_text("\n".join(SAMPLES * 40), encoding="utf-8")
    return train_tokenizer([corpus], vocab_size=320, out_path=d / "tok.json")


def test_special_tokens_have_fixed_ids(tok):
    assert N_SPECIAL == 32
    for i, name in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(name) == i


def test_special_token_string_encodes_to_its_id(tok):
    assert encode_text(tok, "<|endoftext|>") == [0]


def test_vocab_size_in_range(tok):
    assert 288 <= tok.get_vocab_size() <= 320


@pytest.mark.parametrize(
    "text", SAMPLES + ["", "line1\r\nline2", "<<>> {} [] ()", "x" * 500]
)
def test_roundtrip_is_lossless(tok, text):
    assert decode_ids(tok, encode_text(tok, text)) == text


def test_digits_are_split_individually(tok):
    ids = encode_text(tok, "2026")
    assert len(ids) == 4
    assert [decode_ids(tok, [i]) for i in ids] == ["2", "0", "2", "6"]


def test_nfc_normalization(tok):
    assert encode_text(tok, "e\u0301") == encode_text(tok, "\u00e9")


def test_save_and_load_give_same_ids(tok, tmp_path):
    path = tmp_path / "t.json"
    tok.save(str(path))
    loaded = load_tokenizer(path)
    for text in SAMPLES:
        assert encode_text(loaded, text) == encode_text(tok, text)


def test_too_small_vocab_rejected(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        train_tokenizer([f], vocab_size=100)