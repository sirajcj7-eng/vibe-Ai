import hashlib

import numpy as np
import pytest

from vibe.corpus import (
    StratumBudget, TokenShardWriter, doc_key, encode_documents, is_val_doc,
    process_documents,
)
from vibe.data import TOKEN_DTYPE, TokenStream
from vibe.tokenizer import train_tokenizer

GOOD = "The quick brown fox jumps over the lazy dog. " * 20  # 900 characters
CC_BY = "Creative Commons - Attribution - https://creativecommons.org/licenses/by/4.0/"
CC_BY_SA = "Creative Commons - Attribution Share-Alike - https://creativecommons.org/licenses/by-sa/4.0/"


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok")
    corpus = d / "c.txt"
    corpus.write_text("\n".join([GOOD, "def f(x): return x + 1", "hello world 2026"] * 30),
                      encoding="utf-8")
    return train_tokenizer([corpus], vocab_size=300)


def make_doc(i, text=GOOD, **meta):
    m = {"license": CC_BY}
    m.update(meta)
    return {"id": f"doc-{i}", "text": text, "metadata": m}


def writers(tmp_path, shard_tokens=100_000):
    return (TokenShardWriter(tmp_path, "train", "src", shard_tokens),
            TokenShardWriter(tmp_path, "val", "src", shard_tokens))


def test_is_val_doc_is_deterministic_and_about_half_a_percent():
    keys = [f"id-{i}" for i in range(20_000)]
    flags = [is_val_doc(k) for k in keys]
    assert flags == [is_val_doc(k) for k in keys]
    assert 0.003 < sum(flags) / len(flags) < 0.007
    assert not any(is_val_doc(k, 0) for k in keys[:500])


def test_doc_key_fallbacks():
    assert doc_key({"id": "x"}) == "x"
    assert doc_key({"metadata": {"provenance": "file.json.gz:7"}}) == "file.json.gz:7"
    k = doc_key({"text": "abc"})
    assert k == hashlib.sha1(b"abc").hexdigest() == doc_key({"text": "abc"})


def test_budget_caps_and_exclusive_strata():
    b = StratumBudget(1000, caps={"A": 0.5})
    assert b.wants("A") and b.wants("B") and b.wants(None)
    b.add("A", 500)
    assert not b.wants("A") and b.wants("B") and not b.done
    b.add("B", 500)
    assert b.done and not b.wants("B")

    ex = StratumBudget(1000, caps={"Py": 0.6, "JS": 0.4}, exclusive=True)
    assert ex.wants("Py") and not ex.wants("Go") and not ex.wants(None)
    ex.add("Py", 600)
    ex.add("JS", 400)
    assert ex.done


def test_shard_writer_roundtrip(tmp_path):
    w = TokenShardWriter(tmp_path, "train", "src", shard_tokens=100)
    rng = np.random.default_rng(0)
    pieces = [rng.integers(0, 300, n).astype(TOKEN_DTYPE) for n in (30, 90, 5, 120, 7)]
    for p in pieces:
        w.add(p)
    names = w.close()
    assert w.total == 252 and len(names) == 3 and names[0] == "train/src_000.bin"
    stream = TokenStream([tmp_path / n for n in names])
    assert stream.total == 252
    assert np.array_equal(stream.read(0, 252), np.concatenate(pieces))


def test_shard_writer_empty_close(tmp_path):
    assert TokenShardWriter(tmp_path, "val", "src").close() == []


def test_encode_documents_appends_eos(tok):
    arrays = encode_documents(tok, ["hello world", ""])
    assert arrays[0][-1] == 0 and len(arrays[0]) > 1
    assert arrays[1].tolist() == [0]


def test_process_filters_and_stops_at_budget(tok, tmp_path):
    docs = [make_doc(0), make_doc(1, license="GPL-3.0"), make_doc(2, text="too short")]
    docs += [make_doc(i) for i in range(3, 20)]
    tw, vw = writers(tmp_path)
    budget = StratumBudget(1500)
    res = process_documents("news", docs, tok, budget, tw, vw, batch_chars=1, val_permille=0)
    assert res.drops["license"] == 1 and res.drops["short"] == 1
    assert res.reached_budget and res.train_tokens >= 1500
    assert res.docs_seen < len(docs)                  # stopped early
    assert tw.total == res.train_tokens and res.val_tokens == 0


def test_process_routes_validation_documents(tok, tmp_path):
    docs = [make_doc(i) for i in range(40)]
    tw, vw = writers(tmp_path)
    res = process_documents("news", docs, tok, StratumBudget(10**9), tw, vw,
                            val_permille=500)
    val_ids = {d["id"] for d in docs if is_val_doc(d["id"], 500)}
    assert 0 < len(val_ids) < 40
    assert res.docs_kept_val == len(val_ids)
    assert res.docs_kept_train == 40 - len(val_ids)
    assert res.val_bytes == len(val_ids) * len(GOOD.encode("utf-8"))
    assert vw.total == res.val_tokens and tw.total == res.train_tokens


def test_process_blocklist_drops_contaminated(tok, tmp_path):
    docs = [make_doc(i, text=GOOD.replace("fox", "FOX") if i % 2 == 0 else GOOD)
            for i in range(10)]
    tw, vw = writers(tmp_path)
    res = process_documents("news", docs, tok, StratumBudget(10**9), tw, vw,
                            blocklist=lambda t: "FOX" in t, val_permille=0)
    assert res.drops["contaminated"] == 5 and res.docs_kept_train == 5


def test_process_caps_a_dominant_stratum(tok, tmp_path):
    docs = []
    for i in range(30):
        site = "stackoverflow.com" if i % 2 == 0 else "diy.stackexchange.com"
        docs.append(make_doc(i, license=CC_BY_SA, site=site))
    tw, vw = writers(tmp_path)
    budget = StratumBudget(6000, caps={"stackoverflow.com": 0.35})
    res = process_documents("stackexchange", docs, tok, budget, tw, vw,
                            batch_chars=1, val_permille=0)
    doc_tokens = len(encode_documents(tok, [GOOD])[0])
    so = res.strata_tokens["stackoverflow.com"]
    diy = res.strata_tokens["diy.stackexchange.com"]
    assert so <= 0.35 * 6000 + doc_tokens
    assert diy > so
    assert res.drops["stratum_full"] > 0