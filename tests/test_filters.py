import pytest

from vibe.filters import (
    CODE_LANGUAGE_TARGETS, STRATA_CAPS, alpha_ratio, check_document, stratum,
)

LONG = "The quick brown fox jumps over the lazy dog. " * 40  # about 1,800 characters
CC_BY = "Creative Commons - Attribution - https://creativecommons.org/licenses/by/4.0/"
CC0 = "Creative Commons Zero - Public Domain - https://creativecommons.org/publicdomain/zero/1.0/"


def meta(**kw):
    base = {"license": CC_BY}
    base.update(kw)
    return base


def test_good_document_passes():
    assert check_document("news", LONG, meta()) == (True, "ok")


def test_license_rules():
    assert check_document("news", LONG, meta(license="GPL-3.0")) == (False, "license")
    assert check_document("news", LONG, {}) == (False, "license")
    assert check_document("cccc", LONG, {}) == (True, "ok")  # verified at domain level


def test_minimum_length():
    assert check_document("news", "x" * 499, meta()) == (False, "short")


def test_per_source_minimum_length():
    text = "abc def " * 45  # 360 characters
    assert check_document("news", text, meta()) == (False, "short")
    assert check_document("arxiv_abstracts", text, meta(license=CC0)) == (True, "ok")


def test_replacement_character_and_alpha_ratio():
    assert alpha_ratio("abc 123") == pytest.approx(3 / 7)
    assert alpha_ratio("") == 0.0
    assert check_document("news", LONG + "\ufffd", meta()) == (False, "replacement_char")
    digits = "1234567890 " * 100
    assert check_document("news", digits, meta()) == (False, "low_alpha")
    assert check_document("stackexchange", digits, meta(site="stackoverflow.com")) == (True, "ok")  # exempt


def test_wikimedia_keeps_only_wikipedia_articles():
    assert check_document("wikimedia", LONG, meta(wiki="wikipedia.com", namespace="0")) == (True, "ok")
    assert check_document("wikimedia", LONG, meta(wiki="wiktionary.com", namespace="0")) == (False, "wiki_filter")
    assert check_document("wikimedia", LONG, meta(wiki="wikipedia.com", namespace="1")) == (False, "wiki_filter")


def test_stackv2_drops_generated_and_vendor_files():
    code = "def f(x):\n    return x + 1\n" * 30
    base = meta(license="MIT")
    assert check_document("stackv2_edu", code, base) == (True, "ok")
    assert check_document("stackv2_edu", code, {**base, "is_generated": True}) == (False, "generated_or_vendor")
    assert check_document("stackv2_edu", code, {**base, "is_vendor": "True"}) == (False, "generated_or_vendor")


def test_strata_and_tables():
    assert stratum("stackexchange", {"site": "diy.stackexchange.com"}) == "diy.stackexchange.com"
    assert stratum("stackv2_edu", {"language": "Python"}) == "Python"
    assert stratum("news", {"site": "x"}) is None and stratum("news", None) is None
    assert STRATA_CAPS["stackexchange"]["stackoverflow.com"] == 0.35
    assert sum(CODE_LANGUAGE_TARGETS.values()) == pytest.approx(1.0)
def test_stackexchange_site_filter_and_code_targets():
    assert check_document("stackexchange", LONG, meta(site="math.stackexchange.com")) == (True, "ok")
    assert check_document("stackexchange", LONG, meta(site="es.stackoverflow.com")) == (False, "site_excluded")
    assert check_document("stackexchange", LONG, meta(site="meta.stackoverflow.com")) == (False, "site_excluded")
    assert "SQL" not in CODE_LANGUAGE_TARGETS
    assert CODE_LANGUAGE_TARGETS["Python"] == pytest.approx(0.32)
