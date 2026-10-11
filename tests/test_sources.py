import pytest
import yaml

from vibe.sources import (
    RESERVE_SHARE, SOURCES, TOTAL_TOKENS, doc_stats, group_shares,
    license_values, metadata_keys, target_tokens, update_manifest,
)


def test_shares_plus_reserve_equal_one():
    assert sum(s.share for s in SOURCES) + RESERVE_SHARE == pytest.approx(1.0)


def test_group_shares_match_doc_008():
    assert group_shares() == pytest.approx({
        "web_spoken_edu": 0.30, "qa_discussion": 0.16, "reference": 0.14,
        "science": 0.11, "code": 0.15, "books": 0.09,
    })


def test_keys_unique_and_candidates_prefer_filtered():
    keys = [s.key for s in SOURCES]
    assert len(keys) == len(set(keys))
    for s in SOURCES:
        assert all(c.startswith("common-pile/") for c in s.candidates)
        assert s.candidates[0].endswith("_filtered")


def test_target_tokens_cover_the_budget_minus_reserve():
    total = sum(target_tokens(s) for s in SOURCES)
    assert total == pytest.approx(TOTAL_TOKENS * (1 - RESERVE_SHARE), rel=1e-6)


def test_doc_stats():
    st = doc_stats([10, 20, 30, 40])
    assert st["docs"] == 4 and st["chars"] == 100 and st["mean"] == 25
    assert st["median"] == 30 and st["max"] == 40 and st["p95"] == 40
    assert doc_stats([])["docs"] == 0


def test_license_values_and_metadata_keys():
    metas = [{"license": "CC BY", "url": "x"}, {"license": "CC BY", "url": "y"},
             {"License_Type": "MIT"}, None, "junk"]
    assert license_values(metas) == {"license": {"CC BY": 2}, "License_Type": {"MIT": 1}}
    assert metadata_keys(metas) == ["License_Type", "license", "url"]


def test_update_manifest_keeps_other_sections(tmp_path):
    path = tmp_path / "manifest.yaml"
    update_manifest(path, "new_section", {"a": 1})  # file does not exist yet
    path.write_text(yaml.safe_dump({"poc_corpus": {"books": 34}}), encoding="utf-8")
    update_manifest(path, "common_pile", {"sources": {}})
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["poc_corpus"] == {"books": 34} and data["common_pile"] == {"sources": {}}