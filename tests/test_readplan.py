from pathlib import Path

import pytest

from vibe.filters import CODE_LANGUAGE_TARGETS
from vibe.readplan import ReadItem, allocate_with_rollover, load_plan, plan_reads
from vibe.sources import SOURCES

ROOT = Path(__file__).resolve().parents[1]


def by_stratum(items):
    out = {}
    for it in items:
        out[it.stratum] = out.get(it.stratum, 0) + it.quota
    return out


def test_rollover_moves_shortfall_within_the_group():
    final = allocate_with_rollover({"a": 100, "b": 100}, {"a": 40, "b": 1000},
                                   {"a": "g", "b": "g"})
    assert final == pytest.approx({"a": 40, "b": 160})


def test_rollover_is_proportional_and_conserves_the_total():
    groups = {"a": "g", "b": "g", "c": "g"}
    final = allocate_with_rollover({"a": 100, "b": 100, "c": 100},
                                   {"a": 10, "b": 120, "c": 1000}, groups)
    assert final["a"] == pytest.approx(10)
    assert final["b"] == pytest.approx(120)   # saturated at its availability
    assert final["c"] == pytest.approx(170)
    assert sum(final.values()) == pytest.approx(300)


def test_rollover_stays_in_the_group_and_can_lose_the_shortfall():
    other = allocate_with_rollover({"a": 100, "b": 100}, {"a": 10, "b": 1000},
                                   {"a": "g1", "b": "g2"})
    assert other == pytest.approx({"a": 10, "b": 100})
    full = allocate_with_rollover({"a": 100, "b": 100}, {"a": 10, "b": 100},
                                  {"a": "g", "b": "g"})
    assert full == pytest.approx({"a": 10, "b": 100})


def test_plan_reads_spreads_the_quota_over_random_shards():
    items = plan_reads("cccc", 1000, 100, max_shards=10, seed=0)
    assert len(items) == 10 and sum(i.quota for i in items) == 1000
    assert len({i.shard for i in items}) == 10 and all(0 <= i.shard < 100 for i in items)
    assert items == plan_reads("cccc", 1000, 100, max_shards=10, seed=0)
    other_seed = plan_reads("cccc", 1000, 100, max_shards=10, seed=1)
    assert [i.shard for i in items] != [i.shard for i in other_seed]
    assert plan_reads("news", 500, 1) == [ReadItem(0, 500, None)]


def test_plan_reads_stackexchange_caps_stackoverflow_and_drops_excluded_sites():
    strata = {"stackoverflow.com": list(range(31)), "math.stackexchange.com": [31, 32],
              "es.stackoverflow.com": [33], "meta.stackoverflow.com": [34],
              "diy.stackexchange.com": [35]}
    by = by_stratum(plan_reads("stackexchange", 1000, 36, strata))
    assert set(by) == {"stackoverflow.com", "math.stackexchange.com", "diy.stackexchange.com"}
    assert by["stackoverflow.com"] == pytest.approx(350, abs=2)
    assert by["math.stackexchange.com"] == pytest.approx(433, abs=3)
    assert by["diy.stackexchange.com"] == pytest.approx(217, abs=3)
    assert sum(by.values()) == pytest.approx(1000, abs=5)


def test_plan_reads_code_languages_are_renormalised():
    strata = {"Python": [0, 1], "JavaScript": [2], "Haskell": [9]}
    by = by_stratum(plan_reads("stackv2_edu", 1000, 10, strata))
    py, js = CODE_LANGUAGE_TARGETS["Python"], CODE_LANGUAGE_TARGETS["JavaScript"]
    assert "Haskell" not in by
    assert by["Python"] == pytest.approx(1000 * py / (py + js), abs=2)
    assert sum(by.values()) == pytest.approx(1000, abs=3)


def test_committed_plan_covers_every_source():
    plan = load_plan(ROOT / "data" / "commonpile_plan.yaml")
    assert set(plan) == {s.key for s in SOURCES}
    for info in plan.values():
        assert len(info["revision"]) == 40 and set(info["revision"]) <= set("0123456789abcdef")
        assert info["repo"].startswith("common-pile/") and info["shards"] >= 1