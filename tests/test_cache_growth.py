"""Regression test for the cache-growth measurement script.

`measure_cache_growth` is the script that produces the "hit rate vs
cache size" curve used in the retrieval writeup. We pin down:
  - the curve is monotonic non-decreasing in cache size at each threshold
  - sizes 1..N produce exactly N rows in the output
  - the mean top-sim is in [0, 1]
  - missing-examples rows in the holdout are skipped gracefully
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS_BENCH = Path(__file__).resolve().parent.parent / "tools" / "benchmarks"
sys.path.insert(0, str(TOOLS_BENCH))

from llm_solution_cache import record  # noqa: E402
from measure_cache_growth import measure  # noqa: E402


@pytest.fixture
def seeded_cache(monkeypatch, tmp_path):
    """Populate a 10-row cache with varied-shape examples, return path."""
    cache_path = tmp_path / "seed.tsv"
    monkeypatch.setenv("NSYNTH_LLM_CACHE_PATH", str(cache_path))
    for i in range(10):
        record(
            f"fp{i}", "test-model",
            f"def f_{i}(x, y):\n    return x + y + {i}",
            examples=[
                {"inputs": [i, i + 1], "expected": i + (i + 1) + i},
                {"inputs": [i * 2, i], "expected": i * 2 + i + i},
            ],
        )
    return cache_path


def test_output_row_count(seeded_cache):
    holdout = [
        {"examples": [{"inputs": [3, 4], "expected": 7},
                      {"inputs": [10, 5], "expected": 15}]},
        {"examples": [{"inputs": [1, 1], "expected": 2}]},
    ]
    results = measure(seeded_cache, holdout, sizes=[2, 5], thresholds=[0.5, 0.7])
    assert len(results) == 4  # 2 sizes × 2 thresholds


def test_hit_rate_monotonic_in_size(seeded_cache):
    """Bigger cache → equal or more hits at a fixed threshold."""
    holdout = [
        {"examples": [{"inputs": [3, 4], "expected": 7}]},
        {"examples": [{"inputs": [10, 5], "expected": 15}]},
        {"examples": [{"inputs": [1, 1], "expected": 2}]},
    ]
    results = measure(seeded_cache, holdout,
                       sizes=[2, 5, 10], thresholds=[0.5])
    by_size = {r["size"]: r["hit_rate"] for r in results}
    assert by_size[2] <= by_size[5] <= by_size[10]


def test_hit_rate_bounded(seeded_cache):
    holdout = [{"examples": [{"inputs": [1, 2], "expected": 3}]}]
    results = measure(seeded_cache, holdout,
                       sizes=[5], thresholds=[0.0, 1.0])
    for r in results:
        assert 0.0 <= r["hit_rate"] <= 1.0
        assert 0.0 <= r["mean_top_sim"] <= 1.0


def test_skips_holdout_without_examples(seeded_cache):
    """Holdout problems missing examples are filtered before lookup
    in the CLI; here measure() sees the filtered list. Verify no
    crash when the list contains empty-example entries anyway."""
    holdout = [
        {"examples": [{"inputs": [1, 2], "expected": 3}]},
        {"examples": []},  # empty — skipped
    ]
    results = measure(seeded_cache, holdout,
                       sizes=[5], thresholds=[0.5])
    # We run against 2 "problems" but only 1 has examples, so
    # the denominator is 2 (total) with hits ≤ 1.
    assert results[0]["total"] == 2
    assert results[0]["hits"] <= 1


def test_impossible_threshold_gives_zero_hits(seeded_cache):
    holdout = [{"examples": [{"inputs": [1, 2], "expected": 3}]}]
    results = measure(seeded_cache, holdout,
                       sizes=[10], thresholds=[1.01])  # unreachable
    assert results[0]["hits"] == 0
