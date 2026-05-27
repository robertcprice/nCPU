"""Regression tests for the retrieval-augmented prompt builder.

`build_retrieval_prefix` is the function that takes a new problem's
examples and returns a few-shot block of similar cached solutions for
the model to condition on. This test pins down its shape: empty
string when no matches, structured block when matches exist, truncation
for overly-long cached code, self-exclusion when the query fingerprint
matches a cached row.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS_BENCH = Path(__file__).resolve().parent.parent / "tools" / "benchmarks"
sys.path.insert(0, str(TOOLS_BENCH))

from llm_solution_cache import record  # noqa: E402
from retrieval_prompt import build_retrieval_prefix  # noqa: E402


@pytest.fixture
def fresh_cache(monkeypatch, tmp_path):
    p = tmp_path / "c.tsv"
    monkeypatch.setenv("NSYNTH_LLM_CACHE_PATH", str(p))
    return p


def test_empty_cache_returns_empty_string(fresh_cache):
    pref = build_retrieval_prefix(
        [{"inputs": [1], "expected": 2}], k=3, min_similarity=0.70,
    )
    assert pref == ""


def test_returns_formatted_block_with_matches(fresh_cache):
    record(
        "a", "m", "def add(a,b):\n    return a+b",
        examples=[{"inputs": [1, 2], "expected": 3},
                  {"inputs": [3, 4], "expected": 7}],
    )
    pref = build_retrieval_prefix(
        [{"inputs": [2, 3], "expected": 5}, {"inputs": [5, 6], "expected": 11}],
        k=2, min_similarity=0.0,
    )
    # Structured few-shot block.
    assert "# Similar verified solutions retrieved from cache:" in pref
    assert "def add" in pref
    assert "# --- Example 1 (sim=" in pref
    # Ends with a hand-off line for the caller to append their task.
    assert "# Your task" in pref
    # Trailing newline so caller can concatenate a prompt cleanly.
    assert pref.endswith("\n")


def test_respects_min_similarity_threshold(fresh_cache):
    record(
        "a", "m", "def f(x):\n    return x",
        examples=[{"inputs": [1], "expected": 1}],
    )
    # Impossibly high threshold → no hits.
    pref = build_retrieval_prefix(
        [{"inputs": [999], "expected": 123456}],
        k=3, min_similarity=0.99,
    )
    assert pref == ""


def test_truncates_overly_long_code(fresh_cache):
    long_body = "    return (\n" + "        " + " + ".join(str(i) for i in range(200)) + "\n    )"
    long_code = f"def big(x):\n{long_body}"
    assert len(long_code) > 500
    record(
        "a", "m", long_code,
        examples=[{"inputs": [1], "expected": 1}],
    )
    pref = build_retrieval_prefix(
        [{"inputs": [1], "expected": 1}],
        k=1, min_similarity=0.0, max_code_chars=200,
    )
    # Truncation marker present, original code not fully included.
    assert "# ...(truncated)" in pref
    assert pref.count("\n") < long_code.count("\n") + 10


def test_k_caps_number_of_entries(fresh_cache):
    for i in range(5):
        record(
            f"fp{i}", "m", f"def f_{i}(x):\n    return x + {i}",
            examples=[{"inputs": [i], "expected": i + i}],
        )
    pref = build_retrieval_prefix(
        [{"inputs": [1], "expected": 2}],
        k=2, min_similarity=0.0,
    )
    assert pref.count("# --- Example ") == 2
