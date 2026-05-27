"""Regression tests for the MCP server's retrieval-related tools.

Covers the three tools that form the retrieval loop over MCP:
  - cache_solution(..., examples)     — persistence with examples
  - semantic_similar(examples, ...)   — raw similarity matches
  - build_retrieval_prefix(examples)  — ready-to-paste few-shot block

Also exercises the math tools added in the same extension:
  - evaluate_expression(expression)
  - check_numeric_answer(predicted, ground_truth)

These tests invoke the tool functions directly (no JSON-RPC layer)
so they run fast and fail cleanly when an impl changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS_MCP = Path(__file__).resolve().parent.parent / "tools" / "mcp"
sys.path.insert(0, str(TOOLS_MCP))

from nsynth_mcp_server import (  # noqa: E402
    tool_cache_solution, tool_semantic_similar, tool_build_retrieval_prefix,
    tool_lookup_solution, tool_evaluate_expression, tool_check_numeric_answer,
)


@pytest.fixture
def fresh_cache(monkeypatch, tmp_path):
    p = tmp_path / "mcp_cache.tsv"
    monkeypatch.setenv("NSYNTH_LLM_CACHE_PATH", str(p))
    return p


# ─── cache_solution ─────────────────────────────────────────────────────────


def test_cache_solution_persists_examples(fresh_cache):
    r = tool_cache_solution({
        "fingerprint": "fp1", "code": "def f(x):\n    return x*2",
        "model": "test-model",
        "examples": [{"inputs": [3], "expected": 6}],
    })
    assert r["status"] == "ok"
    assert r["examples_stored"] is True

    back = tool_lookup_solution({"fingerprint": "fp1"})
    assert back["status"] == "hit"
    assert back["code"].startswith("def f")


def test_cache_solution_without_examples_still_works(fresh_cache):
    r = tool_cache_solution({
        "fingerprint": "fp2", "code": "def g():\n    pass",
    })
    assert r["status"] == "ok"
    assert r["examples_stored"] is False


def test_cache_solution_rejects_bad_examples_type(fresh_cache):
    r = tool_cache_solution({
        "fingerprint": "fp3", "code": "x",
        "examples": "not a list",
    })
    assert "error" in r


# ─── semantic_similar ───────────────────────────────────────────────────────


def test_semantic_similar_finds_persisted_row(fresh_cache):
    tool_cache_solution({
        "fingerprint": "s1", "code": "def add(a,b):\n    return a+b",
        "examples": [{"inputs": [1, 2], "expected": 3},
                     {"inputs": [5, 5], "expected": 10}],
    })
    r = tool_semantic_similar({
        "examples": [{"inputs": [2, 3], "expected": 5}],
        "k": 3, "min_similarity": 0.0,
    })
    assert len(r["matches"]) >= 1
    assert r["matches"][0]["fingerprint"] == "s1"
    assert 0.0 <= r["matches"][0]["similarity"] <= 1.0


def test_semantic_similar_empty_cache(fresh_cache):
    r = tool_semantic_similar({
        "examples": [{"inputs": [1], "expected": 1}],
        "k": 3, "min_similarity": 0.5,
    })
    assert r["matches"] == []


# ─── build_retrieval_prefix ─────────────────────────────────────────────────


def test_build_retrieval_prefix_returns_ready_to_paste_block(fresh_cache):
    tool_cache_solution({
        "fingerprint": "p1", "code": "def neg(x):\n    return -x",
        "examples": [{"inputs": [3], "expected": -3},
                     {"inputs": [-5], "expected": 5}],
    })
    r = tool_build_retrieval_prefix({
        "examples": [{"inputs": [7], "expected": -7}],
        "k": 2, "min_similarity": 0.0,
    })
    assert r["hits"] >= 1
    assert "def neg" in r["prefix"]
    assert "# Similar verified solutions" in r["prefix"]
    assert 0.0 <= r["top_similarity"] <= 1.0


def test_build_retrieval_prefix_empty_when_below_threshold(fresh_cache):
    tool_cache_solution({
        "fingerprint": "p2", "code": "def x(): pass",
        "examples": [{"inputs": [1], "expected": 1}],
    })
    r = tool_build_retrieval_prefix({
        "examples": [{"inputs": [999999], "expected": 42}],
        "k": 3, "min_similarity": 0.99,
    })
    assert r["prefix"] == ""
    assert r["hits"] == 0
    assert r["top_similarity"] is None


def test_build_retrieval_prefix_rejects_empty_examples(fresh_cache):
    r = tool_build_retrieval_prefix({"examples": [], "k": 3})
    assert "error" in r


# ─── evaluate_expression ────────────────────────────────────────────────────


def test_evaluate_expression_basic_arithmetic():
    r = tool_evaluate_expression({"expression": "3 * 45 + 7"})
    assert r["value"] == 142


def test_evaluate_expression_math_module():
    r = tool_evaluate_expression({"expression": "sqrt(16) + floor(pi)"})
    assert abs(r["value"] - 7.0) < 1e-9


def test_evaluate_expression_rejects_import():
    r = tool_evaluate_expression({"expression": "__import__('os').listdir('.')"})
    assert "error" in r


def test_evaluate_expression_rejects_import_keyword():
    r = tool_evaluate_expression({"expression": "import os"})
    assert "error" in r


# ─── check_numeric_answer ───────────────────────────────────────────────────


def test_check_numeric_answer_exact_match():
    r = tool_check_numeric_answer({"predicted": 142, "ground_truth": 142})
    assert r["match"] is True
    assert r["abs_error"] == 0


def test_check_numeric_answer_within_tolerance():
    r = tool_check_numeric_answer({
        "predicted": 142.0001, "ground_truth": 142, "tolerance": 1e-3,
    })
    assert r["match"] is True


def test_check_numeric_answer_outside_tolerance():
    r = tool_check_numeric_answer({
        "predicted": 141.5, "ground_truth": 142, "tolerance": 1e-3,
    })
    assert r["match"] is False


def test_check_numeric_answer_string_inputs_coerce():
    r = tool_check_numeric_answer({"predicted": "7.5", "ground_truth": "7.5"})
    assert r["match"] is True


def test_check_numeric_answer_rejects_non_numeric():
    r = tool_check_numeric_answer({"predicted": "abc", "ground_truth": "5"})
    assert "error" in r
