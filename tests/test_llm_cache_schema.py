"""Regression tests for the LLM-solution cache schema extension.

The cache grew a 6th column (examples JSON) in the retrieval-augmented
generation work. This test pins down:
  - legacy 5-col rows still load cleanly (read-through compat)
  - new writes produce 6-col rows with examples attached
  - semantic retrieval uses stored examples when present
  - missing examples falls back to code-shape features without crashing

These invariants protect the deployed TSV caches (a handful of rows
live on users' machines) — a schema change that broke old rows would
silently discard cached solutions.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

TOOLS_BENCH = Path(__file__).resolve().parent.parent / "tools" / "benchmarks"
sys.path.insert(0, str(TOOLS_BENCH))

from llm_solution_cache import _load_all, lookup, record  # noqa: E402
from semantic_cache import semantic_lookup  # noqa: E402


@pytest.fixture
def temp_cache(monkeypatch, tmp_path):
    path = tmp_path / "cache.tsv"
    monkeypatch.setenv("NSYNTH_LLM_CACHE_PATH", str(path))
    return path


def test_loads_legacy_5col_rows(temp_cache):
    """Rows written by the pre-extension cache are still readable."""
    temp_cache.write_text(
        "abc\tclaude-haiku\t1\t1700000000\tdef f(x):\\n    return x+1\n"
    )
    got = lookup("abc")
    assert got is not None
    assert got["model"] == "claude-haiku"
    assert got["code"] == "def f(x):\n    return x+1"
    assert got["examples"] == []


def test_writes_6col_rows_with_examples(temp_cache):
    examples = [{"inputs": [1, 2], "expected": 3}]
    record("fp1", "m", "def g(a,b):\n    return a+b", examples=examples)
    txt = temp_cache.read_text()
    cols = txt.strip().split("\t")
    assert len(cols) == 6, f"expected 6 columns, got {len(cols)}: {txt!r}"
    # Sixth column is JSON of the examples.
    decoded = json.loads(cols[5].replace("\\n", "\n").replace("\\t", "\t"))
    assert decoded == examples


def test_roundtrip_preserves_examples(temp_cache):
    examples = [
        {"inputs": [3, 4], "expected": 7},
        {"inputs": [-1, 1], "expected": 0},
    ]
    record("fp2", "m", "code", examples=examples)
    got = lookup("fp2")
    assert got["examples"] == examples


def test_record_without_examples_persists_empty_list(temp_cache):
    record("fp3", "m", "code")  # no examples arg
    got = lookup("fp3")
    assert got["examples"] == []


def test_record_preserves_existing_examples_on_update(temp_cache):
    """Updating an existing entry without passing examples shouldn't
    wipe the previously-stored examples. This is the 'incrementing
    success_count' code path when the same (fp, model, code) re-solves."""
    examples = [{"inputs": [5], "expected": 25}]
    record("fp4", "m", "def sq(x):\n    return x*x", examples=examples)
    record("fp4", "m", "def sq(x):\n    return x*x")  # no examples
    got = lookup("fp4")
    assert got["examples"] == examples
    assert got["success_count"] == 2


def test_semantic_lookup_uses_stored_examples(temp_cache):
    """Two cached rows with very different examples should yield very
    different similarity scores to a query — proving retrieval uses
    the persisted examples, not code shape."""
    record(
        "row_sum", "m", "def add(a,b):\n    return a+b",
        examples=[{"inputs": [1, 2], "expected": 3},
                  {"inputs": [4, 5], "expected": 9}],
    )
    record(
        "row_big", "m", "def big(a,b):\n    return a*1000+b",
        examples=[{"inputs": [1, 2], "expected": 1002},
                  {"inputs": [3, 4], "expected": 3004}],
    )
    # Query is a small-sum shape; should match row_sum better.
    q = [{"inputs": [2, 3], "expected": 5}, {"inputs": [10, 4], "expected": 14}]
    hits = semantic_lookup(q, k=2, min_similarity=0.0)
    assert len(hits) == 2
    sum_sim = next(h["similarity"] for h in hits if h["fingerprint"] == "row_sum")
    big_sim = next(h["similarity"] for h in hits if h["fingerprint"] == "row_big")
    assert sum_sim > big_sim, f"sum={sum_sim} big={big_sim}"


def test_mixed_5col_and_6col_in_same_file(temp_cache):
    """A cache file with a mix of old and new rows should load every
    row. The legacy row ends up with examples=[] and is retrievable
    via exact-fingerprint lookup but loses semantic-retrieval coverage
    until re-recorded with examples."""
    temp_cache.write_text(
        "legacy\tm\t1\t100\tdef f(x):\\n    return x\n"
    )
    record("new", "m", "def g(x):\n    return x+1",
           examples=[{"inputs": [1], "expected": 2}])
    all_rows = _load_all()
    assert "legacy" in all_rows and "new" in all_rows
    assert all_rows["legacy"]["examples"] == []
    assert all_rows["new"]["examples"] == [{"inputs": [1], "expected": 2}]


def test_roundtrip_preserves_question_and_metadata(temp_cache):
    record(
        "fp5",
        "m",
        "def f(x):\n    return x",
        question="How many apples are left?",
        metadata={
            "task_kind": "humaneval",
            "task_id": "HumanEval/0",
            "prompt": "Complete the function.",
        },
    )
    got = lookup("fp5")
    assert got["question"] == "How many apples are left?"
    assert got["metadata"]["task_kind"] == "humaneval"
    assert got["metadata"]["task_id"] == "HumanEval/0"
    txt = temp_cache.read_text()
    cols = txt.strip().split("\t")
    assert len(cols) == 8, f"expected 8 columns, got {len(cols)}: {txt!r}"


def test_loads_legacy_7col_question_rows(temp_cache):
    row = "\t".join([
        "fp6",
        "m",
        "1",
        "1700000000",
        "def%20unused".replace("%20", " "),
        "",
        "word_problem_text",
    ])
    temp_cache.write_text(row + "\n")
    got = lookup("fp6")
    assert got is not None
    assert got["question"] == "word_problem_text"
    assert got["metadata"] == {}
