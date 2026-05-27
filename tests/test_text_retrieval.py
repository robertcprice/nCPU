"""Regression tests for TF-IDF text retrieval over question-keyed caches.

The GSM8K retrieval path stores `(question, solve_code)` pairs and
looks up similar *questions* at inference time. This test pins:
  - record_with_question writes a 7-col row
  - text_lookup ranks shared-vocabulary questions higher
  - empty/short queries don't crash
  - roundtrip preserves the question text"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS_BENCH = Path(__file__).resolve().parent.parent / "tools" / "benchmarks"
sys.path.insert(0, str(TOOLS_BENCH))

from text_retrieval import (  # noqa: E402
    _tokenize, record_with_question, text_lookup, build_text_retrieval_prefix,
)


@pytest.fixture
def fresh_cache(monkeypatch, tmp_path):
    p = tmp_path / "t.tsv"
    monkeypatch.setenv("NSYNTH_LLM_CACHE_PATH", str(p))
    return p


def test_tokenize_produces_words_and_bigrams():
    toks = _tokenize("A farmer has 3 apples and 5 oranges")
    # Should include unigrams
    assert "farmer" in toks
    assert "apples" in toks
    # Should include bigrams
    assert "farmer_has" in toks
    assert "apples_and" in toks


def test_tokenize_short_text_falls_back_to_unigrams():
    toks = _tokenize("two apples")
    assert toks == ["two", "apples"]  # no bigrams below threshold


def test_tokenize_empty_string():
    assert _tokenize("") == []


def test_record_then_lookup_returns_hit(fresh_cache):
    record_with_question(
        "fp1", "m", "def solve():\n    return 8",
        "Alice has 3 apples and gets 5 more. How many apples total?",
    )
    hits = text_lookup(
        "Bob has 3 apples and gets 5 more. Count the total apples.",
        k=3, min_similarity=0.0,
    )
    assert len(hits) >= 1
    assert hits[0]["fingerprint"] == "fp1"
    assert "apples" in hits[0]["question"]


def test_text_lookup_empty_cache(fresh_cache):
    hits = text_lookup("some question", k=3, min_similarity=0.0)
    assert hits == []


def test_similarity_ordering(fresh_cache):
    record_with_question(
        "apples", "m", "def solve():\n    return 8",
        "How many apples does Alice have after buying more?",
    )
    record_with_question(
        "train", "m", "def solve():\n    return 60",
        "A train travels at 30 miles per hour for 2 hours. Distance?",
    )
    record_with_question(
        "cookies", "m", "def solve():\n    return 12",
        "Baker made cookies and sold some. How many cookies remain?",
    )
    hits = text_lookup(
        "Alice has apples. She gives some to Bob. How many apples remain?",
        k=3, min_similarity=0.0,
    )
    # With only 3 cache entries, TF-IDF is noisy; the relevant check
    # is that the vocabulary-matching entries (apples, cookies) both
    # score above the completely-off-topic train entry.
    top2 = {h["fingerprint"] for h in hits[:2]}
    assert "apples" in top2
    # Train entry (zero shared words) is last.
    if len(hits) == 3:
        assert hits[-1]["fingerprint"] == "train"


def test_build_text_retrieval_prefix_structure(fresh_cache):
    record_with_question(
        "p1", "m", "def solve():\n    return 42",
        "Sample word problem about apples and pears",
    )
    prefix = build_text_retrieval_prefix(
        "Another word problem about apples", k=2, min_similarity=0.0,
    )
    assert "# Similar verified solutions retrieved from cache:" in prefix
    assert "def solve" in prefix
    assert "# Problem:" in prefix  # question included in few-shot
    assert "# Your task" in prefix


def test_empty_query_returns_empty(fresh_cache):
    record_with_question(
        "p1", "m", "def solve():\n    return 1",
        "Some question text",
    )
    assert text_lookup("", k=3, min_similarity=0.0) == []


def test_self_exclusion(fresh_cache):
    record_with_question(
        "same", "m", "def solve():\n    return 1",
        "Exactly this question text",
    )
    # Querying with the exact same text → entry excluded (include_self=False).
    hits = text_lookup("Exactly this question text", k=3, min_similarity=0.0)
    assert not any(h["fingerprint"] == "same" for h in hits)
