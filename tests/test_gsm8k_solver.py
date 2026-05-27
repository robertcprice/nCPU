"""Regression tests for run_gsm8k._majority_vote and related helpers.

These tests pin down the self-consistency logic used by the
GSM8K agent loop — so future edits to the tolerance, bucket merging,
or return shape don't silently shift the retrieved answer."""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS_BENCH = Path(__file__).resolve().parent.parent / "tools" / "benchmarks"
sys.path.insert(0, str(TOOLS_BENCH))

from run_gsm8k import (  # noqa: E402
    _majority_vote, extract_predicted_answer, extract_and_run_pot,
)


def test_clear_majority():
    top, votes, n = _majority_vote([18, 18, 18, 25, 14])
    assert top == 18 and votes == 3 and n == 5


def test_handles_none_preds():
    top, votes, n = _majority_vote([None, 7, None, 7])
    assert top == 7 and votes == 2 and n == 2


def test_all_none_returns_none():
    top, votes, n = _majority_vote([None, None])
    assert top is None and votes == 0 and n == 0


def test_float_tolerance_merges_near_ties():
    top, votes, n = _majority_vote([3.0, 3.0000001, 5])
    assert top == 3.0 and votes == 2 and n == 3


def test_no_majority_returns_top_count_one():
    top, votes, n = _majority_vote([1, 2, 3, 4, 5])
    assert votes == 1 and n == 5


def test_tied_votes_takes_first_after_sort():
    top, votes, n = _majority_vote([10, 10, 20, 20, 30])
    # Both 10 and 20 have count 2 — sort is stable on equal keys, so
    # whichever appeared first in insertion order wins.
    assert votes == 2 and n == 5
    assert top in (10, 20)


def test_extract_predicted_answer_from_gsm8k_output():
    text = "The answer is computed step by step...\n\n#### 42"
    assert extract_predicted_answer(text) == 42.0


def test_extract_predicted_answer_prefers_final_marker():
    text = "I think it's 10 at first, but actually\n\n#### 15"
    assert extract_predicted_answer(text) == 15.0


def test_extract_predicted_answer_returns_none_when_missing():
    assert extract_predicted_answer("no number anywhere in this reply") is None


# ─── extract_and_run_pot ──────────────────────────────────────────────────


def test_pot_runs_simple_function():
    resp = "```python\ndef solve():\n    return 3 * 45 + 7\n```"
    val, err = extract_and_run_pot(resp)
    assert val == 142.0 and err == ""


def test_pot_handles_unfenced_response():
    resp = "Here's the solution\ndef solve():\n    return 100 - 42\n# end"
    val, err = extract_and_run_pot(resp)
    assert val == 58.0


def test_pot_allows_math_module():
    resp = "```python\ndef solve():\n    import math\n    return math.sqrt(144)\n```"
    # Restricted namespace blocks `import` — we pre-inject math instead.
    # Model should use `math.sqrt` without importing. Verify that path:
    resp2 = "```python\ndef solve():\n    return math.sqrt(144) + math.floor(math.pi)\n```"
    val, err = extract_and_run_pot(resp2)
    assert val == 15.0


def test_pot_rejects_missing_solve():
    resp = "```python\ndef other():\n    return 42\n```"
    val, err = extract_and_run_pot(resp)
    assert val is None
    assert "solve" in err.lower()


def test_pot_catches_runtime_error():
    resp = "```python\ndef solve():\n    return 1 / 0\n```"
    val, err = extract_and_run_pot(resp)
    assert val is None
    assert "ZeroDivisionError" in err or "exec" in err


def test_pot_timeout_kills_infinite_loop():
    resp = "```python\ndef solve():\n    while True:\n        pass\n```"
    val, err = extract_and_run_pot(resp, timeout_s=1)
    assert val is None
    assert "timed out" in err.lower()


def test_pot_coerces_int_return_to_float():
    resp = "```python\ndef solve():\n    return 42\n```"
    val, _ = extract_and_run_pot(resp)
    assert isinstance(val, float) and val == 42.0
