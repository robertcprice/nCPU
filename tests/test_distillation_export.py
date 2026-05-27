"""Regression tests for distillation dataset export quality.

These tests pin the two failure modes that make a coding adapter go bad:
  - exporting non-code cache rows (for example GSM8K reasoning traces)
  - throwing away the original coding prompt when richer prompt metadata
    or benchmark recovery is available
"""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from export_distillation_dataset import _build_export_record  # noqa: E402


def test_export_skips_noncode_rows_by_default():
    row = {
        "model": "claude-haiku",
        "success_count": 1,
        "last_used_at": 1,
        "code": "Let's reason it out step by step.\n#### 42",
        "examples": [],
        "metadata": {},
    }
    record = _build_export_record(
        "fp",
        row,
        fmt="hf",
        prompt_catalog={},
        include_noncode=False,
    )
    assert record is None


def test_export_prefers_cached_prompt_metadata():
    row = {
        "model": "claude-haiku",
        "success_count": 3,
        "last_used_at": 10,
        "code": "def add(a, b):\n    return a + b",
        "examples": [{"inputs": [1, 2], "expected": 3}],
        "metadata": {
            "task_kind": "mbpp",
            "prompt": "Write a Python function matching these assertions.",
            "task_id": 17,
        },
    }
    record = _build_export_record(
        "fp",
        row,
        fmt="hf",
        prompt_catalog={},
        include_noncode=False,
    )
    assert record is not None
    assert record["prompt"] == "Write a Python function matching these assertions."
    assert record["metadata"]["task_kind"] == "mbpp"
    assert record["metadata"]["prompt_source"] == "cache_metadata"


def test_export_uses_benchmark_recovery_when_cache_lacks_prompt():
    row = {
        "model": "claude-haiku",
        "success_count": 2,
        "last_used_at": 11,
        "code": "def add(a, b):\n    return a + b",
        "examples": [{"inputs": [1, 2], "expected": 3}],
        "metadata": {},
    }
    record = _build_export_record(
        "fp",
        row,
        fmt="hf",
        prompt_catalog={
            "fp": {
                "prompt": "Recovered benchmark prompt",
                "metadata": {
                    "task_kind": "humaneval",
                    "task_id": "HumanEval/0",
                    "prompt_source": "benchmark_recovery",
                },
            }
        },
        include_noncode=False,
    )
    assert record is not None
    assert record["prompt"] == "Recovered benchmark prompt"
    assert record["metadata"]["task_kind"] == "humaneval"
    assert record["metadata"]["prompt_source"] == "benchmark_recovery"
