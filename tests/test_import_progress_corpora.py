"""Regression tests for benchmark progress-log import."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS_DISTILL = Path(__file__).resolve().parent.parent / "tools" / "distillation"
sys.path.insert(0, str(TOOLS_DISTILL))

from import_progress_corpora import import_progress_files  # noqa: E402


def test_import_progress_keeps_fastest_success_and_strips_fences(tmp_path: Path):
    path = tmp_path / "progress.jsonl"
    rows = [
        {
            "event": "task_complete",
            "success": True,
            "task_name": "Task/1",
            "model": "a",
            "dataset": "humaneval",
            "approach": "standard",
            "task_result": {
                "attempt_details": [{
                    "prompt": "Write code",
                    "response_text": "```python\ndef foo():\n    return 1\n```",
                    "elapsed_seconds": 3.0,
                    "verification": {"passed": True},
                }]
            },
        },
        {
            "event": "task_complete",
            "success": True,
            "task_name": "Task/1",
            "model": "b",
            "dataset": "humaneval",
            "approach": "standard",
            "task_result": {
                "attempt_details": [{
                    "prompt": "Write code",
                    "response_text": "def foo():\n    return 2",
                    "elapsed_seconds": 7.0,
                    "verification": {"passed": True},
                }]
            },
        },
    ]
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    imported, stats = import_progress_files([path])
    assert len(imported) == 1
    assert imported[0]["completion"] == "def foo():\n    return 1"
    assert imported[0]["metadata"]["success_variants"] == 2
    assert imported[0]["metadata"]["source_model"] == "a"
    assert stats["rows_kept"] == 1


def test_import_progress_reads_summary_json(tmp_path: Path):
    path = tmp_path / "summary.json"
    payload = {
        "model": "qwen3.5:4b",
        "dataset": "humaneval",
        "baseline": {
            "task_results": [
                {
                    "name": "HumanEval/0",
                    "category": "humaneval",
                    "success": True,
                    "attempt_details": [
                        {
                            "prompt": "Write code",
                            "response_text": "```python\ndef foo():\n    return 7\n```",
                            "elapsed_seconds": 2.5,
                            "verification": {"passed": True},
                        }
                    ],
                }
            ]
        },
    }
    path.write_text(json.dumps(payload))

    imported, stats = import_progress_files([], [path])
    assert len(imported) == 1
    assert imported[0]["completion"] == "def foo():\n    return 7"
    assert imported[0]["metadata"]["task_name"] == "HumanEval/0"
    assert imported[0]["metadata"]["approach"] == "baseline"
    assert imported[0]["metadata"]["source_model"] == "qwen3.5:4b"
    assert stats["rows_kept"] == 1
