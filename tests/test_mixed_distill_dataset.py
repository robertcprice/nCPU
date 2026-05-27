"""Regression tests for the mixed distillation dataset builder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS_DISTILL = Path(__file__).resolve().parent.parent / "tools" / "distillation"
sys.path.insert(0, str(TOOLS_DISTILL))

from build_mixed_codegen_dataset import build_dataset  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_builder_caps_utility_share(tmp_path: Path):
    coding = tmp_path / "coding.jsonl"
    utility = tmp_path / "utility.jsonl"
    _write_jsonl(coding, [
        {"prompt": f"Write foo_{i}", "completion": f"def foo_{i}():\n    return {i}"}
        for i in range(10)
    ])
    _write_jsonl(utility, [
        {"prompt": f"Reimplement wc #{i}", "completion": f"def solve(stdin: str) -> str:\n    return '{i}'"}
        for i in range(100)
    ])

    splits = build_dataset(
        coding_jsonl=[coding],
        utility_jsonl=[utility],
        coding_split_dirs=[],
        utility_split_dirs=[],
        max_utility_share=0.25,
        valid_fraction=0.1,
        test_fraction=0.1,
        seed=0,
    )
    total = sum(len(v) for v in splits.values())
    utility_total = sum(
        1 for rows in splits.values() for row in rows
        if row["metadata"]["source_group"] == "utility"
    )
    coding_total = total - utility_total
    assert coding_total == 10
    assert utility_total <= 3


def test_builder_drops_noncode_rows(tmp_path: Path):
    coding = tmp_path / "coding.jsonl"
    _write_jsonl(coding, [
        {"prompt": "Good", "completion": "def good():\n    return 1"},
        {"prompt": "Bad", "completion": "Reasoning only\n#### 7"},
    ])
    splits = build_dataset(
        coding_jsonl=[coding],
        utility_jsonl=[],
        coding_split_dirs=[],
        utility_split_dirs=[],
        max_utility_share=0.4,
        valid_fraction=0.1,
        test_fraction=0.1,
        seed=0,
    )
    total = sum(len(v) for v in splits.values())
    assert total == 1


def test_builder_reads_split_dirs(tmp_path: Path):
    utility_dir = tmp_path / "utility"
    _write_jsonl(utility_dir / "train.jsonl", [
        {"prompt": "Reimplement head", "completion": "def solve(stdin: str) -> str:\n    return stdin"}
    ])
    _write_jsonl(utility_dir / "valid.jsonl", [
        {"prompt": "Reimplement tail", "completion": "def solve(stdin: str) -> str:\n    return stdin"}
    ])
    splits = build_dataset(
        coding_jsonl=[],
        utility_jsonl=[],
        coding_split_dirs=[],
        utility_split_dirs=[utility_dir],
        max_utility_share=0.4,
        valid_fraction=0.1,
        test_fraction=0.1,
        seed=0,
    )
    total = sum(len(v) for v in splits.values())
    assert total == 2
