"""Tests for the synthesis-API refusal WorkItem source."""

from __future__ import annotations

import json
from pathlib import Path

from ncpu.autoresearch.sources.synthesis_api import (
    mine_synthesis_refusals,
    work_item_from_refusal,
)


GOOD_REFUSAL = {
    "name": "junk_scalar",
    "examples": [
        {"inputs": [1], "expected": 7},
        {"inputs": [2, 3], "expected": 99},
        {"inputs": ["ab"], "expected": -1},
    ],
    "error": "no program found",
    "method": "timeout",
    "elapsed_ms": 3000.0,
    "ts": "2026-06-14T20:00:00+00:00",
}


UNREPRESENTABLE = {
    "name": "floats_only",
    "examples": [
        {"inputs": [1.5, 2.5], "expected": 4.0},  # float args, float expected
    ],
}


KWARG_REFUSAL = {
    "name": "kwargs_only",
    "examples": [
        {"inputs": [1], "kwargs": {"a": 2}, "expected": 3},
    ],
}


def test_work_item_from_refusal_succeeds_for_mixed_inputs():
    item = work_item_from_refusal(GOOD_REFUSAL)
    assert item is not None
    assert item.entry_point == "junk_scalar"
    assert item.source_benchmark == "synthesis_api"
    # All three examples are representable (int, [int,int], str -> int).
    assert len(item.io_pairs) == 3
    assert item.io_pairs[0].args == [1]
    assert item.io_pairs[1].args == [2, 3]
    assert item.io_pairs[2].args == ["ab"]
    assert item.provenance["synth_error"] == "no program found"


def test_work_item_from_refusal_skips_unrepresentable_examples():
    assert work_item_from_refusal(UNREPRESENTABLE) is None
    assert work_item_from_refusal(KWARG_REFUSAL) is None


def test_work_item_from_refusal_sanitizes_name():
    item = work_item_from_refusal({**GOOD_REFUSAL, "name": "junk-scalar!"})
    assert item is not None
    assert item.entry_point == "junk_scalar_"


def test_mine_synthesis_refusals_writes_work_items(tmp_path: Path):
    refusals = tmp_path / "refusals.jsonl"
    out = tmp_path / "queue.jsonl"
    refusals.write_text(
        json.dumps(GOOD_REFUSAL) + "\n"
        + json.dumps(UNREPRESENTABLE) + "\n"
        + "\n"
    )
    counters = mine_synthesis_refusals(refusals, out)
    assert counters == {"read": 2, "emitted": 1, "skipped": 1}
    payload = json.loads(out.read_text().strip().splitlines()[0])
    assert payload["source_benchmark"] == "synthesis_api"
    assert payload["entry_point"] == "junk_scalar"
    assert len(payload["io_pairs"]) == 3


def test_mine_synthesis_refusals_handles_missing_file(tmp_path: Path):
    counters = mine_synthesis_refusals(tmp_path / "absent.jsonl", tmp_path / "out.jsonl")
    assert counters == {"read": 0, "emitted": 0, "skipped": 0}
