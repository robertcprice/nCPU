"""Tests for the registry-miss WorkItem source."""

from __future__ import annotations

import json
from pathlib import Path

from ncpu.autoresearch.sources.registry import (
    mine_registry_misses,
    work_item_from_miss,
)


SUM_MISS = {
    "name": "sum",
    "author": "alice",
    "examples": [
        {"data": [1.0, 2.0, 3.0], "n_points": 3, "target": 6.0},
        {"data": [5.0], "n_points": 1, "target": 5.0},
    ],
    "error": "verification failed: max_err=2.0",
    "first_failure": {"example_index": 0, "expected": 6.0, "got": 4.0},
}


V3_TRACE_MISS = {
    "name": "counter",
    "author": "dana",
    "examples": [
        {"data": [5.0, -2.0, 0.0, 9.0], "n_points": 4,
         "targets": [1.0, 2.0, 3.0, 4.0]},
    ],
}


def test_work_item_from_miss_succeeds_for_scalar_target():
    item = work_item_from_miss(SUM_MISS)
    assert item is not None
    assert item.entry_point == "sum"
    assert item.source_benchmark == "registry"
    assert len(item.io_pairs) == 2
    assert item.io_pairs[0].args == [[1.0, 2.0, 3.0]]
    assert item.io_pairs[0].expected == 6
    assert "def check(candidate):" in item.test_source
    assert "candidate([1.0, 2.0, 3.0])" in item.test_source
    assert item.provenance["registry_author"] == "alice"


def test_work_item_from_miss_skips_v3_trace_examples():
    assert work_item_from_miss(V3_TRACE_MISS) is None


def test_work_item_from_miss_rejects_missing_examples():
    assert work_item_from_miss({"name": "x"}) is None


def test_work_item_from_miss_sanitizes_name():
    item = work_item_from_miss({**SUM_MISS, "name": "sum-thing?"})
    assert item is not None
    assert item.entry_point == "sum_thing_"


def test_mine_registry_misses_writes_work_items(tmp_path: Path):
    misses = tmp_path / "misses.jsonl"
    out = tmp_path / "queue.jsonl"
    misses.write_text(
        json.dumps(SUM_MISS) + "\n" + json.dumps(V3_TRACE_MISS) + "\n" + "\n"
    )
    counters = mine_registry_misses(misses, out)
    assert counters == {"read": 2, "emitted": 1, "skipped": 1}
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["entry_point"] == "sum"
    assert payload["source_benchmark"] == "registry"
    assert len(payload["io_pairs"]) == 2


def test_mine_registry_misses_handles_missing_file(tmp_path: Path):
    counters = mine_registry_misses(tmp_path / "absent.jsonl", tmp_path / "out.jsonl")
    assert counters == {"read": 0, "emitted": 0, "skipped": 0}
