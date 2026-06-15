"""Tests for the autoresearch CLI subcommands, including the registry source."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ncpu.autoresearch.cli", *args],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_mine_registry_subcommand_emits_work_items(tmp_path: Path):
    misses = tmp_path / "misses.jsonl"
    misses.write_text(
        json.dumps({
            "name": "sum",
            "author": "alice",
            "examples": [
                {"data": [1.0, 2.0, 3.0], "n_points": 3, "target": 6.0},
            ],
        })
        + "\n"
    )
    result = _run_cli(
        "--artifact-dir", str(tmp_path), "mine-registry", "--misses", str(misses),
    )
    assert result.returncode == 0, result.stderr
    counters = json.loads(result.stdout.strip())
    assert counters == {"read": 1, "emitted": 1, "skipped": 0}
    queue = tmp_path / "registry_queue.jsonl"
    assert queue.exists()
    line = queue.read_text().strip().splitlines()[0]
    payload = json.loads(line)
    assert payload["entry_point"] == "sum"
    assert payload["source_benchmark"] == "registry"
    assert len(payload["io_pairs"]) == 1


def test_mine_registry_skips_unrepresentable_misses(tmp_path: Path):
    misses = tmp_path / "misses.jsonl"
    misses.write_text(
        json.dumps({
            "name": "counter",
            "examples": [
                {"data": [1.0, 2.0, 3.0], "n_points": 3,
                 "targets": [1.0, 2.0, 3.0]},
            ],
        })
        + "\n"
    )
    result = _run_cli(
        "--artifact-dir", str(tmp_path), "mine-registry", "--misses", str(misses),
    )
    assert result.returncode == 0, result.stderr
    counters = json.loads(result.stdout.strip())
    assert counters == {"read": 1, "emitted": 0, "skipped": 1}
    assert not (tmp_path / "registry_queue.jsonl").exists()


def test_run_once_supports_registry_benchmark(tmp_path: Path):
    # Empty queue: run-once prints a friendly error and exits 2.
    result = _run_cli(
        "--artifact-dir", str(tmp_path), "run-once", "--benchmark", "registry",
    )
    assert result.returncode == 2
    assert "queue" in result.stderr
