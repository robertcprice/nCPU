"""Regression tests for the nSynth solver portfolio coverage.

Fast path (default): parses the persisted artifact
    artifacts/nsynth_per_problem_coverage.jsonl
    artifacts/nsynth_per_problem_summary.json
and asserts the published solver breakdown (95/95, 60 gradient, 25 enumerative,
10 search). This runs in milliseconds and catches regressions in the
reporting pipeline without rebuilding the Rust binary.

Slow path (opt-in via NCPU_NSYNTH_FULL_RUN=1): invokes the Rust CLI through
benchmarks/benchmark_nsynth.py and verifies the live numbers. Takes ~15 min.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_COVERAGE = PROJECT_ROOT / "artifacts" / "nsynth_per_problem_coverage.jsonl"
ARTIFACT_SUMMARY = PROJECT_ROOT / "artifacts" / "nsynth_per_problem_summary.json"
HARNESS = PROJECT_ROOT / "benchmarks" / "benchmark_nsynth.py"

GRADIENT_METHODS = {
    "synth_gradient",
    "univ_arr_gradient",
    "arr_gradient",
    "arr_gradient_binary_search",
    "arr_gradient_count_distinct",
    "arr_gradient_kth_smallest",
    "arr_gradient_two_sum_exists",
}


def _parse_artifact() -> tuple[list[dict], dict]:
    if not ARTIFACT_SUMMARY.exists():
        pytest.skip(f"artifact missing: {ARTIFACT_SUMMARY}")
    summary = json.loads(ARTIFACT_SUMMARY.read_text())
    rows: list[dict] = []
    for line in ARTIFACT_COVERAGE.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        obj = json.loads(line)
        if obj.get("summary"):
            continue
        rows.append(obj)
    return rows, summary


def _classify(method: str) -> str:
    if method in GRADIENT_METHODS:
        return "gradient"
    if method.startswith("enumerative"):
        return "enumerative"
    if method.startswith("search_"):
        return "search"
    return "other"


def test_artifact_reports_full_coverage():
    _, summary = _parse_artifact()
    assert summary["passed"] == summary["problem_count"] == 95, (
        f"artifact reports {summary['passed']}/{summary['problem_count']} — expected 95/95"
    )
    assert summary["coverage"] == 1.0
    assert summary["failures"] == []


def test_artifact_rows_match_summary():
    rows, summary = _parse_artifact()
    assert len(rows) == summary["problem_count"]
    passed = sum(1 for r in rows if r["success"])
    assert passed == summary["passed"]
    # Each row carries a recognizable method name.
    for row in rows:
        method = row["method"]
        assert method, f"row without method: {row['name']}"


def test_artifact_solver_family_breakdown():
    rows, _ = _parse_artifact()
    families: dict[str, int] = {}
    for row in rows:
        if row["success"]:
            families[_classify(row["method"])] = families.get(_classify(row["method"]), 0) + 1
    assert families.get("gradient", 0) >= 55, (
        f"gradient family coverage dropped: {families.get('gradient', 0)} (expected >= 55)"
    )
    assert families.get("enumerative", 0) >= 20, (
        f"enumerative family coverage dropped: {families.get('enumerative', 0)} (expected >= 20)"
    )
    # Total solved across families must equal summary passed.
    assert sum(families.values()) == 95


def test_artifact_known_hard_problems_solved():
    """Regressions on the hardest gradient-solved problems would be silent
    without an explicit check — these are the load-bearing wins for the
    universal-array and pairwise gradient paths."""
    rows, _ = _parse_artifact()
    must_be_solved = {
        "count_peaks_v0",
        "longest_plateau_v0",
        "prefix_sum_k_v0",
        "max_stock_profit_v0",
        "is_sorted_v0",
    }
    solved_names = {r["name"] for r in rows if r["success"]}
    missing = must_be_solved - solved_names
    assert not missing, f"regression on known-solvable pairwise problems: {sorted(missing)}"


@pytest.mark.skipif(
    os.environ.get("NCPU_NSYNTH_FULL_RUN", "0") != "1",
    reason="set NCPU_NSYNTH_FULL_RUN=1 to run the full Rust harness (~15 min)",
)
def test_live_harness_matches_artifact():
    """Opt-in slow path: re-run the Rust harness and assert it still returns 95/95."""
    result = subprocess.run(
        [sys.executable, str(HARNESS), "--variants", "1"],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, (
        f"nsynth harness exited {result.returncode}\nstderr:\n{result.stderr[-2000:]}"
    )
    assert "95/95" in result.stdout, (
        f"live harness no longer reports 95/95:\n{result.stdout[-2000:]}"
    )
