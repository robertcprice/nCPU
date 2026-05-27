"""Artifact integrity regressions.

Each publishable claim has a committed artifact under artifacts/. These tests
verify that every artifact parses, has the expected shape, and is internally
consistent (row counts match summary totals). They catch silent corruption
introduced by repo reorganizations, merge conflicts, or accidental overwrites.

The content-level regressions live in the per-benchmark test files
(test_nsynth_coverage.py, test_mog_diff_compiler.py). These tests are
structural only and run in milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT_ROOT / "artifacts"


def test_mog_synthesis_artifact_structural_integrity():
    path = ARTIFACTS / "mog_synthesis_coverage.json"
    if not path.exists():
        pytest.skip(f"artifact missing: {path}")
    data = json.loads(path.read_text())
    assert set(data.keys()) >= {"summary", "rows"}
    summary = data["summary"]
    rows = data["rows"]
    required_summary_keys = {
        "problem_count", "passed", "coverage", "wall_seconds",
        "variants_per_factory", "factory_count", "method_counts",
        "per_factory", "failing_factories", "timing",
    }
    missing = required_summary_keys - set(summary.keys())
    assert not missing, f"summary missing keys: {sorted(missing)}"
    assert len(rows) == summary["problem_count"], (
        f"row count {len(rows)} != summary.problem_count {summary['problem_count']}"
    )
    method_total = sum(summary["method_counts"].values())
    assert method_total == summary["passed"], (
        f"method counts sum to {method_total}, summary.passed={summary['passed']}"
    )
    # Coverage = passed / problem_count, up to rounding.
    expected = summary["passed"] / summary["problem_count"]
    assert abs(summary["coverage"] - expected) < 1e-4, (
        f"coverage {summary['coverage']} != passed/total {expected:.4f}"
    )


def test_nsynth_per_problem_coverage_structural_integrity():
    jsonl = ARTIFACTS / "nsynth_per_problem_coverage.jsonl"
    summary_path = ARTIFACTS / "nsynth_per_problem_summary.json"
    if not (jsonl.exists() and summary_path.exists()):
        pytest.skip(f"nsynth artifacts missing: {jsonl}, {summary_path}")
    summary = json.loads(summary_path.read_text())
    assert set(summary.keys()) >= {
        "summary", "problem_count", "passed", "coverage",
        "method_counts", "failures", "wall_seconds",
    }
    # Row count from JSONL must agree with summary.problem_count.
    row_count = 0
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        obj = json.loads(line)
        if obj.get("summary"):
            continue
        assert {"name", "method", "seconds"} <= set(obj.keys()), (
            f"row missing required keys: {obj}"
        )
        row_count += 1
    assert row_count == summary["problem_count"], (
        f"jsonl row count {row_count} != summary.problem_count {summary['problem_count']}"
    )
    # method_counts sum to passed.
    assert sum(summary["method_counts"].values()) == summary["passed"]


def test_nsynth_live_coverage_structural_integrity():
    path = ARTIFACTS / "nsynth_coverage.json"
    if not path.exists():
        pytest.skip(f"artifact missing: {path}")
    data = json.loads(path.read_text())
    assert set(data.keys()) >= {"summary", "rows"}
    summary = data["summary"]
    # Family counts must exist and sum to passed.
    assert "family_counts" in summary
    family_total = sum(summary["family_counts"].values())
    assert family_total == summary["passed"], (
        f"family counts sum to {family_total}, summary.passed={summary['passed']}"
    )
    # Timing block present.
    assert "timing" in summary


def test_superblock_promotion_benchmark_artifact_exists():
    path = ARTIFACTS / "superblock_promotion_benchmark.txt"
    if not path.exists():
        pytest.skip(f"artifact missing: {path}")
    text = path.read_text()
    # The benchmark output has a header row and at least one measured row.
    # The contract is that "threshold", "wall_s", "trace_skip", "tpl_hit"
    # all appear in the output.
    for keyword in ("threshold", "wall_s", "trace_skip", "tpl_hit"):
        assert keyword in text, f"expected marker '{keyword}' in benchmark output"


def test_nsynth_gradient_first_summary_integrity():
    """Pin the gradient-first coverage claim from paper/section_solver_portfolio.md.

    Measured: with `--prefer-differentiable` and 1200 s budget per problem,
    gradient-family methods solve 75/95 of the benchmark. Regressions that drop
    this below 70 indicate the gradient path has lost ground.
    """
    path = ARTIFACTS / "nsynth_gradient_first_summary.json"
    if not path.exists():
        pytest.skip(f"artifact missing: {path}")
    data = json.loads(path.read_text())
    assert data["total"] == 95, f"expected 95 total rows, got {data['total']}"
    families = data["family_counts"]
    assert families.get("gradient", 0) >= 70, (
        f"gradient-first coverage dropped: {families.get('gradient', 0)} (expected >= 70)"
    )
    # Family counts must sum to total.
    assert sum(families.values()) == data["total"], (
        f"family counts {families} don't sum to total {data['total']}"
    )
