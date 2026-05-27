from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import benchmark_full_neural_pipeline as bench
from demos.neural import full_neural_demo


def test_full_neural_workload_catalog_has_benchmark_suite():
    names = full_neural_demo.available_workloads()

    assert names == ["single-1k", "single-4k", "dual-1k", "triple-512", "staggered"]
    assert full_neural_demo.resolve_workload("dual-1k").loop_counts == (1000, 1000)
    assert full_neural_demo.resolve_workload("staggered").expected_total == 3328

    with pytest.raises(ValueError):
        full_neural_demo.resolve_workload("not-a-workload")


def test_build_payload_and_write_artifacts(tmp_path: Path):
    records = [
        {
            "workload": "single-1k",
            "workload_title": "Single Counter Loop (1K)",
            "loop_counts": [1000],
            "expected_total_iterations": 1000,
            "status": "completed",
            "executed_instructions": 3210,
            "throughput_ips": 1200.0,
            "execution_time_s": 2.675,
            "render_time_ms": 88.4,
            "counter_verified": True,
        },
        {
            "workload": "dual-1k",
            "workload_title": "Dual Counter Loop (1K + 1K)",
            "loop_counts": [1000, 1000],
            "expected_total_iterations": 2000,
            "status": "failed",
            "error_message": "demo exited with return code 1",
        },
    ]

    payload = bench.build_payload(records, requested_workloads=["single-1k", "dual-1k"])
    paths = bench.write_artifacts(tmp_path, payload)

    data = json.loads(paths["json"].read_text())
    markdown = paths["md"].read_text()

    assert data["benchmark"] == "full_neural_pipeline"
    assert data["summary"]["requested"] == 2
    assert data["summary"]["completed"] == 1
    assert data["summary"]["failed"] == 1
    assert data["summary"]["all_counters_verified"] is False
    assert data["summary"]["avg_throughput_ips"] == pytest.approx(1200.0)
    assert data["summary"]["total_executed_instructions"] == 3210
    assert "## Workloads" in markdown
    assert "| single-1k | 1000 | 1,000 | 3,210 | 1,200 | 2.675 | 88.4 | OK | completed |" in markdown
    assert "Note: demo exited with return code 1" in markdown
