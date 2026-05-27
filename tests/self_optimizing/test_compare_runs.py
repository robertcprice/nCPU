"""Tests for compare_runs (multi-model sweep aggregator)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ncpu.self_optimizing.compare_runs import (
    collect_reports,
    render_markdown,
    render_table,
    summarize,
)


def _write_report(
    path: Path, *, model: str, pass_at_1: float, total_problems: int, use_npcot: bool
) -> None:
    payload = {
        "mode": "humaneval_real_run",
        "config": {
            "model": model,
            "use_npcot": use_npcot,
            "library_path": "lib.json" if use_npcot else None,
        },
        "results": {
            "pass_at_1": pass_at_1,
            "pass_count": int(round(pass_at_1 * total_problems)),
            "total_problems": total_problems,
            "total_seconds": 120.0,
        },
    }
    path.write_text(json.dumps(payload))


class TestCollectReports(unittest.TestCase):
    def test_collects_only_matching_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_report(d / "humaneval_a.json", model="A", pass_at_1=0.5, total_problems=100, use_npcot=False)
            _write_report(d / "humaneval_b.json", model="B", pass_at_1=0.6, total_problems=100, use_npcot=False)
            _write_report(d / "mbpp_c.json", model="C", pass_at_1=0.7, total_problems=100, use_npcot=False)
            reports = collect_reports(d, "humaneval")
            self.assertEqual(len(reports), 2)

    def test_skips_non_result_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "humaneval_noisy.json").write_text('{"no_results_key": true}')
            self.assertEqual(collect_reports(d, "humaneval"), [])


class TestSummarize(unittest.TestCase):
    def test_summarize_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_report(d / "humaneval_x.json", model="Qwen2.5-1.5B", pass_at_1=0.42, total_problems=50, use_npcot=False)
            rows = summarize(collect_reports(d, "humaneval"))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["model"], "Qwen2.5-1.5B")
            self.assertFalse(row["use_npcot"])
            self.assertEqual(row["problems"], 50)
            self.assertAlmostEqual(row["pass_at_1"], 0.42)

    def test_baseline_before_npcot_within_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_report(d / "humaneval_a_npcot.json", model="A", pass_at_1=0.5, total_problems=10, use_npcot=True)
            _write_report(d / "humaneval_a_base.json", model="A", pass_at_1=0.4, total_problems=10, use_npcot=False)
            rows = summarize(collect_reports(d, "humaneval"))
            # Baseline should sort first when model is the same.
            self.assertFalse(rows[0]["use_npcot"])
            self.assertTrue(rows[1]["use_npcot"])


class TestRender(unittest.TestCase):
    def test_render_table_includes_headers(self):
        rows = [{
            "file": "x.json", "model": "M", "use_npcot": False, "library": None,
            "problems": 10, "pass_count": 4, "pass_at_1": 0.4, "total_seconds": 5.0,
        }]
        txt = render_table(rows)
        self.assertIn("model", txt)
        self.assertIn("pass@1", txt)
        self.assertIn("40.0%", txt)

    def test_markdown_has_pipe_table(self):
        rows = [{
            "file": "x.json", "model": "M", "use_npcot": True, "library": "l.json",
            "problems": 10, "pass_count": 7, "pass_at_1": 0.7, "total_seconds": 5.0,
        }]
        md = render_markdown(rows, "humaneval")
        self.assertIn("| model", md)
        self.assertIn("+NPCoT", md)
        self.assertIn("70.0%", md)


class TestCLI(unittest.TestCase):
    def test_cli_reports_nonexistent_dir(self):
        result = subprocess.run(
            [sys.executable, "-m", "ncpu.self_optimizing.compare_runs",
             "/tmp/definitely-nowhere-xyz"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
