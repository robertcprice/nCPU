"""Tests for autoresearch.distiller."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ncpu.autoresearch.distiller import (
    append_solved,
    dedupe_solved,
    load_solved,
    summarize_solved,
)
from ncpu.autoresearch.types import SolvedItem


class TestDistillerIO(unittest.TestCase):
    def test_append_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "solved.jsonl"
            items = [
                SolvedItem(
                    task_id="h/1", source_benchmark="humaneval",
                    solver="template_match", program_python="    return 1\n",
                    wall_seconds=0.1,
                ),
                SolvedItem(
                    task_id="h/2", source_benchmark="humaneval",
                    solver="llm_resample", program_python="    return 2\n",
                    wall_seconds=10.5,
                ),
            ]
            for it in items:
                append_solved(it, out_path=path)
            loaded = load_solved(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].task_id, "h/1")
            self.assertEqual(loaded[1].solver, "llm_resample")

    def test_dedupe_keeps_latest(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "solved.jsonl"
            a = SolvedItem(task_id="h/1", source_benchmark="humaneval",
                           solver="template_match", program_python="    return 1\n",
                           wall_seconds=0.1)
            b = SolvedItem(task_id="h/1", source_benchmark="humaneval",
                           solver="llm_resample", program_python="    return 42\n",
                           wall_seconds=20.0)
            append_solved(a, out_path=path)
            append_solved(b, out_path=path)
            n = dedupe_solved(path)
            self.assertEqual(n, 1)
            loaded = load_solved(path)
            self.assertEqual(loaded[0].solver, "llm_resample")
            self.assertIn("42", loaded[0].program_python)

    def test_summarize(self):
        items = [
            SolvedItem(task_id="a", source_benchmark="h", solver="template_match",
                       program_python="", wall_seconds=0.5),
            SolvedItem(task_id="b", source_benchmark="h", solver="template_match",
                       program_python="", wall_seconds=1.0),
            SolvedItem(task_id="c", source_benchmark="h", solver="llm_resample",
                       program_python="", wall_seconds=30.0),
        ]
        s = summarize_solved(items)
        self.assertEqual(s["total_solved"], 3)
        self.assertEqual(s["by_solver"]["template_match"], 2)
        self.assertEqual(s["by_solver"]["llm_resample"], 1)
        self.assertEqual(s["total_wall_seconds"], 31.5)


if __name__ == "__main__":
    unittest.main()
