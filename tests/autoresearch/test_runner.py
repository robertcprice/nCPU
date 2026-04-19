"""Tests for autoresearch.runner (session loop + budgets)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ncpu.autoresearch.cascade import CascadeConfig
from ncpu.autoresearch.runner import run_session
from ncpu.autoresearch.types import Budget, IoPair, WorkItem


def _write_queue(path: Path, items: list[WorkItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for it in items:
            fh.write(json.dumps(it.to_dict()) + "\n")


SUM_PROMPT = """def total(xs):
    \"\"\"Return sum of xs.\"\"\"
"""

SUM_TEST = """
def check(candidate):
    assert candidate([]) == 0
    assert candidate([1, 2, 3]) == 6
"""


class TestRunner(unittest.TestCase):
    def _item(self, task_id: str, pairs: list[IoPair]) -> WorkItem:
        return WorkItem(
            task_id=task_id, source_benchmark="humaneval",
            prompt=SUM_PROMPT, entry_point="total",
            test_source=SUM_TEST, io_pairs=pairs, priority=1.0,
        )

    def test_skips_already_solved(self):
        with tempfile.TemporaryDirectory() as td:
            qpath = Path(td) / "q.jsonl"
            spath = Path(td) / "solved.jsonl"
            items = [
                self._item("h/1", [IoPair(args=[[1, 2, 3]], kwargs={}, expected=6),
                                   IoPair(args=[[]], kwargs={}, expected=0)]),
                self._item("h/2", [IoPair(args=[[10]], kwargs={}, expected=10),
                                   IoPair(args=[[]], kwargs={}, expected=0)]),
            ]
            _write_queue(qpath, items)

            # First run: solve both.
            r1 = run_session(
                queue_path=qpath, solved_path=spath,
                cascade_config=CascadeConfig(solver_names=["template_match"]),
                budget=Budget(wall_seconds=30, max_problems=10),
            )
            self.assertEqual(r1.problems_solved, 2)

            # Second run: both should be skipped, 0 new attempts.
            r2 = run_session(
                queue_path=qpath, solved_path=spath,
                cascade_config=CascadeConfig(solver_names=["template_match"]),
                budget=Budget(wall_seconds=30, max_problems=10),
            )
            self.assertEqual(r2.problems_attempted, 0)
            self.assertEqual(r2.problems_already_solved_skipped, 2)

    def test_problem_budget_stops_loop(self):
        with tempfile.TemporaryDirectory() as td:
            qpath = Path(td) / "q.jsonl"
            spath = Path(td) / "solved.jsonl"
            items = [
                self._item(f"h/{i}", [IoPair(args=[[1, 2]], kwargs={}, expected=3),
                                       IoPair(args=[[]], kwargs={}, expected=0)])
                for i in range(5)
            ]
            _write_queue(qpath, items)
            report = run_session(
                queue_path=qpath, solved_path=spath,
                cascade_config=CascadeConfig(solver_names=["template_match"]),
                budget=Budget(wall_seconds=30, max_problems=2),
            )
            self.assertEqual(report.problems_attempted, 2)
            self.assertEqual(report.stopped_reason, "problem_budget")

    def test_status_file_written(self):
        with tempfile.TemporaryDirectory() as td:
            qpath = Path(td) / "q.jsonl"
            spath = Path(td) / "solved.jsonl"
            stpath = Path(td) / "status.json"
            _write_queue(qpath, [
                self._item("h/1", [IoPair(args=[[1, 2]], kwargs={}, expected=3),
                                   IoPair(args=[[]], kwargs={}, expected=0)])
            ])
            run_session(
                queue_path=qpath, solved_path=spath, status_path=stpath,
                cascade_config=CascadeConfig(solver_names=["template_match"]),
            )
            self.assertTrue(stpath.exists())
            data = json.loads(stpath.read_text())
            self.assertIn("problems_attempted", data)


if __name__ == "__main__":
    unittest.main()
