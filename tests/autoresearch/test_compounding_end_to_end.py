"""End-to-end test: every solve permanently improves the next run.

This is the "always compounding" contract — after the runner has solved
a problem once, the next session returns it via a store hit without
invoking the cascade again, even if the legacy solved-log is deleted.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ncpu.autoresearch.cascade import CascadeConfig
from ncpu.autoresearch.compounding_store import (
    CompoundingStore,
    CompoundingStoreConfig,
)
from ncpu.autoresearch.runner import run_session
from ncpu.autoresearch.types import Budget, IoPair, WorkItem


def _write_queue(path: Path, items: list[WorkItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for it in items:
            fh.write(json.dumps(it.to_dict()) + "\n")


SUM_TEST = """
def check(candidate):
    assert candidate([1, 2, 3]) == 6
    assert candidate([]) == 0
"""


class TestAlwaysCompounding(unittest.TestCase):
    def test_store_hit_survives_legacy_log_deletion(self):
        """If we nuke solved_programs.jsonl, the prompt cache should
        *still* catch the re-run because the cache file persists."""
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            qpath = art / "queue.jsonl"
            spath = art / "solved.jsonl"
            store = CompoundingStore(CompoundingStoreConfig(
                artifact_dir=art, solved_log_name="solved.jsonl",
            ))

            item = WorkItem(
                task_id="h/1", source_benchmark="humaneval",
                prompt="def total(xs):\n    \"\"\"sum\"\"\"\n",
                entry_point="total", test_source=SUM_TEST,
                io_pairs=[IoPair(args=[[1, 2, 3]], kwargs={}, expected=6),
                          IoPair(args=[[]], kwargs={}, expected=0)],
                priority=1.0,
            )
            _write_queue(qpath, [item])

            r1 = run_session(
                queue_path=qpath, solved_path=spath,
                cascade_config=CascadeConfig(solver_names=["template_match"]),
                budget=Budget(wall_seconds=30, max_problems=5),
                store=store,
            )
            self.assertEqual(r1.problems_solved, 1)
            self.assertEqual(r1.problems_attempted, 1)

            # Nuke the legacy log; prompt cache + temp stats remain.
            spath.unlink()

            r2 = run_session(
                queue_path=qpath, solved_path=spath,
                cascade_config=CascadeConfig(solver_names=["template_match"]),
                budget=Budget(wall_seconds=30, max_problems=5),
                store=store,
            )
            # Store hit takes effect → cascade was NOT run.
            self.assertEqual(r2.problems_attempted, 0)
            self.assertEqual(r2.store_hits, 1)
            self.assertEqual(r2.problems_solved, 1)

    def test_summary_reflects_growth_across_sessions(self):
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            qpath = art / "queue.jsonl"
            spath = art / "solved.jsonl"
            store = CompoundingStore(CompoundingStoreConfig(
                artifact_dir=art, solved_log_name="solved.jsonl",
            ))

            items = [
                WorkItem(task_id=f"h/{i}", source_benchmark="humaneval",
                         prompt=f"def total_{i}(xs):\n    \"\"\"sum_{i}\"\"\"\n",
                         entry_point=f"total_{i}", test_source=SUM_TEST,
                         io_pairs=[IoPair(args=[[1, 2]], kwargs={}, expected=3),
                                   IoPair(args=[[]], kwargs={}, expected=0)],
                         priority=1.0)
                for i in range(3)
            ]
            _write_queue(qpath, items)

            run_session(
                queue_path=qpath, solved_path=spath,
                cascade_config=CascadeConfig(solver_names=["template_match"]),
                store=store,
            )
            summary = store.summary()
            self.assertEqual(summary["solved_programs"], 3)
            self.assertEqual(summary["prompt_cache_size"], 3)


if __name__ == "__main__":
    unittest.main()
