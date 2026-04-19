"""Tests for the always-compounding persistent store."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ncpu.autoresearch.compounding_store import (
    CompoundingStore,
    CompoundingStoreConfig,
    hash_prompt,
)
from ncpu.autoresearch.types import IoPair, SolvedItem, WorkItem


def _mk_item(task_id: str, entry_point: str = "f",
              prompt: str = "def f(x):\n    pass\n") -> WorkItem:
    return WorkItem(
        task_id=task_id, source_benchmark="humaneval",
        prompt=prompt, entry_point=entry_point,
        test_source="", io_pairs=[], priority=1.0,
    )


def _mk_solved(task_id: str, **provenance) -> SolvedItem:
    return SolvedItem(
        task_id=task_id, source_benchmark="humaneval",
        solver="llm_resample", program_python="    return 1\n",
        verifier_passed=True, wall_seconds=1.0,
        provenance=dict(provenance),
    )


class TestCompoundingStore(unittest.TestCase):
    def test_prompt_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            store = CompoundingStore(CompoundingStoreConfig(artifact_dir=Path(td)))
            item = _mk_item("h/1", "g", "def g(n):\n    pass\n")
            self.assertIsNone(store.check_prompt(item))
            solved = _mk_solved("h/1",
                                winning_temperature=0.5,
                                winning_sample_idx=2,
                                prompt=item.prompt,
                                entry_point=item.entry_point)
            store.record(solved, work_item=item)
            hit = store.check_prompt(item)
            self.assertIsNotNone(hit)
            self.assertEqual(hit.task_id, "h/1")
            self.assertEqual(hit.source, "prompt_exact")

    def test_temperature_stats_accumulate(self):
        with tempfile.TemporaryDirectory() as td:
            store = CompoundingStore(CompoundingStoreConfig(artifact_dir=Path(td)))
            store.record(_mk_solved("h/1", winning_temperature=0.7), work_item=_mk_item("h/1"))
            store.record(_mk_solved("h/2", winning_temperature=0.7), work_item=_mk_item("h/2", "f", "def f(y):\n    pass\n"))
            store.record(_mk_solved("h/3", winning_temperature=0.3), work_item=_mk_item("h/3", "f", "def f(z):\n    pass\n"))
            stats = store.temperature_stats()
            self.assertEqual(stats["0.70"], 2)
            self.assertEqual(stats["0.30"], 1)

    def test_task_id_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            store = CompoundingStore(CompoundingStoreConfig(artifact_dir=Path(td)))
            store.record(_mk_solved("h/1"), work_item=_mk_item("h/1"))
            # Different prompt but same task_id — task_id fallback still hits.
            hit = store.check_task_id("h/1")
            self.assertIsNotNone(hit)
            self.assertEqual(hit.source, "task_id")

    def test_hash_prompt_stable(self):
        h1 = hash_prompt("abc", "x")
        h2 = hash_prompt("abc", "x")
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, hash_prompt("abc", "y"))
        self.assertNotEqual(h1, hash_prompt("abcd", "x"))

    def test_rebuild_indices(self):
        with tempfile.TemporaryDirectory() as td:
            store = CompoundingStore(CompoundingStoreConfig(artifact_dir=Path(td)))
            items = [
                _mk_item("h/1"),
                _mk_item("h/2", "g", "def g(a):\n    pass\n"),
            ]
            solves = [
                _mk_solved("h/1", winning_temperature=0.5,
                            prompt=items[0].prompt, entry_point=items[0].entry_point),
                _mk_solved("h/2", winning_temperature=0.7,
                            prompt=items[1].prompt, entry_point=items[1].entry_point),
            ]
            for it, s in zip(items, solves):
                store.record(s, work_item=it)

            # Remove the cache + stats files; rebuild should restore them.
            store.prompt_cache_path.unlink()
            store.temp_stats_path.unlink()
            store._prompt_cache = None
            store._temp_stats = None
            counters = store.rebuild_indices()
            self.assertEqual(counters["prompt_cache"], 2)
            self.assertEqual(counters["temperature_stats"], 2)

            # Cache hit still works after rebuild.
            hit = store.check_prompt(items[0])
            self.assertIsNotNone(hit)

    def test_summary(self):
        with tempfile.TemporaryDirectory() as td:
            store = CompoundingStore(CompoundingStoreConfig(artifact_dir=Path(td)))
            store.record(_mk_solved("h/1", winning_temperature=0.3),
                         work_item=_mk_item("h/1"))
            s = store.summary()
            self.assertEqual(s["solved_programs"], 1)
            self.assertEqual(s["prompt_cache_size"], 1)
            self.assertEqual(s["temperature_stats"]["0.30"], 1)


if __name__ == "__main__":
    unittest.main()
