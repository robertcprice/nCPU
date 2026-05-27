"""Snapshot + diff tests for ArrayProgramLibrary and sessions (N4-next)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    ArrayProgramLibraryConfig,
    DiscreteArrayProgram,
)
from ncpu.self_optimizing.program_library_session import (
    ProgramLibrarySession,
    ProgramLibrarySessionConfig,
)


class TestLibrarySnapshotDiff(unittest.TestCase):
    def _make(self) -> ArrayProgramLibrary:
        return ArrayProgramLibrary(
            ArrayProgramLibraryConfig(similarity_threshold=0.85)
        )

    def test_snapshot_is_list_of_dicts(self):
        lib = self._make()
        lib.record(
            torch.tensor([1.0, 0.0, 0.0]),
            DiscreteArrayProgram(0, 0, 0, 0, 0.0),
            task_name="sum",
        )
        snap = lib.snapshot()
        self.assertIsInstance(snap, list)
        self.assertEqual(len(snap), 1)
        self.assertIn("signature", snap[0])
        self.assertIn("program", snap[0])

    def test_diff_empty_on_no_changes(self):
        lib = self._make()
        lib.record(
            torch.tensor([1.0, 0.0]),
            DiscreteArrayProgram(0, 0, 0, 0, 0.0),
            task_name="sum",
        )
        snap = lib.snapshot()
        diff = lib.diff_against(snap)
        self.assertEqual(diff["added"], [])
        self.assertEqual(diff["removed"], [])
        self.assertEqual(diff["changed"], [])
        self.assertEqual(diff["unchanged"], 1)

    def test_diff_detects_added(self):
        lib = self._make()
        lib.record(
            torch.tensor([1.0, 0.0]),
            DiscreteArrayProgram(0, 0, 0, 0, 0.0),
            task_name="sum",
        )
        snap = lib.snapshot()
        lib.record(
            torch.tensor([0.0, 1.0]),
            DiscreteArrayProgram(2, 0, 2, 0, 0.0),
            task_name="max",
        )
        diff = lib.diff_against(snap)
        self.assertEqual(len(diff["added"]), 1)
        self.assertEqual(diff["added"][0]["task_name"], "max")
        self.assertEqual(len(diff["removed"]), 0)

    def test_diff_detects_program_change(self):
        lib = self._make()
        lib.record(
            torch.tensor([1.0, 0.0]),
            DiscreteArrayProgram(0, 0, 0, 0, 0.0),
            task_name="sum",
        )
        snap = lib.snapshot()
        # Same signature, different program — simulates a library "upgrade"
        # where a later convergence overwrote a prior discrete program.
        lib.record(
            torch.tensor([1.0, 0.0]),
            DiscreteArrayProgram(0, 2, 0, 0, 0.0),
            task_name="sum",
        )
        diff = lib.diff_against(snap)
        self.assertEqual(len(diff["changed"]), 1)
        self.assertEqual(diff["changed"][0]["before"]["task_name"], "sum")

    def test_diff_reports_hits_since_snapshot(self):
        lib = self._make()
        lib.record(
            torch.tensor([1.0, 0.0]),
            DiscreteArrayProgram(0, 0, 0, 0, 0.0),
            task_name="sum",
        )
        snap = lib.snapshot()
        # Two lookups count as 2 hits.
        lib.lookup(torch.tensor([1.0, 0.0]))
        lib.lookup(torch.tensor([1.0, 0.0]))
        diff = lib.diff_against(snap)
        self.assertEqual(diff["hits_since_snapshot"], 2)
        self.assertEqual(diff["unchanged"], 1)


class TestSessionDiff(unittest.TestCase):
    def test_end_task_attaches_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            session = ProgramLibrarySession(
                ProgramLibrarySessionConfig(library_path=path)
            )
            session.begin_task("task_1")
            session.library.record(
                torch.tensor([1.0, 0.0, 0.0]),
                DiscreteArrayProgram(0, 0, 0, 0, 0.0),
                task_name="sum",
            )
            summary = session.end_task()
            self.assertIsNotNone(summary.diff)
            self.assertEqual(len(summary.diff["added"]), 1)
            self.assertEqual(summary.diff["added"][0]["task_name"], "sum")

    def test_end_task_diff_skipped_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            session = ProgramLibrarySession(
                ProgramLibrarySessionConfig(library_path=path)
            )
            session.begin_task("task_1")
            session.library.record(
                torch.tensor([1.0, 0.0]),
                DiscreteArrayProgram(0, 0, 0, 0, 0.0),
                task_name="x",
            )
            summary = session.end_task(include_diff=False)
            self.assertIsNone(summary.diff)

    def test_second_session_diff_shows_only_new_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            # First session: add entry.
            session_a = ProgramLibrarySession(
                ProgramLibrarySessionConfig(library_path=path)
            )
            session_a.begin_task("a")
            session_a.library.record(
                torch.tensor([1.0, 0.0]),
                DiscreteArrayProgram(0, 0, 0, 0, 0.0),
                task_name="sum",
            )
            session_a.end_task()

            # Second session: add another entry.
            session_b = ProgramLibrarySession(
                ProgramLibrarySessionConfig(library_path=path)
            )
            session_b.begin_task("b")
            session_b.library.record(
                torch.tensor([0.0, 1.0]),
                DiscreteArrayProgram(2, 0, 2, 0, 0.0),
                task_name="max",
            )
            summary_b = session_b.end_task()
            # Only "max" is added in session B's diff — "sum" was already
            # there from A and shows as unchanged.
            self.assertEqual(len(summary_b.diff["added"]), 1)
            self.assertEqual(summary_b.diff["added"][0]["task_name"], "max")
            self.assertEqual(summary_b.diff["unchanged"], 1)


if __name__ == "__main__":
    unittest.main()
