"""Tests for `ProgramLibrarySession` (N1)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ncpu.self_optimizing.array_executable_thought_head import (
    ArrayExecutableThoughtHead,
    ArrayExecutableThoughtHeadConfig,
)
from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    ArrayProgramLibraryConfig,
    DiscreteArrayProgram,
)
from ncpu.self_optimizing.program_library_session import (
    ProgramLibrarySession,
    ProgramLibrarySessionConfig,
    ProgramLibraryTaskSummary,
    attach_session_to_provider,
)


class TestProgramLibrarySession(unittest.TestCase):
    def test_begin_task_creates_library_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            session = ProgramLibrarySession(
                ProgramLibrarySessionConfig(library_path=path)
            )
            meta = session.begin_task("task_1")
            self.assertEqual(meta["task_name"], "task_1")
            self.assertEqual(meta["entry_count"], 0)
            self.assertFalse(meta["reused_from_disk"])

    def test_begin_task_loads_existing_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            seed = ArrayProgramLibrary(ArrayProgramLibraryConfig())
            seed.record(
                torch.tensor([1.0, 0.0, 0.0]),
                DiscreteArrayProgram(0, 0, 0, 0, 0.0),
                task_name="sum",
            )
            seed.save(path)

            session = ProgramLibrarySession(
                ProgramLibrarySessionConfig(library_path=path)
            )
            meta = session.begin_task("task_2")
            self.assertEqual(meta["entry_count"], 1)
            self.assertTrue(meta["reused_from_disk"])

    def test_end_task_persists_library(self):
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
            self.assertIsInstance(summary, ProgramLibraryTaskSummary)
            self.assertTrue(summary.saved)
            self.assertTrue(path.exists())
            restored = ArrayProgramLibrary.load(path)
            self.assertEqual(len(restored), 1)

    def test_end_task_save_override(self):
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
            summary = session.end_task(save=False)
            self.assertFalse(summary.saved)
            self.assertFalse(path.exists())

    def test_apply_converged_program_integrates_with_head(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            head = ArrayExecutableThoughtHead(
                ArrayExecutableThoughtHeadConfig(
                    hidden_dim=8,
                    array_max_len=6,
                    trace_projection_dim=8,
                    trace_hidden_dim=16,
                    state_patch_dim=8,
                )
            )
            session = ProgramLibrarySession(
                ProgramLibrarySessionConfig(
                    library_path=path,
                    convergence_gap_threshold=5.0,
                )
            )
            session.begin_task("demo")
            hidden = torch.randn(2, 8)
            arrays = torch.tensor(
                [
                    [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
                    [-1.0, -2.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
            lengths = torch.tensor([3.0, 2.0])
            result = session.apply_converged_program(
                head,
                hidden,
                arrays,
                lengths=lengths,
                temperature=1.0,
            )
            self.assertEqual(len(result.programs), 2)
            session.end_task()

    def test_missing_library_raises_when_not_creating(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nonexistent.json"
            session = ProgramLibrarySession(
                ProgramLibrarySessionConfig(
                    library_path=path, create_if_missing=False
                )
            )
            with self.assertRaises(FileNotFoundError):
                session.begin_task("task_1")

    def test_session_rejects_access_before_begin(self):
        session = ProgramLibrarySession()
        with self.assertRaises(RuntimeError):
            _ = session.library
        with self.assertRaises(RuntimeError):
            session.end_task()


class _MockProvider:
    """Minimal provider mimicking HFTaskLocalFastWeightsProvider's lifecycle."""

    def __init__(self):
        self.begin_calls = []
        self.end_calls = 0

    def begin_task(self, task_name: str, task_prompt: str = "") -> dict:
        self.begin_calls.append(task_name)
        return {"task_name": task_name, "enabled": True}

    def end_task(self) -> dict:
        self.end_calls += 1
        return {"update_count": 0}


class TestAttachSession(unittest.TestCase):
    def test_attach_wraps_begin_and_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            provider = _MockProvider()
            session = attach_session_to_provider(
                provider,
                ProgramLibrarySessionConfig(library_path=path),
            )
            begin_result = provider.begin_task("demo")
            self.assertEqual(begin_result["task_name"], "demo")
            # Add a program so end-task has something to save.
            session.library.record(
                torch.tensor([1.0, 0.0, 0.0]),
                DiscreteArrayProgram(0, 0, 0, 0, 0.0),
                task_name="sum",
            )
            end_result = provider.end_task()
            self.assertEqual(provider.end_calls, 1)
            self.assertIn("program_library", end_result)
            self.assertEqual(end_result["program_library"]["entries"], 1)
            self.assertTrue(path.exists())

    def test_attach_rejects_provider_without_lifecycle(self):
        class Broken:
            pass

        with self.assertRaises(TypeError):
            attach_session_to_provider(Broken())

    def test_session_persists_across_attached_provider_invocations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            provider_a = _MockProvider()
            session_a = attach_session_to_provider(
                provider_a,
                ProgramLibrarySessionConfig(library_path=path),
            )
            provider_a.begin_task("run_1")
            session_a.library.record(
                torch.tensor([1.0, 0.0, 0.0]),
                DiscreteArrayProgram(0, 0, 0, 0, 0.0),
                task_name="sum",
            )
            provider_a.end_task()

            # Now a new provider in a fresh process-equivalent should see
            # the persisted library.
            provider_b = _MockProvider()
            session_b = attach_session_to_provider(
                provider_b,
                ProgramLibrarySessionConfig(library_path=path),
            )
            begin_meta = provider_b.begin_task("run_2")
            self.assertEqual(begin_meta["task_name"], "run_2")
            self.assertEqual(len(session_b.library), 1)


if __name__ == "__main__":
    unittest.main()
