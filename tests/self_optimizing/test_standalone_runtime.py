"""Standalone pure-Rust runtime tests (N5-next)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    ArrayProgramLibraryConfig,
    DiscreteArrayProgram,
    get_native_backend,
    reset_native_backend_cache,
)


class TestNpcotStandaloneRuntime(unittest.TestCase):
    def setUp(self):
        reset_native_backend_cache()
        self._backend = get_native_backend()
        if self._backend is None or not hasattr(
            self._backend, "NpcotStandaloneRuntime"
        ):
            self.skipTest("standalone runtime not available")

    def _save_library(self, tmp: Path) -> Path:
        library = ArrayProgramLibrary(
            ArrayProgramLibraryConfig(similarity_threshold=0.85)
        )
        library.record(
            torch.tensor([1.0, 0.0, 0.0]),
            DiscreteArrayProgram(0, 0, 0, 0, 0.0),
            task_name="sum",
        )
        library.record(
            torch.tensor([0.0, 1.0, 0.0]),
            DiscreteArrayProgram(2, 0, 2, 0, 0.0),
            task_name="max",
        )
        path = tmp / "lib.json"
        library.save(path)
        return path

    def test_load_from_path_and_consult_sum(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib_path = self._save_library(Path(tmp))
            runtime = self._backend.NpcotStandaloneRuntime.from_json_path(
                str(lib_path)
            )
            self.assertEqual(runtime.entry_count(), 2)
            self.assertAlmostEqual(
                runtime.similarity_threshold(), 0.85, places=3
            )
            # Consult with hidden aligned to the SUM signature; expect 6.
            result = runtime.consult(
                [1.0, 0.0, 0.0], [1.0, 2.0, 3.0, 0.0, 0.0], 3
            )
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result, 6.0, places=4)

    def test_consult_miss_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib_path = self._save_library(Path(tmp))
            runtime = self._backend.NpcotStandaloneRuntime.from_json_path(
                str(lib_path)
            )
            # A hidden orthogonal to all stored signatures should miss
            # when similarity is below threshold. [0.3, 0.3, 0.3] is not
            # sufficiently aligned with either axis.
            # (We set threshold to 0.85 so 1/sqrt(3)=0.577 cosine is a miss.)
            result = runtime.consult(
                [0.3, 0.3, 0.3], [1.0, 2.0, 3.0], 3
            )
            self.assertIsNone(result)

    def test_consult_from_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib_path = self._save_library(Path(tmp))
            payload = lib_path.read_bytes()
            runtime = self._backend.NpcotStandaloneRuntime.from_json_bytes(
                payload
            )
            self.assertEqual(runtime.entry_count(), 2)
            result = runtime.consult(
                [0.0, 1.0, 0.0], [5.0, -2.0, 3.0], 3
            )
            self.assertIsNotNone(result)
            # MAX program: max of [5, -2, 3] = 5
            self.assertAlmostEqual(result, 5.0, places=4)

    def test_consult_agrees_with_python_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib_path = self._save_library(Path(tmp))
            runtime = self._backend.NpcotStandaloneRuntime.from_json_path(
                str(lib_path)
            )
            library = ArrayProgramLibrary.load(lib_path)
            hidden = torch.tensor([1.0, 0.0, 0.0])
            array = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0]])
            lengths = torch.tensor([3.0])
            # Python path
            entry = library.lookup(hidden)
            self.assertIsNotNone(entry)
            python_out = entry.program.execute(array, lengths).item()
            # Rust standalone path
            rust_out = runtime.consult(
                hidden.tolist(), array[0].tolist(), 3
            )
            self.assertAlmostEqual(python_out, rust_out, places=4)


if __name__ == "__main__":
    unittest.main()
