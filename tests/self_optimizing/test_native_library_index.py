"""Native (Rust-backed) sharded library index tests (N3-next)."""

from __future__ import annotations

import unittest

import torch

from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    ArrayProgramLibraryConfig,
    DiscreteArrayProgram,
    get_native_backend,
    reset_native_backend_cache,
)


class TestNativeLibraryIndex(unittest.TestCase):
    def setUp(self):
        reset_native_backend_cache()
        self._has_native = (
            get_native_backend() is not None
            and hasattr(get_native_backend(), "NpcotLibraryIndex")
        )

    def test_build_returns_false_without_native(self):
        # Can't really simulate "no native" here; just assert the return
        # type is bool and the library still works either way.
        lib = ArrayProgramLibrary()
        lib.record(
            torch.tensor([1.0, 0.0]),
            DiscreteArrayProgram(0, 0, 0, 0, 0.0),
            task_name="x",
        )
        result = lib.build_native_index()
        self.assertIsInstance(result, bool)

    def test_native_index_lookup_agrees_with_python(self):
        if not self._has_native:
            self.skipTest("native library index not available")
        lib = ArrayProgramLibrary(
            ArrayProgramLibraryConfig(similarity_threshold=0.85)
        )
        lib.record(
            torch.tensor([1.0, 0.0, 0.0]),
            DiscreteArrayProgram(0, 0, 0, 0, 0.0),
            task_name="sum",
        )
        lib.record(
            torch.tensor([0.0, 1.0, 0.0]),
            DiscreteArrayProgram(2, 0, 2, 0, 0.0),
            task_name="max",
        )
        built = lib.build_native_index()
        self.assertTrue(built)
        hit = lib.lookup(torch.tensor([0.95, 0.1, 0.1]))
        self.assertIsNotNone(hit)
        self.assertEqual(hit.task_name, "sum")

    def test_native_index_large_library_still_correct(self):
        if not self._has_native:
            self.skipTest("native library index not available")
        lib = ArrayProgramLibrary(
            ArrayProgramLibraryConfig(similarity_threshold=0.9)
        )
        # Seed 20 entries along different basis axes in 8-D.
        for axis in range(8):
            sig = torch.zeros(8)
            sig[axis] = 1.0
            program = DiscreteArrayProgram(0, axis % 6, 0, 0, float(axis) * 0.1)
            lib.record(sig, program, task_name=f"axis_{axis}")
        lib.build_native_index()
        # Query along axis 3 should return the axis_3 program.
        query = torch.zeros(8)
        query[3] = 1.0
        hit = lib.lookup(query)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.task_name, "axis_3")
        self.assertEqual(hit.program.transform_idx, 3)

    def test_drop_native_index_reverts_to_python(self):
        if not self._has_native:
            self.skipTest("native library index not available")
        lib = ArrayProgramLibrary(
            ArrayProgramLibraryConfig(similarity_threshold=0.9)
        )
        lib.record(
            torch.tensor([1.0, 0.0, 0.0]),
            DiscreteArrayProgram(0, 0, 0, 0, 0.0),
            task_name="x",
        )
        lib.build_native_index()
        self.assertIsNotNone(lib._native_index)
        lib.drop_native_index()
        self.assertIsNone(lib._native_index)
        # Lookup still works via Python path.
        hit = lib.lookup(torch.tensor([1.0, 0.0, 0.0]))
        self.assertIsNotNone(hit)


if __name__ == "__main__":
    unittest.main()
