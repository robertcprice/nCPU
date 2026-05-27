"""Library merge tests (NV4b)."""

from __future__ import annotations

import unittest

import torch

from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    ArrayProgramLibraryConfig,
    DiscreteArrayProgram,
    merge_libraries,
)


def _lib_with(*entries: tuple[list[float], DiscreteArrayProgram, str, int]) -> ArrayProgramLibrary:
    lib = ArrayProgramLibrary(
        ArrayProgramLibraryConfig(similarity_threshold=0.85)
    )
    for sig, prog, task, hits in entries:
        lib.record(torch.tensor(sig), prog, task_name=task)
        # Artificially bump hit count to simulate usage.
        entry = lib._entries[-1]
        entry.hit_count = hits
    return lib


class TestMergeLibraries(unittest.TestCase):
    def test_empty_list_returns_empty_library(self):
        merged = merge_libraries([])
        self.assertEqual(len(merged), 0)

    def test_union_non_overlapping_libraries(self):
        lib_a = _lib_with(
            ([1.0, 0.0, 0.0], DiscreteArrayProgram(0, 0, 0, 0, 0.0), "sum", 0),
        )
        lib_b = _lib_with(
            ([0.0, 1.0, 0.0], DiscreteArrayProgram(2, 0, 2, 0, 0.0), "max", 0),
        )
        merged = merge_libraries([lib_a, lib_b])
        self.assertEqual(len(merged), 2)
        tasks = {e.task_name for e in merged.entries}
        self.assertEqual(tasks, {"sum", "max"})

    def test_overlap_keep_more_hits(self):
        lib_old = _lib_with(
            ([1.0, 0.0, 0.0], DiscreteArrayProgram(0, 0, 0, 0, 0.0), "sum_v1", 100),
        )
        lib_new = _lib_with(
            ([1.0, 0.0, 0.0], DiscreteArrayProgram(0, 2, 0, 0, 0.0), "sum_v2", 5),
        )
        merged = merge_libraries(
            [lib_old, lib_new], conflict_resolution="keep_more_hits"
        )
        self.assertEqual(len(merged), 1)
        # Higher-hit version won.
        self.assertEqual(merged.entries[0].task_name, "sum_v1")
        self.assertEqual(merged.entries[0].program.transform_idx, 0)

    def test_overlap_keep_newer(self):
        lib_old = _lib_with(
            ([1.0, 0.0, 0.0], DiscreteArrayProgram(0, 0, 0, 0, 0.0), "sum_v1", 100),
        )
        lib_new = _lib_with(
            ([1.0, 0.0, 0.0], DiscreteArrayProgram(0, 2, 0, 0, 0.0), "sum_v2", 5),
        )
        merged = merge_libraries(
            [lib_old, lib_new], conflict_resolution="keep_newer"
        )
        self.assertEqual(len(merged), 1)
        # Newer (last-in-list) program wins.
        self.assertEqual(merged.entries[0].task_name, "sum_v2")
        self.assertEqual(merged.entries[0].program.transform_idx, 2)

    def test_overlap_keep_both(self):
        lib_old = _lib_with(
            ([1.0, 0.0, 0.0], DiscreteArrayProgram(0, 0, 0, 0, 0.0), "sum_v1", 0),
        )
        lib_new = _lib_with(
            ([1.0, 0.0, 0.0], DiscreteArrayProgram(0, 2, 0, 0, 0.0), "sum_v2", 0),
        )
        merged = merge_libraries(
            [lib_old, lib_new], conflict_resolution="keep_both"
        )
        self.assertEqual(len(merged), 2)

    def test_signature_dim_mismatch_raises(self):
        lib_a = _lib_with(
            ([1.0, 0.0, 0.0], DiscreteArrayProgram(0, 0, 0, 0, 0.0), "x", 0),
        )
        lib_b = _lib_with(
            ([1.0, 0.0], DiscreteArrayProgram(0, 0, 0, 0, 0.0), "y", 0),
        )
        with self.assertRaises(ValueError):
            merge_libraries([lib_a, lib_b])

    def test_invalid_conflict_resolution_raises(self):
        # Empty libraries list short-circuits, so supply a non-empty one.
        lib = _lib_with(
            ([1.0, 0.0, 0.0], DiscreteArrayProgram(0, 0, 0, 0, 0.0), "x", 0),
        )
        with self.assertRaises(ValueError):
            merge_libraries([lib], conflict_resolution="bogus")

    def test_merge_preserves_capacity_cap(self):
        lib_a = _lib_with(
            ([1.0, 0.0, 0.0], DiscreteArrayProgram(0, 0, 0, 0, 0.0), "sum", 10),
            ([0.0, 1.0, 0.0], DiscreteArrayProgram(2, 0, 2, 0, 0.0), "max", 5),
        )
        lib_b = _lib_with(
            ([0.0, 0.0, 1.0], DiscreteArrayProgram(0, 4, 0, 0, 0.0), "count", 20),
        )
        merged = merge_libraries(
            [lib_a, lib_b],
            target_config=ArrayProgramLibraryConfig(
                similarity_threshold=0.85, max_entries=2
            ),
        )
        self.assertEqual(len(merged), 2)
        # Highest-hit entries survive.
        task_names = {e.task_name for e in merged.entries}
        self.assertIn("count", task_names)
        self.assertIn("sum", task_names)

    def test_merge_three_libraries(self):
        lib_a = _lib_with(
            ([1.0, 0.0, 0.0], DiscreteArrayProgram(0, 0, 0, 0, 0.0), "sum", 0),
        )
        lib_b = _lib_with(
            ([0.0, 1.0, 0.0], DiscreteArrayProgram(2, 0, 2, 0, 0.0), "max", 0),
        )
        lib_c = _lib_with(
            ([0.0, 0.0, 1.0], DiscreteArrayProgram(0, 4, 0, 0, 0.0), "count", 0),
        )
        merged = merge_libraries([lib_a, lib_b, lib_c])
        self.assertEqual(len(merged), 3)


if __name__ == "__main__":
    unittest.main()
