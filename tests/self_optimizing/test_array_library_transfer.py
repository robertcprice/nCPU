"""Cross-model library-transfer tests (NV2)."""

from __future__ import annotations

import unittest

import torch

from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    ArrayProgramLibraryConfig,
    DiscreteArrayProgram,
    transfer_library,
)


def _make_source_library() -> ArrayProgramLibrary:
    library = ArrayProgramLibrary(
        ArrayProgramLibraryConfig(similarity_threshold=0.85, max_entries=32)
    )
    library.record(
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        DiscreteArrayProgram(0, 0, 0, 0, 0.0),
        task_name="sum",
    )
    library.record(
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        DiscreteArrayProgram(2, 0, 2, 0, 0.0),
        task_name="max",
    )
    library.record(
        torch.tensor([0.0, 0.0, 1.0, 0.0]),
        DiscreteArrayProgram(0, 4, 0, 0, 0.0),
        task_name="count",
    )
    return library


class TestTransferLibrary(unittest.TestCase):
    def test_identity_projection_preserves_signatures(self):
        source = _make_source_library()
        projection = torch.eye(4)
        target = transfer_library(source, projection=projection)
        self.assertEqual(len(target), 3)

        # Each target signature should be identical to its source (since
        # input signatures were unit-norm already and projection is identity).
        source_sigs = [entry.signature for entry in source.entries]
        target_sigs = [entry.signature for entry in target.entries]
        for src_sig, tgt_sig in zip(source_sigs, target_sigs):
            self.assertEqual(len(src_sig), len(tgt_sig))
            for a, b in zip(src_sig, tgt_sig):
                self.assertAlmostEqual(a, b, places=6)

    def test_dim_change_projection_to_smaller_dim(self):
        source = _make_source_library()
        # Project 4 -> 2 via a learned-looking map.
        projection = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ]
        )
        target = transfer_library(source, projection=projection)
        self.assertEqual(len(target), 3)
        for entry in target.entries:
            self.assertEqual(len(entry.signature), 2)
            # Unit norm preserved.
            norm = sum(v * v for v in entry.signature) ** 0.5
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_collapsed_signatures_are_skipped(self):
        # A projection with a zero row will collapse some signatures —
        # they must not be recorded as zero-norm entries.
        source = ArrayProgramLibrary(
            ArrayProgramLibraryConfig(similarity_threshold=0.85)
        )
        source.record(
            torch.tensor([1.0, 0.0]),
            DiscreteArrayProgram(0, 0, 0, 0, 0.0),
            task_name="x",
        )
        source.record(
            torch.tensor([0.0, 1.0]),
            DiscreteArrayProgram(0, 1, 0, 0, 0.0),
            task_name="y",
        )
        projection = torch.tensor([[1.0, 0.0]])  # projects to 1-D, kills "y"
        target = transfer_library(source, projection=projection)
        # "y" signature was [0, 1] which the projection maps to 0 — dropped.
        self.assertEqual(len(target), 1)
        self.assertEqual(target.entries[0].task_name, "x_xfer")

    def test_programs_carry_over_unchanged(self):
        source = _make_source_library()
        projection = torch.eye(4)
        target = transfer_library(source, projection=projection)
        for src_entry, tgt_entry in zip(source.entries, target.entries):
            self.assertEqual(src_entry.program.key(), tgt_entry.program.key())
            self.assertAlmostEqual(
                src_entry.program.offset, tgt_entry.program.offset, places=6
            )

    def test_task_name_suffix_applied(self):
        source = _make_source_library()
        projection = torch.eye(4)
        target = transfer_library(
            source, projection=projection, task_name_suffix="_student"
        )
        names = {entry.task_name for entry in target.entries}
        self.assertEqual(names, {"sum_student", "max_student", "count_student"})

    def test_invalid_projection_shape_raises(self):
        source = _make_source_library()
        with self.assertRaises(ValueError):
            transfer_library(source, projection=torch.zeros(4))

    def test_signature_dim_mismatch_raises(self):
        source = _make_source_library()
        # Projection expects dim=3 inputs, but source entries have dim=4.
        with self.assertRaises(ValueError):
            transfer_library(source, projection=torch.zeros(3, 3))

    def test_end_to_end_transfer_enables_lookup_on_target_hidden_state(self):
        """Verify a transferred library answers lookups from the target model."""
        source = _make_source_library()
        # Rotate from 4-D to 3-D via a deterministic projection.
        projection = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        target = transfer_library(source, projection=projection)

        # A hidden state that's highly aligned with the projected "sum"
        # signature should hit.
        query = torch.tensor([1.0, 0.0, 0.0])
        hit = target.lookup(query)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.program.key(), DiscreteArrayProgram(0, 0, 0, 0, 0.0).key())


if __name__ == "__main__":
    unittest.main()
