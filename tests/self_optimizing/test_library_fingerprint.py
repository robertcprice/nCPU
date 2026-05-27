"""Library fingerprint tests (NV3b)."""

from __future__ import annotations

import unittest

import torch

from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    ArrayProgramLibraryConfig,
    DiscreteArrayProgram,
)


class TestFingerprint(unittest.TestCase):
    def _populate(self, lib: ArrayProgramLibrary, *, order: int = 0) -> None:
        entries = [
            (
                torch.tensor([1.0, 0.0, 0.0]),
                DiscreteArrayProgram(0, 0, 0, 0, 0.0),
                "sum",
            ),
            (
                torch.tensor([0.0, 1.0, 0.0]),
                DiscreteArrayProgram(2, 0, 2, 0, 0.0),
                "max",
            ),
            (
                torch.tensor([0.0, 0.0, 1.0]),
                DiscreteArrayProgram(0, 4, 0, 0, 0.0),
                "count",
            ),
        ]
        if order:
            entries = list(reversed(entries))
        for sig, prog, task in entries:
            lib.record(sig, prog, task_name=task)

    def test_fingerprint_is_stable_prefix(self):
        lib = ArrayProgramLibrary()
        self._populate(lib)
        fp = lib.fingerprint()
        self.assertTrue(fp.startswith("npcot1:"))
        self.assertEqual(len(fp), len("npcot1:") + 32)

    def test_ordering_does_not_change_fingerprint(self):
        lib_a = ArrayProgramLibrary()
        lib_b = ArrayProgramLibrary()
        self._populate(lib_a, order=0)
        self._populate(lib_b, order=1)
        self.assertEqual(lib_a.fingerprint(), lib_b.fingerprint())

    def test_hit_counts_do_not_change_fingerprint(self):
        lib = ArrayProgramLibrary()
        self._populate(lib)
        original = lib.fingerprint()
        for _ in range(10):
            lib.lookup(torch.tensor([1.0, 0.0, 0.0]))
        self.assertEqual(lib.fingerprint(), original)

    def test_program_change_breaks_fingerprint(self):
        lib_a = ArrayProgramLibrary()
        lib_b = ArrayProgramLibrary()
        self._populate(lib_a)
        self._populate(lib_b)
        # Modify one program in B.
        lib_b._entries[0].program = DiscreteArrayProgram(0, 2, 0, 0, 0.0)
        self.assertNotEqual(lib_a.fingerprint(), lib_b.fingerprint())

    def test_extra_entry_breaks_fingerprint(self):
        lib_a = ArrayProgramLibrary()
        lib_b = ArrayProgramLibrary()
        self._populate(lib_a)
        self._populate(lib_b)
        lib_b.record(
            torch.tensor([0.5, 0.5, 0.0]),
            DiscreteArrayProgram(1, 0, 1, 0, 0.0),
            task_name="extra",
        )
        self.assertNotEqual(lib_a.fingerprint(), lib_b.fingerprint())

    def test_similarity_threshold_included_in_fingerprint(self):
        lib_a = ArrayProgramLibrary(
            ArrayProgramLibraryConfig(similarity_threshold=0.85)
        )
        lib_b = ArrayProgramLibrary(
            ArrayProgramLibraryConfig(similarity_threshold=0.95)
        )
        self._populate(lib_a)
        self._populate(lib_b)
        self.assertNotEqual(lib_a.fingerprint(), lib_b.fingerprint())

    def test_empty_library_fingerprint_is_stable(self):
        lib_a = ArrayProgramLibrary()
        lib_b = ArrayProgramLibrary()
        self.assertEqual(lib_a.fingerprint(), lib_b.fingerprint())


if __name__ == "__main__":
    unittest.main()
