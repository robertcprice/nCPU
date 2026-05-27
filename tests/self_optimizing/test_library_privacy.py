"""Differential-privacy perturbation tests (NV5b)."""

from __future__ import annotations

import math
import unittest

import torch

from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    DiscreteArrayProgram,
)
from ncpu.self_optimizing.library_privacy import (
    DPCertificate,
    dp_perturb_library,
)


def _basic_library() -> ArrayProgramLibrary:
    lib = ArrayProgramLibrary()
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
    return lib


class TestDPPerturbation(unittest.TestCase):
    def test_certificate_fields_populated(self):
        lib = _basic_library()
        _, cert = dp_perturb_library(lib, epsilon=1.0, delta=1e-5, seed=0)
        self.assertIsInstance(cert, DPCertificate)
        self.assertEqual(cert.epsilon, 1.0)
        self.assertEqual(cert.delta, 1e-5)
        self.assertEqual(cert.entries_perturbed, 2)
        self.assertEqual(cert.signature_dim, 3)
        # sigma = sqrt(2 ln(1.25/delta)) * sensitivity / epsilon
        expected_sigma = (
            math.sqrt(2.0 * math.log(1.25 / 1e-5)) * 2.0 / 1.0
        )
        self.assertAlmostEqual(cert.sigma, expected_sigma, places=5)

    def test_programs_preserved_across_perturbation(self):
        lib = _basic_library()
        perturbed, _ = dp_perturb_library(
            lib, epsilon=1.0, delta=1e-5, seed=0
        )
        original_keys = {e.program.key() for e in lib.entries}
        perturbed_keys = {e.program.key() for e in perturbed.entries}
        self.assertEqual(original_keys, perturbed_keys)

    def test_signatures_remain_unit_norm(self):
        lib = _basic_library()
        perturbed, _ = dp_perturb_library(
            lib, epsilon=1.0, delta=1e-5, seed=0
        )
        for entry in perturbed.entries:
            norm = math.sqrt(sum(v * v for v in entry.signature))
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_signatures_differ_from_originals(self):
        lib = _basic_library()
        perturbed, _ = dp_perturb_library(
            lib, epsilon=0.5, delta=1e-5, seed=0
        )
        for orig_entry, new_entry in zip(lib.entries, perturbed.entries):
            self.assertNotEqual(orig_entry.signature, new_entry.signature)

    def test_larger_epsilon_means_less_noise(self):
        lib = _basic_library()
        # sigma is larger for smaller epsilon → strong-privacy perturbation
        # moves signatures further from originals.
        _, cert_tight = dp_perturb_library(lib, epsilon=0.1, delta=1e-5, seed=0)
        _, cert_loose = dp_perturb_library(lib, epsilon=10.0, delta=1e-5, seed=0)
        self.assertGreater(cert_tight.sigma, cert_loose.sigma)

    def test_invalid_epsilon_raises(self):
        lib = _basic_library()
        with self.assertRaises(ValueError):
            dp_perturb_library(lib, epsilon=0.0, delta=1e-5, seed=0)

    def test_invalid_delta_raises(self):
        lib = _basic_library()
        with self.assertRaises(ValueError):
            dp_perturb_library(lib, epsilon=1.0, delta=0.0, seed=0)
        with self.assertRaises(ValueError):
            dp_perturb_library(lib, epsilon=1.0, delta=1.0, seed=0)

    def test_lookups_still_work_at_loose_privacy(self):
        # With large epsilon, noise is small → lookups on the perturbed
        # library should still hit the right programs.
        lib = _basic_library()
        perturbed, _ = dp_perturb_library(
            lib, epsilon=50.0, delta=1e-5, seed=0
        )
        # Direct lookup on near-original signature should succeed.
        hit = perturbed.lookup(torch.tensor([1.0, 0.0, 0.0]))
        self.assertIsNotNone(hit)
        self.assertEqual(hit.program.key(), DiscreteArrayProgram(0, 0, 0, 0, 0.0).key())

    def test_seed_determinism(self):
        lib = _basic_library()
        a, _ = dp_perturb_library(lib, epsilon=1.0, delta=1e-5, seed=42)
        b, _ = dp_perturb_library(lib, epsilon=1.0, delta=1e-5, seed=42)
        for ea, eb in zip(a.entries, b.entries):
            self.assertEqual(ea.signature, eb.signature)


if __name__ == "__main__":
    unittest.main()
