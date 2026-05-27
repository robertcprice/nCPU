"""Post-offset bounds verifier tests (N4b)."""

from __future__ import annotations

import unittest

from ncpu.self_optimizing.array_program_library import DiscreteArrayProgram
from ncpu.self_optimizing.program_verifier import (
    RISK_SAFE,
    RISK_WARN,
    RangeBound,
    VerifierConfig,
    verify_program,
)


class TestOffsetMagnitudeClaim(unittest.TestCase):
    def test_tiny_offset_flagged_safe(self):
        program = DiscreteArrayProgram(0, 0, 0, 0, 0.1)  # offset 0.1
        report = verify_program(
            program,
            config=VerifierConfig(
                input_bound=RangeBound(-5.0, 5.0), max_length=8
            ),
        )
        offset_claim = next(
            c for c in report.claims if c.name == "offset_magnitude"
        )
        self.assertTrue(offset_claim.verdict)
        self.assertEqual(offset_claim.risk_level, RISK_SAFE)

    def test_dominant_offset_flagged_warn(self):
        # Program: reduce=min over transform=1 (constant) → pre-offset bound
        # has width 0 (always 1.0, assuming init=1). Offset of +100 dominates.
        program = DiscreteArrayProgram(1, 3, 3, 0, 100.0)
        report = verify_program(
            program,
            config=VerifierConfig(
                input_bound=RangeBound(-1.0, 1.0), max_length=4
            ),
        )
        offset_claim = next(
            c for c in report.claims if c.name == "offset_magnitude"
        )
        self.assertFalse(offset_claim.verdict)
        self.assertEqual(offset_claim.risk_level, RISK_WARN)

    def test_post_offset_bound_includes_offset(self):
        program = DiscreteArrayProgram(0, 0, 0, 0, 50.0)
        report = verify_program(
            program,
            config=VerifierConfig(
                input_bound=RangeBound(-2.0, 2.0), max_length=4
            ),
        )
        # Pre-offset range is [-8, 8]; post-offset should be [42, 58].
        self.assertAlmostEqual(report.output_bound.lower, 42.0, places=5)
        self.assertAlmostEqual(report.output_bound.upper, 58.0, places=5)

    def test_negative_offset_bound(self):
        program = DiscreteArrayProgram(0, 2, 0, 0, -3.0)  # sum of abs(x) - 3
        report = verify_program(
            program,
            config=VerifierConfig(
                input_bound=RangeBound(-4.0, 4.0), max_length=5
            ),
        )
        # Per-element: [0, 4]; sum: [0, 20]; post-offset: [-3, 17].
        self.assertAlmostEqual(report.output_bound.lower, -3.0, places=5)
        self.assertAlmostEqual(report.output_bound.upper, 17.0, places=5)


if __name__ == "__main__":
    unittest.main()
