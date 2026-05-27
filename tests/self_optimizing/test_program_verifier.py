"""Tests for the static program verifier (NV1)."""

from __future__ import annotations

import unittest

import torch

from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    DiscreteArrayProgram,
)
from ncpu.self_optimizing.program_verifier import (
    RISK_HIGH,
    RISK_SAFE,
    RISK_WARN,
    RangeBound,
    VerifierConfig,
    verify_library,
    verify_program,
)


class TestRangeBound(unittest.TestCase):
    def test_width_is_nonnegative(self):
        self.assertEqual(RangeBound(-3.0, 5.0).width(), 8.0)


class TestSumProgramVerification(unittest.TestCase):
    def test_sum_program_is_safe_and_bounded(self):
        program = DiscreteArrayProgram(0, 0, 0, 0, 0.0)  # sum
        report = verify_program(
            program,
            config=VerifierConfig(
                input_bound=RangeBound(-3.0, 3.0), max_length=8
            ),
        )
        self.assertTrue(report.overall_safe)
        self.assertEqual(report.worst_risk, RISK_SAFE)
        # Output ∈ [-3*8, 3*8] = [-24, 24]
        self.assertAlmostEqual(report.output_bound.lower, -24.0, places=5)
        self.assertAlmostEqual(report.output_bound.upper, 24.0, places=5)
        names = [c.name for c in report.claims]
        self.assertIn("termination", names)
        self.assertIn("division_safety", names)
        self.assertIn("overflow_risk", names)

    def test_sum_flags_large_sums_as_warn(self):
        program = DiscreteArrayProgram(0, 0, 0, 0, 0.0)  # sum
        report = verify_program(
            program,
            config=VerifierConfig(
                input_bound=RangeBound(-1e4, 1e4),
                max_length=100,
                overflow_threshold=1e6,
            ),
        )
        # Output magnitude = 1e4 * 100 = 1e6 → at threshold, warn.
        overflow_claim = next(
            c for c in report.claims if c.name == "overflow_risk"
        )
        self.assertIn(overflow_claim.risk_level, (RISK_WARN, RISK_HIGH))


class TestProductStabilityCheck(unittest.TestCase):
    def test_naive_product_flagged_as_warn(self):
        # reduce=*, transform=x, long array → warn.
        program = DiscreteArrayProgram(1, 0, 1, 0, 0.0)
        report = verify_program(
            program,
            config=VerifierConfig(
                input_bound=RangeBound(-3.0, 3.0), max_length=8
            ),
        )
        product_claims = [
            c for c in report.claims if c.name == "product_stability"
        ]
        self.assertEqual(len(product_claims), 1)
        self.assertEqual(product_claims[0].risk_level, RISK_WARN)
        self.assertFalse(report.overall_safe)

    def test_log_domain_product_not_flagged(self):
        # transform=log|x|, reduce=+, post_scale=exp — stable product path.
        program = DiscreteArrayProgram(0, 5, 0, 2, 0.0)
        report = verify_program(
            program,
            config=VerifierConfig(
                input_bound=RangeBound(-3.0, 3.0), max_length=8
            ),
        )
        product_claims = [
            c for c in report.claims if c.name == "product_stability"
        ]
        self.assertEqual(product_claims, [])
        exp_claims = [c for c in report.claims if c.name == "exp_clamp"]
        self.assertEqual(len(exp_claims), 1)
        self.assertEqual(exp_claims[0].risk_level, RISK_SAFE)

    def test_short_array_product_not_flagged(self):
        # Short arrays don't trip the warning even with reduce=*.
        program = DiscreteArrayProgram(1, 0, 1, 0, 0.0)
        report = verify_program(
            program,
            config=VerifierConfig(
                input_bound=RangeBound(-2.0, 2.0), max_length=3
            ),
        )
        product_claims = [
            c for c in report.claims if c.name == "product_stability"
        ]
        self.assertEqual(product_claims, [])


class TestTransformBounds(unittest.TestCase):
    def test_indicator_transform_bounds_to_0_1(self):
        program = DiscreteArrayProgram(0, 4, 0, 0, 0.0)  # count_positive
        report = verify_program(
            program,
            config=VerifierConfig(
                input_bound=RangeBound(-5.0, 5.0), max_length=10
            ),
        )
        # Each element contributes 0 or 1; sum in [0, 10].
        self.assertAlmostEqual(report.output_bound.lower, 0.0, places=5)
        self.assertAlmostEqual(report.output_bound.upper, 10.0, places=5)

    def test_square_transform_bounds_nonnegative(self):
        program = DiscreteArrayProgram(0, 1, 0, 0, 0.0)  # sum of squares
        report = verify_program(
            program,
            config=VerifierConfig(
                input_bound=RangeBound(-4.0, 4.0), max_length=5
            ),
        )
        # Per-elem bound: [0, 16]; sum over 5: [0, 80].
        self.assertAlmostEqual(report.output_bound.lower, 0.0, places=5)
        self.assertAlmostEqual(report.output_bound.upper, 80.0, places=5)

    def test_max_reduce_bound(self):
        # init=-large, reduce=max over x; bound = max(-20, 5) = 5.
        program = DiscreteArrayProgram(2, 0, 2, 0, 0.0)
        report = verify_program(
            program,
            config=VerifierConfig(
                input_bound=RangeBound(-5.0, 5.0), max_length=10
            ),
        )
        self.assertEqual(report.output_bound.upper, 5.0)


class TestDivisionSafety(unittest.TestCase):
    def test_acc_over_len_is_certified_safe(self):
        program = DiscreteArrayProgram(0, 0, 0, 1, 0.0)  # mean
        report = verify_program(program)
        div = next(c for c in report.claims if c.name == "division_safety")
        self.assertTrue(div.verdict)
        self.assertEqual(div.risk_level, RISK_SAFE)

    def test_no_division_in_plain_sum(self):
        program = DiscreteArrayProgram(0, 0, 0, 0, 0.0)
        report = verify_program(program)
        div = next(c for c in report.claims if c.name == "division_safety")
        self.assertTrue(div.verdict)


class TestVerifyLibrary(unittest.TestCase):
    def test_verifies_all_entries(self):
        lib = ArrayProgramLibrary()
        lib.record(
            torch.tensor([1.0, 0.0, 0.0]),
            DiscreteArrayProgram(0, 0, 0, 0, 0.0),
            task_name="sum",
        )
        lib.record(
            torch.tensor([0.0, 1.0, 0.0]),
            DiscreteArrayProgram(1, 0, 1, 0, 0.0),
            task_name="naive_product",
        )
        reports = verify_library(
            lib.entries,
            config=VerifierConfig(
                input_bound=RangeBound(-3.0, 3.0), max_length=8
            ),
        )
        self.assertEqual(len(reports), 2)
        # One entry is safe; the other should flag product instability.
        safe_count = sum(1 for r in reports if r["overall_safe"])
        self.assertEqual(safe_count, 1)
        tasks = [r["task_name"] for r in reports]
        self.assertEqual(set(tasks), {"sum", "naive_product"})


if __name__ == "__main__":
    unittest.main()
