"""Transform fidelity tests: soft forward vs discrete execution (N2).

The M2 head uses a sigmoid relaxation for the `1{x>0}` indicator transform:

    soft:  sigmoid(x / 0.25)
    hard:  (x > 0).float()

The soft version is a smooth surrogate needed to let gradients flow through
the argmax-selected transform during training. The hard version is what the
discrete `DiscreteArrayProgram` actually executes when the library is hit.

These tests characterize exactly when the two paths diverge — specifically at
x ≈ 0, where the sigmoid is near 0.5 but the step is 0 or 1. They also
confirm that once the softmax over transforms has collapsed to argmax ≈ 1
on the indicator slot, the soft forward and the discrete execution agree to
within a small residual on integer-valued inputs away from zero.
"""

from __future__ import annotations

import unittest

import torch

from ncpu.self_optimizing.array_executable_thought_head import (
    ArrayExecutableThoughtHead,
    ArrayExecutableThoughtHeadConfig,
)
from ncpu.self_optimizing.array_program_library import (
    DiscreteArrayProgram,
    _apply_transform,
)


class TestTransformSoftVsHard(unittest.TestCase):
    def test_identity_transform_exact(self):
        # idx=0 is "x" — soft and hard are identical.
        x = torch.tensor([-3.0, -0.5, 0.0, 0.25, 2.0])
        self.assertTrue(torch.allclose(_apply_transform(x, 0), x))

    def test_square_transform_exact(self):
        x = torch.tensor([-3.0, 0.0, 2.5])
        self.assertTrue(torch.allclose(_apply_transform(x, 1), x * x))

    def test_abs_transform_exact(self):
        x = torch.tensor([-3.0, 0.0, 2.5])
        self.assertTrue(torch.allclose(_apply_transform(x, 2), x.abs()))

    def test_constant_transform_exact(self):
        x = torch.tensor([-3.0, 0.0, 2.5])
        self.assertTrue(torch.allclose(_apply_transform(x, 3), torch.ones_like(x)))

    def test_indicator_matches_hard_on_integer_inputs_away_from_zero(self):
        # The discrete indicator is (x > 0). On integer inputs strictly
        # away from zero (|x| >= 1), the discrete form agrees exactly.
        x = torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
        hard = _apply_transform(x, 4)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        self.assertTrue(torch.allclose(hard, expected))

    def test_indicator_is_zero_at_boundary(self):
        # The discrete 1{x>0} is strictly >0 — x=0 evaluates to 0.
        x = torch.tensor([0.0])
        self.assertEqual(float(_apply_transform(x, 4).item()), 0.0)

    def test_unknown_transform_raises(self):
        with self.assertRaises(ValueError):
            _apply_transform(torch.tensor([1.0]), 99)


class TestSoftVsDiscreteIndicatorGap(unittest.TestCase):
    """Quantify the soft↔hard gap for the indicator transform near zero."""

    def test_sigmoid_surrogate_near_half_at_zero(self):
        # M2 soft forward uses sigmoid(x / 0.25). At x=0 that's exactly 0.5.
        # At |x|=0.25 it's ~0.731 / ~0.269. At |x|>=1 it's ~0.982 / ~0.018.
        x = torch.tensor([0.0, 0.25, -0.25, 1.0, -1.0])
        soft = torch.sigmoid(x / 0.25)
        hard = (x > 0).to(torch.float32)
        gaps = (soft - hard).abs()
        self.assertAlmostEqual(float(soft[0].item()), 0.5, places=6)
        # At zero the sigmoid is exactly 0.5 away from hard=0.
        self.assertAlmostEqual(float(gaps[0].item()), 0.5, places=6)
        # Far from zero the gap is tiny.
        self.assertLess(float(gaps[3].item()), 0.02)
        self.assertLess(float(gaps[4].item()), 0.02)


class TestArgmaxCollapseConvergence(unittest.TestCase):
    """Once softmax over transforms collapses, soft and discrete agree."""

    def test_count_positive_soft_vs_discrete_after_argmax_collapse(self):
        # Build a fresh head; hand-force the transform distribution to
        # one-hot on slot 4 (indicator). Measure soft forward vs discrete
        # execution on integer inputs — they should agree to within ~|L|*0.5
        # (worst case: 0.5 sigmoid error at each of L elements where x==0).
        torch.manual_seed(0)
        config = ArrayExecutableThoughtHeadConfig(
            hidden_dim=4,
            array_max_len=6,
            trace_projection_dim=8,
            trace_hidden_dim=16,
            state_patch_dim=8,
        )
        head = ArrayExecutableThoughtHead(config)

        # Integer inputs strictly away from zero: hard and soft agree to
        # within the residual sigmoid slope.
        arrays = torch.tensor([[1.0, -2.0, 3.0, -1.0, 2.0, -3.0]])
        lengths = torch.tensor([6.0])

        # Force a one-hot discrete program in place of the head.
        distributions = {
            "init": torch.tensor([[1.0, 0.0, 0.0]]),
            "transform": torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
            "reduce": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "post_scale": torch.tensor([[1.0, 0.0, 0.0]]),
            "post_offset": torch.tensor([0.0]),
        }
        discrete = DiscreteArrayProgram.from_soft_distributions(distributions, 0)
        discrete_out = discrete.execute(arrays, lengths).item()
        # Run the head's soft executor with the same forced distributions.
        soft_out = head._execute_batched(arrays, lengths, distributions).item()

        # For inputs strictly away from zero, soft ≈ hard to high precision.
        self.assertAlmostEqual(discrete_out, 3.0, places=4)
        self.assertLess(abs(soft_out - discrete_out), 0.2)

    def test_worst_case_zero_valued_elements_widen_gap(self):
        # When the array contains zeros, each contributes a 0.5 error
        # per sigmoid — documenting the worst-case divergence explicitly.
        torch.manual_seed(0)
        config = ArrayExecutableThoughtHeadConfig(
            hidden_dim=4,
            array_max_len=6,
            trace_projection_dim=8,
            trace_hidden_dim=16,
            state_patch_dim=8,
        )
        head = ArrayExecutableThoughtHead(config)

        arrays = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        lengths = torch.tensor([6.0])

        distributions = {
            "init": torch.tensor([[1.0, 0.0, 0.0]]),
            "transform": torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
            "reduce": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "post_scale": torch.tensor([[1.0, 0.0, 0.0]]),
            "post_offset": torch.tensor([0.0]),
        }
        discrete = DiscreteArrayProgram.from_soft_distributions(distributions, 0)
        discrete_out = discrete.execute(arrays, lengths).item()
        soft_out = head._execute_batched(arrays, lengths, distributions).item()

        # Discrete returns 0 (nothing is strictly > 0).
        self.assertEqual(discrete_out, 0.0)
        # Soft returns sum_i active_i * sigmoid(0/0.25). The length mask is
        # ALSO a sigmoid with boundary softness, so the final element gets
        # weight ~0.84 not 1.0. Expected gap ≈ 5 * 0.5 + 1 * ~0.42 ≈ 2.92.
        self.assertGreater(soft_out, 2.5)
        self.assertLess(soft_out, 3.1)

        # This ~3.0 gap is exactly what the convergence_gap_threshold is
        # designed to detect and prevent from caching — a program that
        # disagrees with its soft parent by this much should NEVER make it
        # into the library.
        self.assertGreater(abs(soft_out - discrete_out), 2.5)


if __name__ == "__main__":
    unittest.main()
