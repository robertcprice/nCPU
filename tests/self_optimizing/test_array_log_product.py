"""Log-product + exp-post-scale tests (N2-ext).

Verifies the new transform_idx=5 (log|x|+eps) and post_scale_idx=2 (exp(acc))
give a numerically stable product-magnitude path without overflow on large
|values| × long arrays. All three backends (Python, Rust CPU, Metal GPU)
must agree to within log-epsilon drift.
"""

from __future__ import annotations

import unittest

import torch

from ncpu.self_optimizing.array_executable_thought_head import (
    _ELEM_TRANSFORMS,
    _POST_SCALES,
)
from ncpu.self_optimizing.array_program_library import (
    DiscreteArrayProgram,
    get_native_backend,
    reset_native_backend_cache,
)


class TestExtendedSchema(unittest.TestCase):
    def test_transforms_table_has_log_abs(self):
        self.assertEqual(_ELEM_TRANSFORMS[5], "log|x|")

    def test_post_scales_table_has_exp(self):
        self.assertEqual(_POST_SCALES[2], "exp(acc)")


class TestLogProductProgram(unittest.TestCase):
    def test_python_log_product_recovers_magnitude(self):
        program = DiscreteArrayProgram(
            init_idx=0,
            transform_idx=5,
            reduce_idx=0,
            post_scale_idx=2,
            offset=0.0,
        )
        arrays = torch.tensor([[2.0, -3.0, 4.0, 0.0, 0.0]])
        lengths = torch.tensor([3.0])
        out = program.execute(arrays, lengths)
        # |2 * -3 * 4| = 24; log-eps drift is well under 0.01 for |x|>=1.
        self.assertAlmostEqual(float(out.item()), 24.0, places=1)

    def test_python_log_product_handles_zero(self):
        program = DiscreteArrayProgram(0, 5, 0, 2, 0.0)
        arrays = torch.tensor([[2.0, 0.0, 4.0]])
        lengths = torch.tensor([3.0])
        out = program.execute(arrays, lengths)
        self.assertTrue(torch.isfinite(out).all())
        self.assertLess(float(out.item()), 0.01)

    def test_exp_post_scale_clamps_large_acc(self):
        # init=1, transform=x*x, reduce=+ => acc = 1 + sum(x_i^2) which
        # blows up for |x_i|=100, L=4. post_scale=exp with clamp must stay finite.
        program = DiscreteArrayProgram(1, 1, 0, 2, 0.0)
        arrays = torch.tensor([[100.0, 100.0, 100.0, 100.0]])
        lengths = torch.tensor([4.0])
        out = program.execute(arrays, lengths)
        self.assertTrue(torch.isfinite(out).all())

    def test_render_log_product_program(self):
        program = DiscreteArrayProgram(0, 5, 0, 2, 0.0)
        text = program.render()
        self.assertIn("ln(|arr[i]| + eps)", text)
        self.assertIn("exp(clamp(acc, -30, 30))", text)


class TestNativeBackendAgreesOnLogProduct(unittest.TestCase):
    def setUp(self):
        reset_native_backend_cache()
        self._has_native = get_native_backend() is not None

    def test_native_matches_python_on_log_product(self):
        program = DiscreteArrayProgram(0, 5, 0, 2, 0.0)
        arrays = torch.tensor(
            [
                [2.0, -3.0, 4.0, 0.0, 0.0],
                [5.0, 2.0, 0.0, 0.0, 0.0],
            ]
        )
        lengths = torch.tensor([3.0, 2.0])
        python_out = program.execute(arrays, lengths)
        native_out = program.execute_native(arrays, lengths)
        self.assertTrue(
            torch.allclose(python_out, native_out, atol=1e-3),
            msg=f"python={python_out}, native={native_out}",
        )

    def test_metal_matches_python_on_log_product(self):
        if not self._has_native:
            self.skipTest("native backend unavailable")
        program = DiscreteArrayProgram(0, 5, 0, 2, 0.0)
        arrays = torch.tensor([[2.0, -3.0, 4.0, 0.0, 0.0]])
        lengths = torch.tensor([3.0])
        python_out = program.execute(arrays, lengths)
        try:
            metal_out = program.execute_native(
                arrays, lengths, backend="metal"
            )
        except RuntimeError:
            self.skipTest("metal device unavailable")
        self.assertAlmostEqual(
            float(python_out.item()),
            float(metal_out.item()),
            places=1,
        )


class TestJSONRoundTripWithNewSchema(unittest.TestCase):
    def test_round_trip_preserves_new_indices(self):
        program = DiscreteArrayProgram(2, 5, 1, 2, -0.25)
        payload = program.to_dict()
        self.assertEqual(payload["transform_idx"], 5)
        self.assertEqual(payload["post_scale_idx"], 2)
        restored = DiscreteArrayProgram.from_dict(payload)
        self.assertEqual(restored.key(), program.key())
        self.assertAlmostEqual(restored.offset, program.offset)


if __name__ == "__main__":
    unittest.main()
