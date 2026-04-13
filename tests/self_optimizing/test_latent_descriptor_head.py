"""Tests for learned latent descriptor feature encoding and runtime."""

from __future__ import annotations

import unittest

from ncpu.self_optimizing import (
    LatentControllerState,
    LatentDescriptorGenerator,
    encode_latent_descriptor_features,
)


class TestLatentDescriptorHead(unittest.TestCase):
    def test_encode_latent_descriptor_features_marks_update_kind(self):
        state = LatentControllerState(
            hidden_plan="Use recursion.",
            last_failure_summary="expected 8",
            failure_patterns=["expected 8"],
            verification_failures=1,
            confidence=0.2,
        )
        vector, summary = encode_latent_descriptor_features(
            latent_state=state,
            update_kind="verify_failure_descriptor",
            error_text="expected 8",
            candidate_text="def fib(n): return 0",
        )

        self.assertGreater(len(vector), 24)
        self.assertEqual(summary["update_kind"], "verify_failure_descriptor")
        self.assertEqual(vector[summary["event_flag_indices"]["verify_failure"]], 1.0)
        self.assertGreater(len(summary["memory_features"]), 4)

    def test_generator_without_head_returns_zero_projection(self):
        state = LatentControllerState(hidden_plan="Use recursion.")
        generator = LatentDescriptorGenerator()
        projection, summary = generator.build_projection(
            latent_state=state,
            update_kind="plan_descriptor",
        )
        self.assertEqual(len(projection), generator.config.output_dim)
        self.assertTrue(all(value == 0.0 for value in projection))
        self.assertEqual(summary["update_kind"], "plan_descriptor")


if __name__ == "__main__":
    unittest.main()
