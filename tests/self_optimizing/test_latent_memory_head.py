"""Tests for learned latent memory feature encoding and runtime."""

from __future__ import annotations

import unittest

from ncpu.self_optimizing import (
    HiddenWorkspace,
    LatentControllerState,
    LatentMemoryUpdater,
    encode_latent_memory_features,
)


class TestLatentMemoryHead(unittest.TestCase):
    def test_encode_latent_memory_features_marks_event_and_memory_projection(self):
        workspace = HiddenWorkspace(
            task_name="demo",
            task_prompt="Write fib(n) in Python.",
            max_generation_attempts=4,
        )
        workspace.generation_attempts = 1
        workspace.last_error = "expected 8"
        workspace.latent_state.record_plan("Use recursion.", kind="plan")
        workspace.latent_state.record_verification(
            success=False,
            error="expected 8",
            verification={"status": "failed"},
        )

        vector, summary = encode_latent_memory_features(
            latent_state=workspace.latent_state,
            workspace=workspace,
            event_kind="verify",
            response_text="expected 8",
            error_text="expected 8",
            success=False,
        )

        self.assertGreater(len(vector), 32)
        self.assertEqual(summary["event_kind"], "verify")
        self.assertEqual(vector[summary["event_flag_indices"]["verify"]], 1.0)
        self.assertEqual(len(summary["memory_projection"]), 4)
        self.assertGreater(len(summary["memory_features"]), 4)

    def test_updater_without_head_returns_zero_delta(self):
        workspace = HiddenWorkspace(
            task_name="demo",
            task_prompt="Return 7.",
            max_generation_attempts=1,
        )
        updater = LatentMemoryUpdater()
        delta, summary = updater.build_memory_delta(
            latent_state=workspace.latent_state,
            workspace=workspace,
            event_kind="write",
            response_text="7",
            success=True,
        )
        self.assertEqual(len(delta), updater.config.output_dim)
        self.assertTrue(all(value == 0.0 for value in delta))
        self.assertEqual(summary["event_kind"], "write")


if __name__ == "__main__":
    unittest.main()
