"""Tests for recurrent latent memory in controller state."""

from __future__ import annotations

import unittest

from ncpu.self_optimizing import LatentControllerState


class TestLatentControllerState(unittest.TestCase):
    def test_recurrent_memory_updates_across_hidden_events(self):
        state = LatentControllerState()

        baseline = list(state.memory_vector)
        state.record_action("think")
        state.record_plan("Use recursion with correct base cases.", kind="plan")
        state.record_candidate("def fib(n): return 0")
        state.record_verification(
            success=False,
            error="expected 8",
            verification={"status": "failed"},
        )

        self.assertEqual(len(state.memory_vector), 16)
        self.assertNotEqual(state.memory_vector, baseline)
        self.assertGreaterEqual(state.memory_updates, 4)
        self.assertEqual(len(state.memory_projection(width=4)), 4)

    def test_prompt_summary_mentions_latent_memory(self):
        state = LatentControllerState()
        state.record_plan("Use recursion.", kind="plan")
        summary = state.to_prompt_summary()
        self.assertIn("latent_memory:", summary)


if __name__ == "__main__":
    unittest.main()
