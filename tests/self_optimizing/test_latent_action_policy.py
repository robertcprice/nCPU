"""Tests for latent action policy and feature encoding."""

from __future__ import annotations

import unittest

from ncpu.self_optimizing import (
    HiddenWorkspace,
    LatentActionPolicy,
    encode_latent_action_features,
)


class TestLatentActionPolicy(unittest.TestCase):
    def test_encode_latent_action_features_sets_allowed_action_flags(self):
        workspace = HiddenWorkspace(
            task_name="demo",
            task_prompt="Write fib(n) in Python.",
            max_generation_attempts=4,
        )
        workspace.generation_attempts = 1
        workspace.last_error = "expected 8"
        workspace.candidate_solution = "def fib(n): return 0"
        workspace.latent_state.record_plan("Use recursion.", kind="plan")
        workspace.latent_state.record_verification(
            success=False,
            error="expected 8",
            verification={"status": "failed"},
        )

        vector, summary = encode_latent_action_features(
            latent_state=workspace.latent_state,
            workspace=workspace,
            allowed_actions=["patch", "think", "fail"],
        )

        self.assertGreater(len(vector), 24)
        self.assertEqual(summary["allowed_actions"], ["patch", "think", "fail"])
        indices = summary["action_flag_indices"]
        self.assertEqual(vector[indices["think"]], 1.0)
        self.assertEqual(vector[indices["write"]], 0.0)
        self.assertEqual(vector[indices["patch"]], 1.0)
        self.assertEqual(vector[indices["fail"]], 1.0)
        self.assertEqual(len(summary["memory_projection"]), 4)
        self.assertGreater(len(summary["memory_features"]), 4)

    def test_policy_prefers_think_then_patch(self):
        policy = LatentActionPolicy()
        workspace = HiddenWorkspace(
            task_name="demo",
            task_prompt="Write fib(n) in Python.",
            max_generation_attempts=3,
        )

        initial = policy.choose_action(
            workspace=workspace,
            allowed_actions=["think", "write"],
            fallback="write",
        )
        self.assertEqual(initial.action, "think")
        self.assertEqual(initial.source, "latent_action_policy")

        workspace.generation_attempts = 1
        workspace.latent_state.record_plan("Use recursion.", kind="plan")
        workspace.latent_state.record_verification(
            success=False,
            error="expected 8",
            verification={"status": "failed"},
        )
        workspace.last_error = "expected 8"

        repair = policy.choose_action(
            workspace=workspace,
            allowed_actions=["patch", "think", "fail"],
            fallback="patch",
        )
        self.assertEqual(repair.action, "patch")
        self.assertGreater(repair.scores["patch"], repair.scores["think"])


if __name__ == "__main__":
    unittest.main()
