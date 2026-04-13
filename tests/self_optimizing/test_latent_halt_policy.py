"""Tests for latent halt policy and feature encoding."""

from __future__ import annotations

import unittest

from ncpu.self_optimizing import (
    HiddenWorkspace,
    LatentHaltPolicy,
    encode_latent_halt_features,
)


class TestLatentHaltPolicy(unittest.TestCase):
    def test_encode_latent_halt_features_sets_action_flags(self):
        workspace = HiddenWorkspace(
            task_name="demo",
            task_prompt="Write fib(n) in Python.",
            max_generation_attempts=4,
        )
        workspace.generation_attempts = 2
        workspace.status = "running"
        workspace.candidate_solution = "def fib(n): return 0"
        workspace.last_error = "expected 8"
        workspace.latent_state.record_plan("Use recursion.", kind="plan")
        workspace.latent_state.record_verification(
            success=False,
            error="expected 8",
            verification={"status": "failed"},
        )

        vector, summary = encode_latent_halt_features(
            latent_state=workspace.latent_state,
            workspace=workspace,
            verification_success=False,
            verification_error="expected 8",
            allowed_actions=["continue", "fail"],
        )

        self.assertEqual(summary["allowed_actions"], ["continue", "fail"])
        indices = summary["action_flag_indices"]
        self.assertEqual(vector[indices["continue"]], 1.0)
        self.assertEqual(vector[indices["commit"]], 0.0)
        self.assertEqual(vector[indices["fail"]], 1.0)
        self.assertEqual(len(summary["memory_projection"]), 4)
        self.assertGreater(len(summary["memory_features"]), 4)

    def test_policy_prefers_commit_on_success(self):
        policy = LatentHaltPolicy()
        workspace = HiddenWorkspace(
            task_name="demo",
            task_prompt="Return 7.",
            max_generation_attempts=3,
        )
        workspace.generation_attempts = 1
        workspace.candidate_solution = "7"
        workspace.latent_state.record_verification(
            success=True,
            verification={"status": "passed"},
            error=None,
        )

        decision = policy.choose_halt_action(
            workspace=workspace,
            verification_success=True,
            allowed_actions=["commit", "continue"],
            fallback="commit",
        )

        self.assertEqual(decision.action, "commit")
        self.assertGreater(decision.scores["commit"], decision.scores["continue"])

    def test_policy_prefers_fail_when_no_attempts_remain(self):
        policy = LatentHaltPolicy()
        workspace = HiddenWorkspace(
            task_name="demo",
            task_prompt="Return 7.",
            max_generation_attempts=1,
        )
        workspace.generation_attempts = 1
        workspace.candidate_solution = "0"
        workspace.last_error = "expected 7"
        workspace.latent_state.record_verification(
            success=False,
            verification={"status": "failed"},
            error="expected 7",
        )

        decision = policy.choose_halt_action(
            workspace=workspace,
            verification_success=False,
            verification_error="expected 7",
            allowed_actions=["continue", "fail"],
            fallback="fail",
        )

        self.assertEqual(decision.action, "fail")
        self.assertGreater(decision.scores["fail"], decision.scores["continue"])


if __name__ == "__main__":
    unittest.main()
