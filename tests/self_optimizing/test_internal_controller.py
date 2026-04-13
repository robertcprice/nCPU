"""Tests for the hidden buffered inference controller."""

import json
import tempfile
import unittest
from pathlib import Path

from ncpu.self_optimizing import (
    BufferedInternalController,
    InternalControllerConfig,
    InternalDeliberationTask,
    LatentActionPolicy,
    LatentHaltPolicy,
    TrajectoryLogger,
)


FIB_TESTS = [
    {"input": {"n": 0}, "expected": 0},
    {"input": {"n": 1}, "expected": 1},
    {"input": {"n": 6}, "expected": 8},
]


class TestBufferedInternalController(unittest.TestCase):
    """Validate hidden planning and repair-before-commit behavior."""

    def test_controller_repairs_code_before_commit(self):
        def fake_provider(prompt: str) -> str:
            if "Think privately" in prompt:
                return "Use recursion with the correct base cases."
            if "repairing a hidden candidate" in prompt.lower():
                return (
                    "def fib(n):\n"
                    "    if n <= 1:\n"
                    "        return n\n"
                    "    return fib(n - 1) + fib(n - 2)\n"
                )
            return "def fib(n):\n    return 0\n"

        controller = BufferedInternalController(
            llm_provider=fake_provider,
            config=InternalControllerConfig(max_generation_attempts=3),
        )
        task = InternalDeliberationTask(
            name="fibonacci",
            prompt="Write fib(n) in Python.",
            test_cases=FIB_TESTS,
        )

        workspace = controller.run_task(task)

        self.assertEqual(workspace.status, "committed")
        self.assertTrue(workspace.committed_verified)
        self.assertIn("return fib(n - 1)", workspace.committed_output or "")
        self.assertEqual(workspace.generation_attempts, 2)
        self.assertEqual([step.action for step in workspace.steps[:5]], ["think", "write", "verify", "patch", "verify"])

    def test_controller_fails_without_unverified_commit(self):
        def fake_provider(_prompt: str) -> str:
            return "def fib(n):\n    return -1\n"

        controller = BufferedInternalController(
            llm_provider=fake_provider,
            config=InternalControllerConfig(max_generation_attempts=2, plan_before_generate=False),
        )
        task = InternalDeliberationTask(
            name="fibonacci",
            prompt="Write fib(n) in Python.",
            test_cases=FIB_TESTS,
        )

        workspace = controller.run_task(task)

        self.assertEqual(workspace.status, "failed")
        self.assertIsNone(workspace.committed_output)
        self.assertIn("expected", workspace.last_error or "")

    def test_trajectory_logger_records_hidden_steps(self):
        def fake_provider(prompt: str) -> str:
            if "Think privately" in prompt:
                return "Return the exact integer."
            return "7"

        def verify(candidate: str) -> tuple[bool, str]:
            return candidate.strip() == "7", "expected 7"

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "trajectory.jsonl"
            controller = BufferedInternalController(
                llm_provider=fake_provider,
                config=InternalControllerConfig(max_generation_attempts=1),
                trajectory_logger=TrajectoryLogger(str(log_path)),
            )
            task = InternalDeliberationTask(
                name="reasoning",
                prompt="Return the integer 7.",
                verifier=verify,
                category="reasoning",
                response_format="the final integer only",
            )

            workspace = controller.run_task(task)

            self.assertEqual(workspace.status, "committed")
            events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[0]["event"], "workspace_initialized")
            self.assertEqual(events[-1]["event"], "workspace_committed")
            self.assertIn("workspace_step", [event["event"] for event in events])
            self.assertEqual(events[-1]["committed_output"], "7")

    def test_action_provider_can_skip_initial_think(self):
        def fake_provider(_prompt: str) -> str:
            return (
                "def fib(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    return fib(n - 1) + fib(n - 2)\n"
            )

        controller = BufferedInternalController(
            llm_provider=fake_provider,
            action_provider=lambda _prompt: "write",
            config=InternalControllerConfig(max_generation_attempts=2),
        )
        task = InternalDeliberationTask(
            name="fibonacci",
            prompt="Write fib(n) in Python.",
            test_cases=FIB_TESTS,
        )

        workspace = controller.run_task(task)

        self.assertEqual(workspace.status, "committed")
        self.assertEqual([step.action for step in workspace.steps], ["write", "verify", "commit"])
        self.assertEqual(workspace.steps[0].metadata["policy_selected_action"], "write")

    def test_action_provider_can_fail_fast_after_bad_attempt(self):
        def fake_provider(_prompt: str) -> str:
            return "def fib(n):\n    return 0\n"

        decisions = iter(["write", "fail"])
        controller = BufferedInternalController(
            llm_provider=fake_provider,
            action_provider=lambda _prompt: next(decisions),
            config=InternalControllerConfig(max_generation_attempts=3, plan_before_generate=False),
        )
        task = InternalDeliberationTask(
            name="fibonacci",
            prompt="Write fib(n) in Python.",
            test_cases=FIB_TESTS,
        )

        workspace = controller.run_task(task)

        self.assertEqual(workspace.status, "failed")
        self.assertEqual(workspace.generation_attempts, 1)
        self.assertIn("terminated", workspace.last_error or "")
        self.assertEqual([step.action for step in workspace.steps], ["write", "verify", "fail"])

    def test_action_provider_can_rethink_before_patch(self):
        def fake_provider(prompt: str) -> str:
            if "repair attempt" in prompt.lower():
                return "The fix is to use the Fibonacci recurrence."
            if "repairing a hidden candidate" in prompt.lower():
                return (
                    "def fib(n):\n"
                    "    if n <= 1:\n"
                    "        return n\n"
                    "    return fib(n - 1) + fib(n - 2)\n"
                )
            return "def fib(n):\n    return 0\n"

        decisions = iter(["write", "think"])
        controller = BufferedInternalController(
            llm_provider=fake_provider,
            action_provider=lambda _prompt: next(decisions),
            config=InternalControllerConfig(max_generation_attempts=3, plan_before_generate=False),
        )
        task = InternalDeliberationTask(
            name="fibonacci",
            prompt="Write fib(n) in Python.",
            test_cases=FIB_TESTS,
        )

        workspace = controller.run_task(task)

        self.assertEqual(workspace.status, "committed")
        self.assertEqual(
            [step.action for step in workspace.steps[:5]],
            ["write", "verify", "think", "patch", "verify"],
        )
        self.assertEqual(workspace.steps[2].metadata["policy_selected_action"], "think")
        self.assertEqual(workspace.steps[3].metadata["policy_selected_action"], "patch")

    def test_latent_action_policy_drives_controller_without_prompt_policy(self):
        def fake_provider(prompt: str) -> str:
            if "Think privately" in prompt:
                return "Use the Fibonacci recurrence with correct base cases."
            if "repairing a hidden candidate" in prompt.lower():
                return (
                    "def fib(n):\n"
                    "    if n <= 1:\n"
                    "        return n\n"
                    "    return fib(n - 1) + fib(n - 2)\n"
                )
            return "def fib(n):\n    return 0\n"

        controller = BufferedInternalController(
            llm_provider=fake_provider,
            latent_action_policy=LatentActionPolicy(),
            config=InternalControllerConfig(max_generation_attempts=3),
        )
        task = InternalDeliberationTask(
            name="fibonacci",
            prompt="Write fib(n) in Python.",
            test_cases=FIB_TESTS,
        )

        workspace = controller.run_task(task)

        self.assertEqual(workspace.status, "committed")
        self.assertEqual([step.action for step in workspace.steps[:5]], ["think", "write", "verify", "patch", "verify"])
        self.assertEqual(workspace.steps[0].metadata["policy_source"], "latent_action_policy")
        self.assertEqual(workspace.steps[1].metadata["policy_source"], "implicit_after_think")
        self.assertEqual(workspace.steps[3].metadata["policy_source"], "latent_action_policy")

    def test_latent_halt_policy_can_continue_after_verified_success_then_commit(self):
        class ContinueThenCommitHaltPolicy:
            def __init__(self):
                self.calls = 0

            def choose_halt_action(self, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    return {
                        "action": "continue",
                        "source": "latent_halt_policy",
                        "scores": {"continue": 0.9, "commit": 0.1},
                        "confidence": 0.9,
                        "feature_summary": {"phase": "verified_refine"},
                    }
                return {
                    "action": "commit",
                    "source": "latent_halt_policy",
                    "scores": {"commit": 0.9, "continue": 0.1},
                    "confidence": 0.9,
                    "feature_summary": {"phase": "final_commit"},
                }

        def fake_provider(prompt: str) -> str:
            if "refining a hidden candidate" in prompt.lower():
                return (
                    "def fib(n):\n"
                    "    if n <= 1:\n"
                    "        return n\n"
                    "    return fib(n - 1) + fib(n - 2)\n"
                )
            return (
                "def fib(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    return fib(n - 1) + fib(n - 2)\n"
            )

        controller = BufferedInternalController(
            llm_provider=fake_provider,
            latent_halt_policy=ContinueThenCommitHaltPolicy(),
            config=InternalControllerConfig(max_generation_attempts=3, plan_before_generate=False),
        )
        task = InternalDeliberationTask(
            name="fibonacci",
            prompt="Write fib(n) in Python.",
            test_cases=FIB_TESTS,
        )

        workspace = controller.run_task(task)

        self.assertEqual(workspace.status, "committed")
        self.assertEqual(
            [step.action for step in workspace.steps],
            ["write", "verify", "patch", "verify", "commit"],
        )
        self.assertEqual(workspace.steps[-1].metadata["halt_source"], "latent_halt_policy")

    def test_latent_halt_policy_can_fail_after_verification_failure(self):
        class FailImmediatelyHaltPolicy:
            def choose_halt_action(self, **_kwargs):
                return {
                    "action": "fail",
                    "source": "latent_halt_policy",
                    "scores": {"fail": 0.95, "continue": 0.05},
                    "confidence": 0.95,
                    "feature_summary": {"phase": "abort"},
                }

        def fake_provider(_prompt: str) -> str:
            return "def fib(n):\n    return 0\n"

        controller = BufferedInternalController(
            llm_provider=fake_provider,
            latent_halt_policy=FailImmediatelyHaltPolicy(),
            config=InternalControllerConfig(max_generation_attempts=3, plan_before_generate=False),
        )
        task = InternalDeliberationTask(
            name="fibonacci",
            prompt="Write fib(n) in Python.",
            test_cases=FIB_TESTS,
        )

        workspace = controller.run_task(task)

        self.assertEqual(workspace.status, "failed")
        self.assertEqual([step.action for step in workspace.steps], ["write", "verify", "fail"])
        self.assertEqual(workspace.steps[-1].metadata["halt_source"], "latent_halt_policy")

    def test_controller_applies_task_local_fast_weight_updates(self):
        class FakeFastWeightProvider:
            def __init__(self):
                self.begin_calls = []
                self.update_calls = []
                self.end_calls = 0

            def begin_task(self, task_name: str, task_prompt: str) -> dict:
                self.begin_calls.append((task_name, task_prompt))
                return {"enabled": True, "updated_modules": ["q_proj"]}

            def apply_self_update(self, *, prompt: str, target_text: str, update_kind: str, task_name: str):
                self.update_calls.append(
                    {
                        "prompt": prompt,
                        "target_text": target_text,
                        "update_kind": update_kind,
                        "task_name": task_name,
                    }
                )
                return {
                    "success": True,
                    "kind": update_kind,
                    "updated_modules": ["q_proj"],
                    "task_name": task_name,
                    "steps": 1,
                    "target_tokens": 12,
                }

            def end_task(self) -> dict:
                self.end_calls += 1
                return {"task_name": "fibonacci", "update_count": len(self.update_calls)}

            def __call__(self, prompt: str) -> str:
                if "Think privately" in prompt:
                    return "Use the Fibonacci recurrence."
                return (
                    "def fib(n):\n"
                    "    if n <= 1:\n"
                    "        return n\n"
                    "    return fib(n - 1) + fib(n - 2)\n"
                )

        provider = FakeFastWeightProvider()
        controller = BufferedInternalController(
            llm_provider=provider,
            config=InternalControllerConfig(max_generation_attempts=2),
        )
        task = InternalDeliberationTask(
            name="fibonacci",
            prompt="Write fib(n) in Python.",
            test_cases=FIB_TESTS,
        )

        workspace = controller.run_task(task)

        self.assertEqual(workspace.status, "committed")
        self.assertEqual(provider.begin_calls, [("fibonacci", "Write fib(n) in Python.")])
        self.assertEqual(provider.end_calls, 1)
        self.assertEqual(len(provider.update_calls), 1)
        self.assertEqual(provider.update_calls[0]["update_kind"], "plan")
        self.assertEqual(
            [step.action for step in workspace.steps],
            ["think", "fast_weight_update", "write", "verify", "commit"],
        )
        self.assertTrue(workspace.steps[1].success)
        self.assertEqual(workspace.metadata["fast_weights"]["updated_modules"], ["q_proj"])
        self.assertEqual(workspace.metadata["fast_weights_last_task"]["update_count"], 1)

    def test_controller_applies_failure_state_to_fast_weights_during_repair(self):
        class FakeFastWeightProvider:
            def __init__(self):
                self.update_calls = []

            def begin_task(self, task_name: str, task_prompt: str) -> dict:
                return {"enabled": True, "task_name": task_name}

            def apply_self_update(self, *, prompt: str, target_text: str, update_kind: str, task_name: str):
                self.update_calls.append(
                    {
                        "prompt": prompt,
                        "target_text": target_text,
                        "update_kind": update_kind,
                        "task_name": task_name,
                    }
                )
                return {
                    "success": True,
                    "kind": update_kind,
                    "updated_modules": ["q_proj"],
                    "task_name": task_name,
                    "steps": 1,
                    "target_tokens": 8,
                }

            def end_task(self) -> dict:
                return {"update_count": len(self.update_calls)}

            def __call__(self, prompt: str) -> str:
                if "Think privately" in prompt:
                    return "Use the Fibonacci recurrence."
                if "repairing a hidden candidate" in prompt.lower():
                    return (
                        "def fib(n):\n"
                        "    if n <= 1:\n"
                        "        return n\n"
                        "    return fib(n - 1) + fib(n - 2)\n"
                    )
                return "def fib(n):\n    return 0\n"

        provider = FakeFastWeightProvider()
        controller = BufferedInternalController(
            llm_provider=provider,
            config=InternalControllerConfig(
                max_generation_attempts=2,
                max_fast_weight_updates_per_task=3,
            ),
        )
        task = InternalDeliberationTask(
            name="fibonacci",
            prompt="Write fib(n) in Python.",
            test_cases=FIB_TESTS,
        )

        workspace = controller.run_task(task)

        self.assertEqual(workspace.status, "committed")
        self.assertEqual([call["update_kind"] for call in provider.update_calls], ["plan", "verify_failure"])
        self.assertIn("Failure summary", provider.update_calls[1]["prompt"])
        self.assertIn("Repair constraints", provider.update_calls[1]["target_text"])
        self.assertEqual(
            [step.action for step in workspace.steps[:7]],
            ["think", "fast_weight_update", "write", "verify", "fast_weight_update", "patch", "verify"],
        )
        self.assertEqual(workspace.latent_state.verification_failures, 1)
        self.assertEqual(workspace.latent_state.verification_passes, 1)
        self.assertEqual(workspace.latent_state.fast_weight_updates_used, 2)
        self.assertIn("expected", workspace.latent_state.failure_patterns[0])

    def test_controller_respects_fast_weight_update_budget(self):
        class FakeFastWeightProvider:
            def __init__(self):
                self.update_calls = []

            def begin_task(self, task_name: str, task_prompt: str) -> dict:
                return {"enabled": True, "task_name": task_name}

            def apply_self_update(self, *, prompt: str, target_text: str, update_kind: str, task_name: str):
                self.update_calls.append(update_kind)
                return {
                    "success": True,
                    "kind": update_kind,
                    "updated_modules": ["q_proj"],
                    "task_name": task_name,
                }

            def end_task(self) -> dict:
                return {"update_count": len(self.update_calls)}

            def __call__(self, prompt: str) -> str:
                if "Think privately" in prompt:
                    return "Use the Fibonacci recurrence."
                return "def fib(n):\n    return 0\n"

        provider = FakeFastWeightProvider()
        controller = BufferedInternalController(
            llm_provider=provider,
            config=InternalControllerConfig(
                max_generation_attempts=2,
                max_fast_weight_updates_per_task=1,
            ),
        )
        task = InternalDeliberationTask(
            name="fibonacci",
            prompt="Write fib(n) in Python.",
            test_cases=FIB_TESTS,
        )

        workspace = controller.run_task(task)

        self.assertEqual(workspace.status, "failed")
        self.assertEqual(provider.update_calls, ["plan"])
        fast_weight_steps = [step for step in workspace.steps if step.action == "fast_weight_update"]
        self.assertEqual(len(fast_weight_steps), 2)
        self.assertTrue(fast_weight_steps[0].success)
        self.assertFalse(fast_weight_steps[1].success)
        self.assertIn("budget exhausted", fast_weight_steps[1].error or "")
        self.assertEqual(workspace.steps[-1].action, "fail")

    def test_workspace_snapshot_includes_latent_state(self):
        def fake_provider(prompt: str) -> str:
            if "Think privately" in prompt:
                return "Use recursion with the correct base cases."
            if "repairing a hidden candidate" in prompt.lower():
                return (
                    "def fib(n):\n"
                    "    if n <= 1:\n"
                    "        return n\n"
                    "    return fib(n - 1) + fib(n - 2)\n"
                )
            return "def fib(n):\n    return 0\n"

        controller = BufferedInternalController(
            llm_provider=fake_provider,
            config=InternalControllerConfig(max_generation_attempts=2),
        )
        task = InternalDeliberationTask(
            name="fibonacci",
            prompt="Write fib(n) in Python.",
            test_cases=FIB_TESTS,
        )

        workspace = controller.run_task(task)
        snapshot = workspace.snapshot()

        self.assertIn("latent_state", snapshot)
        self.assertEqual(snapshot["latent_state"]["verification_failures"], 1)
        self.assertEqual(snapshot["latent_state"]["verification_passes"], 1)
        self.assertIn("descriptor_updates_used", snapshot["latent_state"])
        self.assertIn("hidden_plan", snapshot["latent_state"])
        self.assertIn("recent_actions", snapshot["latent_state"])
        self.assertIn("memory_vector", snapshot["latent_state"])
        self.assertEqual(len(snapshot["latent_state"]["memory_vector"]), 16)
        self.assertIn("memory_updates", snapshot["latent_state"])

    def test_latent_memory_updater_replaces_heuristic_memory_path(self):
        class FakeLatentMemoryUpdater:
            def __init__(self):
                self.calls = []

            def build_memory_delta(self, **kwargs):
                self.calls.append(kwargs["event_kind"])
                return [0.5] * 16, {"event_kind": kwargs["event_kind"]}

        def fake_provider(prompt: str) -> str:
            if "Think privately" in prompt:
                return "Use recursion."
            if "repairing a hidden candidate" in prompt.lower():
                return (
                    "def fib(n):\n"
                    "    if n <= 1:\n"
                    "        return n\n"
                    "    return fib(n - 1) + fib(n - 2)\n"
                )
            return "def fib(n):\n    return 0\n"

        updater = FakeLatentMemoryUpdater()
        controller = BufferedInternalController(
            llm_provider=fake_provider,
            latent_memory_updater=updater,
            config=InternalControllerConfig(max_generation_attempts=2),
        )
        task = InternalDeliberationTask(
            name="fibonacci",
            prompt="Write fib(n) in Python.",
            test_cases=FIB_TESTS,
        )

        workspace = controller.run_task(task)

        self.assertEqual(workspace.status, "committed")
        self.assertFalse(workspace.latent_state.enable_heuristic_memory_updates)
        self.assertGreater(workspace.latent_state.memory_updates, 0)
        self.assertIn("think", updater.calls)
        self.assertIn("verify", updater.calls)
        self.assertIn("latent_memory_update", workspace.steps[0].metadata)

    def test_controller_can_apply_latent_descriptor_updates(self):
        class FakeDescriptorProvider:
            def __init__(self):
                self.descriptor_calls = []

            def begin_task(self, task_name: str, task_prompt: str) -> dict:
                return {"enabled": True, "task_name": task_name}

            def apply_state_descriptor_update(
                self,
                *,
                task_name: str,
                update_kind: str,
                latent_state,
                error_text: str = "",
                candidate_text: str = "",
            ) -> dict:
                self.descriptor_calls.append(
                    {
                        "task_name": task_name,
                        "update_kind": update_kind,
                        "latent_state": latent_state.to_dict() if hasattr(latent_state, "to_dict") else latent_state,
                        "error_text": error_text,
                        "candidate_text": candidate_text,
                    }
                )
                return {
                    "success": True,
                    "kind": update_kind,
                    "updated_modules": ["q_proj"],
                    "task_name": task_name,
                    "steps": 1,
                    "adaptation_descriptor": {
                        "implementation": "ncpu_gradient_protocol",
                        "source": "latent_state",
                    },
                }

            def end_task(self) -> dict:
                return {"descriptor_updates": len(self.descriptor_calls)}

            def __call__(self, prompt: str) -> str:
                if "Think privately" in prompt:
                    return "Use the Fibonacci recurrence."
                if "repairing a hidden candidate" in prompt.lower():
                    return (
                        "def fib(n):\n"
                        "    if n <= 1:\n"
                        "        return n\n"
                        "    return fib(n - 1) + fib(n - 2)\n"
                    )
                return "def fib(n):\n    return 0\n"

        provider = FakeDescriptorProvider()
        controller = BufferedInternalController(
            llm_provider=provider,
            config=InternalControllerConfig(max_generation_attempts=2),
        )
        task = InternalDeliberationTask(
            name="fibonacci",
            prompt="Write fib(n) in Python.",
            test_cases=FIB_TESTS,
        )

        workspace = controller.run_task(task)

        self.assertEqual(workspace.status, "committed")
        self.assertEqual(
            [call["update_kind"] for call in provider.descriptor_calls],
            ["plan_descriptor", "verify_failure_descriptor"],
        )
        descriptor_steps = [step for step in workspace.steps if step.action == "descriptor_update"]
        self.assertEqual(len(descriptor_steps), 2)
        self.assertTrue(all(step.success for step in descriptor_steps))
        self.assertEqual(workspace.latent_state.descriptor_updates_used, 2)
        self.assertIn("expected", provider.descriptor_calls[1]["error_text"])


if __name__ == "__main__":
    unittest.main()
