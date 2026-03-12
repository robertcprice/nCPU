"""Tests for the buffered internal inference API."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from ncpu.self_optimizing.api_server import BufferedInferenceRequest, app, _create_llm_provider
from ncpu.self_optimizing.controller_bundle import (
    ControllerBundle,
    ControllerComponentConfig,
    save_controller_bundle,
)


FIB_TESTS = [
    {"input": {"n": 0}, "expected": 0},
    {"input": {"n": 1}, "expected": 1},
    {"input": {"n": 6}, "expected": 8},
]


class TestBufferedInferenceAPI(unittest.TestCase):
    """Validate the FastAPI surface for hidden buffered inference."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_internal_infer_repairs_and_commits_code(self):
        def fake_provider(prompt: str) -> str:
            if "Think privately" in prompt:
                return "Use recursion with correct base cases."
            if "repairing a hidden candidate" in prompt.lower():
                return (
                    "def fib(n):\n"
                    "    if n <= 1:\n"
                    "        return n\n"
                    "    return fib(n - 1) + fib(n - 2)\n"
                )
            return "def fib(n):\n    return 0\n"

        with patch("ncpu.self_optimizing.api_server._create_llm_provider", return_value=fake_provider):
            response = self.client.post(
                "/internal/infer",
                json={
                    "task_name": "fibonacci",
                    "prompt": "Write fib(n) in Python.",
                    "provider": "local",
                    "model": "fake-model",
                    "test_cases": FIB_TESTS,
                    "max_generation_attempts": 3,
                    "persist_trajectory": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "committed")
        self.assertTrue(body["committed_verified"])
        self.assertEqual(body["verification_mode"], "code_tests")
        self.assertEqual(body["generation_attempts"], 2)
        self.assertEqual(body["hidden_actions"], ["think", "write", "verify", "patch", "verify", "commit"])
        self.assertIn("return fib(n - 1)", body["committed_output"])
        self.assertIsNone(body["hidden_workspace"])
        self.assertIsNone(body["trajectory_path"])

    def test_internal_infer_can_return_hidden_trace_and_persist_trajectory(self):
        def fake_provider(prompt: str) -> str:
            if "repairing a hidden candidate" in prompt.lower():
                return "7"
            return "5"

        with tempfile.TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl"
            with patch("ncpu.self_optimizing.api_server._create_llm_provider", return_value=fake_provider):
                response = self.client.post(
                    "/internal/infer",
                    json={
                        "task_name": "return_seven",
                        "prompt": "Return the integer 7.",
                        "category": "reasoning",
                        "response_format": "the final integer only",
                        "provider": "local",
                        "model": "fake-model",
                        "verification_mode": "exact_match",
                        "expected_output": 7,
                        "max_generation_attempts": 2,
                        "plan_before_generate": False,
                        "include_hidden_trace": True,
                        "trajectory_path": str(trajectory_path),
                    },
                )

            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["status"], "committed")
            self.assertEqual(body["verification_mode"], "exact_match")
            self.assertEqual(body["committed_output"], "7")
            self.assertEqual(body["hidden_workspace"]["task_name"], "return_seven")
            self.assertEqual(body["hidden_workspace"]["status"], "committed")
            self.assertEqual(body["hidden_workspace"]["steps"][0]["action"], "write")
            self.assertEqual(body["hidden_workspace"]["steps"][2]["action"], "patch")
            self.assertEqual(body["trajectory_path"], str(trajectory_path))
            self.assertTrue(trajectory_path.exists())

            events = [
                json.loads(line)
                for line in trajectory_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(events[0]["event"], "workspace_initialized")
            self.assertEqual(events[-1]["event"], "workspace_committed")

    def test_internal_infer_rejects_invalid_exact_match_request(self):
        response = self.client.post(
            "/internal/infer",
            json={
                "task_name": "broken",
                "prompt": "Return the integer 7.",
                "provider": "local",
                "model": "fake-model",
                "verification_mode": "exact_match",
                "persist_trajectory": False,
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("expected_output", response.json()["detail"])

    def test_internal_infer_can_use_separate_action_policy_provider(self):
        def response_provider(prompt: str) -> str:
            self.assertNotIn("Think privately", prompt)
            return "7"

        def action_provider(_prompt: str) -> str:
            return "write"

        with patch(
            "ncpu.self_optimizing.api_server._create_llm_provider",
            side_effect=[response_provider, action_provider],
        ):
            response = self.client.post(
                "/internal/infer",
                json={
                    "task_name": "action_policy_reasoning",
                    "prompt": "Return the integer 7.",
                    "category": "reasoning",
                    "response_format": "the final integer only",
                    "provider": "local",
                    "model": "fake-response-model",
                    "action_provider": "hf_local",
                    "action_model": "fake-action-model",
                    "verification_mode": "exact_match",
                    "expected_output": 7,
                    "persist_trajectory": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "committed")
        self.assertEqual(body["action_provider"], "hf_local")
        self.assertEqual(body["action_model"], "fake-action-model")
        self.assertEqual(body["hidden_actions"], ["write", "verify", "commit"])
        self.assertTrue(body["committed_verified"])

    def test_create_llm_provider_uses_action_timeout_overrides(self):
        request = BufferedInferenceRequest(
            prompt="Return 7.",
            provider="local",
            model="response-model",
            request_timeout_seconds=90.0,
            action_provider="hf_local",
            action_model="action-model",
            action_request_timeout_seconds=12.0,
            action_temperature=0.0,
            action_max_tokens=32,
        )

        with patch("ncpu.self_optimizing.llm_provider.LLMProviderFactory.create_provider") as factory:
            factory.return_value = lambda _prompt: "ok"
            _create_llm_provider(request, action=True)

        _, kwargs = factory.call_args
        self.assertEqual(kwargs["provider"], "hf_local")
        self.assertEqual(kwargs["model"], "action-model")
        self.assertEqual(kwargs["request_timeout"], 12.0)
        self.assertEqual(kwargs["max_tokens"], 32)

    def test_internal_infer_can_load_controller_bundle_defaults(self):
        def bundled_provider(prompt: str) -> str:
            self.assertNotIn("Think privately", prompt)
            return "7"

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "controller_bundle.json"
            save_controller_bundle(
                ControllerBundle(
                    name="demo",
                    response=ControllerComponentConfig(
                        provider="hf_local",
                        model="response-adapter",
                        temperature=0.0,
                    ),
                    action=ControllerComponentConfig(
                        provider="hf_local",
                        model="action-adapter",
                        temperature=0.0,
                    ),
                ),
                bundle_path,
            )

            with patch(
                "ncpu.self_optimizing.api_server._create_llm_provider",
                side_effect=[bundled_provider, lambda _prompt: "write"],
            ):
                response = self.client.post(
                    "/internal/infer",
                    json={
                        "task_name": "bundle_reasoning",
                        "prompt": "Return the integer 7.",
                        "category": "reasoning",
                        "response_format": "the final integer only",
                        "controller_bundle_path": str(bundle_path),
                        "verification_mode": "exact_match",
                        "expected_output": 7,
                        "persist_trajectory": False,
                    },
                )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["provider"], "hf_local")
        self.assertEqual(body["model"], "response-adapter")
        self.assertEqual(body["action_provider"], "hf_local")
        self.assertEqual(body["action_model"], "action-adapter")
        self.assertEqual(body["controller_bundle_path"], str(bundle_path))
        self.assertEqual(body["hidden_actions"], ["write", "verify", "commit"])

    def test_internal_infer_uses_bundle_controller_config(self):
        def bundled_provider(prompt: str) -> str:
            self.assertNotIn("Think privately", prompt)
            return "7"

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "controller_bundle.json"
            save_controller_bundle(
                ControllerBundle(
                    name="demo",
                    response=ControllerComponentConfig(
                        provider="hf_local",
                        model="response-adapter",
                        temperature=0.0,
                    ),
                    controller_config={"plan_before_generate": False},
                ),
                bundle_path,
            )

            with patch(
                "ncpu.self_optimizing.api_server._create_llm_provider",
                return_value=bundled_provider,
            ):
                response = self.client.post(
                    "/internal/infer",
                    json={
                        "task_name": "bundle_weight_first_reasoning",
                        "prompt": "Return the integer 7.",
                        "category": "reasoning",
                        "response_format": "the final integer only",
                        "controller_bundle_path": str(bundle_path),
                        "verification_mode": "exact_match",
                        "expected_output": 7,
                        "persist_trajectory": False,
                    },
                )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "committed")
        self.assertEqual(body["hidden_actions"], ["write", "verify", "commit"])

    def test_create_llm_provider_passes_bundle_provider_kwargs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "controller_bundle.json"
            save_controller_bundle(
                ControllerBundle(
                    name="demo",
                    response=ControllerComponentConfig(
                        provider="hf_fast_weights",
                        model="response-adapter",
                        provider_kwargs={"fast_weights_rank": 4},
                    ),
                ),
                bundle_path,
            )
            request = BufferedInferenceRequest(
                task_name="reasoning",
                prompt="Return the integer 7.",
                category="reasoning",
                response_format="the final integer only",
                controller_bundle_path=str(bundle_path),
            )

            with patch("ncpu.self_optimizing.llm_provider.LLMProviderFactory.create_provider") as factory:
                factory.return_value = lambda _prompt: "ok"
                _create_llm_provider(request)

        _, kwargs = factory.call_args
        self.assertEqual(kwargs["provider"], "hf_fast_weights")
        self.assertEqual(kwargs["model"], "response-adapter")
        self.assertEqual(kwargs["fast_weights_rank"], 4)

    def test_async_internal_task_exposes_sanitized_events_and_status(self):
        def fake_provider(prompt: str) -> str:
            time.sleep(0.01)
            if "Think privately" in prompt:
                return "Use recursion with correct base cases."
            if "repairing a hidden candidate" in prompt.lower():
                return (
                    "def fib(n):\n"
                    "    if n <= 1:\n"
                    "        return n\n"
                    "    return fib(n - 1) + fib(n - 2)\n"
                )
            return "def fib(n):\n    return 0\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "async_trajectory.jsonl"
            with patch("ncpu.self_optimizing.api_server._create_llm_provider", return_value=fake_provider):
                submit = self.client.post(
                    "/internal/tasks",
                    json={
                        "task_name": "fibonacci_async",
                        "prompt": "Write fib(n) in Python.",
                        "provider": "local",
                        "model": "fake-model",
                        "test_cases": FIB_TESTS,
                        "max_generation_attempts": 3,
                        "trajectory_path": str(trajectory_path),
                    },
                )

                self.assertEqual(submit.status_code, 200)
                task_id = submit.json()["task_id"]

                deadline = time.time() + 3.0
                status = None
                while time.time() < deadline:
                    poll = self.client.get(f"/internal/tasks/{task_id}")
                    self.assertEqual(poll.status_code, 200)
                    status = poll.json()
                    if status["status"] in {"success", "failed"}:
                        break
                    time.sleep(0.02)

            self.assertIsNotNone(status)
            self.assertEqual(status["status"], "success")
            self.assertTrue(status["committed_verified"])
            self.assertIn("return fib(n - 1)", status["committed_output"])
            self.assertEqual(status["trajectory_path"], str(trajectory_path))
            self.assertIsNone(status["action_provider"])
            self.assertIsNone(status["action_model"])
            self.assertTrue(trajectory_path.exists())

            events_response = self.client.get(f"/internal/tasks/{task_id}/events")
            self.assertEqual(events_response.status_code, 200)
            events = events_response.json()
            self.assertGreaterEqual(len(events), 3)
            actions = [event["action"] for event in events if event.get("action")]
            self.assertEqual(actions[:5], ["think", "write", "verify", "patch", "verify"])
            self.assertTrue(all("response_text" not in event for event in events))
            self.assertTrue(all("prompt" not in event for event in events))
            self.assertTrue(all("committed_output" not in event for event in events))

    def test_async_internal_task_stream_replays_sanitized_events(self):
        def fake_provider(prompt: str) -> str:
            if "Think privately" in prompt:
                return "Return the exact integer."
            return "7"

        with patch("ncpu.self_optimizing.api_server._create_llm_provider", return_value=fake_provider):
            submit = self.client.post(
                "/internal/tasks",
                json={
                    "task_name": "stream_reasoning",
                    "prompt": "Return the integer 7.",
                    "category": "reasoning",
                    "response_format": "the final integer only",
                    "provider": "local",
                    "model": "fake-model",
                    "verification_mode": "exact_match",
                    "expected_output": 7,
                    "persist_trajectory": False,
                },
            )

            self.assertEqual(submit.status_code, 200)
            task_id = submit.json()["task_id"]

            deadline = time.time() + 3.0
            while time.time() < deadline:
                poll = self.client.get(f"/internal/tasks/{task_id}")
                self.assertEqual(poll.status_code, 200)
                if poll.json()["status"] in {"success", "failed"}:
                    break
                time.sleep(0.02)

            with self.client.stream("GET", f"/internal/tasks/{task_id}/stream") as response:
                self.assertEqual(response.status_code, 200)
                stream_body = "".join(chunk for chunk in response.iter_text())

        self.assertIn("workspace_initialized", stream_body)
        self.assertIn("workspace_step", stream_body)
        self.assertIn("stream_complete", stream_body)
        self.assertNotIn("response_text", stream_body)
        self.assertNotIn("prompt", stream_body)


if __name__ == "__main__":
    unittest.main()
