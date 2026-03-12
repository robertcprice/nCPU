"""Tests for the Qwen benchmark runner."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ncpu.self_optimizing.controller_bundle import (
    ControllerBundle,
    ControllerComponentConfig,
    save_controller_bundle,
)
from ncpu.self_optimizing.code_verifier import CodeVerifier
from ncpu.self_optimizing.llm_benchmark import ProviderResponse
from ncpu.self_optimizing.run_qwen_benchmark import (
    extract_json_object,
    make_reasoning_verifier,
    ReasoningTaskSpec,
    run_model_benchmark,
)


class TestCodeVerifierExtraction(unittest.TestCase):
    """Test code extraction for benchmarked model responses."""

    def test_extracts_fenced_python(self):
        verifier = CodeVerifier()
        response = """Here is the function:

```python
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
```
"""
        extracted = verifier.extract_code(response)
        self.assertTrue(extracted.startswith("def fib"))
        self.assertNotIn("```", extracted)


class TestReasoningHelpers(unittest.TestCase):
    """Test reasoning response parsing and verification."""

    def test_extract_json_object_from_fenced_response(self):
        payload, error = extract_json_object(
            '```json\n{"answer": 8, "explanation": "Shortest path is A-C-B-D."}\n```'
        )
        self.assertIsNone(error)
        self.assertEqual(payload["answer"], 8)

    def test_reasoning_verifier_rejects_wrong_answer(self):
        verifier = make_reasoning_verifier(
            ReasoningTaskSpec(
                name="weighted_path",
                prompt="Graph task",
                expected_answer=8,
                normalizer=int,
            )
        )
        result = verifier('{"answer": 7, "explanation": "I made a mistake."}')
        self.assertFalse(result.success)
        self.assertIn("expected 8", result.error)


class TestQwenBenchmarkFlow(unittest.TestCase):
    """Test end-to-end model benchmark flow with a mocked provider."""

    def test_run_model_benchmark_reports_improvements(self):
        def fake_provider(prompt: str) -> ProviderResponse:
            prompt_lower = prompt.lower()
            if "fibonacci" in prompt_lower:
                if "verification" in prompt_lower or "repairing a hidden candidate" in prompt_lower:
                    return ProviderResponse(
                        text=(
                            "def fib(n):\n"
                            "    if n <= 1:\n"
                            "        return n\n"
                            "    return fib(n - 1) + fib(n - 2)"
                        )
                    )
                return ProviderResponse(text="def fib(n):\n    return 0")
            if "factorial" in prompt_lower:
                return ProviderResponse(
                    text=(
                        "def factorial(n):\n"
                        "    result = 1\n"
                        "    for value in range(2, n + 1):\n"
                        "        result *= value\n"
                        "    return result"
                    )
                )
            if "is_prime" in prompt_lower:
                return ProviderResponse(
                    text=(
                        "def is_prime(n):\n"
                        "    if n < 2:\n"
                        "        return False\n"
                        "    for value in range(2, int(n ** 0.5) + 1):\n"
                        "        if n % value == 0:\n"
                        "            return False\n"
                        "    return True"
                    )
                )
            if "binary_search" in prompt_lower:
                return ProviderResponse(
                    text=(
                        "def binary_search(arr, target):\n"
                        "    left, right = 0, len(arr) - 1\n"
                        "    while left <= right:\n"
                        "        mid = (left + right) // 2\n"
                        "        if arr[mid] == target:\n"
                        "            return mid\n"
                        "        if arr[mid] < target:\n"
                        "            left = mid + 1\n"
                        "        else:\n"
                        "            right = mid - 1\n"
                        "    return -1"
                    )
                )
            if "quick_sort" in prompt_lower:
                return ProviderResponse(
                    text=(
                        "def quick_sort(arr):\n"
                        "    if len(arr) <= 1:\n"
                        "        return arr[:]\n"
                        "    pivot = arr[len(arr) // 2]\n"
                        "    left = [x for x in arr if x < pivot]\n"
                        "    middle = [x for x in arr if x == pivot]\n"
                        "    right = [x for x in arr if x > pivot]\n"
                        "    return quick_sort(left) + middle + quick_sort(right)"
                    )
                )
            raise AssertionError(f"Unexpected prompt: {prompt}")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("ncpu.self_optimizing.run_qwen_benchmark.build_provider", return_value=fake_provider):
                report = run_model_benchmark(
                    provider_name="local",
                    model="fake-qwen",
                    include_coding=True,
                    include_reasoning=False,
                    repeats=1,
                    max_retries=2,
                    trajectory_dir=tmpdir,
                )

        self.assertAlmostEqual(report["baseline"]["summary"]["success_rate"], 0.8)
        self.assertAlmostEqual(report["some"]["summary"]["success_rate"], 1.0)
        improved = {item["name"] for item in report["delta"]["improved_tasks"]}
        self.assertIn("fibonacci", improved)
        self.assertEqual(report["some"]["trajectory_dir"], tmpdir)

    def test_run_model_benchmark_records_action_policy_config(self):
        def fake_provider(prompt: str) -> ProviderResponse:
            prompt_lower = prompt.lower()
            if "fibonacci" in prompt_lower and "repairing a hidden candidate" not in prompt_lower:
                return ProviderResponse(text="def fib(n):\n    return 0")
            return ProviderResponse(
                text=(
                    "def fib(n):\n"
                    "    if n <= 1:\n"
                    "        return n\n"
                    "    return fib(n - 1) + fib(n - 2)"
                )
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("ncpu.self_optimizing.run_qwen_benchmark.build_provider", return_value=fake_provider):
                report = run_model_benchmark(
                    provider_name="hf_local",
                    model="response-adapter",
                    action_provider_name="hf_local",
                    action_model="action-adapter",
                    include_coding=True,
                    include_reasoning=False,
                    repeats=1,
                    max_retries=2,
                    trajectory_dir=tmpdir,
                )

        self.assertEqual(report["some"]["action_policy"]["provider"], "hf_local")
        self.assertEqual(report["some"]["action_policy"]["model"], "action-adapter")

    def test_run_model_benchmark_can_use_controller_bundle_defaults(self):
        def fake_provider(prompt: str) -> ProviderResponse:
            prompt_lower = prompt.lower()
            if "fibonacci" in prompt_lower and "repairing a hidden candidate" not in prompt_lower:
                return ProviderResponse(text="def fib(n):\n    return 0")
            return ProviderResponse(
                text=(
                    "def fib(n):\n"
                    "    if n <= 1:\n"
                    "        return n\n"
                    "    return fib(n - 1) + fib(n - 2)"
                )
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "controller_bundle.json"
            save_controller_bundle(
                ControllerBundle(
                    name="demo",
                    response=ControllerComponentConfig(
                        provider="hf_fast_weights",
                        model="response-adapter",
                        max_tokens=128,
                        provider_kwargs={"fast_weights_rank": 4},
                    ),
                    action=ControllerComponentConfig(
                        provider="hf_local",
                        model="action-adapter",
                        max_tokens=16,
                        provider_kwargs={"role": "policy"},
                    ),
                ),
                bundle_path,
            )
            with patch("ncpu.self_optimizing.run_qwen_benchmark.build_provider", return_value=fake_provider) as build_provider:
                report = run_model_benchmark(
                    provider_name=None,
                    model=None,
                    controller_bundle_path=str(bundle_path),
                    include_coding=True,
                    include_reasoning=False,
                    repeats=1,
                    max_retries=2,
                    trajectory_dir=tmpdir,
                )

        self.assertEqual(report["provider"], "hf_fast_weights")
        self.assertEqual(report["model"], "response-adapter")
        self.assertEqual(report["controller_bundle_path"], str(bundle_path))
        self.assertEqual(report["some"]["action_policy"]["provider"], "hf_local")
        self.assertEqual(report["some"]["action_policy"]["model"], "action-adapter")
        self.assertEqual(report["max_tokens"], 128)
        self.assertEqual(report["provider_kwargs"], {"fast_weights_rank": 4})
        self.assertEqual(report["some"]["action_policy"]["max_tokens"], 16)
        self.assertEqual(report["some"]["action_policy"]["provider_kwargs"], {"role": "policy"})
        self.assertEqual(build_provider.call_args_list[0].kwargs["provider_name"], "hf_fast_weights")
        self.assertEqual(build_provider.call_args_list[0].kwargs["max_tokens"], 128)
        self.assertEqual(build_provider.call_args_list[0].kwargs["provider_kwargs"], {"fast_weights_rank": 4})
        self.assertEqual(build_provider.call_args_list[1].kwargs["provider_kwargs"], {"role": "policy"})


if __name__ == "__main__":
    unittest.main()
