"""Tests for the benchmark core."""
import tempfile
import unittest
from pathlib import Path

from ncpu.self_optimizing.code_verifier import VerificationResult
from ncpu.self_optimizing.controller_bundle import ControllerBundle, ControllerComponentConfig, save_controller_bundle
from ncpu.self_optimizing.llm_benchmark import (
    BenchmarkResult,
    BenchmarkTask,
    LLMBenchmark,
    ProviderResponse,
)


class TestBenchmarkResult(unittest.TestCase):
    """Test BenchmarkResult."""

    def test_create_result(self):
        result = BenchmarkResult(
            approach="test",
            num_samples=10,
            success_rate=0.8,
            avg_execution_time=1.5,
            avg_attempts=1.2,
        )
        self.assertEqual(result.num_samples, 10)
        self.assertEqual(result.success_rate, 0.8)
        self.assertEqual(result.task_results, [])


class TestLLMBenchmark(unittest.TestCase):
    """Test LLMBenchmark."""

    def test_create_benchmark(self):
        benchmark = LLMBenchmark()
        self.assertIsNotNone(benchmark.llm_provider)
        self.assertIsNotNone(benchmark.verify_fn)

    def test_benchmark_standard_accepts_verification_object(self):
        def provider(_prompt: str) -> ProviderResponse:
            return ProviderResponse(
                text="def test():\n    return 42",
                metadata={"eval_count": 6},
            )

        def verify(code: str) -> VerificationResult:
            namespace = {}
            exec(code, namespace)
            return VerificationResult(
                success=namespace["test"]() == 42,
                error=None,
                output=None,
                test_results=[],
            )

        benchmark = LLMBenchmark(llm_provider=provider, verify_fn=verify)
        result = benchmark.benchmark_standard(["Write test"])

        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.num_samples, 1)
        self.assertEqual(result.success_rate, 1.0)
        self.assertEqual(result.task_results[0]["attempt_details"][0]["metadata"]["eval_count"], 6)

    def test_benchmark_with_some_retries_until_success(self):
        attempts = {"count": 0}

        def provider(prompt: str) -> str:
            attempts["count"] += 1
            if "failed verification" in prompt.lower():
                return "def test():\n    return 42"
            return "def test():\n    return 0"

        def verify(code: str) -> tuple[bool, str]:
            namespace = {}
            exec(code, namespace)
            value = namespace["test"]()
            if value == 42:
                return True, ""
            return False, f"expected 42, got {value}"

        benchmark = LLMBenchmark(llm_provider=provider, verify_fn=verify)
        task = BenchmarkTask(name="test", prompt="Write test()", verify_fn=verify)

        result = benchmark.benchmark_with_some([task], max_retries=3)

        self.assertEqual(result.success_rate, 1.0)
        self.assertEqual(result.avg_attempts, 2.0)
        self.assertEqual(attempts["count"], 2)
        self.assertEqual(result.task_results[0]["attempts"], 2)

    def test_benchmark_with_buffered_some_writes_trajectory_artifact(self):
        attempts = {"count": 0}

        def provider(prompt: str) -> ProviderResponse:
            attempts["count"] += 1
            if "repairing a hidden candidate" in prompt.lower():
                return ProviderResponse(
                    text="def test():\n    return 42",
                    metadata={"eval_count": 7},
                )
            if "think privately" in prompt.lower():
                return ProviderResponse(text="Use the exact integer result.")
            return ProviderResponse(
                text="def test():\n    return 0",
                metadata={"eval_count": 3},
            )

        def verify(code: str) -> tuple[bool, str]:
            namespace = {}
            exec(code, namespace)
            value = namespace["test"]()
            if value == 42:
                return True, ""
            return False, f"expected 42, got {value}"

        benchmark = LLMBenchmark(llm_provider=provider, verify_fn=verify)
        task = BenchmarkTask(
            name="buffered_test",
            prompt="Write test()",
            verify_fn=verify,
            response_format="raw Python code",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = benchmark.benchmark_with_some(
                [task],
                max_retries=3,
                use_buffered_controller=True,
                trajectory_dir=tmpdir,
            )
            trajectory_path = result.task_results[0]["trajectory_path"]
            self.assertTrue(trajectory_path.endswith(".jsonl"))
            self.assertTrue(Path(trajectory_path).exists())

        self.assertEqual(result.success_rate, 1.0)
        self.assertEqual(result.avg_attempts, 2.0)
        self.assertEqual(result.task_results[0]["attempts"], 2)
        self.assertEqual(attempts["count"], 3)

    def test_buffered_some_uses_controller_bundle_config(self):
        def provider(prompt: str) -> ProviderResponse:
            self.assertNotIn("Think privately", prompt)
            return ProviderResponse(text="7", metadata={"eval_count": 1})

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "controller_bundle.json"
            save_controller_bundle(
                ControllerBundle(
                    name="weight-first",
                    response=ControllerComponentConfig(provider="hf_local", model="demo"),
                    controller_config={"plan_before_generate": False},
                ),
                bundle_path,
            )
            benchmark = LLMBenchmark(llm_provider=provider, verify_fn=lambda code: (code.strip() == "7", "expected 7"))
            task = BenchmarkTask(
                name="bundle_reasoning",
                prompt="Return the integer 7.",
                verify_fn=lambda code: (code.strip() == "7", "expected 7"),
                category="reasoning",
                response_format="the final integer only",
            )

            result = benchmark.benchmark_with_some(
                [task],
                max_retries=2,
                use_buffered_controller=True,
                trajectory_dir=tmpdir,
                controller_bundle_path=str(bundle_path),
            )

        self.assertEqual(result.success_rate, 1.0)
        self.assertEqual(result.task_results[0]["attempts"], 1)

    def test_buffered_some_uses_action_provider(self):
        def provider(prompt: str) -> ProviderResponse:
            if "repairing a hidden candidate" in prompt.lower():
                return ProviderResponse(text="def test():\n    return 42")
            return ProviderResponse(text="def test():\n    return 0")

        decisions = iter(["write", "fail"])
        benchmark = LLMBenchmark(
            llm_provider=provider,
            action_provider=lambda _prompt: next(decisions),
            verify_fn=lambda code: (("return 42" in code), "expected 42"),
        )
        task = BenchmarkTask(
            name="action_controlled",
            prompt="Write test()",
            response_format="raw Python code",
        )

        result = benchmark.benchmark_with_some(
            [task],
            max_retries=3,
            use_buffered_controller=True,
        )

        self.assertEqual(result.success_rate, 0.0)
        self.assertEqual(result.task_results[0]["attempts"], 1)
        self.assertIn("terminated", result.task_results[0]["final_error"])

    def test_benchmark_standard_handles_provider_exception(self):
        def provider(_prompt: str) -> str:
            raise TimeoutError("provider timed out")

        benchmark = LLMBenchmark(llm_provider=provider, verify_fn=lambda code: (True, None))
        result = benchmark.benchmark_standard([BenchmarkTask(name="test", prompt="Write test()")])

        self.assertEqual(result.success_rate, 0.0)
        self.assertEqual(len(result.task_results), 1)
        self.assertFalse(result.task_results[0]["success"])
        self.assertIn("provider exception", result.task_results[0]["final_error"])

    def test_default_llm(self):
        benchmark = LLMBenchmark()
        code = benchmark._default_llm("calculate fibonacci")
        self.assertIsInstance(code, str)
        self.assertIn("fib", code.lower())

    def test_default_verify(self):
        benchmark = LLMBenchmark()

        success, _error = benchmark._default_verify("def test(): return 42")
        self.assertTrue(success)

        success, _error = benchmark._default_verify("def test():")
        self.assertFalse(success)


class TestMockLLM(unittest.TestCase):
    """Test mock LLM behavior."""

    def test_mock_llm_behavior(self):
        benchmark = LLMBenchmark()
        code1 = benchmark._default_llm("fibonacci")
        code2 = benchmark._default_llm("sort")
        self.assertNotEqual(code1, code2)


if __name__ == "__main__":
    unittest.main()
