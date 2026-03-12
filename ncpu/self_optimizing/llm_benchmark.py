"""
LLM benchmark support for baseline vs SOME-style verification loops.

The benchmark core is deliberately provider-agnostic:
- Providers may return raw text or structured metadata.
- Verifiers may return booleans, tuples, or rich result objects.
- Tasks may be plain prompts or prompt objects with custom verifiers.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Callable, Optional


@dataclass
class ProviderResponse:
    """Normalized model response with optional provider metadata."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkTask:
    """Single benchmark task."""

    name: str
    prompt: str
    verify_fn: Optional[Callable[[str], Any]] = None
    category: str = "coding"
    feedback_builder: Optional[Callable[[str, dict[str, Any]], str]] = None
    response_format: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Aggregate result for a benchmark run."""

    approach: str
    num_samples: int
    success_rate: float
    avg_execution_time: float
    avg_attempts: float
    errors: list[str] = field(default_factory=list)
    task_results: list[dict[str, Any]] = field(default_factory=list)


class LLMBenchmark:
    """
    Benchmarks baseline prompting against a SOME-style verify/retry loop.

    Compares:
    1. Standard prompting (single attempt)
    2. With SOME (generate -> verify -> feedback -> retry)
    """

    def __init__(
        self,
        llm_provider: Optional[Callable[[str], Any]] = None,
        action_provider: Optional[Callable[[str], Any]] = None,
        verify_fn: Optional[Callable[[str], Any]] = None,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ):
        self.llm_provider = llm_provider or self._default_llm
        self.action_provider = action_provider
        self.verify_fn = verify_fn or self._default_verify
        self.progress_callback = progress_callback

    def _default_llm(self, prompt: str) -> str:
        """Default mock LLM."""
        prompt_lower = prompt.lower()
        if "fibonacci" in prompt_lower:
            return "def fib(n): return [0,1,1,2,3,5,8][n%7]"
        if "sort" in prompt_lower:
            return "def sort(x): return sorted(x)"
        return "def answer(): return 42"

    def _default_verify(self, code: str) -> tuple[bool, Optional[str]]:
        """Default verifier: checks that the code executes."""
        try:
            exec(code, {})
            return True, None
        except Exception as exc:  # pragma: no cover - trivial path
            return False, str(exc)

    def _normalize_tasks(self, prompts: list[Any]) -> list[BenchmarkTask]:
        tasks: list[BenchmarkTask] = []
        for index, item in enumerate(prompts):
            if isinstance(item, BenchmarkTask):
                tasks.append(item)
            else:
                tasks.append(
                    BenchmarkTask(
                        name=f"task_{index + 1}",
                        prompt=str(item),
                    )
                )
        return tasks

    def _normalize_generation(self, response: Any) -> ProviderResponse:
        if isinstance(response, ProviderResponse):
            return response
        if isinstance(response, str):
            return ProviderResponse(text=response)
        if isinstance(response, dict):
            if "text" in response:
                metadata = {k: v for k, v in response.items() if k != "text"}
                return ProviderResponse(text=str(response["text"]), metadata=metadata)
            return ProviderResponse(text=str(response))
        if hasattr(response, "text"):
            return ProviderResponse(
                text=str(getattr(response, "text")),
                metadata=dict(getattr(response, "metadata", {}) or {}),
            )
        return ProviderResponse(text=str(response))

    def _normalize_verification(self, verification: Any) -> tuple[bool, Optional[str], Any]:
        if isinstance(verification, tuple):
            success = bool(verification[0]) if verification else False
            error = verification[1] if len(verification) > 1 else None
            return success, error, verification
        if isinstance(verification, bool):
            return verification, (None if verification else "Verification returned False"), verification
        if hasattr(verification, "success"):
            success = bool(getattr(verification, "success"))
            error = getattr(verification, "error", None)
            return success, error, verification
        raise TypeError(
            "Verifier must return a bool, (success, error) tuple, "
            "or an object with 'success' and optional 'error' attributes"
        )

    def _serialize_verification(self, verification: Any) -> Optional[dict[str, Any]]:
        if isinstance(verification, tuple):
            return {
                "success": bool(verification[0]) if verification else False,
                "error": verification[1] if len(verification) > 1 else None,
            }
        if isinstance(verification, bool):
            return {"success": verification}
        if hasattr(verification, "__dict__"):
            details: dict[str, Any] = {}
            for key, value in verification.__dict__.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    details[key] = value
                elif isinstance(value, (list, dict)):
                    details[key] = value
                else:
                    details[key] = repr(value)
            return details
        return None

    def _summarize_error(self, error: Optional[str], verification_details: Optional[dict[str, Any]]) -> str:
        if error:
            return error
        if not verification_details:
            return "verification failed"

        failed_tests = [
            item for item in verification_details.get("test_results", [])
            if not item.get("passed", False)
        ]
        if not failed_tests:
            return "verification failed"

        first = failed_tests[0]
        if "error" in first:
            return f"test {first.get('test', '?')} raised {first['error']}"
        return (
            f"test {first.get('test', '?')} expected {first.get('expected')!r} "
            f"but got {first.get('actual')!r}"
        )

    def _default_feedback_builder(self, task: BenchmarkTask, attempt: dict[str, Any]) -> str:
        failure_summary = self._summarize_error(attempt["error"], attempt["verification"])
        return (
            f"{task.prompt}\n\n"
            f"Your previous answer failed verification.\n"
            f"Failure details: {failure_summary}\n"
            "Return only a corrected solution."
        )

    def _run_attempt(self, task: BenchmarkTask, prompt: str) -> dict[str, Any]:
        verifier = task.verify_fn or self.verify_fn

        started = time.perf_counter()
        try:
            response = self._normalize_generation(self.llm_provider(prompt))
        except Exception as exc:
            elapsed = time.perf_counter() - started
            return {
                "prompt": prompt,
                "response_text": "",
                "metadata": {},
                "elapsed_seconds": elapsed,
                "success": False,
                "error": f"provider exception: {type(exc).__name__}: {exc}",
                "verification": None,
            }

        elapsed = time.perf_counter() - started

        try:
            verification_raw = verifier(response.text)
            success, error, verification = self._normalize_verification(verification_raw)
            verification_details = self._serialize_verification(verification)
        except Exception as exc:
            success = False
            error = f"verifier exception: {type(exc).__name__}: {exc}"
            verification_details = None

        return {
            "prompt": prompt,
            "response_text": response.text,
            "metadata": response.metadata,
            "elapsed_seconds": elapsed,
            "success": success,
            "error": error,
            "verification": verification_details,
        }

    def _emit_progress(self, payload: dict[str, Any]) -> None:
        if not self.progress_callback:
            return
        self.progress_callback(payload)

    def _default_response_format(self, task: BenchmarkTask) -> str:
        if task.response_format:
            return task.response_format
        if task.category == "reasoning":
            return "JSON only with keys 'answer' and 'explanation'"
        return "raw Python code"

    def _build_trajectory_path(
        self,
        trajectory_dir: Optional[str],
        *,
        task: BenchmarkTask,
        task_index: int,
    ) -> Optional[str]:
        if not trajectory_dir:
            return None

        slug = "".join(char if char.isalnum() else "_" for char in task.name.lower()).strip("_") or f"task_{task_index}"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = Path(trajectory_dir) / f"{task_index:04d}_{slug}_{timestamp}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _task_to_internal_task(self, task: BenchmarkTask) -> Any:
        from ncpu.self_optimizing.internal_controller import InternalDeliberationTask

        verifier = task.verify_fn or self.verify_fn
        return InternalDeliberationTask(
            name=task.name,
            prompt=task.prompt,
            verifier=verifier,
            category=task.category,
            response_format=self._default_response_format(task),
            feedback_builder=task.feedback_builder,
        )

    def _workspace_to_attempts(self, workspace: Any) -> list[dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        pending_attempt: Optional[dict[str, Any]] = None

        for step in workspace.steps:
            if step.action not in {"write", "patch", "verify"}:
                continue

            if step.action in {"write", "patch"}:
                pending_attempt = {
                    "prompt": step.prompt,
                    "response_text": step.response_text,
                    "metadata": dict(step.metadata or {}),
                    "elapsed_seconds": float((step.metadata or {}).get("model_elapsed_seconds", 0.0) or 0.0),
                    "success": False,
                    "error": None,
                    "verification": None,
                }
                continue

            if pending_attempt is None:
                continue

            verification_metadata = dict(step.metadata or {})
            verification_elapsed = float(verification_metadata.pop("verification_elapsed_seconds", 0.0) or 0.0)
            pending_attempt["elapsed_seconds"] += verification_elapsed
            pending_attempt["success"] = bool(step.success)
            pending_attempt["error"] = step.error
            pending_attempt["verification"] = verification_metadata or None
            attempts.append(pending_attempt)
            pending_attempt = None

        return attempts

    def _resolve_buffered_controller_config(
        self,
        *,
        max_retries: int,
        controller_bundle_path: Optional[str] = None,
    ) -> Any:
        from ncpu.self_optimizing.internal_controller import InternalControllerConfig

        config_kwargs: dict[str, Any] = {}
        if controller_bundle_path:
            from ncpu.self_optimizing.controller_bundle import load_controller_bundle

            config_kwargs.update(load_controller_bundle(controller_bundle_path).controller_config)

        config_kwargs.setdefault("max_generation_attempts", max_retries)
        config_kwargs.setdefault("plan_before_generate", True)
        config_kwargs.setdefault("allow_unverified_commit", False)
        config_kwargs.setdefault("commit_on_first_success", True)
        return InternalControllerConfig(**config_kwargs)

    def _buffered_task_result(
        self,
        task: BenchmarkTask,
        *,
        task_index: int,
        max_retries: int,
        trajectory_dir: Optional[str],
        controller_bundle_path: Optional[str] = None,
    ) -> dict[str, Any]:
        from ncpu.self_optimizing.internal_controller import BufferedInternalController
        from ncpu.self_optimizing.controller_runtime import (
            load_bundle_latent_action_policy,
            load_bundle_latent_halt_policy,
            load_bundle_latent_memory_updater,
        )
        from ncpu.self_optimizing.trajectory_logger import TrajectoryLogger

        trajectory_path = self._build_trajectory_path(trajectory_dir, task=task, task_index=task_index)
        controller = BufferedInternalController(
            llm_provider=self.llm_provider,
            action_provider=self.action_provider,
            latent_action_policy=load_bundle_latent_action_policy(
                controller_bundle_path=controller_bundle_path,
            ),
            latent_memory_updater=load_bundle_latent_memory_updater(
                controller_bundle_path=controller_bundle_path,
            ),
            latent_halt_policy=load_bundle_latent_halt_policy(
                controller_bundle_path=controller_bundle_path,
            ),
            config=self._resolve_buffered_controller_config(
                max_retries=max_retries,
                controller_bundle_path=controller_bundle_path,
            ),
            trajectory_logger=TrajectoryLogger(trajectory_path),
        )
        try:
            workspace = controller.run_task(self._task_to_internal_task(task))
        except Exception as exc:
            final_error = f"buffered controller exception: {type(exc).__name__}: {exc}"
            return {
                "name": task.name,
                "category": task.category,
                "success": False,
                "attempts": 0,
                "elapsed_seconds": 0.0,
                "final_error": final_error,
                "attempt_details": [],
                "trajectory_path": trajectory_path,
                "hidden_step_count": 0,
                "hidden_workspace_status": "failed",
                "committed_verified": False,
            }
        attempts = self._workspace_to_attempts(workspace)
        final_attempt = attempts[-1] if attempts else {
            "prompt": task.prompt,
            "response_text": "",
            "metadata": {},
            "elapsed_seconds": 0.0,
            "success": False,
            "error": workspace.last_error or "buffered inference failed",
            "verification": workspace.last_verification,
        }
        if final_attempt["success"]:
            final_error = None
        else:
            final_error = workspace.last_error or self._summarize_error(
                final_attempt["error"],
                final_attempt["verification"],
            )
        return {
            "name": task.name,
            "category": task.category,
            "success": final_attempt["success"],
            "attempts": len(attempts),
            "elapsed_seconds": sum(item["elapsed_seconds"] for item in attempts),
            "final_error": final_error,
            "attempt_details": attempts,
            "trajectory_path": trajectory_path,
            "hidden_step_count": len(workspace.steps),
            "hidden_workspace_status": workspace.status,
            "committed_verified": workspace.committed_verified,
        }

    def benchmark_standard(
        self,
        prompts: list[Any],
        max_retries: int = 1,
    ) -> BenchmarkResult:
        """Standard approach: generate once and verify once."""
        del max_retries  # Standard mode is always pass@1.

        tasks = self._normalize_tasks(prompts)
        outcomes: list[bool] = []
        errors: list[str] = []
        task_results: list[dict[str, Any]] = []
        total_time = 0.0

        for index, task in enumerate(tasks, start=1):
            self._emit_progress(
                {
                    "approach": "standard",
                    "event": "task_start",
                    "task_index": index,
                    "task_total": len(tasks),
                    "task_name": task.name,
                }
            )
            attempt = self._run_attempt(task, task.prompt)
            total_time += attempt["elapsed_seconds"]
            outcomes.append(attempt["success"])

            if not attempt["success"]:
                errors.append(self._summarize_error(attempt["error"], attempt["verification"])[:200])

            task_result = {
                "name": task.name,
                "category": task.category,
                "success": attempt["success"],
                "attempts": 1,
                "elapsed_seconds": attempt["elapsed_seconds"],
                "final_error": None if attempt["success"] else self._summarize_error(
                    attempt["error"], attempt["verification"]
                ),
                "attempt_details": [attempt],
            }
            task_results.append(task_result)
            self._emit_progress(
                {
                    "approach": "standard",
                    "event": "task_complete",
                    "task_index": index,
                    "task_total": len(tasks),
                    "task_name": task.name,
                    "success": attempt["success"],
                    "attempts": 1,
                    "elapsed_seconds": attempt["elapsed_seconds"],
                    "running_success_rate": sum(outcomes) / len(outcomes),
                    "final_error": None if attempt["success"] else self._summarize_error(
                        attempt["error"], attempt["verification"]
                    ),
                    "task_result": task_result,
                }
            )

        success_rate = sum(outcomes) / len(tasks) if tasks else 0.0

        return BenchmarkResult(
            approach="Standard (single attempt)",
            num_samples=len(tasks),
            success_rate=success_rate,
            avg_execution_time=(total_time / len(tasks)) if tasks else 0.0,
            avg_attempts=1.0 if tasks else 0.0,
            errors=errors,
            task_results=task_results,
        )

    def benchmark_with_some(
        self,
        prompts: list[Any],
        max_retries: int = 5,
        *,
        use_buffered_controller: bool = False,
        trajectory_dir: Optional[str] = None,
        controller_bundle_path: Optional[str] = None,
    ) -> BenchmarkResult:
        """SOME-style approach: verify each attempt and retry with feedback."""
        tasks = self._normalize_tasks(prompts)
        outcomes: list[bool] = []
        errors: list[str] = []
        task_results: list[dict[str, Any]] = []
        total_attempts = 0
        total_time = 0.0

        for index, task in enumerate(tasks, start=1):
            self._emit_progress(
                {
                    "approach": "some",
                    "event": "task_start",
                    "task_index": index,
                    "task_total": len(tasks),
                    "task_name": task.name,
                }
            )
            if use_buffered_controller:
                task_result = self._buffered_task_result(
                    task,
                    task_index=index,
                    max_retries=max_retries,
                    trajectory_dir=trajectory_dir,
                    controller_bundle_path=controller_bundle_path,
                )
                attempts = task_result["attempt_details"]
                final_attempt = attempts[-1] if attempts else {
                    "success": False,
                    "error": task_result["final_error"],
                    "verification": None,
                    "elapsed_seconds": 0.0,
                }
            else:
                attempts = []
                current_prompt = task.prompt

                for _ in range(max_retries):
                    attempt = self._run_attempt(task, current_prompt)
                    attempts.append(attempt)

                    if attempt["success"]:
                        break

                    feedback_builder = task.feedback_builder or (
                        lambda original_prompt, attempt_data, task_obj=task: self._default_feedback_builder(
                            task_obj,
                            attempt_data,
                        )
                    )
                    current_prompt = feedback_builder(task.prompt, attempt)

                final_attempt = attempts[-1]
                final_error = None if final_attempt["success"] else self._summarize_error(
                    final_attempt["error"],
                    final_attempt["verification"],
                )
                task_result = {
                    "name": task.name,
                    "category": task.category,
                    "success": final_attempt["success"],
                    "attempts": len(attempts),
                    "elapsed_seconds": sum(item["elapsed_seconds"] for item in attempts),
                    "final_error": final_error,
                    "attempt_details": attempts,
                }

            final_error = task_result["final_error"]
            outcomes.append(task_result["success"])
            total_attempts += task_result["attempts"]
            total_time += task_result["elapsed_seconds"]

            if final_error:
                errors.append(final_error[:200])

            task_results.append(task_result)
            self._emit_progress(
                {
                    "approach": "some",
                    "event": "task_complete",
                    "task_index": index,
                    "task_total": len(tasks),
                    "task_name": task.name,
                    "success": task_result["success"],
                    "attempts": task_result["attempts"],
                    "elapsed_seconds": task_result["elapsed_seconds"],
                    "running_success_rate": sum(outcomes) / len(outcomes),
                    "final_error": final_error,
                    "task_result": task_result,
                }
            )

        success_rate = sum(outcomes) / len(tasks) if tasks else 0.0

        return BenchmarkResult(
            approach="WITH SOME (verify + retry)",
            num_samples=len(tasks),
            success_rate=success_rate,
            avg_execution_time=(total_time / total_attempts) if total_attempts else 0.0,
            avg_attempts=(total_attempts / len(tasks)) if tasks else 0.0,
            errors=errors,
            task_results=task_results,
        )

    def run_comparison(
        self,
        prompts: list[Any],
        max_retries: int = 5,
    ) -> dict[str, Any]:
        """Run both approaches and return the comparison."""
        tasks = self._normalize_tasks(prompts)

        print("=" * 60)
        print("LLM BENCHMARK: Standard vs SOME")
        print("=" * 60)
        print(f"Tasks: {len(tasks)}")
        print(f"Max retries: {max_retries}")
        print()

        print("Running standard approach...")
        standard = self.benchmark_standard(tasks)
        print(f"  Success rate: {standard.success_rate:.1%}")
        print(f"  Avg attempts: {standard.avg_attempts:.1f}")
        print(f"  Avg time: {standard.avg_execution_time:.2f}s")
        print()

        print("Running SOME approach...")
        some = self.benchmark_with_some(tasks, max_retries)
        print(f"  Success rate: {some.success_rate:.1%}")
        print(f"  Avg attempts: {some.avg_attempts:.1f}")
        print(f"  Avg time: {some.avg_execution_time:.2f}s")
        print()

        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Standard success: {standard.success_rate:.1%}")
        print(f"SOME success:      {some.success_rate:.1%}")
        print(f"Improvement:       {(some.success_rate - standard.success_rate):.1%}")
        print()

        if some.errors:
            print("Sample errors:")
            for err in some.errors[:3]:
                print(f"  - {err}")

        return {
            "standard": standard,
            "some": some,
            "improvement": some.success_rate - standard.success_rate,
        }


def demo_benchmark():
    """Demo the benchmark with a mock LLM."""
    print("\n" + "=" * 60)
    print("DEMO: LLM Benchmark with SOME")
    print("=" * 60 + "\n")

    tasks = [
        BenchmarkTask(name="fibonacci", prompt="Write a function that calculates fibonacci numbers"),
        BenchmarkTask(name="sort", prompt="Write a function that sorts a list"),
        BenchmarkTask(name="palindrome", prompt="Write a function that checks if a string is a palindrome"),
    ]

    def mock_llm(prompt: str) -> str:
        import random

        prompt_lower = prompt.lower()
        if "fibonacci" in prompt_lower and "failed verification" not in prompt_lower:
            return random.choice([
                "def fib(n): return [0,1][n]",
                "def fib(n): return [0,1,1,2,3,5,8][n%7]",
            ])
        if "fibonacci" in prompt_lower:
            return "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)"
        if "sort" in prompt_lower and "failed verification" not in prompt_lower:
            return random.choice([
                "def sort(x): return x.sort()",
                "def sort(x): return sorted(x)",
            ])
        if "sort" in prompt_lower:
            return "def sort(x):\n    return sorted(x)"
        return "def palindrome(s): return s == s[::-1]"

    def mock_verify(code: str) -> tuple[bool, Optional[str]]:
        try:
            namespace: dict[str, Any] = {}
            exec(code, namespace)

            if "fib" in namespace and namespace["fib"](7) != 13:
                return False, "fib(7) should be 13"
            if "sort" in namespace and namespace["sort"]([3, 1, 2]) != [1, 2, 3]:
                return False, "sort([3,1,2]) should return [1,2,3]"
            return True, None
        except Exception as exc:
            return False, str(exc)

    benchmark = LLMBenchmark(llm_provider=mock_llm, verify_fn=mock_verify)
    return benchmark.run_comparison(tasks, max_retries=3)


def real_llm_benchmark_example():
    """Example of the benchmark interface with a real LLM provider."""
    print("\n" + "=" * 60)
    print("REAL LLM BENCHMARK EXAMPLE")
    print("=" * 60 + "\n")

    try:
        import openai
    except ImportError:
        print("OpenAI not installed. Install with: pip install openai")
        return None

    def openai_llm(prompt: str) -> str:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content

    tasks = [
        BenchmarkTask(name="fibonacci", prompt="Write a Python function to calculate fibonacci numbers recursively"),
        BenchmarkTask(name="prime", prompt="Write a Python function to check if a number is prime"),
    ]

    benchmark = LLMBenchmark(llm_provider=openai_llm)
    return benchmark.run_comparison(tasks, max_retries=3)


if __name__ == "__main__":
    results = demo_benchmark()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    if results["improvement"] > 0:
        print(f"SOME improved success rate by {results['improvement']:.1%}")
    elif results["improvement"] < 0:
        print(f"SOME decreased success rate by {abs(results['improvement']):.1%}")
    else:
        print("No difference in this run")
