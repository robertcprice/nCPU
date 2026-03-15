#!/usr/bin/env python3
"""
Qwen benchmark runner for baseline vs SOME-style self-optimization.

This runner focuses on deterministic tasks so the output is useful:
- Coding tasks are scored by execution against test cases.
- Reasoning tasks are scored by exact answers, not vague prose quality.
- "SOME" is measured as verify + feedback + retry, compared to pass@1.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
import statistics
import sys
from typing import Any, Callable, Optional

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ncpu.self_optimizing.code_verifier import (
    BINARY_SEARCH_TESTS,
    CODE_PROMPTS,
    FACTORIAL_TESTS,
    FIBONACCI_TESTS,
    GRAPH_TESTS,
    HARD_CODE_PROMPTS,
    IS_PRIME_TESTS,
    LONGEST_SUBSTRING_TESTS,
    MERGE_INTERVALS_TESTS,
    QUICK_SORT_TESTS,
    CodeVerifier,
    VerificationResult,
    verify_lru_cache,
    verify_topological_sort,
)
from ncpu.self_optimizing.controller_runtime import resolve_controller_runtime
from ncpu.self_optimizing.llm_benchmark import BenchmarkTask, LLMBenchmark, ProviderResponse
from ncpu.self_optimizing.llm_provider import LLMProviderFactory
from ncpu.self_optimizing.reasoning_analyzer import ReasoningAnalyzer


HUMANEVAL_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "HumanEval.jsonl"
)

BENCHMARK_SYSTEM_PROMPT = (
    "You are participating in a deterministic benchmark. "
    "Follow the requested output format exactly."
)


@dataclass
class ReasoningVerification:
    """Structured verification result for reasoning tasks."""

    success: bool
    error: Optional[str]
    parsed_answer: Any = None
    expected_answer: Any = None
    explanation: str = ""
    raw_payload: Optional[dict[str, Any]] = None


@dataclass
class ReasoningTaskSpec:
    """Definition for a deterministic reasoning task."""

    name: str
    prompt: str
    expected_answer: Any
    normalizer: Callable[[Any], Any]


def extract_json_object(text: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Extract the first JSON object from a model response."""
    stripped = text.strip()
    if not stripped:
        return None, "empty response"

    fenced = stripped
    if stripped.startswith("```"):
        parts = stripped.split("```")
        for part in parts:
            candidate = part.strip()
            if not candidate or candidate.lower() == "json":
                continue
            if candidate.lower().startswith("json\n"):
                candidate = candidate[5:]
            fenced = candidate.strip()
            break

    decoder = json.JSONDecoder()
    for index, char in enumerate(fenced):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(fenced[index:])
            if isinstance(payload, dict):
                return payload, None
        except json.JSONDecodeError:
            continue

    return None, "response did not contain a valid JSON object"


def _identity(value: Any) -> Any:
    return value


def _normalize_coordinate(answer: Any) -> list[int]:
    if not isinstance(answer, (list, tuple)) or len(answer) != 2:
        raise ValueError("answer must be a coordinate pair")
    return [int(answer[0]), int(answer[1])]


def _normalize_clock(value: Any) -> str:
    return str(value).strip()


def make_reasoning_verifier(spec: ReasoningTaskSpec) -> Callable[[str], ReasoningVerification]:
    expected = spec.normalizer(spec.expected_answer)

    def verify(text: str) -> ReasoningVerification:
        payload, error = extract_json_object(text)
        if error:
            return ReasoningVerification(success=False, error=error)

        if "answer" not in payload:
            return ReasoningVerification(
                success=False,
                error="JSON response is missing the 'answer' field",
                raw_payload=payload,
                explanation=str(payload.get("explanation", "")),
            )

        explanation = str(payload.get("explanation", "")).strip()
        try:
            actual = spec.normalizer(payload["answer"])
        except Exception as exc:
            return ReasoningVerification(
                success=False,
                error=f"could not normalize answer: {exc}",
                parsed_answer=payload.get("answer"),
                expected_answer=expected,
                explanation=explanation,
                raw_payload=payload,
            )

        if actual != expected:
            return ReasoningVerification(
                success=False,
                error=f"expected {expected!r}, got {actual!r}",
                parsed_answer=actual,
                expected_answer=expected,
                explanation=explanation,
                raw_payload=payload,
            )

        return ReasoningVerification(
            success=True,
            error=None,
            parsed_answer=actual,
            expected_answer=expected,
            explanation=explanation,
            raw_payload=payload,
        )

    return verify


def build_code_feedback(_original_prompt: str, attempt: dict[str, Any]) -> str:
    verification = attempt.get("verification") or {}
    failed_tests = [
        item for item in verification.get("test_results", [])
        if not item.get("passed", False)
    ]

    details: list[str] = []
    if attempt.get("error"):
        details.append(attempt["error"])

    for item in failed_tests[:3]:
        if "error" in item:
            details.append(f"test {item.get('test', '?')} raised {item['error']}")
        else:
            details.append(
                f"test {item.get('test', '?')} expected {item.get('expected')!r} "
                f"but got {item.get('actual')!r}"
            )

    summary = "; ".join(details) if details else "verification failed"
    return (
        f"{_original_prompt}\n\n"
        "Your previous code did not pass verification.\n"
        f"Failure details: {summary}\n"
        "Return only corrected raw Python code."
    )


def build_reasoning_feedback(original_prompt: str, attempt: dict[str, Any]) -> str:
    verification = attempt.get("verification") or {}
    failure = attempt.get("error") or verification.get("error") or "verification failed"
    return (
        f"{original_prompt}\n\n"
        "Your previous answer failed verification.\n"
        f"Failure details: {failure}\n"
        "Return JSON only with keys 'answer' and 'explanation'."
    )


def build_coding_tasks(repeats: int = 1) -> list[BenchmarkTask]:
    verifier = CodeVerifier()

    specs = [
        ("fibonacci", CODE_PROMPTS["fibonacci"], FIBONACCI_TESTS),
        ("factorial", CODE_PROMPTS["factorial"], FACTORIAL_TESTS),
        ("is_prime", CODE_PROMPTS["is_prime"], IS_PRIME_TESTS),
        ("binary_search", CODE_PROMPTS["binary_search"], BINARY_SEARCH_TESTS),
        ("quick_sort", CODE_PROMPTS["quick_sort"], QUICK_SORT_TESTS),
    ]

    tasks: list[BenchmarkTask] = []
    for _ in range(repeats):
        for name, prompt, tests in specs:
            tasks.append(
                BenchmarkTask(
                    name=name,
                    category="coding",
                    prompt=(
                        "Write raw Python only. No markdown, no backticks, no explanation.\n\n"
                        f"{prompt}"
                    ),
                    verify_fn=lambda code, test_cases=tests, local_verifier=verifier: local_verifier.verify(
                        code,
                        test_cases,
                    ),
                    feedback_builder=build_code_feedback,
                    response_format="raw Python code",
                )
            )
    return tasks


def build_hard_coding_tasks(repeats: int = 1) -> list[BenchmarkTask]:
    """Build hard coding tasks that challenge smaller models."""
    verifier = CodeVerifier()

    # Function-based tasks using existing test suites
    function_specs = [
        ("dijkstra", HARD_CODE_PROMPTS["dijkstra"], GRAPH_TESTS[0]["tests"]),
        ("merge_intervals", HARD_CODE_PROMPTS["merge_intervals"], MERGE_INTERVALS_TESTS[0]["tests"]),
        ("longest_substring", HARD_CODE_PROMPTS["longest_substring"], LONGEST_SUBSTRING_TESTS[0]["tests"]),
    ]

    tasks: list[BenchmarkTask] = []
    for _ in range(repeats):
        for name, prompt, tests in function_specs:
            tasks.append(
                BenchmarkTask(
                    name=name,
                    category="coding",
                    prompt=(
                        "Write raw Python only. No markdown, no backticks, no explanation.\n\n"
                        f"{prompt}"
                    ),
                    verify_fn=lambda code, test_cases=tests, local_verifier=verifier: local_verifier.verify(
                        code,
                        test_cases,
                    ),
                    feedback_builder=build_code_feedback,
                    response_format="raw Python code",
                )
            )

        # LRU cache: class-based, custom verifier
        tasks.append(
            BenchmarkTask(
                name="lru_cache",
                category="coding",
                prompt=(
                    "Write raw Python only. No markdown, no backticks, no explanation.\n\n"
                    f"{HARD_CODE_PROMPTS['lru_cache']}"
                ),
                verify_fn=verify_lru_cache,
                feedback_builder=build_code_feedback,
                response_format="raw Python code",
            )
        )

        # Topological sort: custom verifier (checks validity, not exact order)
        tasks.append(
            BenchmarkTask(
                name="topological_sort",
                category="coding",
                prompt=(
                    "Write raw Python only. No markdown, no backticks, no explanation.\n\n"
                    f"{HARD_CODE_PROMPTS['topological_sort']}"
                ),
                verify_fn=verify_topological_sort,
                feedback_builder=build_code_feedback,
                response_format="raw Python code",
            )
        )

    return tasks


def build_reasoning_tasks(repeats: int = 1) -> list[BenchmarkTask]:
    specs = [
        ReasoningTaskSpec(
            name="weighted_path",
            prompt=(
                "A weighted graph has edges: A-B=4, A-C=2, C-B=1, B-D=5, C-D=8. "
                "What is the shortest distance from A to D?"
            ),
            expected_answer=8,
            normalizer=int,
        ),
        ReasoningTaskSpec(
            name="meeting_slot",
            prompt=(
                "A person is busy from 09:00-09:45, 10:15-11:00, and 11:30-12:00. "
                "What is the earliest 30-minute free slot that starts at or after 09:00 "
                "and ends by 12:00? Return the start time in HH:MM."
            ),
            expected_answer="09:45",
            normalizer=_normalize_clock,
        ),
        ReasoningTaskSpec(
            name="robot_grid",
            prompt=(
                "A robot starts at [0, 0] facing north. It executes: forward 2, right, "
                "forward 3, right, forward 1, left, forward 2. What is the final coordinate?"
            ),
            expected_answer=[5, 1],
            normalizer=_normalize_coordinate,
        ),
        ReasoningTaskSpec(
            name="logic_constraint",
            prompt=(
                "Exactly one of A, B, and C is true. If A is true then B is true. "
                "If C is true then B is false. Which variable must be true?"
            ),
            expected_answer="C",
            normalizer=lambda value: str(value).strip().upper(),
        ),
        ReasoningTaskSpec(
            name="price_reasoning",
            prompt=(
                "An item costs 80 dollars. Apply a 25% discount, then apply 10% sales tax. "
                "What is the final price?"
            ),
            expected_answer=66.0,
            normalizer=lambda value: round(float(value), 2),
        ),
    ]

    tasks: list[BenchmarkTask] = []
    for _ in range(repeats):
        for spec in specs:
            tasks.append(
                BenchmarkTask(
                    name=spec.name,
                    category="reasoning",
                    prompt=(
                        "Solve the following reasoning task.\n"
                        "Return JSON only in the format "
                        '{"answer": <final answer>, "explanation": "<brief visible reasoning summary>"}.\n'
                        "Do not include markdown or extra text.\n\n"
                        f"{spec.prompt}"
                    ),
                    verify_fn=make_reasoning_verifier(spec),
                    feedback_builder=build_reasoning_feedback,
                    response_format="JSON only with keys 'answer' and 'explanation'",
                )
            )
    return tasks


def build_hard_reasoning_tasks(repeats: int = 1) -> list[BenchmarkTask]:
    """Build hard reasoning tasks where small models struggle."""
    specs = [
        ReasoningTaskSpec(
            name="seating_constraint",
            prompt=(
                "Five people A, B, C, D, E sit in seats numbered 1 through 5. "
                "B is in seat 2. A is in seat 5. D sits immediately to the right of C "
                "(i.e., D's seat number is exactly one more than C's). "
                "What seat number is E in?"
            ),
            expected_answer=1,
            normalizer=int,
        ),
        ReasoningTaskSpec(
            name="probability_calc",
            prompt=(
                "A bag contains 3 red, 4 blue, and 2 green marbles (9 total). "
                "Two marbles are drawn without replacement. "
                "What is the probability that both marbles are the same color? "
                "Express your answer as a simplified fraction like '5/18'."
            ),
            expected_answer="5/18",
            normalizer=lambda v: str(v).strip(),
        ),
        ReasoningTaskSpec(
            name="modular_arithmetic",
            prompt=(
                "What is the remainder when 7^123 is divided by 13? "
                "Hint: consider Fermat's little theorem."
            ),
            expected_answer=5,
            normalizer=int,
        ),
        ReasoningTaskSpec(
            name="train_meeting",
            prompt=(
                "Train A leaves a station at 8:15 AM traveling east at 60 mph. "
                "Train B leaves the same station at 9:00 AM traveling east at 90 mph. "
                "At what time does Train B catch up to Train A? "
                "Return the time in HH:MM format (e.g., '10:30')."
            ),
            expected_answer="10:30",
            normalizer=_normalize_clock,
        ),
        ReasoningTaskSpec(
            name="counting_strings",
            prompt=(
                "How many 4-letter strings can be formed from the set {A, B, C} "
                "such that no two adjacent letters are the same? "
                "For example, 'ABAB' is valid but 'AABB' is not."
            ),
            expected_answer=24,
            normalizer=int,
        ),
    ]

    tasks: list[BenchmarkTask] = []
    for _ in range(repeats):
        for spec in specs:
            tasks.append(
                BenchmarkTask(
                    name=spec.name,
                    category="reasoning",
                    prompt=(
                        "Solve the following reasoning task.\n"
                        "Return JSON only in the format "
                        '{"answer": <final answer>, "explanation": "<brief visible reasoning summary>"}.\n'
                        "Do not include markdown or extra text.\n\n"
                        f"{spec.prompt}"
                    ),
                    verify_fn=make_reasoning_verifier(spec),
                    feedback_builder=build_reasoning_feedback,
                    response_format="JSON only with keys 'answer' and 'explanation'",
                )
            )
    return tasks


def _load_humaneval() -> list[dict[str, Any]]:
    """Load HumanEval problems from bundled JSONL."""
    problems = []
    with open(HUMANEVAL_DATA_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def _make_humaneval_verifier(
    prompt: str, entry_point: str, test_code: str,
) -> Callable[[str], VerificationResult]:
    """Build a verify_fn for a single HumanEval problem."""
    extractor = CodeVerifier()

    def verify(code: str) -> VerificationResult:
        prepared = extractor.extract_code(code)

        # If model only output the body, prepend the function signature
        if f"def {entry_point}" not in prepared:
            prepared = prompt + prepared

        try:
            compile(prepared, "<generated>", "exec")
        except SyntaxError as e:
            return VerificationResult(
                success=False,
                error=f"Syntax error at line {e.lineno}: {e.msg}",
                output=None,
                test_results=[],
            )

        try:
            ns: dict = {}
            exec(prepared, ns)
        except Exception as e:
            return VerificationResult(
                success=False,
                error=f"Runtime error: {e}",
                output=None,
                test_results=[],
            )

        if entry_point not in ns:
            return VerificationResult(
                success=False,
                error=f"Function '{entry_point}' not found in generated code",
                output=None,
                test_results=[],
            )

        # Run the HumanEval test suite
        try:
            test_ns = dict(ns)
            exec(test_code, test_ns)
            test_ns["check"](ns[entry_point])
            return VerificationResult(
                success=True,
                error=None,
                output=None,
                test_results=[{"test": "check", "passed": True}],
            )
        except AssertionError as e:
            return VerificationResult(
                success=False,
                error=f"Assertion failed: {e}",
                output=None,
                test_results=[{"test": "check", "passed": False, "error": str(e)}],
            )
        except Exception as e:
            return VerificationResult(
                success=False,
                error=f"Test error: {type(e).__name__}: {e}",
                output=None,
                test_results=[{"test": "check", "passed": False, "error": str(e)}],
            )

    return verify


def _humaneval_feedback(original_prompt: str, attempt: dict[str, Any]) -> str:
    """Feedback builder for HumanEval tasks."""
    verification = attempt.get("verification") or {}
    error = attempt.get("error") or verification.get("error") or "test failed"
    return (
        f"{original_prompt}\n\n"
        "Your previous code did not pass the test suite.\n"
        f"Error: {error}\n"
        "Return only corrected raw Python code."
    )


def build_humaneval_tasks(repeats: int = 1) -> list[BenchmarkTask]:
    """Build tasks from OpenAI's HumanEval benchmark (164 problems)."""
    problems = _load_humaneval()

    tasks: list[BenchmarkTask] = []
    for _ in range(repeats):
        for problem in problems:
            task_id = problem["task_id"]
            short_name = task_id.replace("HumanEval/", "he_")
            entry_point = problem["entry_point"]
            prompt = problem["prompt"]
            test_code = problem["test"]

            tasks.append(
                BenchmarkTask(
                    name=short_name,
                    category="coding",
                    prompt=(
                        "Complete the following Python function. "
                        "Write raw Python only. No markdown, no backticks, no explanation.\n\n"
                        f"{prompt}"
                    ),
                    verify_fn=_make_humaneval_verifier(prompt, entry_point, test_code),
                    feedback_builder=_humaneval_feedback,
                    response_format="raw Python code",
                )
            )
    return tasks


def build_tasks(
    include_coding: bool = True,
    include_reasoning: bool = True,
    repeats: int = 1,
    difficulty: str = "easy",
    benchmark: str = "custom",
) -> list[BenchmarkTask]:
    if benchmark == "humaneval":
        return build_humaneval_tasks(repeats=repeats)

    tasks: list[BenchmarkTask] = []
    if difficulty in ("easy", "all"):
        if include_coding:
            tasks.extend(build_coding_tasks(repeats=repeats))
        if include_reasoning:
            tasks.extend(build_reasoning_tasks(repeats=repeats))
    if difficulty in ("hard", "all"):
        if include_coding:
            tasks.extend(build_hard_coding_tasks(repeats=repeats))
        if include_reasoning:
            tasks.extend(build_hard_reasoning_tasks(repeats=repeats))
    return tasks


def build_provider(
    provider_name: str,
    model: str,
    temperature: float,
    base_url: str,
    max_tokens: Optional[int] = None,
    provider_kwargs: Optional[dict[str, Any]] = None,
    request_timeout: float = 240.0,
) -> Callable[[str], ProviderResponse]:
    """Build an LLM provider that returns normalized text plus metadata."""
    if provider_name == "local":
        import requests

        endpoint = f"{base_url.rstrip('/')}/api/generate"

        def provider(prompt: str) -> ProviderResponse:
            response = requests.post(
                endpoint,
                json={
                    "model": model,
                    "system": BENCHMARK_SYSTEM_PROMPT,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens or 2048,
                    },
                },
                timeout=request_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            metadata = {
                key: payload.get(key)
                for key in (
                    "total_duration",
                    "load_duration",
                    "prompt_eval_count",
                    "prompt_eval_duration",
                    "eval_count",
                    "eval_duration",
                )
                if payload.get(key) is not None
            }
            return ProviderResponse(
                text=str(payload.get("response", "")).strip(),
                metadata=metadata,
            )

        return provider

    raw_provider = LLMProviderFactory.create_provider(
        provider=provider_name,
        model=model,
        temperature=temperature,
        base_url=base_url,
        max_tokens=max_tokens or 2048,
        request_timeout=request_timeout,
        **(provider_kwargs or {}),
    )

    def provider(prompt: str) -> ProviderResponse:
        raw_response = raw_provider(f"{BENCHMARK_SYSTEM_PROMPT}\n\n{prompt}")
        if isinstance(raw_response, ProviderResponse):
            return raw_response
        if isinstance(raw_response, dict):
            if "text" in raw_response:
                metadata = {key: value for key, value in raw_response.items() if key != "text"}
                return ProviderResponse(text=str(raw_response["text"]), metadata=metadata)
            return ProviderResponse(text=str(raw_response))
        return ProviderResponse(text=str(raw_response))

    return provider


def _mean(values: list[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def summarize_result(result: Any) -> dict[str, Any]:
    all_tasks = result.task_results
    all_attempts = [
        attempt
        for task in all_tasks
        for attempt in task["attempt_details"]
    ]

    categories: dict[str, dict[str, Any]] = {}
    for category in ("coding", "reasoning"):
        subset = [task for task in all_tasks if task["category"] == category]
        if not subset:
            continue
        categories[category] = {
            "num_samples": len(subset),
            "success_rate": sum(task["success"] for task in subset) / len(subset),
            "avg_attempts": sum(task["attempts"] for task in subset) / len(subset),
        }

    eval_counts = [
        float(attempt["metadata"]["eval_count"])
        for attempt in all_attempts
        if attempt.get("metadata", {}).get("eval_count") is not None
    ]
    throughputs = []
    for attempt in all_attempts:
        metadata = attempt.get("metadata", {})
        eval_count = metadata.get("eval_count")
        eval_duration = metadata.get("eval_duration")
        if eval_count and eval_duration:
            throughputs.append(float(eval_count) / (float(eval_duration) / 1_000_000_000))

    task_rollup: dict[str, dict[str, Any]] = {}
    for task in all_tasks:
        bucket = task_rollup.setdefault(task["name"], {"runs": 0, "successes": 0, "avg_attempts": []})
        bucket["runs"] += 1
        bucket["successes"] += int(task["success"])
        bucket["avg_attempts"].append(task["attempts"])

    by_task = {
        name: {
            "runs": values["runs"],
            "success_rate": values["successes"] / values["runs"],
            "avg_attempts": statistics.mean(values["avg_attempts"]),
        }
        for name, values in sorted(task_rollup.items())
    }

    return {
        "num_samples": result.num_samples,
        "success_rate": result.success_rate,
        "avg_attempts": result.avg_attempts,
        "avg_attempt_time_seconds": result.avg_execution_time,
        "categories": categories,
        "avg_output_tokens": _mean(eval_counts),
        "avg_tokens_per_second": _mean(throughputs),
        "by_task": by_task,
    }


def summarize_reasoning_quality(task_results: list[dict[str, Any]]) -> dict[str, Any]:
    analyzer = ReasoningAnalyzer()
    metrics: list[dict[str, Any]] = []

    for task in task_results:
        if task["category"] != "reasoning" or not task["attempt_details"]:
            continue

        final_attempt = task["attempt_details"][-1]
        verification = final_attempt.get("verification") or {}
        explanation = str(verification.get("explanation", "")).strip()
        if not explanation:
            continue

        analyzed = analyzer.analyze_output(
            explanation,
            final_attempt["elapsed_seconds"] * 1000,
        )
        metrics.append(
            {
                "explanation_tokens": analyzed.total_tokens,
                "reasoning_depth": analyzed.reasoning_depth,
                "steps_identified": analyzed.steps_identified,
                "visible_reasoning_markers": analyzed.chain_of_thought_detected,
            }
        )
        final_attempt["reasoning_metrics"] = metrics[-1]

    if not metrics:
        return {}

    return {
        "num_reasoning_samples": len(metrics),
        "avg_explanation_tokens": _mean([item["explanation_tokens"] for item in metrics]),
        "avg_reasoning_depth": _mean([item["reasoning_depth"] for item in metrics]),
        "avg_steps_identified": _mean([item["steps_identified"] for item in metrics]),
        "visible_reasoning_marker_rate": (
            sum(item["visible_reasoning_markers"] for item in metrics) / len(metrics)
        ),
    }


def build_delta_report(standard_summary: dict[str, Any], some_summary: dict[str, Any]) -> dict[str, Any]:
    delta = {
        "overall_success_delta": some_summary["success_rate"] - standard_summary["success_rate"],
        "avg_attempts_delta": some_summary["avg_attempts"] - standard_summary["avg_attempts"],
    }

    for category in ("coding", "reasoning"):
        standard_category = standard_summary.get("categories", {}).get(category)
        some_category = some_summary.get("categories", {}).get(category)
        if not standard_category or not some_category:
            continue
        delta[f"{category}_success_delta"] = (
            some_category["success_rate"] - standard_category["success_rate"]
        )

    improved_tasks = []
    for task_name, standard_task in standard_summary.get("by_task", {}).items():
        some_task = some_summary.get("by_task", {}).get(task_name)
        if not some_task:
            continue
        improvement = some_task["success_rate"] - standard_task["success_rate"]
        if improvement > 0:
            improved_tasks.append(
                {
                    "name": task_name,
                    "success_delta": improvement,
                    "standard_success_rate": standard_task["success_rate"],
                    "some_success_rate": some_task["success_rate"],
                }
            )

    delta["improved_tasks"] = improved_tasks
    return delta


def default_trajectory_dir(model: str) -> str:
    safe_model = model.replace("/", "--").replace(":", "_")
    return os.path.join(
        PROJECT_ROOT,
        "benchmarks",
        "internal_trajectories",
        "qwen_benchmark",
        safe_model,
    )


def run_model_benchmark(
    provider_name: Optional[str],
    model: Optional[str],
    controller_bundle_path: Optional[str] = None,
    action_provider_name: Optional[str] = None,
    action_model: Optional[str] = None,
    action_temperature: Optional[float] = None,
    action_base_url: Optional[str] = None,
    temperature: Optional[float] = 0.0,
    max_retries: int = 3,
    include_coding: bool = True,
    include_reasoning: bool = True,
    repeats: int = 1,
    base_url: Optional[str] = "http://localhost:11434",
    request_timeout: float = 240.0,
    trajectory_dir: Optional[str] = None,
    difficulty: str = "easy",
) -> dict[str, Any]:
    tasks = build_tasks(
        include_coding=include_coding,
        include_reasoning=include_reasoning,
        repeats=repeats,
        difficulty=difficulty,
    )
    runtime = resolve_controller_runtime(
        controller_bundle_path=controller_bundle_path,
        provider_name=provider_name,
        model=model,
        temperature=temperature,
        base_url=base_url,
        request_timeout=request_timeout,
        action_provider_name=action_provider_name,
        action_model=action_model,
        action_temperature=action_temperature,
        action_base_url=action_base_url,
        default_provider="local",
        default_temperature=0.0,
        default_base_url="http://localhost:11434",
        default_request_timeout=240.0,
    )
    provider = build_provider(
        provider_name=runtime.provider_name,
        model=runtime.model,
        temperature=runtime.temperature,
        base_url=runtime.base_url,
        max_tokens=runtime.max_tokens,
        provider_kwargs=runtime.provider_kwargs,
        request_timeout=runtime.request_timeout or 240.0,
    )
    action_provider = None
    if runtime.action_provider_name and runtime.action_model:
        action_provider = build_provider(
            provider_name=runtime.action_provider_name,
            model=runtime.action_model,
            temperature=runtime.action_temperature or runtime.temperature,
            base_url=runtime.action_base_url or runtime.base_url,
            max_tokens=runtime.action_max_tokens,
            provider_kwargs=runtime.action_provider_kwargs,
            request_timeout=runtime.action_request_timeout or runtime.request_timeout or 240.0,
        )
    benchmark = LLMBenchmark(llm_provider=provider, action_provider=action_provider)
    some_trajectory_dir = trajectory_dir or default_trajectory_dir(runtime.model)

    print(f"\n{'=' * 72}")
    print(f"Benchmark: {runtime.provider_name} / {runtime.model}")
    print(f"{'=' * 72}")
    print(f"Tasks: {len(tasks)} | Coding: {include_coding} | Reasoning: {include_reasoning} | Repeats: {repeats}")

    standard = benchmark.benchmark_standard(tasks)
    some = benchmark.benchmark_with_some(
        tasks,
        max_retries=max_retries,
        use_buffered_controller=True,
        trajectory_dir=some_trajectory_dir,
        controller_bundle_path=controller_bundle_path,
    )

    standard_summary = summarize_result(standard)
    some_summary = summarize_result(some)
    standard_reasoning = summarize_reasoning_quality(standard.task_results)
    some_reasoning = summarize_reasoning_quality(some.task_results)
    delta = build_delta_report(standard_summary, some_summary)

    print(
        "Baseline: "
        f"{standard_summary['success_rate']:.1%} overall | "
        f"{standard_summary.get('categories', {}).get('coding', {}).get('success_rate', 0.0):.1%} coding | "
        f"{standard_summary.get('categories', {}).get('reasoning', {}).get('success_rate', 0.0):.1%} reasoning | "
        f"{standard_summary['avg_attempt_time_seconds']:.2f}s/attempt"
    )
    print(
        "SOME:     "
        f"{some_summary['success_rate']:.1%} overall | "
        f"{some_summary.get('categories', {}).get('coding', {}).get('success_rate', 0.0):.1%} coding | "
        f"{some_summary.get('categories', {}).get('reasoning', {}).get('success_rate', 0.0):.1%} reasoning | "
        f"{some_summary['avg_attempts']:.2f} attempts/task"
    )
    print(f"Delta:    {delta['overall_success_delta']:+.1%} overall success")
    if delta.get("improved_tasks"):
        print("Improved tasks: " + ", ".join(item["name"] for item in delta["improved_tasks"]))
    if some_reasoning:
        print(
            "Reasoning: "
            f"{some_reasoning.get('avg_explanation_tokens', 0.0):.1f} explanation tokens | "
            f"{some_reasoning.get('avg_steps_identified', 0.0):.1f} step markers | "
            f"{some_reasoning.get('visible_reasoning_marker_rate', 0.0):.1%} visible reasoning markers"
        )

    return {
        "provider": runtime.provider_name,
        "model": runtime.model,
        "controller_bundle_path": controller_bundle_path,
        "temperature": runtime.temperature,
        "max_tokens": runtime.max_tokens,
        "request_timeout": runtime.request_timeout,
        "provider_kwargs": runtime.provider_kwargs,
        "max_retries": max_retries,
        "repeats": repeats,
        "baseline": {
            "summary": standard_summary,
            "reasoning_quality": standard_reasoning,
            "task_results": standard.task_results,
        },
        "some": {
            "summary": some_summary,
            "reasoning_quality": some_reasoning,
            "task_results": some.task_results,
            "trajectory_dir": some_trajectory_dir,
            "action_policy": {
                "provider": runtime.action_provider_name,
                "model": runtime.action_model,
                "temperature": runtime.action_temperature,
                "base_url": runtime.action_base_url,
                "max_tokens": runtime.action_max_tokens,
                "request_timeout": runtime.action_request_timeout,
                "provider_kwargs": runtime.action_provider_kwargs,
            },
        },
        "delta": delta,
    }


def run_benchmark(
    provider_name: Optional[str],
    models: list[str],
    controller_bundle_path: Optional[str] = None,
    action_provider_name: Optional[str] = None,
    action_model: Optional[str] = None,
    action_temperature: Optional[float] = None,
    action_base_url: Optional[str] = None,
    temperature: Optional[float] = 0.0,
    max_retries: int = 3,
    include_coding: bool = True,
    include_reasoning: bool = True,
    repeats: int = 1,
    base_url: Optional[str] = "http://localhost:11434",
    request_timeout: float = 240.0,
    trajectory_root: Optional[str] = None,
    difficulty: str = "easy",
) -> dict[str, Any]:
    model_reports = [
        run_model_benchmark(
            provider_name=provider_name,
            model=model,
            controller_bundle_path=controller_bundle_path,
            action_provider_name=action_provider_name,
            action_model=action_model,
            action_temperature=action_temperature,
            action_base_url=action_base_url,
            temperature=temperature,
            max_retries=max_retries,
            include_coding=include_coding,
            include_reasoning=include_reasoning,
            repeats=repeats,
            base_url=base_url,
            request_timeout=request_timeout,
            trajectory_dir=(
                os.path.join(trajectory_root, model.replace("/", "--").replace(":", "_"))
                if trajectory_root
                else None
            ),
            difficulty=difficulty,
        )
        for model in models
    ]

    leaderboard = sorted(
        [
            {
                "model": report["model"],
                "baseline_success_rate": report["baseline"]["summary"]["success_rate"],
                "some_success_rate": report["some"]["summary"]["success_rate"],
                "success_delta": report["delta"]["overall_success_delta"],
                "avg_attempt_time_seconds": report["some"]["summary"]["avg_attempt_time_seconds"],
            }
            for report in model_reports
        ],
        key=lambda item: (
            item["some_success_rate"],
            item["success_delta"],
            -item["avg_attempt_time_seconds"],
        ),
        reverse=True,
    )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": model_reports[0]["provider"] if model_reports else provider_name,
        "controller_bundle_path": controller_bundle_path,
        "action_policy": {
            "provider": model_reports[0]["some"]["action_policy"]["provider"] if model_reports else action_provider_name,
            "model": model_reports[0]["some"]["action_policy"]["model"] if model_reports else action_model,
            "temperature": model_reports[0]["some"]["action_policy"]["temperature"] if model_reports else action_temperature,
            "base_url": model_reports[0]["some"]["action_policy"]["base_url"] if model_reports else action_base_url,
            "max_tokens": model_reports[0]["some"]["action_policy"]["max_tokens"] if model_reports else None,
            "request_timeout": model_reports[0]["some"]["action_policy"]["request_timeout"] if model_reports else None,
            "provider_kwargs": model_reports[0]["some"]["action_policy"]["provider_kwargs"] if model_reports else {},
        },
        "models": model_reports,
        "leaderboard": leaderboard,
    }


def parse_models(args: argparse.Namespace) -> list[str]:
    models: list[str] = []
    if args.models:
        models.extend(item.strip() for item in args.models.split(",") if item.strip())
    if args.model:
        models.extend(args.model)
    if models:
        return models
    if args.controller_bundle:
        runtime = resolve_controller_runtime(
            controller_bundle_path=args.controller_bundle,
            provider_name=args.provider,
            model=None,
            temperature=args.temp,
            base_url=args.base_url,
            request_timeout=args.request_timeout,
            action_provider_name=args.action_provider,
            action_model=args.action_model,
            action_temperature=args.action_temp,
            action_base_url=args.action_base_url,
            default_provider="local",
            default_temperature=0.0,
            default_base_url="http://localhost:11434",
            default_request_timeout=240.0,
        )
        return [runtime.model]
    return ["qwen3.5:4b", "qwen3.5:9b", "qwen3.5:27b"]


def print_leaderboard(report: dict[str, Any]) -> None:
    print(f"\n{'=' * 72}")
    print("Leaderboard")
    print(f"{'=' * 72}")
    for item in report["leaderboard"]:
        print(
            f"{item['model']}: SOME {item['some_success_rate']:.1%} | "
            f"baseline {item['baseline_success_rate']:.1%} | "
            f"delta {item['success_delta']:+.1%} | "
            f"{item['avg_attempt_time_seconds']:.2f}s/attempt"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen benchmark runner with baseline vs SOME comparison")
    parser.add_argument(
        "--provider",
        "-p",
        default=None,
        choices=["local", "hf_local", "hf_segmented_cache", "hf_fast_weights", "qwen", "deepseek", "together", "replicate", "openai", "anthropic"],
        help="Provider to use",
    )
    parser.add_argument(
        "--controller-bundle",
        help="Optional controller bundle manifest that provides response/action model defaults.",
    )
    parser.add_argument(
        "--action-provider",
        choices=["local", "hf_local", "hf_segmented_cache", "hf_fast_weights", "qwen", "deepseek", "together", "replicate", "openai", "anthropic"],
        help="Optional separate provider for hidden action-policy decisions.",
    )
    parser.add_argument(
        "--model",
        "-m",
        action="append",
        help="Model name. Repeat the flag to benchmark multiple models.",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated model list. Used in addition to repeated --model flags.",
    )
    parser.add_argument(
        "--action-model",
        help="Optional separate hidden action-policy model. Defaults to the response model when --action-provider is set.",
    )
    parser.add_argument(
        "--temp",
        "-t",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--action-temp",
        type=float,
        help="Optional sampling temperature for the action-policy model.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry budget for SOME mode",
    )
    parser.add_argument(
        "--repeats",
        "-n",
        type=int,
        default=1,
        help="Repeat each task this many times",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for local/OpenAI-compatible providers",
    )
    parser.add_argument(
        "--action-base-url",
        help="Optional base URL for the action-policy provider.",
    )
    parser.add_argument(
        "--tasks",
        default="coding,reasoning",
        help="Comma-separated task categories to run: coding, reasoning",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=240.0,
        help="Per-request timeout in seconds for local/OpenAI-compatible providers",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path",
    )
    parser.add_argument(
        "--trajectory-root",
        help="Optional directory root for SOME hidden trajectory JSONL artifacts",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "hard", "all"],
        default="easy",
        help="Task difficulty: easy (original 10 tasks), hard (10 harder tasks), all (20 tasks)",
    )

    args = parser.parse_args()
    requested_tasks = {item.strip().lower() for item in args.tasks.split(",") if item.strip()}
    include_coding = "coding" in requested_tasks
    include_reasoning = "reasoning" in requested_tasks

    if not include_coding and not include_reasoning:
        raise SystemExit("No valid task categories selected. Use coding, reasoning, or both.")

    report = run_benchmark(
        provider_name=args.provider,
        models=parse_models(args),
        controller_bundle_path=args.controller_bundle,
        action_provider_name=args.action_provider,
        action_model=args.action_model,
        action_temperature=args.action_temp,
        action_base_url=args.action_base_url,
        temperature=args.temp,
        max_retries=args.max_retries,
        include_coding=include_coding,
        include_reasoning=include_reasoning,
        repeats=args.repeats,
        base_url=args.base_url,
        request_timeout=args.request_timeout,
        trajectory_root=args.trajectory_root,
        difficulty=args.difficulty,
    )

    print_leaderboard(report)

    output_path = args.output
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(PROJECT_ROOT, f"qwen_benchmark_{timestamp}.json")

    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"\nSaved detailed report to {output_path}")


if __name__ == "__main__":
    main()
