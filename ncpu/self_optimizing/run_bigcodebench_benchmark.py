#!/usr/bin/env python3
"""
Run BigCodeBench with baseline vs SOME-style verify/retry.

This targets the official BigCodeBench dataset/checker rather than custom tasks.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import requests

from ncpu.self_optimizing.controller_runtime import resolve_controller_runtime
from ncpu.self_optimizing.llm_benchmark import BenchmarkTask, LLMBenchmark, ProviderResponse
from ncpu.self_optimizing.llm_provider import LLMProviderFactory

try:
    from bigcodebench.data import get_bigcodebench, get_bigcodebench_hash
    from bigcodebench.data.utils import CACHE_DIR
    from bigcodebench.eval import PASS, untrusted_check
    from bigcodebench.gen.util import trusted_check
    from bigcodebench.sanitize import sanitize
except ImportError as exc:  # pragma: no cover - handled at runtime on benchmark boxes
    raise SystemExit(
        "BigCodeBench is required. Install the lightweight local-eval set with:\n"
        "python3 -m pip install --no-deps bigcodebench appdirs tempdir wget pqdm termcolor tqdm "
        "tree_sitter tree-sitter-python datasets numpy rich"
    ) from exc


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DEFAULT_MAX_AS_LIMIT = 30 * 1024
DEFAULT_MAX_DATA_LIMIT = 30 * 1024
DEFAULT_MAX_STACK_LIMIT = 10
DEFAULT_MIN_TIME_LIMIT = 1.0
DEFAULT_GT_TIME_LIMIT = 20.0
DEFAULT_GROUNDTRUTH_WORKERS = 4


SAFE_EXEC_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "TOKENIZERS_PARALLELISM": "false",
}

for env_key, env_value in SAFE_EXEC_ENV_DEFAULTS.items():
    os.environ.setdefault(env_key, env_value)


@dataclass
class BigCodeBenchVerification:
    success: bool
    task_id: str
    entry_point: str
    status: str
    error: Optional[str]
    sanitized_solution: str
    calibrated_solution: str
    failing_tests: list[str]
    failure_details: dict[str, str]


def _compute_groundtruth_time(args: tuple[str, str, str, float, float, float, float]) -> dict[str, Any]:
    code, test_code, task_id, max_as_limit, max_data_limit, max_stack_limit, min_time_limit = args
    return trusted_check(
        code,
        test_code,
        task_id,
        max_as_limit,
        max_data_limit,
        max_stack_limit,
        min_time_limit,
    )


def get_groundtruth_times(
    problems: dict[str, dict[str, Any]],
    hashcode: str,
    n_workers: int,
    max_as_limit: float = DEFAULT_MAX_AS_LIMIT,
    max_data_limit: float = DEFAULT_MAX_DATA_LIMIT,
    max_stack_limit: float = DEFAULT_MAX_STACK_LIMIT,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
) -> dict[str, Optional[float]]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{hashcode}.pkl")
    if os.path.exists(cache_file):
        print(f"Load from ground-truth from {cache_file}")
        with open(cache_file, "rb") as handle:
            return pickle.load(handle)

    print("\nAsserting the groundtruth...")
    started = time.time()
    expected_time: dict[str, Optional[float]] = {}
    work_items = [
        (
            problem["complete_prompt"] + "\n" + problem["canonical_solution"],
            problem["test"],
            problem["task_id"],
            max_as_limit,
            max_data_limit,
            max_stack_limit,
            min_time_limit,
        )
        for problem in problems.values()
    ]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_compute_groundtruth_time, item) for item in work_items]
        for future in as_completed(futures):
            result = future.result()
            expected_time[result["task_id"]] = result["time"]

    print(f"Expected outputs computed in {time.time() - started:.2f}s")
    with open(cache_file, "wb") as handle:
        pickle.dump(expected_time, handle)
    return expected_time


def make_bigcodebench_verifier(
    problem: dict[str, Any],
    gt_time_limit: Optional[float],
    calibrated: bool = True,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    max_as_limit: float = DEFAULT_MAX_AS_LIMIT,
    max_data_limit: float = DEFAULT_MAX_DATA_LIMIT,
    max_stack_limit: float = DEFAULT_MAX_STACK_LIMIT,
) -> Callable[[str], BigCodeBenchVerification]:
    def verify(raw_output: str) -> BigCodeBenchVerification:
        sanitized = sanitize(raw_output, entrypoint=problem["entry_point"]).strip()
        calibrated_solution = sanitized
        if calibrated:
            calibrated_solution = problem["code_prompt"] + "\n    pass\n" + sanitized

        if not sanitized:
            return BigCodeBenchVerification(
                success=False,
                task_id=problem["task_id"],
                entry_point=problem["entry_point"],
                status="fail",
                error="sanitizer produced empty solution",
                sanitized_solution="",
                calibrated_solution="",
                failing_tests=[],
                failure_details={},
            )

        try:
            status, details = untrusted_check(
                calibrated_solution,
                problem["test"],
                problem["entry_point"],
                max_as_limit,
                max_data_limit,
                max_stack_limit,
                min_time_limit=min_time_limit,
                gt_time_limit=gt_time_limit or DEFAULT_GT_TIME_LIMIT,
            )
        except Exception as exc:  # pragma: no cover - depends on runtime/model outputs
            return BigCodeBenchVerification(
                success=False,
                task_id=problem["task_id"],
                entry_point=problem["entry_point"],
                status="fail",
                error=f"checker exception: {exc}",
                sanitized_solution=sanitized,
                calibrated_solution=calibrated_solution,
                failing_tests=[],
                failure_details={},
            )

        failure_details = details if isinstance(details, dict) else {}
        failing_tests = sorted(failure_details)
        error = None if status == PASS else (
            failure_details.get("ALL") or f"{len(failing_tests)} failing tests"
        )

        return BigCodeBenchVerification(
            success=status == PASS,
            task_id=problem["task_id"],
            entry_point=problem["entry_point"],
            status=status,
            error=error,
            sanitized_solution=sanitized,
            calibrated_solution=calibrated_solution,
            failing_tests=failing_tests,
            failure_details=failure_details,
        )

    return verify


def build_bigcodebench_feedback(original_prompt: str, attempt: dict[str, Any]) -> str:
    verification = attempt.get("verification") or {}
    entry_point = verification.get("entry_point", "target")
    failing_tests = verification.get("failing_tests") or []
    failure_details = verification.get("failure_details") or {}
    error = attempt.get("error") or verification.get("error") or "BigCodeBench tests failed"

    details: list[str] = [f"Previous solution for `{entry_point}` failed BigCodeBench checks: {error}."]
    if failing_tests:
        details.append(f"Failing tests include: {', '.join(failing_tests[:5])}.")
        first_failure = failure_details.get(failing_tests[0], "")
        if first_failure:
            snippet = " ".join(str(first_failure).strip().split())[:500]
            details.append(f"Failure trace excerpt: {snippet}")

    return f"{original_prompt}\n\n" + " ".join(details)


def build_tasks(
    split: str,
    subset: str,
    limit: Optional[int] = None,
    offset: int = 0,
    calibrated: bool = True,
    groundtruth_workers: Optional[int] = None,
) -> tuple[list[BenchmarkTask], str]:
    problems = get_bigcodebench(subset=subset)
    dataset_hash = get_bigcodebench_hash(subset=subset)
    task_items = list(problems.items())
    if offset:
        task_items = task_items[offset:]
    if limit is not None:
        task_items = task_items[:limit]

    selected = {task_id: problem for task_id, problem in task_items}
    workers = groundtruth_workers or DEFAULT_GROUNDTRUTH_WORKERS
    expected_times = get_groundtruth_times(selected, dataset_hash, workers)

    tasks: list[BenchmarkTask] = []
    for task_id, problem in task_items:
        prompt = str(problem[f"{split}_prompt"]).rstrip() + "\n"
        tasks.append(
            BenchmarkTask(
                name=task_id,
                category=f"bigcodebench-{subset}-{split}",
                prompt=prompt,
                verify_fn=make_bigcodebench_verifier(
                    problem,
                    gt_time_limit=expected_times.get(task_id),
                    calibrated=calibrated,
                ),
                feedback_builder=build_bigcodebench_feedback,
            )
        )

    return tasks, dataset_hash


def build_provider(
    provider_name: str,
    model: str,
    temperature: float = 0.0,
    base_url: str = "http://localhost:11434",
    max_tokens: Optional[int] = None,
    provider_kwargs: Optional[dict[str, Any]] = None,
    request_timeout: float = 1800.0,
) -> Callable[[str], ProviderResponse]:
    if provider_name == "local":
        def provider(prompt: str) -> ProviderResponse:
            response = requests.post(
                f"{base_url.rstrip('/')}/api/generate",
                json={
                    "model": model,
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
            return ProviderResponse(text=str(payload.get("response", "")).strip(), metadata=metadata)

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
        raw_response = raw_provider(prompt)
        if isinstance(raw_response, ProviderResponse):
            return raw_response
        if isinstance(raw_response, dict):
            if "text" in raw_response:
                metadata = {key: value for key, value in raw_response.items() if key != "text"}
                return ProviderResponse(text=str(raw_response["text"]), metadata=metadata)
            return ProviderResponse(text=str(raw_response))
        return ProviderResponse(text=str(raw_response))

    return provider


def summarize_result(result: Any) -> dict[str, Any]:
    task_results = result.task_results
    successes = sum(1 for task in task_results if task["success"])
    return {
        "num_tasks": len(task_results),
        "num_passed": successes,
        "success_rate": (successes / len(task_results)) if task_results else 0.0,
        "avg_attempts": result.avg_attempts,
        "avg_attempt_time_seconds": result.avg_execution_time,
        "failed_tasks": [task["name"] for task in task_results if not task["success"]],
    }


def build_delta_report(standard_summary: dict[str, Any], some_summary: dict[str, Any]) -> dict[str, Any]:
    improved = sorted(set(standard_summary["failed_tasks"]) - set(some_summary["failed_tasks"]))
    return {
        "overall_success_delta": some_summary["success_rate"] - standard_summary["success_rate"],
        "improved_tasks": improved,
    }


def default_trajectory_dir(subset: str, split: str, model: str) -> str:
    safe_model = model.replace("/", "--").replace(":", "_")
    return os.path.join(
        PROJECT_ROOT,
        "benchmarks",
        "internal_trajectories",
        "bigcodebench",
        subset,
        split,
        safe_model,
    )


def _append_progress_record(progress_path: str, record: dict[str, Any]) -> None:
    with open(progress_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def make_progress_callback(
    model: str,
    subset: str,
    split: str,
    progress_path: Optional[str] = None,
) -> Callable[[dict[str, Any]], None]:
    prefix = f"{model} {subset}/{split}"

    def callback(event: dict[str, Any]) -> None:
        if progress_path:
            _append_progress_record(
                progress_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": model,
                    "subset": subset,
                    "split": split,
                    **event,
                },
            )
        if event.get("event") == "task_start":
            print(
                f"[{prefix}] {event.get('approach')} "
                f"{event.get('task_index')}/{event.get('task_total')} "
                f"{event.get('task_name')}: START",
                flush=True,
            )
            return
        if event.get("event") != "task_complete":
            return
        status = "PASS" if event.get("success") else "FAIL"
        print(
            f"[{prefix}] {event.get('approach')} "
            f"{event.get('task_index')}/{event.get('task_total')} "
            f"{event.get('task_name')}: {status} | "
            f"attempts={event.get('attempts')} | "
            f"elapsed={event.get('elapsed_seconds', 0.0):.2f}s | "
            f"running_success={event.get('running_success_rate', 0.0):.1%}",
            flush=True,
        )
        final_error = event.get("final_error")
        if final_error and not event.get("success"):
            print(f"[{prefix}] error: {final_error}", flush=True)

    return callback


def run_model_benchmark(
    provider_name: Optional[str],
    model: Optional[str],
    split: str,
    subset: str,
    controller_bundle_path: Optional[str] = None,
    action_provider_name: Optional[str] = None,
    action_model: Optional[str] = None,
    action_temperature: Optional[float] = None,
    action_base_url: Optional[str] = None,
    action_request_timeout: Optional[float] = None,
    temperature: Optional[float] = 0.0,
    max_retries: int = 3,
    base_url: Optional[str] = "http://localhost:11434",
    request_timeout: float = 1800.0,
    limit: Optional[int] = None,
    offset: int = 0,
    calibrated: bool = True,
    groundtruth_workers: Optional[int] = None,
    progress_path: Optional[str] = None,
    trajectory_dir: Optional[str] = None,
) -> dict[str, Any]:
    tasks, dataset_hash = build_tasks(
        split=split,
        subset=subset,
        limit=limit,
        offset=offset,
        calibrated=calibrated,
        groundtruth_workers=groundtruth_workers,
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
        action_request_timeout=action_request_timeout,
        default_provider="local",
        default_temperature=0.0,
        default_base_url="http://localhost:11434",
        default_request_timeout=1800.0,
    )
    provider = build_provider(
        provider_name=runtime.provider_name,
        model=runtime.model,
        temperature=runtime.temperature,
        base_url=runtime.base_url,
        max_tokens=runtime.max_tokens,
        provider_kwargs=runtime.provider_kwargs,
        request_timeout=runtime.request_timeout or 1800.0,
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
            request_timeout=runtime.action_request_timeout or runtime.request_timeout or 1800.0,
        )
    benchmark = LLMBenchmark(
        llm_provider=provider,
        action_provider=action_provider,
        progress_callback=make_progress_callback(
            model=runtime.model,
            subset=subset,
            split=split,
            progress_path=progress_path,
        ),
    )
    some_trajectory_dir = trajectory_dir or default_trajectory_dir(subset, split, runtime.model)

    print(f"\n{'=' * 72}")
    print(f"BigCodeBench Benchmark: {subset}/{split} / {runtime.provider_name} / {runtime.model}")
    print(f"{'=' * 72}")
    print(f"Tasks: {len(tasks)} | Offset: {offset} | Limit: {limit if limit is not None else 'all'}")

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
    delta = build_delta_report(standard_summary, some_summary)

    print(
        "Baseline: "
        f"{standard_summary['success_rate']:.1%} "
        f"({standard_summary['num_passed']}/{standard_summary['num_tasks']}) | "
        f"{standard_summary['avg_attempt_time_seconds']:.2f}s/attempt"
    )
    print(
        "SOME:     "
        f"{some_summary['success_rate']:.1%} "
        f"({some_summary['num_passed']}/{some_summary['num_tasks']}) | "
        f"{some_summary['avg_attempts']:.2f} attempts/task"
    )
    print(f"Delta:    {delta['overall_success_delta']:+.1%} overall success")
    if delta["improved_tasks"]:
        print("Improved tasks: " + ", ".join(delta["improved_tasks"]))

    return {
        "provider": runtime.provider_name,
        "model": runtime.model,
        "controller_bundle_path": controller_bundle_path,
        "subset": subset,
        "split": split,
        "dataset_hash": dataset_hash,
        "temperature": runtime.temperature,
        "max_tokens": runtime.max_tokens,
        "request_timeout": runtime.request_timeout,
        "provider_kwargs": runtime.provider_kwargs,
        "max_retries": max_retries,
        "task_limit": limit,
        "task_offset": offset,
        "calibrated": calibrated,
        "baseline": {
            "summary": standard_summary,
            "task_results": standard.task_results,
        },
        "some": {
            "summary": some_summary,
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
            action_request_timeout=args.action_request_timeout,
            default_provider="local",
            default_temperature=0.0,
            default_base_url="http://localhost:11434",
            default_request_timeout=1800.0,
        )
        return [runtime.model]
    return ["qwen3.5:4b", "qwen3.5:9b", "qwen3.5:27b"]


def run_benchmark(
    provider_name: Optional[str],
    models: list[str],
    split: str,
    subset: str,
    controller_bundle_path: Optional[str] = None,
    action_provider_name: Optional[str] = None,
    action_model: Optional[str] = None,
    action_temperature: Optional[float] = None,
    action_base_url: Optional[str] = None,
    action_request_timeout: Optional[float] = None,
    temperature: Optional[float] = 0.0,
    max_retries: int = 3,
    base_url: Optional[str] = "http://localhost:11434",
    request_timeout: float = 1800.0,
    limit: Optional[int] = None,
    offset: int = 0,
    calibrated: bool = True,
    groundtruth_workers: Optional[int] = None,
    progress_path: Optional[str] = None,
    trajectory_root: Optional[str] = None,
) -> dict[str, Any]:
    model_reports = [
        run_model_benchmark(
            provider_name=provider_name,
            model=model,
            split=split,
            subset=subset,
            controller_bundle_path=controller_bundle_path,
            action_provider_name=action_provider_name,
            action_model=action_model,
            action_temperature=action_temperature,
            action_base_url=action_base_url,
            action_request_timeout=action_request_timeout,
            temperature=temperature,
            max_retries=max_retries,
            base_url=base_url,
            request_timeout=request_timeout,
            limit=limit,
            offset=offset,
            calibrated=calibrated,
            groundtruth_workers=groundtruth_workers,
            progress_path=progress_path,
            trajectory_dir=(
                os.path.join(trajectory_root, model.replace("/", "--").replace(":", "_"))
                if trajectory_root
                else None
            ),
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
        "subset": subset,
        "split": split,
        "action_policy": {
            "provider": model_reports[0]["some"]["action_policy"]["provider"] if model_reports else action_provider_name,
            "model": model_reports[0]["some"]["action_policy"]["model"] if model_reports else action_model,
            "temperature": model_reports[0]["some"]["action_policy"]["temperature"] if model_reports else action_temperature,
            "base_url": model_reports[0]["some"]["action_policy"]["base_url"] if model_reports else action_base_url,
            "max_tokens": model_reports[0]["some"]["action_policy"]["max_tokens"] if model_reports else None,
            "request_timeout": model_reports[0]["some"]["action_policy"]["request_timeout"] if model_reports else action_request_timeout,
            "provider_kwargs": model_reports[0]["some"]["action_policy"]["provider_kwargs"] if model_reports else {},
        },
        "models": model_reports,
        "leaderboard": leaderboard,
    }


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
    parser = argparse.ArgumentParser(description="Run BigCodeBench with baseline vs SOME")
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
        "--subset",
        default="hard",
        choices=["full", "hard"],
        help="BigCodeBench subset to run",
    )
    parser.add_argument(
        "--split",
        default="instruct",
        choices=["instruct", "complete"],
        help="Prompt style to use",
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
        "--base-url",
        default=None,
        help="Base URL for local/OpenAI-compatible providers",
    )
    parser.add_argument(
        "--action-base-url",
        help="Optional base URL for the action-policy provider.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=1800.0,
        help="Per-request timeout in seconds for local/OpenAI-compatible providers",
    )
    parser.add_argument(
        "--action-request-timeout",
        type=float,
        help="Optional per-request timeout for the action-policy provider.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional task limit for pilot runs",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Optional task offset for shard/pilot runs",
    )
    parser.add_argument(
        "--no-calibrated",
        action="store_true",
        help="Disable the official calibrated evaluation wrapper",
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
        "--groundtruth-workers",
        type=int,
        default=DEFAULT_GROUNDTRUTH_WORKERS,
        help="Worker count for BigCodeBench ground-truth timing checks",
    )

    args = parser.parse_args()
    output_path = args.output
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(PROJECT_ROOT, f"bigcodebench_{args.subset}_{args.split}_{timestamp}.json")
    progress_path = output_path + ".progress.jsonl"
    if os.path.exists(progress_path):
        os.remove(progress_path)

    report = run_benchmark(
        provider_name=args.provider,
        models=parse_models(args),
        split=args.split,
        subset=args.subset,
        controller_bundle_path=args.controller_bundle,
        action_provider_name=args.action_provider,
        action_model=args.action_model,
        action_temperature=args.action_temp,
        action_base_url=args.action_base_url,
        action_request_timeout=args.action_request_timeout,
        temperature=args.temp,
        max_retries=args.max_retries,
        base_url=args.base_url,
        request_timeout=args.request_timeout,
        limit=args.limit,
        offset=args.offset,
        calibrated=not args.no_calibrated,
        groundtruth_workers=args.groundtruth_workers,
        progress_path=progress_path,
        trajectory_root=args.trajectory_root,
    )

    print_leaderboard(report)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"\nSaved detailed report to {output_path}")
    print(f"Saved incremental progress to {progress_path}")


if __name__ == "__main__":
    main()
