#!/usr/bin/env python3
"""
Run HumanEval+/MBPP+ with baseline vs SOME-style verify/retry.

This uses EvalPlus task definitions and EvalPlus correctness checks so the
benchmark is grounded in a public benchmark suite rather than custom toy tasks.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional

import requests

from ncpu.self_optimizing.controller_runtime import resolve_controller_runtime
from ncpu.self_optimizing.llm_benchmark import BenchmarkTask, LLMBenchmark, ProviderResponse
from ncpu.self_optimizing.llm_provider import LLMProviderFactory

try:
    from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash, get_mbpp_plus, get_mbpp_plus_hash
    from evalplus.eval import PASS
    from evalplus.evaluate import check_correctness, get_groundtruth
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.sanitize import sanitize
except ImportError as exc:  # pragma: no cover - handled at runtime on benchmark boxes
    raise SystemExit(
        "EvalPlus is required. Install it with: python3 -m pip install evalplus"
    ) from exc


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


@dataclass
class EvalPlusVerification:
    success: bool
    dataset: str
    task_id: str
    entry_point: str
    error: Optional[str]
    sanitized_solution: str
    base_status: str
    plus_status: str
    base_passed: int
    base_total: int
    plus_passed: int
    plus_total: int
    base_failed_examples: list[str]
    plus_failed_examples: list[str]


def _trim_examples(inputs: Iterable[Any], details: list[bool], limit: int = 3) -> list[str]:
    failures: list[str] = []
    for candidate, passed in zip(inputs, details):
        if passed:
            continue
        failures.append(repr(candidate))
        if len(failures) >= limit:
            break
    return failures


def _status_tuple(value: Any) -> tuple[str, list[bool]]:
    if not value:
        return "fail", []
    status = value[0]
    details = list(value[1]) if len(value) > 1 and value[1] is not None else []
    return status, [bool(item) for item in details]


def make_evalplus_verifier(dataset: str, problem: dict[str, Any], expected_output: dict[str, Any]) -> Callable[[str], EvalPlusVerification]:
    def verify(raw_output: str) -> EvalPlusVerification:
        sanitized = sanitize(raw_output, entrypoint=problem["entry_point"]).strip()
        if not sanitized:
            return EvalPlusVerification(
                success=False,
                dataset=dataset,
                task_id=problem["task_id"],
                entry_point=problem["entry_point"],
                error="sanitizer produced empty solution",
                sanitized_solution="",
                base_status="fail",
                plus_status="fail",
                base_passed=0,
                base_total=len(problem["base_input"]),
                plus_passed=0,
                plus_total=len(problem["plus_input"]),
                base_failed_examples=[],
                plus_failed_examples=[],
            )

        try:
            result = check_correctness(
                dataset=dataset,
                completion_id=0,
                problem=problem,
                solution=sanitized,
                expected_output=expected_output,
                base_only=False,
                fast_check=False,
            )
        except Exception as exc:  # pragma: no cover - depends on model outputs/runtime
            return EvalPlusVerification(
                success=False,
                dataset=dataset,
                task_id=problem["task_id"],
                entry_point=problem["entry_point"],
                error=f"evaluator exception: {exc}",
                sanitized_solution=sanitized,
                base_status="fail",
                plus_status="fail",
                base_passed=0,
                base_total=len(problem["base_input"]),
                plus_passed=0,
                plus_total=len(problem["plus_input"]),
                base_failed_examples=[],
                plus_failed_examples=[],
            )

        base_status, base_details = _status_tuple(result.get("base"))
        plus_status, plus_details = _status_tuple(result.get("plus"))
        base_failed = _trim_examples(problem["base_input"], base_details)
        plus_failed = _trim_examples(problem["plus_input"], plus_details)

        base_total = len(problem["base_input"])
        plus_total = len(problem["plus_input"])
        base_passed = sum(base_details) if base_details else (base_total if base_status == PASS else 0)
        plus_passed = sum(plus_details) if plus_details else (plus_total if plus_status == PASS else 0)
        success = base_status == PASS and plus_status == PASS

        error_parts: list[str] = []
        if base_status != PASS:
            error_parts.append(f"base {base_passed}/{base_total}")
        if plus_status != PASS:
            error_parts.append(f"plus {plus_passed}/{plus_total}")
        error = None if success else ", ".join(error_parts) or "EvalPlus verification failed"

        return EvalPlusVerification(
            success=success,
            dataset=dataset,
            task_id=problem["task_id"],
            entry_point=problem["entry_point"],
            error=error,
            sanitized_solution=sanitized,
            base_status=base_status,
            plus_status=plus_status,
            base_passed=base_passed,
            base_total=base_total,
            plus_passed=plus_passed,
            plus_total=plus_total,
            base_failed_examples=base_failed,
            plus_failed_examples=plus_failed,
        )

    return verify


def build_evalplus_feedback(original_prompt: str, attempt: dict[str, Any]) -> str:
    verification = attempt.get("verification") or {}
    entry_point = verification.get("entry_point", "target function")
    error = attempt.get("error") or verification.get("error") or "EvalPlus tests failed"

    details: list[str] = [f"Previous solution for `{entry_point}` failed EvalPlus checks: {error}."]

    base_failed = verification.get("base_failed_examples") or []
    plus_failed = verification.get("plus_failed_examples") or []
    if base_failed:
        details.append(f"Base failing inputs include: {', '.join(base_failed)}.")
    if plus_failed:
        details.append(f"Extended failing inputs include: {', '.join(plus_failed)}.")

    return (
        "Produce a corrected, self-contained Python solution.\n"
        "Return only raw Python code. Do not include markdown, backticks, or explanation.\n\n"
        f"{original_prompt}\n\n"
        + " ".join(details)
    )


def _dataset_bundle(dataset: str, noextreme: bool) -> tuple[dict[str, dict[str, Any]], str, list[str]]:
    if dataset == "humaneval":
        problems = get_human_eval_plus(noextreme=noextreme)
        dataset_hash = get_human_eval_plus_hash(noextreme=noextreme)
        tasks_only_output_not_none: list[str] = []
    elif dataset == "mbpp":
        problems = get_mbpp_plus(noextreme=noextreme)
        dataset_hash = get_mbpp_plus_hash(noextreme=noextreme)
        tasks_only_output_not_none = MBPP_OUTPUT_NOT_NONE_TASKS
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return problems, dataset_hash, tasks_only_output_not_none


def build_tasks(
    dataset: str,
    limit: Optional[int] = None,
    offset: int = 0,
    noextreme: bool = True,
) -> tuple[list[BenchmarkTask], str]:
    problems, dataset_hash, tasks_only_output_not_none = _dataset_bundle(dataset, noextreme=noextreme)
    expected_output = get_groundtruth(problems, dataset_hash, tasks_only_output_not_none)

    task_items = list(problems.items())
    if offset:
        task_items = task_items[offset:]
    if limit is not None:
        task_items = task_items[:limit]

    tasks: list[BenchmarkTask] = []
    for task_id, problem in task_items:
        prompt = (
            "Write a self-contained Python solution for the following benchmark task.\n"
            "Return only raw Python code. Do not include markdown, backticks, or explanation.\n\n"
            f"{problem['prompt'].strip()}\n"
        )
        tasks.append(
            BenchmarkTask(
                name=task_id,
                category=dataset,
                prompt=prompt,
                verify_fn=make_evalplus_verifier(dataset, problem, expected_output[task_id]),
                feedback_builder=build_evalplus_feedback,
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
    request_timeout: float = 600.0,
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


def default_trajectory_dir(dataset: str, model: str) -> str:
    safe_model = model.replace("/", "--").replace(":", "_")
    return os.path.join(
        PROJECT_ROOT,
        "benchmarks",
        "internal_trajectories",
        "evalplus",
        dataset,
        safe_model,
    )


def _append_progress_record(progress_path: str, record: dict[str, Any]) -> None:
    with open(progress_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def make_progress_callback(
    model: str,
    dataset: str,
    progress_path: Optional[str] = None,
) -> Callable[[dict[str, Any]], None]:
    prefix = f"{model} {dataset}"

    def callback(event: dict[str, Any]) -> None:
        if progress_path:
            _append_progress_record(
                progress_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": model,
                    "dataset": dataset,
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
    dataset: str,
    controller_bundle_path: Optional[str] = None,
    action_provider_name: Optional[str] = None,
    action_model: Optional[str] = None,
    action_temperature: Optional[float] = None,
    action_base_url: Optional[str] = None,
    action_request_timeout: Optional[float] = None,
    temperature: Optional[float] = 0.0,
    max_retries: int = 3,
    base_url: Optional[str] = "http://localhost:11434",
    request_timeout: float = 600.0,
    limit: Optional[int] = None,
    offset: int = 0,
    noextreme: bool = True,
    progress_path: Optional[str] = None,
    trajectory_dir: Optional[str] = None,
) -> dict[str, Any]:
    tasks, dataset_hash = build_tasks(dataset=dataset, limit=limit, offset=offset, noextreme=noextreme)
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
        default_request_timeout=600.0,
    )
    provider = build_provider(
        provider_name=runtime.provider_name,
        model=runtime.model,
        temperature=runtime.temperature,
        base_url=runtime.base_url,
        max_tokens=runtime.max_tokens,
        provider_kwargs=runtime.provider_kwargs,
        request_timeout=runtime.request_timeout or 600.0,
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
            request_timeout=runtime.action_request_timeout or runtime.request_timeout or 600.0,
        )
    benchmark = LLMBenchmark(
        llm_provider=provider,
        action_provider=action_provider,
        progress_callback=make_progress_callback(model=runtime.model, dataset=dataset, progress_path=progress_path),
    )
    some_trajectory_dir = trajectory_dir or default_trajectory_dir(dataset, runtime.model)

    print(f"\n{'=' * 72}")
    print(f"EvalPlus Benchmark: {dataset} / {runtime.provider_name} / {runtime.model}")
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
        "dataset": dataset,
        "dataset_hash": dataset_hash,
        "temperature": runtime.temperature,
        "max_tokens": runtime.max_tokens,
        "request_timeout": runtime.request_timeout,
        "provider_kwargs": runtime.provider_kwargs,
        "max_retries": max_retries,
        "task_limit": limit,
        "task_offset": offset,
        "noextreme": noextreme,
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
            default_request_timeout=600.0,
        )
        return [runtime.model]
    return ["qwen3.5:4b", "qwen3.5:9b", "qwen3.5:27b"]


def run_benchmark(
    provider_name: Optional[str],
    models: list[str],
    dataset: str,
    controller_bundle_path: Optional[str] = None,
    action_provider_name: Optional[str] = None,
    action_model: Optional[str] = None,
    action_temperature: Optional[float] = None,
    action_base_url: Optional[str] = None,
    action_request_timeout: Optional[float] = None,
    temperature: float = 0.0,
    max_retries: int = 3,
    base_url: str = "http://localhost:11434",
    request_timeout: float = 600.0,
    limit: Optional[int] = None,
    offset: int = 0,
    noextreme: bool = True,
    progress_path: Optional[str] = None,
    trajectory_root: Optional[str] = None,
) -> dict[str, Any]:
    model_reports = [
        run_model_benchmark(
            provider_name=provider_name,
            model=model,
            dataset=dataset,
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
            noextreme=noextreme,
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
        "dataset": dataset,
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
    parser = argparse.ArgumentParser(description="Run EvalPlus with baseline vs SOME")
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
        "--dataset",
        default="humaneval",
        choices=["humaneval", "mbpp"],
        help="EvalPlus dataset to run",
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
        default=600.0,
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
        "--full",
        action="store_true",
        help="Include extreme EvalPlus inputs instead of the safer noextreme split.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path",
    )
    parser.add_argument(
        "--trajectory-root",
        help="Optional directory root for SOME hidden trajectory JSONL artifacts",
    )

    args = parser.parse_args()
    output_path = args.output
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(PROJECT_ROOT, f"evalplus_{args.dataset}_{timestamp}.json")
    progress_path = output_path + ".progress.jsonl"
    if os.path.exists(progress_path):
        os.remove(progress_path)

    report = run_benchmark(
        provider_name=args.provider,
        models=parse_models(args),
        dataset=args.dataset,
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
        noextreme=not args.full,
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
