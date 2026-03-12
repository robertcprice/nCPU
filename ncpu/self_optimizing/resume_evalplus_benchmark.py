#!/usr/bin/env python3
"""
Resume an EvalPlus benchmark from a saved progress JSONL checkpoint.

The checkpoint file is the append-only `.progress.jsonl` produced by
`run_evalplus_benchmark.py`. This runner recovers completed task results for the
standard and SOME phases, continues only the missing work, and writes a final
per-model JSON report when complete.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent
for entry in (str(PROJECT_ROOT), str(MODULE_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from ncpu.self_optimizing.llm_benchmark import BenchmarkTask, LLMBenchmark, ProviderResponse
from ncpu.self_optimizing.run_evalplus_benchmark import default_trajectory_dir

try:
    from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash, get_mbpp_plus, get_mbpp_plus_hash
    from evalplus.eval import PASS
    from evalplus.evaluate import check_correctness, get_groundtruth
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.sanitize import sanitize
except ImportError as exc:  # pragma: no cover - handled on benchmark boxes
    raise SystemExit(
        "EvalPlus is required. Install it with: python3 -m pip install evalplus"
    ) from exc


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


def make_evalplus_verifier(
    dataset: str,
    problem: dict[str, Any],
    expected_output: dict[str, Any],
) -> Callable[[str], EvalPlusVerification]:
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
        except Exception as exc:
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
    noextreme: bool = True,
) -> tuple[list[BenchmarkTask], str]:
    problems, dataset_hash, tasks_only_output_not_none = _dataset_bundle(dataset, noextreme=noextreme)
    expected_output = get_groundtruth(problems, dataset_hash, tasks_only_output_not_none)

    tasks: list[BenchmarkTask] = []
    for task_id, problem in problems.items():
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
    request_timeout: float = 300.0,
) -> Callable[[str], ProviderResponse]:
    if provider_name != "local":
        raise ValueError("resume_evalplus_benchmark.py currently supports only --provider local")

    def provider(prompt: str) -> ProviderResponse:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
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


def build_delta_report(standard_summary: dict[str, Any], some_summary: dict[str, Any]) -> dict[str, Any]:
    improved = sorted(set(standard_summary["failed_tasks"]) - set(some_summary["failed_tasks"]))
    return {
        "overall_success_delta": some_summary["success_rate"] - standard_summary["success_rate"],
        "improved_tasks": improved,
    }


def _append_progress_record(progress_path: str, record: dict[str, Any]) -> None:
    with open(progress_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _canonical_task_name(
    *,
    task_order: list[str],
    task_index: Any,
    task_name: Any = None,
    task_result_name: Any = None,
) -> str:
    try:
        index = int(task_index)
    except (TypeError, ValueError):
        index = 0

    if 1 <= index <= len(task_order):
        return task_order[index - 1]

    for candidate in (task_result_name, task_name):
        if candidate:
            return str(candidate)
    return ""


def normalize_progress_file(progress_path: str, task_order: list[str]) -> None:
    path = Path(progress_path)
    if not path.exists():
        return

    normalized_lines: list[str] = []
    changed = False

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            canonical_name = _canonical_task_name(
                task_order=task_order,
                task_index=record.get("task_index"),
                task_name=record.get("task_name"),
                task_result_name=(record.get("task_result") or {}).get("name")
                if isinstance(record.get("task_result"), dict)
                else None,
            )
            if canonical_name:
                if record.get("task_name") != canonical_name:
                    record["task_name"] = canonical_name
                    changed = True
                if isinstance(record.get("task_result"), dict) and record["task_result"].get("name") != canonical_name:
                    record["task_result"] = dict(record["task_result"])
                    record["task_result"]["name"] = canonical_name
                    changed = True
            normalized_lines.append(json.dumps(record))

    if changed:
        path.write_text("\n".join(normalized_lines) + "\n", encoding="utf-8")


def load_checkpoint(progress_path: str, task_order: list[str]) -> dict[str, list[dict[str, Any]]]:
    path = Path(progress_path)
    if not path.exists():
        return {"standard": [], "some": []}

    latest_by_phase: dict[str, dict[str, dict[str, Any]]] = {"standard": {}, "some": {}}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("event") != "task_complete":
                continue
            approach = record.get("approach")
            if approach not in latest_by_phase:
                continue
            task_result = record.get("task_result")
            if not isinstance(task_result, dict):
                continue
            task_name = _canonical_task_name(
                task_order=task_order,
                task_index=record.get("task_index"),
                task_name=record.get("task_name"),
                task_result_name=task_result.get("name"),
            )
            if not task_name:
                continue
            normalized_result = dict(task_result)
            normalized_result["name"] = task_name
            latest_by_phase[approach][task_name] = normalized_result

    ordered: dict[str, list[dict[str, Any]]] = {"standard": [], "some": []}
    for approach in ("standard", "some"):
        ordered[approach] = [
            latest_by_phase[approach][task_name]
            for task_name in task_order
            if task_name in latest_by_phase[approach]
        ]
    return ordered


def summarize_task_results(task_results: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(1 for task in task_results if task["success"])
    total_attempts = sum(int(task.get("attempts", 0) or 0) for task in task_results)
    total_elapsed = sum(float(task.get("elapsed_seconds", 0.0) or 0.0) for task in task_results)
    return {
        "num_tasks": len(task_results),
        "num_passed": successes,
        "success_rate": (successes / len(task_results)) if task_results else 0.0,
        "avg_attempts": (total_attempts / len(task_results)) if task_results else 0.0,
        "avg_attempt_time_seconds": (total_elapsed / total_attempts) if total_attempts else 0.0,
        "failed_tasks": [task["name"] for task in task_results if not task["success"]],
    }


def make_resuming_progress_callback(
    *,
    model: str,
    dataset: str,
    progress_path: str,
    approach: str,
    index_offset: int,
    total_tasks: int,
    initial_passes: int,
    task_order: list[str],
) -> Callable[[dict[str, Any]], None]:
    prefix = f"{model} {dataset}"
    completed_passes = 0

    def callback(event: dict[str, Any]) -> None:
        nonlocal completed_passes
        adjusted = dict(event)
        adjusted["task_index"] = index_offset + int(event.get("task_index", 0) or 0)
        adjusted["task_total"] = total_tasks
        adjusted["approach"] = approach
        adjusted["task_name"] = _canonical_task_name(
            task_order=task_order,
            task_index=adjusted["task_index"],
            task_name=event.get("task_name"),
        )
        if isinstance(event.get("task_result"), dict):
            adjusted_task_result = dict(event["task_result"])
            adjusted_task_result["name"] = _canonical_task_name(
                task_order=task_order,
                task_index=adjusted["task_index"],
                task_name=event.get("task_name"),
                task_result_name=adjusted_task_result.get("name"),
            )
            adjusted["task_result"] = adjusted_task_result

        if event.get("event") == "task_complete" and event.get("success"):
            completed_passes += 1
        if event.get("event") == "task_complete":
            adjusted["running_success_rate"] = (
                (initial_passes + completed_passes) / adjusted["task_index"]
                if adjusted["task_index"]
                else 0.0
            )

        _append_progress_record(
            progress_path,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "dataset": dataset,
                **adjusted,
            },
        )

        if adjusted.get("event") == "task_start":
            print(
                f"[{prefix}] {adjusted.get('approach')} "
                f"{adjusted.get('task_index')}/{adjusted.get('task_total')} "
                f"{adjusted.get('task_name')}: START",
                flush=True,
            )
            return

        if adjusted.get("event") != "task_complete":
            return

        status = "PASS" if adjusted.get("success") else "FAIL"
        print(
            f"[{prefix}] {adjusted.get('approach')} "
            f"{adjusted.get('task_index')}/{adjusted.get('task_total')} "
            f"{adjusted.get('task_name')}: {status} | "
            f"attempts={adjusted.get('attempts')} | "
            f"elapsed={adjusted.get('elapsed_seconds', 0.0):.2f}s | "
            f"running_success={adjusted.get('running_success_rate', 0.0):.1%}",
            flush=True,
        )
        final_error = adjusted.get("final_error")
        if final_error and not adjusted.get("success"):
            print(f"[{prefix}] error: {final_error}", flush=True)

    return callback


def _run_remaining_phase(
    *,
    model: str,
    dataset: str,
    provider_name: str,
    provider: Callable[[str], Any],
    progress_path: str,
    tasks: list[BenchmarkTask],
    existing_results: list[dict[str, Any]],
    max_retries: int,
    phase: str,
    trajectory_dir: str | None = None,
) -> list[dict[str, Any]]:
    total_tasks = len(tasks)
    completed = len(existing_results)
    if completed >= total_tasks:
        print(
            f"{phase} already complete for {model}: "
            f"{completed}/{total_tasks} tasks",
            flush=True,
        )
        return existing_results

    remaining = tasks[completed:]
    initial_passes = sum(1 for task in existing_results if task.get("success"))
    callback = make_resuming_progress_callback(
        model=model,
        dataset=dataset,
        progress_path=progress_path,
        approach=phase,
        index_offset=completed,
        total_tasks=total_tasks,
        initial_passes=initial_passes,
        task_order=[task.name for task in tasks],
    )
    benchmark = LLMBenchmark(llm_provider=provider, progress_callback=callback)

    print(
        f"Resuming {phase} for {model}: "
        f"{completed}/{total_tasks} already complete, {len(remaining)} remaining",
        flush=True,
    )

    if phase == "standard":
        result = benchmark.benchmark_standard(remaining)
    elif phase == "some":
        result = benchmark.benchmark_with_some(
            remaining,
            max_retries=max_retries,
            use_buffered_controller=True,
            trajectory_dir=trajectory_dir,
        )
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported phase: {phase}")

    return existing_results + result.task_results


def build_model_report(
    *,
    provider_name: str,
    model: str,
    dataset: str,
    dataset_hash: str,
    temperature: float,
    request_timeout: float,
    max_retries: int,
    noextreme: bool,
    task_results_standard: list[dict[str, Any]],
    task_results_some: list[dict[str, Any]],
    trajectory_dir: str | None = None,
) -> dict[str, Any]:
    standard_summary = summarize_task_results(task_results_standard)
    some_summary = summarize_task_results(task_results_some)
    delta = build_delta_report(standard_summary, some_summary)
    return {
        "provider": provider_name,
        "model": model,
        "dataset": dataset,
        "dataset_hash": dataset_hash,
        "temperature": temperature,
        "request_timeout": request_timeout,
        "max_retries": max_retries,
        "task_limit": None,
        "task_offset": 0,
        "noextreme": noextreme,
        "resumed_from_progress": True,
        "baseline": {
            "summary": standard_summary,
            "task_results": task_results_standard,
        },
        "some": {
            "summary": some_summary,
            "task_results": task_results_some,
            "trajectory_dir": trajectory_dir,
        },
        "delta": delta,
    }


def default_output_path(progress_path: str) -> str:
    if progress_path.endswith(".progress.jsonl"):
        return progress_path[: -len(".progress.jsonl")]
    return progress_path + ".json"


def print_checkpoint_status(checkpoint: dict[str, list[dict[str, Any]]], total_tasks: int) -> None:
    for phase in ("standard", "some"):
        task_results = checkpoint[phase]
        passed = sum(1 for task in task_results if task.get("success"))
        print(
            f"{phase}: {len(task_results)}/{total_tasks} complete, "
            f"{passed} passed",
            flush=True,
        )
        if task_results:
            last = task_results[-1]
            print(
                f"  last: {last.get('name')} "
                f"success={last.get('success')} "
                f"error={last.get('final_error')}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume EvalPlus from a progress checkpoint")
    parser.add_argument("--provider", "-p", default="local")
    parser.add_argument("--dataset", default="humaneval", choices=["humaneval", "mbpp"])
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--progress-path", required=True, help="Existing .progress.jsonl checkpoint path")
    parser.add_argument("--output", help="Final JSON output path")
    parser.add_argument("--temp", "-t", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--trajectory-root", help="Optional directory root for SOME hidden trajectory JSONL artifacts")
    parser.add_argument("--status-only", action="store_true", help="Print checkpoint status and exit")

    args = parser.parse_args()

    progress_path = args.progress_path
    output_path = args.output or default_output_path(progress_path)
    tasks, dataset_hash = build_tasks(dataset=args.dataset, noextreme=not args.full)
    task_order = [task.name for task in tasks]
    normalize_progress_file(progress_path, task_order)
    checkpoint = load_checkpoint(progress_path, task_order)

    print_checkpoint_status(checkpoint, total_tasks=len(tasks))
    if args.status_only:
        return

    provider = build_provider(
        provider_name=args.provider,
        model=args.model,
        temperature=args.temp,
        base_url=args.base_url,
        request_timeout=args.request_timeout,
    )
    some_trajectory_dir = (
        os.path.join(args.trajectory_root, args.model.replace("/", "--").replace(":", "_"))
        if args.trajectory_root
        else default_trajectory_dir(args.dataset, args.model)
    )

    standard_results = _run_remaining_phase(
        model=args.model,
        dataset=args.dataset,
        provider_name=args.provider,
        provider=provider,
        progress_path=progress_path,
        tasks=tasks,
        existing_results=checkpoint["standard"],
        max_retries=args.max_retries,
        phase="standard",
    )
    some_results = _run_remaining_phase(
        model=args.model,
        dataset=args.dataset,
        provider_name=args.provider,
        provider=provider,
        progress_path=progress_path,
        tasks=tasks,
        existing_results=checkpoint["some"],
        max_retries=args.max_retries,
        phase="some",
        trajectory_dir=some_trajectory_dir,
    )

    report = build_model_report(
        provider_name=args.provider,
        model=args.model,
        dataset=args.dataset,
        dataset_hash=dataset_hash,
        temperature=args.temp,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        noextreme=not args.full,
        task_results_standard=standard_results,
        task_results_some=some_results,
        trajectory_dir=some_trajectory_dir,
    )

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"\nSaved resumed report to {output_path}", flush=True)


if __name__ == "__main__":
    main()
