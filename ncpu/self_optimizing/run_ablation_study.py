#!/usr/bin/env python3
"""SOME ablation study: isolate contribution of each component."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ncpu.self_optimizing.internal_controller import (
    BufferedInternalController,
    InternalControllerConfig,
    InternalDeliberationTask,
)
from ncpu.self_optimizing.llm_benchmark import BenchmarkTask, LLMBenchmark
from ncpu.self_optimizing.run_qwen_benchmark import (
    build_tasks,
    build_provider,
    summarize_result,
)
from ncpu.self_optimizing.trajectory_logger import TrajectoryLogger


# ---------------------------------------------------------------------------
# Ablation conditions
# ---------------------------------------------------------------------------

ABLATION_CONDITIONS: dict[str, dict[str, Any]] = {
    "1_baseline": {
        "description": "Single attempt, no SOME",
        "use_some": False,
    },
    "2_verify_only": {
        "description": "Retry loop (4 attempts), no planning/heads/memory/weights",
        "use_some": True,
        "config": {
            "max_generation_attempts": 4,
            "plan_before_generate": False,
            "prefer_latent_action_policy": False,
            "prefer_latent_halt_policy": False,
            "prefer_latent_memory_updater": False,
            "fast_weight_updates_on_plan": False,
            "fast_weight_updates_on_repair_plan": False,
            "fast_weight_updates_on_verify_failure": False,
            "fast_weight_updates_on_verify_success": False,
            "descriptor_updates_on_plan": False,
            "descriptor_updates_on_verify_failure": False,
            "commit_on_first_success": True,
            "allow_unverified_commit": False,
        },
    },
    "3_verify_plan": {
        "description": "Retry + planning, no heads/memory/weights",
        "use_some": True,
        "config": {
            "max_generation_attempts": 4,
            "plan_before_generate": True,
            "prefer_latent_action_policy": False,
            "prefer_latent_halt_policy": False,
            "prefer_latent_memory_updater": False,
            "fast_weight_updates_on_plan": False,
            "fast_weight_updates_on_repair_plan": False,
            "fast_weight_updates_on_verify_failure": False,
            "fast_weight_updates_on_verify_success": False,
            "descriptor_updates_on_plan": False,
            "descriptor_updates_on_verify_failure": False,
            "commit_on_first_success": True,
            "allow_unverified_commit": False,
        },
    },
    "4_verify_plan_heads": {
        "description": "Retry + planning + learned action/halt heads",
        "use_some": True,
        "config": {
            "max_generation_attempts": 4,
            "plan_before_generate": True,
            "prefer_latent_action_policy": True,
            "prefer_latent_halt_policy": True,
            "prefer_latent_memory_updater": False,
            "fast_weight_updates_on_plan": False,
            "fast_weight_updates_on_repair_plan": False,
            "fast_weight_updates_on_verify_failure": False,
            "fast_weight_updates_on_verify_success": False,
            "descriptor_updates_on_plan": False,
            "descriptor_updates_on_verify_failure": False,
            "commit_on_first_success": True,
            "allow_unverified_commit": False,
        },
    },
    "5_verify_plan_heads_memory": {
        "description": "Retry + planning + heads + recurrent memory",
        "use_some": True,
        "config": {
            "max_generation_attempts": 4,
            "plan_before_generate": True,
            "prefer_latent_action_policy": True,
            "prefer_latent_halt_policy": True,
            "prefer_latent_memory_updater": True,
            "fast_weight_updates_on_plan": False,
            "fast_weight_updates_on_repair_plan": False,
            "fast_weight_updates_on_verify_failure": False,
            "fast_weight_updates_on_verify_success": False,
            "descriptor_updates_on_plan": False,
            "descriptor_updates_on_verify_failure": False,
            "commit_on_first_success": True,
            "allow_unverified_commit": False,
        },
    },
    "6_full_some": {
        "description": "Full SOME: retry + plan + heads + memory + fast weights",
        "use_some": True,
        "config": {
            "max_generation_attempts": 4,
            "plan_before_generate": True,
            "prefer_latent_action_policy": True,
            "prefer_latent_halt_policy": True,
            "prefer_latent_memory_updater": True,
            "fast_weight_updates_on_plan": True,
            "fast_weight_updates_on_repair_plan": True,
            "fast_weight_updates_on_verify_failure": True,
            "fast_weight_updates_on_verify_success": False,
            "descriptor_updates_on_plan": True,
            "descriptor_updates_on_verify_failure": True,
            "commit_on_first_success": True,
            "allow_unverified_commit": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run_single_task_some(
    *,
    benchmark: LLMBenchmark,
    task: BenchmarkTask,
    task_index: int,
    config: InternalControllerConfig,
    trajectory_dir: Optional[str],
) -> dict[str, Any]:
    """Run one task through the buffered controller with a specific config."""
    internal_task = InternalDeliberationTask(
        name=task.name,
        prompt=task.prompt,
        verifier=task.verify_fn or benchmark.verify_fn,
        category=task.category,
        response_format=benchmark._default_response_format(task),
        feedback_builder=task.feedback_builder,
    )

    traj_path: Optional[str] = None
    if trajectory_dir:
        traj_path = str(
            Path(trajectory_dir)
            / f"{task_index:04d}_{task.name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
        )

    controller = BufferedInternalController(
        llm_provider=benchmark.llm_provider,
        action_provider=benchmark.action_provider,
        config=config,
        trajectory_logger=TrajectoryLogger(traj_path) if traj_path else None,
    )

    t0 = time.monotonic()
    try:
        workspace = controller.run_task(internal_task)
    except Exception as exc:
        return {
            "name": task.name,
            "category": task.category,
            "success": False,
            "attempts": 0,
            "elapsed_seconds": time.monotonic() - t0,
            "error": f"{type(exc).__name__}: {exc}",
        }

    elapsed = time.monotonic() - t0
    attempts = benchmark._workspace_to_attempts(workspace)
    final = attempts[-1] if attempts else {"success": False, "error": "no attempts"}
    return {
        "name": task.name,
        "category": task.category,
        "success": bool(final.get("success", False)),
        "attempts": len(attempts),
        "elapsed_seconds": elapsed,
        "error": None if final.get("success") else final.get("error", "unknown"),
    }


def run_condition(
    *,
    condition_name: str,
    condition_spec: dict[str, Any],
    benchmark: LLMBenchmark,
    tasks: list[BenchmarkTask],
    output_dir: Path,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Run one ablation condition across all tasks."""
    desc = condition_spec["description"]
    print(f"\n{'─' * 60}")
    print(f"Condition: {condition_name}")
    print(f"  {desc}")
    print(f"  Tasks: {len(tasks)} | Workers: {max_workers}")
    print(f"{'─' * 60}", flush=True)

    traj_dir = str(output_dir / "trajectories" / condition_name)
    Path(traj_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()

    if not condition_spec.get("use_some"):
        # Baseline: use standard benchmark (single attempt), now parallel
        result = benchmark.benchmark_standard(tasks, max_workers=max_workers)
        summary = summarize_result(result)
        task_results = []
        for tr in result.task_results:
            task_results.append({
                "name": tr["name"],
                "category": tr["category"],
                "success": tr["success"],
                "attempts": 1,
                "elapsed_seconds": tr.get("elapsed_seconds", 0.0),
            })
        condition_result = {
            "condition": condition_name,
            "description": desc,
            "summary": summary,
            "task_results": task_results,
            "wall_clock_seconds": time.monotonic() - t0,
        }
    else:
        config_kwargs = dict(condition_spec.get("config", {}))
        config = InternalControllerConfig(**config_kwargs)

        task_results: list[dict[str, Any]] = []
        completed = 0

        if max_workers <= 1:
            for i, task in enumerate(tasks, start=1):
                tr = _run_single_task_some(
                    benchmark=benchmark,
                    task=task,
                    task_index=i,
                    config=config,
                    trajectory_dir=traj_dir,
                )
                status = "OK" if tr["success"] else "FAIL"
                print(f"  [{i}/{len(tasks)}] {task.name}: {status} ({tr['attempts']} attempts, {tr['elapsed_seconds']:.1f}s)", flush=True)
                task_results.append(tr)
        else:
            # Parallel execution
            indexed_tasks = list(enumerate(tasks, start=1))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_to_idx = {}
                for i, task in indexed_tasks:
                    fut = pool.submit(
                        _run_single_task_some,
                        benchmark=benchmark,
                        task=task,
                        task_index=i,
                        config=config,
                        trajectory_dir=traj_dir,
                    )
                    future_to_idx[fut] = (i, task)

                for fut in as_completed(future_to_idx):
                    i, task = future_to_idx[fut]
                    tr = fut.result()
                    completed += 1
                    status = "OK" if tr["success"] else "FAIL"
                    print(f"  [{completed}/{len(tasks)}] {task.name}: {status} ({tr['attempts']} attempts, {tr['elapsed_seconds']:.1f}s)", flush=True)
                    task_results.append(tr)

        successes = sum(1 for tr in task_results if tr["success"])
        total_attempts = sum(tr["attempts"] for tr in task_results)
        total_time = sum(tr["elapsed_seconds"] for tr in task_results)

        # Per-category breakdown
        categories: dict[str, dict[str, Any]] = {}
        for tr in task_results:
            cat = tr.get("category", "unknown")
            if cat not in categories:
                categories[cat] = {"successes": 0, "total": 0, "attempts": 0}
            categories[cat]["total"] += 1
            categories[cat]["attempts"] += tr["attempts"]
            if tr["success"]:
                categories[cat]["successes"] += 1

        category_summary = {}
        for cat, stats in categories.items():
            category_summary[cat] = {
                "success_rate": stats["successes"] / stats["total"] if stats["total"] else 0.0,
                "num_samples": stats["total"],
                "avg_attempts": stats["attempts"] / stats["total"] if stats["total"] else 0.0,
            }

        # Per-task breakdown
        by_task: dict[str, dict[str, Any]] = {}
        for tr in task_results:
            name = tr["name"]
            if name not in by_task:
                by_task[name] = {"runs": 0, "successes": 0, "total_attempts": 0}
            by_task[name]["runs"] += 1
            by_task[name]["total_attempts"] += tr["attempts"]
            if tr["success"]:
                by_task[name]["successes"] += 1
        by_task_summary = {}
        for name, stats in by_task.items():
            by_task_summary[name] = {
                "success_rate": stats["successes"] / stats["runs"],
                "runs": stats["runs"],
                "avg_attempts": stats["total_attempts"] / stats["runs"],
            }

        condition_result = {
            "condition": condition_name,
            "description": desc,
            "summary": {
                "success_rate": successes / len(tasks) if tasks else 0.0,
                "num_samples": len(tasks),
                "avg_attempts": total_attempts / len(tasks) if tasks else 0.0,
                "avg_attempt_time_seconds": total_time / total_attempts if total_attempts else 0.0,
                "categories": category_summary,
                "by_task": by_task_summary,
            },
            "task_results": task_results,
            "wall_clock_seconds": time.monotonic() - t0,
        }

    sr = condition_result["summary"]["success_rate"]
    wc = condition_result["wall_clock_seconds"]
    print(f"  Result: {sr:.1%} success rate, {wc:.1f}s wall clock")
    return condition_result


def run_ablation_study(
    *,
    provider_name: str,
    model: str,
    output_dir: str | Path,
    repeats: int = 4,
    base_url: str = "http://localhost:11434",
    request_timeout: float = 240.0,
    temperature: float = 0.0,
    conditions: Optional[list[str]] = None,
    difficulty: str = "easy",
    benchmark: str = "custom",
    max_workers: int = 1,
) -> dict[str, Any]:
    """Run the full ablation study."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(
        include_coding=True, include_reasoning=True,
        repeats=repeats, difficulty=difficulty, benchmark=benchmark,
    )

    provider = build_provider(
        provider_name=provider_name,
        model=model,
        temperature=temperature,
        base_url=base_url,
        request_timeout=request_timeout,
    )
    benchmark = LLMBenchmark(llm_provider=provider)

    print(f"\n{'=' * 60}")
    print(f"SOME Ablation Study: {provider_name} / {model}")
    print(f"{'=' * 60}")
    print(f"Tasks: {len(tasks)} | Repeats: {repeats}")

    selected = conditions or list(ABLATION_CONDITIONS.keys())
    results: dict[str, Any] = {}

    for cond_name in selected:
        if cond_name not in ABLATION_CONDITIONS:
            print(f"WARNING: Unknown condition '{cond_name}', skipping")
            continue
        cond_result = run_condition(
            condition_name=cond_name,
            condition_spec=ABLATION_CONDITIONS[cond_name],
            benchmark=benchmark,
            tasks=tasks,
            output_dir=output_root,
            max_workers=max_workers,
        )
        results[cond_name] = cond_result

        # Save incremental progress
        progress_path = output_root / "ablation_progress.json"
        progress_path.write_text(
            json.dumps(
                {"model": model, "provider": provider_name, "conditions": results},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    # Build comparison
    comparison = _build_comparison(results)

    final = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "provider": provider_name,
        "num_tasks": len(tasks),
        "repeats": repeats,
        "conditions": results,
        "comparison": comparison,
    }

    report_path = output_root / "ablation_report.json"
    report_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _print_summary(comparison, model)
    return final


def _build_comparison(results: dict[str, Any]) -> dict[str, Any]:
    """Build a comparison table from all condition results."""
    rows = []
    baseline_sr = None
    prev_sr = None

    for cond_name in sorted(results.keys()):
        cond = results[cond_name]
        sr = cond["summary"]["success_rate"]
        avg_att = cond["summary"].get("avg_attempts", 1.0)
        wc = cond.get("wall_clock_seconds", 0.0)
        cats = cond["summary"].get("categories", {})
        coding_sr = cats.get("coding", {}).get("success_rate", 0.0)
        reasoning_sr = cats.get("reasoning", {}).get("success_rate", 0.0)

        if baseline_sr is None:
            baseline_sr = sr
        delta_from_baseline = sr - baseline_sr
        delta_from_prev = sr - prev_sr if prev_sr is not None else 0.0
        prev_sr = sr

        rows.append({
            "condition": cond_name,
            "description": cond["description"],
            "success_rate": sr,
            "coding_success_rate": coding_sr,
            "reasoning_success_rate": reasoning_sr,
            "avg_attempts": avg_att,
            "wall_clock_seconds": wc,
            "delta_from_baseline": delta_from_baseline,
            "marginal_delta": delta_from_prev,
        })

    return {"rows": rows}


def _print_summary(comparison: dict[str, Any], model: str) -> None:
    """Print human-readable ablation summary."""
    print(f"\n{'=' * 72}")
    print(f"ABLATION RESULTS: {model}")
    print(f"{'=' * 72}")
    print(f"{'Condition':<35} {'Success':>8} {'Coding':>8} {'Reason':>8} {'Delta':>8} {'Margin':>8} {'Att':>5}")
    print(f"{'─' * 35} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 5}")

    for row in comparison["rows"]:
        name = row["condition"]
        sr = f"{row['success_rate']:.1%}"
        cod = f"{row['coding_success_rate']:.1%}"
        rea = f"{row['reasoning_success_rate']:.1%}"
        delta = f"{row['delta_from_baseline']:+.1%}" if row["delta_from_baseline"] != 0 else "base"
        margin = f"{row['marginal_delta']:+.1%}" if row["marginal_delta"] != 0 else "—"
        att = f"{row['avg_attempts']:.1f}"
        print(f"{name:<35} {sr:>8} {cod:>8} {rea:>8} {delta:>8} {margin:>8} {att:>5}")

    print(f"{'─' * 35} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 5}")
    print()

    # Highlight biggest marginal contributor
    rows = comparison["rows"]
    if len(rows) > 1:
        best_margin = max(rows[1:], key=lambda r: r["marginal_delta"])
        print(f"Largest marginal contributor: {best_margin['condition']} (+{best_margin['marginal_delta']:.1%})")
        print(f"  = {best_margin['description']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="SOME ablation study")
    parser.add_argument("--provider", default="local")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--request-timeout", type=float, default=240.0)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument(
        "--conditions",
        nargs="*",
        help="Specific conditions to run (default: all). E.g. 1_baseline 2_verify_only",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "hard", "all"],
        default="easy",
        help="Task difficulty: easy (original 10), hard (10 harder), all (20 tasks)",
    )
    parser.add_argument(
        "--benchmark",
        choices=["custom", "humaneval"],
        default="custom",
        help="Benchmark suite: custom (our tasks) or humaneval (OpenAI HumanEval 164)",
    )
    parser.add_argument(
        "--parallel", "-j",
        type=int,
        default=1,
        help="Number of parallel workers for task execution (default: 1, try 4-8 for GPU)",
    )
    args = parser.parse_args()

    run_ablation_study(
        provider_name=args.provider,
        model=args.model,
        output_dir=args.output_dir,
        repeats=args.repeats,
        base_url=args.base_url,
        request_timeout=args.request_timeout,
        temperature=args.temp,
        conditions=args.conditions,
        difficulty=args.difficulty,
        benchmark=args.benchmark,
        max_workers=args.parallel,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
