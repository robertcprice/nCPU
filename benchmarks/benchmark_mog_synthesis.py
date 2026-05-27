#!/usr/bin/env python3
"""Reproducible harness for the differentiable Mog synthesis benchmark.

Runs every factory in egdc.mog.benchmark.PROBLEM_FACTORIES across N variants
(default 5), solves each problem through egdc.mog.solvers.search_solver, and
emits a structured summary suitable for paper tables and regression tracking.

Exit code is non-zero when observed coverage falls below --min-coverage so CI
detects regressions automatically.

Usage:
    python benchmarks/benchmark_mog_synthesis.py
    python benchmarks/benchmark_mog_synthesis.py --variants 3 --json out.json
    python benchmarks/benchmark_mog_synthesis.py --use-compiler
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from egdc.mog.benchmark import PROBLEM_FACTORIES, get_benchmark  # noqa: E402
from egdc.mog.solvers.search_solver import solve_problem  # noqa: E402


def _run_one(problem, use_compiler: bool):
    t0 = time.perf_counter()
    result = solve_problem(problem, use_compiler=use_compiler)
    return result, time.perf_counter() - t0


def run_benchmark(
    *,
    variants: int,
    seed: int,
    use_compiler: bool,
    verbose: bool,
):
    problems = get_benchmark(seed=seed, variants_per_factory=variants)
    rows = []
    by_factory_total: dict[str, int] = defaultdict(int)
    by_factory_passed: dict[str, int] = defaultdict(int)
    method_counts: dict[str, int] = defaultdict(int)
    times: list[float] = []
    started = time.perf_counter()

    for idx, problem in enumerate(problems):
        result, elapsed = _run_one(problem, use_compiler)
        times.append(elapsed)
        factory = problem.name
        by_factory_total[factory] += 1
        if result.success:
            by_factory_passed[factory] += 1
            method_counts[result.method or "unknown"] += 1
        rows.append(
            {
                "index": idx,
                "factory": factory,
                "category": problem.category,
                "success": bool(result.success),
                "method": result.method,
                "loss": float(result.loss) if result.loss is not None else None,
                "compiler_pass": bool(result.compiler_pass),
                "seconds": round(elapsed, 4),
            }
        )
        if verbose:
            status = "OK " if result.success else "FAIL"
            print(
                f"  [{idx+1:4d}/{len(problems)}] {status} {factory:32s} "
                f"method={result.method!s:20s} loss={result.loss!s:10s} "
                f"dt={elapsed:5.2f}s",
                flush=True,
            )

    wall = time.perf_counter() - started
    total = len(problems)
    passed = sum(by_factory_passed.values())
    failing_factories = sorted(
        f for f in by_factory_total if by_factory_passed[f] < by_factory_total[f]
    )
    summary = {
        "seed": seed,
        "variants_per_factory": variants,
        "factory_count": len(PROBLEM_FACTORIES),
        "problem_count": total,
        "passed": passed,
        "coverage": round(passed / total, 6) if total else 0.0,
        "wall_seconds": round(wall, 3),
        "use_compiler": use_compiler,
        "method_counts": dict(sorted(method_counts.items(), key=lambda x: -x[1])),
        "per_factory": {
            f: {"passed": by_factory_passed[f], "total": by_factory_total[f]}
            for f in sorted(by_factory_total)
        },
        "failing_factories": failing_factories,
        "timing": {
            "mean_seconds": round(statistics.mean(times), 4) if times else 0.0,
            "median_seconds": round(statistics.median(times), 4) if times else 0.0,
            "max_seconds": round(max(times), 4) if times else 0.0,
            "p95_seconds": round(
                statistics.quantiles(times, n=20)[-1] if len(times) >= 20 else max(times),
                4,
            ) if times else 0.0,
        },
    }
    return {"summary": summary, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=int, default=5, help="variants per factory")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for variant generation")
    parser.add_argument("--use-compiler", action="store_true", help="enable the differentiable compiler path")
    parser.add_argument("--json", type=Path, default=None, help="write full structured result as JSON")
    parser.add_argument("--verbose", action="store_true", help="print each problem as it runs")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=1.0,
        help="fail with non-zero exit when coverage falls below this fraction (default 1.0)",
    )
    args = parser.parse_args()

    report = run_benchmark(
        variants=args.variants,
        seed=args.seed,
        use_compiler=args.use_compiler,
        verbose=args.verbose,
    )
    summary = report["summary"]
    print(
        f"mog-synthesis: {summary['passed']}/{summary['problem_count']} "
        f"({summary['coverage']*100:.2f}%) in {summary['wall_seconds']}s "
        f"| factories={summary['factory_count']} variants={summary['variants_per_factory']} "
        f"seed={summary['seed']} compiler={summary['use_compiler']}"
    )
    print(f"timing: mean={summary['timing']['mean_seconds']}s "
          f"median={summary['timing']['median_seconds']}s "
          f"p95={summary['timing']['p95_seconds']}s "
          f"max={summary['timing']['max_seconds']}s")
    print("method counts:")
    for method, count in summary["method_counts"].items():
        print(f"  {method}: {count}")
    if summary["failing_factories"]:
        print("\nfailing factories (per_factory counts):")
        for factory in summary["failing_factories"]:
            detail = summary["per_factory"][factory]
            print(f"  {factory}: {detail['passed']}/{detail['total']}")
    else:
        print("\nno failing factories.")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nwrote {args.json}")

    if summary["coverage"] < args.min_coverage:
        print(
            f"\nCOVERAGE REGRESSION: {summary['coverage']*100:.2f}% < "
            f"{args.min_coverage*100:.2f}%",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
