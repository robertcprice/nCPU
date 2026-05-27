#!/usr/bin/env python3
"""Compare two benchmark artifact directories and flag regressions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.utils.artifact_compare import (  # noqa: E402
    ComparisonThresholds,
    compare_artifact_dirs,
    render_markdown_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare publication artifact directories")
    parser.add_argument("--baseline-dir", type=Path, required=True, help="Reference artifact directory")
    parser.add_argument("--candidate-dir", type=Path, required=True, help="New artifact directory to compare")
    parser.add_argument("--json-output", type=Path, help="Optional path for the JSON comparison report")
    parser.add_argument("--markdown-output", type=Path, help="Optional path for the Markdown comparison report")
    parser.add_argument(
        "--allow-platform-mismatch",
        action="store_true",
        help="Allow performance comparisons across different system/machine pairs",
    )
    parser.add_argument(
        "--max-baseline-overhead-increase-points",
        type=float,
        default=5.0,
        help="Maximum allowed increase in baseline_comparison overhead percentage points",
    )
    parser.add_argument(
        "--max-baseline-ips-regression-pct",
        type=float,
        default=10.0,
        help="Maximum allowed IPS regression percentage for baseline comparison checks",
    )
    parser.add_argument(
        "--max-ablation-ips-regression-pct",
        type=float,
        default=10.0,
        help="Maximum allowed IPS regression percentage for ablation checks",
    )
    parser.add_argument(
        "--max-real-workload-ips-regression-pct",
        type=float,
        default=10.0,
        help="Maximum allowed IPS regression percentage for real-workload checks",
    )
    parser.add_argument(
        "--max-real-workload-overhead-increase-points",
        type=float,
        default=10.0,
        help="Maximum allowed increase in real-workload aggregate overhead percentage points",
    )
    parser.add_argument(
        "--max-gpu-only-matrix-ips-regression-pct",
        type=float,
        default=10.0,
        help="Maximum allowed IPS regression percentage for GPU-only matrix workloads",
    )
    args = parser.parse_args()

    thresholds = ComparisonThresholds(
        max_baseline_overhead_increase_points=args.max_baseline_overhead_increase_points,
        max_baseline_ips_regression_pct=args.max_baseline_ips_regression_pct,
        max_ablation_ips_regression_pct=args.max_ablation_ips_regression_pct,
        max_real_workload_ips_regression_pct=args.max_real_workload_ips_regression_pct,
        max_real_workload_overhead_increase_points=args.max_real_workload_overhead_increase_points,
        max_gpu_only_matrix_ips_regression_pct=args.max_gpu_only_matrix_ips_regression_pct,
    )

    report = compare_artifact_dirs(
        args.baseline_dir,
        args.candidate_dir,
        thresholds=thresholds,
        allow_platform_mismatch=args.allow_platform_mismatch,
    )

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[compare] Wrote JSON report to {args.json_output}")

    markdown = render_markdown_report(report)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown)
        print(f"[compare] Wrote Markdown report to {args.markdown_output}")

    summary = report["summary"]
    print(
        f"[compare] Overall={summary['overall_status'].upper()} "
        f"pass={summary['passed']} fail={summary['failed']} "
        f"warn={summary['warnings']} skip={summary['skipped']}"
    )
    for check in report["checks"]:
        if check["status"] in {"fail", "warn"}:
            print(f"[{check['status']}] {check['name']}: {check['summary']}")

    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
