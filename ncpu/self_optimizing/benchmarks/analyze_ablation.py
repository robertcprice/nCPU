#!/usr/bin/env python3
"""Cross-scale ablation analysis for SOME (Self-Optimizing Model Engine).

Loads ablation results from multiple model scales and produces a comprehensive
comparison report showing which components drive improvement at each scale.

Usage:
    python3 -m ncpu.self_optimizing.benchmarks.analyze_ablation \
        --results training_results/ablation_4b_*/*/ablation_report.json \
                  training_results/ablation_9b_*/*/ablation_report.json \
                  training_results/ablation_27b_*/*/ablation_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONDITION_ORDER = [
    "1_baseline",
    "2_verify_only",
    "3_verify_plan",
    "4_verify_plan_heads",
    "5_verify_plan_heads_memory",
    "6_full_some",
]

CONDITION_SHORT = {
    "1_baseline": "Baseline",
    "2_verify_only": "+Retry",
    "3_verify_plan": "+Plan",
    "4_verify_plan_heads": "+Heads",
    "5_verify_plan_heads_memory": "+Memory",
    "6_full_some": "Full SOME",
}

COMPONENT_LABELS = {
    "2_verify_only": "Verify/Retry Loop",
    "3_verify_plan": "Planning Step",
    "4_verify_plan_heads": "Learned Heads",
    "5_verify_plan_heads_memory": "Recurrent Memory",
    "6_full_some": "Fast Weights",
}


def load_report(path: str | Path) -> dict[str, Any]:
    """Load an ablation report JSON."""
    with open(path, "r") as f:
        return json.load(f)


def extract_model_name(report: dict[str, Any]) -> str:
    """Extract model name from report metadata or path."""
    # Try to find model name in the data
    for cond in report.get("conditions", {}).values():
        for tr in cond.get("task_results", []):
            if "model" in tr:
                return tr["model"]
    # Fallback: look at directory structure
    return report.get("model", "unknown")


def print_cross_scale_table(reports: list[tuple[str, dict[str, Any]]]) -> None:
    """Print the main cross-scale comparison table."""
    header_parts = [f"{'Condition':<18}"]
    for label, _ in reports:
        header_parts.append(f"{label + ' Rate':>12}")
        header_parts.append(f"{'Margin':>8}")
    print("".join(header_parts))
    print("-" * (18 + len(reports) * 20))

    for cond in CONDITION_ORDER:
        short = CONDITION_SHORT.get(cond, cond)
        parts = [f"{short:<18}"]
        for _label, report in reports:
            cdata = report["conditions"].get(cond, {})
            sr = cdata.get("summary", {}).get("success_rate", 0)
            # Compute margin from previous condition
            idx = CONDITION_ORDER.index(cond)
            if idx == 0:
                margin_str = "—"
            else:
                prev_cond = CONDITION_ORDER[idx - 1]
                prev_sr = report["conditions"].get(prev_cond, {}).get("summary", {}).get("success_rate", 0)
                margin = (sr - prev_sr) * 100
                margin_str = f"+{margin:.0f}%" if margin > 0 else ("0%" if margin == 0 else f"{margin:.0f}%")
            parts.append(f"{sr*100:>11.0f}%")
            parts.append(f"{margin_str:>8}")
        print("".join(parts))


def print_per_task_breakdown(reports: list[tuple[str, dict[str, Any]]]) -> None:
    """Print per-task results across scales, only for interesting tasks."""
    # Collect all task names
    all_tasks: set[str] = set()
    for _, report in reports:
        for cond in report["conditions"].values():
            all_tasks.update(cond.get("summary", {}).get("by_task", {}).keys())

    for task in sorted(all_tasks):
        # Check if this task has any variance across conditions or models
        has_variance = False
        for _, report in reports:
            rates = []
            for cond_name in CONDITION_ORDER:
                cdata = report["conditions"].get(cond_name, {})
                bt = cdata.get("summary", {}).get("by_task", {}).get(task, {})
                rates.append(bt.get("success_rate", 0))
            if len(set(rates)) > 1 or min(rates) < 1.0:
                has_variance = True
                break

        if not has_variance:
            continue

        print(f"\n  {task}:")
        for cond_name in CONDITION_ORDER:
            short = CONDITION_SHORT.get(cond_name, cond_name)
            parts = [f"    {short:<16}"]
            for label, report in reports:
                cdata = report["conditions"].get(cond_name, {})
                bt = cdata.get("summary", {}).get("by_task", {}).get(task, {})
                sr = bt.get("success_rate", 0)
                att = bt.get("avg_attempts", 1)
                status = "OK" if sr == 1.0 else f"{sr*100:.0f}%"
                parts.append(f"  {label}: {status}({att:.1f}att)")
            print("".join(parts))


def print_marginal_contributions(reports: list[tuple[str, dict[str, Any]]]) -> None:
    """Print which component provides the marginal contribution at each scale."""
    print(f"\n{'Component':<22}", end="")
    for label, _ in reports:
        print(f"{label:>15}", end="")
    print()
    print("-" * (22 + len(reports) * 15))

    for i, cond in enumerate(CONDITION_ORDER[1:], start=1):
        comp = COMPONENT_LABELS.get(cond, cond)
        print(f"{comp:<22}", end="")
        for _, report in reports:
            sr = report["conditions"][cond]["summary"]["success_rate"]
            prev_sr = report["conditions"][CONDITION_ORDER[i - 1]]["summary"]["success_rate"]
            delta = (sr - prev_sr) * 100
            delta_str = f"+{delta:.0f}%" if delta > 0 else ("0%" if delta == 0 else f"{delta:.0f}%")
            star = " ***" if delta > 0 else ""
            print(f"{delta_str + star:>15}", end="")
        print()


def print_efficiency_analysis(reports: list[tuple[str, dict[str, Any]]]) -> None:
    """Print wall-clock and attempt efficiency."""
    print(f"\n{'Condition':<18}", end="")
    for label, _ in reports:
        print(f"{label + ' Wall':>12}  {label + ' Att':>10}", end="")
    print()
    print("-" * (18 + len(reports) * 24))

    for cond in CONDITION_ORDER:
        short = CONDITION_SHORT.get(cond, cond)
        print(f"{short:<18}", end="")
        for _, report in reports:
            cdata = report["conditions"].get(cond, {})
            wall = cdata.get("wall_clock_seconds", 0)
            att = cdata.get("summary", {}).get("avg_attempts", 1)
            print(f"{wall:>11.0f}s  {att:>10.2f}", end="")
        print()


def print_findings(reports: list[tuple[str, dict[str, Any]]]) -> None:
    """Print key findings from the cross-scale analysis."""
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Find which component provides the margin at each scale
    for label, report in reports:
        baseline_sr = report["conditions"]["1_baseline"]["summary"]["success_rate"]
        full_sr = report["conditions"]["6_full_some"]["summary"]["success_rate"]
        delta = (full_sr - baseline_sr) * 100

        # Find the first condition that achieves full improvement
        driver = None
        for i, cond in enumerate(CONDITION_ORDER[1:], start=1):
            sr = report["conditions"][cond]["summary"]["success_rate"]
            if sr >= full_sr:
                driver = cond
                break

        driver_label = COMPONENT_LABELS.get(driver, driver) if driver else "None"
        print(f"\n  {label}: Baseline {baseline_sr*100:.0f}% -> SOME {full_sr*100:.0f}% (delta: +{delta:.0f}%)")
        print(f"    Primary driver: {driver_label}")
        if driver:
            print(f"    All subsequent components add: 0%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-scale SOME ablation analysis")
    parser.add_argument("--results", nargs="+", required=True, help="Paths to ablation_report.json files")
    parser.add_argument("--json-output", type=str, help="Optional: write combined JSON report")
    args = parser.parse_args()

    reports: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(args.results):
        report = load_report(path)
        # Extract scale from path
        p = Path(path)
        for part in p.parts:
            if "4b" in part:
                label = "4B"
                break
            elif "9b" in part:
                label = "9B"
                break
            elif "27b" in part:
                label = "27B"
                break
        else:
            label = p.parent.name
        reports.append((label, report))

    print("=" * 80)
    print(f"SOME ABLATION: CROSS-SCALE COMPARISON ({', '.join(l for l, _ in reports)})")
    print("=" * 80)

    print("\n## Overall Success Rate & Marginal Gains\n")
    print_cross_scale_table(reports)

    print("\n\n## Marginal Component Contributions\n")
    print_marginal_contributions(reports)

    print("\n\n## Efficiency (Wall Clock & Average Attempts)\n")
    print_efficiency_analysis(reports)

    print("\n\n## Per-Task Breakdown (Failing/Variable Tasks Only)")
    print_per_task_breakdown(reports)

    print_findings(reports)

    if args.json_output:
        combined = {
            "models": {label: report for label, report in reports},
            "analysis": {
                "model_count": len(reports),
                "conditions": CONDITION_ORDER,
                "component_labels": COMPONENT_LABELS,
            },
        }
        with open(args.json_output, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\nCombined JSON written to: {args.json_output}")


if __name__ == "__main__":
    main()
