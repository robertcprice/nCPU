#!/usr/bin/env python3
"""Run coprocessor improvement experiments on Qwen2.5-0.5B.

Five experiments testing how to improve coprocessor integration:
  1. Baseline rerun — frozen backbone, 1 layer, 8-bit (control)
  2. Backbone unfreezing — unfreeze last 2 layers with dual LR
  3. Multi-layer injection — inject at 4 layers instead of 1
  4. Wider expert — 16-bit ALU (double info bandwidth)
  5. Combined — best of all three

Each experiment trains for 2000 steps on synthetic+gsm8k, same as baseline.
Results saved to training_results/coprocessor_experiments/.

Key insight: when unfreezing backbone, we need DUAL learning rates.
The coprocessor needs ~1e-3 to learn routing/projections, while the
backbone needs ~5e-5 to gently adapt. A single LR means one or the
other can't learn properly.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENTS = [
    {
        "name": "baseline_rerun",
        "desc": "Baseline rerun (frozen backbone, 1 layer, 8-bit)",
        "args": [
            "--model", "Qwen/Qwen2.5-0.5B",
            "--dataset", "synthetic+gsm8k",
            "--steps", "2000",
            "--lr", "1e-3",
            "--batch-size", "16",
            "--layers", "-1",
            "--n-bits", "8",
            "--unfreeze-last-n", "0",
            "--eval-every", "500",
            "--verbose",
        ],
    },
    {
        "name": "unfreeze_last2",
        "desc": "Unfreeze last 2 transformer layers + LM head (dual LR)",
        "args": [
            "--model", "Qwen/Qwen2.5-0.5B",
            "--dataset", "synthetic+gsm8k",
            "--steps", "2000",
            "--lr", "1e-3",           # coprocessor LR (high, for routing/projections)
            "--backbone-lr", "5e-5",  # backbone LR (conservative for fine-tuning)
            "--batch-size", "16",
            "--layers", "-1",
            "--n-bits", "8",
            "--unfreeze-last-n", "2",
            "--eval-every", "500",
            "--verbose",
        ],
    },
    {
        "name": "multi_layer",
        "desc": "Inject at 4 layers: [-1, -2, -4, -8]",
        "args": [
            "--model", "Qwen/Qwen2.5-0.5B",
            "--dataset", "synthetic+gsm8k",
            "--steps", "2000",
            "--lr", "1e-3",
            "--batch-size", "16",
            "--layers", "-1", "-2", "-4", "-8",
            "--n-bits", "8",
            "--unfreeze-last-n", "0",
            "--eval-every", "500",
            "--verbose",
        ],
    },
    {
        "name": "wide_16bit",
        "desc": "16-bit ALU expert (double information bandwidth)",
        "args": [
            "--model", "Qwen/Qwen2.5-0.5B",
            "--dataset", "synthetic+gsm8k",
            "--steps", "2000",
            "--lr", "1e-3",
            "--batch-size", "16",
            "--layers", "-1",
            "--n-bits", "16",
            "--unfreeze-last-n", "0",
            "--eval-every", "500",
            "--verbose",
        ],
    },
    {
        "name": "combined_best",
        "desc": "Combined: unfreeze 2 + multi-layer [-1,-2,-4] + 16-bit (dual LR)",
        "args": [
            "--model", "Qwen/Qwen2.5-0.5B",
            "--dataset", "synthetic+gsm8k",
            "--steps", "2000",
            "--lr", "1e-3",
            "--backbone-lr", "5e-5",
            "--batch-size", "16",
            "--layers", "-1", "-2", "-4",
            "--n-bits", "16",
            "--unfreeze-last-n", "2",
            "--eval-every", "500",
            "--verbose",
        ],
    },
]


def run_experiment(exp: dict, output_base: str) -> dict:
    """Run a single experiment and return its report."""
    name = exp["name"]
    output_dir = f"{output_base}/{name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "ncpu.coprocessor.train",
        *exp["args"],
        "--output-dir", output_dir,
    ]

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  {exp['desc']}")
    print(f"  Output: {output_dir}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    # Load report if it exists
    report_path = Path(output_dir) / "training_report.json"
    report = {}
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)

    return {
        "name": name,
        "desc": exp["desc"],
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "report": report,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run coprocessor experiments")
    parser.add_argument("--only", nargs="+",
                        choices=[e["name"] for e in EXPERIMENTS],
                        help="Run only specific experiments")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline rerun")
    args = parser.parse_args()

    output_base = "training_results/coprocessor_experiments"
    Path(output_base).mkdir(parents=True, exist_ok=True)

    to_run = EXPERIMENTS
    if args.only:
        to_run = [e for e in EXPERIMENTS if e["name"] in args.only]
    elif args.skip_baseline:
        to_run = [e for e in EXPERIMENTS if e["name"] != "baseline_rerun"]

    results = []
    for exp in to_run:
        try:
            r = run_experiment(exp, output_base)
            results.append(r)
            if r["returncode"] != 0:
                print(f"\nWARNING: {exp['name']} exited with code {r['returncode']}")
        except Exception as e:
            print(f"\nERROR running {exp['name']}: {e}")
            results.append({
                "name": exp["name"],
                "desc": exp["desc"],
                "error": str(e),
            })

    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"{'Experiment':<25} {'Baseline':>10} {'Final':>10} {'Delta':>8} {'Params':>10} {'Time':>8}")
    print("-" * 75)

    for r in results:
        name = r["name"]
        report = r.get("report", {})
        baseline = report.get("baseline_eval", {}).get("overall_accuracy", 0)
        final = report.get("final_eval", {}).get("overall_accuracy", 0)
        delta = final - baseline
        params = report.get("result", {}).get("trainable_params", 0)
        elapsed = r.get("elapsed_seconds", 0)

        print(f"{name:<25} {baseline:>9.1%} {final:>9.1%} {delta:>+7.1%} {params:>10,} {elapsed:>7.0f}s")

    # Save summary
    summary_path = Path(output_base) / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {summary_path}")


if __name__ == "__main__":
    main()
