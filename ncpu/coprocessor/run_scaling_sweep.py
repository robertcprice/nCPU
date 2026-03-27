#!/usr/bin/env python3
"""Run coprocessor scaling sweep across multiple model sizes.

Tests the dual-LR unfreezing approach across the Qwen model family
to produce a scaling curve showing how coprocessor benefit varies
with model capacity.

Usage:
    python3 -m ncpu.coprocessor.run_scaling_sweep
    python3 -m ncpu.coprocessor.run_scaling_sweep --only qwen3-0.6b qwen3-8b
"""

import json
import subprocess
import sys
import time
from pathlib import Path

# Batch sizes tuned for 48GB VRAM with bf16 + unfreezing 2 layers
MODELS = [
    # --- Qwen3.5 family (latest generation) ---
    {
        "name": "qwen3.5-0.8b",
        "model_id": "Qwen/Qwen3.5-0.8B",
        "batch_size": 16,
        "desc": "Qwen3.5 0.8B (smallest Qwen3.5)",
    },
    {
        "name": "qwen3.5-2b",
        "model_id": "Qwen/Qwen3.5-2B",
        "batch_size": 16,
        "desc": "Qwen3.5 2B",
    },
    {
        "name": "qwen3.5-4b",
        "model_id": "Qwen/Qwen3.5-4B",
        "batch_size": 8,
        "desc": "Qwen3.5 4B",
    },
    {
        "name": "qwen3.5-9b",
        "model_id": "Qwen/Qwen3.5-9B",
        "batch_size": 4,
        "desc": "Qwen3.5 9B",
    },
    # --- Qwen3 family ---
    {
        "name": "qwen3-0.6b",
        "model_id": "Qwen/Qwen3-0.6B",
        "batch_size": 16,
        "desc": "Qwen3 0.6B (smallest Qwen3)",
    },
    {
        "name": "qwen3-1.7b",
        "model_id": "Qwen/Qwen3-1.7B",
        "batch_size": 16,
        "desc": "Qwen3 1.7B",
    },
    {
        "name": "qwen3-4b",
        "model_id": "Qwen/Qwen3-4B",
        "batch_size": 8,
        "desc": "Qwen3 4B",
    },
    {
        "name": "qwen3-8b",
        "model_id": "Qwen/Qwen3-8B",
        "batch_size": 4,
        "desc": "Qwen3 8B",
    },
    # --- Qwen2.5 family ---
    {
        "name": "qwen2.5-0.5b",
        "model_id": "Qwen/Qwen2.5-0.5B",
        "batch_size": 16,
        "desc": "Qwen2.5 0.5B (smallest Qwen2.5)",
    },
    {
        "name": "qwen2.5-1.5b",
        "model_id": "Qwen/Qwen2.5-1.5B",
        "batch_size": 16,
        "desc": "Qwen2.5 1.5B",
    },
    {
        "name": "qwen2.5-3b",
        "model_id": "Qwen/Qwen2.5-3B",
        "batch_size": 8,
        "desc": "Qwen2.5 3B",
    },
]


def run_model(m: dict, output_base: str) -> dict:
    """Run coprocessor training for a single model."""
    name = m["name"]
    output_dir = f"{output_base}/{name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "ncpu.coprocessor.train",
        "--model", m["model_id"],
        "--dataset", "synthetic+gsm8k",
        "--steps", "2000",
        "--lr", "1e-3",
        "--backbone-lr", "5e-5",
        "--batch-size", str(m["batch_size"]),
        "--layers", "-1",
        "--n-bits", "8",
        "--unfreeze-last-n", "2",
        "--eval-every", "500",
        "--output-dir", output_dir,
        "--verbose",
    ]

    print(f"\n{'='*70}")
    print(f"MODEL: {name} ({m['desc']})")
    print(f"  HF ID: {m['model_id']}")
    print(f"  Batch size: {m['batch_size']}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    report_path = Path(output_dir) / "training_report.json"
    report = {}
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)

    return {
        "name": name,
        "model_id": m["model_id"],
        "desc": m["desc"],
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "report": report,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Coprocessor scaling sweep")
    parser.add_argument("--only", nargs="+",
                        choices=[m["name"] for m in MODELS],
                        help="Run only specific models")
    args = parser.parse_args()

    output_base = "training_results/scaling_sweep"
    Path(output_base).mkdir(parents=True, exist_ok=True)

    to_run = MODELS
    if args.only:
        to_run = [m for m in MODELS if m["name"] in args.only]

    results = []
    for m in to_run:
        try:
            r = run_model(m, output_base)
            results.append(r)
            if r["returncode"] != 0:
                print(f"\nWARNING: {m['name']} exited with code {r['returncode']}")
        except Exception as e:
            print(f"\nERROR running {m['name']}: {e}")
            results.append({"name": m["name"], "error": str(e)})

    # Summary table
    print(f"\n{'='*90}")
    print("SCALING SWEEP SUMMARY")
    print(f"{'='*90}")
    print(f"{'Model':<20} {'Params':>12} {'Trainable':>12} {'Baseline':>10} {'Final':>10} {'Delta':>8} {'Time':>8}")
    print("-" * 85)

    for r in results:
        name = r["name"]
        report = r.get("report", {})
        result_data = report.get("result", {})
        baseline = report.get("baseline_eval", {}).get("overall_accuracy", 0)
        final = report.get("final_eval", {}).get("overall_accuracy", 0)
        delta = final - baseline
        total_p = result_data.get("total_params", 0)
        train_p = result_data.get("trainable_params", 0)
        elapsed = r.get("elapsed_seconds", 0)

        def fmt_params(n):
            if n >= 1e9: return f"{n/1e9:.1f}B"
            if n >= 1e6: return f"{n/1e6:.0f}M"
            if n >= 1e3: return f"{n/1e3:.0f}K"
            return str(n)

        print(f"{name:<20} {fmt_params(total_p):>12} {fmt_params(train_p):>12} "
              f"{baseline:>9.1%} {final:>9.1%} {delta:>+7.1%} {elapsed:>7.0f}s")

    # Save summary
    summary_path = Path(output_base) / "scaling_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {summary_path}")


if __name__ == "__main__":
    main()
