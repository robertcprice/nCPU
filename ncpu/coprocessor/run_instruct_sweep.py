#!/usr/bin/env python3
"""Run coprocessor training sweep on instruct/chat models.

Fixes the base→instruct transfer problem by training coprocessor weights
directly ON instruct models with confidence-aware gating. This prevents
the activation distribution mismatch that caused -5% to -40% degradation
when applying base-trained weights to instruct models.

Usage:
    python3 -m ncpu.coprocessor.run_instruct_sweep
    python3 -m ncpu.coprocessor.run_instruct_sweep --only qwen3.5-2b
    python3 -m ncpu.coprocessor.run_instruct_sweep --no-confidence-aware
"""

import json
import subprocess
import sys
import time
from pathlib import Path

INSTRUCT_MODELS = [
    {
        "name": "qwen3.5-1.5b",
        "model_id": "Qwen/Qwen3.5-1.5B",
        "batch_size": 16,
        "desc": "Qwen3.5 1.5B (VL, text-only extraction)",
    },
    {
        "name": "qwen3.5-2b",
        "model_id": "Qwen/Qwen3.5-2B",
        "batch_size": 16,
        "desc": "Qwen3.5 2B (VL, text-only extraction) — best delta in sweep",
    },
    {
        "name": "qwen3.5-4b",
        "model_id": "Qwen/Qwen3.5-4B",
        "batch_size": 8,
        "desc": "Qwen3.5 4B (VL, text-only extraction) — +51% delta",
    },
    {
        "name": "qwen3.5-9b",
        "model_id": "Qwen/Qwen3.5-9B",
        "batch_size": 4,
        "desc": "Qwen3.5 9B (VL, text-only extraction) — +51% delta",
    },
]


def run_model(
    m: dict,
    output_base: str,
    confidence_aware: bool,
    max_gate: float,
    deterministic_alu: bool = False,
    gate_warmup_steps: int = 200,
    calibrate: bool = False,
) -> dict:
    """Run coprocessor training for a single instruct model."""
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
        "--target-load", "0.01",
        "--max-gate", str(max_gate),
        "--gate-warmup-steps", str(gate_warmup_steps),
        "--eval-every", "500",
        "--output-dir", output_dir,
        "--verbose",
    ]

    if confidence_aware:
        cmd.append("--confidence-aware")
    if deterministic_alu:
        cmd.append("--deterministic-alu")
    if calibrate:
        cmd.append("--calibrate")

    print(f"\n{'='*70}")
    print(f"INSTRUCT MODEL: {name} ({m['desc']})")
    print(f"  HF ID: {m['model_id']}")
    print(f"  Batch size: {m['batch_size']}")
    print(f"  Confidence-aware: {confidence_aware}")
    print(f"  Deterministic ALU: {deterministic_alu}")
    print(f"  Gate warmup: {gate_warmup_steps} steps")
    print(f"  Max gate: {max_gate}")
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
        "confidence_aware": confidence_aware,
        "max_gate": max_gate,
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "report": report,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Coprocessor instruct model sweep")
    parser.add_argument("--only", nargs="+",
                        choices=[m["name"] for m in INSTRUCT_MODELS],
                        help="Run only specific models")
    parser.add_argument("--no-confidence-aware", action="store_true",
                        help="Disable confidence-aware gating (for ablation)")
    parser.add_argument("--max-gate", type=float, default=0.1,
                        help="Hard cap on gate value (default: 0.1)")
    parser.add_argument("--deterministic-alu", action="store_true",
                        help="Use exact arithmetic with STE")
    parser.add_argument("--gate-warmup-steps", type=int, default=200,
                        help="Anneal max_gate from 0→max_gate over N steps")
    parser.add_argument("--calibrate", action="store_true",
                        help="Pre-calibrate confidence projection from MLP statistics")
    args = parser.parse_args()

    confidence_aware = not args.no_confidence_aware
    output_base = "training_results/instruct_sweep"
    Path(output_base).mkdir(parents=True, exist_ok=True)

    to_run = INSTRUCT_MODELS
    if args.only:
        to_run = [m for m in INSTRUCT_MODELS if m["name"] in args.only]

    results = []
    for m in to_run:
        try:
            r = run_model(
                m, output_base, confidence_aware, args.max_gate,
                deterministic_alu=args.deterministic_alu,
                gate_warmup_steps=args.gate_warmup_steps,
                calibrate=args.calibrate,
            )
            results.append(r)
            if r["returncode"] != 0:
                print(f"\nWARNING: {m['name']} exited with code {r['returncode']}")
        except Exception as e:
            print(f"\nERROR running {m['name']}: {e}")
            results.append({"name": m["name"], "error": str(e)})

    # Summary table
    print(f"\n{'='*100}")
    print("INSTRUCT SWEEP SUMMARY")
    print(f"  Confidence-aware: {confidence_aware}")
    print(f"  Max gate: {args.max_gate}")
    print(f"{'='*100}")
    print(f"{'Model':<28} {'Params':>12} {'Trainable':>12} {'Baseline':>10} {'Final':>10} {'Delta':>8} {'Time':>8}")
    print("-" * 95)

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

        print(f"{name:<28} {fmt_params(total_p):>12} {fmt_params(train_p):>12} "
              f"{baseline:>9.1%} {final:>9.1%} {delta:>+7.1%} {elapsed:>7.0f}s")

    # Save summary
    summary_path = Path(output_base) / "instruct_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {summary_path}")


if __name__ == "__main__":
    main()
