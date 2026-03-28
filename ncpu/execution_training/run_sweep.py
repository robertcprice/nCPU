"""Scaling sweep for differentiable execution training across model families.

Runs the full execution-grounded training pipeline across multiple model
sizes to measure how the execution training signal scales. Produces a
JSON report and a summary table.

Usage:
    # Quick sweep (small models, fewer steps)
    python -m ncpu.execution_training.run_sweep --quick

    # Full sweep
    python -m ncpu.execution_training.run_sweep --output-dir sweep_results/

    # Single model
    python -m ncpu.execution_training.run_sweep --models Qwen/Qwen3.5-0.8B --steps 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .train import ExecutionTrainingConfig, train_execution_grounded

logger = logging.getLogger(__name__)


# ── Model configurations ──

QUICK_MODELS = [
    "Qwen/Qwen3.5-0.8B",
]

STANDARD_MODELS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
]

FULL_MODELS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
]

# Baseline (no execution loss) vs execution-trained
ABLATION_CONFIGS = {
    "baseline": {"exec_loss_weight": 0.0, "aux_loss_weight": 1.0},
    "exec_only": {"exec_loss_weight": 1.0, "aux_loss_weight": 0.0},
    "exec_plus_copro": {"exec_loss_weight": 1.0, "aux_loss_weight": 1.0},
    "exec_heavy": {"exec_loss_weight": 5.0, "aux_loss_weight": 1.0},
    "exec_with_trace": {
        "exec_loss_weight": 1.0,
        "aux_loss_weight": 1.0,
        "trace_loss_weight": 0.5,
    },
    "deterministic_alu": {
        "exec_loss_weight": 1.0,
        "aux_loss_weight": 1.0,
        "deterministic_alu": True,
    },
}


@dataclass
class SweepConfig:
    """Configuration for a scaling sweep."""

    models: list[str] = field(default_factory=lambda: STANDARD_MODELS)
    ablations: list[str] = field(
        default_factory=lambda: ["baseline", "exec_plus_copro"]
    )
    steps: int = 2000
    data_size: int = 5000
    batch_size: int = 8
    eval_every: int = 500
    layers: list[int] = field(default_factory=lambda: [-1])
    output_dir: str = "sweep_results/exec_training"
    max_value: int = 100
    # Per-model overrides
    model_batch_sizes: dict = field(default_factory=dict)


@dataclass
class SweepResult:
    """Results from one model+ablation combination."""

    model: str
    ablation: str
    config: dict
    final_loss: float
    final_exec_loss: float
    parse_success_rate: float
    exec_accuracy: float
    eval_accuracy: float = 0.0
    eval_parse_rate: float = 0.0
    eval_exec_accuracy: float = 0.0
    trainable_params: int = 0
    wall_time_seconds: float = 0.0
    error: Optional[str] = None


def run_sweep(sweep_config: SweepConfig) -> list[SweepResult]:
    """Run a full scaling sweep."""
    output_dir = Path(sweep_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_runs = len(sweep_config.models) * len(sweep_config.ablations)
    run_idx = 0

    for model_name in sweep_config.models:
        for ablation_name in sweep_config.ablations:
            run_idx += 1
            logger.info(
                f"\n{'=' * 60}\n"
                f"Run {run_idx}/{total_runs}: {model_name} / {ablation_name}\n"
                f"{'=' * 60}"
            )

            ablation_overrides = ABLATION_CONFIGS.get(ablation_name, {})
            batch_size = sweep_config.model_batch_sizes.get(
                model_name, sweep_config.batch_size
            )

            # Build training config
            train_config = ExecutionTrainingConfig(
                model_name=model_name,
                layers=sweep_config.layers,
                steps=sweep_config.steps,
                data_size=sweep_config.data_size,
                batch_size=batch_size,
                max_value=sweep_config.max_value,
                eval_every=sweep_config.eval_every,
                output_dir=str(
                    output_dir / f"{_safe_name(model_name)}_{ablation_name}"
                ),
                # Defaults
                lr=1e-3,
                warmup_steps=100,
                freeze_backbone=True,
                confidence_aware=True,
                max_gate=0.1,
                gate_warmup_steps=200,
                n_bits=8,
            )

            # Apply ablation overrides
            for key, value in ablation_overrides.items():
                if hasattr(train_config, key):
                    setattr(train_config, key, value)

            # Run training
            try:
                train_result = train_execution_grounded(train_config)

                sweep_result = SweepResult(
                    model=model_name,
                    ablation=ablation_name,
                    config=ablation_overrides,
                    final_loss=train_result.final_loss,
                    final_exec_loss=train_result.final_exec_loss,
                    parse_success_rate=train_result.parse_success_rate,
                    exec_accuracy=train_result.exec_accuracy,
                    eval_accuracy=train_result.eval_result.get("accuracy", 0)
                    if train_result.eval_result
                    else 0,
                    eval_parse_rate=train_result.eval_result.get("parse_rate", 0)
                    if train_result.eval_result
                    else 0,
                    eval_exec_accuracy=train_result.eval_result.get(
                        "exec_accuracy", 0
                    )
                    if train_result.eval_result
                    else 0,
                    trainable_params=train_result.trainable_params,
                    wall_time_seconds=train_result.wall_time_seconds,
                )

            except Exception as e:
                logger.error(f"Run failed: {e}")
                sweep_result = SweepResult(
                    model=model_name,
                    ablation=ablation_name,
                    config=ablation_overrides,
                    final_loss=float("nan"),
                    final_exec_loss=float("nan"),
                    parse_success_rate=0.0,
                    exec_accuracy=0.0,
                    error=str(e),
                )

            results.append(sweep_result)

            # Save incremental results
            _save_results(results, output_dir)

    # Final report
    _print_summary(results)
    _save_results(results, output_dir)

    return results


def _safe_name(model_name: str) -> str:
    """Convert model name to filesystem-safe string."""
    return model_name.replace("/", "_").replace(".", "_")


def _save_results(results: list[SweepResult], output_dir: Path):
    """Save sweep results to JSON."""
    data = []
    for r in results:
        d = {
            "model": r.model,
            "ablation": r.ablation,
            "config": r.config,
            "final_loss": r.final_loss,
            "final_exec_loss": r.final_exec_loss,
            "parse_success_rate": r.parse_success_rate,
            "exec_accuracy": r.exec_accuracy,
            "eval_accuracy": r.eval_accuracy,
            "eval_parse_rate": r.eval_parse_rate,
            "eval_exec_accuracy": r.eval_exec_accuracy,
            "trainable_params": r.trainable_params,
            "wall_time_seconds": r.wall_time_seconds,
            "error": r.error,
        }
        data.append(d)

    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump(data, f, indent=2, default=str)


def _print_summary(results: list[SweepResult]):
    """Print a summary table."""
    print(f"\n{'=' * 90}")
    print("EXECUTION TRAINING SCALING SWEEP RESULTS")
    print(f"{'=' * 90}")
    print(
        f"{'Model':<30} {'Ablation':<20} {'ExecLoss':>10} "
        f"{'ParseRate':>10} {'ExecAcc':>10} {'Time':>8}"
    )
    print("-" * 90)

    for r in results:
        if r.error:
            print(f"{r.model:<30} {r.ablation:<20} {'ERROR':>10} {r.error}")
        else:
            print(
                f"{r.model:<30} {r.ablation:<20} "
                f"{r.final_exec_loss:>10.2f} "
                f"{r.parse_success_rate:>9.1%} "
                f"{r.exec_accuracy:>9.1%} "
                f"{r.wall_time_seconds:>7.0f}s"
            )

    print(f"{'=' * 90}")


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(
        description="Scaling sweep for execution-grounded training"
    )

    parser.add_argument(
        "--models", nargs="+", default=None, help="Model names to sweep"
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=["baseline", "exec_plus_copro"],
        choices=list(ABLATION_CONFIGS.keys()),
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--data-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--layers", nargs="+", type=int, default=[-1])
    parser.add_argument(
        "--output-dir", default="sweep_results/exec_training"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick sweep: single small model, 500 steps",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full sweep: all models and ablations",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.quick:
        models = QUICK_MODELS
        steps = 500
        data_size = 1000
        ablations = ["baseline", "exec_plus_copro"]
    elif args.full:
        models = FULL_MODELS
        steps = args.steps
        data_size = args.data_size
        ablations = list(ABLATION_CONFIGS.keys())
    else:
        models = args.models or STANDARD_MODELS
        steps = args.steps
        data_size = args.data_size
        ablations = args.ablations

    sweep_config = SweepConfig(
        models=models,
        ablations=ablations,
        steps=steps,
        data_size=data_size,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        layers=args.layers,
        output_dir=args.output_dir,
    )

    run_sweep(sweep_config)


if __name__ == "__main__":
    main()
