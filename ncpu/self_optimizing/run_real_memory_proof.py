#!/usr/bin/env python3
"""Collect fresh real-model SOME trajectories and evaluate the latent-memory path."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ncpu.self_optimizing.latent_memory_evaluation import evaluate_latent_memory_head
from ncpu.self_optimizing.latent_memory_head import LatentMemoryHeadConfig
from ncpu.self_optimizing.latent_memory_training import (
    build_latent_memory_training_bundle,
    train_latent_memory_head,
)
from ncpu.self_optimizing.run_qwen_benchmark import run_model_benchmark


@dataclass
class RealMemoryProofResult:
    timestamp: str
    provider: str
    model: str
    benchmark_output_path: str
    trajectory_root: str
    training_output_dir: str
    memory_head_checkpoint: str
    benchmark_summary: dict[str, Any]
    memory_dataset: dict[str, Any]
    memory_training_metrics: dict[str, Any]
    memory_eval: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_real_memory_proof(
    *,
    provider_name: str,
    model: str,
    output_dir: str | Path,
    repeats: int = 4,
    max_retries: int = 3,
    include_coding: bool = True,
    include_reasoning: bool = True,
    base_url: Optional[str] = "http://localhost:11434",
    request_timeout: float = 240.0,
    temperature: float = 0.0,
    latent_memory_epochs: int = 20,
    latent_memory_batch_size: int = 8,
    latent_memory_learning_rate: float = 1e-3,
    latent_memory_hidden_dim: int = 64,
    latent_memory_output_dim: int = 16,
) -> RealMemoryProofResult:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    trajectory_root = output_root / "trajectories"
    benchmark_output_path = output_root / "benchmark_report.json"
    training_output_dir = output_root / "latent_memory_training"
    evaluation_output_path = training_output_dir / "latent_memory_eval.json"
    checkpoint_path = training_output_dir / "latent_memory_head.pt"

    benchmark_report = run_model_benchmark(
        provider_name=provider_name,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        include_coding=include_coding,
        include_reasoning=include_reasoning,
        repeats=repeats,
        base_url=base_url,
        request_timeout=request_timeout,
        trajectory_dir=str(trajectory_root),
    )
    benchmark_output_path.write_text(
        json.dumps(benchmark_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    memory_bundle = build_latent_memory_training_bundle(
        trajectory_root,
        training_output_dir,
        require_verified_commit=False,
        val_ratio=0.2,
        config=LatentMemoryHeadConfig(
            hidden_dim=latent_memory_hidden_dim,
            output_dim=latent_memory_output_dim,
            dropout=0.0,
        ),
    )

    training_metrics = train_latent_memory_head(
        train_path=memory_bundle["train_path"],
        val_path=memory_bundle["val_path"],
        output_path=checkpoint_path,
        config=LatentMemoryHeadConfig(
            hidden_dim=latent_memory_hidden_dim,
            output_dim=latent_memory_output_dim,
            dropout=0.0,
        ),
        epochs=latent_memory_epochs,
        batch_size=latent_memory_batch_size,
        learning_rate=latent_memory_learning_rate,
        device="cpu",
    )

    evaluation_report = evaluate_latent_memory_head(
        train_path=memory_bundle["train_path"],
        val_path=memory_bundle["val_path"],
        checkpoint_path=checkpoint_path,
        output_path=evaluation_output_path,
        device="cpu",
    )

    result = RealMemoryProofResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        provider=provider_name,
        model=model,
        benchmark_output_path=str(benchmark_output_path),
        trajectory_root=str(trajectory_root),
        training_output_dir=str(training_output_dir),
        memory_head_checkpoint=str(checkpoint_path),
        benchmark_summary={
            "baseline": benchmark_report["baseline"]["summary"],
            "some": benchmark_report["some"]["summary"],
            "delta": benchmark_report["delta"],
        },
        memory_dataset=memory_bundle,
        memory_training_metrics=training_metrics,
        memory_eval=evaluation_report,
    )
    proof_path = output_root / "real_memory_proof.json"
    proof_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real Qwen trajectory collection + latent-memory proof")
    parser.add_argument("--provider", default="local")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--tasks", default="coding,reasoning")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--request-timeout", type=float, default=240.0)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--latent-memory-epochs", type=int, default=20)
    parser.add_argument("--latent-memory-batch-size", type=int, default=8)
    parser.add_argument("--latent-memory-learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-memory-hidden-dim", type=int, default=64)
    parser.add_argument("--latent-memory-output-dim", type=int, default=16)
    args = parser.parse_args()

    requested = {item.strip().lower() for item in args.tasks.split(",") if item.strip()}
    include_coding = "coding" in requested
    include_reasoning = "reasoning" in requested
    if not include_coding and not include_reasoning:
        raise SystemExit("No valid tasks selected. Use coding, reasoning, or both.")

    result = run_real_memory_proof(
        provider_name=args.provider,
        model=args.model,
        output_dir=args.output_dir,
        repeats=args.repeats,
        max_retries=args.max_retries,
        include_coding=include_coding,
        include_reasoning=include_reasoning,
        base_url=args.base_url,
        request_timeout=args.request_timeout,
        temperature=args.temp,
        latent_memory_epochs=args.latent_memory_epochs,
        latent_memory_batch_size=args.latent_memory_batch_size,
        latent_memory_learning_rate=args.latent_memory_learning_rate,
        latent_memory_hidden_dim=args.latent_memory_hidden_dim,
        latent_memory_output_dim=args.latent_memory_output_dim,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
