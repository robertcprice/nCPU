#!/usr/bin/env python3
"""Collect fresh descriptor-update trajectories and train/evaluate executable thought."""

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

from ncpu.self_optimizing.controller_bundle import load_controller_bundle
from ncpu.self_optimizing.executable_thought_evaluation import evaluate_executable_thought_head
from ncpu.self_optimizing.executable_thought_head import (
    ExecutableThoughtHeadConfig,
    train_executable_thought_head,
)
from ncpu.self_optimizing.executable_thought_training import build_executable_thought_training_bundle
from ncpu.self_optimizing.run_qwen_benchmark import run_model_benchmark


@dataclass
class ExecutableThoughtProofResult:
    timestamp: str
    controller_bundle_path: str
    provider: str
    model: str
    benchmark_output_path: str
    trajectory_root: str
    training_output_dir: str
    executable_thought_checkpoint: str
    benchmark_summary: dict[str, Any]
    executable_thought_dataset: dict[str, Any]
    executable_thought_training_metrics: dict[str, Any]
    executable_thought_eval: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_executable_thought_proof(
    *,
    controller_bundle_path: str | Path,
    output_dir: str | Path,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    repeats: int = 4,
    max_retries: int = 3,
    include_coding: bool = True,
    include_reasoning: bool = True,
    base_url: Optional[str] = "http://localhost:11434",
    request_timeout: float = 240.0,
    temperature: float = 0.0,
    executable_thought_train_steps: int = 60,
    executable_thought_batch_size: int = 8,
    executable_thought_learning_rate: float = 1e-2,
    executable_thought_hidden_dim: int = 0,
    executable_thought_compiler_d_model: int = 64,
    executable_thought_compiler_max_program_len: int = 4,
    executable_thought_num_registers: int = 8,
    executable_thought_execution_max_steps: int = 4,
    executable_thought_output_register: int = 2,
    executable_thought_trace_projection_dim: int = 16,
    executable_thought_trace_hidden_dim: int = 64,
    executable_thought_state_patch_dim: int = 16,
    executable_thought_start_temperature: float = 1.0,
    executable_thought_end_temperature: float = 0.35,
    val_ratio: float = 0.2,
    device: str = "cpu",
) -> ExecutableThoughtProofResult:
    bundle = load_controller_bundle(controller_bundle_path)
    controller_path = str(Path(controller_bundle_path))
    resolved_model_for_encoding = model or bundle.base_model or bundle.response.model

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    trajectory_root = output_root / "trajectories"
    benchmark_output_path = output_root / "benchmark_report.json"
    training_output_dir = output_root / "executable_thought_training"
    evaluation_output_path = training_output_dir / "executable_thought_eval.json"
    checkpoint_path = training_output_dir / "executable_thought_head.pt"

    benchmark_report = run_model_benchmark(
        provider_name=provider_name,
        model=model,
        controller_bundle_path=controller_path,
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

    executable_bundle = build_executable_thought_training_bundle(
        trajectory_root,
        training_output_dir,
        require_verified_commit=True,
        num_registers=executable_thought_num_registers,
        output_dim=executable_thought_state_patch_dim,
        val_ratio=val_ratio,
    )

    training_metrics = train_executable_thought_head(
        output_path=checkpoint_path,
        config=ExecutableThoughtHeadConfig(
            hidden_dim=executable_thought_hidden_dim,
            compiler_d_model=executable_thought_compiler_d_model,
            compiler_max_program_len=executable_thought_compiler_max_program_len,
            num_registers=executable_thought_num_registers,
            execution_max_steps=executable_thought_execution_max_steps,
            output_register=executable_thought_output_register,
            trace_projection_dim=executable_thought_trace_projection_dim,
            trace_hidden_dim=executable_thought_trace_hidden_dim,
            state_patch_dim=executable_thought_state_patch_dim,
            temperature=executable_thought_start_temperature,
        ),
        model_name_or_path=resolved_model_for_encoding,
        steps=executable_thought_train_steps,
        batch_size=executable_thought_batch_size,
        learning_rate=executable_thought_learning_rate,
        start_temperature=executable_thought_start_temperature,
        end_temperature=executable_thought_end_temperature,
        device=device,
        train_path=executable_bundle["train_path"],
        val_path=executable_bundle["val_path"],
    )

    evaluation_report = evaluate_executable_thought_head(
        train_path=executable_bundle["train_path"],
        val_path=executable_bundle["val_path"],
        checkpoint_path=checkpoint_path,
        model_name_or_path=resolved_model_for_encoding,
        output_path=evaluation_output_path,
        device=device,
        temperature=executable_thought_end_temperature,
    )

    result = ExecutableThoughtProofResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        controller_bundle_path=controller_path,
        provider=str(benchmark_report["provider"]),
        model=str(benchmark_report["model"]),
        benchmark_output_path=str(benchmark_output_path),
        trajectory_root=str(trajectory_root),
        training_output_dir=str(training_output_dir),
        executable_thought_checkpoint=str(checkpoint_path),
        benchmark_summary={
            "baseline": benchmark_report["baseline"]["summary"],
            "some": benchmark_report["some"]["summary"],
            "delta": benchmark_report["delta"],
        },
        executable_thought_dataset=executable_bundle,
        executable_thought_training_metrics=training_metrics,
        executable_thought_eval=evaluation_report,
    )
    proof_path = output_root / "executable_thought_proof.json"
    proof_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect fresh descriptor-update trajectories and train/evaluate executable thought"
    )
    parser.add_argument("--controller-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--tasks", default="coding,reasoning")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--request-timeout", type=float, default=240.0)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--executable-thought-train-steps", type=int, default=60)
    parser.add_argument("--executable-thought-batch-size", type=int, default=8)
    parser.add_argument("--executable-thought-learning-rate", type=float, default=1e-2)
    parser.add_argument("--executable-thought-hidden-dim", type=int, default=0)
    parser.add_argument("--executable-thought-compiler-d-model", type=int, default=64)
    parser.add_argument("--executable-thought-compiler-max-program-len", type=int, default=4)
    parser.add_argument("--executable-thought-num-registers", type=int, default=8)
    parser.add_argument("--executable-thought-execution-max-steps", type=int, default=4)
    parser.add_argument("--executable-thought-output-register", type=int, default=2)
    parser.add_argument("--executable-thought-trace-projection-dim", type=int, default=16)
    parser.add_argument("--executable-thought-trace-hidden-dim", type=int, default=64)
    parser.add_argument("--executable-thought-state-patch-dim", type=int, default=16)
    parser.add_argument("--executable-thought-start-temperature", type=float, default=1.0)
    parser.add_argument("--executable-thought-end-temperature", type=float, default=0.35)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    requested = {item.strip().lower() for item in args.tasks.split(",") if item.strip()}
    include_coding = "coding" in requested
    include_reasoning = "reasoning" in requested
    if not include_coding and not include_reasoning:
        raise SystemExit("No valid tasks selected. Use coding, reasoning, or both.")

    result = run_executable_thought_proof(
        controller_bundle_path=args.controller_bundle,
        output_dir=args.output_dir,
        provider_name=args.provider,
        model=args.model,
        repeats=args.repeats,
        max_retries=args.max_retries,
        include_coding=include_coding,
        include_reasoning=include_reasoning,
        base_url=args.base_url,
        request_timeout=args.request_timeout,
        temperature=args.temp,
        executable_thought_train_steps=args.executable_thought_train_steps,
        executable_thought_batch_size=args.executable_thought_batch_size,
        executable_thought_learning_rate=args.executable_thought_learning_rate,
        executable_thought_hidden_dim=args.executable_thought_hidden_dim,
        executable_thought_compiler_d_model=args.executable_thought_compiler_d_model,
        executable_thought_compiler_max_program_len=args.executable_thought_compiler_max_program_len,
        executable_thought_num_registers=args.executable_thought_num_registers,
        executable_thought_execution_max_steps=args.executable_thought_execution_max_steps,
        executable_thought_output_register=args.executable_thought_output_register,
        executable_thought_trace_projection_dim=args.executable_thought_trace_projection_dim,
        executable_thought_trace_hidden_dim=args.executable_thought_trace_hidden_dim,
        executable_thought_state_patch_dim=args.executable_thought_state_patch_dim,
        executable_thought_start_temperature=args.executable_thought_start_temperature,
        executable_thought_end_temperature=args.executable_thought_end_temperature,
        val_ratio=args.val_ratio,
        device=args.device,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
