"""Ablation runner and benchmark suite for nCPU execution training.

Runs a proper A/B comparison between:
  - Baseline: pre-trained model, no training
  - Copro-only: trained with coprocessor but exec_loss_weight=0
  - Exec-trained: trained with full execution loss (exec_loss_weight=1.0)

Each condition is evaluated using model.generate() → parse → execute,
so we measure the model's actual code generation quality, not just
reference code pipeline correctness.

Usage:
    # Quick smoke test (no real LM)
    python -m ncpu.execution_training.run_ablation --synthetic-only --steps 100

    # Full ablation with a model
    python -m ncpu.execution_training.run_ablation \
        --model Qwen/Qwen3.5-0.8B --steps 1000 --eval-samples 100

    # Compare two model sizes
    python -m ncpu.execution_training.run_ablation \
        --model Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B \
        --steps 500 --output-dir ablation_results/
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ncpu.differentiable.execution import DifferentiableEngine

from .code_parser import CodeToISAParser, ParseError
from .data import ExecutionTrainingDataset, ExecutionTrainingSample
from .evaluate import EvaluationResult, ExecutionEvaluator
from .execution_loss import ExecutionLoss, ExecutionLossWithParsing

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class AblationCondition:
    """One experimental condition in the ablation."""
    name: str
    exec_loss_weight: float
    description: str
    eval_result: Optional[EvaluationResult] = None
    train_losses: List[float] = field(default_factory=list)
    exec_losses: List[float] = field(default_factory=list)
    wall_time: float = 0.0
    steps_completed: int = 0


@dataclass
class AblationModelResult:
    """Results for one model across all conditions."""
    model_name: str
    baseline: Optional[AblationCondition] = None
    conditions: List[AblationCondition] = field(default_factory=list)


@dataclass
class AblationReport:
    """Full ablation report across models."""
    model_results: List[AblationModelResult] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    total_wall_time: float = 0.0


# ── Core ablation logic ─────────────────────────────────────────────────────

def run_ablation(
    model_names: List[str],
    steps: int = 1000,
    eval_samples: int = 100,
    data_size: int = 2000,
    batch_size: int = 4,
    lr: float = 1e-3,
    max_value: int = 100,
    max_loop_n: int = 10,
    max_length: int = 256,
    output_dir: str = "ablation_results",
    layers: Optional[List[int]] = None,
    eval_every: int = 200,
    max_new_tokens: int = 128,
    gen_temperature: float = 0.1,
    conditions: Optional[List[Dict]] = None,
    models_dir: Optional[str] = None,
    synthetic_only: bool = False,
) -> AblationReport:
    """Run full ablation study.

    Args:
        model_names: List of HuggingFace model names to test
        steps: Training steps per condition
        eval_samples: Number of samples for evaluation
        data_size: Training dataset size
        batch_size: Training batch size
        lr: Learning rate
        max_value: Max integer value in generated problems
        max_loop_n: Max loop iterations
        max_length: Max token sequence length
        output_dir: Where to save results
        layers: Which transformer layers to inject coprocessor
        eval_every: Evaluate during training every N steps
        max_new_tokens: Max tokens for model generation during eval
        gen_temperature: Temperature for generation
        conditions: Custom conditions; default is copro-only + exec-trained
        models_dir: Path to local model weights
        synthetic_only: Skip real LM, test pipeline only

    Returns:
        AblationReport with all results
    """
    start_time = time.time()
    device = _get_device()
    logger.info(f"Ablation device: {device}")

    if layers is None:
        layers = [-1]

    # Default conditions: copro-only vs exec-trained
    if conditions is None:
        conditions = [
            {"name": "copro_only", "exec_loss_weight": 0.0,
             "description": "Coprocessor only, no execution gradient"},
            {"name": "exec_trained", "exec_loss_weight": 1.0,
             "description": "Full execution gradient (alpha=1.0)"},
        ]

    # Build eval dataset (shared across all conditions)
    logger.info(f"Building evaluation dataset ({eval_samples} samples)...")
    eval_dataset = ExecutionTrainingDataset(
        size=eval_samples,
        seed=999,
        max_value=max_value,
        max_loop_n=max_loop_n,
    )
    eval_samples_list = eval_dataset.samples[:eval_samples]

    if synthetic_only:
        report = _run_synthetic_ablation(
            conditions=conditions,
            eval_samples=eval_samples_list,
            data_size=data_size,
            steps=steps,
            max_value=max_value,
            max_loop_n=max_loop_n,
            device=device,
        )
        report.config = _build_config_dict(locals())
        report.total_wall_time = time.time() - start_time
        _save_report(report, output_dir)
        return report

    report = AblationReport()
    report.config = _build_config_dict(locals())

    for model_name in model_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"ABLATION: {model_name}")
        logger.info(f"{'='*60}")

        model_result = _run_model_ablation(
            model_name=model_name,
            conditions=conditions,
            eval_samples=eval_samples_list,
            data_size=data_size,
            batch_size=batch_size,
            lr=lr,
            steps=steps,
            max_value=max_value,
            max_loop_n=max_loop_n,
            max_length=max_length,
            layers=layers,
            eval_every=eval_every,
            max_new_tokens=max_new_tokens,
            gen_temperature=gen_temperature,
            models_dir=models_dir,
            device=device,
        )
        report.model_results.append(model_result)

    report.total_wall_time = time.time() - start_time
    _save_report(report, output_dir)
    return report


def _run_model_ablation(
    model_name: str,
    conditions: List[Dict],
    eval_samples: List[ExecutionTrainingSample],
    data_size: int,
    batch_size: int,
    lr: float,
    steps: int,
    max_value: int,
    max_loop_n: int,
    max_length: int,
    layers: List[int],
    eval_every: int,
    max_new_tokens: int,
    gen_temperature: float,
    models_dir: Optional[str],
    device: str,
) -> AblationModelResult:
    """Run ablation for a single model across all conditions."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ncpu.coprocessor.config import NCPUCoprocessorConfig
    from ncpu.coprocessor.inject import (
        collect_aux_losses,
        freeze_backbone,
        get_coprocessor_params,
        inject_ncpu_coprocessor,
    )

    result = AblationModelResult(model_name=model_name)

    # ── Load base model + tokenizer ──
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_state = None  # Will store initial state for reloading

    def _load_fresh_model():
        """Load a fresh copy of the model with coprocessor injected."""
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
            trust_remote_code=True,
        ).to(device)

        copro_config = NCPUCoprocessorConfig(
            layer_indices=layers,
            models_dir=models_dir,
            freeze_backbone=True,
            max_gate=0.1,
        )
        inject_ncpu_coprocessor(model, copro_config)
        freeze_backbone(model, unfreeze_last_n=0, freeze_alu=True)
        return model

    # ── 1. Baseline eval (before any training) ──
    logger.info("Phase 1: Baseline evaluation (pre-training)...")
    model = _load_fresh_model()
    model.eval()

    engine = DifferentiableEngine(device=device)
    evaluator = ExecutionEvaluator(engine=engine, device=device)

    baseline_start = time.time()
    baseline_eval = evaluator.evaluate(
        model, tokenizer, eval_samples,
        max_new_tokens=max_new_tokens,
        temperature=gen_temperature,
    )
    baseline_time = time.time() - baseline_start

    baseline_cond = AblationCondition(
        name="baseline",
        exec_loss_weight=0.0,
        description="Pre-trained model, no fine-tuning",
        eval_result=baseline_eval,
        wall_time=baseline_time,
    )
    result.baseline = baseline_cond
    logger.info(f"Baseline: accuracy={baseline_eval.accuracy:.1%} "
                f"parse={baseline_eval.parse_rate:.1%} "
                f"exec_acc={baseline_eval.exec_accuracy:.1%}")

    del model
    _clear_memory(device)

    # ── 2. Train + evaluate each condition ──
    for cond_spec in conditions:
        cond_name = cond_spec["name"]
        exec_weight = cond_spec["exec_loss_weight"]
        cond_desc = cond_spec.get("description", f"exec_loss_weight={exec_weight}")

        logger.info(f"\nPhase 2: Training condition '{cond_name}' "
                    f"(exec_loss_weight={exec_weight})...")

        cond = AblationCondition(
            name=cond_name,
            exec_loss_weight=exec_weight,
            description=cond_desc,
        )

        # Load fresh model
        model = _load_fresh_model()
        model.train()

        # Build training data
        train_dataset = ExecutionTrainingDataset(
            size=data_size,
            seed=42,
            max_value=max_value,
            max_loop_n=max_loop_n,
            tokenizer=tokenizer,
            max_length=max_length,
        )

        # Training components
        engine = DifferentiableEngine(device=device)
        exec_loss_fn = ExecutionLossWithParsing(
            execution_loss=ExecutionLoss(
                engine=engine,
                correctness_tolerance=0.5,
                max_exec_steps=64,
                device=device,
            ),
            use_soft_programs=True,
            temperature=1.0,
            device=device,
        )

        # Optimizer (coprocessor params only)
        copro_params = get_coprocessor_params(model)
        optimizer = torch.optim.AdamW(
            [{"params": copro_params, "lr": lr}],
            weight_decay=0.01,
        )

        # Training loop
        from torch.utils.data import DataLoader

        def _collate_fn(batch):
            tensors = {}
            samples = []
            for item in batch:
                for k, v in item.items():
                    if k == "sample":
                        samples.append(v)
                    else:
                        tensors.setdefault(k, []).append(v)
            result = {k: torch.stack(v) for k, v in tensors.items()}
            result["sample"] = samples
            return result

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            collate_fn=_collate_fn,
        )

        train_start = time.time()
        data_iter = iter(train_loader)
        step = 0

        while step < steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            batch_samples = batch["sample"]

            # Forward: LM loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            lm_loss = outputs.loss

            # Aux loss
            aux_loss = collect_aux_losses(model)

            # Execution loss
            exec_loss = torch.tensor(0.0, device=device)
            n_exec = 0

            if exec_weight > 0:
                for sample in batch_samples:
                    if isinstance(sample, ExecutionTrainingSample):
                        code = sample.reference_code
                        test_cases = sample.test_cases
                        arg_names = sample.arg_names
                        output_var = sample.output_var
                        is_func = sample.is_function
                    elif isinstance(sample, dict):
                        code = sample.get("reference_code", "")
                        test_cases = sample.get("test_cases", [])
                        arg_names = sample.get("arg_names", [])
                        output_var = sample.get("output_var", None)
                        is_func = sample.get("is_function", False)
                    else:
                        continue

                    try:
                        elr = exec_loss_fn(
                            code=code,
                            test_cases=test_cases,
                            arg_names=arg_names if arg_names else None,
                            output_var=output_var,
                            is_function=is_func,
                        )
                        if elr.total_loss.requires_grad:
                            exec_loss = exec_loss + elr.total_loss
                            n_exec += 1
                    except Exception:
                        pass

                if n_exec > 0:
                    exec_loss = exec_loss / n_exec

            # Combined loss
            total_loss = lm_loss + aux_loss
            if n_exec > 0 and exec_loss.requires_grad:
                total_loss = total_loss + exec_weight * exec_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            cond.train_losses.append(total_loss.item())
            cond.exec_losses.append(
                exec_loss.item() if isinstance(exec_loss, torch.Tensor) else 0.0
            )

            if step % max(steps // 10, 1) == 0:
                logger.info(
                    f"  [{cond_name}] Step {step}/{steps} | "
                    f"loss={total_loss.item():.4f} "
                    f"exec={exec_loss.item() if isinstance(exec_loss, torch.Tensor) else 0:.4f}"
                )

            # Mid-training evaluation
            if eval_every > 0 and step % eval_every == 0 and step < steps:
                model.eval()
                mid_eval = evaluator.evaluate(
                    model, tokenizer, eval_samples[:20],
                    max_new_tokens=max_new_tokens,
                    temperature=gen_temperature,
                )
                model.train()
                logger.info(
                    f"  [{cond_name}] Mid-eval step {step}: "
                    f"acc={mid_eval.accuracy:.1%} parse={mid_eval.parse_rate:.1%}"
                )

        cond.steps_completed = step
        cond.wall_time = time.time() - train_start

        # Final evaluation
        logger.info(f"  [{cond_name}] Final evaluation...")
        model.eval()
        cond.eval_result = evaluator.evaluate(
            model, tokenizer, eval_samples,
            max_new_tokens=max_new_tokens,
            temperature=gen_temperature,
        )
        logger.info(
            f"  [{cond_name}] Final: accuracy={cond.eval_result.accuracy:.1%} "
            f"parse={cond.eval_result.parse_rate:.1%} "
            f"exec_acc={cond.eval_result.exec_accuracy:.1%}"
        )

        result.conditions.append(cond)

        del model
        _clear_memory(device)

    return result


def _run_synthetic_ablation(
    conditions: List[Dict],
    eval_samples: List[ExecutionTrainingSample],
    data_size: int,
    steps: int,
    max_value: int,
    max_loop_n: int,
    device: str,
) -> AblationReport:
    """Synthetic ablation: tests pipeline without a real LM.

    Evaluates reference code through the execution pipeline to verify
    that parsing, execution, and evaluation all work correctly.
    """
    logger.info("Running synthetic ablation (no LM)...")

    engine = DifferentiableEngine(device=device)
    evaluator = ExecutionEvaluator(engine=engine, device=device)

    # Reference-only eval (no model generation)
    ref_eval = evaluator.evaluate_reference_only(eval_samples)

    report = AblationReport()
    model_result = AblationModelResult(model_name="synthetic_pipeline")

    baseline = AblationCondition(
        name="reference_code",
        exec_loss_weight=0.0,
        description="Reference code pipeline validation",
        eval_result=ref_eval,
    )
    model_result.baseline = baseline

    # For synthetic, simulate conditions with reference eval
    for cond_spec in conditions:
        cond = AblationCondition(
            name=cond_spec["name"],
            exec_loss_weight=cond_spec["exec_loss_weight"],
            description=cond_spec.get("description", ""),
            eval_result=ref_eval,  # Same as baseline for synthetic
        )
        model_result.conditions.append(cond)

    report.model_results.append(model_result)

    logger.info(f"Synthetic ablation: parse_rate={ref_eval.parse_rate:.1%} "
                f"exec_acc={ref_eval.exec_accuracy:.1%}")
    return report


# ── Reporting ────────────────────────────────────────────────────────────────

def format_summary_table(report: AblationReport) -> str:
    """Generate a readable ASCII summary table."""
    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("ABLATION STUDY RESULTS")
    lines.append("=" * 90)
    lines.append(f"Total wall time: {report.total_wall_time:.1f}s")
    lines.append("")

    for mr in report.model_results:
        lines.append(f"Model: {mr.model_name}")
        lines.append("-" * 90)

        # Header
        header = (
            f"{'Condition':<20s} | {'Accuracy':>10s} | {'Parse Rate':>10s} | "
            f"{'Exec Acc':>10s} | {'Exec Loss':>10s} | {'Time':>8s} | {'Steps':>6s}"
        )
        lines.append(header)
        lines.append("-" * 90)

        # Baseline row
        if mr.baseline and mr.baseline.eval_result:
            er = mr.baseline.eval_result
            lines.append(
                f"{'baseline':<20s} | {er.accuracy:>10.1%} | {er.parse_rate:>10.1%} | "
                f"{er.exec_accuracy:>10.1%} | {er.mean_execution_loss:>10.4f} | "
                f"{mr.baseline.wall_time:>7.1f}s | {'N/A':>6s}"
            )

        # Condition rows
        for cond in mr.conditions:
            if cond.eval_result:
                er = cond.eval_result
                lines.append(
                    f"{cond.name:<20s} | {er.accuracy:>10.1%} | {er.parse_rate:>10.1%} | "
                    f"{er.exec_accuracy:>10.1%} | {er.mean_execution_loss:>10.4f} | "
                    f"{cond.wall_time:>7.1f}s | {cond.steps_completed:>6d}"
                )

        lines.append("")

        # Deltas vs baseline
        if mr.baseline and mr.baseline.eval_result:
            base_acc = mr.baseline.eval_result.accuracy
            base_parse = mr.baseline.eval_result.parse_rate
            base_exec = mr.baseline.eval_result.exec_accuracy

            lines.append("Deltas vs baseline:")
            for cond in mr.conditions:
                if cond.eval_result:
                    er = cond.eval_result
                    d_acc = er.accuracy - base_acc
                    d_parse = er.parse_rate - base_parse
                    d_exec = er.exec_accuracy - base_exec
                    sign = lambda x: "+" if x >= 0 else ""
                    lines.append(
                        f"  {cond.name:<18s}: "
                        f"accuracy {sign(d_acc)}{d_acc:.1%}, "
                        f"parse {sign(d_parse)}{d_parse:.1%}, "
                        f"exec_acc {sign(d_exec)}{d_exec:.1%}"
                    )

        # Deltas between conditions (if 2+ conditions)
        if len(mr.conditions) >= 2:
            c0 = mr.conditions[0]
            c1 = mr.conditions[1]
            if c0.eval_result and c1.eval_result:
                lines.append("")
                lines.append(f"Head-to-head: {c1.name} vs {c0.name}:")
                e0 = c0.eval_result
                e1 = c1.eval_result
                d_acc = e1.accuracy - e0.accuracy
                d_parse = e1.parse_rate - e0.parse_rate
                d_exec = e1.exec_accuracy - e0.exec_accuracy
                sign = lambda x: "+" if x >= 0 else ""
                lines.append(
                    f"  accuracy {sign(d_acc)}{d_acc:.1%}, "
                    f"parse {sign(d_parse)}{d_parse:.1%}, "
                    f"exec_acc {sign(d_exec)}{d_exec:.1%}"
                )

        lines.append("")

    lines.append("=" * 90)
    return "\n".join(lines)


def _eval_result_to_dict(er: Optional[EvaluationResult]) -> dict:
    """Convert EvaluationResult to JSON-serializable dict."""
    if er is None:
        return {}
    return {
        "total_samples": er.total_samples,
        "total_correct": er.total_correct,
        "total_parseable": er.total_parseable,
        "total_executable": er.total_executable,
        "total_exec_correct": er.total_exec_correct,
        "accuracy": er.accuracy,
        "parse_rate": er.parse_rate,
        "exec_accuracy": er.exec_accuracy,
        "mean_execution_loss": er.mean_execution_loss,
        "mean_output_loss": er.mean_output_loss,
        "eval_time_seconds": er.eval_time_seconds,
        "category_results": er.category_results,
        "difficulty_results": er.difficulty_results,
    }


def _save_report(report: AblationReport, output_dir: str) -> None:
    """Save ablation report as JSON and text summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON report
    json_data = {
        "config": report.config,
        "total_wall_time": report.total_wall_time,
        "models": [],
    }

    for mr in report.model_results:
        model_data = {
            "model_name": mr.model_name,
            "baseline": {
                "name": mr.baseline.name if mr.baseline else "N/A",
                "eval": _eval_result_to_dict(mr.baseline.eval_result if mr.baseline else None),
                "wall_time": mr.baseline.wall_time if mr.baseline else 0,
            },
            "conditions": [],
        }
        for cond in mr.conditions:
            cond_data = {
                "name": cond.name,
                "exec_loss_weight": cond.exec_loss_weight,
                "description": cond.description,
                "eval": _eval_result_to_dict(cond.eval_result),
                "steps_completed": cond.steps_completed,
                "wall_time": cond.wall_time,
                "final_train_loss": cond.train_losses[-1] if cond.train_losses else None,
                "final_exec_loss": cond.exec_losses[-1] if cond.exec_losses else None,
            }
            model_data["conditions"].append(cond_data)
        json_data["models"].append(model_data)

    json_path = out / "ablation_report.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    logger.info(f"JSON report saved to {json_path}")

    # Text summary
    summary = format_summary_table(report)
    txt_path = out / "ablation_summary.txt"
    with open(txt_path, "w") as f:
        f.write(summary)
    logger.info(f"Text summary saved to {txt_path}")

    # Print to stdout
    print(summary)


# ── Utilities ────────────────────────────────────────────────────────────────

def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _clear_memory(device: str) -> None:
    """Free GPU/MPS memory between runs."""
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def _build_config_dict(local_vars: dict) -> dict:
    """Build a JSON-safe config dict from function locals."""
    safe_keys = [
        "model_names", "steps", "eval_samples", "data_size", "batch_size",
        "lr", "max_value", "max_loop_n", "max_length", "output_dir",
        "layers", "eval_every", "max_new_tokens", "gen_temperature",
        "conditions", "synthetic_only",
    ]
    return {k: local_vars[k] for k in safe_keys if k in local_vars}


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study: baseline vs copro-only vs exec-trained"
    )
    parser.add_argument(
        "--model", nargs="+", default=["Qwen/Qwen3.5-0.8B"],
        help="HuggingFace model name(s) to test",
    )
    parser.add_argument("--steps", type=int, default=1000,
                        help="Training steps per condition")
    parser.add_argument("--eval-samples", type=int, default=100,
                        help="Number of samples for evaluation")
    parser.add_argument("--data-size", type=int, default=2000,
                        help="Training dataset size")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--max-value", type=int, default=100)
    parser.add_argument("--max-loop-n", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output-dir", default="ablation_results",
                        help="Output directory for reports")
    parser.add_argument("--layers", nargs="+", type=int, default=[-1],
                        help="Transformer layers for coprocessor injection")
    parser.add_argument("--eval-every", type=int, default=200,
                        help="Mid-training evaluation interval (0=disabled)")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Max tokens for model generation")
    parser.add_argument("--gen-temperature", type=float, default=0.1,
                        help="Generation temperature")
    parser.add_argument("--models-dir", default=None,
                        help="Local path for model weights")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Test pipeline without real LM")

    # Extra conditions
    parser.add_argument("--exec-weights", nargs="+", type=float, default=None,
                        help="Custom exec_loss_weight values to test "
                             "(default: 0.0 and 1.0)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Build conditions
    conditions = None
    if args.exec_weights:
        conditions = []
        for w in args.exec_weights:
            name = f"exec_w{w:.1f}".replace(".", "p")
            conditions.append({
                "name": name,
                "exec_loss_weight": w,
                "description": f"exec_loss_weight={w}",
            })

    report = run_ablation(
        model_names=args.model,
        steps=args.steps,
        eval_samples=args.eval_samples,
        data_size=args.data_size,
        batch_size=args.batch_size,
        lr=args.lr,
        max_value=args.max_value,
        max_loop_n=args.max_loop_n,
        max_length=args.max_length,
        output_dir=args.output_dir,
        layers=args.layers,
        eval_every=args.eval_every,
        max_new_tokens=args.max_new_tokens,
        gen_temperature=args.gen_temperature,
        conditions=conditions,
        models_dir=args.models_dir,
        synthetic_only=args.synthetic_only,
    )


if __name__ == "__main__":
    main()
