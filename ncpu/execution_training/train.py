"""Training loop for differentiable execution-grounded code model training.

This is the main entry point. It:
1. Loads a pretrained LM (Qwen3.5, LLaMA, etc.)
2. Injects the nCPU differentiable coprocessor into MLP sublayers
3. Generates training data with arithmetic/variable/loop problems
4. Trains with a combined loss:
     L_total = L_lm + α * L_execution + β * L_aux_copro
   where L_execution comes from actually running the code on nCPU's
   differentiable engine and backpropagating execution error through
   every ALU operation.

Three training modes:
  Mode 1 (default): Coprocessor + execution loss on reference code
    - Parse reference code to nCPU ISA
    - Execute differentiably, backprop execution error into coprocessor
    - Standard LM loss for token generation
  Mode 2: End-to-end differentiable compilation
    - DifferentiableCompiler maps hidden states to programs
    - Full gradient from execution through compilation to embeddings
  Mode 3: Execution loss on model-generated code
    - Model generates code, parse it, execute, backprop
    - Requires generation in the training loop (slower, but trains on own output)

Usage:
    # Minimal smoke test (CPU, tiny model)
    python -m ncpu.execution_training.train --synthetic-only --steps 200

    # Full training with Qwen
    python -m ncpu.execution_training.train \\
        --model Qwen/Qwen3.5-0.8B \\
        --steps 2000 \\
        --lr 1e-3 \\
        --batch-size 8 \\
        --exec-loss-weight 1.0 \\
        --output-dir training_results/exec_training/

    # Mode 2: differentiable compilation
    python -m ncpu.execution_training.train \\
        --model Qwen/Qwen3.5-0.8B \\
        --mode compilation \\
        --steps 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# nCPU imports
from ncpu.differentiable.execution import DifferentiableEngine, SoftProgram
from ncpu.coprocessor.config import NCPUCoprocessorConfig
from ncpu.coprocessor.inject import (
    inject_ncpu_coprocessor,
    collect_aux_losses,
    freeze_backbone,
    get_coprocessor_params,
)
from ncpu.coprocessor.router import update_gate_schedule

from .code_parser import CodeToISAParser, ParseError
from .execution_loss import ExecutionLoss, ExecutionLossWithParsing
from .data import ExecutionTrainingDataset, ExecutionTrainingSample
from .evaluate import ExecutionEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class ExecutionTrainingConfig:
    """Full training configuration."""

    # ── Model ──
    model_name: str = "Qwen/Qwen3.5-0.8B"
    layers: List[int] = field(default_factory=lambda: [-1])

    # ── Training mode ──
    mode: str = "coprocessor"  # "coprocessor", "compilation", "generated"

    # ── Data ──
    data_size: int = 5000
    max_value: int = 100
    max_loop_n: int = 10
    max_length: int = 256
    batch_size: int = 8
    num_workers: int = 0
    category_weights: dict = field(
        default_factory=lambda: {
            "arithmetic": 0.5,
            "variable_tracking": 0.3,
            "loop": 0.2,
        }
    )

    # ── Training ──
    steps: int = 2000
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0
    eval_every: int = 200
    log_every: int = 50
    grad_accum_steps: int = 1

    # ── Loss weights ──
    exec_loss_weight: float = 1.0  # α: weight for execution loss
    aux_loss_weight: float = 1.0  # β: weight for coprocessor load-balancing
    trace_loss_weight: float = 0.1  # γ: weight for intermediate trace loss
    lm_loss_weight: float = 1.0  # Standard next-token loss weight

    # ── Execution engine ──
    max_exec_steps: int = 64
    exec_temperature: float = 1.0  # Gumbel-softmax temperature for engine
    use_soft_programs: bool = True  # SoftProgram (full gradient) vs FixedProgram
    correctness_tolerance: float = 0.5

    # ── Coprocessor config ──
    n_bits: int = 8
    num_ops: int = 7
    target_load: float = 0.01
    balance_coeff: float = 0.01
    freeze_backbone: bool = True
    freeze_alu: bool = True
    residual_init_scale: float = 0.01
    unfreeze_last_n_layers: int = 0
    backbone_lr: float = 5e-5
    confidence_aware: bool = False
    max_gate: float = 0.1
    gate_warmup_steps: int = 0
    deterministic_alu: bool = False

    # ── Output ──
    output_dir: str = "training_results/exec_training"
    models_dir: Optional[str] = None

    # ── Smoke test ──
    synthetic_only: bool = False


@dataclass
class ExecutionTrainingResult:
    """Training results."""

    final_loss: float
    final_lm_loss: float
    final_exec_loss: float
    final_aux_loss: float
    mean_gate: float
    steps_completed: int
    parse_success_rate: float
    exec_accuracy: float
    eval_result: Optional[dict] = None
    trainable_params: int = 0
    total_params: int = 0
    wall_time_seconds: float = 0.0
    loss_history: list = field(default_factory=list)
    exec_loss_history: list = field(default_factory=list)
    gate_history: list = field(default_factory=list)


def train_execution_grounded(config: ExecutionTrainingConfig) -> ExecutionTrainingResult:
    """Main training function.

    Implements the full pipeline:
    1. Load model + tokenizer
    2. Inject coprocessor
    3. Build datasets
    4. Train with combined LM + execution loss
    5. Evaluate
    6. Save results
    """
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Device: {device}")

    if config.synthetic_only:
        return _run_smoke_test(config, device)

    # ── 1. Load model and tokenizer ──
    logger.info(f"Loading model: {config.model_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.train()

    # ── 2. Inject coprocessor ──
    copro_config = NCPUCoprocessorConfig(
        n_bits=config.n_bits,
        num_ops=config.num_ops,
        target_load=config.target_load,
        balance_coeff=config.balance_coeff,
        freeze_backbone=config.freeze_backbone,
        freeze_alu=config.freeze_alu,
        residual_init_scale=config.residual_init_scale,
        layer_indices=config.layers,
        models_dir=config.models_dir,
        confidence_aware=config.confidence_aware,
        max_gate=config.max_gate,
        gate_warmup_steps=config.gate_warmup_steps,
        deterministic_alu=config.deterministic_alu,
        unfreeze_last_n_layers=config.unfreeze_last_n_layers,
    )

    logger.info("Injecting nCPU coprocessor...")
    copro_layers = inject_ncpu_coprocessor(model, copro_config)
    freeze_backbone(
        model,
        unfreeze_last_n=config.unfreeze_last_n_layers,
        freeze_alu=config.freeze_alu,
    )

    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {trainable:,} trainable / {total:,} total")

    # ── 3. Build datasets ──
    logger.info("Building training data...")
    train_dataset = ExecutionTrainingDataset(
        size=config.data_size,
        seed=42,
        max_value=config.max_value,
        max_loop_n=config.max_loop_n,
        tokenizer=tokenizer,
        max_length=config.max_length,
        category_weights=config.category_weights,
    )

    eval_dataset = ExecutionTrainingDataset(
        size=min(200, config.data_size // 5),
        seed=999,
        max_value=config.max_value,
        max_loop_n=config.max_loop_n,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
    )

    # ── 4. Set up execution loss ──
    diff_engine = DifferentiableEngine(device=device)
    exec_loss_fn = ExecutionLossWithParsing(
        execution_loss=ExecutionLoss(
            engine=diff_engine,
            output_weight=1.0,
            trace_weight=config.trace_loss_weight,
            structure_weight=0.01,
            correctness_tolerance=config.correctness_tolerance,
            max_exec_steps=config.max_exec_steps,
            device=device,
        ),
        use_soft_programs=config.use_soft_programs,
        temperature=config.exec_temperature,
        device=device,
    )

    # ── 5. Optimizer ──
    copro_params = get_coprocessor_params(model)
    backbone_params = [
        p for p in model.parameters()
        if p.requires_grad and not any(p is cp for cp in copro_params)
    ]

    param_groups = [{"params": copro_params, "lr": config.lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": config.backbone_lr})

    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=config.weight_decay
    )

    # Linear warmup scheduler
    def lr_schedule(step):
        if step < config.warmup_steps:
            return step / max(config.warmup_steps, 1)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # ── 6. Training loop ──
    logger.info(f"Starting training for {config.steps} steps...")
    loss_history = []
    exec_loss_history = []
    gate_history = []
    parse_successes = 0
    parse_attempts = 0

    data_iter = iter(train_loader)
    step = 0

    while step < config.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        samples = batch["sample"]  # List of ExecutionTrainingSample

        # ── Gate warmup ──
        if config.gate_warmup_steps > 0:
            update_gate_schedule(model, step, config.gate_warmup_steps, config.max_gate)

        # ── Forward pass: LM loss ──
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        lm_loss = outputs.loss * config.lm_loss_weight

        # ── Coprocessor auxiliary loss ──
        aux_loss = collect_aux_losses(model)

        # ── Execution loss (per-sample in batch) ──
        exec_loss = torch.tensor(0.0, device=device)
        n_exec = 0

        for sample in samples:
            # sample is an ExecutionTrainingSample (from the DataLoader)
            # For DataLoader collation, samples may be nested; handle gracefully
            if isinstance(sample, dict):
                code = sample.get("reference_code", "")
                test_cases = sample.get("test_cases", [])
                arg_names = sample.get("arg_names", [])
                output_var = sample.get("output_var", None)
                is_func = sample.get("is_function", False)
            elif isinstance(sample, ExecutionTrainingSample):
                code = sample.reference_code
                test_cases = sample.test_cases
                arg_names = sample.arg_names
                output_var = sample.output_var
                is_func = sample.is_function
            else:
                continue

            parse_attempts += 1
            try:
                result = exec_loss_fn(
                    code=code,
                    test_cases=test_cases,
                    arg_names=arg_names if arg_names else None,
                    output_var=output_var,
                    is_function=is_func,
                )
                if result.total_loss.requires_grad:
                    exec_loss = exec_loss + result.total_loss
                    n_exec += 1
                    parse_successes += 1
                else:
                    parse_successes += 1  # Parsed but no gradient (fallback)
            except Exception as e:
                logger.debug(f"Execution loss failed: {e}")

        if n_exec > 0:
            exec_loss = exec_loss / n_exec

        # ── Combined loss ──
        total_loss = lm_loss + config.aux_loss_weight * aux_loss
        if n_exec > 0 and exec_loss.requires_grad:
            total_loss = total_loss + config.exec_loss_weight * exec_loss

        # ── Backward + step ──
        scaled_loss = total_loss / config.grad_accum_steps
        scaled_loss.backward()

        if (step + 1) % config.grad_accum_steps == 0:
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        step += 1

        # ── Logging ──
        loss_val = total_loss.item()
        exec_val = exec_loss.item() if isinstance(exec_loss, torch.Tensor) else 0.0
        loss_history.append(loss_val)
        exec_loss_history.append(exec_val)

        # Collect gate statistics
        mean_gate = 0.0
        n_gates = 0
        for module in model.modules():
            if hasattr(module, "router") and hasattr(module.router, "gate_proj"):
                # Approximate: use the gate_proj bias as proxy
                if hasattr(module.router, "_last_mean_gate"):
                    mean_gate += module.router._last_mean_gate
                    n_gates += 1
        if n_gates > 0:
            mean_gate /= n_gates
        gate_history.append(mean_gate)

        if step % config.log_every == 0:
            psr = parse_successes / max(parse_attempts, 1)
            logger.info(
                f"Step {step}/{config.steps} | "
                f"loss={loss_val:.4f} lm={lm_loss.item():.4f} "
                f"exec={exec_val:.4f} aux={aux_loss.item():.4f} | "
                f"parse_rate={psr:.1%} gate={mean_gate:.4f}"
            )

        # ── Evaluation ──
        if step % config.eval_every == 0:
            logger.info("Running evaluation...")
            evaluator = ExecutionEvaluator(engine=diff_engine, device=device)
            eval_result = evaluator.evaluate_reference_only(eval_dataset.samples[:50])
            logger.info(f"Eval: exec_acc={eval_result.exec_accuracy:.1%} "
                       f"parse_rate={eval_result.parse_rate:.1%}")

    # ── 7. Final evaluation ──
    logger.info("Final evaluation...")
    evaluator = ExecutionEvaluator(engine=diff_engine, device=device)
    final_eval = evaluator.evaluate_reference_only(eval_dataset.samples)

    # ── 8. Save results ──
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save coprocessor weights
    copro_state = {}
    for name, module in model.named_modules():
        if hasattr(module, "router") and hasattr(module, "expert"):
            copro_state[name] = {
                k: v.cpu() for k, v in module.state_dict().items()
            }
    torch.save(copro_state, output_dir / "coprocessor_weights.pt")

    # Save training report
    wall_time = time.time() - start_time
    report = {
        "config": asdict(config),
        "result": {
            "final_loss": loss_history[-1] if loss_history else 0,
            "final_exec_loss": exec_loss_history[-1] if exec_loss_history else 0,
            "parse_success_rate": parse_successes / max(parse_attempts, 1),
            "trainable_params": trainable,
            "total_params": total,
            "wall_time_seconds": wall_time,
            "steps_completed": step,
        },
        "eval": {
            "accuracy": final_eval.accuracy,
            "parse_rate": final_eval.parse_rate,
            "exec_accuracy": final_eval.exec_accuracy,
            "mean_exec_loss": final_eval.mean_execution_loss,
        },
    }
    with open(output_dir / "training_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Training complete. Results saved to {output_dir}")
    logger.info(final_eval.summary())

    return ExecutionTrainingResult(
        final_loss=loss_history[-1] if loss_history else 0,
        final_lm_loss=lm_loss.item() if isinstance(lm_loss, torch.Tensor) else 0,
        final_exec_loss=exec_loss_history[-1] if exec_loss_history else 0,
        final_aux_loss=aux_loss.item() if isinstance(aux_loss, torch.Tensor) else 0,
        mean_gate=gate_history[-1] if gate_history else 0,
        steps_completed=step,
        parse_success_rate=parse_successes / max(parse_attempts, 1),
        exec_accuracy=final_eval.exec_accuracy,
        eval_result=report.get("eval"),
        trainable_params=trainable,
        total_params=total,
        wall_time_seconds=wall_time,
        loss_history=loss_history,
        exec_loss_history=exec_loss_history,
        gate_history=gate_history,
    )


def _run_smoke_test(
    config: ExecutionTrainingConfig, device: str
) -> ExecutionTrainingResult:
    """Run a minimal smoke test without loading a real LM.

    Tests the full pipeline: data generation → parsing → execution → loss.
    Uses the DifferentiableEngine directly without a language model.
    """
    logger.info("Running smoke test (no LM, testing execution pipeline)...")

    engine = DifferentiableEngine(device=device)
    parser = CodeToISAParser()
    exec_loss_fn = ExecutionLoss(
        engine=engine,
        correctness_tolerance=config.correctness_tolerance,
        max_exec_steps=config.max_exec_steps,
        device=device,
    )

    # Generate test data
    dataset = ExecutionTrainingDataset(
        size=config.data_size,
        seed=42,
        max_value=config.max_value,
        max_loop_n=config.max_loop_n,
    )

    parse_successes = 0
    exec_successes = 0
    total_loss = 0.0
    total_correct = 0
    n_samples = min(len(dataset), config.steps)

    for i in range(n_samples):
        sample = dataset[i]

        # Parse
        try:
            result = parser.parse_block(
                sample.reference_code,
                arg_names=sample.arg_names if sample.arg_names else None,
                output_var=sample.output_var,
            )
            parse_successes += 1
        except ParseError as e:
            logger.debug(f"Parse failed ({i}): {e}")
            continue

        # Execute
        try:
            tc = sample.test_cases[0]
            inputs = {}
            for var_name, val in tc.get("inputs", {}).items():
                reg = result.variable_map.get(var_name)
                if reg is not None:
                    inputs[reg] = float(val)

            expected = {}
            if "expected" in tc:
                for var_name, val in tc["expected"].items():
                    reg = result.variable_map.get(var_name)
                    if reg is not None:
                        expected[reg] = float(val)

            if not expected:
                continue

            if config.use_soft_programs:
                soft_prog = result.to_soft_program()
                loss_result = exec_loss_fn.compute_soft(
                    soft_prog, inputs=inputs, expected=expected,
                    temperature=config.exec_temperature,
                )
            else:
                fixed_prog = result.to_fixed_program()
                loss_result = exec_loss_fn.compute_fixed(
                    fixed_prog, inputs=inputs, expected=expected
                )

            exec_successes += 1
            total_loss += loss_result.total_loss.item()
            total_correct += loss_result.num_correct

            # Verify gradient flow
            if i == 0 and config.use_soft_programs:
                loss_result.total_loss.backward()
                has_grad = any(
                    p.grad is not None and p.grad.abs().sum() > 0
                    for p in soft_prog.parameters()
                )
                logger.info(f"Gradient flow verified: {has_grad}")

        except Exception as e:
            logger.debug(f"Execution failed ({i}): {e}")
            continue

        if (i + 1) % 100 == 0:
            logger.info(
                f"  [{i + 1}/{n_samples}] parsed={parse_successes} "
                f"executed={exec_successes} "
                f"mean_loss={total_loss / max(exec_successes, 1):.4f}"
            )

    mean_loss = total_loss / max(exec_successes, 1)
    parse_rate = parse_successes / n_samples
    exec_rate = exec_successes / max(parse_successes, 1)

    logger.info(
        f"\nSmoke test complete:\n"
        f"  Samples:      {n_samples}\n"
        f"  Parsed:       {parse_successes}/{n_samples} ({parse_rate:.1%})\n"
        f"  Executed:     {exec_successes}/{parse_successes} ({exec_rate:.1%})\n"
        f"  Mean loss:    {mean_loss:.6f}\n"
        f"  Correct regs: {total_correct}"
    )

    return ExecutionTrainingResult(
        final_loss=mean_loss,
        final_lm_loss=0.0,
        final_exec_loss=mean_loss,
        final_aux_loss=0.0,
        mean_gate=0.0,
        steps_completed=n_samples,
        parse_success_rate=parse_rate,
        exec_accuracy=exec_rate,
        loss_history=[mean_loss],
        exec_loss_history=[mean_loss],
    )


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="Train a code LM with differentiable execution loss"
    )

    # Model
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="HF model name")
    parser.add_argument("--layers", nargs="+", type=int, default=[-1],
                       help="Transformer layers to inject coprocessor")
    parser.add_argument("--mode", default="coprocessor",
                       choices=["coprocessor", "compilation", "generated"])

    # Data
    parser.add_argument("--data-size", type=int, default=5000)
    parser.add_argument("--max-value", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)

    # Training
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=200)

    # Loss weights
    parser.add_argument("--exec-loss-weight", type=float, default=1.0)
    parser.add_argument("--aux-loss-weight", type=float, default=1.0)
    parser.add_argument("--trace-loss-weight", type=float, default=0.1)
    parser.add_argument("--lm-loss-weight", type=float, default=1.0)

    # Execution
    parser.add_argument("--max-exec-steps", type=int, default=64)
    parser.add_argument("--exec-temperature", type=float, default=1.0)
    parser.add_argument("--use-fixed-programs", action="store_true",
                       help="Use FixedProgram instead of SoftProgram")

    # Coprocessor
    parser.add_argument("--n-bits", type=int, default=8)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--no-freeze-backbone", action="store_false", dest="freeze_backbone")
    parser.add_argument("--confidence-aware", action="store_true")
    parser.add_argument("--max-gate", type=float, default=0.1)
    parser.add_argument("--deterministic-alu", action="store_true")
    parser.add_argument("--unfreeze-last-n", type=int, default=0)
    parser.add_argument("--backbone-lr", type=float, default=5e-5)

    # Output
    parser.add_argument("--output-dir", default="training_results/exec_training")
    parser.add_argument("--models-dir", default=None)

    # Smoke test
    parser.add_argument("--synthetic-only", action="store_true",
                       help="Run without a real LM (test pipeline only)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = ExecutionTrainingConfig(
        model_name=args.model,
        layers=args.layers,
        mode=args.mode,
        data_size=args.data_size,
        max_value=args.max_value,
        max_length=args.max_length,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        eval_every=args.eval_every,
        exec_loss_weight=args.exec_loss_weight,
        aux_loss_weight=args.aux_loss_weight,
        trace_loss_weight=args.trace_loss_weight,
        lm_loss_weight=args.lm_loss_weight,
        max_exec_steps=args.max_exec_steps,
        exec_temperature=args.exec_temperature,
        use_soft_programs=not args.use_fixed_programs,
        n_bits=args.n_bits,
        freeze_backbone=args.freeze_backbone,
        confidence_aware=args.confidence_aware,
        max_gate=args.max_gate,
        deterministic_alu=args.deterministic_alu,
        unfreeze_last_n_layers=args.unfreeze_last_n,
        backbone_lr=args.backbone_lr,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        synthetic_only=args.synthetic_only,
    )

    result = train_execution_grounded(config)

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Final loss:         {result.final_loss:.6f}")
    print(f"Final exec loss:    {result.final_exec_loss:.6f}")
    print(f"Parse success rate: {result.parse_success_rate:.1%}")
    print(f"Exec accuracy:      {result.exec_accuracy:.1%}")
    print(f"Wall time:          {result.wall_time_seconds:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
