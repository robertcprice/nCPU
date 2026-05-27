"""Additive coprocessor trainer — fine-tunes the gate on code-relevant arithmetic.

The key insight: previous training teaches the coprocessor to DO arithmetic (ALU),
but not WHEN to activate during code generation. The gate learns to stay shut
because disrupting language generation always hurts more than arithmetic helps.

This trainer:
1. Loads existing coprocessor weights (ALU + projections are frozen)
2. Trains ONLY the gate_proj parameters (tiny: hidden_dim × 1 per layer)
3. Uses code-embedded arithmetic where activation SHOULD help
4. Uses contrastive pairs: same problem with/without correct answer,
   so the gate learns that activating on arithmetic tokens reduces loss
5. Progressive curriculum: start with easy arithmetic, add harder code patterns

Usage:
    # Fine-tune gate on code-embedded arithmetic, starting from instruct weights:
    python -m ncpu.coprocessor.train_additive \
        --model Qwen/Qwen3.5-4B \
        --checkpoint training_results/instruct_sweep/qwen3.5-4b/coprocessor_weights.pt \
        --steps 1000 \
        --max-gate 0.05 \
        --output-dir training_results/additive/qwen3.5-4b
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import NCPUCoprocessorConfig
from .inject import (
    inject_ncpu_coprocessor,
    collect_aux_losses,
    freeze_backbone,
    get_coprocessor_params,
    load_coprocessor_weights,
)
from .train import (
    evaluate_arithmetic_accuracy,
    evaluate_code_accuracy,
    TrainingResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contrastive code-arithmetic dataset
# ---------------------------------------------------------------------------

class CodeArithmeticContrastiveDataset(Dataset):
    """Generates (code_with_arithmetic, code_without_arithmetic) pairs.

    The contrastive loss encourages the gate to open wider on tokens
    where arithmetic is embedded in code, because the coprocessor's
    contribution reduces perplexity on the result tokens.
    """

    PATTERNS = [
        # (template, op_fn, operand_range)
        ("arr[{a} + {b}] = value  # index {result}", lambda a, b: a + b, (0, 50, 1, 20)),
        ("data[{a} * {b}]  # flat index {result}", lambda a, b: a * b, (0, 20, 1, 10)),
        ("if i < {a} - {b}:  # bound {result}", lambda a, b: a - b, (20, 100, 1, 15)),
        ("mask = {a} | {b}  # flags {result}", lambda a, b: a | b, (0, 255, 0, 255)),
        ("flags = {a} & {b}  # bits {result}", lambda a, b: a & b, (0, 255, 0, 255)),
        ("total += {a}  # now {result}", lambda a, b: a + b, (0, 100, 1, 50)),
        ("offset = {a} + {b} * 4  # {result}", lambda a, b: a + b * 4, (0, 30, 1, 10)),
        ("step = {a} % {b}  # remainder {result}", lambda a, b: a % b if b else 0, (10, 100, 2, 20)),
    ]

    def __init__(self, tokenizer, size: int = 5000, max_length: int = 48, seed: int = 42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        rng = random.Random(seed)

        for _ in range(size):
            template, op_fn, ranges = rng.choice(self.PATTERNS)
            lo_a, hi_a, lo_b, hi_b = ranges
            a = rng.randint(lo_a, hi_a)
            b = rng.randint(lo_b, hi_b)
            result = op_fn(a, b)
            text = template.format(a=a, b=b, result=result)

            # Tokenize: loss only on result tokens
            prompt_part = text.split("#")[0] if "#" in text else text.rsplit(str(result), 1)[0]
            full_ids = tokenizer(
                text, return_tensors="pt", padding="max_length",
                max_length=max_length, truncation=True, add_special_tokens=False,
            ).input_ids[0]
            prompt_ids = tokenizer(
                prompt_part, return_tensors="pt", padding=False,
                add_special_tokens=False,
            ).input_ids[0]

            labels = full_ids.clone()
            labels[:len(prompt_ids)] = -100
            attention_mask = (full_ids != tokenizer.pad_token_id).long()

            self.samples.append({
                "input_ids": full_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "operation": "CODE",
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


import random


@dataclass
class AdditiveTrainingConfig:
    """Configuration for additive gate training."""
    # Model
    model_name: str = "Qwen/Qwen3.5-4B"
    layers: List[int] = field(default_factory=lambda: [-1, -2])

    # Checkpoint to resume from
    checkpoint: Optional[str] = None

    # Data
    dataset_size: int = 5000
    max_length: int = 48

    # Training
    steps: int = 1000
    batch_size: int = 8
    lr: float = 5e-4
    warmup_steps: int = 100
    eval_every: int = 200
    log_every: int = 10
    grad_clip: float = 1.0

    # Gate-specific
    max_gate: float = 0.05
    gate_warmup_steps: int = 200
    target_load: float = 0.02
    balance_coeff: float = 0.001

    # What to freeze
    freeze_alu: bool = True
    freeze_projections: bool = True
    unfreeze_last_n_layers: int = 0
    backbone_lr: Optional[float] = None

    # Output
    output_dir: str = "training_results/additive"


def _get_gate_params(model: nn.Module) -> List[nn.Parameter]:
    """Extract only gate projection parameters from injected modules."""
    from .coprocessor_layer import NCPUCoprocessorMLP
    params = []
    for m in model.modules():
        if isinstance(m, NCPUCoprocessorMLP):
            params.extend(m.router.gate_proj.parameters())
            if m.router.confidence_proj is not None:
                params.extend(m.router.confidence_proj.parameters())
    return params


def train_additive(config: AdditiveTrainingConfig) -> TrainingResult:
    """Additive gate training: loads existing weights, trains only the gate."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model_dtype = torch.float32 if device == "cpu" else torch.bfloat16

    logger.info(f"Loading {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
    if hasattr(model_config, "text_config") and not hasattr(model_config, "vocab_size"):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        text_cfg_dict = (model_config.text_config if isinstance(model_config.text_config, dict)
                         else model_config.text_config.to_dict())
        text_config = Qwen3_5TextConfig(**text_cfg_dict)
        model = Qwen3_5ForCausalLM.from_pretrained(
            config.model_name, config=text_config,
            torch_dtype=model_dtype,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True, ignore_mismatched_sizes=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=model_dtype,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
        )
    if device == "cpu":
        model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total_params:,} params on {device}")

    # Evaluate baseline BEFORE injection
    baseline_eval = evaluate_arithmetic_accuracy(model, tokenizer, device=device)
    logger.info(f"Baseline arithmetic accuracy: {baseline_eval['overall_accuracy']:.1%}")

    # Inject coprocessor
    copro_config = NCPUCoprocessorConfig(
        layer_indices=config.layers,
        confidence_aware=True,
        max_gate=config.max_gate,
        target_load=config.target_load,
        balance_coeff=config.balance_coeff,
        freeze_alu=config.freeze_alu,
    )
    injected = inject_ncpu_coprocessor(model, copro_config)
    logger.info(f"Injected {len(injected)} layers")

    # Load existing weights if provided
    if config.checkpoint:
        loaded = load_coprocessor_weights(model, config.checkpoint)
        logger.info(f"Loaded checkpoint: {config.checkpoint} ({loaded} params)")

    # Freeze EVERYTHING except gate params
    freeze_backbone(model, unfreeze_last_n=config.unfreeze_last_n_layers, freeze_alu=True)

    # Now selectively unfreeze gate params
    gate_params = _get_gate_params(model)
    for p in model.parameters():
        p.requires_grad = False
    for p in gate_params:
        p.requires_grad = True

    # Also unfreeze backbone layers if requested
    if config.unfreeze_last_n_layers != 0:
        # Re-unfreeze backbone layers (freeze_backbone froze them)
        layers = model.model.layers if hasattr(model, "model") and hasattr(model.model, "layers") else model.layers
        if config.unfreeze_last_n_layers == -1:
            unfreeze_indices = range(len(layers))
        else:
            unfreeze_indices = range(len(layers) - config.unfreeze_last_n_layers, len(layers))
        for idx in unfreeze_indices:
            for p in layers[idx].parameters():
                p.requires_grad = True
        if hasattr(model, "lm_head"):
            for p in model.lm_head.parameters():
                p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    logger.info(f"Trainable (gate-only): {trainable_count:,} / {total_params:,} params "
                f"({100*trainable_count/total_params:.6f}%)")

    # Build dataset
    from .data import ArithmeticDataset, CombinedArithmeticDataset

    synth_ds = ArithmeticDataset(
        size=config.dataset_size // 2,
        max_value=999, tokenizer=tokenizer,
        max_length=config.max_length, difficulty="mixed",
    )
    code_ds = CodeArithmeticContrastiveDataset(
        tokenizer, size=config.dataset_size // 2,
        max_length=config.max_length,
    )
    combined = CombinedArithmeticDataset([synth_ds, code_ds])
    loader = DataLoader(combined, batch_size=config.batch_size, shuffle=True, drop_last=True)

    logger.info(f"Training data: {len(combined)} samples")

    # Optimizer — only gate params get the main LR
    if config.unfreeze_last_n_layers != 0 and config.backbone_lr is not None:
        gate_ids = {id(p) for p in gate_params}
        backbone_params = [p for p in trainable if id(p) not in gate_ids]
        optimizer = torch.optim.AdamW([
            {"params": gate_params, "lr": config.lr},
            {"params": backbone_params, "lr": config.backbone_lr},
        ], weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(gate_params, lr=config.lr, weight_decay=0.01)

    def get_lr(step):
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # Training loop
    result = TrainingResult(trainable_params=trainable_count, total_params=total_params)
    start_time = time.time()
    model.train()

    from .router import update_gate_schedule
    from .coprocessor_layer import NCPUCoprocessorMLP
    copro_modules = [m for m in model.modules() if isinstance(m, NCPUCoprocessorMLP)]

    data_iter = iter(loader)
    optimizer.zero_grad()

    for step in range(config.steps):
        if config.gate_warmup_steps > 0:
            update_gate_schedule(model, step, config.gate_warmup_steps, config.max_gate)

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        lm_loss = outputs.loss
        aux_loss = collect_aux_losses(model) * config.balance_coeff
        loss = lm_loss + aux_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
        optimizer.step()
        scheduler.step()

        # Gate stats
        mean_gate = 0.0
        if step % config.log_every == 0 and copro_modules:
            gate_vals = []
            with torch.no_grad():
                test_h = torch.randn(1, 1, model.config.hidden_size, device=device, dtype=model_dtype)
                for m in copro_modules:
                    g, _ = m.router(test_h)
                    gate_vals.append(g.mean().item())
            mean_gate = sum(gate_vals) / len(gate_vals)

        result.loss_history.append(loss.item())
        result.gate_history.append(mean_gate)

        if step % config.log_every == 0:
            logger.info(
                f"Step {step}/{config.steps}: loss={loss.item():.4f} "
                f"lm={lm_loss.item():.4f} aux={aux_loss.item():.6f} "
                f"gate={mean_gate:.4f} lr={scheduler.get_last_lr()[0]:.6f}"
            )

        if step > 0 and step % config.eval_every == 0:
            model.eval()
            eval_result = evaluate_arithmetic_accuracy(model, tokenizer, device=device)
            logger.info(f"  Eval: {eval_result['overall_accuracy']:.1%} "
                        f"(per-op: {eval_result['per_operation']})")
            result.eval_accuracy = eval_result["overall_accuracy"]
            model.train()

    # Final eval
    model.eval()
    final_eval = evaluate_arithmetic_accuracy(model, tokenizer, device=device)
    code_eval = evaluate_code_accuracy(model, tokenizer, device=device)
    result.eval_accuracy = final_eval["overall_accuracy"]
    result.final_loss = result.loss_history[-1] if result.loss_history else 0.0
    result.steps_completed = config.steps
    result.wall_time_seconds = time.time() - start_time

    logger.info(f"\nAdditive training complete in {result.wall_time_seconds:.1f}s")
    logger.info(f"Baseline accuracy:    {baseline_eval['overall_accuracy']:.1%}")
    logger.info(f"Arithmetic accuracy:  {final_eval['overall_accuracy']:.1%} "
                f"({final_eval['overall_accuracy'] - baseline_eval['overall_accuracy']:+.1%})")
    logger.info(f"Code accuracy:        {code_eval['overall_accuracy']:.1%}")

    # Save weights
    coprocessor_state = {
        "_config": {
            "confidence_aware": True,
            "max_gate": config.max_gate,
            "target_load": config.target_load,
            "additive_from": config.checkpoint,
        },
    }
    for i, module in enumerate(injected):
        coprocessor_state[f"layer_{i}_router"] = module.router.state_dict()
        coprocessor_state[f"layer_{i}_expert"] = module.expert.state_dict()
    save_path = output_dir / "coprocessor_weights.pt"
    torch.save(coprocessor_state, save_path)
    logger.info(f"Saved to {save_path}")

    # Report
    report = {
        "config": asdict(config),
        "baseline_eval": baseline_eval,
        "final_eval": final_eval,
        "code_eval": code_eval,
        "result": {
            "final_loss": result.final_loss,
            "mean_gate": result.gate_history[-1] if result.gate_history else 0.0,
            "steps_completed": result.steps_completed,
            "eval_accuracy": result.eval_accuracy,
            "code_accuracy": code_eval["overall_accuracy"],
            "trainable_params": result.trainable_params,
            "total_params": result.total_params,
            "wall_time_seconds": result.wall_time_seconds,
        },
    }
    with open(output_dir / "training_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    return result


def main():
    parser = argparse.ArgumentParser(description="Additive gate training for nCPU coprocessor")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--checkpoint", default=None,
                        help="Existing coprocessor weights to fine-tune from")
    parser.add_argument("--layers", nargs="+", type=int, default=[-1, -2])
    parser.add_argument("--dataset-size", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--max-gate", type=float, default=0.05)
    parser.add_argument("--gate-warmup-steps", type=int, default=200)
    parser.add_argument("--target-load", type=float, default=0.02)
    parser.add_argument("--balance-coeff", type=float, default=0.001)
    parser.add_argument("--unfreeze-last-n", type=int, default=0)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--output-dir", default="training_results/additive")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = AdditiveTrainingConfig(
        model_name=args.model,
        checkpoint=args.checkpoint,
        layers=args.layers,
        dataset_size=args.dataset_size,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        max_gate=args.max_gate,
        gate_warmup_steps=args.gate_warmup_steps,
        target_load=args.target_load,
        balance_coeff=args.balance_coeff,
        unfreeze_last_n_layers=args.unfreeze_last_n,
        backbone_lr=args.backbone_lr,
        output_dir=args.output_dir,
    )

    result = train_additive(config)

    print(f"\nAdditive Training Result:")
    print(f"  Steps: {result.steps_completed}")
    print(f"  Final loss: {result.final_loss:.4f}")
    print(f"  Eval accuracy: {result.eval_accuracy:.1%}")
    print(f"  Gate params: {result.trainable_params:,}")
    print(f"  Wall time: {result.wall_time_seconds:.1f}s")


if __name__ == "__main__":
    main()
