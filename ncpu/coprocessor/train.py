"""Training harness for the nCPU differentiable coprocessor.

Injects the coprocessor into a transformer model (Qwen2, LLaMA, etc.),
freezes the backbone, and trains only the ~113K coprocessor parameters
on arithmetic data. The router learns WHEN to activate nCPU, while the
projection layers learn HOW to extract operands from the hidden state.

Usage:
    # Minimal (synthetic data, no GPU model needed):
    python -m ncpu.coprocessor.train --synthetic-only --steps 500

    # Full training with Qwen:
    python -m ncpu.coprocessor.train \
        --model Qwen/Qwen2.5-0.5B \
        --dataset synthetic+gsm8k \
        --steps 2000 \
        --lr 1e-3 \
        --batch-size 16 \
        --layers -1 -2 \
        --output-dir training_results/coprocessor/

    # Multi-layer injection:
    python -m ncpu.coprocessor.train \
        --model Qwen/Qwen2.5-0.5B \
        --layers 0 8 16 -1 \
        --steps 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import NCPUCoprocessorConfig
from .data import ArithmeticDataset, CombinedArithmeticDataset
from .inject import (
    inject_ncpu_coprocessor,
    collect_aux_losses,
    freeze_backbone,
    get_coprocessor_params,
    calibrate_confidence,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Full training configuration."""
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B"
    layers: List[int] = field(default_factory=lambda: [-1])

    # Data
    dataset: str = "synthetic"  # "synthetic", "gsm8k", "math", "synthetic+gsm8k"
    synthetic_size: int = 10000
    max_value: int = 999
    difficulty: str = "mixed"
    max_length: int = 32
    batch_size: int = 16
    num_workers: int = 0

    # Training
    steps: int = 2000
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    aux_loss_weight: float = 1.0
    grad_clip: float = 1.0
    eval_every: int = 100
    log_every: int = 10
    grad_accum_steps: int = 1  # gradient accumulation steps
    compile_model: bool = False  # use torch.compile on CUDA

    # Coprocessor
    n_bits: int = 8
    num_ops: int = 7
    target_load: float = 0.01
    balance_coeff: float = 0.01
    freeze_backbone: bool = True
    freeze_alu: bool = True
    residual_init_scale: float = 0.01
    unfreeze_last_n_layers: int = 0  # 0=none, -1=all, N=last N layers
    backbone_lr: Optional[float] = None  # separate LR for unfrozen backbone params (default: same as lr)
    confidence_aware: bool = False  # modulate gate using MLP output uncertainty
    max_gate: float = 0.1  # hard cap on gate value to prevent aggressive activation
    gate_warmup_steps: int = 0  # anneal max_gate from 0→max_gate over N steps (0=disabled)
    layer_gate_strategy: str = "uniform"  # "uniform" or "linear_decay"
    deterministic_alu: bool = False  # exact arithmetic mode (bypasses neural approximation)
    calibrate: bool = False  # pre-calibrate confidence_proj using held-out MLP variance stats

    # Output
    output_dir: str = "training_results/coprocessor"
    models_dir: str = "models"

    # Mode
    synthetic_only: bool = False  # Skip loading HF model, train on synthetic smoke test


@dataclass
class TrainingResult:
    """Collected metrics from a training run."""
    final_loss: float = 0.0
    final_aux_loss: float = 0.0
    mean_gate: float = 0.0
    steps_completed: int = 0
    eval_accuracy: float = 0.0
    trainable_params: int = 0
    total_params: int = 0
    wall_time_seconds: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    gate_history: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _extract_first_int(text: str) -> Optional[int]:
    match = re.search(r"-?\d+", text)
    return int(match.group(0)) if match else None

def evaluate_arithmetic_accuracy(
    model: nn.Module,
    tokenizer,
    num_samples: int = 200,
    max_value: int = 99,
    device: str = "cpu",
) -> dict:
    """Evaluate model on arithmetic problems by generating answers.

    Returns dict with per-operation and overall accuracy.
    """
    from .data import ArithmeticDataset

    ds = ArithmeticDataset(
        size=num_samples, max_value=max_value, seed=9999,
        difficulty="easy",
    )

    correct = 0
    total = 0
    per_op = {}

    model.eval()
    with torch.no_grad():
        for sample in ds.samples:
            prompt = sample.expression.split("=")[0] + "="
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            ).strip()

            predicted = _extract_first_int(generated)

            op = sample.operation
            if op not in per_op:
                per_op[op] = {"correct": 0, "total": 0}
            per_op[op]["total"] += 1
            total += 1

            if predicted == sample.result:
                correct += 1
                per_op[op]["correct"] += 1

    accuracy = correct / total if total > 0 else 0.0
    per_op_acc = {
        op: v["correct"] / v["total"] if v["total"] > 0 else 0.0
        for op, v in per_op.items()
    }

    return {
        "overall_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_operation": per_op_acc,
    }


def evaluate_code_accuracy(
    model: nn.Module,
    tokenizer,
    num_samples: int = 200,
    max_value: int = 255,
    device: str = "cpu",
) -> dict:
    """Evaluate model on code-embedded prompts by generating numeric results."""
    from .data import CodeArithmeticDataset

    ds = CodeArithmeticDataset(size=num_samples, max_value=max_value, seed=9999)

    correct = 0
    total = 0
    per_op = {}

    model.eval()
    with torch.no_grad():
        for sample in ds.raw_samples:
            prompt = sample.code_snippet + "\nResult:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            ).strip()
            predicted = _extract_first_int(generated)

            op = sample.operation
            if op not in per_op:
                per_op[op] = {"correct": 0, "total": 0}
            per_op[op]["total"] += 1
            total += 1

            if predicted == sample.result:
                correct += 1
                per_op[op]["correct"] += 1

    accuracy = correct / total if total > 0 else 0.0
    per_op_acc = {
        op: v["correct"] / v["total"] if v["total"] > 0 else 0.0
        for op, v in per_op.items()
    }

    return {
        "overall_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_operation": per_op_acc,
    }


def get_eval_mode(dataset: str) -> str:
    return "code" if "code" in dataset else "arithmetic"


def evaluate_training_accuracy(
    model: nn.Module,
    tokenizer,
    dataset: str,
    device: str = "cpu",
    max_value: int = 255,
) -> dict:
    if get_eval_mode(dataset) == "code":
        return evaluate_code_accuracy(
            model,
            tokenizer,
            max_value=max_value,
            device=device,
        )
    return evaluate_arithmetic_accuracy(model, tokenizer, device=device)


# ---------------------------------------------------------------------------
# Synthetic-only smoke test (no HF model needed)
# ---------------------------------------------------------------------------

class TinyTransformerLayer(nn.Module):
    """Minimal transformer layer for synthetic smoke testing."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, **kwargs):
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TinyTransformer(nn.Module):
    """Minimal transformer for smoke-testing the coprocessor training loop."""
    def __init__(self, vocab_size=1000, hidden_dim=128, n_layers=4):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_dim})()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            TinyTransformerLayer(hidden_dim) for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return type("Output", (), {"loss": loss, "logits": logits})()


def run_synthetic_smoke_test(config: TrainingConfig) -> TrainingResult:
    """Train coprocessor on a tiny model with synthetic data.

    This validates the full training pipeline without requiring a real
    LLM. Useful for CI and debugging.
    """
    logger.info("Running synthetic-only smoke test")

    device = "cpu"
    hidden_dim = 128
    vocab_size = 1000

    model = TinyTransformer(vocab_size=vocab_size, hidden_dim=hidden_dim, n_layers=4)

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
        layer_gate_strategy=config.layer_gate_strategy,
        deterministic_alu=config.deterministic_alu,
    )

    injected = inject_ncpu_coprocessor(model, copro_config)
    logger.info(f"Injected {len(injected)} coprocessor layers")

    if config.freeze_backbone:
        freeze_backbone(model, unfreeze_last_n=config.unfreeze_last_n_layers, freeze_alu=config.freeze_alu)

    trainable = get_coprocessor_params(model)
    if config.unfreeze_last_n_layers != 0 or not config.freeze_backbone:
        trainable = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable)
    logger.info(f"Trainable: {trainable_count:,} / {total_params:,} params "
                f"({100*trainable_count/total_params:.3f}%)")

    if config.unfreeze_last_n_layers != 0 and config.backbone_lr is not None:
        copro_params = get_coprocessor_params(model)
        copro_ids = {id(p) for p in copro_params}
        backbone_params = [p for p in trainable if id(p) not in copro_ids]
        param_groups = [
            {"params": copro_params, "lr": config.lr},
            {"params": backbone_params, "lr": config.backbone_lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)

    # Generate synthetic batch data
    batch_size = config.batch_size
    seq_len = 16
    result = TrainingResult(trainable_params=trainable_count, total_params=total_params)

    start_time = time.time()
    model.train()

    from .router import update_gate_schedule

    for step in range(config.steps):
        # Adaptive gate warmup
        if config.gate_warmup_steps > 0:
            update_gate_schedule(model, step, config.gate_warmup_steps, config.max_gate)

        # Random token sequences (synthetic — no real tokenizer)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Mask first half as prompt
        labels[:, :seq_len//2] = -100

        outputs = model(input_ids=input_ids, labels=labels)
        lm_loss = outputs.loss
        aux_loss = collect_aux_losses(model) * config.aux_loss_weight
        loss = lm_loss + aux_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
        optimizer.step()

        # Collect gate statistics (sample a forward pass through the router)
        gate_vals = []
        for m in model.modules():
            from .coprocessor_layer import NCPUCoprocessorMLP
            if isinstance(m, NCPUCoprocessorMLP) and m._aux_loss is not None:
                with torch.no_grad():
                    test_h = torch.randn(1, 1, 128, device="cpu")  # smoke test is on CPU
                    g, _ = m.router(test_h)
                    gate_vals.append(g.mean().item())
        mean_gate = sum(gate_vals) / len(gate_vals) if gate_vals else 0.0

        result.loss_history.append(loss.item())
        result.gate_history.append(mean_gate)

        if step % config.log_every == 0:
            logger.info(
                f"Step {step}/{config.steps}: "
                f"loss={loss.item():.4f} lm={lm_loss.item():.4f} "
                f"aux={aux_loss.item():.6f} gate={mean_gate:.4f}"
            )

    result.final_loss = result.loss_history[-1] if result.loss_history else 0.0
    result.final_aux_loss = aux_loss.item()
    result.mean_gate = mean_gate
    result.steps_completed = config.steps
    result.wall_time_seconds = time.time() - start_time

    logger.info(f"Smoke test complete in {result.wall_time_seconds:.1f}s")
    logger.info(f"Final loss: {result.final_loss:.4f}, gate: {result.mean_gate:.4f}")

    return result


# ---------------------------------------------------------------------------
# Full training with real model
# ---------------------------------------------------------------------------

def train_coprocessor(config: TrainingConfig) -> TrainingResult:
    """Full coprocessor training on a real transformer model.

    1. Load model + tokenizer from HuggingFace
    2. Inject coprocessor at specified layers
    3. Freeze backbone, train only coprocessor params
    4. Evaluate arithmetic accuracy before/after
    5. Save coprocessor weights
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Device: {device}")

    model_dtype = torch.float32 if device == "cpu" else torch.bfloat16

    # Load model (with VL/composite model detection for Qwen3.5)
    logger.info(f"Loading {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
    if hasattr(model_config, "text_config") and not hasattr(model_config, "vocab_size"):
        # Qwen3.5 VL model — load text-only CausalLM
        logger.info("Detected composite (VL) config, loading text-only CausalLM...")
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        text_cfg_dict = (model_config.text_config if isinstance(model_config.text_config, dict)
                         else model_config.text_config.to_dict())
        text_config = Qwen3_5TextConfig(**text_cfg_dict)
        model = Qwen3_5ForCausalLM.from_pretrained(
            config.model_name,
            config=text_config,
            torch_dtype=model_dtype,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=model_dtype,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
        )
    if device == "cpu":
        model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total_params:,} params on {device}")

    # Evaluate BEFORE injection
    eval_mode = get_eval_mode(config.dataset)
    logger.info(f"Evaluating baseline {eval_mode} accuracy...")
    baseline_eval = evaluate_training_accuracy(
        model,
        tokenizer,
        config.dataset,
        device=device,
        max_value=config.max_value,
    )
    logger.info(f"Baseline {eval_mode} accuracy: {baseline_eval['overall_accuracy']:.1%}")

    # Inject coprocessor
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
    )
    injected = inject_ncpu_coprocessor(model, copro_config)
    logger.info(f"Injected {len(injected)} coprocessor layer(s) at indices {config.layers}")
    if config.confidence_aware:
        logger.info(f"  Confidence-aware gating ENABLED (max_gate={config.max_gate})")

    if config.freeze_backbone:
        freeze_backbone(model, unfreeze_last_n=config.unfreeze_last_n_layers, freeze_alu=config.freeze_alu)
    else:
        for p in model.parameters():
            p.requires_grad = True

    # Pre-calibrate confidence projection if requested
    if config.confidence_aware and getattr(config, "calibrate", False):
        logger.info("Running confidence calibration on held-out samples...")
        cal_stats = calibrate_confidence(model, tokenizer, device=device)
        logger.info(f"  Calibration: {cal_stats}")

    trainable = get_coprocessor_params(model)
    if config.unfreeze_last_n_layers != 0 or not config.freeze_backbone:
        trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    logger.info(f"Trainable: {trainable_count:,} / {total_params:,} params "
                f"({100*trainable_count/total_params:.4f}%)")

    # Build dataset
    datasets = []
    if "synthetic" in config.dataset:
        datasets.append(ArithmeticDataset(
            size=config.synthetic_size,
            max_value=config.max_value,
            tokenizer=tokenizer,
            max_length=config.max_length,
            difficulty=config.difficulty,
        ))
    if "code" in config.dataset:
        from .data import CodeArithmeticDataset
        code_ds = CodeArithmeticDataset(
            size=config.synthetic_size,
            tokenizer=tokenizer,
            max_length=config.max_length,
            max_value=config.max_value,
        )
        datasets.append(code_ds)
        logger.info(f"Code-embedded: {len(code_ds)} samples")
    if "gsm8k" in config.dataset:
        from .data import GSM8KArithmeticDataset
        gsm8k_ds = GSM8KArithmeticDataset(tokenizer=tokenizer, max_length=config.max_length)
        if len(gsm8k_ds) > 0:
            datasets.append(gsm8k_ds)
            logger.info(f"GSM8K: {len(gsm8k_ds)} arithmetic samples extracted")
        else:
            logger.warning("GSM8K dataset not available, using synthetic only")
    if "math" in config.dataset:
        from .data import MATHArithmeticDataset
        math_ds = MATHArithmeticDataset(tokenizer=tokenizer, max_length=config.max_length)
        if len(math_ds) > 0:
            datasets.append(math_ds)
            logger.info(f"MATH: {len(math_ds)} arithmetic samples extracted")

    combined = CombinedArithmeticDataset(datasets) if len(datasets) > 1 else datasets[0]
    # Auto-tune num_workers for CUDA
    num_workers = config.num_workers
    if num_workers == 0 and device == "cuda":
        import os
        num_workers = min(4, os.cpu_count() or 1)
    loader = DataLoader(
        combined, batch_size=config.batch_size,
        shuffle=True, num_workers=num_workers,
        drop_last=True, pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    logger.info(f"Training data: {len(combined)} samples (workers={num_workers})")

    # Optimizer with parameter groups (separate LR for backbone vs coprocessor)
    if config.unfreeze_last_n_layers != 0 and config.backbone_lr is not None:
        copro_params = get_coprocessor_params(model)
        copro_ids = {id(p) for p in copro_params}
        backbone_params = [p for p in trainable if id(p) not in copro_ids]
        param_groups = [
            {"params": copro_params, "lr": config.lr},
            {"params": backbone_params, "lr": config.backbone_lr},
        ]
        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=config.weight_decay
        )
        logger.info(f"Using dual LR: coprocessor={config.lr}, backbone={config.backbone_lr}")
    else:
        optimizer = torch.optim.AdamW(
            trainable, lr=config.lr, weight_decay=config.weight_decay
        )

    def get_lr(step):
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # Optional torch.compile for CUDA speedup
    if config.compile_model and device == "cuda":
        try:
            logger.info("Compiling model with torch.compile()...")
            model = torch.compile(model)
            logger.info("Compilation complete")
        except Exception as e:
            logger.warning(f"torch.compile failed ({e}), continuing without compilation")

    # Training loop
    result = TrainingResult(trainable_params=trainable_count, total_params=total_params)
    start_time = time.time()
    model.train()

    from .router import update_gate_schedule
    from .coprocessor_layer import NCPUCoprocessorMLP

    # Pre-find coprocessor modules for gate stats (avoid repeated search)
    copro_modules = [m for m in model.modules() if isinstance(m, NCPUCoprocessorMLP)]

    grad_accum = config.grad_accum_steps
    step = 0
    data_iter = iter(loader)
    optimizer.zero_grad()

    while step < config.steps:
        # Adaptive gate warmup
        if config.gate_warmup_steps > 0:
            update_gate_schedule(model, step, config.gate_warmup_steps, config.max_gate)

        accum_loss = 0.0
        accum_lm_loss = 0.0
        accum_aux_loss = 0.0

        for accum_step in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            lm_loss = outputs.loss / grad_accum
            aux_loss = collect_aux_losses(model) * config.aux_loss_weight / grad_accum
            loss = lm_loss + aux_loss
            loss.backward()

            accum_loss += loss.item()
            accum_lm_loss += lm_loss.item()
            accum_aux_loss += aux_loss.item()

        torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Gate statistics (only on log steps to avoid overhead)
        mean_gate = 0.0
        if step % config.log_every == 0 and copro_modules:
            gate_vals = []
            with torch.no_grad():
                test_hidden = torch.randn(1, 1, model.config.hidden_size, device=device,
                                          dtype=model_dtype)
                for m in copro_modules:
                    gate, _ = m.router(test_hidden)
                    gate_vals.append(gate.mean().item())
            mean_gate = sum(gate_vals) / len(gate_vals)

        result.loss_history.append(accum_loss)
        result.gate_history.append(mean_gate)

        if step % config.log_every == 0:
            logger.info(
                f"Step {step}/{config.steps}: "
                f"loss={accum_loss:.4f} lm={accum_lm_loss:.4f} "
                f"aux={accum_aux_loss:.6f} gate={mean_gate:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )

        if step > 0 and step % config.eval_every == 0:
            model.eval()
            eval_result = evaluate_training_accuracy(
                model,
                tokenizer,
                config.dataset,
                device=device,
                max_value=config.max_value,
            )
            logger.info(f"  Eval {eval_mode} accuracy: {eval_result['overall_accuracy']:.1%} "
                        f"(per-op: {eval_result['per_operation']})")
            result.eval_accuracy = eval_result["overall_accuracy"]
            model.train()

        step += 1

    # Final evaluation
    model.eval()
    final_eval = evaluate_training_accuracy(
        model,
        tokenizer,
        config.dataset,
        device=device,
        max_value=config.max_value,
    )
    result.eval_accuracy = final_eval["overall_accuracy"]
    result.final_loss = result.loss_history[-1] if result.loss_history else 0.0
    result.final_aux_loss = aux_loss.item()
    result.mean_gate = mean_gate
    result.steps_completed = step
    result.wall_time_seconds = time.time() - start_time

    logger.info(f"\nTraining complete in {result.wall_time_seconds:.1f}s")
    logger.info(f"Baseline {eval_mode} accuracy: {baseline_eval['overall_accuracy']:.1%}")
    logger.info(f"Final {eval_mode} accuracy:    {result.eval_accuracy:.1%}")
    logger.info(f"Delta:                         {result.eval_accuracy - baseline_eval['overall_accuracy']:+.1%}")

    # Save coprocessor weights (include config metadata for benchmark loading)
    coprocessor_state = {
        "_config": {
            "confidence_aware": config.confidence_aware,
            "max_gate": config.max_gate,
            "target_load": config.target_load,
            "deterministic_alu": config.deterministic_alu,
            "layer_gate_strategy": config.layer_gate_strategy,
        },
    }
    for i, module in enumerate(injected):
        coprocessor_state[f"layer_{i}_router"] = module.router.state_dict()
        coprocessor_state[f"layer_{i}_expert"] = module.expert.state_dict()
    save_path = output_dir / "coprocessor_weights.pt"
    torch.save(coprocessor_state, save_path)
    logger.info(f"Saved coprocessor weights to {save_path}")

    # Save training report
    report = {
        "config": asdict(config) if hasattr(config, "__dataclass_fields__") else vars(config),
        "baseline_eval": baseline_eval,
        "final_eval": final_eval,
        "result": {
            "final_loss": result.final_loss,
            "final_aux_loss": result.final_aux_loss,
            "mean_gate": result.mean_gate,
            "steps_completed": result.steps_completed,
            "eval_accuracy": result.eval_accuracy,
            "eval_mode": eval_mode,
            "trainable_params": result.trainable_params,
            "total_params": result.total_params,
            "wall_time_seconds": result.wall_time_seconds,
        },
    }
    report_path = output_dir / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Saved report to {report_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train nCPU coprocessor")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--layers", nargs="+", type=int, default=[-1])
    parser.add_argument("--dataset", default="synthetic",
                        choices=["synthetic", "gsm8k", "math", "synthetic+gsm8k", "synthetic+math",
                                 "code", "synthetic+code", "code+gsm8k"])
    parser.add_argument("--synthetic-size", type=int, default=10000)
    parser.add_argument("--max-value", type=int, default=999)
    parser.add_argument("--difficulty", default="mixed",
                        choices=["easy", "medium", "hard", "mixed"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--target-load", type=float, default=0.01)
    parser.add_argument("--output-dir", default="training_results/coprocessor")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Smoke test with tiny model, no HF download")
    parser.add_argument("--n-bits", type=int, default=8,
                        help="Bit width for ALU operations (default: 8)")
    parser.add_argument("--unfreeze-last-n", type=int, default=0,
                        help="Unfreeze last N transformer layers (0=none, -1=all)")
    parser.add_argument("--backbone-lr", type=float, default=None,
                        help="Separate LR for unfrozen backbone layers (default: same as --lr)")
    parser.add_argument("--confidence-aware", action="store_true",
                        help="Enable confidence-aware gating (modulates gate by MLP uncertainty)")
    parser.add_argument("--max-gate", type=float, default=0.1,
                        help="Hard cap on gate activation (default: 0.1)")
    parser.add_argument("--gate-warmup-steps", type=int, default=0,
                        help="Anneal max_gate from 0→max_gate over N steps (default: 0=disabled)")
    parser.add_argument("--layer-gate-strategy", default="uniform",
                        choices=["uniform", "linear_decay"],
                        help="Per-layer gate scaling strategy")
    parser.add_argument("--deterministic-alu", action="store_true",
                        help="Use exact arithmetic (bypasses neural approximation)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Pre-calibrate confidence_proj from MLP variance stats")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for CUDA speedup")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = TrainingConfig(
        model_name=args.model,
        layers=args.layers,
        dataset=args.dataset,
        synthetic_size=args.synthetic_size,
        max_value=args.max_value,
        difficulty=args.difficulty,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        target_load=args.target_load,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        synthetic_only=args.synthetic_only,
        n_bits=args.n_bits,
        unfreeze_last_n_layers=args.unfreeze_last_n,
        backbone_lr=args.backbone_lr,
        confidence_aware=args.confidence_aware,
        max_gate=args.max_gate,
        gate_warmup_steps=args.gate_warmup_steps,
        layer_gate_strategy=args.layer_gate_strategy,
        deterministic_alu=args.deterministic_alu,
        calibrate=args.calibrate,
        grad_accum_steps=args.grad_accum_steps,
        compile_model=args.compile,
    )

    if config.synthetic_only:
        result = run_synthetic_smoke_test(config)
    else:
        result = train_coprocessor(config)

    print(f"\nTraining result:")
    print(f"  Steps: {result.steps_completed}")
    print(f"  Final loss: {result.final_loss:.4f}")
    print(f"  Mean gate: {result.mean_gate:.4f}")
    print(f"  Eval accuracy: {result.eval_accuracy:.1%}")
    print(f"  Trainable params: {result.trainable_params:,}")
    print(f"  Wall time: {result.wall_time_seconds:.1f}s")


if __name__ == "__main__":
    main()
