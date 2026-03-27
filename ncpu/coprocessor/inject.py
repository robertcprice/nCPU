"""Injection utilities for patching nCPU coprocessors into transformer models.

Supports Qwen2/LLaMA-style architectures where layers are accessed via
model.model.layers[idx].mlp. Uses the same _replace_module pattern as
the SOME fast-weight system.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from .config import NCPUCoprocessorConfig
from .coprocessor_layer import NCPUCoprocessorMLP


def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    """Replace a named submodule in a model. Handles dotted paths and integer indices."""
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    final = parts[-1]
    if final.isdigit():
        parent[int(final)] = new_module
    else:
        setattr(parent, final, new_module)


def _resolve_layer_indices(model: nn.Module, indices: List[int]) -> List[int]:
    """Convert negative indices to positive, relative to total layer count."""
    # Find layers via model.model.layers (Qwen2/LLaMA pattern)
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h

    if layers is None:
        raise ValueError(
            "Cannot find transformer layers. Expected model.model.layers, "
            "model.layers, or model.transformer.h"
        )

    n_layers = len(layers)
    resolved = []
    for idx in indices:
        if idx < 0:
            idx = n_layers + idx
        if 0 <= idx < n_layers:
            resolved.append(idx)
        else:
            raise IndexError(f"Layer index {idx} out of range [0, {n_layers})")
    return resolved


def _count_total_layers(model: nn.Module) -> int:
    """Count total transformer layers in the model."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    elif hasattr(model, "layers"):
        return len(model.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    return 1


def _get_hidden_dim(model: nn.Module) -> int:
    """Extract hidden dimension from model config."""
    if hasattr(model, "config"):
        if hasattr(model.config, "hidden_size"):
            return model.config.hidden_size
        if hasattr(model.config, "d_model"):
            return model.config.d_model
    raise ValueError("Cannot determine hidden_dim from model.config")


def _get_layer_and_mlp_name(model: nn.Module, layer_idx: int) -> tuple:
    """Get the layer module and the dotted path to its MLP."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer = model.model.layers[layer_idx]
        mlp_path = f"model.layers.{layer_idx}.mlp"
        return layer, mlp_path, layer.mlp
    elif hasattr(model, "layers"):
        layer = model.layers[layer_idx]
        mlp_path = f"layers.{layer_idx}.mlp"
        return layer, mlp_path, layer.mlp
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layer = model.transformer.h[layer_idx]
        mlp_path = f"transformer.h.{layer_idx}.mlp"
        return layer, mlp_path, layer.mlp

    raise ValueError("Unsupported model architecture for MLP injection")


def inject_ncpu_coprocessor(
    model: nn.Module,
    config: Optional[NCPUCoprocessorConfig] = None,
) -> List[NCPUCoprocessorMLP]:
    """Inject nCPU coprocessor layers into a transformer model.

    Replaces MLP sublayers at specified layer indices with NCPUCoprocessorMLP
    wrappers that blend the original MLP with an nCPU expert.

    Args:
        model: transformer model (Qwen2, LLaMA, GPT-2, etc.)
        config: coprocessor configuration (uses defaults if None)

    Returns:
        List of injected NCPUCoprocessorMLP modules
    """
    if config is None:
        config = NCPUCoprocessorConfig()

    layer_indices = _resolve_layer_indices(model, config.layer_indices)
    hidden_dim = _get_hidden_dim(model)
    models_dir = config.resolve_models_dir()

    # Determine total layers for per-layer gate scaling
    n_total_layers = _count_total_layers(model)

    injected = []
    for idx in layer_indices:
        _, mlp_path, original_mlp = _get_layer_and_mlp_name(model, idx)

        # Per-layer gate scaling
        layer_config = config
        strategy = getattr(config, "layer_gate_strategy", "uniform")
        if strategy == "linear_decay" and n_total_layers > 1:
            # Later layers get lower max_gate (early layers benefit more from routing)
            # Layer 0 gets full max_gate, last layer gets max_gate * 0.25
            position_ratio = idx / max(1, n_total_layers - 1)  # 0.0 → 1.0
            layer_max_gate = config.max_gate * (1.0 - 0.75 * position_ratio)
            # Create a per-layer config copy with adjusted max_gate
            from dataclasses import replace
            layer_config = replace(config, max_gate=layer_max_gate)

        # Create coprocessor wrapper
        coprocessor_mlp = NCPUCoprocessorMLP(
            original_mlp=original_mlp,
            hidden_dim=hidden_dim,
            config=layer_config,
        )

        # Load pretrained ALU weights if available
        coprocessor_mlp.expert.load_pretrained_alu(
            models_dir=models_dir,
            freeze=config.freeze_alu,
        )

        # Move to same device AND dtype as the original MLP
        param = next(original_mlp.parameters())
        coprocessor_mlp = coprocessor_mlp.to(device=param.device, dtype=param.dtype)

        # Patch into model
        _replace_module(model, mlp_path, coprocessor_mlp)

        injected.append(coprocessor_mlp)

    return injected


def collect_aux_losses(model: nn.Module) -> torch.Tensor:
    """Sum load-balancing losses from all injected coprocessor layers."""
    total = None
    for module in model.modules():
        if isinstance(module, NCPUCoprocessorMLP) and module.aux_loss is not None:
            if total is None:
                total = module.aux_loss
            else:
                total = total + module.aux_loss
    if total is None:
        return torch.tensor(0.0)
    return total


def freeze_backbone(model: nn.Module, unfreeze_last_n: int = 0, freeze_alu: bool = True) -> None:
    """Freeze all model parameters except coprocessor modules.

    Args:
        model: transformer model
        unfreeze_last_n: also unfreeze the last N transformer layers + LM head.
            0 = only coprocessor params trainable (default)
            -1 = unfreeze everything (full fine-tune)
            N > 0 = unfreeze last N layers + lm_head
        freeze_alu: keep pretrained ALU weights frozen (default True)
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Then unfreeze coprocessor params
    for module in model.modules():
        if isinstance(module, NCPUCoprocessorMLP):
            # Unfreeze router
            for p in module.router.parameters():
                p.requires_grad = True
            # Unfreeze expert (optionally keeping ALU frozen)
            for name, p in module.expert.named_parameters():
                if freeze_alu and ("soft_logical" in name or "soft_adder" in name):
                    continue
                p.requires_grad = True

    # Optionally unfreeze backbone layers
    if unfreeze_last_n != 0:
        layers = None
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers

        if layers is not None:
            n_layers = len(layers)
            if unfreeze_last_n == -1:
                start = 0
            else:
                start = max(0, n_layers - unfreeze_last_n)
            for i in range(start, n_layers):
                for p in layers[i].parameters():
                    p.requires_grad = True

        # Always unfreeze LM head when unfreezing backbone layers
        if hasattr(model, "lm_head"):
            for p in model.lm_head.parameters():
                p.requires_grad = True


def get_coprocessor_params(model: nn.Module) -> List[nn.Parameter]:
    """Return only trainable coprocessor parameters for optimizer."""
    params = []
    for module in model.modules():
        if isinstance(module, NCPUCoprocessorMLP):
            for p in module.parameters():
                if p.requires_grad:
                    params.append(p)
    return params


def load_coprocessor_weights(model: nn.Module, weights_path: str) -> dict:
    """Load coprocessor weights into an already-injected model.

    Args:
        model: transformer model with injected coprocessor layers
        weights_path: path to coprocessor_weights.pt file

    Returns:
        The config dict from the weights file
    """
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Get config
    config = weights.get("_config", {})

    # Find all coprocessor modules
    coprocessor_modules = []
    for module in model.modules():
        if isinstance(module, NCPUCoprocessorMLP):
            coprocessor_modules.append(module)

    if not coprocessor_modules:
        raise ValueError("No coprocessor layers found in model. Did you call inject_ncpu_coprocessor()?")

    # Load weights for each layer
    for idx, coproc in enumerate(coprocessor_modules):
        router_key = f"layer_{idx}_router"
        expert_key = f"layer_{idx}_expert"

        if router_key in weights:
            coproc.router.load_state_dict(weights[router_key])

        if expert_key in weights:
            coproc.expert.load_state_dict(weights[expert_key])

    return config


@torch.no_grad()
def calibrate_confidence(
    model: nn.Module,
    tokenizer,
    calibration_texts: Optional[List[str]] = None,
    num_samples: int = 100,
    device: str = "cpu",
) -> dict:
    """Pre-calibrate confidence_proj using MLP output variance statistics.

    Runs a set of calibration texts through the model, measures the MLP output
    variance distribution, then sets the confidence_proj weight/bias so that
    the uncertainty signal is well-calibrated:
    - Median variance → 0.5 uncertainty (moderate activation)
    - Low variance (25th percentile) → ~0.1 uncertainty (mostly off)
    - High variance (75th percentile) → ~0.9 uncertainty (strong activation)

    This gives the confidence-aware router a better starting point than the
    default bias=2.0 initialization.

    Returns dict with calibration statistics.
    """
    import math

    if calibration_texts is None:
        # Default calibration: mix of arithmetic and natural language
        calibration_texts = [
            "What is 42 + 17?",
            "Calculate 256 * 128.",
            "The sum of 99 and 101 is",
            "If x = 5, then 3x + 7 =",
            "The weather today is sunny and warm.",
            "Once upon a time, there was a small village.",
            "In Python, to sort a list you can use",
            "The capital of France is",
            "def fibonacci(n):",
            "SELECT * FROM users WHERE",
        ] * (num_samples // 10 + 1)
        calibration_texts = calibration_texts[:num_samples]

    # Collect MLP output variances from the ORIGINAL MLPs (before coprocessor blending)
    variances = []

    model.eval()
    for text in calibration_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)

        # Hook to capture MLP output variance at each coprocessor layer
        layer_vars = []

        def make_hook(layer_vars_list):
            def hook_fn(module, input, output):
                # The original_mlp output variance
                if isinstance(output, torch.Tensor):
                    var = output.var(dim=-1).mean().item()
                    layer_vars_list.append(var)
            return hook_fn

        hooks = []
        for m in model.modules():
            if isinstance(m, NCPUCoprocessorMLP):
                h = m.original_mlp.register_forward_hook(make_hook(layer_vars))
                hooks.append(h)

        model(**inputs)
        for h in hooks:
            h.remove()

    if not variances:
        return {"status": "no_variances_collected"}

    variances_t = torch.tensor(variances)
    median_var = variances_t.median().item()
    p25 = variances_t.quantile(0.25).item()
    p75 = variances_t.quantile(0.75).item()

    # Calibrate confidence_proj so that:
    # sigmoid(weight * median_var + bias) = 0.5  → weight * median_var + bias = 0
    # sigmoid(weight * p25 + bias) ≈ 0.1        → weight * p25 + bias ≈ -2.2
    # Solving: weight = -2.2 / (p25 - median_var), bias = -weight * median_var
    if abs(p25 - median_var) > 1e-8:
        target_weight = -2.2 / (p25 - median_var)
        target_bias = -target_weight * median_var
    else:
        target_weight = 1.0
        target_bias = 0.0

    # Apply calibration to all confidence-aware routers
    calibrated_count = 0
    for m in model.modules():
        if isinstance(m, NCPUCoprocessorMLP) and m.router.confidence_aware:
            m.router.confidence_proj.weight.data.fill_(target_weight)
            m.router.confidence_proj.bias.data.fill_(target_bias)
            calibrated_count += 1

    stats = {
        "status": "calibrated",
        "num_samples": len(calibration_texts),
        "num_variances": len(variances),
        "median_variance": median_var,
        "p25_variance": p25,
        "p75_variance": p75,
        "calibrated_weight": target_weight,
        "calibrated_bias": target_bias,
        "layers_calibrated": calibrated_count,
    }
    return stats
