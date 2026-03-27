"""Configuration for the nCPU differentiable coprocessor."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class NCPUCoprocessorConfig:
    """All tunables for the nCPU coprocessor injection."""

    # ALU dimensions
    n_bits: int = 8
    num_ops: int = 7  # ADD, SUB, MUL, AND, OR, XOR, CMP

    # Router
    target_load: float = 0.01
    balance_coeff: float = 0.01
    confidence_aware: bool = False
    max_gate: float = 0.1
    gate_warmup_steps: int = 0  # Anneal max_gate from 0→max_gate over N steps (0=disabled)

    # Per-layer gating: scale max_gate by layer position
    # "uniform" = same max_gate everywhere, "linear_decay" = later layers get less
    layer_gate_strategy: str = "uniform"

    # Deterministic ALU mode: bypass neural approximation for exact arithmetic
    deterministic_alu: bool = False

    # Training control
    freeze_backbone: bool = True
    freeze_alu: bool = True
    residual_init_scale: float = 0.01

    # Which transformer layers to inject into
    layer_indices: list[int] = field(default_factory=lambda: [-1])

    # Path to pretrained nCPU ALU models
    models_dir: Optional[str] = None

    # Soft bit decomposition temperature
    bit_temperature: float = 10.0

    # Partial backbone unfreezing: unfreeze last N transformer layers
    # 0 = freeze all backbone (default), -1 = unfreeze all, N = last N layers
    unfreeze_last_n_layers: int = 0

    def resolve_models_dir(self) -> Path:
        if self.models_dir is not None:
            return Path(self.models_dir)
        return Path("models")
