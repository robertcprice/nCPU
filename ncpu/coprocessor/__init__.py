"""nCPU Differentiable Coprocessor for Transformer Forward Passes.

Embeds nCPU's trained neural ALU components inside a transformer layer as a
routed expert. Neural truth tables provide differentiable logic (AND/OR/XOR),
tensor ops provide differentiable arithmetic (ADD/SUB/MUL), and a learned
per-token router gates computation through the nCPU path vs the original MLP.
"""

from .config import NCPUCoprocessorConfig
from .coprocessor_layer import NCPUCoprocessorMLP
from .inject import (
    inject_ncpu_coprocessor,
    collect_aux_losses,
    freeze_backbone,
    get_coprocessor_params,
)
from .data import ArithmeticDataset, GSM8KArithmeticDataset, MATHArithmeticDataset
from .train import TrainingConfig, train_coprocessor, run_synthetic_smoke_test

__all__ = [
    "NCPUCoprocessorConfig",
    "NCPUCoprocessorMLP",
    "inject_ncpu_coprocessor",
    "collect_aux_losses",
    "freeze_backbone",
    "get_coprocessor_params",
    "ArithmeticDataset",
    "GSM8KArithmeticDataset",
    "MATHArithmeticDataset",
    "TrainingConfig",
    "train_coprocessor",
    "run_synthetic_smoke_test",
]
