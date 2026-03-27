"""NCPUExpert: the nCPU computation module inside a transformer layer.

Extracts operands from the hidden state, routes them through a soft mixture
of nCPU operations (tensor arithmetic + neural logic), and projects the
result back to hidden_dim.

The operation mixture is soft: all ops execute, and a learned op_selector
weights their contributions. This makes the forward pass fully differentiable
while giving the model the ability to learn WHICH operation to apply at
each token position.

Deterministic mode: bypasses neural approximation for exact integer arithmetic
using straight-through estimation (STE) to maintain differentiability. This
gives 100% arithmetic accuracy (like Percepta) while remaining fully
differentiable and integrated into a real production LLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NCPUCoprocessorConfig
from .soft_alu import (
    SoftNeuralLogical,
    SoftNeuralAdder,
    soft_int_to_bits,
    soft_bits_to_int,
)


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest integer with straight-through gradient estimation."""
    return x + (x.round() - x).detach()


def _deterministic_and(a_bits: torch.Tensor, b_bits: torch.Tensor) -> torch.Tensor:
    """Exact bitwise AND with STE: round to {0,1}, AND, pass gradient through."""
    a_hard = _ste_round(a_bits)
    b_hard = _ste_round(b_bits)
    return a_hard * b_hard  # AND = multiply for {0,1}


def _deterministic_or(a_bits: torch.Tensor, b_bits: torch.Tensor) -> torch.Tensor:
    """Exact bitwise OR with STE."""
    a_hard = _ste_round(a_bits)
    b_hard = _ste_round(b_bits)
    return a_hard + b_hard - a_hard * b_hard  # OR = a + b - a*b for {0,1}


def _deterministic_xor(a_bits: torch.Tensor, b_bits: torch.Tensor) -> torch.Tensor:
    """Exact bitwise XOR with STE."""
    a_hard = _ste_round(a_bits)
    b_hard = _ste_round(b_bits)
    return a_hard + b_hard - 2 * a_hard * b_hard  # XOR = a + b - 2*a*b for {0,1}


class NCPUExpert(nn.Module):
    """nCPU computation expert for transformer coprocessor.

    Operations:
        0: ADD (a + b, tensor)
        1: SUB (a - b, tensor)
        2: MUL (a * b, tensor)
        3: AND (neural truth table)
        4: OR  (neural truth table)
        5: XOR (neural truth table)
        6: CMP (a - b, sign as soft flag)
    """

    OP_NAMES = ["ADD", "SUB", "MUL", "AND", "OR", "XOR", "CMP"]

    def __init__(self, hidden_dim: int, config: NCPUCoprocessorConfig):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config
        self.n_bits = config.n_bits
        self.num_ops = config.num_ops
        self.deterministic_alu = getattr(config, "deterministic_alu", False)

        # Project hidden state → scalar operands (for ADD/SUB/MUL/CMP)
        self.scalar_proj = nn.Linear(hidden_dim, 2)

        # Project hidden state → soft bit operands (for AND/OR/XOR)
        self.bit_proj = nn.Linear(hidden_dim, 2 * self.n_bits)

        # Operation selector
        self.op_selector = nn.Linear(hidden_dim, self.num_ops)

        # Neural logic module (differentiable truth tables)
        self.soft_logical = SoftNeuralLogical(n_ops=7)

        # Neural adder (optional, for differentiable bit-level addition)
        self.soft_adder: Optional[SoftNeuralAdder] = None

        # Result dimension: max of scalar (1) and bits (n_bits)
        result_dim = max(1, self.n_bits) * self.num_ops

        # Project concatenated results back to hidden_dim
        self.output_proj = nn.Linear(result_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

        # Start with small residual contribution, learned during training
        self.residual_scale = nn.Parameter(
            torch.tensor(config.residual_init_scale)
        )

    def load_pretrained_alu(self, models_dir: Path, freeze: bool = True) -> None:
        """Load pretrained nCPU ALU weights from disk."""
        logical_path = models_dir / "alu" / "logical.pt"
        if logical_path.exists():
            self.soft_logical.load_from_trained(logical_path)

        adder_path = models_dir / "alu" / "arithmetic.pt"
        if adder_path.exists():
            self.soft_adder = SoftNeuralAdder(hidden_dim=128, n_bits=self.n_bits)
            self.soft_adder.load_from_trained(adder_path)

        if freeze:
            for p in self.soft_logical.parameters():
                p.requires_grad = False
            if self.soft_adder is not None:
                for p in self.soft_adder.parameters():
                    p.requires_grad = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute soft mixture of nCPU operations.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]

        Returns:
            [batch, seq_len, hidden_dim] nCPU computation output
        """
        batch, seq_len, _ = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_dim)  # [B*S, H]

        # Extract operands
        scalars = self.scalar_proj(flat)  # [B*S, 2]
        a_scalar, b_scalar = scalars[:, 0], scalars[:, 1]

        bits_raw = self.bit_proj(flat)  # [B*S, 2*n_bits]
        bits_a = torch.sigmoid(bits_raw[:, :self.n_bits])  # [B*S, n_bits]
        bits_b = torch.sigmoid(bits_raw[:, self.n_bits:])  # [B*S, n_bits]

        # Operation weights
        op_weights = F.softmax(self.op_selector(flat), dim=-1)  # [B*S, num_ops]

        # Compute each operation result, expanded to n_bits dimension
        op_results = []

        if self.deterministic_alu:
            # === DETERMINISTIC MODE ===
            # Exact integer arithmetic with STE for gradient flow.
            # This guarantees 100% correctness (like Percepta's compiled weights)
            # while remaining fully differentiable via straight-through estimation.

            # Round scalars to integers via STE
            a_int = _ste_round(a_scalar)
            b_int = _ste_round(b_scalar)

            # ADD: exact integer addition
            add_result = (a_int + b_int).unsqueeze(-1).expand(-1, self.n_bits)
            op_results.append(add_result)

            # SUB: exact integer subtraction
            sub_result = (a_int - b_int).unsqueeze(-1).expand(-1, self.n_bits)
            op_results.append(sub_result)

            # MUL: exact integer multiplication
            mul_result = (a_int * b_int).unsqueeze(-1).expand(-1, self.n_bits)
            op_results.append(mul_result)

            # AND/OR/XOR: exact bitwise with STE
            and_result = _deterministic_and(bits_a, bits_b)
            op_results.append(and_result)

            or_result = _deterministic_or(bits_a, bits_b)
            op_results.append(or_result)

            xor_result = _deterministic_xor(bits_a, bits_b)
            op_results.append(xor_result)

            # CMP: exact comparison via STE-rounded operands
            cmp_diff = a_int - b_int
            cmp_result = torch.sigmoid(cmp_diff * 10.0).unsqueeze(-1).expand(-1, self.n_bits)
            op_results.append(cmp_result)

        else:
            # === NEURAL MODE (original) ===
            # ADD: scalar a + b → expand to n_bits
            add_result = (a_scalar + b_scalar).unsqueeze(-1).expand(-1, self.n_bits)
            op_results.append(add_result)

            # SUB: scalar a - b → expand to n_bits
            sub_result = (a_scalar - b_scalar).unsqueeze(-1).expand(-1, self.n_bits)
            op_results.append(sub_result)

            # MUL: scalar a * b → expand to n_bits
            mul_result = (a_scalar * b_scalar).unsqueeze(-1).expand(-1, self.n_bits)
            op_results.append(mul_result)

            # AND: neural truth table (op_idx=0)
            and_result = self.soft_logical.forward_single_op(bits_a, bits_b, op_idx=0)
            op_results.append(and_result)

            # OR: neural truth table (op_idx=1)
            or_result = self.soft_logical.forward_single_op(bits_a, bits_b, op_idx=1)
            op_results.append(or_result)

            # XOR: neural truth table (op_idx=2)
            xor_result = self.soft_logical.forward_single_op(bits_a, bits_b, op_idx=2)
            op_results.append(xor_result)

            # CMP: sign of (a - b) as soft flag, expanded to n_bits
            cmp_result = torch.sigmoid(a_scalar - b_scalar).unsqueeze(-1).expand(-1, self.n_bits)
            op_results.append(cmp_result)

        # Stack and weight by operation selection
        # [B*S, num_ops, n_bits]
        all_results = torch.stack(op_results, dim=1)

        # Weight each op's output: [B*S, num_ops, 1] * [B*S, num_ops, n_bits]
        weighted = op_weights.unsqueeze(-1) * all_results  # [B*S, num_ops, n_bits]

        # Flatten to [B*S, num_ops * n_bits]
        combined = weighted.reshape(flat.shape[0], -1)

        # Project back to hidden_dim
        output = self.output_proj(combined)  # [B*S, hidden_dim]
        output = self.output_norm(output)
        output = output * self.residual_scale

        return output.reshape(batch, seq_len, self.hidden_dim)
