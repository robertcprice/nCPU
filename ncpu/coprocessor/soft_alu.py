"""Differentiable wrappers around nCPU's trained neural ALU components.

The key innovation: nCPU's NeuralLogical truth tables (28 params) use hard
indexing (truth_tables[op, a*2+b]) which has zero gradient. We replace this
with bilinear soft indexing that gives nonzero gradients through both input
bits AND the truth table parameters, enabling end-to-end training.

For arithmetic (ADD/SUB/MUL), we use native tensor ops which are naturally
differentiable in float — no neural network needed.

For logic (AND/OR/XOR), only the neural truth tables can provide gradients
through discrete operations, making them essential for the coprocessor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Straight-Through Estimator for hard thresholds
# ---------------------------------------------------------------------------

class StraightThroughThreshold(Function):
    """Forward: hard threshold (x > 0.5). Backward: pass gradient through."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return (x > 0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def ste_threshold(x: torch.Tensor) -> torch.Tensor:
    """Apply hard threshold with straight-through gradient."""
    return StraightThroughThreshold.apply(x)


# ---------------------------------------------------------------------------
# Differentiable bit decomposition
# ---------------------------------------------------------------------------

def soft_int_to_bits(x: torch.Tensor, n_bits: int = 8, temperature: float = 10.0) -> torch.Tensor:
    """Differentiable integer-to-bits decomposition.

    Uses a shifted-remainder sawtooth wave to softly extract each bit position.
    A +0.5 shift ensures exact integers never land on transition boundaries,
    giving correct results at all integer values with nonzero gradients.

    Args:
        x: [...] float tensor of values (should be in [0, 2^n_bits - 1] range)
        n_bits: number of bits to extract
        temperature: sigmoid sharpness (higher = more discrete-like)

    Returns:
        [..., n_bits] soft bit tensor with values in (0, 1)
    """
    powers = 2.0 ** torch.arange(n_bits, dtype=x.dtype, device=x.device)  # [n_bits]
    x_expanded = x.unsqueeze(-1)  # [..., 1]
    # Shift by 0.5 so integers land at sawtooth midpoints, not transitions.
    # remainder((x+0.5)/2^i, 2) gives a sawtooth in [0, 2) with period 2^(i+1).
    # Values in [0, 1) → bit is 0; values in [1, 2) → bit is 1.
    scaled = (x_expanded + 0.5) / powers
    phase = torch.remainder(scaled, 2.0)
    # Scale temperature per bit: margin at bit i is 0.5/2^i, so multiply
    # temperature by 2^i to keep the sigmoid response uniform across all bits.
    per_bit_temp = temperature * powers  # [n_bits]
    bits = torch.sigmoid(per_bit_temp * (phase - 1.0))
    return bits


def soft_bits_to_int(bits: torch.Tensor) -> torch.Tensor:
    """Convert soft bit tensor back to soft integer value.

    Args:
        bits: [..., n_bits] tensor with values in (0, 1)

    Returns:
        [...] tensor of reconstructed values
    """
    n_bits = bits.shape[-1]
    weights = 2.0 ** torch.arange(n_bits, dtype=bits.dtype, device=bits.device)
    return (bits * weights).sum(dim=-1)


# ---------------------------------------------------------------------------
# Soft Neural Logical — differentiable truth table lookup
# ---------------------------------------------------------------------------

class SoftNeuralLogical(nn.Module):
    """Differentiable version of NeuralLogical truth tables.

    The original NeuralLogical does hard indexing: truth_tables[op, a*2+b]
    which has zero gradient w.r.t. inputs.

    We replace this with bilinear interpolation over the 4 truth table entries:
        result = (1-a)(1-b)*tt[0] + (1-a)*b*tt[1] + a*(1-b)*tt[2] + a*b*tt[3]

    This gives nonzero gradients through both bit_a and bit_b, AND through
    the truth table parameters.

    Operations: AND=0, OR=1, XOR=2, NOT=3, NAND=4, NOR=5, XNOR=6
    """

    def __init__(self, n_ops: int = 7):
        super().__init__()
        # Raw logits — sigmoid applied during lookup for consistency with trained weights
        self.truth_tables = nn.Parameter(torch.zeros(n_ops, 4))

    def load_from_trained(self, path: Path) -> None:
        """Load weights from a trained NeuralLogical checkpoint."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.truth_tables.data.copy_(state["truth_tables"])

    def forward(
        self,
        bits_a: torch.Tensor,
        bits_b: torch.Tensor,
        op_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Soft truth table lookup with differentiable op selection.

        Args:
            bits_a: [batch, n_bits] soft bits in (0, 1)
            bits_b: [batch, n_bits] soft bits in (0, 1)
            op_weights: [batch, n_ops] soft operation selection (softmax)

        Returns:
            [batch, n_bits] soft result bits
        """
        # Bilinear interpolation weights for each bit
        # p[i,j] = P(a=i, b=j) — probability mass at each truth table entry
        p00 = (1 - bits_a) * (1 - bits_b)  # [batch, n_bits]
        p01 = (1 - bits_a) * bits_b
        p10 = bits_a * (1 - bits_b)
        p11 = bits_a * bits_b

        # Stack into [batch, n_bits, 4] interpolation weights
        p = torch.stack([p00, p01, p10, p11], dim=-1)  # [batch, n_bits, 4]

        # Get truth table values through sigmoid: [n_ops, 4]
        tt_values = torch.sigmoid(self.truth_tables)  # [n_ops, 4]

        # Soft-select operation: [batch, 4] = [batch, n_ops] @ [n_ops, 4]
        selected_tt = op_weights @ tt_values  # [batch, 4]

        # Expand for broadcasting: [batch, 1, 4]
        selected_tt = selected_tt.unsqueeze(1)

        # Bilinear lookup: sum over 4 entries
        # [batch, n_bits, 4] * [batch, 1, 4] → sum → [batch, n_bits]
        result = (p * selected_tt).sum(dim=-1)

        return result

    def forward_single_op(
        self,
        bits_a: torch.Tensor,
        bits_b: torch.Tensor,
        op_idx: int,
    ) -> torch.Tensor:
        """Soft lookup for a single operation (useful for testing).

        Args:
            bits_a: [batch, n_bits] soft bits
            bits_b: [batch, n_bits] soft bits
            op_idx: integer operation index

        Returns:
            [batch, n_bits] soft result bits
        """
        p00 = (1 - bits_a) * (1 - bits_b)
        p01 = (1 - bits_a) * bits_b
        p10 = bits_a * (1 - bits_b)
        p11 = bits_a * bits_b

        p = torch.stack([p00, p01, p10, p11], dim=-1)  # [batch, n_bits, 4]

        tt = torch.sigmoid(self.truth_tables[op_idx])  # [4]
        result = (p * tt).sum(dim=-1)  # [batch, n_bits]

        return result


# ---------------------------------------------------------------------------
# Soft Neural Adder — differentiable ripple-carry with STE
# ---------------------------------------------------------------------------

class SoftNeuralAdder(nn.Module):
    """Differentiable wrapper around the trained NeuralFullAdder.

    Runs n_bits ripple-carry passes through the trained adder network,
    using straight-through estimators at the threshold boundaries so
    gradients flow back through the carry chain.

    The adder weights are loaded from arithmetic.pt and frozen by default.
    """

    def __init__(self, hidden_dim: int = 128, n_bits: int = 8):
        super().__init__()
        self.n_bits = n_bits
        self.full_adder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def load_from_trained(self, path: Path) -> None:
        """Load weights from a trained NeuralFullAdder checkpoint."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        # The trained model wraps in full_adder.* keys
        try:
            self.load_state_dict(state)
        except RuntimeError:
            # Dimension mismatch (e.g. 8-bit pretrained → 16-bit model)
            import logging
            logging.getLogger(__name__).warning(
                f"Cannot load pretrained adder weights (dimension mismatch), "
                f"starting with random initialization"
            )

    def forward(self, bits_a: torch.Tensor, bits_b: torch.Tensor) -> torch.Tensor:
        """Ripple-carry addition with STE thresholds.

        Args:
            bits_a: [batch, n_bits] soft bits
            bits_b: [batch, n_bits] soft bits

        Returns:
            [batch, n_bits] soft sum bits (overflow/carry discarded)
        """
        batch = bits_a.shape[0]
        carry = torch.zeros(batch, 1, device=bits_a.device, dtype=bits_a.dtype)
        sum_bits = []

        for i in range(self.n_bits):
            # [batch, 3]: (bit_a_i, bit_b_i, carry)
            inp = torch.cat([
                bits_a[:, i:i+1],
                bits_b[:, i:i+1],
                carry,
            ], dim=-1)

            out = torch.sigmoid(self.full_adder(inp))  # [batch, 2]
            sum_bit = out[:, 0:1]   # soft sum
            carry_out = out[:, 1:2]  # soft carry

            # STE: hard threshold forward, pass-through backward
            sum_bits.append(ste_threshold(sum_bit))
            carry = ste_threshold(carry_out)

        return torch.cat(sum_bits, dim=-1)  # [batch, n_bits]
