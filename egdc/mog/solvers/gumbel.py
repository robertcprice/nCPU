"""Gumbel-softmax straight-through estimators for differentiable discretization.

This is the bridge between soft continuous programs and hard discrete Mog code.
Instead of separate soft optimization + combinatorial discrete search,
we anneal Gumbel-softmax temperature to near-zero so the soft program
IS the discrete program at the end of training.

The straight-through estimator gives gradient signal through argmax:
forward: hard one-hot (discrete)
backward: soft gradient (differentiable)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gumbel_softmax(logits: torch.Tensor, temperature: float = 1.0,
                   hard: bool = False) -> torch.Tensor:
    """Gumbel-softmax with optional straight-through estimator.

    Args:
        logits: unnormalized log-probabilities
        temperature: softmax temperature (lower = more discrete)
        hard: if True, use straight-through estimator (hard forward, soft backward)
    """
    # Sample Gumbel noise
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = (logits + gumbels) / max(temperature, 1e-8)
    y_soft = F.softmax(y, dim=-1)

    if hard:
        # Straight-through: hard one-hot forward, soft gradient backward
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        # Use straight-through trick: y_hard in forward, y_soft gradient in backward
        return (y_hard - y_soft).detach() + y_soft
    return y_soft


def gumbel_read(storage: torch.Tensor, logits: torch.Tensor,
                temperature: float, hard: bool = False) -> torch.Tensor:
    """Read from storage using Gumbel-softmax attention."""
    weights = gumbel_softmax(logits, temperature, hard)
    return (weights * storage).sum()


def gumbel_op(a: torch.Tensor, b: torch.Tensor, op_logits: torch.Tensor,
              temperature: float, hard: bool = False) -> torch.Tensor:
    """Soft-select binary operation using Gumbel-softmax."""
    safe_b = torch.where(torch.abs(b) < 1e-6, torch.ones_like(b), b)
    results = torch.stack([
        a + b,
        a - b,
        a * b,
        a / safe_b,
        torch.remainder(torch.round(a), torch.clamp(torch.round(torch.abs(safe_b)), min=1.0)),
    ])
    weights = gumbel_softmax(op_logits, temperature, hard)
    return (weights * results).sum()


def gumbel_cmp(a: torch.Tensor, b: torch.Tensor, cmp_logits: torch.Tensor,
               temperature: float, hard: bool = False) -> torch.Tensor:
    """Soft comparison using Gumbel-softmax."""
    diff = a - b
    results = torch.stack([
        torch.sigmoid(diff / 0.25),
        torch.sigmoid(-diff / 0.25),
        torch.sigmoid(diff / 0.25),
        torch.sigmoid(-diff / 0.25),
        torch.exp(-(diff ** 2) / 0.125),
        1.0 - torch.exp(-(diff ** 2) / 0.125),
    ])
    weights = gumbel_softmax(cmp_logits, temperature, hard)
    return (weights * results).sum()
