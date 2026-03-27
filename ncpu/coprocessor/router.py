"""Per-token router that gates computation through nCPU vs original MLP.

Uses a sigmoid gate (not top-k) so the gating is fully differentiable.
A load-balancing auxiliary loss encourages a target fraction of tokens
to route through the nCPU path.

Confidence-aware mode (optional): modulates the gate using MLP output
variance as an uncertainty signal. When the MLP is confident (low variance),
the gate stays near zero; when uncertain (high variance), the gate scales up.

Adaptive gate scheduling: max_gate can be annealed from 0 to target over
a warmup period, teaching the router WHEN to activate before giving it budget.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class NCPURouter(nn.Module):
    """Learned per-token gate deciding how much to use nCPU vs original MLP.

    gate = sigmoid(linear(hidden_state))

    Output blend: (1 - gate) * mlp_out + gate * ncpu_out

    The aux_loss encourages mean gate activation toward target_load,
    preventing the router from collapsing to always-on or always-off.

    When confidence_aware=True, the gate is further modulated by MLP output
    uncertainty (variance), and hard-capped at max_gate to prevent the
    0.3-0.9 gate problem observed in base→instruct transfer.
    """

    def __init__(
        self,
        hidden_dim: int,
        target_load: float = 0.01,
        balance_coeff: float = 0.01,
        confidence_aware: bool = False,
        max_gate: float = 0.1,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, 1)
        self.target_load = target_load
        self.balance_coeff = balance_coeff
        self.confidence_aware = confidence_aware
        self.max_gate = max_gate
        # Effective max_gate can be different from max_gate during warmup
        self._effective_max_gate = max_gate

        if confidence_aware:
            # Learns when the model is uncertain from MLP output variance.
            # Bias init at +2.0 so sigmoid starts near 0.88 — but multiplied
            # with a low-variance signal this keeps the gate mostly OFF.
            self.confidence_proj = nn.Linear(1, 1)
            nn.init.constant_(self.confidence_proj.bias, 2.0)

    def set_effective_max_gate(self, value: float) -> None:
        """Set the effective max_gate (used by adaptive scheduling)."""
        self._effective_max_gate = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        mlp_output: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token routing gate and auxiliary loss.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            mlp_output: [batch, seq_len, hidden_dim] (optional, for confidence-aware mode)

        Returns:
            gate: [batch, seq_len, 1] values in (0, effective_max_gate)
            aux_loss: scalar load-balancing loss
        """
        gate = torch.sigmoid(self.gate_proj(hidden_states))  # [B, S, 1]

        if self.confidence_aware and mlp_output is not None:
            # Per-token variance of MLP output → uncertainty signal
            mlp_var = mlp_output.var(dim=-1, keepdim=True)  # [B, S, 1]
            uncertainty = torch.sigmoid(self.confidence_proj(mlp_var))
            gate = gate * uncertainty

        # Hard cap to prevent overly aggressive gating (may be annealed)
        gate = gate * self._effective_max_gate

        # Load-balancing: penalize deviation from target activation rate
        mean_gate = gate.mean()
        aux_loss = self.balance_coeff * (mean_gate - self.target_load) ** 2

        return gate, aux_loss


def update_gate_schedule(model: nn.Module, step: int, warmup_steps: int, max_gate: float) -> float:
    """Update effective max_gate for all routers based on training step.

    Linear warmup: max_gate scales from 0 → max_gate over warmup_steps.
    After warmup, stays at max_gate.

    Returns the current effective max_gate value.
    """
    if warmup_steps <= 0:
        effective = max_gate
    else:
        progress = min(1.0, step / warmup_steps)
        effective = max_gate * progress

    from .coprocessor_layer import NCPUCoprocessorMLP
    for module in model.modules():
        if isinstance(module, NCPUCoprocessorMLP):
            module.router.set_effective_max_gate(effective)

    return effective
