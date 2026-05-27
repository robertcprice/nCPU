"""NCPUCoprocessorMLP: drop-in replacement for a transformer MLP sublayer.

Wraps the original MLP and adds an nCPU expert path with learned routing.
The forward signature matches the original MLP so the transformer layer
doesn't need any changes.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import NCPUCoprocessorConfig
from .ncpu_expert import NCPUExpert
from .router import NCPURouter


class NCPUCoprocessorMLP(nn.Module):
    """MLP wrapper that blends original MLP output with nCPU expert output.

    output = (1 - gate) * mlp_out + gate * ncpu_out

    The gate is per-token and learned. The original MLP can be frozen
    during coprocessor training so only ~113K new parameters are updated.
    """

    def __init__(
        self,
        original_mlp: nn.Module,
        hidden_dim: int,
        config: NCPUCoprocessorConfig,
    ):
        super().__init__()
        self.original_mlp = original_mlp
        self.router = NCPURouter(
            hidden_dim=hidden_dim,
            target_load=config.target_load,
            balance_coeff=config.balance_coeff,
            confidence_aware=config.confidence_aware,
            max_gate=config.max_gate,
        )
        self.expert = NCPUExpert(hidden_dim=hidden_dim, config=config)

        # Stored after each forward pass for collection during training
        self._aux_loss: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Blended forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            **kwargs: passed through to original MLP (e.g. residual)

        Returns:
            [batch, seq_len, hidden_dim] blended output
        """
        # Ensure hidden_states matches the original MLP's weight dtype. This
        # defends against upstream mixed-precision layernorms that may
        # promote activations to float32 even when weights stay bf16.
        original_weight = None
        for p in self.original_mlp.parameters():
            if p.is_floating_point():
                original_weight = p
                break
        import os
        _dbg = os.environ.get("NPCOT_DTYPE_DEBUG") == "1"
        if _dbg:
            print(f"[coproc-fwd] hidden={hidden_states.dtype} weight={original_weight.dtype if original_weight is not None else None}", flush=True)
        if original_weight is not None and hidden_states.dtype != original_weight.dtype:
            hidden_states = hidden_states.to(dtype=original_weight.dtype)
            if _dbg:
                print(f"[coproc-fwd] cast hidden → {hidden_states.dtype}", flush=True)
        # Original MLP path
        mlp_out = self.original_mlp(hidden_states, **kwargs)

        # nCPU expert path
        ncpu_out = self.expert(hidden_states)

        # Routing gate (pass mlp_out for confidence-aware mode)
        gate, aux_loss = self.router(hidden_states, mlp_output=mlp_out)
        self._aux_loss = aux_loss

        # Blend: (1 - gate) * MLP + gate * nCPU
        output = (1 - gate) * mlp_out + gate * ncpu_out

        return output

    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        return self._aux_loss
