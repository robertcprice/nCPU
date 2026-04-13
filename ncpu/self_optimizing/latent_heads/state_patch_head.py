"""Learned state-to-weight patch head for descriptor-first fast-weight updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn


@dataclass
class StatePatchHeadConfig:
    """Configuration for a lightweight learned latent-state patch head."""

    input_dim: int = 16
    hidden_dim: int = 64
    output_dim: int = 16
    dropout: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StatePatchHead(nn.Module):
    """
    Small MLP that turns a latent-state descriptor signal into a weight patch signal.

    This is the learned replacement for the current hand-written projection rule.
    """

    def __init__(self, config: Optional[StatePatchHeadConfig] = None):
        super().__init__()
        self.config = config or StatePatchHeadConfig()
        self.network = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

    def transform(self, signal_projection: list[float], *, device: str | torch.device) -> list[float]:
        vector = torch.zeros(self.config.input_dim, dtype=torch.float32, device=device)
        if signal_projection:
            source = torch.tensor(signal_projection, dtype=torch.float32, device=device)
            width = min(source.numel(), vector.numel())
            vector[:width] = source[:width]
        with torch.no_grad():
            result = self.forward(vector.unsqueeze(0)).squeeze(0)
        return [float(value) for value in result.detach().cpu().tolist()]


def load_state_patch_head(
    *,
    path: str | Path,
    device: str | torch.device,
    config: Optional[StatePatchHeadConfig] = None,
) -> StatePatchHead:
    """Load a trained state-patch head checkpoint."""
    checkpoint_path = Path(path).expanduser()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    resolved_config = config
    if resolved_config is None and isinstance(payload, dict) and "config" in payload:
        resolved_config = StatePatchHeadConfig(**dict(payload["config"]))
    head = StatePatchHead(resolved_config)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    head.load_state_dict(state_dict)
    head = head.to(device)
    head.eval()
    return head
