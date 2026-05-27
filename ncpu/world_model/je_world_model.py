"""
JEPA Execution / Machine World Model (J-EWM)

Core idea:
Learn a fast predictor that models the dynamics of an nCPU machine
in a learned abstract latent space.

(state_t, action) -> predicted_latent_state_{t+1}

Trained self-supervised against traces from the exact differentiable
execution engine (or sampled from the GPU kernel).

This enables cheap multi-step "mental simulation" in latent space,
uncertainty-aware fallbacks to exact execution, and powerful new
training signals for the latent controller stack.

See the full design:
docs/architecture/JEPA_MACHINE_WORLD_MODEL.md
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class JEWMConfig:
    """Configuration for a JEPA-style Machine World Model."""

    state_dim: int = 128          # dimensionality of the machine state embedding
    action_dim: int = 64          # dimensionality of encoded action (instruction / short program)
    hidden_dim: int = 256
    num_predictor_layers: int = 3
    dropout: float = 0.1
    # Future: multi-step prediction horizon, uncertainty head, etc.
    use_uncertainty: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StateEncoder(nn.Module):
    """
    Encodes machine state features into the world model's latent space.
    Expects the caller to provide a fixed-size feature vector (we use 22 in the v0 prototype).
    """

    def __init__(self, config: JEWMConfig, input_dim: int = 22):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.state_dim),
        )

    def forward(self, raw_state: torch.Tensor) -> torch.Tensor:
        return self.proj(raw_state)


class ActionEncoder(nn.Module):
    """Encodes the 'action' the machine will take next (instruction or short program fragment)."""

    def __init__(self, config: JEWMConfig, input_dim: int = 8):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

    def forward(self, action_features: torch.Tensor) -> torch.Tensor:
        return self.proj(action_features)


class Predictor(nn.Module):
    """
    The heart of the JEPA model: (z_t, action) -> ẑ_{t+1}

    In a proper JEPA setup we would have:
    - A predictor network
    - A target encoder (EMA of state encoder, stop-gradient)
    - A loss that makes predicted representation close to the actual future representation
      (often with variance regularization, covariance, etc.)
    """

    def __init__(self, config: JEWMConfig):
        super().__init__()
        self.config = config
        layers = []
        input_dim = config.state_dim + config.action_dim
        for _ in range(config.num_predictor_layers - 1):
            layers += [nn.Linear(input_dim, config.hidden_dim), nn.SiLU(), nn.Dropout(config.dropout)]
            input_dim = config.hidden_dim
        layers += [nn.Linear(input_dim, config.state_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, state_latent: torch.Tensor, action_latent: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state_latent, action_latent], dim=-1)
        return self.net(x)


class JEWorldModel(nn.Module):
    """
    Full JEPA Machine World Model.

    This is the object you will interact with from ExecutableThoughtHead,
    the latent controller, the coprocessor router, etc.
    """

    def __init__(self, config: Optional[JEWMConfig] = None):
        super().__init__()
        self.config = config or JEWMConfig()
        # v0 prototype uses 22-dim state features and 8-dim actions
        self.state_encoder = StateEncoder(self.config, input_dim=22)
        self.action_encoder = ActionEncoder(self.config, input_dim=8)
        self.predictor = Predictor(self.config)

        # Placeholder for target encoder (will be EMA copy in proper training)
        self.target_state_encoder: Optional[nn.Module] = None

    def encode_state(self, raw_state: torch.Tensor) -> torch.Tensor:
        return self.state_encoder(raw_state)

    def encode_action(self, action_features: torch.Tensor) -> torch.Tensor:
        return self.action_encoder(action_features)

    def predict_next_latent(
        self,
        current_latent: torch.Tensor,
        action_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fast forward prediction in latent space. This is the cheap operation."""
        a = self.encode_action(action_features)
        return self.predictor(current_latent, a)

    # ------------------------------------------------------------------
    # Training hooks (to be filled in Phase 2-3)
    # ------------------------------------------------------------------
    def forward_for_training(
        self,
        current_state: torch.Tensor,
        action: torch.Tensor,
        target_next_state: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns predictions + loss terms.

        In the real training loop you will:
        1. Get actual next state by running the action on DifferentiableEngine / fast path.
        2. Encode current + target next with (possibly stopped) encoders.
        3. Predict with the predictor.
        4. Compute JEPA-style similarity loss + regularizers.
        """
        z = self.encode_state(current_state)
        pred = self.predict_next_latent(z, action)

        out = {"predicted_latent": pred}

        if target_next_state is not None:
            with torch.no_grad():  # target encoder is usually detached / EMA
                z_target = self.encode_state(target_next_state)  # later: use separate target encoder
            # Simple MSE as starting point — replace with proper JEPA loss
            loss = F.mse_loss(pred, z_target)
            out["loss"] = loss
            out["target_latent"] = z_target

        return out


# Convenience factory for quick experiments
def create_small_jewm() -> JEWorldModel:
    """Small model suitable for fast iteration and early integration experiments."""
    cfg = JEWMConfig(state_dim=64, action_dim=32, hidden_dim=128, num_predictor_layers=2)
    return JEWorldModel(cfg)