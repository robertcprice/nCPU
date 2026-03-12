"""Learned recurrent latent-memory updater for hidden SOME control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from ncpu.self_optimizing.latent_action_policy import _normalized_length, _stable_bucket
from ncpu.self_optimizing.latent_controller_state import LatentControllerState


LATENT_MEMORY_EVENT_LABELS = (
    "think",
    "write",
    "patch",
    "verify",
    "fast_weight_update",
    "descriptor_update",
    "commit",
    "fail",
)


@dataclass
class LatentMemoryHeadConfig:
    """Configuration for a learned recurrent latent-memory updater."""

    numeric_feature_count: int = 32
    hash_bucket_count: int = 16
    hidden_dim: int = 64
    output_dim: int = 16
    dropout: float = 0.0

    @property
    def input_dim(self) -> int:
        return self.numeric_feature_count + self.hash_bucket_count

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _event_flag(event_kind: str, target: str) -> float:
    return 1.0 if event_kind == target else 0.0


def encode_latent_memory_features(
    *,
    latent_state: LatentControllerState,
    workspace: Any,
    event_kind: str,
    response_text: str = "",
    error_text: str = "",
    success: Optional[bool] = None,
    config: Optional[LatentMemoryHeadConfig] = None,
) -> tuple[list[float], dict[str, Any]]:
    """Encode pre-update hidden state plus event context into a dense feature vector."""
    resolved = config or LatentMemoryHeadConfig()
    vector = [0.0] * resolved.input_dim
    max_attempts = max(int(getattr(workspace, "max_generation_attempts", 0) or 0), 1)
    generation_attempts = int(getattr(workspace, "generation_attempts", 0) or 0)
    remaining_attempts = max(max_attempts - generation_attempts, 0)
    status = str(getattr(workspace, "status", "running") or "running")

    numeric_values = [
        float(latent_state.confidence),
        min(latent_state.verification_passes / 8.0, 1.0),
        min(latent_state.verification_failures / 8.0, 1.0),
        min(latent_state.fast_weight_updates_used / 8.0, 1.0),
        min(latent_state.descriptor_updates_used / 8.0, 1.0),
        min(len(latent_state.failure_patterns) / 8.0, 1.0),
        min(len(latent_state.verified_constraints) / 8.0, 1.0),
        min(len(latent_state.recent_actions) / 8.0, 1.0),
        1.0 if latent_state.hidden_plan else 0.0,
        1.0 if latent_state.repair_plan else 0.0,
        1.0 if latent_state.last_failure_summary else 0.0,
        generation_attempts / float(max_attempts),
        remaining_attempts / float(max_attempts),
        1.0 if status == "running" else 0.0,
        1.0 if status == "failed" else 0.0,
        1.0 if status == "committed" else 0.0,
        1.0 if success is True else 0.0,
        1.0 if success is False else 0.0,
        _normalized_length(response_text),
        _normalized_length(error_text),
        _event_flag(event_kind, "think"),
        _event_flag(event_kind, "write"),
        _event_flag(event_kind, "patch"),
        _event_flag(event_kind, "verify"),
        _event_flag(event_kind, "fast_weight_update"),
        _event_flag(event_kind, "descriptor_update"),
        _event_flag(event_kind, "commit"),
        _event_flag(event_kind, "fail"),
        *latent_state.memory_projection(width=4),
    ]
    for index, value in enumerate(numeric_values):
        vector[index] = value

    hashed_offset = resolved.numeric_feature_count
    for token_source in (
        event_kind,
        latent_state.active_strategy,
        latent_state.hidden_plan,
        latent_state.repair_plan,
        latent_state.last_candidate_digest,
        latent_state.last_failure_summary,
        " ".join(latent_state.failure_patterns),
        " ".join(latent_state.verified_constraints),
        " ".join(latent_state.recent_actions),
        response_text[:256],
        error_text[:256],
    ):
        for raw_token in str(token_source or "").lower().split():
            bucket = _stable_bucket(raw_token, bucket_count=resolved.hash_bucket_count)
            vector[hashed_offset + bucket] += 1.0

    hash_total = sum(vector[hashed_offset:])
    if hash_total > 0:
        for index in range(hashed_offset, len(vector)):
            vector[index] /= hash_total

    summary = {
        "event_kind": event_kind,
        "status": status,
        "generation_attempts": generation_attempts,
        "remaining_attempts": remaining_attempts,
        "success": success,
        "confidence": float(latent_state.confidence),
        "memory_projection": latent_state.memory_projection(width=4),
    }
    return vector, summary


class LatentMemoryHead(nn.Module):
    """Small MLP that predicts a recurrent memory delta from hidden event state."""

    def __init__(self, config: Optional[LatentMemoryHeadConfig] = None):
        super().__init__()
        self.config = config or LatentMemoryHeadConfig()
        self.network = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class LatentMemoryUpdater:
    """Runtime wrapper for a learned latent-memory update head."""

    def __init__(
        self,
        head: Optional[LatentMemoryHead] = None,
        *,
        device: str | torch.device = "cpu",
        config: Optional[LatentMemoryHeadConfig] = None,
    ):
        self.head = head
        self.device = device
        self.config = config or (head.config if head is not None else LatentMemoryHeadConfig())
        if self.head is not None:
            self.head = self.head.to(device)
            self.head.eval()

    def build_memory_delta(
        self,
        *,
        latent_state: LatentControllerState,
        workspace: Any,
        event_kind: str,
        response_text: str = "",
        error_text: str = "",
        success: Optional[bool] = None,
    ) -> tuple[list[float], dict[str, Any]]:
        feature_vector, feature_summary = encode_latent_memory_features(
            latent_state=latent_state,
            workspace=workspace,
            event_kind=event_kind,
            response_text=response_text,
            error_text=error_text,
            success=success,
            config=self.config,
        )
        if self.head is None:
            return [0.0] * self.config.output_dim, feature_summary
        with torch.no_grad():
            inputs = torch.tensor(feature_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
            outputs = self.head(inputs).squeeze(0).detach().cpu().tolist()
        return [float(value) for value in outputs], feature_summary


def load_latent_memory_head(
    *,
    path: str | Path,
    device: str | torch.device,
    config: Optional[LatentMemoryHeadConfig] = None,
) -> LatentMemoryHead:
    """Load a trained latent memory head checkpoint."""
    checkpoint_path = Path(path).expanduser()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    resolved_config = config
    if resolved_config is None and isinstance(payload, dict) and "config" in payload:
        resolved_config = LatentMemoryHeadConfig(**dict(payload["config"]))
    head = LatentMemoryHead(resolved_config)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    head.load_state_dict(state_dict)
    head = head.to(device)
    head.eval()
    return head
