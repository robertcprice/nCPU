"""Learned latent-state to descriptor-projection head."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from ncpu.self_optimizing.latent_action_policy import _normalized_length, _stable_bucket
from ncpu.self_optimizing.latent_controller_state import LatentControllerState


@dataclass
class LatentDescriptorHeadConfig:
    """Configuration for a learned latent-state descriptor head."""

    numeric_feature_count: int = 24
    hash_bucket_count: int = 16
    hidden_dim: int = 64
    output_dim: int = 16
    dropout: float = 0.0

    @property
    def input_dim(self) -> int:
        return self.numeric_feature_count + self.hash_bucket_count

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def encode_latent_descriptor_features(
    *,
    latent_state: LatentControllerState,
    update_kind: str,
    error_text: str = "",
    candidate_text: str = "",
    config: Optional[LatentDescriptorHeadConfig] = None,
) -> tuple[list[float], dict[str, Any]]:
    """Encode latent controller state for learned descriptor generation."""
    resolved = config or LatentDescriptorHeadConfig()
    vector = [0.0] * resolved.input_dim
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
        _normalized_length(latent_state.hidden_plan or ""),
        _normalized_length(latent_state.repair_plan or ""),
        _normalized_length(latent_state.last_candidate_digest or ""),
        _normalized_length(latent_state.last_failure_summary or ""),
        _normalized_length(error_text),
        _normalized_length(candidate_text),
        1.0 if "verify_failure" in update_kind else 0.0,
        1.0 if "plan" in update_kind else 0.0,
        1.0 if "repair" in update_kind else 0.0,
        1.0 if latent_state.active_strategy else 0.0,
        *latent_state.memory_projection(width=4),
    ]
    for index, value in enumerate(numeric_values):
        vector[index] = value

    hashed_offset = resolved.numeric_feature_count
    for token_source in (
        update_kind,
        latent_state.active_strategy,
        latent_state.hidden_plan,
        latent_state.repair_plan,
        latent_state.last_candidate_digest,
        latent_state.last_failure_summary,
        " ".join(latent_state.failure_patterns),
        " ".join(latent_state.verified_constraints),
        " ".join(latent_state.recent_actions),
        error_text[:256],
        candidate_text[:256],
    ):
        for raw_token in str(token_source or "").lower().split():
            bucket = _stable_bucket(raw_token, bucket_count=resolved.hash_bucket_count)
            vector[hashed_offset + bucket] += 1.0

    hash_total = sum(vector[hashed_offset:])
    if hash_total > 0:
        for index in range(hashed_offset, len(vector)):
            vector[index] /= hash_total

    summary = {
        "update_kind": update_kind,
        "confidence": float(latent_state.confidence),
        "verification_passes": int(latent_state.verification_passes),
        "verification_failures": int(latent_state.verification_failures),
        "recent_actions": list(latent_state.recent_actions[-4:]),
        "memory_projection": latent_state.memory_projection(width=4),
    }
    return vector, summary


class LatentDescriptorHead(nn.Module):
    """Small MLP that predicts a descriptor projection from latent state."""

    def __init__(self, config: Optional[LatentDescriptorHeadConfig] = None):
        super().__init__()
        self.config = config or LatentDescriptorHeadConfig()
        self.network = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class LatentDescriptorGenerator:
    """Runtime wrapper for a learned latent-state descriptor head."""

    def __init__(
        self,
        head: Optional[LatentDescriptorHead] = None,
        *,
        device: str | torch.device = "cpu",
        config: Optional[LatentDescriptorHeadConfig] = None,
    ):
        self.head = head
        self.device = device
        self.config = config or (head.config if head is not None else LatentDescriptorHeadConfig())
        if self.head is not None:
            self.head = self.head.to(device)
            self.head.eval()

    def build_projection(
        self,
        *,
        latent_state: LatentControllerState,
        update_kind: str,
        error_text: str = "",
        candidate_text: str = "",
    ) -> tuple[list[float], dict[str, Any]]:
        feature_vector, feature_summary = encode_latent_descriptor_features(
            latent_state=latent_state,
            update_kind=update_kind,
            error_text=error_text,
            candidate_text=candidate_text,
            config=self.config,
        )
        if self.head is None:
            return [0.0] * self.config.output_dim, feature_summary
        with torch.no_grad():
            inputs = torch.tensor(feature_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
            outputs = self.head(inputs).squeeze(0).detach().cpu().tolist()
        return [float(value) for value in outputs], feature_summary


def load_latent_descriptor_head(
    *,
    path: str | Path,
    device: str | torch.device,
    config: Optional[LatentDescriptorHeadConfig] = None,
) -> LatentDescriptorHead:
    """Load a trained latent descriptor head checkpoint."""
    checkpoint_path = Path(path).expanduser()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    resolved_config = config
    if resolved_config is None and isinstance(payload, dict) and "config" in payload:
        resolved_config = LatentDescriptorHeadConfig(**dict(payload["config"]))
    head = LatentDescriptorHead(resolved_config)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    head.load_state_dict(state_dict)
    head = head.to(device)
    head.eval()
    return head
