"""Latent halt/commit policy for hidden SOME control flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from ncpu.self_optimizing.latent_action_policy import _normalized_length, _stable_bucket
from ncpu.self_optimizing.latent_controller_state import LatentControllerState


LATENT_HALT_LABELS = ("continue", "commit", "fail")


@dataclass
class LatentHaltHeadConfig:
    """Configuration for the latent halt/commit head."""

    numeric_feature_count: int = 24
    hash_bucket_count: int = 12
    hidden_dim: int = 48
    dropout: float = 0.0
    action_labels: tuple[str, ...] = LATENT_HALT_LABELS

    @property
    def input_dim(self) -> int:
        return self.numeric_feature_count + self.hash_bucket_count

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["action_labels"] = list(self.action_labels)
        return payload


@dataclass
class LatentHaltDecision:
    """One latent halt decision."""

    action: str
    source: str
    scores: dict[str, float]
    confidence: float
    feature_summary: dict[str, Any]


def encode_latent_halt_features(
    *,
    latent_state: LatentControllerState,
    workspace: Any,
    verification_success: bool,
    verification_error: str = "",
    allowed_actions: list[str],
    config: Optional[LatentHaltHeadConfig] = None,
) -> tuple[list[float], dict[str, Any]]:
    """Encode hidden state plus verification outcome for halt decisions."""
    resolved = config or LatentHaltHeadConfig()
    vector = [0.0] * resolved.input_dim
    max_attempts = max(int(getattr(workspace, "max_generation_attempts", 0) or 0), 1)
    generation_attempts = int(getattr(workspace, "generation_attempts", 0) or 0)
    remaining_attempts = max(max_attempts - generation_attempts, 0)
    status = str(getattr(workspace, "status", "running") or "running")
    candidate_solution = str(getattr(workspace, "candidate_solution", "") or "")
    last_error = str(verification_error or getattr(workspace, "last_error", "") or "")

    numeric_values = [
        float(latent_state.confidence),
        min(latent_state.verification_passes / 8.0, 1.0),
        min(latent_state.verification_failures / 8.0, 1.0),
        min(latent_state.fast_weight_updates_used / 8.0, 1.0),
        min(latent_state.descriptor_updates_used / 8.0, 1.0),
        generation_attempts / float(max_attempts),
        remaining_attempts / float(max_attempts),
        1.0 if verification_success else 0.0,
        0.0 if verification_success else 1.0,
        1.0 if latent_state.hidden_plan else 0.0,
        1.0 if latent_state.repair_plan else 0.0,
        min(len(latent_state.failure_patterns) / 8.0, 1.0),
        min(len(latent_state.verified_constraints) / 8.0, 1.0),
        min(len(latent_state.recent_actions) / 8.0, 1.0),
        _normalized_length(candidate_solution),
        _normalized_length(last_error),
        1.0 if status == "running" else 0.0,
        1.0 if "continue" in allowed_actions else 0.0,
        1.0 if "commit" in allowed_actions else 0.0,
        1.0 if "fail" in allowed_actions else 0.0,
        *latent_state.memory_projection(width=4),
    ]
    for index, value in enumerate(numeric_values):
        vector[index] = value

    hashed_offset = resolved.numeric_feature_count
    for token_source in (
        latent_state.active_strategy,
        latent_state.hidden_plan,
        latent_state.repair_plan,
        latent_state.last_candidate_digest,
        latent_state.last_failure_summary,
        " ".join(latent_state.failure_patterns),
        candidate_solution[:256],
        last_error[:256],
    ):
        for raw_token in str(token_source or "").lower().split():
            bucket = _stable_bucket(raw_token, bucket_count=resolved.hash_bucket_count)
            vector[hashed_offset + bucket] += 1.0

    hash_total = sum(vector[hashed_offset:])
    if hash_total > 0:
        for index in range(hashed_offset, len(vector)):
            vector[index] /= hash_total

    summary = {
        "verification_success": verification_success,
        "generation_attempts": generation_attempts,
        "remaining_attempts": remaining_attempts,
        "allowed_actions": list(allowed_actions),
        "confidence": float(latent_state.confidence),
        "status": status,
        "failure_patterns": list(latent_state.failure_patterns[-4:]),
        "memory_projection": latent_state.memory_projection(width=4),
    }
    return vector, summary


class LatentHaltHead(nn.Module):
    """Small MLP over latent halt features."""

    def __init__(self, config: Optional[LatentHaltHeadConfig] = None):
        super().__init__()
        self.config = config or LatentHaltHeadConfig()
        self.network = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, len(self.config.action_labels)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class LatentHaltPolicy:
    """Hybrid heuristic + learned halt policy."""

    def __init__(
        self,
        head: Optional[LatentHaltHead] = None,
        *,
        device: str | torch.device = "cpu",
        config: Optional[LatentHaltHeadConfig] = None,
    ):
        self.head = head
        self.device = device
        self.config = config or (head.config if head is not None else LatentHaltHeadConfig())
        if self.head is not None:
            self.head = self.head.to(device)
            self.head.eval()

    def choose_halt_action(
        self,
        *,
        workspace: Any,
        verification_success: bool,
        verification_error: str = "",
        allowed_actions: list[str],
        fallback: str,
    ) -> LatentHaltDecision:
        feature_vector, feature_summary = encode_latent_halt_features(
            latent_state=workspace.latent_state,
            workspace=workspace,
            verification_success=verification_success,
            verification_error=verification_error,
            allowed_actions=allowed_actions,
            config=self.config,
        )
        logits = self._heuristic_logits(
            workspace=workspace,
            verification_success=verification_success,
            allowed_actions=allowed_actions,
        )
        if self.head is not None:
            with torch.no_grad():
                inputs = torch.tensor(feature_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
                learned_logits = self.head(inputs).squeeze(0).detach().cpu().tolist()
            for action, score in zip(self.config.action_labels, learned_logits):
                logits[action] = logits.get(action, 0.0) + float(score)

        masked = {
            action: (logits.get(action, -1e9) if action in allowed_actions else -1e9)
            for action in self.config.action_labels
        }
        action = max(masked, key=masked.get)
        if masked[action] <= -1e8:
            action = fallback
            source = "latent_halt_policy_fallback"
        else:
            source = "latent_halt_policy"
        probabilities = self._softmax(masked, allowed_actions=allowed_actions)
        return LatentHaltDecision(
            action=action,
            source=source,
            scores=probabilities,
            confidence=probabilities.get(action, 0.0),
            feature_summary=feature_summary,
        )

    def _heuristic_logits(
        self,
        *,
        workspace: Any,
        verification_success: bool,
        allowed_actions: list[str],
    ) -> dict[str, float]:
        logits = {label: 0.0 for label in self.config.action_labels}
        remaining_attempts = max(
            int(getattr(workspace, "max_generation_attempts", 0) or 0)
            - int(getattr(workspace, "generation_attempts", 0) or 0),
            0,
        )
        if verification_success and "commit" in allowed_actions:
            logits["commit"] += 1.5
        if (not verification_success) and "continue" in allowed_actions and remaining_attempts > 0:
            logits["continue"] += 1.2
        if (not verification_success) and "fail" in allowed_actions and remaining_attempts == 0:
            logits["fail"] += 1.5
        if verification_success and "continue" in allowed_actions:
            logits["continue"] += 0.2
        return logits

    def _softmax(
        self,
        scores: dict[str, float],
        *,
        allowed_actions: list[str],
    ) -> dict[str, float]:
        allowed_scores = {action: scores[action] for action in allowed_actions if action in scores}
        if not allowed_scores:
            return {}
        best = max(allowed_scores.values())
        exps = {
            action: torch.exp(torch.tensor(score - best, dtype=torch.float32)).item()
            for action, score in allowed_scores.items()
        }
        total = sum(exps.values()) or 1.0
        return {action: value / total for action, value in exps.items()}


def load_latent_halt_head(
    *,
    path: str | Path,
    device: str | torch.device,
    config: Optional[LatentHaltHeadConfig] = None,
) -> LatentHaltHead:
    """Load a trained latent halt head checkpoint."""
    checkpoint_path = Path(path).expanduser()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    resolved_config = config
    if resolved_config is None and isinstance(payload, dict) and "config" in payload:
        config_payload = dict(payload["config"])
        if "action_labels" in config_payload:
            config_payload["action_labels"] = tuple(config_payload["action_labels"])
        resolved_config = LatentHaltHeadConfig(**config_payload)
    head = LatentHaltHead(resolved_config)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    head.load_state_dict(state_dict)
    head = head.to(device)
    head.eval()
    return head
