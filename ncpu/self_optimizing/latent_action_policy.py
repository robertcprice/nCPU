"""Latent action-selection policy for hidden SOME control flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from ncpu.self_optimizing.latent_controller_state import LatentControllerState


LATENT_ACTION_LABELS = ("think", "write", "patch", "commit", "fail")


@dataclass
class LatentActionHeadConfig:
    """Configuration for the latent action-selection head."""

    numeric_feature_count: int = 28
    hash_bucket_count: int = 16
    hidden_dim: int = 64
    dropout: float = 0.0
    action_labels: tuple[str, ...] = LATENT_ACTION_LABELS

    @property
    def input_dim(self) -> int:
        return self.numeric_feature_count + self.hash_bucket_count

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["action_labels"] = list(self.action_labels)
        return payload


@dataclass
class LatentActionDecision:
    """One latent policy decision."""

    action: str
    source: str
    scores: dict[str, float]
    confidence: float
    feature_summary: dict[str, Any]


def _stable_bucket(value: str, *, bucket_count: int) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(bucket_count, 1)


def _normalized_length(value: str, *, scale: float = 256.0) -> float:
    return min(len(str(value or "")) / scale, 1.0)


def encode_latent_action_features(
    *,
    latent_state: LatentControllerState,
    workspace: Any,
    allowed_actions: list[str],
    config: Optional[LatentActionHeadConfig] = None,
) -> tuple[list[float], dict[str, Any]]:
    """Turn latent controller state plus workspace status into a dense feature vector."""
    resolved = config or LatentActionHeadConfig()
    vector = [0.0] * resolved.input_dim
    max_attempts = max(int(getattr(workspace, "max_generation_attempts", 0) or 0), 1)
    generation_attempts = int(getattr(workspace, "generation_attempts", 0) or 0)
    remaining_attempts = max(max_attempts - generation_attempts, 0)
    status = str(getattr(workspace, "status", "running") or "running")
    candidate_solution = str(getattr(workspace, "candidate_solution", "") or "")
    last_error = str(getattr(workspace, "last_error", "") or "")

    numeric_values = [
        float(latent_state.confidence),
        min(latent_state.verification_passes / 8.0, 1.0),
        min(latent_state.verification_failures / 8.0, 1.0),
        min(latent_state.fast_weight_updates_used / 8.0, 1.0),
        min(latent_state.descriptor_updates_used / 8.0, 1.0),
        generation_attempts / float(max_attempts),
        remaining_attempts / float(max_attempts),
        1.0 if latent_state.hidden_plan else 0.0,
        1.0 if latent_state.repair_plan else 0.0,
        1.0 if latent_state.last_failure_summary else 0.0,
        min(len(latent_state.failure_patterns) / 8.0, 1.0),
        min(len(latent_state.verified_constraints) / 8.0, 1.0),
        min(len(latent_state.recent_actions) / 8.0, 1.0),
        _normalized_length(candidate_solution),
        _normalized_length(last_error),
        1.0 if getattr(workspace, "committed_verified", False) else 0.0,
        1.0 if status == "running" else 0.0,
        1.0 if status == "failed" else 0.0,
        1.0 if status == "committed" else 0.0,
        1.0 if "think" in allowed_actions else 0.0,
        1.0 if "write" in allowed_actions else 0.0,
        1.0 if "patch" in allowed_actions else 0.0,
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
        " ".join(latent_state.recent_actions),
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
        "generation_attempts": generation_attempts,
        "remaining_attempts": remaining_attempts,
        "status": status,
        "allowed_actions": list(allowed_actions),
        "confidence": float(latent_state.confidence),
        "failure_patterns": list(latent_state.failure_patterns[-4:]),
        "recent_actions": list(latent_state.recent_actions[-6:]),
        "memory_projection": latent_state.memory_projection(width=4),
    }
    return vector, summary


class LatentActionHead(nn.Module):
    """Small MLP over latent controller features."""

    def __init__(self, config: Optional[LatentActionHeadConfig] = None):
        super().__init__()
        self.config = config or LatentActionHeadConfig()
        self.network = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, len(self.config.action_labels)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class LatentActionPolicy:
    """Hybrid heuristic + learned latent action policy."""

    def __init__(
        self,
        head: Optional[LatentActionHead] = None,
        *,
        device: str | torch.device = "cpu",
        config: Optional[LatentActionHeadConfig] = None,
    ):
        self.head = head
        self.device = device
        self.config = config or (head.config if head is not None else LatentActionHeadConfig())
        if self.head is not None:
            self.head = self.head.to(device)
            self.head.eval()

    def choose_action(
        self,
        *,
        workspace: Any,
        allowed_actions: list[str],
        fallback: str,
    ) -> LatentActionDecision:
        feature_vector, feature_summary = encode_latent_action_features(
            latent_state=workspace.latent_state,
            workspace=workspace,
            allowed_actions=allowed_actions,
            config=self.config,
        )
        heuristic_logits = self._heuristic_logits(
            feature_summary=feature_summary,
            workspace=workspace,
            allowed_actions=allowed_actions,
        )
        logits = dict(heuristic_logits)

        if self.head is not None:
            with torch.no_grad():
                inputs = torch.tensor(feature_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
                learned_logits = self.head(inputs).squeeze(0).detach().cpu().tolist()
            for action, score in zip(self.config.action_labels, learned_logits):
                logits[action] = logits.get(action, 0.0) + float(score)

        masked_scores = {
            action: (logits.get(action, -1e9) if action in allowed_actions else -1e9)
            for action in self.config.action_labels
        }
        action = max(masked_scores, key=masked_scores.get)
        if masked_scores[action] <= -1e8:
            action = fallback
            source = "latent_action_policy_fallback"
        else:
            source = "latent_action_policy"
        probabilities = self._softmax(masked_scores, allowed_actions=allowed_actions)
        confidence = probabilities.get(action, 0.0)
        return LatentActionDecision(
            action=action,
            source=source,
            scores=probabilities,
            confidence=confidence,
            feature_summary=feature_summary,
        )

    def _heuristic_logits(
        self,
        *,
        feature_summary: dict[str, Any],
        workspace: Any,
        allowed_actions: list[str],
    ) -> dict[str, float]:
        logits = {action: 0.0 for action in self.config.action_labels}
        generation_attempts = int(feature_summary["generation_attempts"])
        remaining_attempts = int(feature_summary["remaining_attempts"])
        recent_actions = set(feature_summary["recent_actions"])
        failure_patterns = feature_summary["failure_patterns"]

        if "think" in allowed_actions and not workspace.latent_state.hidden_plan and generation_attempts == 0:
            logits["think"] += 0.7
        if "write" in allowed_actions and generation_attempts == 0:
            logits["write"] += 0.6
        if "patch" in allowed_actions and failure_patterns:
            logits["patch"] += 1.0
        if "think" in allowed_actions and failure_patterns and "patch" in recent_actions:
            logits["think"] += 0.45
        if "commit" in allowed_actions and workspace.last_verification:
            logits["commit"] += 1.2 if workspace.committed_verified or workspace.last_error is None else 0.2
        if "fail" in allowed_actions and remaining_attempts == 0:
            logits["fail"] += 1.4
        if "fail" in allowed_actions and generation_attempts > 0 and not failure_patterns:
            logits["fail"] -= 0.3
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
            action: torch.exp(torch.tensor(value - best, dtype=torch.float32)).item()
            for action, value in allowed_scores.items()
        }
        total = sum(exps.values()) or 1.0
        return {action: value / total for action, value in exps.items()}


def load_latent_action_head(
    *,
    path: str | Path,
    device: str | torch.device,
    config: Optional[LatentActionHeadConfig] = None,
) -> LatentActionHead:
    """Load a trained latent action head checkpoint."""
    checkpoint_path = Path(path).expanduser()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    resolved_config = config
    if resolved_config is None and isinstance(payload, dict) and "config" in payload:
        config_payload = dict(payload["config"])
        if "action_labels" in config_payload:
            config_payload["action_labels"] = tuple(config_payload["action_labels"])
        resolved_config = LatentActionHeadConfig(**config_payload)
    head = LatentActionHead(resolved_config)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    head.load_state_dict(state_dict)
    head = head.to(device)
    head.eval()
    return head
