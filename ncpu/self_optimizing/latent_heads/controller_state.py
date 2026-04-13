"""Structured latent controller state for hidden SOME inference."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


def _shorten(text: str, *, limit: int = 160) -> str:
    normalized = " ".join(text.strip().split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


@dataclass
class LatentControllerState:
    """Compact structured state carried across hidden inference steps."""

    active_strategy: Optional[str] = None
    hidden_plan: Optional[str] = None
    repair_plan: Optional[str] = None
    last_candidate_digest: Optional[str] = None
    last_failure_summary: Optional[str] = None
    verified_constraints: list[str] = field(default_factory=list)
    failure_patterns: list[str] = field(default_factory=list)
    recent_actions: list[str] = field(default_factory=list)
    fast_weight_updates_used: int = 0
    descriptor_updates_used: int = 0
    verification_passes: int = 0
    verification_failures: int = 0
    confidence: float = 0.0
    memory_vector: list[float] = field(default_factory=lambda: [0.0] * 16)
    memory_updates: int = 0
    enable_heuristic_memory_updates: bool = True

    def _ensure_memory_width(self, width: int = 16) -> None:
        if len(self.memory_vector) < width:
            self.memory_vector.extend([0.0] * (width - len(self.memory_vector)))
        elif len(self.memory_vector) > width:
            self.memory_vector = self.memory_vector[:width]

    def _text_signal(self, text: str, *, width: int = 16) -> list[float]:
        self._ensure_memory_width(width)
        signal = [0.0] * width
        normalized = " ".join(str(text or "").strip().split()).lower()
        if not normalized:
            return signal
        for token in normalized.split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % width
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            magnitude = 0.25 + (digest[3] / 255.0)
            signal[index] += sign * magnitude
        return signal

    def _blend_memory(self, signal: list[float], *, scale: float, momentum: float = 0.82) -> None:
        self._ensure_memory_width(len(signal) or 16)
        if not self.enable_heuristic_memory_updates:
            return
        if not signal:
            return
        signal_norm = math.sqrt(sum(value * value for value in signal))
        normalized = signal if signal_norm <= 1e-8 else [value / signal_norm for value in signal]
        next_memory = []
        for current, update in zip(self.memory_vector, normalized):
            next_memory.append(max(-1.0, min(1.0, current * momentum + update * scale)))
        self.memory_vector = next_memory
        self.memory_updates += 1

    def apply_memory_delta(
        self,
        delta: list[float],
        *,
        scale: float = 1.0,
        clamp: float = 1.0,
    ) -> None:
        self._ensure_memory_width(len(delta) or 16)
        if not delta:
            return
        next_memory = list(self.memory_vector)
        for index, value in enumerate(delta[: len(next_memory)]):
            next_memory[index] = max(-clamp, min(clamp, next_memory[index] + float(value) * scale))
        self.memory_vector = next_memory
        self.memory_updates += 1

    def memory_projection(self, *, width: int = 4) -> list[float]:
        self._ensure_memory_width()
        width = max(1, min(width, len(self.memory_vector)))
        chunk_size = max(1, math.ceil(len(self.memory_vector) / width))
        pooled: list[float] = []
        for start in range(0, len(self.memory_vector), chunk_size):
            chunk = self.memory_vector[start : start + chunk_size]
            pooled.append(sum(chunk) / max(len(chunk), 1))
        if len(pooled) < width:
            pooled.extend([0.0] * (width - len(pooled)))
        return pooled[:width]

    def memory_feature_vector(
        self,
        *,
        width: int = 8,
        include_stats: bool = True,
    ) -> list[float]:
        self._ensure_memory_width()
        pooled = self.memory_projection(width=width)
        if not include_stats:
            return pooled

        if not self.memory_vector:
            return pooled + [0.0, 0.0, 0.0, 0.0]

        mean_value = sum(self.memory_vector) / len(self.memory_vector)
        mean_abs = sum(abs(value) for value in self.memory_vector) / len(self.memory_vector)
        max_abs = max(abs(value) for value in self.memory_vector)
        update_norm = min(self.memory_updates / 16.0, 1.0)
        return pooled + [mean_value, mean_abs, max_abs, update_norm]

    def record_action(self, action: str) -> None:
        self.recent_actions.append(action)
        if len(self.recent_actions) > 8:
            self.recent_actions = self.recent_actions[-8:]
        self._blend_memory(self._text_signal(f"action:{action}"), scale=0.06)

    def record_plan(self, plan_text: str, *, kind: str) -> None:
        if kind == "repair_plan":
            self.repair_plan = _shorten(plan_text)
        else:
            self.hidden_plan = _shorten(plan_text)
        if not self.active_strategy:
            self.active_strategy = _shorten(plan_text, limit=96)
        self.confidence = min(self.confidence + 0.05, 0.95)
        self._blend_memory(self._text_signal(f"{kind}:{plan_text}"), scale=0.14)

    def record_candidate(self, candidate: str) -> None:
        self.last_candidate_digest = _shorten(candidate)
        self._blend_memory(self._text_signal(candidate), scale=0.08)

    def record_verification(
        self,
        *,
        success: bool,
        error: Optional[str],
        verification: Optional[dict[str, Any]],
    ) -> None:
        if success:
            self.verification_passes += 1
            self.confidence = min(self.confidence + 0.2, 1.0)
            if verification:
                for key in ("entry_point", "task_id", "status"):
                    value = verification.get(key)
                    if value is not None:
                        self._add_unique(self.verified_constraints, f"{key}={value}")
            self._blend_memory(self._text_signal("verification:pass"), scale=0.1)
        else:
            self.verification_failures += 1
            summary = error or (verification or {}).get("error") or "verification failed"
            self.last_failure_summary = _shorten(summary)
            self._add_unique(self.failure_patterns, _shorten(summary))
            self.confidence = max(self.confidence - 0.15, 0.0)
            self._blend_memory(self._text_signal(f"verification:fail {summary}"), scale=0.16)

    def record_fast_weight_update(self, *, success: bool, kind: str) -> None:
        if success:
            self.fast_weight_updates_used += 1
            self._add_unique(self.verified_constraints, f"fast_weight_update={kind}")
            self._blend_memory(self._text_signal(f"fast_weight:{kind}"), scale=0.08)

    def record_descriptor_update(self, *, success: bool, kind: str) -> None:
        if success:
            self.descriptor_updates_used += 1
            self._add_unique(self.verified_constraints, f"descriptor_update={kind}")
            self._blend_memory(self._text_signal(f"descriptor:{kind}"), scale=0.1)

    def to_dict(self) -> dict[str, Any]:
        self._ensure_memory_width()
        return asdict(self)

    def to_prompt_summary(self) -> str:
        lines: list[str] = []
        if self.active_strategy:
            lines.append(f"strategy: {self.active_strategy}")
        if self.hidden_plan:
            lines.append(f"hidden_plan: {self.hidden_plan}")
        if self.repair_plan:
            lines.append(f"repair_plan: {self.repair_plan}")
        if self.last_candidate_digest:
            lines.append(f"last_candidate: {self.last_candidate_digest}")
        if self.last_failure_summary:
            lines.append(f"last_failure: {self.last_failure_summary}")
        if self.verified_constraints:
            lines.append("verified_constraints: " + "; ".join(self.verified_constraints[-4:]))
        if self.failure_patterns:
            lines.append("failure_patterns: " + "; ".join(self.failure_patterns[-4:]))
        if self.recent_actions:
            lines.append("recent_actions: " + ", ".join(self.recent_actions[-6:]))
        lines.append(
            "latent_memory: "
            + ", ".join(f"{value:.2f}" for value in self.memory_projection(width=4))
            + f" | updates={self.memory_updates}"
        )
        lines.append(
            "verification: "
            f"{self.verification_passes} pass / {self.verification_failures} fail | "
            f"fast_weight_updates={self.fast_weight_updates_used} | "
            f"descriptor_updates={self.descriptor_updates_used} | "
            f"confidence={self.confidence:.2f}"
        )
        return "\n".join(lines) if lines else "(no latent state yet)"

    def _add_unique(self, target: list[str], item: str) -> None:
        if item in target:
            return
        target.append(item)
        if len(target) > 8:
            del target[:-8]
