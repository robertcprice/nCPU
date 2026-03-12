"""Hidden workspace state for buffered internal inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from ncpu.self_optimizing.latent_controller_state import LatentControllerState


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class WorkspaceStep:
    """Single hidden step taken during buffered inference."""

    index: int
    action: str
    prompt: str
    response_text: str
    success: Optional[bool] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_utc_now)


@dataclass
class HiddenWorkspace:
    """Internal state visible to the controller but not the end user."""

    task_name: str
    task_prompt: str
    category: str = "coding"
    status: str = "initialized"
    candidate_solution: str = ""
    committed_output: Optional[str] = None
    committed_verified: bool = False
    generation_attempts: int = 0
    max_generation_attempts: int = 0
    last_error: Optional[str] = None
    last_verification: Optional[dict[str, Any]] = None
    latent_state: LatentControllerState = field(default_factory=LatentControllerState)
    steps: list[WorkspaceStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def record_step(
        self,
        *,
        action: str,
        prompt: str,
        response_text: str,
        success: Optional[bool] = None,
        error: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> WorkspaceStep:
        step = WorkspaceStep(
            index=len(self.steps) + 1,
            action=action,
            prompt=prompt,
            response_text=response_text,
            success=success,
            error=error,
            metadata=dict(metadata or {}),
        )
        self.steps.append(step)
        self.updated_at = step.timestamp
        return step

    def set_candidate(self, candidate_solution: str) -> None:
        self.candidate_solution = candidate_solution
        self.updated_at = _utc_now()

    def record_verification(
        self,
        *,
        success: bool,
        verification: Optional[dict[str, Any]],
        error: Optional[str],
    ) -> None:
        self.last_verification = verification
        self.last_error = error if not success else None
        self.updated_at = _utc_now()

    def commit(
        self,
        output: str,
        *,
        verified: bool,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.committed_output = output
        self.committed_verified = verified
        self.status = "committed"
        if metadata:
            self.metadata.update(metadata)
        self.updated_at = _utc_now()

    def fail(self, error: str) -> None:
        self.status = "failed"
        self.last_error = error
        self.updated_at = _utc_now()

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "task_prompt": self.task_prompt,
            "category": self.category,
            "status": self.status,
            "candidate_solution": self.candidate_solution,
            "committed_output": self.committed_output,
            "committed_verified": self.committed_verified,
            "generation_attempts": self.generation_attempts,
            "max_generation_attempts": self.max_generation_attempts,
            "last_error": self.last_error,
            "last_verification": self.last_verification,
            "latent_state": self.latent_state.to_dict(),
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "steps": [
                {
                    "index": step.index,
                    "action": step.action,
                    "prompt": step.prompt,
                    "response_text": step.response_text,
                    "success": step.success,
                    "error": step.error,
                    "metadata": step.metadata,
                    "timestamp": step.timestamp,
                }
                for step in self.steps
            ],
        }
