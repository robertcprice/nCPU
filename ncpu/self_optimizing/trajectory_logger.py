"""JSONL trajectory logging for buffered internal inference."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ncpu.self_optimizing.hidden_workspace import HiddenWorkspace, WorkspaceStep


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrajectoryLogger:
    """Persists hidden controller trajectories for later distillation."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else None

    def _append(self, event: dict[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    def log_workspace_initialized(self, workspace: HiddenWorkspace) -> None:
        self._append(
            {
                "timestamp": _utc_now(),
                "event": "workspace_initialized",
                "task_name": workspace.task_name,
                "category": workspace.category,
                "status": workspace.status,
                "max_generation_attempts": workspace.max_generation_attempts,
                "latent_state": workspace.latent_state.to_dict(),
            }
        )

    def log_step(self, workspace: HiddenWorkspace, step: WorkspaceStep) -> None:
        self._append(
            {
                "timestamp": step.timestamp,
                "event": "workspace_step",
                "task_name": workspace.task_name,
                "status": workspace.status,
                "step_index": step.index,
                "action": step.action,
                "success": step.success,
                "error": step.error,
                "metadata": step.metadata,
                "latent_state": workspace.latent_state.to_dict(),
                "prompt": step.prompt,
                "response_text": step.response_text,
            }
        )

    def log_commit(self, workspace: HiddenWorkspace) -> None:
        self._append(
            {
                "timestamp": _utc_now(),
                "event": "workspace_committed",
                "task_name": workspace.task_name,
                "status": workspace.status,
                "committed_verified": workspace.committed_verified,
                "generation_attempts": workspace.generation_attempts,
                "last_error": workspace.last_error,
                "latent_state": workspace.latent_state.to_dict(),
                "committed_output": workspace.committed_output,
            }
        )

    def log_failed(self, workspace: HiddenWorkspace) -> None:
        self._append(
            {
                "timestamp": _utc_now(),
                "event": "workspace_failed",
                "task_name": workspace.task_name,
                "status": workspace.status,
                "generation_attempts": workspace.generation_attempts,
                "last_error": workspace.last_error,
                "latent_state": workspace.latent_state.to_dict(),
            }
        )
