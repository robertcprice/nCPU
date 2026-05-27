"""Prepare executable-thought supervision from real SOME trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable, Optional

from ncpu.self_optimizing.controller.hidden_workspace import HiddenWorkspace
from ncpu.self_optimizing.executable_thought_context import (
    build_executable_register_values,
    build_executable_thought_prompt,
)
from ncpu.self_optimizing.latent_heads.controller_state import LatentControllerState
from ncpu.self_optimizing.training.trajectory_dataset import LoadedTrajectory, iter_trajectories


@dataclass
class ExecutableThoughtTrainingExample:
    """One hidden-state executable-thought supervision example."""

    example_id: str
    source_path: str
    task_name: str
    category: str
    update_kind: str
    step_index: int
    prompt_text: str
    register_inputs: list[float]
    target_vector: list[float]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_vector(values: list[float], *, width: int) -> list[float]:
    vector = [0.0] * width
    for index, value in enumerate(values[:width]):
        vector[index] = float(value)
    return vector


def _make_workspace(trajectory: LoadedTrajectory) -> HiddenWorkspace:
    workspace = HiddenWorkspace(
        task_name=trajectory.task_name,
        task_prompt="",
        category=trajectory.category,
        status="running",
        max_generation_attempts=int(trajectory.metadata.get("max_generation_attempts") or trajectory.generation_attempts or 0),
    )
    initial_state = dict(trajectory.metadata.get("initial_latent_state") or {})
    if initial_state:
        workspace.latent_state = LatentControllerState(**initial_state)
    return workspace


def _current_failure_summary(workspace: HiddenWorkspace) -> str:
    return str(
        workspace.metadata.get("last_failure_summary")
        or workspace.last_error
        or ""
    )


def _advance_workspace(workspace: HiddenWorkspace, step: dict[str, Any]) -> None:
    action = str(step.get("action") or "")
    metadata = dict(step.get("metadata") or {})
    if action in {"write", "patch"}:
        workspace.generation_attempts += 1
        workspace.set_candidate(str(step.get("response_text") or ""))
    elif action == "verify":
        workspace.record_verification(
            success=bool(step.get("success")),
            verification=metadata or None,
            error=step.get("error"),
        )
        if step.get("success") is False:
            workspace.metadata["last_failure_summary"] = str(step.get("response_text") or step.get("error") or "")
        else:
            workspace.metadata["last_failure_summary"] = ""
    elif action == "fail":
        workspace.fail(str(step.get("error") or step.get("response_text") or "failed"))
    elif action == "commit":
        workspace.commit(
            workspace.candidate_solution,
            verified=bool(metadata.get("verified", False)),
            metadata=metadata,
        )

    state_payload = dict(step.get("latent_state") or {})
    if state_payload:
        workspace.latent_state = LatentControllerState(**state_payload)


def build_executable_thought_training_examples(
    trajectory: LoadedTrajectory,
    *,
    require_verified_commit: bool = True,
    num_registers: int = 8,
    output_dim: int = 16,
) -> list[ExecutableThoughtTrainingExample]:
    """Build executable-thought supervision from successful descriptor updates."""
    if require_verified_commit and not trajectory.committed_verified:
        return []

    workspace = _make_workspace(trajectory)
    examples: list[ExecutableThoughtTrainingExample] = []

    for step in trajectory.steps:
        action = str(step.get("action") or "")
        if action == "descriptor_update" and step.get("success") is True:
            metadata = dict(step.get("metadata") or {})
            descriptor = dict(metadata.get("adaptation_descriptor") or {})
            signal_projection = descriptor.get("signal_projection") or []
            if isinstance(signal_projection, list) and signal_projection:
                update_kind = str(metadata.get("kind") or descriptor.get("update_kind") or "descriptor_update")
                error_text = _current_failure_summary(workspace) if "verify_failure" in update_kind else ""
                candidate_text = workspace.candidate_solution
                prompt_text = build_executable_thought_prompt(
                    task_name=trajectory.task_name,
                    update_kind=update_kind,
                    latent_state=workspace.latent_state,
                    error_text=error_text,
                    candidate_text=candidate_text,
                )
                examples.append(
                    ExecutableThoughtTrainingExample(
                        example_id=(
                            f"{Path(trajectory.source_path).stem}:executable_thought:"
                            f"{int(step.get('step_index') or step.get('index') or len(examples) + 1)}"
                        ),
                        source_path=trajectory.source_path,
                        task_name=trajectory.task_name,
                        category=trajectory.category,
                        update_kind=update_kind,
                        step_index=int(step.get("step_index") or step.get("index") or 0),
                        prompt_text=prompt_text,
                        register_inputs=build_executable_register_values(
                            latent_state=workspace.latent_state,
                            error_text=error_text,
                            candidate_text=candidate_text,
                            num_registers=num_registers,
                        ),
                        target_vector=_normalize_vector(signal_projection, width=output_dim),
                        metadata={
                            "step_metadata": metadata,
                            "candidate_text": candidate_text[:512],
                            "error_text": error_text[:512],
                        },
                    )
                )
        _advance_workspace(workspace, step)

    return examples


def build_executable_thought_dataset(
    root: str | Path,
    *,
    require_verified_commit: bool = True,
    num_registers: int = 8,
    output_dim: int = 16,
) -> list[ExecutableThoughtTrainingExample]:
    dataset: list[ExecutableThoughtTrainingExample] = []
    for trajectory in iter_trajectories(root):
        dataset.extend(
            build_executable_thought_training_examples(
                trajectory,
                require_verified_commit=require_verified_commit,
                num_registers=num_registers,
                output_dim=output_dim,
            )
        )
    return dataset


def _split_by_source(
    items: list[ExecutableThoughtTrainingExample],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[ExecutableThoughtTrainingExample], list[ExecutableThoughtTrainingExample]]:
    grouped: dict[str, list[ExecutableThoughtTrainingExample]] = {}
    for item in items:
        grouped.setdefault(item.source_path, []).append(item)
    source_paths = list(grouped)
    rng = random.Random(seed)
    rng.shuffle(source_paths)
    if len(source_paths) <= 1:
        return items, []
    num_val = max(1, int(round(len(source_paths) * val_ratio)))
    num_val = min(num_val, len(source_paths) - 1)
    val_sources = set(source_paths[:num_val])
    train: list[ExecutableThoughtTrainingExample] = []
    val: list[ExecutableThoughtTrainingExample] = []
    for source_path, group in grouped.items():
        if source_path in val_sources:
            val.extend(group)
        else:
            train.extend(group)
    return train, val


def write_executable_thought_dataset(
    examples: Iterable[ExecutableThoughtTrainingExample],
    path: str | Path,
) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict()) + "\n")
            count += 1
    return count


def build_executable_thought_training_bundle(
    trajectory_root: str | Path,
    output_dir: str | Path,
    *,
    require_verified_commit: bool = True,
    num_registers: int = 8,
    output_dim: int = 16,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    examples = build_executable_thought_dataset(
        trajectory_root,
        require_verified_commit=require_verified_commit,
        num_registers=num_registers,
        output_dim=output_dim,
    )
    train_examples, val_examples = _split_by_source(examples, val_ratio=val_ratio, seed=seed)
    train_path = output_root / "executable_thought_train.jsonl"
    val_path = output_root / "executable_thought_val.jsonl"
    train_count = write_executable_thought_dataset(train_examples, train_path)
    val_count = write_executable_thought_dataset(val_examples, val_path)
    return {
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_examples": train_count,
        "val_examples": val_count,
        "train_source_files": len({example.source_path for example in train_examples}),
        "val_source_files": len({example.source_path for example in val_examples}),
        "num_registers": num_registers,
        "output_dim": output_dim,
    }


__all__ = [
    "ExecutableThoughtTrainingExample",
    "build_executable_thought_training_examples",
    "build_executable_thought_dataset",
    "write_executable_thought_dataset",
    "build_executable_thought_training_bundle",
]
