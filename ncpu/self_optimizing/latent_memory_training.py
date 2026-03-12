"""Prepare and train learned recurrent latent-memory heads from SOME trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ncpu.self_optimizing.hidden_workspace import HiddenWorkspace
from ncpu.self_optimizing.latent_controller_state import LatentControllerState
from ncpu.self_optimizing.latent_memory_head import (
    LATENT_MEMORY_EVENT_LABELS,
    LatentMemoryHead,
    LatentMemoryHeadConfig,
    encode_latent_memory_features,
)
from ncpu.self_optimizing.trajectory_dataset import LoadedTrajectory, iter_trajectories


@dataclass
class LatentMemoryTrainingExample:
    """One latent-state + event to memory-delta supervision example."""

    example_id: str
    source_path: str
    task_name: str
    category: str
    event_kind: str
    step_index: int
    feature_vector: list[float]
    target_vector: list[float]
    feature_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_vector(values: list[float], *, width: int) -> list[float]:
    vector = [0.0] * width
    for index, value in enumerate(values[:width]):
        vector[index] = float(value)
    return vector


def _subtract_vectors(current: list[float], previous: list[float], *, width: int) -> list[float]:
    delta = [0.0] * width
    for index in range(width):
        current_value = float(current[index]) if index < len(current) else 0.0
        previous_value = float(previous[index]) if index < len(previous) else 0.0
        delta[index] = current_value - previous_value
    return delta


def _make_workspace(trajectory: LoadedTrajectory) -> HiddenWorkspace:
    return HiddenWorkspace(
        task_name=trajectory.task_name,
        task_prompt="",
        category=trajectory.category,
        status="running",
        max_generation_attempts=int(trajectory.metadata.get("max_generation_attempts") or trajectory.generation_attempts or 0),
    )


def _record_auxiliary_step(workspace: HiddenWorkspace, step: dict[str, Any]) -> None:
    action = str(step.get("action") or "")
    metadata = dict(step.get("metadata") or {})
    workspace.latent_state.record_action(action)
    if action == "think":
        kind = "repair_plan" if workspace.generation_attempts > 0 or workspace.last_error else "plan"
        workspace.latent_state.record_plan(str(step.get("response_text") or ""), kind=kind)
    elif action in {"write", "patch"}:
        workspace.generation_attempts += 1
        workspace.set_candidate(str(step.get("response_text") or ""))
        workspace.latent_state.record_candidate(workspace.candidate_solution)
    elif action == "verify":
        workspace.record_verification(
            success=bool(step.get("success")),
            verification=metadata or None,
            error=step.get("error"),
        )
        workspace.latent_state.record_verification(
            success=bool(step.get("success")),
            verification=metadata or None,
            error=step.get("error"),
        )
    elif action == "fast_weight_update":
        workspace.latent_state.record_fast_weight_update(
            success=bool(step.get("success")),
            kind=str(metadata.get("kind", "unknown")),
        )
    elif action == "descriptor_update":
        workspace.latent_state.record_descriptor_update(
            success=bool(step.get("success")),
            kind=str(metadata.get("kind", "unknown")),
        )
    elif action == "commit":
        workspace.commit(
            str(step.get("response_text") or workspace.candidate_solution),
            verified=bool(metadata.get("verified", False)),
            metadata=metadata,
        )
    elif action == "fail":
        workspace.fail(str(step.get("error") or step.get("response_text") or "failed"))


def build_latent_memory_training_examples(
    trajectory: LoadedTrajectory,
    *,
    require_verified_commit: bool = False,
    config: Optional[LatentMemoryHeadConfig] = None,
) -> list[LatentMemoryTrainingExample]:
    """Build latent-memory update supervision from hidden workspace steps."""
    if require_verified_commit and not trajectory.committed_verified:
        return []

    resolved_config = config or LatentMemoryHeadConfig()
    workspace = _make_workspace(trajectory)
    examples: list[LatentMemoryTrainingExample] = []
    previous_memory = _normalize_vector(workspace.latent_state.memory_vector, width=resolved_config.output_dim)

    for step in trajectory.steps:
        action = str(step.get("action") or "")
        if action not in LATENT_MEMORY_EVENT_LABELS:
            _record_auxiliary_step(workspace, step)
            previous_memory = _normalize_vector(workspace.latent_state.memory_vector, width=resolved_config.output_dim)
            continue

        feature_vector, feature_summary = encode_latent_memory_features(
            latent_state=workspace.latent_state,
            workspace=workspace,
            event_kind=action,
            response_text=str(step.get("response_text") or ""),
            error_text=str(step.get("error") or ""),
            success=step.get("success"),
            config=resolved_config,
        )
        current_state = LatentControllerState(**dict(step.get("latent_state") or {}))
        current_memory = _normalize_vector(current_state.memory_vector, width=resolved_config.output_dim)
        target_vector = _subtract_vectors(current_memory, previous_memory, width=resolved_config.output_dim)
        examples.append(
            LatentMemoryTrainingExample(
                example_id=(
                    f"{Path(trajectory.source_path).stem}:latent_memory:"
                    f"{int(step.get('step_index') or step.get('index') or len(examples) + 1)}"
                ),
                source_path=trajectory.source_path,
                task_name=trajectory.task_name,
                category=trajectory.category,
                event_kind=action,
                step_index=int(step.get("step_index") or step.get("index") or 0),
                feature_vector=feature_vector,
                target_vector=target_vector,
                feature_summary=feature_summary,
            )
        )
        _record_auxiliary_step(workspace, step)
        previous_memory = current_memory

    return examples


def build_latent_memory_dataset(
    root: str | Path,
    *,
    require_verified_commit: bool = False,
    config: Optional[LatentMemoryHeadConfig] = None,
) -> list[LatentMemoryTrainingExample]:
    dataset: list[LatentMemoryTrainingExample] = []
    for trajectory in iter_trajectories(root):
        dataset.extend(
            build_latent_memory_training_examples(
                trajectory,
                require_verified_commit=require_verified_commit,
                config=config,
            )
        )
    return dataset


def _split_by_source(
    items: list[LatentMemoryTrainingExample],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[LatentMemoryTrainingExample], list[LatentMemoryTrainingExample]]:
    grouped: dict[str, list[LatentMemoryTrainingExample]] = {}
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
    train: list[LatentMemoryTrainingExample] = []
    val: list[LatentMemoryTrainingExample] = []
    for source_path, group in grouped.items():
        if source_path in val_sources:
            val.extend(group)
        else:
            train.extend(group)
    return train, val


def write_latent_memory_dataset(
    examples: Iterable[LatentMemoryTrainingExample],
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


def build_latent_memory_training_bundle(
    trajectory_root: str | Path,
    output_dir: str | Path,
    *,
    require_verified_commit: bool = False,
    config: Optional[LatentMemoryHeadConfig] = None,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_config = config or LatentMemoryHeadConfig()
    examples = build_latent_memory_dataset(
        trajectory_root,
        require_verified_commit=require_verified_commit,
        config=resolved_config,
    )
    train_examples, val_examples = _split_by_source(examples, val_ratio=val_ratio, seed=seed)
    train_path = output_root / "latent_memory_train.jsonl"
    val_path = output_root / "latent_memory_val.jsonl"
    train_count = write_latent_memory_dataset(train_examples, train_path)
    val_count = write_latent_memory_dataset(val_examples, val_path)
    return {
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_examples": train_count,
        "val_examples": val_count,
        "train_source_files": len({example.source_path for example in train_examples}),
        "val_source_files": len({example.source_path for example in val_examples}),
        "output_dim": resolved_config.output_dim,
    }


def _load_jsonl_examples(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def train_latent_memory_head(
    *,
    train_path: str | Path,
    val_path: str | Path,
    output_path: str | Path,
    config: Optional[LatentMemoryHeadConfig] = None,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train a latent memory head on hidden event-state examples."""
    resolved_config = config or LatentMemoryHeadConfig()
    train_rows = _load_jsonl_examples(train_path)
    val_rows = _load_jsonl_examples(val_path)
    model = LatentMemoryHead(resolved_config).to(device)

    if not train_rows:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": resolved_config.to_dict(),
                "metrics": {
                    "train_examples": 0,
                    "val_examples": len(val_rows),
                    "train_loss": 0.0,
                    "val_loss": 0.0,
                    "trained": False,
                },
            },
            destination,
        )
        return {
            "output_path": str(destination),
            "train_examples": 0,
            "val_examples": len(val_rows),
            "train_loss": 0.0,
            "val_loss": 0.0,
            "trained": False,
        }

    def to_tensors(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.tensor([row["feature_vector"] for row in rows], dtype=torch.float32)
        targets = torch.tensor([row["target_vector"] for row in rows], dtype=torch.float32)
        return features, targets

    train_features, train_targets = to_tensors(train_rows)
    if val_rows:
        val_features, val_targets = to_tensors(val_rows)
    else:
        val_features = torch.zeros((0, resolved_config.input_dim), dtype=torch.float32)
        val_targets = torch.zeros((0, resolved_config.output_dim), dtype=torch.float32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    train_loader = DataLoader(TensorDataset(train_features, train_targets), batch_size=batch_size, shuffle=True)

    model.train()
    for _epoch in range(max(1, epochs)):
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_loss = criterion(model(train_features.to(device)), train_targets.to(device)).item()
        val_loss = (
            criterion(model(val_features.to(device)), val_targets.to(device)).item()
            if len(val_rows) > 0
            else 0.0
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "trained": True,
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": resolved_config.to_dict(),
            "metrics": metrics,
        },
        destination,
    )
    return {"output_path": str(destination), **metrics}
