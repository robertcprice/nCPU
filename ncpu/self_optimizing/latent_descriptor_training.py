"""Prepare and train learned latent descriptor heads from hidden SOME trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ncpu.self_optimizing.latent_descriptor_head import (
    LatentDescriptorHead,
    LatentDescriptorHeadConfig,
    encode_latent_descriptor_features,
)
from ncpu.self_optimizing.latent_controller_state import LatentControllerState
from ncpu.self_optimizing.trajectory_dataset import LoadedTrajectory, iter_trajectories


@dataclass
class LatentDescriptorTrainingExample:
    """One latent-state to descriptor-projection training example."""

    example_id: str
    source_path: str
    task_name: str
    category: str
    update_kind: str
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


def build_latent_descriptor_training_examples(
    trajectory: LoadedTrajectory,
    *,
    require_verified_commit: bool = True,
    config: Optional[LatentDescriptorHeadConfig] = None,
) -> list[LatentDescriptorTrainingExample]:
    """Build descriptor-head supervision from successful descriptor update steps."""
    if require_verified_commit and not trajectory.committed_verified:
        return []

    resolved_config = config or LatentDescriptorHeadConfig()
    examples: list[LatentDescriptorTrainingExample] = []
    for step in trajectory.steps:
        if str(step.get("action") or "") != "descriptor_update" or step.get("success") is not True:
            continue
        metadata = dict(step.get("metadata") or {})
        descriptor = dict(metadata.get("adaptation_descriptor") or {})
        signal_projection = descriptor.get("signal_projection") or []
        if not isinstance(signal_projection, list) or not signal_projection:
            continue
        latent_state_payload = dict(step.get("latent_state") or {})
        latent_state = LatentControllerState(**latent_state_payload)
        update_kind = str(metadata.get("kind") or descriptor.get("update_kind") or "descriptor_update")
        feature_vector, feature_summary = encode_latent_descriptor_features(
            latent_state=latent_state,
            update_kind=update_kind,
            error_text=str(step.get("error") or latent_state.last_failure_summary or ""),
            candidate_text=str(latent_state.last_candidate_digest or ""),
            config=resolved_config,
        )
        examples.append(
            LatentDescriptorTrainingExample(
                example_id=(
                    f"{Path(trajectory.source_path).stem}:latent_descriptor:"
                    f"{int(step.get('step_index') or step.get('index') or len(examples) + 1)}"
                ),
                source_path=trajectory.source_path,
                task_name=trajectory.task_name,
                category=trajectory.category,
                update_kind=update_kind,
                step_index=int(step.get("step_index") or step.get("index") or 0),
                feature_vector=feature_vector,
                target_vector=_normalize_vector(signal_projection, width=resolved_config.output_dim),
                feature_summary=feature_summary,
            )
        )
    return examples


def build_latent_descriptor_dataset(
    root: str | Path,
    *,
    require_verified_commit: bool = True,
    config: Optional[LatentDescriptorHeadConfig] = None,
) -> list[LatentDescriptorTrainingExample]:
    dataset: list[LatentDescriptorTrainingExample] = []
    for trajectory in iter_trajectories(root):
        dataset.extend(
            build_latent_descriptor_training_examples(
                trajectory,
                require_verified_commit=require_verified_commit,
                config=config,
            )
        )
    return dataset


def _split_by_source(
    items: list[LatentDescriptorTrainingExample],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[LatentDescriptorTrainingExample], list[LatentDescriptorTrainingExample]]:
    grouped: dict[str, list[LatentDescriptorTrainingExample]] = {}
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
    train: list[LatentDescriptorTrainingExample] = []
    val: list[LatentDescriptorTrainingExample] = []
    for source_path, group in grouped.items():
        if source_path in val_sources:
            val.extend(group)
        else:
            train.extend(group)
    return train, val


def write_latent_descriptor_dataset(
    examples: Iterable[LatentDescriptorTrainingExample],
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


def build_latent_descriptor_training_bundle(
    trajectory_root: str | Path,
    output_dir: str | Path,
    *,
    require_verified_commit: bool = True,
    config: Optional[LatentDescriptorHeadConfig] = None,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_config = config or LatentDescriptorHeadConfig()
    examples = build_latent_descriptor_dataset(
        trajectory_root,
        require_verified_commit=require_verified_commit,
        config=resolved_config,
    )
    train_examples, val_examples = _split_by_source(examples, val_ratio=val_ratio, seed=seed)
    train_path = output_root / "latent_descriptor_train.jsonl"
    val_path = output_root / "latent_descriptor_val.jsonl"
    train_count = write_latent_descriptor_dataset(train_examples, train_path)
    val_count = write_latent_descriptor_dataset(val_examples, val_path)
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


def train_latent_descriptor_head(
    *,
    train_path: str | Path,
    val_path: str | Path,
    output_path: str | Path,
    config: Optional[LatentDescriptorHeadConfig] = None,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train a latent descriptor head on latent-state examples."""
    resolved_config = config or LatentDescriptorHeadConfig()
    train_rows = _load_jsonl_examples(train_path)
    val_rows = _load_jsonl_examples(val_path)
    model = LatentDescriptorHead(resolved_config).to(device)

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
        train_predictions = model(train_features.to(device))
        train_loss = float(criterion(train_predictions, train_targets.to(device)).item())
        if val_rows:
            val_predictions = model(val_features.to(device))
            val_loss = float(criterion(val_predictions, val_targets.to(device)).item())
        else:
            val_loss = train_loss

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": resolved_config.to_dict(),
            "metrics": {
                "train_examples": len(train_rows),
                "val_examples": len(val_rows),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
        },
        destination,
    )
    return {
        "output_path": str(destination),
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
