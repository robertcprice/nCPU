"""Prepare and train latent action-selection heads from hidden SOME trajectories."""

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
from ncpu.self_optimizing.latent_action_policy import (
    LATENT_ACTION_LABELS,
    LatentActionHead,
    LatentActionHeadConfig,
    encode_latent_action_features,
)
from ncpu.self_optimizing.trajectory_dataset import LoadedTrajectory, iter_trajectories


@dataclass
class LatentActionTrainingExample:
    """One latent-state action-selection training example."""

    example_id: str
    source_path: str
    task_name: str
    category: str
    allowed_actions: list[str]
    target_action: str
    step_index: int
    feature_vector: list[float]
    feature_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _find_next_decision_action(steps: list[dict[str, Any]], start_index: int) -> Optional[dict[str, Any]]:
    for candidate in steps[start_index + 1 :]:
        action = str(candidate.get("action") or "")
        if action in {"think", "patch", "fail"}:
            return candidate
        if action in {"commit", "write"}:
            return None
    return None


def build_latent_action_training_examples(
    trajectory: LoadedTrajectory,
    *,
    require_verified_commit: bool = True,
) -> list[LatentActionTrainingExample]:
    """Build latent-state action examples from a hidden workspace trajectory."""
    if require_verified_commit and not trajectory.committed_verified:
        return []

    workspace = _make_workspace(trajectory)
    steps = trajectory.steps
    examples: list[LatentActionTrainingExample] = []

    first_decision_step = next(
        (step for step in steps if str(step.get("action") or "") in {"think", "write"}),
        None,
    )
    if first_decision_step is not None:
        features, summary = encode_latent_action_features(
            latent_state=workspace.latent_state,
            workspace=workspace,
            allowed_actions=["think", "write"],
        )
        examples.append(
            LatentActionTrainingExample(
                example_id=f"{Path(trajectory.source_path).stem}:latent:init",
                source_path=trajectory.source_path,
                task_name=trajectory.task_name,
                category=trajectory.category,
                allowed_actions=["think", "write"],
                target_action=str(first_decision_step.get("action")),
                step_index=int(first_decision_step.get("step_index") or first_decision_step.get("index") or 0),
                feature_vector=features,
                feature_summary=summary,
            )
        )

    for index, step in enumerate(steps):
        _record_auxiliary_step(workspace, step)
        if str(step.get("action") or "") != "verify" or bool(step.get("success")):
            continue

        next_decision = _find_next_decision_action(steps, index)
        if next_decision is None:
            continue
        features, summary = encode_latent_action_features(
            latent_state=workspace.latent_state,
            workspace=workspace,
            allowed_actions=["patch", "think", "fail"],
        )
        examples.append(
            LatentActionTrainingExample(
                example_id=(
                    f"{Path(trajectory.source_path).stem}:latent:"
                    f"{int(next_decision.get('step_index') or next_decision.get('index') or len(examples) + 1)}"
                ),
                source_path=trajectory.source_path,
                task_name=trajectory.task_name,
                category=trajectory.category,
                allowed_actions=["patch", "think", "fail"],
                target_action=str(next_decision.get("action")),
                step_index=int(next_decision.get("step_index") or next_decision.get("index") or 0),
                feature_vector=features,
                feature_summary=summary,
            )
        )

    return examples


def build_latent_action_dataset(
    root: str | Path,
    *,
    require_verified_commit: bool = True,
) -> list[LatentActionTrainingExample]:
    dataset: list[LatentActionTrainingExample] = []
    for trajectory in iter_trajectories(root):
        dataset.extend(
            build_latent_action_training_examples(
                trajectory,
                require_verified_commit=require_verified_commit,
            )
        )
    return dataset


def _split_by_source(
    items: list[LatentActionTrainingExample],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[LatentActionTrainingExample], list[LatentActionTrainingExample]]:
    grouped: dict[str, list[LatentActionTrainingExample]] = {}
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
    train: list[LatentActionTrainingExample] = []
    val: list[LatentActionTrainingExample] = []
    for source_path, group in grouped.items():
        if source_path in val_sources:
            val.extend(group)
        else:
            train.extend(group)
    return train, val


def write_latent_action_dataset(
    examples: Iterable[LatentActionTrainingExample],
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


def build_latent_action_training_bundle(
    trajectory_root: str | Path,
    output_dir: str | Path,
    *,
    require_verified_commit: bool = True,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    examples = build_latent_action_dataset(
        trajectory_root,
        require_verified_commit=require_verified_commit,
    )
    train_examples, val_examples = _split_by_source(examples, val_ratio=val_ratio, seed=seed)
    train_path = output_root / "latent_action_train.jsonl"
    val_path = output_root / "latent_action_val.jsonl"
    train_count = write_latent_action_dataset(train_examples, train_path)
    val_count = write_latent_action_dataset(val_examples, val_path)
    return {
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_examples": train_count,
        "val_examples": val_count,
        "train_source_files": len({example.source_path for example in train_examples}),
        "val_source_files": len({example.source_path for example in val_examples}),
        "action_labels": list(LATENT_ACTION_LABELS),
    }


def _load_jsonl_examples(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def train_latent_action_head(
    *,
    train_path: str | Path,
    val_path: str | Path,
    output_path: str | Path,
    config: Optional[LatentActionHeadConfig] = None,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train a lightweight latent action head on trajectory-derived features."""
    resolved_config = config or LatentActionHeadConfig()
    train_rows = _load_jsonl_examples(train_path)
    val_rows = _load_jsonl_examples(val_path)
    label_to_index = {label: index for index, label in enumerate(resolved_config.action_labels)}

    def to_tensors(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.tensor([row["feature_vector"] for row in rows], dtype=torch.float32)
        targets = torch.tensor([label_to_index[row["target_action"]] for row in rows], dtype=torch.long)
        return features, targets

    train_features, train_targets = to_tensors(train_rows)
    val_features, val_targets = to_tensors(val_rows) if val_rows else (
        torch.zeros((0, resolved_config.input_dim), dtype=torch.float32),
        torch.zeros((0,), dtype=torch.long),
    )

    model = LatentActionHead(resolved_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(train_features, train_targets), batch_size=batch_size, shuffle=True)

    model.train()
    for _epoch in range(max(1, epochs)):
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_logits = model(train_features.to(device))
        train_predictions = train_logits.argmax(dim=-1).cpu()
        train_accuracy = float((train_predictions == train_targets).float().mean().item())
        if val_rows:
            val_logits = model(val_features.to(device))
            val_predictions = val_logits.argmax(dim=-1).cpu()
            val_accuracy = float((val_predictions == val_targets).float().mean().item())
        else:
            val_accuracy = train_accuracy

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": resolved_config.to_dict(),
            "metrics": {
                "train_examples": len(train_rows),
                "val_examples": len(val_rows),
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            },
        },
        destination,
    )
    return {
        "output_path": str(destination),
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
    }
