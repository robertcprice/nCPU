"""Prepare and train latent halt/commit heads from hidden SOME trajectories."""

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
from ncpu.self_optimizing.latent_halt_policy import (
    LATENT_HALT_LABELS,
    LatentHaltHead,
    LatentHaltHeadConfig,
    encode_latent_halt_features,
)
from ncpu.self_optimizing.trajectory_dataset import LoadedTrajectory, iter_trajectories


@dataclass
class LatentHaltTrainingExample:
    """One latent halt/commit training example."""

    example_id: str
    source_path: str
    task_name: str
    category: str
    allowed_actions: list[str]
    target_action: str
    step_index: int
    verification_success: bool
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


def _record_step(workspace: HiddenWorkspace, step: dict[str, Any]) -> None:
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


def _next_terminal_decision(steps: list[dict[str, Any]], start_index: int) -> tuple[Optional[str], list[str]]:
    remaining_actions: list[str] = []
    for candidate in steps[start_index + 1 :]:
        action = str(candidate.get("action") or "")
        remaining_actions.append(action)
        if action == "commit":
            return "commit", remaining_actions
        if action == "fail":
            return "fail", remaining_actions
        if action in {"think", "patch", "write"}:
            return "continue", remaining_actions
    return None, remaining_actions


def build_latent_halt_training_examples(
    trajectory: LoadedTrajectory,
    *,
    require_verified_commit: bool = True,
) -> list[LatentHaltTrainingExample]:
    """Build halt-policy supervision examples from a hidden trajectory."""
    if require_verified_commit and not trajectory.committed_verified:
        return []

    workspace = _make_workspace(trajectory)
    examples: list[LatentHaltTrainingExample] = []
    for index, step in enumerate(trajectory.steps):
        _record_step(workspace, step)
        if str(step.get("action") or "") != "verify":
            continue

        verification_success = bool(step.get("success"))
        target_action, remaining_actions = _next_terminal_decision(trajectory.steps, index)
        if target_action is None:
            continue
        allowed_actions = ["commit", "continue"] if verification_success else ["continue", "fail"]
        if target_action not in allowed_actions:
            continue
        features, summary = encode_latent_halt_features(
            latent_state=workspace.latent_state,
            workspace=workspace,
            verification_success=verification_success,
            verification_error=str(step.get("error") or step.get("response_text") or ""),
            allowed_actions=allowed_actions,
        )
        examples.append(
            LatentHaltTrainingExample(
                example_id=f"{Path(trajectory.source_path).stem}:halt:{int(step.get('step_index') or step.get('index') or len(examples) + 1)}",
                source_path=trajectory.source_path,
                task_name=trajectory.task_name,
                category=trajectory.category,
                allowed_actions=allowed_actions,
                target_action=target_action,
                step_index=int(step.get("step_index") or step.get("index") or 0),
                verification_success=verification_success,
                feature_vector=features,
                feature_summary={**summary, "remaining_actions": remaining_actions[:4]},
            )
        )
    return examples


def build_latent_halt_dataset(
    root: str | Path,
    *,
    require_verified_commit: bool = True,
) -> list[LatentHaltTrainingExample]:
    dataset: list[LatentHaltTrainingExample] = []
    for trajectory in iter_trajectories(root):
        dataset.extend(
            build_latent_halt_training_examples(
                trajectory,
                require_verified_commit=require_verified_commit,
            )
        )
    return dataset


def _split_by_source(
    items: list[LatentHaltTrainingExample],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[LatentHaltTrainingExample], list[LatentHaltTrainingExample]]:
    grouped: dict[str, list[LatentHaltTrainingExample]] = {}
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
    train: list[LatentHaltTrainingExample] = []
    val: list[LatentHaltTrainingExample] = []
    for source_path, group in grouped.items():
        if source_path in val_sources:
            val.extend(group)
        else:
            train.extend(group)
    return train, val


def write_latent_halt_dataset(
    examples: Iterable[LatentHaltTrainingExample],
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


def build_latent_halt_training_bundle(
    trajectory_root: str | Path,
    output_dir: str | Path,
    *,
    require_verified_commit: bool = True,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    examples = build_latent_halt_dataset(
        trajectory_root,
        require_verified_commit=require_verified_commit,
    )
    train_examples, val_examples = _split_by_source(examples, val_ratio=val_ratio, seed=seed)
    train_path = output_root / "latent_halt_train.jsonl"
    val_path = output_root / "latent_halt_val.jsonl"
    train_count = write_latent_halt_dataset(train_examples, train_path)
    val_count = write_latent_halt_dataset(val_examples, val_path)
    return {
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_examples": train_count,
        "val_examples": val_count,
        "train_source_files": len({example.source_path for example in train_examples}),
        "val_source_files": len({example.source_path for example in val_examples}),
        "action_labels": list(LATENT_HALT_LABELS),
    }


def _load_jsonl_examples(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def train_latent_halt_head(
    *,
    train_path: str | Path,
    val_path: str | Path,
    output_path: str | Path,
    config: Optional[LatentHaltHeadConfig] = None,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train a latent halt head on trajectory-derived features."""
    resolved_config = config or LatentHaltHeadConfig()
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

    model = LatentHaltHead(resolved_config).to(device)
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
