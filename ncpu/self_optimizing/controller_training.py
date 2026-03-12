"""Build trainable controller datasets from hidden SOME trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
from typing import Any, Iterable, Optional, TypeVar

from ncpu.self_optimizing.trajectory_dataset import (
    DistillationExample,
    LoadedTrajectory,
    build_distillation_dataset,
    iter_trajectories,
)
from ncpu.self_optimizing.latent_action_training import build_latent_action_training_bundle
from ncpu.self_optimizing.latent_descriptor_training import build_latent_descriptor_training_bundle
from ncpu.self_optimizing.latent_halt_training import build_latent_halt_training_bundle
from ncpu.self_optimizing.latent_memory_training import build_latent_memory_training_bundle
from ncpu.self_optimizing.state_patch_training import build_state_patch_training_bundle


ACTION_LABELS = ("think", "write", "patch", "commit", "fail")


@dataclass
class ActionPolicyExample:
    """Single action-selection supervision example."""

    example_id: str
    source_path: str
    task_name: str
    category: str
    prompt: str
    target_action: str
    step_index: int
    success_label: bool
    committed_verified: bool
    workspace_status: str
    metadata: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ControllerTrainingBundle:
    """Collection of train/validation datasets for controller distillation."""

    output_dir: str
    response_train_path: str
    response_val_path: str
    action_train_path: Optional[str]
    action_val_path: Optional[str]
    latent_action_train_path: Optional[str]
    latent_action_val_path: Optional[str]
    latent_descriptor_train_path: Optional[str]
    latent_descriptor_val_path: Optional[str]
    latent_memory_train_path: Optional[str]
    latent_memory_val_path: Optional[str]
    latent_halt_train_path: Optional[str]
    latent_halt_val_path: Optional[str]
    state_patch_train_path: Optional[str]
    state_patch_val_path: Optional[str]
    manifest_path: str
    summary: dict[str, Any]


def _truncate(text: Any, *, limit: int = 220) -> str:
    value = str(text or "").strip().replace("\n", "\\n")
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _summarize_prior_steps(steps: list[dict[str, Any]], *, limit: int = 6) -> str:
    if not steps:
        return "(none)"
    recent = steps[-limit:]
    lines: list[str] = []
    for step in recent:
        action = step.get("action", "?")
        pieces = [f"{action}"]
        if step.get("success") is True:
            pieces.append("success")
        elif step.get("success") is False:
            pieces.append("failure")
        if step.get("error"):
            pieces.append(f"error={_truncate(step['error'], limit=100)}")
        response_text = step.get("response_text")
        if response_text and action in {"think", "write", "patch"}:
            pieces.append(f"response={_truncate(response_text, limit=120)}")
        lines.append("- " + " | ".join(pieces))
    return "\n".join(lines)


def _last_verification_error(steps: list[dict[str, Any]]) -> Optional[str]:
    for step in reversed(steps):
        if step.get("action") == "verify" and step.get("success") is False:
            return str(step.get("error") or step.get("response_text") or "").strip() or None
    return None


def build_action_policy_examples(
    trajectory: LoadedTrajectory,
    *,
    require_verified_commit: bool = True,
) -> list[ActionPolicyExample]:
    """Build action-decision supervision from a hidden workspace trace."""
    if require_verified_commit and not trajectory.committed_verified:
        return []

    examples: list[ActionPolicyExample] = []
    prior_steps: list[dict[str, Any]] = []
    for step in trajectory.steps:
        action = str(step.get("action") or "")
        if action == "verify":
            prior_steps.append(step)
            continue
        if action not in ACTION_LABELS:
            prior_steps.append(step)
            continue

        prompt = (
            "You are choosing the next hidden controller action inside a private "
            "code/reasoning workspace.\n"
            f"Task: {trajectory.task_name}\n"
            f"Category: {trajectory.category}\n"
            f"Workspace status: {trajectory.workspace_status}\n"
            f"Committed output verified: {'yes' if trajectory.committed_verified else 'no'}\n"
            "Recent hidden history:\n"
            f"{_summarize_prior_steps(prior_steps)}\n\n"
            f"Last verification error: {_last_verification_error(prior_steps) or 'none'}\n"
            "Choose the single best next action from: think, write, patch, commit, fail.\n"
            "Return only the action label."
        )
        target_action = action
        example = ActionPolicyExample(
            example_id=f"{Path(trajectory.source_path).stem}:action:{int(step.get('step_index') or step.get('index') or len(examples) + 1)}",
            source_path=trajectory.source_path,
            task_name=trajectory.task_name,
            category=trajectory.category,
            prompt=prompt,
            target_action=target_action,
            step_index=int(step.get("step_index") or step.get("index") or 0),
            success_label=step.get("success") is not False,
            committed_verified=trajectory.committed_verified,
            workspace_status=trajectory.workspace_status,
            metadata={
                "step_timestamp": step.get("timestamp"),
                "step_metadata": dict(step.get("metadata") or {}),
                "max_generation_attempts": trajectory.metadata.get("max_generation_attempts"),
            },
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target_action},
            ],
        )
        examples.append(example)
        prior_steps.append(step)

    return examples


T = TypeVar("T")


def _split_by_source(items: list[T], *, val_ratio: float, seed: int) -> tuple[list[T], list[T]]:
    grouped: dict[str, list[T]] = {}
    for item in items:
        source_path = getattr(item, "source_path")
        grouped.setdefault(source_path, []).append(item)

    source_paths = list(grouped)
    rng = random.Random(seed)
    rng.shuffle(source_paths)

    if len(source_paths) <= 1:
        return items, []

    num_val = max(1, int(round(len(source_paths) * val_ratio)))
    num_val = min(num_val, len(source_paths) - 1)
    val_sources = set(source_paths[:num_val])

    train: list[T] = []
    val: list[T] = []
    for source_path, group in grouped.items():
        if source_path in val_sources:
            val.extend(group)
        else:
            train.extend(group)
    return train, val


def _write_examples(examples: Iterable[Any], path: str | Path) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for example in examples:
            payload = example.to_dict() if hasattr(example, "to_dict") else asdict(example)
            handle.write(json.dumps(payload) + "\n")
            count += 1
    return count


def _summarize_sources(examples: Iterable[Any]) -> int:
    return len({getattr(example, "source_path") for example in examples})


def build_controller_training_bundle(
    trajectory_root: str | Path,
    output_dir: str | Path,
    *,
    include_failed_response_steps: bool = False,
    include_think_steps: bool = True,
    allow_unverified_trajectories: bool = False,
    include_action_policy: bool = True,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> ControllerTrainingBundle:
    """Create train/validation datasets for response and action supervision."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    response_examples = build_distillation_dataset(
        trajectory_root,
        include_failed_steps=include_failed_response_steps,
        include_think_steps=include_think_steps,
        require_verified_commit=not allow_unverified_trajectories,
    )
    response_train, response_val = _split_by_source(response_examples, val_ratio=val_ratio, seed=seed)

    action_examples: list[ActionPolicyExample] = []
    if include_action_policy:
        for trajectory in iter_trajectories(trajectory_root):
            action_examples.extend(
                build_action_policy_examples(
                    trajectory,
                    require_verified_commit=not allow_unverified_trajectories,
                )
            )
    action_train, action_val = _split_by_source(action_examples, val_ratio=val_ratio, seed=seed) if action_examples else ([], [])

    response_train_path = output_root / "response_sft_train.jsonl"
    response_val_path = output_root / "response_sft_val.jsonl"
    action_train_path = output_root / "action_policy_train.jsonl" if include_action_policy else None
    action_val_path = output_root / "action_policy_val.jsonl" if include_action_policy else None
    latent_action_summary = (
        build_latent_action_training_bundle(
            trajectory_root,
            output_root,
            require_verified_commit=not allow_unverified_trajectories,
            val_ratio=val_ratio,
            seed=seed,
        )
        if include_action_policy
        else {
            "train_path": None,
            "val_path": None,
            "train_examples": 0,
            "val_examples": 0,
            "train_source_files": 0,
            "val_source_files": 0,
            "action_labels": list(ACTION_LABELS),
        }
    )
    latent_halt_summary = (
        build_latent_halt_training_bundle(
            trajectory_root,
            output_root,
            require_verified_commit=not allow_unverified_trajectories,
            val_ratio=val_ratio,
            seed=seed,
        )
        if include_action_policy
        else {
            "train_path": None,
            "val_path": None,
            "train_examples": 0,
            "val_examples": 0,
            "train_source_files": 0,
            "val_source_files": 0,
            "action_labels": ["continue", "commit", "fail"],
        }
    )
    latent_descriptor_summary = (
        build_latent_descriptor_training_bundle(
            trajectory_root,
            output_root,
            require_verified_commit=not allow_unverified_trajectories,
            val_ratio=val_ratio,
            seed=seed,
        )
        if include_action_policy
        else {
            "train_path": None,
            "val_path": None,
            "train_examples": 0,
            "val_examples": 0,
            "train_source_files": 0,
            "val_source_files": 0,
            "output_dim": 16,
        }
    )
    latent_memory_summary = (
        build_latent_memory_training_bundle(
            trajectory_root,
            output_root,
            require_verified_commit=False,
            val_ratio=val_ratio,
            seed=seed,
        )
        if include_action_policy
        else {
            "train_path": None,
            "val_path": None,
            "train_examples": 0,
            "val_examples": 0,
            "train_source_files": 0,
            "val_source_files": 0,
            "output_dim": 16,
        }
    )
    state_patch_summary = (
        build_state_patch_training_bundle(
            trajectory_root,
            output_root,
            require_verified_commit=not allow_unverified_trajectories,
            val_ratio=val_ratio,
            seed=seed,
        )
        if include_action_policy
        else {
            "train_path": None,
            "val_path": None,
            "train_examples": 0,
            "val_examples": 0,
            "train_source_files": 0,
            "val_source_files": 0,
            "input_dim": 16,
            "output_dim": 16,
        }
    )
    manifest_path = output_root / "training_manifest.json"

    response_train_count = _write_examples(response_train, response_train_path)
    response_val_count = _write_examples(response_val, response_val_path)
    action_train_count = _write_examples(action_train, action_train_path) if action_train_path else 0
    action_val_count = _write_examples(action_val, action_val_path) if action_val_path else 0

    summary = {
        "trajectory_root": str(Path(trajectory_root)),
        "output_dir": str(output_root),
        "response_sft": {
            "train_path": str(response_train_path),
            "val_path": str(response_val_path),
            "train_examples": response_train_count,
            "val_examples": response_val_count,
            "train_source_files": _summarize_sources(response_train),
            "val_source_files": _summarize_sources(response_val),
        },
        "action_policy": {
            "enabled": include_action_policy,
            "train_path": str(action_train_path) if action_train_path else None,
            "val_path": str(action_val_path) if action_val_path else None,
            "train_examples": action_train_count,
            "val_examples": action_val_count,
            "train_source_files": _summarize_sources(action_train),
            "val_source_files": _summarize_sources(action_val),
            "action_labels": list(ACTION_LABELS),
        },
        "latent_action_policy": dict(latent_action_summary),
        "latent_descriptor_head": dict(latent_descriptor_summary),
        "latent_memory_head": dict(latent_memory_summary),
        "latent_halt_policy": dict(latent_halt_summary),
        "state_patch_head": dict(state_patch_summary),
        "options": {
            "include_failed_response_steps": include_failed_response_steps,
            "include_think_steps": include_think_steps,
            "allow_unverified_trajectories": allow_unverified_trajectories,
            "val_ratio": val_ratio,
            "seed": seed,
        },
        "recommended_training": {
            "base_model": "Qwen/Qwen2.5-Coder-1.5B",
            "response_objective": "causal_lm_sft",
            "action_objective": "causal_lm_sft",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
    }
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return ControllerTrainingBundle(
        output_dir=str(output_root),
        response_train_path=str(response_train_path),
        response_val_path=str(response_val_path),
        action_train_path=str(action_train_path) if action_train_path else None,
        action_val_path=str(action_val_path) if action_val_path else None,
        latent_action_train_path=str(latent_action_summary["train_path"]) if latent_action_summary["train_path"] else None,
        latent_action_val_path=str(latent_action_summary["val_path"]) if latent_action_summary["val_path"] else None,
        latent_descriptor_train_path=(
            str(latent_descriptor_summary["train_path"]) if latent_descriptor_summary["train_path"] else None
        ),
        latent_descriptor_val_path=(
            str(latent_descriptor_summary["val_path"]) if latent_descriptor_summary["val_path"] else None
        ),
        latent_memory_train_path=(
            str(latent_memory_summary["train_path"]) if latent_memory_summary["train_path"] else None
        ),
        latent_memory_val_path=(
            str(latent_memory_summary["val_path"]) if latent_memory_summary["val_path"] else None
        ),
        latent_halt_train_path=str(latent_halt_summary["train_path"]) if latent_halt_summary["train_path"] else None,
        latent_halt_val_path=str(latent_halt_summary["val_path"]) if latent_halt_summary["val_path"] else None,
        state_patch_train_path=str(state_patch_summary["train_path"]) if state_patch_summary["train_path"] else None,
        state_patch_val_path=str(state_patch_summary["val_path"]) if state_patch_summary["val_path"] else None,
        manifest_path=str(manifest_path),
        summary=summary,
    )
