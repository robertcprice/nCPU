"""Prepare hidden SOME trajectories for distillation and controller training."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


@dataclass
class LoadedTrajectory:
    """In-memory representation of a single hidden workspace trajectory."""

    source_path: str
    task_name: str
    category: str
    workspace_status: str
    committed_verified: bool
    committed_output: Optional[str]
    generation_attempts: int
    last_error: Optional[str]
    steps: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistillationExample:
    """Single supervised training example derived from a hidden trajectory."""

    example_id: str
    source_path: str
    task_name: str
    category: str
    example_type: str
    prompt: str
    target: str
    success_label: bool
    committed_verified: bool
    workspace_status: str
    step_index: int
    generation_attempt: Optional[int] = None
    verification_error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_trajectory(path: str | Path) -> LoadedTrajectory:
    """Load a hidden workspace trajectory JSONL file."""
    jsonl_path = Path(path)
    records = _load_jsonl(jsonl_path)
    if not records:
        raise ValueError(f"Trajectory file is empty: {jsonl_path}")

    init_event = next((record for record in records if record.get("event") == "workspace_initialized"), None)
    if init_event is None:
        raise ValueError(f"Missing workspace_initialized event: {jsonl_path}")

    terminal_event = next(
        (record for record in reversed(records) if record.get("event") in {"workspace_committed", "workspace_failed"}),
        None,
    )
    if terminal_event is None:
        raise ValueError(f"Missing terminal workspace event: {jsonl_path}")

    steps = [record for record in records if record.get("event") == "workspace_step"]
    return LoadedTrajectory(
        source_path=str(jsonl_path),
        task_name=str(init_event.get("task_name") or ""),
        category=str(init_event.get("category") or "coding"),
        workspace_status=str(terminal_event.get("status") or "unknown"),
        committed_verified=bool(terminal_event.get("committed_verified", False)),
        committed_output=terminal_event.get("committed_output"),
        generation_attempts=int(terminal_event.get("generation_attempts") or 0),
        last_error=terminal_event.get("last_error"),
        steps=steps,
        metadata={
            "max_generation_attempts": init_event.get("max_generation_attempts"),
            "terminal_event": terminal_event.get("event"),
        },
    )


def iter_trajectories(root: str | Path) -> Iterator[LoadedTrajectory]:
    """Yield valid hidden trajectories below a root directory."""
    root_path = Path(root)
    if root_path.is_file():
        yield load_trajectory(root_path)
        return

    for path in sorted(root_path.rglob("*.jsonl")):
        try:
            yield load_trajectory(path)
        except ValueError:
            continue


def _find_following_verify(steps: list[dict[str, Any]], index: int) -> Optional[dict[str, Any]]:
    for candidate in steps[index + 1 :]:
        action = candidate.get("action")
        if action == "verify":
            return candidate
        if action in {"write", "patch"}:
            return None
    return None


def _coerce_verification_error(step: dict[str, Any], verify_step: Optional[dict[str, Any]], trajectory: LoadedTrajectory) -> Optional[str]:
    if verify_step is not None:
        if verify_step.get("error"):
            return str(verify_step["error"])
        response_text = verify_step.get("response_text")
        if response_text and response_text != "pass":
            return str(response_text)
    return step.get("error") or trajectory.last_error


def build_distillation_examples(
    trajectory: LoadedTrajectory,
    *,
    include_failed_steps: bool = False,
    include_think_steps: bool = True,
) -> list[DistillationExample]:
    """Convert a hidden workspace trajectory into SFT-ready examples."""
    examples: list[DistillationExample] = []
    generation_attempt = 0

    for index, step in enumerate(trajectory.steps):
        action = str(step.get("action") or "")
        if action == "verify":
            continue
        if action == "think" and not include_think_steps:
            continue

        verify_step = _find_following_verify(trajectory.steps, index)
        if action in {"write", "patch"}:
            generation_attempt += 1
            success_label = bool(verify_step and verify_step.get("success"))
        else:
            success_label = trajectory.committed_verified

        if not include_failed_steps and not success_label:
            continue

        verification_error = _coerce_verification_error(step, verify_step, trajectory)
        prompt = str(step.get("prompt") or "")
        target = str(step.get("response_text") or "")
        if not prompt or not target:
            continue

        example = DistillationExample(
            example_id=f"{Path(trajectory.source_path).stem}:{int(step.get('step_index') or step.get('index') or len(examples) + 1)}",
            source_path=trajectory.source_path,
            task_name=trajectory.task_name,
            category=trajectory.category,
            example_type=action,
            prompt=prompt,
            target=target,
            success_label=success_label,
            committed_verified=trajectory.committed_verified,
            workspace_status=trajectory.workspace_status,
            step_index=int(step.get("step_index") or step.get("index") or 0),
            generation_attempt=(generation_attempt if action in {"write", "patch"} else None),
            verification_error=verification_error if not success_label else None,
            metadata={
                "step_timestamp": step.get("timestamp"),
                "step_metadata": dict(step.get("metadata") or {}),
                "verification_metadata": dict(verify_step.get("metadata") or {}) if verify_step else {},
                "trajectory_metadata": dict(trajectory.metadata),
            },
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ],
        )
        examples.append(example)

    return examples


def build_distillation_dataset(
    root: str | Path,
    *,
    include_failed_steps: bool = False,
    include_think_steps: bool = True,
    require_verified_commit: bool = True,
) -> list[DistillationExample]:
    """Build a distillation dataset from all valid trajectories beneath a root."""
    dataset: list[DistillationExample] = []
    for trajectory in iter_trajectories(root):
        if require_verified_commit and not trajectory.committed_verified:
            continue
        dataset.extend(
            build_distillation_examples(
                trajectory,
                include_failed_steps=include_failed_steps,
                include_think_steps=include_think_steps,
            )
        )
    return dataset


def summarize_distillation_dataset(examples: Iterable[DistillationExample]) -> dict[str, Any]:
    """Summarize a prepared distillation dataset."""
    cached = list(examples)
    by_type = Counter(example.example_type for example in cached)
    by_category = Counter(example.category for example in cached)
    by_success = Counter("success" if example.success_label else "failure" for example in cached)
    return {
        "num_examples": len(cached),
        "example_types": dict(by_type),
        "categories": dict(by_category),
        "success_labels": dict(by_success),
        "verified_commits": sum(1 for example in cached if example.committed_verified),
        "source_files": len({example.source_path for example in cached}),
    }


def write_distillation_dataset(
    examples: Iterable[DistillationExample],
    output_path: str | Path,
) -> dict[str, Any]:
    """Write a JSONL distillation dataset and return a summary."""
    cached = list(examples)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for example in cached:
            handle.write(json.dumps(example.to_dict()) + "\n")
    return summarize_distillation_dataset(cached)
