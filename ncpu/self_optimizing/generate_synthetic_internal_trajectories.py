"""Generate synthetic hidden SOME trajectories with nonzero latent-memory dynamics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ncpu.self_optimizing.internal_controller import (
    BufferedInternalController,
    InternalControllerConfig,
    InternalDeliberationTask,
)
from ncpu.self_optimizing.trajectory_logger import TrajectoryLogger


@dataclass(frozen=True)
class SyntheticTaskSpec:
    name: str
    prompt: str
    test_cases: list[dict[str, object]]
    failure_mode: str


TASK_SPECS = [
    SyntheticTaskSpec(
        name="fibonacci",
        prompt="Write fib(n) in Python.",
        test_cases=[
            {"input": {"n": 0}, "expected": 0},
            {"input": {"n": 1}, "expected": 1},
            {"input": {"n": 6}, "expected": 8},
        ],
        failure_mode="zero_stub",
    ),
    SyntheticTaskSpec(
        name="factorial",
        prompt="Write factorial(n) in Python.",
        test_cases=[
            {"input": {"n": 0}, "expected": 1},
            {"input": {"n": 1}, "expected": 1},
            {"input": {"n": 5}, "expected": 120},
        ],
        failure_mode="identity_stub",
    ),
    SyntheticTaskSpec(
        name="sum_to_n",
        prompt="Write sum_to_n(n) in Python.",
        test_cases=[
            {"input": {"n": 0}, "expected": 0},
            {"input": {"n": 1}, "expected": 1},
            {"input": {"n": 5}, "expected": 15},
        ],
        failure_mode="off_by_one",
    ),
    SyntheticTaskSpec(
        name="is_even",
        prompt="Write is_even(n) in Python.",
        test_cases=[
            {"input": {"n": 0}, "expected": True},
            {"input": {"n": 1}, "expected": False},
            {"input": {"n": 8}, "expected": True},
        ],
        failure_mode="always_false",
    ),
]


def _bad_candidate(spec: SyntheticTaskSpec, *, variant: int) -> str:
    if spec.name == "fibonacci":
        return "def fib(n):\n    return 0\n"
    if spec.name == "factorial":
        return "def factorial(n):\n    return n\n"
    if spec.name == "sum_to_n":
        return "def sum_to_n(n):\n    return sum(range(n))\n"
    if spec.name == "is_even":
        return "def is_even(n):\n    return False\n"
    return "def solve(x):\n    return x\n"


def _good_candidate(spec: SyntheticTaskSpec) -> str:
    if spec.name == "fibonacci":
        return (
            "def fib(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fib(n - 1) + fib(n - 2)\n"
        )
    if spec.name == "factorial":
        return (
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
        )
    if spec.name == "sum_to_n":
        return "def sum_to_n(n):\n    return sum(range(n + 1))\n"
    if spec.name == "is_even":
        return "def is_even(n):\n    return n % 2 == 0\n"
    return "def solve(x):\n    return x\n"


def build_synthetic_provider(spec: SyntheticTaskSpec, *, behavior_variant: int):
    good = _good_candidate(spec)
    bad = _bad_candidate(spec, variant=behavior_variant)

    def provider(prompt: str) -> str:
        normalized = prompt.lower()
        if "think privately" in normalized:
            if behavior_variant % 3 == 0:
                return f"Use the exact {spec.name} recurrence and watch the base case."
            return f"Plan a repair for {spec.name}; the current draft misses an edge case."
        if "re-planning a hidden repair attempt" in normalized:
            return f"Repair {spec.name} by fixing {spec.failure_mode} and preserving the function signature."
        if "repairing a hidden candidate" in normalized:
            return good
        if "refining a hidden candidate" in normalized:
            return good
        return bad

    return provider


def generate_synthetic_internal_trajectories(
    *,
    output_dir: str | Path,
    num_trajectories: int,
) -> dict[str, object]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    counts_by_task: dict[str, int] = {}
    generated_paths: list[str] = []

    for index in range(num_trajectories):
        spec = TASK_SPECS[index % len(TASK_SPECS)]
        counts_by_task[spec.name] = counts_by_task.get(spec.name, 0) + 1
        provider = build_synthetic_provider(spec, behavior_variant=index)
        trajectory_path = destination / f"{index + 1:04d}_{spec.name}.jsonl"
        controller = BufferedInternalController(
            llm_provider=provider,
            config=InternalControllerConfig(max_generation_attempts=3),
            trajectory_logger=TrajectoryLogger(str(trajectory_path)),
        )
        workspace = controller.run_task(
            InternalDeliberationTask(
                name=spec.name,
                prompt=spec.prompt,
                test_cases=spec.test_cases,
            )
        )
        if workspace.status != "committed" or not workspace.committed_verified:
            raise RuntimeError(f"Synthetic trajectory generation failed for {spec.name} at index {index}")
        generated_paths.append(str(trajectory_path))

    return {
        "output_dir": str(destination),
        "num_trajectories": len(generated_paths),
        "tasks": counts_by_task,
        "paths": generated_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic hidden SOME trajectories")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-trajectories", type=int, default=32)
    args = parser.parse_args()

    summary = generate_synthetic_internal_trajectories(
        output_dir=args.output_dir,
        num_trajectories=args.num_trajectories,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
