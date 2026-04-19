"""Dataclasses shared across the autoresearch package."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class IoPair:
    """One concrete input/output pair extracted from a test suite."""

    args: list[Any]
    kwargs: dict[str, Any]
    expected: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "args_repr": [repr(v) for v in self.args],
            "kwargs_repr": {k: repr(v) for k, v in self.kwargs.items()},
            "expected_repr": repr(self.expected),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "IoPair":
        if "args_repr" in d or "expected_repr" in d:
            return cls(
                args=[_literal_from_repr(v) for v in d.get("args_repr", [])],
                kwargs={k: _literal_from_repr(v) for k, v in d.get("kwargs_repr", {}).items()},
                expected=_literal_from_repr(d.get("expected_repr")),
            )
        return cls(
            args=d.get("args", []),
            kwargs=d.get("kwargs", {}),
            expected=d.get("expected"),
        )


@dataclass
class WorkItem:
    """A hard-fail problem the autoresearch loop should attempt to solve."""

    task_id: str
    source_benchmark: str                # "humaneval" | "mbpp" | "custom"
    prompt: str                          # function signature + docstring
    entry_point: str                     # name of the function to solve
    test_source: str                     # full Python test script (may reference `candidate`)
    io_pairs: list[IoPair]               # best-effort I/O pairs for fast synthesizers
    canonical_solution: Optional[str] = None   # reference, if available
    priority: float = 1.0                # higher = try first
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["io_pairs"] = [p.to_dict() for p in self.io_pairs]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkItem":
        io = [IoPair.from_dict(p) for p in d.get("io_pairs", [])]
        return cls(
            task_id=d["task_id"],
            source_benchmark=d["source_benchmark"],
            prompt=d["prompt"],
            entry_point=d["entry_point"],
            test_source=d["test_source"],
            io_pairs=io,
            canonical_solution=d.get("canonical_solution"),
            priority=d.get("priority", 1.0),
            provenance=d.get("provenance", {}),
        )


@dataclass
class SolvedItem:
    """A WorkItem that the cascade solved."""

    task_id: str
    source_benchmark: str
    solver: str                          # which stage in the cascade found the solution
    program_python: str                  # source code that passes the tests
    program_5tuple: Optional[dict[str, Any]] = None  # DiscreteArrayProgram if translatable
    verifier_passed: bool = True
    wall_seconds: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Budget:
    """Budget for one autoresearch session."""

    wall_seconds: float = 1800.0         # 30 min
    max_cost_usd: float = 1.0
    max_problems: int = 50
    per_problem_seconds: float = 60.0
    cost_per_gpu_hour_usd: float = 0.19  # vast.ai 3090 baseline


@dataclass
class CascadeResult:
    """Summary of a cascade run for one WorkItem."""

    task_id: str
    solved: bool
    solver: Optional[str]
    wall_seconds: float
    solved_item: Optional[SolvedItem] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.solved_item:
            d["solved_item"] = self.solved_item.to_dict()
        return d


DEFAULT_ARTIFACT_DIR = Path(".nCPU_autoresearch")
"""Where autoresearch writes work queues, solved items, status cards."""


def _literal_from_repr(source: Any) -> Any:
    """Decode a repr-serialized literal, falling back to the raw value."""
    if not isinstance(source, str):
        return source
    try:
        return ast.literal_eval(source)
    except (SyntaxError, ValueError):
        return source
