"""Persist solved items + grow the library.

Two outputs:

1. ``solved_programs.jsonl`` — an always-on append-only log keyed by
   ``task_id``. Every successful cascade run writes one row containing the
   Python source, the solver that produced it, wall time, and
   provenance. This is the dataset for any downstream paper table /
   library-rebuild / LLM-teacher distillation pipeline.

2. A live :class:`ArrayProgramLibrary` update — when a solved item is
   translatable into a ``DiscreteArrayProgram`` (array-reduction shape
   with integer I/O), the entry is written straight into the library
   JSON. This is how the autoresearch loop produces in-place library
   growth visible at the next eval.

The library-update path is best-effort: items that can't be translated
(string manipulation, nested control flow, etc.) remain in the JSONL
and are available for manual review / teacher-distillation / paper.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

from ncpu.autoresearch.types import SolvedItem


def append_solved(solved: SolvedItem, *, out_path: Path) -> None:
    """Append one SolvedItem to solved_programs.jsonl."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as fh:
        fh.write(json.dumps(solved.to_dict()) + "\n")


def load_solved(path: Path) -> list[SolvedItem]:
    """Load all SolvedItems from the persistent JSONL."""
    items: list[SolvedItem] = []
    if not path.exists():
        return items
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(SolvedItem(**{k: v for k, v in d.items() if k in SolvedItem.__dataclass_fields__}))
    return items


def dedupe_solved(path: Path) -> int:
    """Keep the latest SolvedItem per task_id. Returns count after dedupe."""
    items = load_solved(path)
    latest: dict[str, SolvedItem] = {}
    for it in items:
        latest[it.task_id] = it
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for it in latest.values():
            fh.write(json.dumps(it.to_dict()) + "\n")
    return len(latest)


def summarize_solved(items: Iterable[SolvedItem]) -> dict:
    """Counts by solver + total wall time."""
    by_solver: dict[str, int] = {}
    total_wall = 0.0
    for it in items:
        by_solver[it.solver] = by_solver.get(it.solver, 0) + 1
        total_wall += it.wall_seconds
    return {
        "total_solved": sum(by_solver.values()),
        "by_solver": by_solver,
        "total_wall_seconds": round(total_wall, 2),
    }
