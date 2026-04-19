"""Autoresearch loop orchestrator.

Ties miner → cascade → distiller into a single session with a budget.
The runner is resumable: both the work queue and the solved-programs
artifact are JSONL append-only files that partial runs can extend.

Session contract:

* Input: a work queue (JSONL of :class:`WorkItem`).
* For each item (in priority order), invoke the cascade.
* On solve: append the :class:`SolvedItem` to the solved artifact and
  mark the work item as done.
* Budget tracking: wall seconds, max problems, and (symbolic) USD cost.
  Whichever hits first stops the loop cleanly.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from ncpu.autoresearch.cascade import CascadeConfig, run_cascade
from ncpu.autoresearch.distiller import append_solved, load_solved
from ncpu.autoresearch.miner import load_queue
from ncpu.autoresearch.types import Budget, CascadeResult, WorkItem, DEFAULT_ARTIFACT_DIR


@dataclass
class SessionReport:
    """Summary of one run of the autoresearch loop."""

    problems_attempted: int = 0
    problems_solved: int = 0
    problems_already_solved_skipped: int = 0
    wall_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    by_solver: dict[str, int] = field(default_factory=dict)
    stopped_reason: str = "done"

    def to_dict(self) -> dict:
        return {
            "problems_attempted": self.problems_attempted,
            "problems_solved": self.problems_solved,
            "problems_already_solved_skipped": self.problems_already_solved_skipped,
            "wall_seconds": round(self.wall_seconds, 2),
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "by_solver": self.by_solver,
            "stopped_reason": self.stopped_reason,
        }


def run_session(
    *,
    queue_path: Path,
    solved_path: Path,
    budget: Optional[Budget] = None,
    cascade_config: Optional[CascadeConfig] = None,
    status_path: Optional[Path] = None,
    on_result: Optional[Callable[[CascadeResult, SessionReport], None]] = None,
) -> SessionReport:
    """Run one autoresearch session.

    Items already present in ``solved_path`` are skipped so repeated runs
    converge; this is the autoresearch loop's cross-run memory.
    """
    budget = budget or Budget()
    cfg = cascade_config or CascadeConfig()

    items = load_queue(queue_path)
    already_solved_ids = {it.task_id for it in load_solved(solved_path)}

    report = SessionReport()
    t_start = time.perf_counter()

    for item in items:
        if item.task_id in already_solved_ids:
            report.problems_already_solved_skipped += 1
            continue

        elapsed = time.perf_counter() - t_start
        if elapsed >= budget.wall_seconds:
            report.stopped_reason = "wall_budget"
            break
        cost_so_far = (elapsed / 3600.0) * budget.cost_per_gpu_hour_usd
        if cost_so_far >= budget.max_cost_usd:
            report.stopped_reason = "cost_budget"
            break
        if report.problems_attempted >= budget.max_problems:
            report.stopped_reason = "problem_budget"
            break

        result = run_cascade(item, config=cfg)
        report.problems_attempted += 1
        if result.solved and result.solved_item is not None:
            report.problems_solved += 1
            report.by_solver[result.solver] = report.by_solver.get(result.solver, 0) + 1
            append_solved(result.solved_item, out_path=solved_path)

        if on_result is not None:
            on_result(result, report)

        if status_path is not None:
            _write_status(status_path, report, elapsed)

    report.wall_seconds = time.perf_counter() - t_start
    report.estimated_cost_usd = (
        report.wall_seconds / 3600.0 * budget.cost_per_gpu_hour_usd
    )
    if status_path is not None:
        _write_status(status_path, report, report.wall_seconds)
    return report


def _write_status(path: Path, report: SessionReport, elapsed: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(
            {
                "as_of_wall_seconds": round(elapsed, 2),
                **report.to_dict(),
            },
            fh,
            indent=2,
        )
