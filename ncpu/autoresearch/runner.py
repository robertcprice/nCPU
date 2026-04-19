"""Autoresearch loop orchestrator.

Ties miner → cascade → distiller into a single session with a budget.
The runner is resumable: both the work queue and the solved-programs
artifact are JSONL append-only files that partial runs can extend.

Session contract:

* Input: a work queue (JSONL of :class:`WorkItem`).
* For each item (in priority order):
    1. Check the :class:`CompoundingStore` for an exact prompt-cache
       hit or a task-id hit; if found, return the cached program
       without invoking the cascade. This is the always-compounding
       short-circuit.
    2. Else invoke the cascade, verify, and on success record the
       solved item into the store (which updates the log, prompt
       cache, and temperature stats in one call).
* Budget tracking: wall seconds, max problems, and (symbolic) USD cost.
  Whichever hits first stops the loop cleanly.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from ncpu.autoresearch.cascade import CascadeConfig, run_cascade, verify_python_solution
from ncpu.autoresearch.compounding_store import (
    CompoundingStore,
    CompoundingStoreConfig,
)
from ncpu.autoresearch.distiller import load_solved
from ncpu.autoresearch.miner import load_queue
from ncpu.autoresearch.types import (
    Budget,
    CascadeResult,
    DEFAULT_ARTIFACT_DIR,
    SolvedItem,
    WorkItem,
)


@dataclass
class SessionReport:
    """Summary of one run of the autoresearch loop."""

    problems_attempted: int = 0
    problems_solved: int = 0
    problems_already_solved_skipped: int = 0
    store_hits: int = 0
    wall_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    by_solver: dict[str, int] = field(default_factory=dict)
    stopped_reason: str = "done"

    def to_dict(self) -> dict:
        return {
            "problems_attempted": self.problems_attempted,
            "problems_solved": self.problems_solved,
            "problems_already_solved_skipped": self.problems_already_solved_skipped,
            "store_hits": self.store_hits,
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
    store: Optional[CompoundingStore] = None,
) -> SessionReport:
    """Run one autoresearch session.

    Items already present in ``solved_path`` (legacy log) or matching
    a cached entry in ``store`` are skipped so repeated runs converge
    on a steady state. The store hit also counts toward
    ``problems_solved`` so composite dashboards see continuous growth.
    """
    budget = budget or Budget()
    cfg = cascade_config or CascadeConfig()
    if store is None:
        store = CompoundingStore(CompoundingStoreConfig(
            artifact_dir=solved_path.parent,
            solved_log_name=solved_path.name,
        ))

    items = load_queue(queue_path)
    already_solved_ids = {it.task_id for it in load_solved(solved_path)}

    report = SessionReport()
    t_start = time.perf_counter()

    for item in items:
        # Task-id already in legacy log → skip entirely.
        if item.task_id in already_solved_ids:
            report.problems_already_solved_skipped += 1
            continue

        # Store hit via exact prompt hash → skip cascade, still verify
        # the cached program against the test suite so a stale entry
        # never silently passes through.
        hit = store.check_prompt(item)
        if hit is not None:
            passed, _detail = verify_python_solution(item, hit.program_python)
            if passed:
                report.store_hits += 1
                report.problems_solved += 1
                fake = CascadeResult(
                    task_id=item.task_id, solved=True, solver="store_hit",
                    wall_seconds=0.0,
                    solved_item=SolvedItem(
                        task_id=item.task_id,
                        source_benchmark=item.source_benchmark,
                        solver="store_hit",
                        program_python=hit.program_python,
                        verifier_passed=True,
                        wall_seconds=0.0,
                        provenance={"source": hit.source, **hit.provenance},
                    ),
                )
                if on_result is not None:
                    on_result(fake, report)
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
            store.record(result.solved_item, work_item=item)

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
