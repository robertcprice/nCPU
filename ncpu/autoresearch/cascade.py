"""Solver cascade — try cheap, escalate on fail.

Each solver is a callable ``(WorkItem) -> Optional[str]`` that returns a
Python source string for the function body, or ``None`` on inability. The
cascade wraps each candidate with the problem's prompt, verifies it
against the test suite, and returns the first passing :class:`SolvedItem`.

The default cascade has one cheap local stage (``template_match``) plus
two slots expected to be filled by callers at runtime: ``llm_resample``
(needs a model + tokenizer; wired by the agent runner) and
``llm_teacher`` (needs an API key; optional).

The verify step delegates to :func:`humaneval_runner._check_solution` so
we inherit the exact same correctness criterion the eval runs use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from ncpu.autoresearch.solvers import SOLVER_FUNCTIONS, SolverFn
from ncpu.autoresearch.types import CascadeResult, SolvedItem, WorkItem


def verify_python_solution(item: WorkItem, candidate_code: str) -> tuple[bool, str]:
    """Run the full test suite against a candidate.

    ``candidate_code`` is a function body (or full function) that, when
    concatenated after ``item.prompt``, gives a runnable Python module.
    Returns ``(passed, detail_message)``.

    HumanEval-style tests expose a ``check(candidate)`` wrapper.
    MBPP-style tests are usually a raw block of top-level ``assert``
    statements over the function name directly. Support both here so the
    cascade can verify mined hard-fails from either benchmark.
    """
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    from ncpu.self_optimizing.humaneval_runner import _check_solution

    full = item.prompt + candidate_code
    if "def check(" in item.test_source:
        problem = {
            "task_id": item.task_id,
            "prompt": item.prompt,
            "test": item.test_source,
            "entry_point": item.entry_point,
        }
        return _check_solution(problem, full)

    harness = "\n".join([
        "import sys",
        "# === solution ===",
        full,
        "# === test ===",
        item.test_source,
        "print('OK')",
    ])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fh:
        fh.write(harness)
        tmp_path = fh.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if result.returncode == 0 and "OK" in result.stdout:
            return True, ""
        error = (result.stderr or result.stdout).strip().splitlines()[-1:]
        return False, (error[0] if error else "nonzero exit")
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass


@dataclass
class CascadeConfig:
    """Which solvers to try, in order, and the budget for each."""

    solver_names: list[str] = field(
        default_factory=lambda: ["template_match"]
    )
    per_solver_seconds: float = 30.0
    extra_solvers: dict[str, SolverFn] = field(default_factory=dict)
    """Solver callables keyed by name — injected by callers (e.g. llm_resample)."""


def run_cascade(
    item: WorkItem,
    *,
    config: Optional[CascadeConfig] = None,
) -> CascadeResult:
    """Try each solver in order, return first passing SolvedItem."""
    cfg = config or CascadeConfig()
    t_start = time.perf_counter()

    table: dict[str, SolverFn] = dict(SOLVER_FUNCTIONS)
    table.update(cfg.extra_solvers)

    last_error: Optional[str] = None
    for name in cfg.solver_names:
        fn = table.get(name)
        if fn is None:
            last_error = f"solver '{name}' not registered"
            continue

        t_solve_start = time.perf_counter()
        try:
            candidate = fn(item, budget_seconds=cfg.per_solver_seconds)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{name}: {type(exc).__name__}: {exc}"
            continue

        if candidate is None:
            continue

        passed, detail = verify_python_solution(item, candidate)
        wall = time.perf_counter() - t_solve_start
        if passed:
            solved = SolvedItem(
                task_id=item.task_id,
                source_benchmark=item.source_benchmark,
                solver=name,
                program_python=candidate,
                verifier_passed=True,
                wall_seconds=wall,
                provenance={"detail": detail, **item.provenance},
            )
            return CascadeResult(
                task_id=item.task_id,
                solved=True,
                solver=name,
                wall_seconds=time.perf_counter() - t_start,
                solved_item=solved,
            )
        last_error = f"{name}: {detail}"

    return CascadeResult(
        task_id=item.task_id,
        solved=False,
        solver=None,
        wall_seconds=time.perf_counter() - t_start,
        error=last_error,
    )


def run_cascade_bulk(
    items: Iterable[WorkItem],
    *,
    config: Optional[CascadeConfig] = None,
    total_budget_seconds: Optional[float] = None,
    progress: Optional[Callable[[CascadeResult, int, int], None]] = None,
) -> list[CascadeResult]:
    """Run the cascade over an iterable of WorkItems, respecting a total budget."""
    results: list[CascadeResult] = []
    items = list(items)
    start = time.perf_counter()
    for idx, item in enumerate(items):
        if total_budget_seconds is not None:
            elapsed = time.perf_counter() - start
            if elapsed >= total_budget_seconds:
                break
        r = run_cascade(item, config=config)
        results.append(r)
        if progress:
            progress(r, idx + 1, len(items))
    return results
