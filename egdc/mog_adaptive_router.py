"""Adaptive router for Mog direct synthesis.

Wraps the direct benchmark router with persistent pathway memory:
- records solved programs
- records failures
- exposes family scores and evaluation summaries

This is the first practical self-improvement loop: the system doesn't merely
solve tasks, it accumulates a reusable pathway/failure history from doing so.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from egdc.mog_benchmark import MogBenchmarkProblem, evaluate_solution, evaluate_solution_with_compiler
from egdc.mog_direct_router import solve_problem_direct
from egdc.mog_pathways import PathwayMemory


@dataclass
class AdaptiveSolveResult:
    success: bool
    family: str | None
    code: str | None
    interp_pass: bool = False
    compiler_pass: bool = False
    error: str | None = None
    loss: float | None = None


class AdaptiveMogRouter:
    def __init__(self, memory_root: str | Path = "egdc/pathway_memory"):
        self.memory = PathwayMemory(memory_root)

    def solve(self, problem: MogBenchmarkProblem, use_real_compiler: bool = True) -> AdaptiveSolveResult:
        synth = solve_problem_direct(problem)
        if synth is None:
            self.memory.record_failure(problem.name, "none", "routing", "no direct family", {})
            self.memory.save()
            return AdaptiveSolveResult(False, None, None, error="no direct family")

        family = synth.template
        interp = evaluate_solution(problem, synth.code)
        comp = evaluate_solution_with_compiler(problem, synth.code) if use_real_compiler else None
        ok = interp.passed and (comp.passed if comp is not None else True)

        if ok:
            self.memory.record_success(problem.name, family, synth.code, {"loss": synth.loss})
        else:
            err = (comp.error if comp is not None else None) or interp.error or "unknown error"
            self.memory.record_failure(problem.name, family, "execution", err, {"code": synth.code, "loss": synth.loss})
        self.memory.save()

        return AdaptiveSolveResult(
            success=ok,
            family=family,
            code=synth.code,
            interp_pass=interp.passed,
            compiler_pass=(comp.passed if comp is not None else False),
            error=((comp.error if comp is not None else None) or interp.error),
            loss=synth.loss,
        )

    def evaluate(self, problems: list[MogBenchmarkProblem], use_real_compiler: bool = True) -> dict[str, Any]:
        rows = []
        solved = 0
        for p in problems:
            r = self.solve(p, use_real_compiler=use_real_compiler)
            if r.success:
                solved += 1
            rows.append({
                "problem": p.name,
                "success": r.success,
                "family": r.family,
                "loss": r.loss,
                "interp_pass": r.interp_pass,
                "compiler_pass": r.compiler_pass,
                "error": r.error,
                "code": r.code,
            })
        return {
            "num_problems": len(problems),
            "num_solved": solved,
            "pass_rate": solved / max(len(problems), 1),
            "family_scores": {fam: self.memory.family_score(fam) for fam in self.memory.successes_by_family().keys()},
            "results": rows,
        }
