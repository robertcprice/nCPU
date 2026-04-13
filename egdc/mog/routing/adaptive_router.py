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

from egdc.mog.benchmark import MogBenchmarkProblem, evaluate_solution, evaluate_solution_with_compiler
from egdc.mog.routing.direct_router import solve_problem_direct, _problem_to_template
from egdc.mog.routing.pathways import PathwayMemory


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
    def __init__(self, memory_root: str | Path = "egdc/mog/routing/pathway_memory"):
        self.memory = PathwayMemory(memory_root)

    def suggest_families(self, problem: MogBenchmarkProblem, top_k: int = 5) -> list[dict[str, Any]]:
        hits = self.memory.retrieve_similar(problem.description, problem.signature, top_k=top_k)
        seen = set()
        suggestions = []
        for h in hits:
            fam = h["family"]
            if fam in seen:
                continue
            seen.add(fam)
            suggestions.append(h)
        # Ensure the hand-mapped family, if any, is present.
        try:
            family, _arg_names, _arg_types, _examples = _problem_to_template(problem)
            if family not in seen:
                suggestions.append({
                    "problem_name": problem.name,
                    "family": family,
                    "code": None,
                    "metadata": {},
                    "similarity": 0.0,
                    "family_score": self.memory.family_score(family),
                    "score": self.memory.family_score(family),
                })
        except Exception:
            pass
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        return suggestions[:top_k]

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

        common_meta = {
            "loss": synth.loss,
            "description": problem.description,
            "signature": problem.signature,
            "category": problem.category,
        }
        if ok:
            self.memory.record_success(problem.name, family, synth.code, common_meta)
        else:
            err = (comp.error if comp is not None else None) or interp.error or "unknown error"
            anti_pattern = None
            if "return return" in (synth.code or ""):
                anti_pattern = "double_return"
            elif "empty separator" in err:
                anti_pattern = "python_split_empty"
            fail_meta = dict(common_meta)
            fail_meta.update({"code": synth.code, "anti_pattern": anti_pattern})
            self.memory.record_failure(problem.name, family, "execution", err, fail_meta)
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
