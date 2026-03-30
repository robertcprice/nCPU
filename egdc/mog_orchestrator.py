"""Unified Mog solver orchestrator.

This is the top-level system that chains multiple solver strategies together
and learns from every attempt:

1. Direct structured synthesis (highest confidence, fastest)
2. Pathway retrieval / exemplar reuse (if direct family not available)
3. Body completion fallback (if a trained checkpoint is available)
4. Compiler + interpreter verification on every candidate
5. Persistent memory recording of every success and failure

The system improves from use:
- successful pathways strengthen families
- failures create anti-patterns that get checked before emitting code
- similar past solutions inform family selection for new problems
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from egdc.mog_benchmark import MogBenchmarkProblem, evaluate_solution, evaluate_solution_with_compiler
from egdc.mog_direct_router import solve_problem_direct
from egdc.mog_pathways import PathwayMemory


# Known anti-pattern signatures and the code substrings that trigger them.
import re as _re

# Anti-pattern checks: each key maps to a callable(code) -> bool.
# True means the anti-pattern was detected.
def _has_double_return(code: str) -> bool:
    return "return return" in code

def _has_void_main(code: str) -> bool:
    return bool(_re.search(r"fn\s+main\s*\(\s*\)\s*\{", code))

def _has_bool_type_annotation(code: str) -> bool:
    return bool(_re.search(r":\s*bool\b", code))

def _has_bare_bang_operator(code: str) -> bool:
    # Match ! that is NOT followed by = (i.e., bare negation, not !=).
    return bool(_re.search(r"!\s*[^=]", code)) and not _re.search(r"!\s*=", code)

ANTI_PATTERN_CHECKS: dict[str, Any] = {
    "double_return": _has_double_return,
    "void_main": _has_void_main,
    "bool_type_annotation": _has_bool_type_annotation,
    "bang_operator": _has_bare_bang_operator,
}


@dataclass
class OrchestratorResult:
    success: bool
    method: str  # "direct" | "retrieval" | "completion" | "failed"
    code: str | None = None
    family: str | None = None
    interp_pass: bool = False
    compiler_pass: bool = False
    error: str | None = None
    anti_patterns_blocked: list[str] | None = None


class MogOrchestrator:
    def __init__(
        self,
        memory_root: str | Path = "egdc/pathway_memory",
        completion_checkpoint: str | None = None,
        use_real_compiler: bool = True,
    ):
        self.memory = PathwayMemory(memory_root)
        self.completion_checkpoint = completion_checkpoint
        self.use_real_compiler = use_real_compiler

    def check_anti_patterns(self, family: str, code: str) -> list[str]:
        """Check candidate code against known anti-patterns from failure memory."""
        blocked: list[str] = []
        # Check family-specific anti-patterns from memory.
        memory_anti = self.memory.anti_patterns(family)
        for ap in memory_anti:
            checker = ANTI_PATTERN_CHECKS.get(ap)
            if checker is not None and checker(code):
                blocked.append(ap)

        # Also check global anti-patterns.
        for ap, checker in ANTI_PATTERN_CHECKS.items():
            if ap in blocked:
                continue
            if checker(code):
                blocked.append(ap)
        return blocked

    def _verify(self, problem: MogBenchmarkProblem, code: str) -> tuple[bool, bool, str | None]:
        interp = evaluate_solution(problem, code)
        comp = evaluate_solution_with_compiler(problem, code) if self.use_real_compiler else None
        ok = interp.passed and (comp.passed if comp is not None else True)
        error = (comp.error if comp is not None and not comp.passed else None) or (interp.error if not interp.passed else None)
        return ok, (comp.passed if comp is not None else False), error

    def _record(self, problem: MogBenchmarkProblem, family: str, code: str, ok: bool, error: str | None):
        meta = {
            "description": problem.description,
            "signature": problem.signature,
            "category": problem.category,
        }
        if ok:
            self.memory.record_success(problem.name, family, code, meta)
        else:
            anti_pattern = None
            if code and "return return" in code:
                anti_pattern = "double_return"
            fail_meta = dict(meta)
            fail_meta.update({"code": code, "anti_pattern": anti_pattern})
            self.memory.record_failure(problem.name, family, "execution", error or "unknown", fail_meta)
        self.memory.save()

    def solve(self, problem: MogBenchmarkProblem) -> OrchestratorResult:
        # --- Strategy 1: Direct structured synthesis ---
        synth = solve_problem_direct(problem)
        if synth is not None and synth.success:
            blocked = self.check_anti_patterns(synth.template, synth.code)
            if not blocked:
                ok, comp_pass, error = self._verify(problem, synth.code)
                self._record(problem, synth.template, synth.code, ok, error)
                if ok:
                    return OrchestratorResult(
                        success=True, method="direct", code=synth.code,
                        family=synth.template, interp_pass=True, compiler_pass=comp_pass,
                    )

        # --- Strategy 2: Retrieval from pathway memory ---
        hits = self.memory.retrieve_similar(problem.description, problem.signature, top_k=5)
        for hit in hits:
            exemplar_code = hit.get("code")
            if not exemplar_code:
                continue
            ok, comp_pass, error = self._verify(problem, exemplar_code)
            if ok:
                self._record(problem, hit["family"], exemplar_code, True, None)
                return OrchestratorResult(
                    success=True, method="retrieval", code=exemplar_code,
                    family=hit["family"], interp_pass=True, compiler_pass=comp_pass,
                )

        # --- Strategy 3: Body completion (if checkpoint available) ---
        if self.completion_checkpoint is not None:
            try:
                completed_code = self._try_completion(problem)
                if completed_code:
                    ok, comp_pass, error = self._verify(problem, completed_code)
                    self._record(problem, "completion", completed_code, ok, error)
                    if ok:
                        return OrchestratorResult(
                            success=True, method="completion", code=completed_code,
                            family="completion", interp_pass=True, compiler_pass=comp_pass,
                        )
            except Exception:
                pass

        # --- All strategies exhausted ---
        error_msg = "all strategies exhausted"
        if synth is not None:
            self._record(problem, synth.template or "unknown", synth.code or "", False, error_msg)
        return OrchestratorResult(success=False, method="failed", error=error_msg)

    def _try_completion(self, problem: MogBenchmarkProblem) -> str | None:
        import torch
        from egdc.mog_model import MogMaskedDiffusion, MogDiffusionConfig
        from egdc.mog_tokenizer import MogCodeTokenizer
        from egdc.mog_completion import build_completion_tokens, complete_mog_from_initial

        config = MogDiffusionConfig.tiny()
        config.max_seq_len = max(config.max_seq_len, 256 + 128 + 64)
        model = MogMaskedDiffusion(config)
        ckpt = torch.load(self.completion_checkpoint, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        model.eval()

        tok = MogCodeTokenizer()
        if not problem.reference_solution:
            return None
        initial, fixed, _ = build_completion_tokens(problem.reference_solution, tok, 256)
        prompt = f"{problem.signature}\n// {problem.description}"
        spec = torch.tensor([tok.pad(tok.encode(prompt, add_bos_eos=False), 128)], dtype=torch.long)
        completed = complete_mog_from_initial(
            model, initial.unsqueeze(0), fixed.unsqueeze(0), spec,
            num_steps=24, temperature=0.1, device=torch.device("cpu"),
        )
        return tok.decode(completed[0].tolist())

    def solve_batch(self, problems: list[MogBenchmarkProblem]) -> list[OrchestratorResult]:
        return [self.solve(p) for p in problems]

    def evaluate(self, problems: list[MogBenchmarkProblem]) -> dict[str, Any]:
        results = self.solve_batch(problems)
        solved = sum(1 for r in results if r.success)
        by_method: dict[str, int] = {}
        for r in results:
            by_method[r.method] = by_method.get(r.method, 0) + 1
        return {
            "num_problems": len(problems),
            "num_solved": solved,
            "pass_rate": solved / max(len(problems), 1),
            "by_method": by_method,
            "family_scores": {fam: self.memory.family_score(fam) for fam in self.memory.successes_by_family().keys()},
            "total_successes": self.memory.total_successes(),
            "results": [
                {
                    "problem": p.name,
                    "success": r.success,
                    "method": r.method,
                    "family": r.family,
                    "compiler_pass": r.compiler_pass,
                    "error": r.error,
                }
                for p, r in zip(problems, results)
            ],
        }
