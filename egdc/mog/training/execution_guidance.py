"""Execution-guided sampling for Mog.

This is the first practical execution-guided layer for Mog generation.
Rather than pretending we already have a full token-logit -> soft-AST ->
soft-executor pipeline, this module does what is useful right now:

1. Sample candidate Mog programs from the diffusion model.
2. Score them with real execution signals:
   - parser / syntax validity
   - differentiable execution loss on benchmark test cases
   - interpreter exact-output pass rate
   - optional real compiler compile/run pass rate
3. Pick the best candidate or rerank a beam.

This gives execution-guided generation today, while the lower-level soft
compiler continues to evolve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from egdc.mog.tokenizer import MogCodeTokenizer
from egdc.mog.train import generate_mog
from egdc.mog.eval import evaluate_mog_program
from egdc.mog.benchmark import MogBenchmarkProblem, evaluate_solution, evaluate_solution_with_compiler
from egdc.mog.solvers.differentiable import DifferentiableMogExecutor, make_numeric_testcases


@dataclass
class MogGuidanceScore:
    total_score: float
    syntax_score: float
    static_score: float
    differentiable_loss: float
    interpreter_pass: float
    compiler_pass: float
    output: str = ""
    error: str | None = None


class MogExecutionGuidedScorer:
    def __init__(
        self,
        problem: MogBenchmarkProblem,
        device: str | torch.device = "cpu",
        use_real_compiler: bool = False,
        weights: dict[str, float] | None = None,
    ):
        self.problem = problem
        self.device = torch.device(device)
        self.use_real_compiler = use_real_compiler
        self.weights = weights or {
            "static": 0.5,
            "interp": 2.0,
            "compiler": 2.5,
            "diff_loss": -0.25,
        }
        self.soft_executor = DifferentiableMogExecutor(device=self.device)

    def score(self, code: str) -> MogGuidanceScore:
        static = evaluate_mog_program(code)
        static_score = float(static.overall_score)
        syntax_score = float(static.syntactic_validity)

        interp_result = evaluate_solution(self.problem, code)
        interpreter_pass = 1.0 if interp_result.passed else 0.0

        compiler_pass = 0.0
        if self.use_real_compiler:
            compiler_result = evaluate_solution_with_compiler(self.problem, code)
            compiler_pass = 1.0 if compiler_result.passed else 0.0

        fn_name = self.problem.signature.split("fn ", 1)[1].split("(", 1)[0].strip()
        numeric_cases = make_numeric_testcases(self.problem)
        diff_loss = float(self.soft_executor.compute_problem_loss(code, fn_name, numeric_cases).detach().item()) if numeric_cases else 0.0

        total = (
            self.weights["static"] * static_score
            + self.weights["interp"] * interpreter_pass
            + self.weights["compiler"] * compiler_pass
            + self.weights["diff_loss"] * diff_loss
        )

        return MogGuidanceScore(
            total_score=total,
            syntax_score=syntax_score,
            static_score=static_score,
            differentiable_loss=diff_loss,
            interpreter_pass=interpreter_pass,
            compiler_pass=compiler_pass,
            output=interp_result.actual_output,
            error=interp_result.error,
        )


@torch.no_grad()
def execution_guided_generate_mog(
    model,
    tokenizer: MogCodeTokenizer,
    problem: MogBenchmarkProblem,
    seq_len: int,
    num_candidates: int = 8,
    num_steps: int = 64,
    temperature: float = 0.8,
    device: Optional[torch.device] = None,
    use_real_compiler: bool = False,
) -> dict[str, Any]:
    """Generate several candidates and pick the best by execution score."""
    if device is None:
        device = next(model.parameters()).device

    # Condition on signature/description prefix.
    prompt = f"{problem.signature}\n// {problem.description}"
    spec_tokens = torch.tensor([tokenizer.pad(tokenizer.encode(prompt, add_bos_eos=False), 64)], dtype=torch.long, device=device)

    scorer = MogExecutionGuidedScorer(problem, device=device, use_real_compiler=use_real_compiler)

    best = None
    candidates = []
    for _ in range(num_candidates):
        sample = generate_mog(
            model,
            spec_tokens=spec_tokens,
            seq_len=seq_len,
            num_steps=num_steps,
            temperature=temperature,
            device=device,
        )
        code = tokenizer.decode(sample[0].tolist())
        score = scorer.score(code)
        record = {"code": code, "score": score}
        candidates.append(record)
        if best is None or score.total_score > best["score"].total_score:
            best = record

    assert best is not None
    return {
        "best_code": best["code"],
        "best_score": best["score"],
        "candidates": candidates,
        "problem": problem,
    }


if __name__ == "__main__":
    from egdc.mog.benchmark import get_benchmark

    benchmark = get_benchmark(seed=42, variants_per_factory=1)
    problem = benchmark[0]
    scorer = MogExecutionGuidedScorer(problem)
    score = scorer.score(problem.reference_solution or "")
    print("problem:", problem.name)
    print("score:", score)
