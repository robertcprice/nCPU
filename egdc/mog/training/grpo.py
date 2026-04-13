"""GRPO-style rewards and utilities for Mog code generation.

This module ports the useful part first: execution-grounded reward computation.
It can be plugged into a larger GRPO trainer the same way egdc/grpo.py does for
nCPU programs, but here the reward is based on real Mog behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from egdc.mog.eval import evaluate_mog_program
from egdc.mog.execute import execute_mog
from egdc.mog.benchmark import MogBenchmarkProblem, evaluate_solution, evaluate_solution_with_compiler
from egdc.mog.solvers.differentiable import DifferentiableMogExecutor, make_numeric_testcases


@dataclass
class MogRewardBreakdown:
    reward: float
    syntax_valid: float
    static_quality: float
    interpreter_pass: float
    compiler_pass: float
    exact_compile_run: float
    differentiable_score: float
    compile_success: float
    run_success: float
    error: str | None = None


@dataclass
class MogRewardConfig:
    syntax_weight: float = 0.25
    static_weight: float = 0.25
    interpreter_weight: float = 2.0
    compiler_weight: float = 2.5
    compile_success_weight: float = 0.5
    run_success_weight: float = 0.5
    diff_exec_weight: float = 0.25
    diff_loss_scale: float = 0.1
    use_real_compiler: bool = True


class MogRewardModel:
    def __init__(self, config: MogRewardConfig | None = None, device: str = "cpu"):
        self.config = config or MogRewardConfig()
        self.soft_executor = DifferentiableMogExecutor(device=device)

    def compute_reward(self, code: str, problem: MogBenchmarkProblem) -> MogRewardBreakdown:
        static = evaluate_mog_program(code)
        syntax_valid = 1.0 if static.syntactic_validity >= 0.8 else 0.0
        static_quality = float(static.overall_score)

        interp = evaluate_solution(problem, code)
        interpreter_pass = 1.0 if interp.passed else 0.0

        compile_success = 0.0
        run_success = 0.0
        compiler_pass = 0.0
        exact_compile_run = 0.0
        compiler_error = None
        if self.config.use_real_compiler:
            compiler = evaluate_solution_with_compiler(problem, code)
            compiler_pass = 1.0 if compiler.passed else 0.0
            raw = execute_mog(code)
            compile_success = 1.0 if raw.compiled else 0.0
            run_success = 1.0 if raw.success else 0.0
            exact_compile_run = 1.0 if raw.success and raw.stdout.rstrip() == compiler.expected_output.rstrip() else 0.0
            compiler_error = raw.error or raw.stderr or compiler.error

        fn_name = problem.signature.split("fn ", 1)[1].split("(", 1)[0].strip()
        numeric_cases = make_numeric_testcases(problem)
        diff_score = 0.0
        if numeric_cases:
            loss = float(self.soft_executor.compute_problem_loss(code, fn_name, numeric_cases).detach().item())
            diff_score = 1.0 / (1.0 + self.config.diff_loss_scale * loss)

        reward = (
            self.config.syntax_weight * syntax_valid
            + self.config.static_weight * static_quality
            + self.config.interpreter_weight * interpreter_pass
            + self.config.compiler_weight * compiler_pass
            + self.config.compile_success_weight * compile_success
            + self.config.run_success_weight * run_success
            + self.config.diff_exec_weight * diff_score
        )

        return MogRewardBreakdown(
            reward=reward,
            syntax_valid=syntax_valid,
            static_quality=static_quality,
            interpreter_pass=interpreter_pass,
            compiler_pass=compiler_pass,
            exact_compile_run=exact_compile_run,
            differentiable_score=diff_score,
            compile_success=compile_success,
            run_success=run_success,
            error=compiler_error or interp.error,
        )

    def compute_batch_rewards(self, items: list[tuple[str, MogBenchmarkProblem]]) -> list[MogRewardBreakdown]:
        return [self.compute_reward(code, problem) for code, problem in items]


if __name__ == "__main__":
    from egdc.mog.benchmark import get_benchmark

    problems = get_benchmark(seed=42, variants_per_factory=1)
    problem = problems[0]
    model = MogRewardModel()
    reward = model.compute_reward(problem.reference_solution or "", problem)
    print(problem.name)
    print(reward)
