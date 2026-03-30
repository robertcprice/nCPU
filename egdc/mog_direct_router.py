"""Route benchmark problems to direct Mog synthesis families.

This turns the direct-synthesis template library into an actual solver: for each
benchmark problem, choose a matching structured family, synthesize Mog code,
and evaluate it with the interpreter and/or real compiler.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional

from egdc.mog_benchmark import MogBenchmarkProblem, evaluate_solution, evaluate_solution_with_compiler
from egdc.mog_direct_synth import synthesize_expression_program, DirectSynthResult


def _extract_struct_numbers(raw_code: str) -> tuple[float, float]:
    nums = re.findall(r"-?\d+", raw_code)
    if len(nums) < 2:
        raise ValueError(f"could not extract two numbers from struct literal: {raw_code}")
    return float(nums[0]), float(nums[1])


def _problem_to_template(problem: MogBenchmarkProblem) -> tuple[str, list[str], list[str] | None, list[tuple[tuple[Any, ...], float]]]:
    name = problem.name

    if name.startswith("add_two"):
        return "binary", ["a", "b"], None, [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("abs_diff"):
        return "if_cmp", ["a", "b"], None, [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("max2"):
        return "if_cmp", ["a", "b"], None, [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("clamp_0_100"):
        return "clamp_0_100", ["x"], None, [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("sign"):
        return "sign3", ["x"], None, [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("sum_to_n"):
        return "sum_to_n", ["n"], None, [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("gcd"):
        return "gcd_euclid", ["a", "b"], None, [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("lcm"):
        return "lcm_via_gcd", ["a", "b"], None, [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("array_sum"):
        return "array_sum_reduce", ["arr"], ["[i64]"], [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("array_max"):
        return "array_max_reduce", ["arr"], ["[i64]"], [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("count_occurrences"):
        return "count_occurrences_reduce", ["arr", "target"], ["[i64]", "i64"], [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("trimmed_len"):
        return "trimmed_len", ["s"], ["string"], [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("vowel_count"):
        return "vowel_count", ["s"], ["string"], [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("contains_cat"):
        return "contains_cat", ["s"], ["string"], [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("point_sum"):
        transformed = []
        for args, out in problem.test_cases:
            x, y = _extract_struct_numbers(args[0].code)
            transformed.append(((x, y), float(out)))
        return "point_sum_struct", ["x", "y"], None, transformed
    if name.startswith("safe_div_or_neg1"):
        return "safe_div_or_neg1", ["a", "b"], None, [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("positive_or_default"):
        return "positive_or_default", ["x"], None, [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("factorial"):
        return "factorial", ["n"], None, [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("fibonacci"):
        return "fibonacci", ["n"], None, [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("closure_map_sum"):
        return "closure_map_sum", ["arr"], ["[i64]"], [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("count_positive"):
        return "count_positive_reduce", ["arr"], ["[i64]"], [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("is_even"):
        return "mod2_eq0", ["x"], None, [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("digit_sum"):
        return "digit_sum_loop", ["n"], None, [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("starts_with_m"):
        return "starts_with_m", ["s"], ["string"], [(args, float(out)) for args, out in problem.test_cases]
    if name.startswith("rectangle_area"):
        transformed = []
        for args, out in problem.test_cases:
            x, y = _extract_struct_numbers(args[0].code)
            transformed.append(((x, y), float(out)))
        return "rectangle_area_struct", ["width", "height"], None, transformed

    raise KeyError(f"no direct synthesis template for problem {problem.name}")


def solve_problem_direct(problem: MogBenchmarkProblem) -> Optional[DirectSynthResult]:
    try:
        template, arg_names, arg_types, examples = _problem_to_template(problem)
    except KeyError:
        return None

    function_name = problem.signature.split("fn ", 1)[1].split("(", 1)[0].strip()
    return synthesize_expression_program(
        function_name=function_name,
        arg_names=arg_names,
        arg_types=arg_types,
        examples=examples,
        template=template,
        seed=0,
    )


def evaluate_direct_solver(problems: list[MogBenchmarkProblem], use_real_compiler: bool = True) -> dict[str, Any]:
    results = []
    solved = 0
    for problem in problems:
        synth = solve_problem_direct(problem)
        if synth is None:
            results.append({"problem": problem.name, "solved": False, "reason": "no template"})
            continue
        interp = evaluate_solution(problem, synth.code)
        comp = evaluate_solution_with_compiler(problem, synth.code) if use_real_compiler else None
        passed = interp.passed and (comp.passed if comp is not None else True)
        if passed:
            solved += 1
        results.append({
            "problem": problem.name,
            "solved": passed,
            "interp_pass": interp.passed,
            "compiler_pass": comp.passed if comp is not None else None,
            "loss": synth.loss,
            "template": synth.template,
            "code": synth.code,
        })
    return {
        "num_problems": len(problems),
        "num_solved": solved,
        "pass_rate": solved / max(len(problems), 1),
        "results": results,
    }


if __name__ == "__main__":
    from egdc.mog_benchmark import get_benchmark

    problems = get_benchmark(seed=42, variants_per_factory=1)[:25]
    summary = evaluate_direct_solver(problems, use_real_compiler=True)
    print(f"Solved {summary['num_solved']}/{summary['num_problems']} ({summary['pass_rate']:.3f})")
    for row in summary['results']:
        print(row['problem'], row.get('solved'), row.get('template'))
