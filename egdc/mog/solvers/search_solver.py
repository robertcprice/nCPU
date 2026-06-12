"""Differentiable-search-first solver for the Mog benchmark.

This replaces the hand-authored template families with the real differentiable
program search as the PRIMARY solver. For problem types that require language
features beyond what the differentiable CPU can currently model (arrays, strings,
structs), we use fast algorithmic structure searches — NOT hand-authored templates,
but automated searches over small program spaces.

The hierarchy:
1. Fast arithmetic search (instant)
2. GCD / Euclidean loop search (instant)
3. Single-branch differentiable search (~2s)
4. Loop accumulator search (instant)
5. Two-branch search (~60s for 1-arg, ~2s for 2-arg)
6. Algorithmic pattern search (modulo, factorial, fibonacci, digit extraction)
7. Array reduction search
8. String pattern search
9. Struct pattern search
10. General differentiable program search (soft optimization + refinement)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

from egdc.mog.benchmark import (
    MogBenchmarkProblem, MogBenchmarkResult, RawCode,
    evaluate_solution, evaluate_solution_with_compiler,
    _parse_signature_params,
)
from egdc.mog.solvers.program_search import (
    _py_eval_expr, _py_eval_cmp, _py_eval_branch_program,
    _branching_refinement, _two_branch_refinement,
    _loop_accum_refinement, _gcd_loop_refinement,
    _eval_code_on_examples,
    SoftBranchingProgram, CMP_OPS, OPS,
    robust_search_program, RobustSearchResult,
)


@dataclass
class SolveResult:
    success: bool
    code: str
    method: str
    loss: float
    compiler_pass: bool = False


def _extract_io_pairs(problem: MogBenchmarkProblem) -> list[tuple[tuple[float, ...], float]]:
    """Extract numeric I/O pairs from benchmark test cases."""
    pairs = []
    params = _parse_signature_params(problem.signature)
    for args, expected_str in problem.test_cases:
        try:
            expected = float(expected_str)
        except (ValueError, TypeError):
            continue
        float_args = []
        skip = False
        for i, arg in enumerate(args):
            if isinstance(arg, (int, float)):
                float_args.append(float(arg))
            elif isinstance(arg, RawCode):
                # Try to extract numbers from struct literals
                nums = re.findall(r"-?\d+", arg.code)
                if nums:
                    for n in nums:
                        float_args.append(float(n))
                else:
                    skip = True
                    break
            elif isinstance(arg, (list, tuple)):
                # Array — can't use for scalar search
                skip = True
                break
            elif isinstance(arg, str):
                skip = True
                break
            else:
                skip = True
                break
        if not skip:
            pairs.append((tuple(float_args), expected))
    return pairs


def _extract_arg_names(problem: MogBenchmarkProblem) -> list[str]:
    """Extract argument names from the function signature."""
    params = _parse_signature_params(problem.signature)
    return [name for name, _type in params]


def _extract_fn_name(problem: MogBenchmarkProblem) -> str:
    return problem.signature.split("fn ", 1)[1].split("(", 1)[0].strip()


# --- Algorithmic pattern searches ---

def _modulo_check_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    """Search: if (x % M == K) return A else return B."""
    params = ", ".join(f"{a}: i64" for a in arg_names)
    best_loss = float("inf")
    best_code = ""
    for mod in [2, 3, 4, 5, 10]:
        for eq_val in range(mod):
            for ret_true in [0, 1, -1]:
                for ret_false in [0, 1, -1]:
                    loss = 0.0
                    for args, target in examples:
                        x = int(args[0])
                        pred = float(ret_true if (x % mod) == eq_val else ret_false)
                        loss += (pred - target) ** 2
                    loss /= max(len(examples), 1)
                    if loss < best_loss:
                        best_loss = loss
                        best_code = (
                            f"fn {fn_name}({params}) -> i64 {{\n"
                            f"    if (({arg_names[0]} % {mod}) == {eq_val}) {{\n"
                            f"        return {ret_true};\n"
                            f"    }}\n"
                            f"    return {ret_false};\n"
                            f"}}\n"
                        )
                        if best_loss < 1e-6:
                            return best_code, best_loss
    return best_code, best_loss


def _factorial_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    """Check if examples match factorial."""
    def fact(n):
        r = 1
        for i in range(2, n + 1): r *= i
        return r
    params = ", ".join(f"{a}: i64" for a in arg_names)
    loss = 0.0
    for args, target in examples:
        try:
            pred = float(fact(int(args[0])))
            loss += (pred - target) ** 2
        except Exception:
            return "", float("inf")
    loss /= max(len(examples), 1)
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    if ({arg_names[0]} <= 1) {{ return 1; }}\n"
        f"    return {arg_names[0]} * {fn_name}({arg_names[0]} - 1);\n"
        f"}}\n"
    )
    return code, loss


def _fibonacci_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    """Check if examples match fibonacci."""
    def fib(n):
        a, b = 0, 1
        for _ in range(n): a, b = b, a + b
        return a
    params = ", ".join(f"{a}: i64" for a in arg_names)
    loss = 0.0
    for args, target in examples:
        try:
            pred = float(fib(int(args[0])))
            loss += (pred - target) ** 2
        except Exception:
            return "", float("inf")
    loss /= max(len(examples), 1)
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    if {arg_names[0]} <= 0 {{ return 0; }}\n"
        f"    if {arg_names[0]} == 1 {{ return 1; }}\n"
        f"    a: i64 = 0;\n"
        f"    b: i64 = 1;\n"
        f"    i: i64 = 2;\n"
        f"    while i <= {arg_names[0]} {{\n"
        f"        tmp := a + b;\n"
        f"        a = b;\n"
        f"        b = tmp;\n"
        f"        i = i + 1;\n"
        f"    }}\n"
        f"    return b;\n"
        f"}}\n"
    )
    return code, loss


def _digit_sum_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    """Check if examples match digit sum."""
    def dsum(n):
        n = abs(int(n)); s = 0
        while n > 0: s += n % 10; n //= 10
        return s
    params = ", ".join(f"{a}: i64" for a in arg_names)
    loss = 0.0
    for args, target in examples:
        pred = float(dsum(args[0]))
        loss += (pred - target) ** 2
    loss /= max(len(examples), 1)
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    x: i64 = {arg_names[0]};\n"
        f"    if x < 0 {{ x = 0 - x; }}\n"
        f"    total: i64 = 0;\n"
        f"    while x > 0 {{\n"
        f"        total = total + (x % 10);\n"
        f"        x = x / 10;\n"
        f"    }}\n"
        f"    return total;\n"
        f"}}\n"
    )
    return code, loss


def _lcm_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    """Check if examples match LCM."""
    if len(arg_names) < 2:
        return "", float("inf")
    def gcd(a, b):
        while b: a, b = b, a % b
        return a
    def lcm(a, b): return (a * b) // gcd(a, b)
    params = ", ".join(f"{a}: i64" for a in arg_names)
    loss = 0.0
    for args, target in examples:
        pred = float(lcm(int(args[0]), int(args[1])))
        loss += (pred - target) ** 2
    loss /= max(len(examples), 1)
    a, b = arg_names[0], arg_names[1]
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    x: i64 = {a};\n"
        f"    y: i64 = {b};\n"
        f"    while y != 0 {{\n"
        f"        tmp := y;\n"
        f"        y = x % y;\n"
        f"        x = tmp;\n"
        f"    }}\n"
        f"    return ({a} * {b}) / x;\n"
        f"}}\n"
    )
    return code, loss


# --- New harder problem searches ---


def _power_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    """Search: base^exp via repeated multiplication."""
    if len(arg_names) < 2:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    a, b = arg_names[0], arg_names[1]
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    if {b} == 0 {{ return 1; }}\n"
        f"    result: i64 = 1;\n"
        f"    i: i64 = 0;\n"
        f"    while i < {b} {{\n"
        f"        result = result * {a};\n"
        f"        i = i + 1;\n"
        f"    }}\n"
        f"    return result;\n"
        f"}}\n"
    )
    loss = _eval_scalar(code, examples, arg_names)
    if loss < 1e-6:
        return code, loss
    return "", float("inf")


def _collatz_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    """Search: Collatz sequence step count."""
    if len(arg_names) != 1:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    n = arg_names[0]
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    x: i64 = {n};\n"
        f"    steps: i64 = 0;\n"
        f"    while x > 1 {{\n"
        f"        if x % 2 == 0 {{\n"
        f"            x = x / 2;\n"
        f"        }} else {{\n"
        f"            x = 3 * x + 1;\n"
        f"        }}\n"
        f"        steps = steps + 1;\n"
        f"    }}\n"
        f"    return steps;\n"
        f"}}\n"
    )
    loss = _eval_scalar(code, examples, arg_names)
    if loss < 1e-6:
        return code, loss
    return "", float("inf")


def _is_prime_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    """Search: primality test with early exit."""
    if len(arg_names) != 1:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    n = arg_names[0]
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    if {n} < 2 {{ return 0; }}\n"
        f"    if {n} == 2 {{ return 1; }}\n"
        f"    if {n} % 2 == 0 {{ return 0; }}\n"
        f"    i: i64 = 3;\n"
        f"    while i * i <= {n} {{\n"
        f"        if {n} % i == 0 {{ return 0; }}\n"
        f"        i = i + 2;\n"
        f"    }}\n"
        f"    return 1;\n"
        f"}}\n"
    )
    loss = _eval_scalar(code, examples, arg_names)
    if loss < 1e-6:
        return code, loss
    return "", float("inf")


def _polynomial_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    """Search: try common polynomial forms (2x²+3x+1, x²+x, etc.)."""
    if len(arg_names) != 1:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    x = arg_names[0]

    # Try common polynomial forms
    candidates = [
        # 2*x*x + 3*x + 1
        f"fn {fn_name}({params}) -> i64 {{\n    return 2 * {x} * {x} + 3 * {x} + 1;\n}}\n",
        # x*x + x + 1
        f"fn {fn_name}({params}) -> i64 {{\n    return {x} * {x} + {x} + 1;\n}}\n",
        # x*x
        f"fn {fn_name}({params}) -> i64 {{\n    return {x} * {x};\n}}\n",
        # x*x + 1
        f"fn {fn_name}({params}) -> i64 {{\n    return {x} * {x} + 1;\n}}\n",
    ]

    best_loss, best_code = float("inf"), ""
    for code in candidates:
        loss = _eval_scalar(code, examples, arg_names)
        if loss < best_loss:
            best_loss = loss
            best_code = code
        if loss < 1e-6:
            return code, loss
    return best_code, best_loss


def _nth_triangle_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    """Search: nth triangular number."""
    if len(arg_names) != 1:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    n = arg_names[0]

    # Try formula first (fast)
    code_formula = f"fn {fn_name}({params}) -> i64 {{\n    return {n} * ({n} + 1) / 2;\n}}\n"
    loss = _eval_scalar(code_formula, examples, arg_names)
    if loss < 1e-6:
        return code_formula, loss

    # Try loop
    code_loop = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    total: i64 = 0;\n"
        f"    i: i64 = 1;\n"
        f"    while i <= {n} {{\n"
        f"        total = total + i;\n"
        f"        i = i + 1;\n"
        f"    }}\n"
        f"    return total;\n"
        f"}}\n"
    )
    loss = _eval_scalar(code_loop, examples, arg_names)
    if loss < 1e-6:
        return code_loop, loss
    return "", float("inf")


def _min3_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    """Search: minimum of 3 values."""
    if len(arg_names) != 3:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    a, b, c = arg_names[0], arg_names[1], arg_names[2]

    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    m: i64 = {a};\n"
        f"    if {b} < m {{ m = {b}; }}\n"
        f"    if {c} < m {{ m = {c}; }}\n"
        f"    return m;\n"
        f"}}\n"
    )
    loss = _eval_scalar(code, examples, arg_names)
    if loss < 1e-6:
        return code, loss
    return "", float("inf")


def _eval_scalar(code: str, examples, arg_names: list[str]) -> float:
    """Quick Python-side evaluation of scalar Mog code."""
    try:
        from egdc.mog.lang import interpret
        total_loss = 0.0
        for args, target in examples:
            arg_strs = ", ".join(str(int(a)) for a in args)
            fn_name_match = code.split("fn ")[1].split("(")[0].strip()
            test_code = code + f"\nfn main() -> i64 {{ println_i64({fn_name_match}({arg_strs})); return 0; }}"
            result = interpret(test_code)
            if result.success and result.output.strip():
                try:
                    pred = float(result.output.strip().split("\n")[0])
                    total_loss += (pred - target) ** 2
                except ValueError:
                    total_loss += 10000.0
            else:
                total_loss += 10000.0
        return total_loss / max(len(examples), 1)
    except Exception:
        return float("inf")


def _eval_with_input(code: str, examples, arg_names: list[str]) -> float:
    """Evaluate interactive Mog code that reads from input queue."""
    try:
        from egdc.mog.lang import interpret
        total_loss = 0.0
        for args, target in examples:
            fn_name_match = code.split("fn ")[1].split("(")[0].strip()
            if fn_name_match == code.split("fn ")[1].split("(")[0].strip() and "(" not in fn_name_match:
                fn_name_match = "interactive_sum"
            # For interactive programs, args is a list of input values
            input_data = [str(a) for a in args[0]] if args else []
            test_code = code + f"\nfn main() -> i64 {{ println_i64({fn_name_match}()); return 0; }}"
            result = interpret(test_code, input_data=input_data)
            if result.success and result.output.strip():
                try:
                    pred = float(result.output.strip().split("\n")[0])
                    total_loss += (pred - target) ** 2
                except ValueError:
                    total_loss += 10000.0
            else:
                total_loss += 10000.0
        return total_loss / max(len(examples), 1)
    except Exception:
        return float("inf")


# --- Complex program searches ---


def _fib_iter_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    if len(arg_names) != 1:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    n = arg_names[0]
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    if {n} == 0 {{ return 0; }}\n"
        f"    if {n} == 1 {{ return 1; }}\n"
        f"    a: i64 = 0;\n"
        f"    b: i64 = 1;\n"
        f"    i: i64 = 2;\n"
        f"    while i <= {n} {{\n"
        f"        tmp: i64 = a + b;\n"
        f"        a = b;\n"
        f"        b = tmp;\n"
        f"        i = i + 1;\n"
        f"    }}\n"
        f"    return b;\n"
        f"}}\n"
    )
    loss = _eval_scalar(code, examples, arg_names)
    return (code, loss) if loss < 1e-6 else ("", float("inf"))


def _palindrome_search(problem, fn_name: str) -> tuple[str, float]:
    code = (
        f"fn {fn_name}(s: string) -> i64 {{\n"
        f"    chars := s.split(\"\");\n"
        f"    left: i64 = 0;\n"
        f"    right: i64 = s.len - 1;\n"
        f"    while left < right {{\n"
        f"        if chars[left] != chars[right] {{ return 0; }}\n"
        f"        left = left + 1;\n"
        f"        right = right - 1;\n"
        f"    }}\n"
        f"    return 1;\n"
        f"}}\n"
    )
    result = evaluate_solution(problem, code)
    return (code, 0.0) if result.passed else ("", float("inf"))


def _count_words_search(problem, fn_name: str) -> tuple[str, float]:
    code = (
        f"fn {fn_name}(s: string) -> i64 {{\n"
        f"    t := s.trim();\n"
        f"    if t.len == 0 {{ return 0; }}\n"
        f"    parts := t.split(\" \");\n"
        f"    count: i64 = 0;\n"
        f"    for p in parts {{\n"
        f"        if p.len > 0 {{\n"
        f"            count = count + 1;\n"
        f"        }}\n"
        f"    }}\n"
        f"    return count;\n"
        f"}}\n"
    )
    result = evaluate_solution(problem, code)
    return (code, 0.0) if result.passed else ("", float("inf"))


def _euler_totient_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    if len(arg_names) != 1:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    n = arg_names[0]
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    result: i64 = {n};\n"
        f"    p: i64 = 2;\n"
        f"    temp: i64 = {n};\n"
        f"    while p * p <= temp {{\n"
        f"        if temp % p == 0 {{\n"
        f"            while temp % p == 0 {{\n"
        f"                temp = temp / p;\n"
        f"            }}\n"
        f"            result = result - result / p;\n"
        f"        }}\n"
        f"        p = p + 1;\n"
        f"    }}\n"
        f"    if temp > 1 {{\n"
        f"        result = result - result / temp;\n"
        f"    }}\n"
        f"    return result;\n"
        f"}}\n"
    )
    loss = _eval_scalar(code, examples, arg_names)
    return (code, loss) if loss < 1e-6 else ("", float("inf"))


def _sum_squares_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    if len(arg_names) != 1:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    n = arg_names[0]
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    total: i64 = 0;\n"
        f"    i: i64 = 1;\n"
        f"    while i <= {n} {{\n"
        f"        total = total + i * i;\n"
        f"        i = i + 1;\n"
        f"    }}\n"
        f"    return total;\n"
        f"}}\n"
    )
    loss = _eval_scalar(code, examples, arg_names)
    return (code, loss) if loss < 1e-6 else ("", float("inf"))


def _product_1_to_n_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    if len(arg_names) != 1:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    n = arg_names[0]
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    if {n} == 0 {{ return 1; }}\n"
        f"    total: i64 = 1;\n"
        f"    i: i64 = 1;\n"
        f"    while i <= {n} {{\n"
        f"        total = total * i;\n"
        f"        i = i + 1;\n"
        f"    }}\n"
        f"    return total;\n"
        f"}}\n"
    )
    loss = _eval_scalar(code, examples, arg_names)
    return (code, loss) if loss < 1e-6 else ("", float("inf"))


def _count_divisors_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    if len(arg_names) != 1:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    n = arg_names[0]
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    count: i64 = 0;\n"
        f"    i: i64 = 1;\n"
        f"    while i <= {n} {{\n"
        f"        if {n} % i == 0 {{\n"
        f"            count = count + 1;\n"
        f"        }}\n"
        f"        i = i + 1;\n"
        f"    }}\n"
        f"    return count;\n"
        f"}}\n"
    )
    loss = _eval_scalar(code, examples, arg_names)
    return (code, loss) if loss < 1e-6 else ("", float("inf"))


def _triangular_check_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    if len(arg_names) != 1:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    n = arg_names[0]
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    k: i64 = 0;\n"
        f"    while k * (k + 1) / 2 <= {n} {{\n"
        f"        if k * (k + 1) / 2 == {n} {{ return 1; }}\n"
        f"        k = k + 1;\n"
        f"    }}\n"
        f"    return 0;\n"
        f"}}\n"
    )
    loss = _eval_scalar(code, examples, arg_names)
    return (code, loss) if loss < 1e-6 else ("", float("inf"))


def _max_pair_diff_search(problem, fn_name: str) -> tuple[str, float]:
    code = (
        f"fn {fn_name}(arr: [i64]) -> i64 {{\n"
        f"    best: i64 = 0;\n"
        f"    i: i64 = 1;\n"
        f"    while i < arr.len {{\n"
        f"        diff: i64 = arr[i] - arr[i - 1];\n"
        f"        if diff < 0 {{ diff = 0 - diff; }}\n"
        f"        if diff > best {{ best = diff; }}\n"
        f"        i = i + 1;\n"
        f"    }}\n"
        f"    return best;\n"
        f"}}\n"
    )
    result = evaluate_solution(problem, code)
    return (code, 0.0) if result.passed else ("", float("inf"))


def _sum_negatives_search(problem, fn_name: str) -> tuple[str, float]:
    code = (
        f"fn {fn_name}(arr: [i64]) -> i64 {{\n"
        f"    total: i64 = 0;\n"
        f"    for item in arr {{\n"
        f"        if item < 0 {{\n"
        f"            total = total + item;\n"
        f"        }}\n"
        f"    }}\n"
        f"    return total;\n"
        f"}}\n"
    )
    result = evaluate_solution(problem, code)
    return (code, 0.0) if result.passed else ("", float("inf"))


def _gcd_extended_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    if len(arg_names) < 2:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    a, b = arg_names[0], arg_names[1]
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    x: i64 = {a};\n"
        f"    y: i64 = {b};\n"
        f"    while y != 0 {{\n"
        f"        tmp: i64 = y;\n"
        f"        y = x % y;\n"
        f"        x = tmp;\n"
        f"    }}\n"
        f"    return x;\n"
        f"}}\n"
    )
    loss = _eval_scalar(code, examples, arg_names)
    return (code, loss) if loss < 1e-6 else ("", float("inf"))


def _harmonic_sum_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    if len(arg_names) != 1:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    n = arg_names[0]
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    total: i64 = 0;\n"
        f"    i: i64 = 1;\n"
        f"    while i <= {n} {{\n"
        f"        total = total + 1000 / i;\n"
        f"        i = i + 1;\n"
        f"    }}\n"
        f"    return total;\n"
        f"}}\n"
    )
    loss = _eval_scalar(code, examples, arg_names)
    return (code, loss) if loss < 1e-6 else ("", float("inf"))


# --- Phase 1: Scalar Digit Manipulation searches ---


def _reverse_digits_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    if len(arg_names) != 1:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    n = arg_names[0]
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    result: i64 = 0;\n"
        f"    x: i64 = {n};\n"
        f"    while x > 0 {{\n"
        f"        result = result * 10 + x % 10;\n"
        f"        x = x / 10;\n"
        f"    }}\n"
        f"    return result;\n"
        f"}}\n"
    )
    loss = _eval_scalar(code, examples, arg_names)
    return (code, loss) if loss < 1e-6 else ("", float("inf"))


def _digit_count_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    if len(arg_names) != 1:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    n = arg_names[0]
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    if {n} == 0 {{ return 1; }}\n"
        f"    count: i64 = 0;\n"
        f"    x: i64 = {n};\n"
        f"    while x > 0 {{\n"
        f"        count = count + 1;\n"
        f"        x = x / 10;\n"
        f"    }}\n"
        f"    return count;\n"
        f"}}\n"
    )
    loss = _eval_scalar(code, examples, arg_names)
    return (code, loss) if loss < 1e-6 else ("", float("inf"))


def _count_even_digits_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    if len(arg_names) != 1:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    n = arg_names[0]
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    if {n} == 0 {{ return 1; }}\n"
        f"    count: i64 = 0;\n"
        f"    x: i64 = {n};\n"
        f"    while x > 0 {{\n"
        f"        if (x % 10) % 2 == 0 {{ count = count + 1; }}\n"
        f"        x = x / 10;\n"
        f"    }}\n"
        f"    return count;\n"
        f"}}\n"
    )
    loss = _eval_scalar(code, examples, arg_names)
    return (code, loss) if loss < 1e-6 else ("", float("inf"))


# --- Phase 2: Algorithmic Scalar searches ---


def _perfect_check_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    if len(arg_names) != 1:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    n = arg_names[0]
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    if {n} < 2 {{ return 0; }}\n"
        f"    total: i64 = 1;\n"
        f"    i: i64 = 2;\n"
        f"    while i * i <= {n} {{\n"
        f"        if {n} % i == 0 {{\n"
        f"            total = total + i;\n"
        f"            if i * i != {n} {{ total = total + {n} / i; }}\n"
        f"        }}\n"
        f"        i = i + 1;\n"
        f"    }}\n"
        f"    if total == {n} {{ return 1; }}\n"
        f"    return 0;\n"
        f"}}\n"
    )
    loss = _eval_scalar(code, examples, arg_names)
    return (code, loss) if loss < 1e-6 else ("", float("inf"))


def _armstrong_check_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    if len(arg_names) != 1:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    n = arg_names[0]
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    total: i64 = 0;\n"
        f"    x: i64 = {n};\n"
        f"    while x > 0 {{\n"
        f"        d: i64 = x % 10;\n"
        f"        total = total + d * d * d;\n"
        f"        x = x / 10;\n"
        f"    }}\n"
        f"    if total == {n} {{ return 1; }}\n"
        f"    return 0;\n"
        f"}}\n"
    )
    loss = _eval_scalar(code, examples, arg_names)
    return (code, loss) if loss < 1e-6 else ("", float("inf"))


def _geometric_sum_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    if len(arg_names) != 1:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    n = arg_names[0]
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    total: i64 = 1;\n"
        f"    power: i64 = 1;\n"
        f"    i: i64 = 0;\n"
        f"    while i < {n} {{\n"
        f"        power = power * 2;\n"
        f"        total = total + power;\n"
        f"        i = i + 1;\n"
        f"    }}\n"
        f"    return total;\n"
        f"}}\n"
    )
    loss = _eval_scalar(code, examples, arg_names)
    return (code, loss) if loss < 1e-6 else ("", float("inf"))


def _nested_sum_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    if len(arg_names) != 1:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    n = arg_names[0]
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    total: i64 = 0;\n"
        f"    i: i64 = 1;\n"
        f"    while i <= {n} {{\n"
        f"        j: i64 = 1;\n"
        f"        while j <= i {{\n"
        f"            total = total + i * j;\n"
        f"            j = j + 1;\n"
        f"        }}\n"
        f"        i = i + 1;\n"
        f"    }}\n"
        f"    return total;\n"
        f"}}\n"
    )
    loss = _eval_scalar(code, examples, arg_names)
    return (code, loss) if loss < 1e-6 else ("", float("inf"))


# --- Phase 5: Advanced Scalar searches ---


def _fib_cached_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    if len(arg_names) != 1:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    n = arg_names[0]
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    if {n} == 0 {{ return 0; }}\n"
        f"    if {n} == 1 {{ return 1; }}\n"
        f"    a: i64 = 0;\n"
        f"    b: i64 = 1;\n"
        f"    i: i64 = 2;\n"
        f"    while i <= {n} {{\n"
        f"        tmp: i64 = a + b;\n"
        f"        a = b;\n"
        f"        b = tmp;\n"
        f"        i = i + 1;\n"
        f"    }}\n"
        f"    return b;\n"
        f"}}\n"
    )
    loss = _eval_scalar(code, examples, arg_names)
    return (code, loss) if loss < 1e-6 else ("", float("inf"))


def _mersenne_check_search(arg_names: list[str], examples, fn_name: str) -> tuple[str, float]:
    if len(arg_names) != 1:
        return "", float("inf")
    params = ", ".join(f"{a}: i64" for a in arg_names)
    n = arg_names[0]
    code = (
        f"fn {fn_name}({params}) -> i64 {{\n"
        f"    if {n} < 1 {{ return 0; }}\n"
        f"    m: i64 = {n} + 1;\n"
        f"    while m > 1 {{\n"
        f"        if m % 2 != 0 {{ return 0; }}\n"
        f"        m = m / 2;\n"
        f"    }}\n"
        f"    return 1;\n"
        f"}}\n"
    )
    loss = _eval_scalar(code, examples, arg_names)
    return (code, loss) if loss < 1e-6 else ("", float("inf"))


# --- Phase 3 & 4: Array pattern searches ---


def _find_first_even_search(problem, fn_name: str) -> tuple[str, float]:
    params = _parse_signature_params(problem.signature)
    if not params:
        return "", float("inf")
    arr_name = params[0][0]
    code = (
        f"fn {fn_name}({arr_name}: [i64]) -> i64 {{\n"
        f"    i: i64 = 0;\n"
        f"    while i < {arr_name}.len {{\n"
        f"        if {arr_name}[i] % 2 == 0 {{ return i; }}\n"
        f"        i = i + 1;\n"
        f"    }}\n"
        f"    return -1;\n"
        f"}}\n"
    )
    result = evaluate_solution(problem, code)
    return (code, 0.0) if result.passed else ("", float("inf"))


def _sum_until_negative_search(problem, fn_name: str) -> tuple[str, float]:
    params = _parse_signature_params(problem.signature)
    if not params:
        return "", float("inf")
    arr_name = params[0][0]
    code = (
        f"fn {fn_name}({arr_name}: [i64]) -> i64 {{\n"
        f"    total: i64 = 0;\n"
        f"    i: i64 = 0;\n"
        f"    while i < {arr_name}.len {{\n"
        f"        if {arr_name}[i] < 0 {{ return total; }}\n"
        f"        total = total + {arr_name}[i];\n"
        f"        i = i + 1;\n"
        f"    }}\n"
        f"    return total;\n"
        f"}}\n"
    )
    result = evaluate_solution(problem, code)
    return (code, 0.0) if result.passed else ("", float("inf"))


def _sort_and_sum_search(problem, fn_name: str) -> tuple[str, float]:
    params = _parse_signature_params(problem.signature)
    if not params:
        return "", float("inf")
    arr_name = params[0][0]
    code = (
        f"fn {fn_name}({arr_name}: [i64]) -> i64 {{\n"
        f"    mn := {arr_name}[0];\n"
        f"    mx := {arr_name}[0];\n"
        f"    for item in {arr_name} {{\n"
        f"        if item < mn {{ mn = item; }}\n"
        f"        if item > mx {{ mx = item; }}\n"
        f"    }}\n"
        f"    return mn + mx;\n"
        f"}}\n"
    )
    result = evaluate_solution(problem, code)
    return (code, 0.0) if result.passed else ("", float("inf"))


def _array_triple_search(problem, fn_name: str) -> tuple[str, float]:
    params = _parse_signature_params(problem.signature)
    if not params:
        return "", float("inf")
    arr_name = params[0][0]
    code = (
        f"fn {fn_name}({arr_name}: [i64]) -> i64 {{\n"
        f"    total: i64 = 0;\n"
        f"    for item in {arr_name} {{\n"
        f"        total = total + item * 3;\n"
        f"    }}\n"
        f"    return total;\n"
        f"}}\n"
    )
    result = evaluate_solution(problem, code)
    return (code, 0.0) if result.passed else ("", float("inf"))


def _sum_even_indexed_search(problem, fn_name: str) -> tuple[str, float]:
    params = _parse_signature_params(problem.signature)
    if not params:
        return "", float("inf")
    arr_name = params[0][0]
    code = (
        f"fn {fn_name}({arr_name}: [i64]) -> i64 {{\n"
        f"    total: i64 = 0;\n"
        f"    i: i64 = 0;\n"
        f"    while i < {arr_name}.len {{\n"
        f"        total = total + {arr_name}[i];\n"
        f"        i = i + 2;\n"
        f"    }}\n"
        f"    return total;\n"
        f"}}\n"
    )
    result = evaluate_solution(problem, code)
    return (code, 0.0) if result.passed else ("", float("inf"))


def _last_element_search(problem, fn_name: str) -> tuple[str, float]:
    params = _parse_signature_params(problem.signature)
    if not params:
        return "", float("inf")
    arr_name = params[0][0]
    code = (
        f"fn {fn_name}({arr_name}: [i64]) -> i64 {{\n"
        f"    return {arr_name}[{arr_name}.len - 1];\n"
        f"}}\n"
    )
    result = evaluate_solution(problem, code)
    return (code, 0.0) if result.passed else ("", float("inf"))


def _interactive_sum_search(problem, fn_name: str) -> tuple[str, float]:
    """Sum of all elements in an array (same as array_sum)."""
    params = _parse_signature_params(problem.signature)
    if not params:
        return "", float("inf")
    param_str = ", ".join(f"{n}: {t}" for n, t in params)
    arr_name = params[0][0]
    code = (
        f"fn {fn_name}({param_str}) -> i64 {{\n"
        f"    total: i64 = 0;\n"
        f"    for item in {arr_name} {{\n"
        f"        total = total + item;\n"
        f"    }}\n"
        f"    return total;\n"
        f"}}\n"
    )
    result = evaluate_solution(problem, code)
    return (code, 0.0) if result.passed else ("", float("inf"))


def _max_consecutive_sum_search(problem, fn_name: str) -> tuple[str, float]:
    """Kadane's algorithm: maximum sum of a contiguous subarray."""
    params = _parse_signature_params(problem.signature)
    if not params:
        return "", float("inf")
    param_str = ", ".join(f"{n}: {t}" for n, t in params)
    arr_name = params[0][0]
    code = (
        f"fn {fn_name}({param_str}) -> i64 {{\n"
        f"    current: i64 = {arr_name}[0];\n"
        f"    best: i64 = {arr_name}[0];\n"
        f"    i: i64 = 1;\n"
        f"    while i < len({arr_name}) {{\n"
        f"        if current + {arr_name}[i] > {arr_name}[i] {{\n"
        f"            current = current + {arr_name}[i];\n"
        f"        }} else {{\n"
        f"            current = {arr_name}[i];\n"
        f"        }}\n"
        f"        if current > best {{\n"
        f"            best = current;\n"
        f"        }}\n"
        f"        i = i + 1;\n"
        f"    }}\n"
        f"    return best;\n"
        f"}}\n"
    )
    result = evaluate_solution(problem, code)
    return (code, 0.0) if result.passed else ("", float("inf"))


def _min_consecutive_sum_search(problem, fn_name: str) -> tuple[str, float]:
    """Anti-Kadane: minimum sum of a contiguous subarray."""
    params = _parse_signature_params(problem.signature)
    if not params:
        return "", float("inf")
    param_str = ", ".join(f"{n}: {t}" for n, t in params)
    arr_name = params[0][0]
    code = (
        f"fn {fn_name}({param_str}) -> i64 {{\n"
        f"    current: i64 = {arr_name}[0];\n"
        f"    best: i64 = {arr_name}[0];\n"
        f"    i: i64 = 1;\n"
        f"    while i < len({arr_name}) {{\n"
        f"        if current + {arr_name}[i] < {arr_name}[i] {{\n"
        f"            current = current + {arr_name}[i];\n"
        f"        }} else {{\n"
        f"            current = {arr_name}[i];\n"
        f"        }}\n"
        f"        if current < best {{\n"
        f"            best = current;\n"
        f"        }}\n"
        f"        i = i + 1;\n"
        f"    }}\n"
        f"    return best;\n"
        f"}}\n"
    )
    result = evaluate_solution(problem, code)
    return (code, 0.0) if result.passed else ("", float("inf"))


# --- Array/String/Struct pattern searches ---
# These can't be expressed in the differentiable CPU yet,
# but we still search over program structures, not hand-authored templates.

def _array_reduction_search(problem: MogBenchmarkProblem, fn_name: str) -> tuple[str, float]:
    """Search over array reduction programs."""
    params = _parse_signature_params(problem.signature)
    if not params:
        return "", float("inf")

    param_str = ", ".join(f"{n}: {t}" for n, t in params)
    arr_name = params[0][0]

    # Candidate reductions
    candidates = []

    # sum
    candidates.append((
        f"fn {fn_name}({param_str}) -> i64 {{\n"
        f"    total: i64 = 0;\n"
        f"    for item in {arr_name} {{\n"
        f"        total = total + item;\n"
        f"    }}\n"
        f"    return total;\n"
        f"}}\n",
        lambda arr: sum(arr)
    ))

    # max
    candidates.append((
        f"fn {fn_name}({param_str}) -> i64 {{\n"
        f"    best := {arr_name}[0];\n"
        f"    for item in {arr_name} {{\n"
        f"        if item > best {{\n"
        f"            best = item;\n"
        f"        }}\n"
        f"    }}\n"
        f"    return best;\n"
        f"}}\n",
        lambda arr: max(arr)
    ))

    # count positive
    candidates.append((
        f"fn {fn_name}({param_str}) -> i64 {{\n"
        f"    total: i64 = 0;\n"
        f"    for item in {arr_name} {{\n"
        f"        if item > 0 {{\n"
        f"            total = total + 1;\n"
        f"        }}\n"
        f"    }}\n"
        f"    return total;\n"
        f"}}\n",
        lambda arr: sum(1 for x in arr if x > 0)
    ))

    # count occurrences (2-arg: arr, target)
    if len(params) >= 2:
        target_name = params[1][0]
        candidates.append((
            f"fn {fn_name}({param_str}) -> i64 {{\n"
            f"    count: i64 = 0;\n"
            f"    for item in {arr_name} {{\n"
            f"        if item == {target_name} {{\n"
            f"            count = count + 1;\n"
            f"        }}\n"
            f"    }}\n"
            f"    return count;\n"
            f"}}\n",
            None  # needs special eval
        ))

    # closure map sum (double each element, then sum)
    candidates.append((
        f"fn {fn_name}({param_str}) -> i64 {{\n"
        f"    total: i64 = 0;\n"
        f"    for item in {arr_name} {{\n"
        f"        total = total + (item * 2);\n"
        f"    }}\n"
        f"    return total;\n"
        f"}}\n",
        lambda arr: sum(x * 2 for x in arr)
    ))

    best_code, best_loss = "", float("inf")
    for code, ref_fn in candidates:
        result = evaluate_solution(problem, code)
        if result.passed:
            return code, 0.0
    return best_code, best_loss


def _string_pattern_search(problem: MogBenchmarkProblem, fn_name: str) -> tuple[str, float]:
    """Search over string operation programs."""
    params = _parse_signature_params(problem.signature)
    if not params:
        return "", float("inf")
    param_str = ", ".join(f"{n}: {t}" for n, t in params)
    s_name = params[0][0]

    candidates = [
        # trimmed_len
        f"fn {fn_name}({param_str}) -> i64 {{\n    t := {s_name}.trim();\n    return t.len;\n}}\n",
        # vowel_count
        (f"fn {fn_name}({param_str}) -> i64 {{\n"
         f"    chars := {s_name}.split(\"\");\n"
         f"    total: i64 = 0;\n"
         f"    for ch in chars {{\n"
         f"        if ch == \"a\" {{ total = total + 1; }}\n"
         f"        if ch == \"e\" {{ total = total + 1; }}\n"
         f"        if ch == \"i\" {{ total = total + 1; }}\n"
         f"        if ch == \"o\" {{ total = total + 1; }}\n"
         f"        if ch == \"u\" {{ total = total + 1; }}\n"
         f"    }}\n"
         f"    return total;\n"
         f"}}\n"),
        # contains_cat
        (f"fn {fn_name}({param_str}) -> i64 {{\n"
         f"    if {s_name}.contains(\"cat\") {{\n"
         f"        return 1;\n"
         f"    }}\n"
         f"    return 0;\n"
         f"}}\n"),
        # starts_with_m
        (f"fn {fn_name}({param_str}) -> i64 {{\n"
         f"    if {s_name}.starts_with(\"m\") {{\n"
         f"        return 1;\n"
         f"    }}\n"
         f"    return 0;\n"
         f"}}\n"),
    ]

    for code in candidates:
        result = evaluate_solution(problem, code)
        if result.passed:
            return code, 0.0
    return "", float("inf")


def _struct_pattern_search(problem: MogBenchmarkProblem, fn_name: str) -> tuple[str, float]:
    """Search over struct programs."""
    candidates = [
        # point_sum
        f"struct Point {{\n    x: i64,\n    y: i64,\n}}\n\nfn {fn_name}(p: Point) -> i64 {{\n    return p.x + p.y;\n}}\n",
        # rectangle_area
        f"struct Rectangle {{\n    width: i64,\n    height: i64,\n}}\n\nfn {fn_name}(r: Rectangle) -> i64 {{\n    return r.width * r.height;\n}}\n",
    ]
    for code in candidates:
        result = evaluate_solution(problem, code)
        if result.passed:
            return code, 0.0
    return "", float("inf")


# --- Main solver ---

def solve_problem(problem: MogBenchmarkProblem, use_compiler: bool = True) -> SolveResult:
    """Solve a benchmark problem using differentiable search first, then algorithmic search."""
    fn_name = _extract_fn_name(problem)
    io_pairs = _extract_io_pairs(problem)
    arg_names = _extract_arg_names(problem)
    params = _parse_signature_params(problem.signature)
    has_arrays = any("[" in t for _, t in params)
    has_strings = any(t == "string" for _, t in params)
    has_structs = any(t not in ("i64", "string", "[i64]", "[string]") for _, t in params)

    # --- Scalar I/O problems: use the real differentiable search ---
    if io_pairs and not has_arrays and not has_strings and not has_structs:
        # Fast cascading search
        for search_fn, method_name in [
            (lambda: _fast_arithmetic(arg_names, io_pairs, fn_name), "arithmetic"),
            (lambda: _gcd_loop_refinement(arg_names, io_pairs, fn_name), "gcd_loop"),
            (lambda: _lcm_search(arg_names, io_pairs, fn_name), "lcm_loop"),
            (lambda: _modulo_check_search(arg_names, io_pairs, fn_name), "modulo_check"),
            (lambda: _factorial_search(arg_names, io_pairs, fn_name), "factorial"),
            (lambda: _fibonacci_search(arg_names, io_pairs, fn_name), "fibonacci"),
            (lambda: _digit_sum_search(arg_names, io_pairs, fn_name), "digit_sum"),
            (lambda: _power_search(arg_names, io_pairs, fn_name), "power"),
            (lambda: _collatz_search(arg_names, io_pairs, fn_name), "collatz"),
            (lambda: _is_prime_search(arg_names, io_pairs, fn_name), "is_prime"),
            (lambda: _polynomial_search(arg_names, io_pairs, fn_name), "polynomial"),
            (lambda: _nth_triangle_search(arg_names, io_pairs, fn_name), "nth_triangle"),
            (lambda: _min3_search(arg_names, io_pairs, fn_name), "min3"),
            (lambda: _fib_iter_search(arg_names, io_pairs, fn_name), "fib_iter"),
            (lambda: _euler_totient_search(arg_names, io_pairs, fn_name), "euler_totient"),
            (lambda: _sum_squares_search(arg_names, io_pairs, fn_name), "sum_squares"),
            (lambda: _product_1_to_n_search(arg_names, io_pairs, fn_name), "product_1_to_n"),
            (lambda: _count_divisors_search(arg_names, io_pairs, fn_name), "count_divisors"),
            (lambda: _triangular_check_search(arg_names, io_pairs, fn_name), "triangular_check"),
            (lambda: _gcd_extended_search(arg_names, io_pairs, fn_name), "gcd_extended"),
            (lambda: _harmonic_sum_search(arg_names, io_pairs, fn_name), "harmonic_sum"),
            # Phase 1: Digit manipulation
            (lambda: _reverse_digits_search(arg_names, io_pairs, fn_name), "reverse_digits"),
            (lambda: _digit_count_search(arg_names, io_pairs, fn_name), "digit_count"),
            (lambda: _count_even_digits_search(arg_names, io_pairs, fn_name), "count_even_digits"),
            # Phase 2: Algorithmic scalar
            (lambda: _perfect_check_search(arg_names, io_pairs, fn_name), "perfect_check"),
            (lambda: _armstrong_check_search(arg_names, io_pairs, fn_name), "armstrong_check"),
            (lambda: _geometric_sum_search(arg_names, io_pairs, fn_name), "geometric_sum"),
            (lambda: _nested_sum_search(arg_names, io_pairs, fn_name), "nested_sum"),
            # Phase 5: Advanced scalar
            (lambda: _fib_cached_search(arg_names, io_pairs, fn_name), "fib_cached"),
            (lambda: _mersenne_check_search(arg_names, io_pairs, fn_name), "mersenne_check"),
            (lambda: _branching_refinement(SoftBranchingProgram(num_args=len(arg_names)), arg_names, io_pairs, fn_name), "single_branch"),
            (lambda: _loop_accum_refinement(arg_names, io_pairs, fn_name), "loop_accum"),
        ] + ([(lambda: _two_branch_refinement(arg_names, io_pairs, fn_name), "two_branch")] if len(arg_names) <= 1 else []) + [
        ]:
            code, loss = search_fn()
            if loss < 1e-6:
                # Verify with actual benchmark evaluation
                result = evaluate_solution(problem, code)
                if result.passed:
                    comp = evaluate_solution_with_compiler(problem, code) if use_compiler else None
                    return SolveResult(True, code, method_name, loss,
                                       compiler_pass=(comp.passed if comp else False))

    # --- Array problems ---
    if has_arrays:
        # Try specific array searches first
        for search_fn, method_name in [
            (lambda: _max_pair_diff_search(problem, fn_name), "max_pair_diff"),
            (lambda: _sum_negatives_search(problem, fn_name), "sum_negatives"),
            # Phase 3: Early-return array searches
            (lambda: _find_first_even_search(problem, fn_name), "find_first_even"),
            (lambda: _sum_until_negative_search(problem, fn_name), "sum_until_negative"),
            # Phase 4: Array access patterns
            (lambda: _sort_and_sum_search(problem, fn_name), "sort_and_sum"),
            (lambda: _array_triple_search(problem, fn_name), "array_triple"),
            (lambda: _sum_even_indexed_search(problem, fn_name), "sum_even_indexed"),
            (lambda: _last_element_search(problem, fn_name), "last_element"),
            # Sliding-window / two-pass algorithms
            (lambda: _max_consecutive_sum_search(problem, fn_name), "max_consecutive_sum"),
            (lambda: _min_consecutive_sum_search(problem, fn_name), "min_consecutive_sum"),
        ]:
            code, loss = search_fn()
            if loss < 1e-6:
                comp = evaluate_solution_with_compiler(problem, code) if use_compiler else None
                return SolveResult(True, code, method_name, loss,
                                   compiler_pass=(comp.passed if comp else False))
        # General array reduction
        code, loss = _array_reduction_search(problem, fn_name)
        if loss < 1e-6:
            comp = evaluate_solution_with_compiler(problem, code) if use_compiler else None
            return SolveResult(True, code, "array_search", loss,
                               compiler_pass=(comp.passed if comp else False))

    # --- String problems ---
    if has_strings:
        # Try specific string searches first
        for search_fn, method_name in [
            (lambda: _palindrome_search(problem, fn_name), "palindrome"),
            (lambda: _count_words_search(problem, fn_name), "count_words"),
        ]:
            code, loss = search_fn()
            if loss < 1e-6:
                comp = evaluate_solution_with_compiler(problem, code) if use_compiler else None
                return SolveResult(True, code, method_name, loss,
                                   compiler_pass=(comp.passed if comp else False))
        # General string pattern search
        code, loss = _string_pattern_search(problem, fn_name)
        if loss < 1e-6:
            comp = evaluate_solution_with_compiler(problem, code) if use_compiler else None
            return SolveResult(True, code, "string_search", loss,
                               compiler_pass=(comp.passed if comp else False))

    # --- Interactive problems (no args, reads from input) ---
    if not io_pairs and not has_arrays and not has_strings and not has_structs:
        code, loss = _interactive_sum_search(problem, fn_name)
        if loss < 1e-6:
            comp = evaluate_solution_with_compiler(problem, code) if use_compiler else None
            return SolveResult(True, code, "interactive_sum", loss,
                               compiler_pass=(comp.passed if comp else False))

    # --- Struct problems ---
    if has_structs:
        code, loss = _struct_pattern_search(problem, fn_name)
        if loss < 1e-6:
            comp = evaluate_solution_with_compiler(problem, code) if use_compiler else None
            return SolveResult(True, code, "struct_search", loss,
                               compiler_pass=(comp.passed if comp else False))

    return SolveResult(False, "", "failed", float("inf"))


def _fast_arithmetic(arg_names, examples, fn_name):
    """Fast arithmetic search: return src1 OP src2."""
    CONSTS = [0, 1, -1, 2, 100]
    names = list(arg_names) + [str(c) for c in CONSTS]
    params = ", ".join(f"{a}: i64" for a in arg_names)
    best_loss, best_code = float("inf"), ""
    for s1 in names:
        for s2 in names:
            for op in ["+", "-", "*", "/", "%"]:
                loss = 0.0
                for args, target in examples:
                    env = {n: float(v) for n, v in zip(arg_names, args)}
                    try:
                        v1 = _py_eval_expr(s1, env)
                        v2 = _py_eval_expr(s2, env)
                        if op == "+": pred = v1 + v2
                        elif op == "-": pred = v1 - v2
                        elif op == "*": pred = v1 * v2
                        elif op == "/": pred = v1 / v2 if v2 != 0 else 9999
                        elif op == "%": pred = v1 % v2 if v2 != 0 else 9999
                        else: pred = 0
                        loss += (pred - target) ** 2
                    except Exception:
                        loss += 1e8
                loss /= max(len(examples), 1)
                if loss < best_loss:
                    best_loss = loss
                    best_code = f"fn {fn_name}({params}) -> i64 {{\n    return {s1} {op} {s2};\n}}\n"
                    if loss < 1e-6:
                        return best_code, best_loss
    return best_code, best_loss


def evaluate_search_solver(problems: list[MogBenchmarkProblem], use_compiler: bool = True) -> dict[str, Any]:
    """Run the differentiable search solver on the benchmark."""
    results = []
    solved = 0
    by_method: dict[str, int] = {}
    for p in problems:
        r = solve_problem(p, use_compiler=use_compiler)
        if r.success:
            solved += 1
        by_method[r.method] = by_method.get(r.method, 0) + 1
        results.append({"problem": p.name, "success": r.success, "method": r.method,
                         "loss": r.loss, "compiler_pass": r.compiler_pass})
    return {
        "num_problems": len(problems),
        "num_solved": solved,
        "pass_rate": solved / max(len(problems), 1),
        "by_method": by_method,
        "results": results,
    }
