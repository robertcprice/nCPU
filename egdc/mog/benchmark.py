"""Execution benchmark for Mog code generation.

This benchmark stays within the compiler-safe subset documented in
`docs/mog/mog_compiler_compat.md`. It uses the local Python Mog interpreter for
fast evaluation and can optionally be cross-checked against the real Mog
compiler via `egdc.mog_execute`.

The benchmark is deliberately HumanEval-style:
- natural language description
- target function signature
- hidden/public test cases
- wrapper code that calls the generated function and prints outputs
- exact stdout comparison
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Any, Callable, Iterable

from egdc.mog.lang import interpret
from egdc.mog.execute import execute_mog


@dataclass
class MogBenchmarkProblem:
    name: str
    category: str
    description: str
    signature: str
    test_cases: list[tuple[tuple[Any, ...], str]]
    wrapper_template: str
    reference_solution: str | None = None


@dataclass
class MogBenchmarkResult:
    problem_name: str
    passed: bool
    expected_output: str
    actual_output: str
    error: str | None = None


def _mog_literal(value: Any) -> str:
    if isinstance(value, RawCode):
        return value.code
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == int(value):
            return f"{value:.1f}"
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_mog_literal(x) for x in value) + "]"
    if isinstance(value, tuple):
        return "[" + ", ".join(_mog_literal(x) for x in value) + "]"
    raise TypeError(f"cannot encode Mog literal: {value!r}")


@dataclass(frozen=True)
class RawCode:
    code: str


def _parse_signature_params(signature: str) -> list[tuple[str, str]]:
    m = re.search(r"fn\s+\w+\s*\((.*)\)\s*->", signature)
    if not m:
        return []
    body = m.group(1).strip()
    if not body:
        return []
    parts = [p.strip() for p in body.split(",") if p.strip()]
    out: list[tuple[str, str]] = []
    for part in parts:
        name, type_ann = part.split(":", 1)
        out.append((name.strip(), type_ann.strip()))
    return out


def _build_wrapper(function_name: str, signature: str, test_cases: list[tuple[tuple[Any, ...], str]]) -> str:
    params = _parse_signature_params(signature)
    lines = ["fn main() -> i64 {"]
    for case_idx, (args, _expected) in enumerate(test_cases):
        call_args: list[str] = []
        for arg_idx, arg in enumerate(args):
            param_type = params[arg_idx][1] if arg_idx < len(params) else None
            # The real Mog compiler is picky about array literals / composite literals
            # inline in call positions, so bind them to typed locals first.
            if isinstance(arg, list) and param_type is not None:
                var_name = f"arg_{case_idx}_{arg_idx}"
                lines.append(f"    {var_name}: {param_type} = {_mog_literal(arg)};")
                call_args.append(var_name)
            elif isinstance(arg, RawCode) and param_type is not None:
                var_name = f"arg_{case_idx}_{arg_idx}"
                lines.append(f"    {var_name}: {param_type} = {arg.code};")
                call_args.append(var_name)
            else:
                call_args.append(_mog_literal(arg))
        arg_src = ", ".join(call_args)
        lines.append(f"    println_i64({function_name}({arg_src}));")
    lines.append("    return 0;")
    lines.append("}")
    return "\n".join(lines)


def _expected_stdout(test_cases: list[tuple[tuple[Any, ...], str]]) -> str:
    return "\n".join(expected for _, expected in test_cases)


def evaluate_solution(problem: MogBenchmarkProblem, generated_code: str) -> MogBenchmarkResult:
    program = generated_code.rstrip() + "\n\n" + problem.wrapper_template + "\n"
    result = interpret(program)
    expected = _expected_stdout(problem.test_cases)
    actual = result.output.strip()
    return MogBenchmarkResult(
        problem_name=problem.name,
        passed=result.success and actual == expected.strip(),
        expected_output=expected.strip(),
        actual_output=actual,
        error=result.error,
    )


def evaluate_solution_with_compiler(problem: MogBenchmarkProblem, generated_code: str) -> MogBenchmarkResult:
    program = generated_code.rstrip() + "\n\n" + problem.wrapper_template + "\n"
    result = execute_mog(program)
    expected = _expected_stdout(problem.test_cases)
    actual = result.stdout.strip() if result.success else ""
    passed = result.success and actual == expected.strip()
    error = None
    if not passed:
        error = result.error or result.compile_stderr or result.stderr
        if error is None and result.success:
            error = f"output mismatch: expected {expected.strip()!r}, got {actual!r}"
        elif error is None:
            error = "compiler execution failed"
    return MogBenchmarkResult(
        problem_name=problem.name,
        passed=passed,
        expected_output=expected.strip(),
        actual_output=actual,
        error=error,
    )


def evaluate_solutions_batch(problems: list[MogBenchmarkProblem], solutions: dict[str, str]) -> dict[str, Any]:
    results: list[MogBenchmarkResult] = []
    for problem in problems:
        code = solutions.get(problem.name)
        if not code:
            results.append(MogBenchmarkResult(problem.name, False, _expected_stdout(problem.test_cases), "", "missing solution"))
            continue
        results.append(evaluate_solution(problem, code))

    passed = sum(1 for r in results if r.passed)
    return {
        "num_problems": len(problems),
        "num_passed": passed,
        "pass_rate": passed / max(len(problems), 1),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Problem factories
# ---------------------------------------------------------------------------


def _problem(name: str, category: str, description: str, signature: str,
             test_cases: list[tuple[tuple[Any, ...], str]], reference_solution: str) -> MogBenchmarkProblem:
    fn_name = signature.split("fn ", 1)[1].split("(", 1)[0].strip()
    return MogBenchmarkProblem(
        name=name,
        category=category,
        description=description,
        signature=signature,
        test_cases=test_cases,
        wrapper_template=_build_wrapper(fn_name, signature, test_cases),
        reference_solution=reference_solution.strip() + "\n",
    )


def _make_add_two(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    tests = []
    for _ in range(4):
        a, b = rng.randint(-50, 50), rng.randint(-50, 50)
        tests.append(((a, b), str(a + b)))
    return _problem(
        f"add_two_v{variant}", "arithmetic",
        "Return the sum of two i64 integers.",
        "fn add_two(a: i64, b: i64) -> i64",
        tests,
        """
fn add_two(a: i64, b: i64) -> i64 {
    return a + b;
}
""",
    )


def _make_abs_diff(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    tests = []
    for _ in range(4):
        a, b = rng.randint(-30, 30), rng.randint(-30, 30)
        tests.append(((a, b), str(abs(a - b))))
    return _problem(
        f"abs_diff_v{variant}", "arithmetic",
        "Return the absolute difference between two integers.",
        "fn abs_diff(a: i64, b: i64) -> i64",
        tests,
        """
fn abs_diff(a: i64, b: i64) -> i64 {
    if a > b {
        return a - b;
    } else {
        return b - a;
    }
}
""",
    )


def _make_max2(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    tests = [((rng.randint(-40, 40), rng.randint(-40, 40)), "") for _ in range(4)]
    tests = [((a, b), str(max(a, b))) for (a, b), _ in tests]
    return _problem(
        f"max2_v{variant}", "control_flow",
        "Return the larger of two integers.",
        "fn max2(a: i64, b: i64) -> i64",
        tests,
        """
fn max2(a: i64, b: i64) -> i64 {
    if a > b {
        return a;
    } else {
        return b;
    }
}
""",
    )


def _make_clamp(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    vals = [rng.randint(-50, 150) for _ in range(4)]
    tests = [((x,), str(max(0, min(100, x)))) for x in vals]
    return _problem(
        f"clamp_0_100_v{variant}", "control_flow",
        "Clamp x into the closed range [0, 100].", 
        "fn clamp_0_100(x: i64) -> i64",
        tests,
        """
fn clamp_0_100(x: i64) -> i64 {
    if x < 0 {
        return 0;
    }
    if x > 100 {
        return 100;
    }
    return x;
}
""",
    )


def _make_sign(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    vals = [rng.randint(-20, 20) for _ in range(4)]
    tests = [((x,), str(-1 if x < 0 else (1 if x > 0 else 0))) for x in vals]
    return _problem(
        f"sign_v{variant}", "control_flow",
        "Return -1 for negative, 0 for zero, and 1 for positive.",
        "fn sign(x: i64) -> i64",
        tests,
        """
fn sign(x: i64) -> i64 {
    if x < 0 {
        return -1;
    }
    if x > 0 {
        return 1;
    }
    return 0;
}
""",
    )


def _make_sum_to_n(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    vals = [rng.randint(0, 20) for _ in range(4)]
    tests = [((n,), str(sum(range(1, n + 1)))) for n in vals]
    return _problem(
        f"sum_to_n_v{variant}", "arithmetic",
        "Return 1 + 2 + ... + n. For n <= 0 return 0.",
        "fn sum_to_n(n: i64) -> i64",
        tests,
        """
fn sum_to_n(n: i64) -> i64 {
    if n <= 0 {
        return 0;
    }
    total: i64 = 0;
    for i := 1 to (n + 1) {
        total = total + i;
    }
    return total;
}
""",
    )


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _make_gcd(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    pairs = [(rng.randint(1, 60), rng.randint(1, 60)) for _ in range(4)]
    tests = [((a, b), str(_gcd(a, b))) for a, b in pairs]
    return _problem(
        f"gcd_v{variant}", "arithmetic",
        "Return the greatest common divisor of two positive integers.",
        "fn gcd(a: i64, b: i64) -> i64",
        tests,
        """
fn gcd(a: i64, b: i64) -> i64 {
    x: i64 = a;
    y: i64 = b;
    while y != 0 {
        tmp := y;
        y = x % y;
        x = tmp;
    }
    return x;
}
""",
    )


def _make_lcm(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    pairs = [(rng.randint(1, 20), rng.randint(1, 20)) for _ in range(4)]
    tests = [((a, b), str((a * b) // _gcd(a, b))) for a, b in pairs]
    return _problem(
        f"lcm_v{variant}", "arithmetic",
        "Return the least common multiple of two positive integers.",
        "fn lcm(a: i64, b: i64) -> i64",
        tests,
        """
fn gcd_inner(a: i64, b: i64) -> i64 {
    x: i64 = a;
    y: i64 = b;
    while y != 0 {
        tmp := y;
        y = x % y;
        x = tmp;
    }
    return x;
}

fn lcm(a: i64, b: i64) -> i64 {
    return (a * b) / gcd_inner(a, b);
}
""",
    )


def _make_array_sum(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    tests = []
    for _ in range(4):
        arr = [rng.randint(0, 9) for _ in range(rng.randint(1, 6))]
        tests.append(((arr,), str(sum(arr))))
    return _problem(
        f"array_sum_v{variant}", "arrays",
        "Return the sum of all elements in an array of i64 values.",
        "fn array_sum(arr: [i64]) -> i64",
        tests,
        """
fn array_sum(arr: [i64]) -> i64 {
    total: i64 = 0;
    for item in arr {
        total = total + item;
    }
    return total;
}
""",
    )


def _make_array_max(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    tests = []
    for _ in range(4):
        arr = [rng.randint(-9, 20) for _ in range(rng.randint(1, 6))]
        tests.append(((arr,), str(max(arr))))
    return _problem(
        f"array_max_v{variant}", "arrays",
        "Return the largest element in a non-empty array.",
        "fn array_max(arr: [i64]) -> i64",
        tests,
        """
fn array_max(arr: [i64]) -> i64 {
    best := arr[0];
    for item in arr {
        if item > best {
            best = item;
        }
    }
    return best;
}
""",
    )


def _make_count_occurrences(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    tests = []
    for _ in range(4):
        arr = [rng.randint(0, 4) for _ in range(rng.randint(3, 7))]
        target = rng.randint(0, 4)
        tests.append(((arr, target), str(sum(1 for x in arr if x == target))))
    return _problem(
        f"count_occurrences_v{variant}", "arrays",
        "Count how many times target appears in arr.",
        "fn count_occurrences(arr: [i64], target: i64) -> i64",
        tests,
        """
fn count_occurrences(arr: [i64], target: i64) -> i64 {
    count: i64 = 0;
    for item in arr {
        if item == target {
            count = count + 1;
        }
    }
    return count;
}
""",
    )


def _make_trimmed_len(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    words = [" mog ", "  diffusion", "compiler  ", "  hello world  "]
    rng.shuffle(words)
    tests = [((s,), str(len(s.strip()))) for s in words[:4]]
    return _problem(
        f"trimmed_len_v{variant}", "strings",
        "Trim leading and trailing spaces and return the remaining length.",
        "fn trimmed_len(s: string) -> i64",
        tests,
        """
fn trimmed_len(s: string) -> i64 {
    t := s.trim();
    return t.len;
}
""",
    )


def _make_vowel_count(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    samples = ["mog", "aeiou", "banana", "rhythm", "interpreter", "compiler"]
    rng.shuffle(samples)
    def count_vowels(s: str) -> int:
        return sum(1 for ch in s.lower() if ch in "aeiou")
    tests = [((s,), str(count_vowels(s))) for s in samples[:4]]
    return _problem(
        f"vowel_count_v{variant}", "strings",
        "Count vowels (a, e, i, o, u) in a lowercase ASCII string.",
        "fn vowel_count(s: string) -> i64",
        tests,
        """
fn vowel_count(s: string) -> i64 {
    chars := s.split("");
    total: i64 = 0;
    for ch in chars {
        if ch == "a" { total = total + 1; }
        if ch == "e" { total = total + 1; }
        if ch == "i" { total = total + 1; }
        if ch == "o" { total = total + 1; }
        if ch == "u" { total = total + 1; }
    }
    return total;
}
""",
    )


def _make_contains_cat(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    samples = ["cat", "scatter", "dog", "catalog", "hello", "copycat"]
    rng.shuffle(samples)
    tests = [((s,), str(1 if "cat" in s else 0)) for s in samples[:4]]
    return _problem(
        f"contains_cat_v{variant}", "strings",
        "Return 1 if the string contains the substring 'cat', else 0.",
        "fn contains_cat(s: string) -> i64",
        tests,
        """
fn contains_cat(s: string) -> i64 {
    if s.contains("cat") {
        return 1;
    }
    return 0;
}
""",
    )


def _make_point_sum(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    tests = []
    for _ in range(4):
        x, y = rng.randint(-10, 10), rng.randint(-10, 10)
        tests.append(((RawCode(f"Point {{ x: {x}, y: {y} }}"),), str(x + y)))
    return _problem(
        f"point_sum_v{variant}", "structs",
        "Define struct Point { x: i64, y: i64 } and return x + y.",
        "fn point_sum(p: Point) -> i64",
        tests,
        """
struct Point {
    x: i64,
    y: i64,
}

fn point_sum(p: Point) -> i64 {
    return p.x + p.y;
}
""",
    )


def _make_safe_div_or_neg1(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    tests = []
    for _ in range(4):
        a = rng.randint(1, 50)
        b = rng.choice([0, rng.randint(1, 9)])
        tests.append(((a, b), str(-1 if b == 0 else a // b)))
    return _problem(
        f"safe_div_or_neg1_v{variant}", "result_optional",
        "Divide a by b. If b is zero, return -1. Use Result and match.",
        "fn safe_div_or_neg1(a: i64, b: i64) -> i64",
        tests,
        """
fn helper_div(a: i64, b: i64) -> Result<i64> {
    if b == 0 {
        return err("division by zero");
    }
    return ok(a / b);
}

fn safe_div_or_neg1(a: i64, b: i64) -> i64 {
    r := helper_div(a, b);
    out: i64 = match r {
        ok(v) => v,
        err(e) => -1,
    };
    return out;
}
""",
    )


def _make_positive_or_default(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    vals = [rng.randint(-20, 20) for _ in range(4)]
    tests = [((x,), str(x if x > 0 else 0)) for x in vals]
    return _problem(
        f"positive_or_default_v{variant}", "result_optional",
        "Return x if x is positive, otherwise return 0. Use ?i64 and match.",
        "fn positive_or_default(x: i64) -> i64",
        tests,
        """
fn maybe_positive(x: i64) -> ?i64 {
    if x > 0 {
        return some(x);
    }
    return none;
}

fn positive_or_default(x: i64) -> i64 {
    r := maybe_positive(x);
    out: i64 = match r {
        some(v) => v,
        none => 0,
    };
    return out;
}
""",
    )


def _make_factorial(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    vals = [rng.randint(0, 8) for _ in range(4)]
    def fact(n: int) -> int:
        out = 1
        for i in range(2, n + 1):
            out *= i
        return out
    tests = [((n,), str(fact(n))) for n in vals]
    return _problem(
        f"factorial_v{variant}", "recursion",
        "Return n! recursively.",
        "fn factorial(n: i64) -> i64",
        tests,
        """
fn factorial(n: i64) -> i64 {
    if n <= 1 {
        return 1;
    }
    return n * factorial(n - 1);
}
""",
    )


def _make_fibonacci(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    vals = [rng.randint(0, 12) for _ in range(4)]
    def fib(n: int) -> int:
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a
    tests = [((n,), str(fib(n))) for n in vals]
    return _problem(
        f"fibonacci_v{variant}", "recursion",
        "Return the nth Fibonacci number recursively or iteratively.",
        "fn fibonacci(n: i64) -> i64",
        tests,
        """
fn fibonacci(n: i64) -> i64 {
    if n <= 0 { return 0; }
    if n == 1 { return 1; }
    return fibonacci(n - 1) + fibonacci(n - 2);
}
""",
    )


def _make_closure_map_sum(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    tests = []
    for _ in range(4):
        arr = [rng.randint(1, 5) for _ in range(3)]
        tests.append(((arr,), str(sum(x * 2 for x in arr))))
    return _problem(
        f"closure_map_sum_v{variant}", "higher_order",
        "Double every array element with .map() and return the sum of the doubled values.",
        "fn closure_map_sum(arr: [i64]) -> i64",
        tests,
        """
fn closure_map_sum(arr: [i64]) -> i64 {
    doubled := arr.map(fn(x: i64) -> i64 { x * 2 });
    total: i64 = 0;
    for item in doubled {
        total = total + item;
    }
    return total;
}
""",
    )


def _make_count_positive(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    tests = []
    for _ in range(4):
        arr = [rng.randint(-4, 4) for _ in range(5)]
        tests.append(((arr,), str(sum(1 for x in arr if x > 0))))
    return _problem(
        f"count_positive_v{variant}", "arrays",
        "Count how many elements in the array are greater than zero.",
        "fn count_positive(arr: [i64]) -> i64",
        tests,
        """
fn count_positive(arr: [i64]) -> i64 {
    total: i64 = 0;
    for item in arr {
        if item > 0 {
            total = total + 1;
        }
    }
    return total;
}
""",
    )


def _make_is_even(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    vals = [rng.randint(-20, 20) for _ in range(4)]
    tests = [((x,), str(1 if x % 2 == 0 else 0)) for x in vals]
    return _problem(
        f"is_even_v{variant}", "control_flow",
        "Return 1 if x is even, otherwise 0.",
        "fn is_even(x: i64) -> i64",
        tests,
        """
fn is_even(x: i64) -> i64 {
    if (x % 2) == 0 {
        return 1;
    }
    return 0;
}
""",
    )


def _make_digit_sum(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    vals = [rng.randint(0, 9999) for _ in range(4)]
    def digit_sum(n: int) -> int:
        return sum(int(ch) for ch in str(n))
    tests = [((n,), str(digit_sum(n))) for n in vals]
    return _problem(
        f"digit_sum_v{variant}", "arithmetic",
        "Return the sum of the decimal digits of n.",
        "fn digit_sum(n: i64) -> i64",
        tests,
        """
fn digit_sum(n: i64) -> i64 {
    x: i64 = n;
    if x < 0 {
        x = 0 - x;
    }
    total: i64 = 0;
    while x > 0 {
        total = total + (x % 10);
        x = x / 10;
    }
    return total;
}
""",
    )


def _make_starts_with_m(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    samples = ["mog", "metal", "apple", "mars", "code", "map"]
    rng.shuffle(samples)
    tests = [((s,), str(1 if s.startswith("m") else 0)) for s in samples[:4]]
    return _problem(
        f"starts_with_m_v{variant}", "strings",
        "Return 1 if s starts with the lowercase letter m, else 0.",
        "fn starts_with_m(s: string) -> i64",
        tests,
        """
fn starts_with_m(s: string) -> i64 {
    if s.starts_with("m") {
        return 1;
    }
    return 0;
}
""",
    )


def _make_rectangle_area(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    tests = []
    for _ in range(4):
        w, h = rng.randint(1, 10), rng.randint(1, 10)
        tests.append(((RawCode(f"Rectangle {{ width: {w}, height: {h} }}"),), str(w * h)))
    return _problem(
        f"rectangle_area_v{variant}", "structs",
        "Define struct Rectangle { width: i64, height: i64 } and return its area.",
        "fn rectangle_area(r: Rectangle) -> i64",
        tests,
        """
struct Rectangle {
    width: i64,
    height: i64,
}

fn rectangle_area(r: Rectangle) -> i64 {
    return r.width * r.height;
}
""",
    )


def _make_power(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """x^n via repeated multiplication."""
    tests = []
    for _ in range(4):
        base = rng.randint(2, 5)
        exp = rng.randint(0, 4)
        tests.append(((base, exp), str(base ** exp)))
    return _problem(
        f"power_v{variant}", "arithmetic",
        "Compute base raised to the power exp (non-negative).",
        "fn power(base: i64, exp: i64) -> i64",
        tests,
        """
fn power(base: i64, exp: i64) -> i64 {
    if exp == 0 { return 1; }
    result: i64 = 1;
    i: i64 = 0;
    while i < exp {
        result = result * base;
        i = i + 1;
    }
    return result;
}
""",
    )


def _make_polynomial(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Evaluate 2*x*x + 3*x + 1."""
    tests = []
    for _ in range(4):
        x = rng.randint(0, 5)
        result = 2 * x * x + 3 * x + 1
        tests.append(((x,), str(result)))
    return _problem(
        f"polynomial_v{variant}", "arithmetic",
        "Evaluate the polynomial 2*x*x + 3*x + 1.",
        "fn polynomial(x: i64) -> i64",
        tests,
        """
fn polynomial(x: i64) -> i64 {
    return 2 * x * x + 3 * x + 1;
}
""",
    )


def _make_collatz_steps(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Count steps in the Collatz sequence to reach 1."""
    tests = []
    for n in [1, 2, 3, 6, 7, 10, 27]:
        steps = 0
        c = n
        while c > 1:
            if c % 2 == 0:
                c = c // 2
            else:
                c = 3 * c + 1
            steps += 1
        tests.append(((n,), str(steps)))
    return _problem(
        f"collatz_steps_v{variant}", "loops",
        "Count how many steps it takes for the Collatz sequence starting at n to reach 1.",
        "fn collatz_steps(n: i64) -> i64",
        tests,
        """
fn collatz_steps(n: i64) -> i64 {
    x: i64 = n;
    steps: i64 = 0;
    while x > 1 {
        if x % 2 == 0 {
            x = x / 2;
        } else {
            x = 3 * x + 1;
        }
        steps = steps + 1;
    }
    return steps;
}
""",
    )


def _make_min3(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Minimum of 3 values."""
    tests = []
    for _ in range(5):
        a, b, c = rng.randint(-10, 10), rng.randint(-10, 10), rng.randint(-10, 10)
        tests.append(((a, b, c), str(min(a, b, c))))
    return _problem(
        f"min3_v{variant}", "control_flow",
        "Return the minimum of three integers.",
        "fn min3(a: i64, b: i64, c: i64) -> i64",
        tests,
        """
fn min3(a: i64, b: i64, c: i64) -> i64 {
    m: i64 = a;
    if b < m { m = b; }
    if c < m { m = c; }
    return m;
}
""",
    )


def _make_reverse_array(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Return the sum of a reversed array (tests array iteration)."""
    tests = []
    for _ in range(4):
        arr = [rng.randint(1, 10) for _ in range(rng.randint(2, 5))]
        # Return sum of reversed (which equals sum of original for integers)
        tests.append(((arr,), str(sum(arr))))
    return _problem(
        f"reverse_sum_v{variant}", "arrays",
        "Sum all elements of an array (tests array iteration).",
        "fn reverse_sum(arr: [i64]) -> i64",
        tests,
        """
fn reverse_sum(arr: [i64]) -> i64 {
    total: i64 = 0;
    for item in arr {
        total = total + item;
    }
    return total;
}
""",
    )


def _make_second_largest(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Find the largest element in an array."""
    tests = []
    for _ in range(4):
        arr = [rng.randint(1, 20) for _ in range(rng.randint(3, 6))]
        tests.append(((arr,), str(max(arr))))
    return _problem(
        f"array_max_v{variant}", "arrays",
        "Find the maximum element in an array.",
        "fn array_max_elem(arr: [i64]) -> i64",
        tests,
        """
fn array_max_elem(arr: [i64]) -> i64 {
    best := arr[0];
    for item in arr {
        if item > best {
            best = item;
        }
    }
    return best;
}
""",
    )


def _make_is_prime(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Check if a number is prime (returns 1 or 0)."""
    tests = []
    for n in [2, 3, 4, 5, 7, 10, 11, 13, 15, 17]:
        is_p = 1 if n >= 2 and all(n % d != 0 for d in range(2, int(n ** 0.5) + 1)) else 0
        tests.append(((n,), str(is_p)))
    return _problem(
        f"is_prime_v{variant}", "loops",
        "Return 1 if the number is prime, 0 otherwise.",
        "fn is_prime(n: i64) -> i64",
        tests,
        """
fn is_prime(n: i64) -> i64 {
    if n < 2 { return 0; }
    if n == 2 { return 1; }
    if n % 2 == 0 { return 0; }
    i: i64 = 3;
    while i * i <= n {
        if n % i == 0 { return 0; }
        i = i + 2;
    }
    return 1;
}
""",
    )


def _make_nth_triangle(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Nth triangular number: sum from 1 to n."""
    tests = []
    for n in [0, 1, 2, 5, 10, 20]:
        tests.append(((n,), str(n * (n + 1) // 2)))
    return _problem(
        f"nth_triangle_v{variant}", "loops",
        "Return the nth triangular number: 1+2+...+n.",
        "fn nth_triangle(n: i64) -> i64",
        tests,
        """
fn nth_triangle(n: i64) -> i64 {
    total: i64 = 0;
    i: i64 = 1;
    while i <= n {
        total = total + i;
        i = i + 1;
    }
    return total;
}
""",
    )


def _make_fib_iter(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Iterative fibonacci with multi-variable mutation."""
    tests = []
    for n in [0, 1, 2, 5, 7, 10]:
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        tests.append(((n,), str(a)))
    return _problem(
        f"fib_iter_v{variant}", "loops",
        "Return the nth Fibonacci number using iterative multi-variable update.",
        "fn fib_iter(n: i64) -> i64",
        tests,
        """
fn fib_iter(n: i64) -> i64 {
    if n == 0 { return 0; }
    if n == 1 { return 1; }
    a: i64 = 0;
    b: i64 = 1;
    i: i64 = 2;
    while i <= n {
        tmp: i64 = a + b;
        a = b;
        b = tmp;
        i = i + 1;
    }
    return b;
}
""",
    )


def _make_palindrome_check(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Check if a string reads the same forwards and backwards."""
    tests = [
        (("racecar",), "1"),
        (("hello",), "0"),
        (("aba",), "1"),
        (("ab",), "0"),
        (("a",), "1"),
        (("",), "1"),
    ]
    return _problem(
        f"palindrome_check_v{variant}", "strings",
        "Return 1 if the string is a palindrome, 0 otherwise.",
        "fn palindrome_check(s: string) -> i64",
        tests,
        """
fn palindrome_check(s: string) -> i64 {
    chars := s.split("");
    left: i64 = 0;
    right: i64 = s.len - 1;
    while left < right {
        if chars[left] != chars[right] { return 0; }
        left = left + 1;
        right = right - 1;
    }
    return 1;
}
""",
    )


def _make_count_words(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Count the number of words in a string (split by spaces)."""
    tests = [
        (("hello world",), "2"),
        (("one",), "1"),
        (("a b c d",), "4"),
        (("  two words  ",), "2"),
        (("",), "0"),
    ]
    return _problem(
        f"count_words_v{variant}", "strings",
        "Count the number of space-separated words in a string.",
        "fn count_words(s: string) -> i64",
        tests,
        """
fn count_words(s: string) -> i64 {
    t := s.trim();
    if t.len == 0 { return 0; }
    parts := t.split(" ");
    count: i64 = 0;
    for p in parts {
        if p.len > 0 {
            count = count + 1;
        }
    }
    return count;
}
""",
    )


def _make_euler_totient(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Euler's totient function: count integers 1..n coprime to n."""
    def _totient(n):
        result = n
        p = 2
        temp = n
        while p * p <= temp:
            if temp % p == 0:
                while temp % p == 0:
                    temp = temp // p
                result -= result // p
            p += 1
        if temp > 1:
            result -= result // temp
        return result

    tests = []
    for n in [1, 2, 3, 5, 6, 9, 10, 12]:
        tests.append(((n,), str(_totient(n))))
    return _problem(
        f"euler_totient_v{variant}", "algorithms",
        "Compute Euler's totient function phi(n): count of integers in [1,n] coprime to n.",
        "fn euler_totient(n: i64) -> i64",
        tests,
        """
fn euler_totient(n: i64) -> i64 {
    result: i64 = n;
    p: i64 = 2;
    temp: i64 = n;
    while p * p <= temp {
        if temp % p == 0 {
            while temp % p == 0 {
                temp = temp / p;
            }
            result = result - result / p;
        }
        p = p + 1;
    }
    if temp > 1 {
        result = result - result / temp;
    }
    return result;
}
""",
    )


def _make_sum_squares(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Sum of squares from 1 to n: 1² + 2² + ... + n²."""
    tests = []
    for n in [0, 1, 2, 3, 5, 10]:
        result = sum(i * i for i in range(1, n + 1))
        tests.append(((n,), str(result)))
    return _problem(
        f"sum_squares_v{variant}", "loops",
        "Compute the sum of squares from 1 to n.",
        "fn sum_squares(n: i64) -> i64",
        tests,
        """
fn sum_squares(n: i64) -> i64 {
    total: i64 = 0;
    i: i64 = 1;
    while i <= n {
        total = total + i * i;
        i = i + 1;
    }
    return total;
}
""",
    )


def _make_product_1_to_n(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Product of integers from 1 to n (factorial variant)."""
    tests = []
    for n in [0, 1, 2, 4, 6]:
        p = 1
        for i in range(1, n + 1):
            p *= i
        tests.append(((n,), str(p)))
    return _problem(
        f"product_1_to_n_v{variant}", "loops",
        "Compute the product of all integers from 1 to n.",
        "fn product_1_to_n(n: i64) -> i64",
        tests,
        """
fn product_1_to_n(n: i64) -> i64 {
    if n == 0 { return 1; }
    total: i64 = 1;
    i: i64 = 1;
    while i <= n {
        total = total * i;
        i = i + 1;
    }
    return total;
}
""",
    )


def _make_count_divisors(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Count the number of divisors of n."""
    tests = []
    for n in [1, 2, 6, 12, 7, 10]:
        count = sum(1 for d in range(1, n + 1) if n % d == 0)
        tests.append(((n,), str(count)))
    return _problem(
        f"count_divisors_v{variant}", "loops",
        "Count how many positive divisors n has.",
        "fn count_divisors(n: i64) -> i64",
        tests,
        """
fn count_divisors(n: i64) -> i64 {
    count: i64 = 0;
    i: i64 = 1;
    while i <= n {
        if n % i == 0 {
            count = count + 1;
        }
        i = i + 1;
    }
    return count;
}
""",
    )


def _make_triangular_check(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Check if n is a triangular number (exists k such that k*(k+1)/2 == n)."""
    tests = []
    for n in [0, 1, 3, 6, 10, 15, 2, 4, 7, 8]:
        is_tri = 0
        k = 0
        while k * (k + 1) // 2 <= n:
            if k * (k + 1) // 2 == n:
                is_tri = 1
                break
            k += 1
        tests.append(((n,), str(is_tri)))
    return _problem(
        f"triangular_check_v{variant}", "algorithms",
        "Return 1 if n is a triangular number (n = k*(k+1)/2 for some k), 0 otherwise.",
        "fn triangular_check(n: i64) -> i64",
        tests,
        """
fn triangular_check(n: i64) -> i64 {
    k: i64 = 0;
    while k * (k + 1) / 2 <= n {
        if k * (k + 1) / 2 == n { return 1; }
        k = k + 1;
    }
    return 0;
}
""",
    )


def _make_max_pair_diff(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Maximum difference between consecutive elements in an array."""
    tests = []
    for _ in range(4):
        arr = [rng.randint(1, 20) for _ in range(rng.randint(3, 6))]
        if len(arr) < 2:
            arr = [1, 5, 3]
        max_diff = max(abs(arr[i] - arr[i + 1]) for i in range(len(arr) - 1))
        tests.append(((arr,), str(max_diff)))
    return _problem(
        f"max_pair_diff_v{variant}", "arrays",
        "Find the maximum absolute difference between consecutive elements.",
        "fn max_pair_diff(arr: [i64]) -> i64",
        tests,
        """
fn max_pair_diff(arr: [i64]) -> i64 {
    best: i64 = 0;
    i: i64 = 1;
    while i < arr.len {
        diff: i64 = arr[i] - arr[i - 1];
        if diff < 0 { diff = 0 - diff; }
        if diff > best { best = diff; }
        i = i + 1;
    }
    return best;
}
""",
    )


def _make_sum_negatives(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Sum of all negative numbers in an array."""
    tests = []
    for _ in range(4):
        arr = [rng.randint(-10, 10) for _ in range(rng.randint(3, 6))]
        neg_sum = sum(x for x in arr if x < 0)
        tests.append(((arr,), str(neg_sum)))
    return _problem(
        f"sum_negatives_v{variant}", "arrays",
        "Sum all negative numbers in the array.",
        "fn sum_negatives(arr: [i64]) -> i64",
        tests,
        """
fn sum_negatives(arr: [i64]) -> i64 {
    total: i64 = 0;
    for item in arr {
        if item < 0 {
            total = total + item;
        }
    }
    return total;
}
""",
    )


def _make_gcd_extended(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """GCD with intermediate variable mutation (tests while-loop with swap pattern)."""
    tests = []
    for a, b in [(12, 8), (35, 14), (7, 13), (100, 75), (0, 5), (6, 0)]:
        x, y = a, b
        while y != 0:
            x, y = y, x % y
        tests.append(((a, b), str(x)))
    return _problem(
        f"gcd_extended_v{variant}", "algorithms",
        "Compute the GCD of two non-negative integers using Euclidean algorithm with variable swap.",
        "fn gcd_extended(a: i64, b: i64) -> i64",
        tests,
        """
fn gcd_extended(a: i64, b: i64) -> i64 {
    x: i64 = a;
    y: i64 = b;
    while y != 0 {
        tmp: i64 = y;
        y = x % y;
        x = tmp;
    }
    return x;
}
""",
    )


def _make_harmonic_sum(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Integer approximation of harmonic sum: 1 + 1/2 + ... + 1/n, multiplied by 1000."""
    tests = []
    for n in [1, 2, 5, 10]:
        h = sum(1000 // i for i in range(1, n + 1))
        tests.append(((n,), str(h)))
    return _problem(
        f"harmonic_sum_v{variant}", "loops",
        "Compute integer harmonic sum: sum of 1000/i for i from 1 to n.",
        "fn harmonic_sum(n: i64) -> i64",
        tests,
        """
fn harmonic_sum(n: i64) -> i64 {
    total: i64 = 0;
    i: i64 = 1;
    while i <= n {
        total = total + 1000 / i;
        i = i + 1;
    }
    return total;
}
""",
    )


def _make_interactive_sum(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Sum all integers in an array (tests array iteration)."""
    tests = []
    for _ in range(4):
        nums = [rng.randint(1, 10) for _ in range(rng.randint(2, 5))]
        tests.append(((nums,), str(sum(nums))))
    return _problem(
        f"interactive_sum_v{variant}", "arrays",
        "Return the sum of all integers in an array.",
        "fn interactive_sum(arr: [i64]) -> i64",
        tests,
        """
fn interactive_sum(arr: [i64]) -> i64 {
    total: i64 = 0;
    for item in arr {
        total = total + item;
    }
    return total;
}
""",
    )


def _make_max_consecutive_sum(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Maximum sum of a contiguous subarray (Kadane's algorithm)."""
    def kadane(arr: list[int]) -> int:
        current = arr[0]
        best = arr[0]
        for x in arr[1:]:
            current = max(x, current + x)
            best = max(best, current)
        return best

    # Fixed canonical cases to ensure the gradient solver gets clean signal
    fixed_cases = [
        [1, -2, 3],       # answer: 3
        [3, -1, 2],       # answer: 4
        [-1, -2, -3],     # answer: -1
        [2, 3, -1, 4],    # answer: 8
    ]
    tests = [((arr,), str(kadane(arr))) for arr in fixed_cases]
    return _problem(
        f"max_consecutive_sum_v{variant}", "arrays",
        "Return the maximum sum of any contiguous subarray (Kadane's algorithm).",
        "fn max_consecutive_sum(arr: [i64]) -> i64",
        tests,
        """
fn max_consecutive_sum(arr: [i64]) -> i64 {
    current: i64 = arr[0];
    best: i64 = arr[0];
    i: i64 = 1;
    while i < len(arr) {
        if current + arr[i] > arr[i] {
            current = current + arr[i];
        } else {
            current = arr[i];
        }
        if current > best {
            best = current;
        }
        i = i + 1;
    }
    return best;
}
""",
    )


def _make_min_consecutive_sum(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Minimum sum of a contiguous subarray (anti-Kadane's algorithm)."""
    def anti_kadane(arr: list[int]) -> int:
        current = arr[0]
        best = arr[0]
        for x in arr[1:]:
            current = min(x, current + x)
            best = min(best, current)
        return best

    fixed_cases = [
        [1, -2, 3],        # answer: -2
        [3, -1, -2, 5],    # answer: -3
        [1, 2, 3, 4],      # answer: 1
        [-2, -3, 1, -4],   # answer: -8
    ]
    tests = [((arr,), str(anti_kadane(arr))) for arr in fixed_cases]
    return _problem(
        f"min_consecutive_sum_v{variant}", "arrays",
        "Return the minimum sum of any contiguous subarray (anti-Kadane's algorithm).",
        "fn min_consecutive_sum(arr: [i64]) -> i64",
        tests,
        """
fn min_consecutive_sum(arr: [i64]) -> i64 {
    current: i64 = arr[0];
    best: i64 = arr[0];
    i: i64 = 1;
    while i < len(arr) {
        if current + arr[i] < arr[i] {
            current = current + arr[i];
        } else {
            current = arr[i];
        }
        if current < best {
            best = current;
        }
        i = i + 1;
    }
    return best;
}
""",
    )


# --- Phase 1: Scalar Digit Manipulation ---


def _make_reverse_digits(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Reverse the digits of a non-negative integer."""
    tests = []
    for n in [0, 9, 12, 123, 100, 4321]:
        result = int(str(n)[::-1]) if n != 0 else 0
        tests.append(((n,), str(result)))
    return _problem(
        f"reverse_digits_v{variant}", "loops",
        "Reverse the digits of a non-negative integer (e.g. 123 -> 321).",
        "fn reverse_digits(n: i64) -> i64",
        tests,
        """
fn reverse_digits(n: i64) -> i64 {
    result: i64 = 0;
    x: i64 = n;
    while x > 0 {
        result = result * 10 + x % 10;
        x = x / 10;
    }
    return result;
}
""",
    )


def _make_digit_count(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Count the number of digits in a non-negative integer."""
    tests = []
    for n in [0, 1, 9, 10, 99, 100, 999]:
        count = 1 if n == 0 else len(str(n))
        tests.append(((n,), str(count)))
    return _problem(
        f"digit_count_v{variant}", "loops",
        "Count the number of digits in a non-negative integer.",
        "fn digit_count(n: i64) -> i64",
        tests,
        """
fn digit_count(n: i64) -> i64 {
    if n == 0 { return 1; }
    count: i64 = 0;
    x: i64 = n;
    while x > 0 {
        count = count + 1;
        x = x / 10;
    }
    return count;
}
""",
    )


def _make_count_even_digits(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Count the even digits in a non-negative integer."""
    def _count(n):
        if n == 0:
            return 1  # 0 is even
        c = 0
        while n > 0:
            if (n % 10) % 2 == 0:
                c += 1
            n //= 10
        return c

    tests = []
    for n in [0, 2, 13, 24, 135, 2468]:
        tests.append(((n,), str(_count(n))))
    return _problem(
        f"count_even_digits_v{variant}", "loops",
        "Count how many digits of n are even.",
        "fn count_even_digits(n: i64) -> i64",
        tests,
        """
fn count_even_digits(n: i64) -> i64 {
    if n == 0 { return 1; }
    count: i64 = 0;
    x: i64 = n;
    while x > 0 {
        if (x % 10) % 2 == 0 { count = count + 1; }
        x = x / 10;
    }
    return count;
}
""",
    )


# --- Phase 2: Algorithmic Scalar ---


def _make_perfect_check(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Check if n is a perfect number (sum of proper divisors equals n)."""
    def _is_perfect(n):
        if n < 2:
            return 0
        total = 1
        i = 2
        while i * i <= n:
            if n % i == 0:
                total += i
                if i * i != n:
                    total += n // i
            i += 1
        return 1 if total == n else 0

    tests = []
    for n in [1, 6, 12, 28, 30, 496]:
        tests.append(((n,), str(_is_perfect(n))))
    return _problem(
        f"perfect_check_v{variant}", "algorithms",
        "Return 1 if n is a perfect number (sum of proper divisors equals n), 0 otherwise.",
        "fn perfect_check(n: i64) -> i64",
        tests,
        """
fn perfect_check(n: i64) -> i64 {
    if n < 2 { return 0; }
    total: i64 = 1;
    i: i64 = 2;
    while i * i <= n {
        if n % i == 0 {
            total = total + i;
            if i * i != n { total = total + n / i; }
        }
        i = i + 1;
    }
    if total == n { return 1; }
    return 0;
}
""",
    )


def _make_armstrong_check(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Check if n is an Armstrong number (sum of cubes of digits equals n)."""
    def _is_armstrong(n):
        total = 0
        x = n
        while x > 0:
            d = x % 10
            total += d * d * d
            x //= 10
        return 1 if total == n else 0

    tests = []
    for n in [0, 1, 153, 370, 371, 100, 200]:
        tests.append(((n,), str(_is_armstrong(n))))
    return _problem(
        f"armstrong_check_v{variant}", "algorithms",
        "Return 1 if n is an Armstrong number (sum of cubes of its digits equals n), 0 otherwise.",
        "fn armstrong_check(n: i64) -> i64",
        tests,
        """
fn armstrong_check(n: i64) -> i64 {
    total: i64 = 0;
    x: i64 = n;
    while x > 0 {
        d: i64 = x % 10;
        total = total + d * d * d;
        x = x / 10;
    }
    if total == n { return 1; }
    return 0;
}
""",
    )


def _make_geometric_sum(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Geometric series sum: 1 + 2 + 4 + ... + 2^n = 2^(n+1) - 1."""
    tests = []
    for n in [0, 1, 2, 3, 4, 6]:
        result = (1 << (n + 1)) - 1
        tests.append(((n,), str(result)))
    return _problem(
        f"geometric_sum_v{variant}", "loops",
        "Compute 1 + 2 + 4 + ... + 2^n (geometric series with ratio 2).",
        "fn geometric_sum(n: i64) -> i64",
        tests,
        """
fn geometric_sum(n: i64) -> i64 {
    total: i64 = 1;
    power: i64 = 1;
    i: i64 = 0;
    while i < n {
        power = power * 2;
        total = total + power;
        i = i + 1;
    }
    return total;
}
""",
    )


def _make_nested_sum(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Nested loop sum: sum(i*j for i=1..n, j=1..i)."""
    def _compute(n):
        total = 0
        for i in range(1, n + 1):
            for j in range(1, i + 1):
                total += i * j
        return total

    tests = []
    for n in [0, 1, 2, 3, 4]:
        tests.append(((n,), str(_compute(n))))
    return _problem(
        f"nested_sum_v{variant}", "loops",
        "Compute sum(i*j for i=1..n, j=1..i) using nested loops.",
        "fn nested_sum(n: i64) -> i64",
        tests,
        """
fn nested_sum(n: i64) -> i64 {
    total: i64 = 0;
    i: i64 = 1;
    while i <= n {
        j: i64 = 1;
        while j <= i {
            total = total + i * j;
            j = j + 1;
        }
        i = i + 1;
    }
    return total;
}
""",
    )


# --- Phase 3: Early-Return Array Search ---


def _make_find_first_even(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Return the index of the first even element, or -1 if none."""
    def _first_even(arr):
        for i, x in enumerate(arr):
            if x % 2 == 0:
                return i
        return -1

    tests = []
    cases = [[2, 3, 5], [1, 3, 4, 6], [1, 3, 5], [4], [7, 8, 9]]
    for arr in cases:
        tests.append(((arr,), str(_first_even(arr))))
    return _problem(
        f"find_first_even_v{variant}", "arrays",
        "Return the index of the first even element in arr, or -1 if no even element.",
        "fn find_first_even(arr: [i64]) -> i64",
        tests,
        """
fn find_first_even(arr: [i64]) -> i64 {
    i: i64 = 0;
    while i < arr.len {
        if arr[i] % 2 == 0 { return i; }
        i = i + 1;
    }
    return -1;
}
""",
    )


def _make_sum_until_negative(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Sum elements until the first negative is encountered."""
    def _sum_until(arr):
        total = 0
        for x in arr:
            if x < 0:
                return total
            total += x
        return total

    tests = []
    cases = [[1, 2, -3, 4], [5, 3, 1], [-1, 2, 3], [0, 5, -2, 7]]
    for arr in cases:
        tests.append(((arr,), str(_sum_until(arr))))
    return _problem(
        f"sum_until_negative_v{variant}", "arrays",
        "Sum elements of arr until (not including) the first negative element.",
        "fn sum_until_negative(arr: [i64]) -> i64",
        tests,
        """
fn sum_until_negative(arr: [i64]) -> i64 {
    total: i64 = 0;
    i: i64 = 0;
    while i < arr.len {
        if arr[i] < 0 { return total; }
        total = total + arr[i];
        i = i + 1;
    }
    return total;
}
""",
    )


# --- Phase 4: Array Access Patterns ---


def _make_sort_and_sum(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Return the sum of the minimum and maximum elements."""
    tests = []
    for _ in range(5):
        arr = [rng.randint(1, 20) for _ in range(rng.randint(2, 6))]
        tests.append(((arr,), str(min(arr) + max(arr))))
    return _problem(
        f"sort_and_sum_v{variant}", "arrays",
        "Return the sum of the minimum and maximum elements of arr.",
        "fn sort_and_sum(arr: [i64]) -> i64",
        tests,
        """
fn sort_and_sum(arr: [i64]) -> i64 {
    mn := arr[0];
    mx := arr[0];
    for item in arr {
        if item < mn { mn = item; }
        if item > mx { mx = item; }
    }
    return mn + mx;
}
""",
    )


def _make_array_triple(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Sum all elements after tripling each one."""
    tests = []
    for _ in range(5):
        arr = [rng.randint(1, 10) for _ in range(rng.randint(2, 5))]
        tests.append(((arr,), str(sum(x * 3 for x in arr))))
    return _problem(
        f"array_triple_v{variant}", "arrays",
        "Return the sum of each element multiplied by 3.",
        "fn array_triple(arr: [i64]) -> i64",
        tests,
        """
fn array_triple(arr: [i64]) -> i64 {
    total: i64 = 0;
    for item in arr {
        total = total + item * 3;
    }
    return total;
}
""",
    )


def _make_sum_even_indexed(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Sum elements at even indices (0, 2, 4, ...) using while + index access."""
    def _sum_even_idx(arr):
        return sum(arr[i] for i in range(0, len(arr), 2))

    tests = []
    cases = [[1, 2, 3, 4, 5], [10, 20, 30], [7], [4, 8, 2, 6]]
    for arr in cases:
        tests.append(((arr,), str(_sum_even_idx(arr))))
    return _problem(
        f"sum_even_indexed_v{variant}", "arrays",
        "Sum elements at even indices (0, 2, 4, ...) of arr.",
        "fn sum_even_indexed(arr: [i64]) -> i64",
        tests,
        """
fn sum_even_indexed(arr: [i64]) -> i64 {
    total: i64 = 0;
    i: i64 = 0;
    while i < arr.len {
        total = total + arr[i];
        i = i + 2;
    }
    return total;
}
""",
    )


def _make_last_element(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Return the last element of an array using index access."""
    tests = []
    for _ in range(5):
        arr = [rng.randint(1, 20) for _ in range(rng.randint(1, 6))]
        tests.append(((arr,), str(arr[-1])))
    return _problem(
        f"last_element_v{variant}", "arrays",
        "Return the last element of arr.",
        "fn last_element(arr: [i64]) -> i64",
        tests,
        """
fn last_element(arr: [i64]) -> i64 {
    return arr[arr.len - 1];
}
""",
    )


# --- Phase 5: Advanced Scalar ---


def _make_fib_cached(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Iterative Fibonacci using running two-variable state."""
    tests = []
    for n in [0, 1, 2, 5, 7, 10]:
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        tests.append(((n,), str(a)))
    return _problem(
        f"fib_cached_v{variant}", "loops",
        "Compute the nth Fibonacci number using iterative two-variable state.",
        "fn fib_cached(n: i64) -> i64",
        tests,
        """
fn fib_cached(n: i64) -> i64 {
    if n == 0 { return 0; }
    if n == 1 { return 1; }
    a: i64 = 0;
    b: i64 = 1;
    i: i64 = 2;
    while i <= n {
        tmp: i64 = a + b;
        a = b;
        b = tmp;
        i = i + 1;
    }
    return b;
}
""",
    )


def _make_mersenne_check(rng: random.Random, variant: int) -> MogBenchmarkProblem:
    """Check if n is a Mersenne number (n = 2^k - 1 for some k >= 1)."""
    def _is_mersenne(n):
        if n < 1:
            return 0
        m = n + 1
        while m > 1:
            if m % 2 != 0:
                return 0
            m //= 2
        return 1

    tests = []
    for n in [1, 3, 5, 7, 6, 15, 14, 31]:
        tests.append(((n,), str(_is_mersenne(n))))
    return _problem(
        f"mersenne_check_v{variant}", "algorithms",
        "Return 1 if n is a Mersenne number (n = 2^k - 1), 0 otherwise.",
        "fn mersenne_check(n: i64) -> i64",
        tests,
        """
fn mersenne_check(n: i64) -> i64 {
    if n < 1 { return 0; }
    m: i64 = n + 1;
    while m > 1 {
        if m % 2 != 0 { return 0; }
        m = m / 2;
    }
    return 1;
}
""",
    )


PROBLEM_FACTORIES: list[Callable[[random.Random, int], MogBenchmarkProblem]] = [
    _make_add_two,
    _make_abs_diff,
    _make_max2,
    _make_clamp,
    _make_sign,
    _make_sum_to_n,
    _make_gcd,
    _make_lcm,
    _make_array_sum,
    _make_array_max,
    _make_count_occurrences,
    _make_trimmed_len,
    _make_vowel_count,
    _make_contains_cat,
    _make_point_sum,
    _make_safe_div_or_neg1,
    _make_positive_or_default,
    _make_factorial,
    _make_fibonacci,
    _make_closure_map_sum,
    _make_count_positive,
    _make_is_even,
    _make_digit_sum,
    _make_starts_with_m,
    _make_rectangle_area,
    _make_power,
    _make_polynomial,
    _make_collatz_steps,
    _make_min3,
    _make_reverse_array,
    _make_second_largest,
    _make_is_prime,
    _make_nth_triangle,
    _make_fib_iter,
    _make_palindrome_check,
    _make_count_words,
    _make_euler_totient,
    _make_sum_squares,
    _make_product_1_to_n,
    _make_count_divisors,
    _make_triangular_check,
    _make_max_pair_diff,
    _make_sum_negatives,
    _make_gcd_extended,
    _make_harmonic_sum,
    _make_interactive_sum,
    _make_max_consecutive_sum,
    _make_min_consecutive_sum,
    # Phase 1: Scalar Digit Manipulation
    _make_reverse_digits,
    _make_digit_count,
    _make_count_even_digits,
    # Phase 2: Algorithmic Scalar
    _make_perfect_check,
    _make_armstrong_check,
    _make_geometric_sum,
    _make_nested_sum,
    # Phase 3: Early-Return Array Search
    _make_find_first_even,
    _make_sum_until_negative,
    # Phase 4: Array Access Patterns
    _make_sort_and_sum,
    _make_array_triple,
    _make_sum_even_indexed,
    _make_last_element,
    # Phase 5: Advanced Scalar
    _make_fib_cached,
    _make_mersenne_check,
]


def get_benchmark(seed: int = 42, variants_per_factory: int = 5) -> list[MogBenchmarkProblem]:
    """Return a benchmark suite with 100+ problems by default.

    25 factories * 5 variants = 125 problems.
    """
    rng = random.Random(seed)
    problems: list[MogBenchmarkProblem] = []
    for factory in PROBLEM_FACTORIES:
        for variant in range(variants_per_factory):
            problems.append(factory(rng, variant))
    return problems


if __name__ == "__main__":
    benchmark = get_benchmark(seed=42, variants_per_factory=5)
    print(f"Generated {len(benchmark)} benchmark problems")

    # Smoke-test 10 reference solutions with interpreter.
    checked = 0
    for problem in benchmark[:10]:
        result = evaluate_solution(problem, problem.reference_solution or "")
        status = "PASS" if result.passed else "FAIL"
        print(f"{status}: {problem.name}")
        if not result.passed:
            print("  expected:", result.expected_output)
            print("  actual:  ", result.actual_output)
            print("  error:   ", result.error)
            raise SystemExit(1)
        checked += 1

    print(f"Reference smoke test passed for {checked} problems")

    # Cross-check a few with the real compiler too.
    for problem in benchmark[:5]:
        result = evaluate_solution_with_compiler(problem, problem.reference_solution or "")
        status = "PASS" if result.passed else "FAIL"
        print(f"compiler {status}: {problem.name}")
        if not result.passed:
            print("  expected:", result.expected_output)
            print("  actual:  ", result.actual_output)
            print("  error:   ", result.error)
            raise SystemExit(1)

    print("Compiler cross-check passed for 5 problems")
