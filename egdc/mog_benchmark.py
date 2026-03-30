"""Execution benchmark for Mog code generation.

This benchmark stays within the compiler-safe subset documented in
`egdc/mog_compiler_compat.md`. It uses the local Python Mog interpreter for
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

from egdc.mog_lang import interpret
from egdc.mog_execute import execute_mog


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
    return MogBenchmarkResult(
        problem_name=problem.name,
        passed=result.success and actual == expected.strip(),
        expected_output=expected.strip(),
        actual_output=actual,
        error=result.stderr if not result.success else None,
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
