#!/usr/bin/env python3
"""
Generate training data for the expr meta-learner program-type classifier.

Generates synthetic problems with known program types, runs the mog_synth
binary to verify solvability and discover which method solves each, then
outputs JSONL records suitable for training train_expr_metalearner.py.

Problem categories:
  - expr:           simple arithmetic (a+b, a*b, a-1, a*10)
  - two_precomp:    two pre-computations (a*b+c, celsius_to_fahrenheit)
  - branch:         conditional expressions (max, min, sign, abs, clamp)
  - loop:           iterative computations (sum_to_n, factorial, fibonacci)
  - chained_branch: two sequential ternary branches (min3, max3, clamp)

Output format (JSONL):
  {"io_pairs": [[inputs, output], ...], "n_args": N,
   "method": "expr"|"two_precomp"|"branch"|"loop"|"chained_branch",
   "code": "fn f(...) -> i64 { ... }"}

Usage:
  python3 scripts/generate_training_data.py --out data/expr_type_train.jsonl
  python3 scripts/generate_training_data.py --out data/expr_type_train.jsonl --count 1000 --verify
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ─── Program type definitions ────────────────────────────────────────────────

PROGRAM_TYPES = ["expr", "two_precomp", "branch", "loop", "chained_branch"]
TYPE_TO_IDX = {t: i for i, t in enumerate(PROGRAM_TYPES)}

MOG_SYNTH_BIN = str(Path(__file__).parent.parent / "target" / "release" / "mog_synth")


# ─── Evaluation helpers ──────────────────────────────────────────────────────

def safe_div(a: int, b: int) -> Optional[int]:
    if b == 0:
        return None
    return int(a / b) if (a < 0) != (b < 0) and a % b != 0 else a // b


def safe_mod(a: int, b: int) -> Optional[int]:
    if b == 0:
        return None
    return a - int(a / b) * b


def eval_fn(fn, args: list[int]) -> Optional[int]:
    """Evaluate a function on integer args, returning None on error."""
    try:
        result = fn(*args)
        if result is None or not isinstance(result, (int, float)):
            return None
        result = int(result)
        if abs(result) > 10**9:
            return None
        return result
    except (ZeroDivisionError, OverflowError, ValueError, RecursionError):
        return None


# ─── Problem generators by program type ──────────────────────────────────────

def _gen_expr_problems(rng: random.Random, count: int) -> list[dict]:
    """Generate simple expression problems (1 or 2 ops, no branches/loops)."""
    records = []

    # Unary: a+c, a-c, a*c, c*a+c2
    unary_templates = [
        # (description, lambda, code, n_args)
        lambda c: (f"a+{c}", lambda a: a + c, f"return a + {c};", 1),
        lambda c: (f"a-{c}", lambda a: a - c, f"return a - {c};", 1),
        lambda c: (f"a*{c}", lambda a: a * c, f"return a * {c};", 1),
        lambda c: (f"{c}*a", lambda a: c * a, f"return {c} * a;", 1),
        lambda c: (f"a*a", lambda a: a * a, f"v0: i64 = a * a;\n    return v0;", 1),
        lambda c: (f"a+a", lambda a: a + a, f"return a + a;", 1),
        lambda c: (f"a*a+{c}", lambda a: a * a + c, f"v0: i64 = a * a;\n    return v0 + {c};", 1),
    ]

    # Binary: a+b, a-b, a*b, a*b+c, etc.
    binary_templates = [
        lambda c: (f"a+b", lambda a, b: a + b, f"return a + b;", 2),
        lambda c: (f"a-b", lambda a, b: a - b, f"return a - b;", 2),
        lambda c: (f"a*b", lambda a, b: a * b, f"return a * b;", 2),
        lambda c: (f"a+b+{c}", lambda a, b: a + b + c, f"v0: i64 = a + b;\n    return v0 + {c};", 2),
        lambda c: (f"a*b+{c}", lambda a, b: a * b + c, f"v0: i64 = a * b;\n    return v0 + {c};", 2),
        lambda c: (f"a*{c}+b", lambda a, b: a * c + b, f"v0: i64 = a * {c};\n    return v0 + b;", 2),
        lambda c: (f"a+b*{c}", lambda a, b: a + b * c, f"v0: i64 = b * {c};\n    return a + v0;", 2),
    ]

    # Ternary
    ternary_templates = [
        lambda c: (f"a+b+c", lambda a, b, c_: a + b + c_, f"v0: i64 = a + b;\n    return v0 + c;", 3),
        lambda c: (f"a*b+c", lambda a, b, c_: a * b + c_, f"v0: i64 = a * b;\n    return v0 + c;", 3),
        lambda c: (f"a*b*c", lambda a, b, c_: a * b * c_, f"v0: i64 = a * b;\n    return v0 * c;", 3),
    ]

    templates = unary_templates + binary_templates + ternary_templates

    while len(records) < count:
        tmpl = rng.choice(templates)
        c = rng.choice([0, 1, -1, 2, -2, 3, 5, 7, 10, -3, -5, 100])
        desc, fn, code_body, n_args = tmpl(c)

        # Generate I/O examples
        examples = []
        seen = set()
        for _ in range(50):
            if len(examples) >= 8:
                break
            args = [rng.randint(-20, 30) for _ in range(n_args)]
            key = tuple(args)
            if key in seen:
                continue
            seen.add(key)
            result = eval_fn(fn, args)
            if result is not None:
                examples.append((args, result))

        if len(examples) < 4:
            continue

        # Check non-trivial
        outputs = set(t for _, t in examples)
        if len(outputs) <= 1:
            continue

        param_names = ["a", "b", "c", "d"][:n_args]
        params_sig = ", ".join(f"{n}: i64" for n in param_names)
        code = f"fn f({params_sig}) -> i64 {{\n    {code_body}\n}}\n"

        records.append({
            "io_pairs": [[list(inp), out] for inp, out in examples],
            "n_args": n_args,
            "method": "expr",
            "code": code,
            "name": desc,
        })

    return records[:count]


def _gen_two_precomp_problems(rng: random.Random, count: int) -> list[dict]:
    """Generate two-precomputation expression problems."""
    records = []

    templates = [
        # a*b + c  (3 args, two precomps: v0=a*b, ret=v0+c)
        lambda c: ("a*b+c_3arg", lambda a, b, c_: a * b + c_,
                    "v0: i64 = a * b;\n    return v0 + c;", 3),
        # (a+b) * (a-b)
        lambda c: ("diff_of_squares", lambda a, b: (a + b) * (a - b),
                    "v0: i64 = a + b;\n    v1: i64 = a - b;\n    return v0 * v1;", 2),
        # a*c1 + c2 (linear: scale + offset)
        lambda c: (f"a*{c}+{c+1}", lambda a: a * c + (c + 1),
                    f"v0: i64 = a * {c};\n    return v0 + {c+1};", 1),
        # a*a + a (a squared plus a)
        lambda c: ("a_sq_plus_a", lambda a: a * a + a,
                    "v0: i64 = a * a;\n    return v0 + a;", 1),
        # a*a - b*b
        lambda c: ("a2_minus_b2", lambda a, b: a * a - b * b,
                    "v0: i64 = a * a;\n    v1: i64 = b * b;\n    return v0 - v1;", 2),
        # (a + b) * c
        lambda c: ("sum_times_c", lambda a, b, c_: (a + b) * c_,
                    "v0: i64 = a + b;\n    return v0 * c;", 3),
        # a * b - a  (product_offset)
        lambda c: ("product_offset", lambda a, b: a * b - a,
                    "v0: i64 = a * b;\n    return v0 - a;", 2),
        # a * a + b * b
        lambda c: ("sum_of_squares_2arg", lambda a, b: a * a + b * b,
                    "v0: i64 = a * a;\n    v1: i64 = b * b;\n    return v0 + v1;", 2),
    ]

    while len(records) < count:
        tmpl = rng.choice(templates)
        c = rng.choice([2, 3, 5, 7, 9, 10, -1, -2])
        desc, fn, code_body, n_args = tmpl(c)

        examples = []
        seen = set()
        for _ in range(50):
            if len(examples) >= 8:
                break
            args = [rng.randint(-10, 20) for _ in range(n_args)]
            key = tuple(args)
            if key in seen:
                continue
            seen.add(key)
            result = eval_fn(fn, args)
            if result is not None:
                examples.append((args, result))

        if len(examples) < 4:
            continue

        outputs = set(t for _, t in examples)
        if len(outputs) <= 1:
            continue

        param_names = ["a", "b", "c", "d"][:n_args]
        params_sig = ", ".join(f"{n}: i64" for n in param_names)
        code = f"fn f({params_sig}) -> i64 {{\n    {code_body}\n}}\n"

        records.append({
            "io_pairs": [[list(inp), out] for inp, out in examples],
            "n_args": n_args,
            "method": "two_precomp",
            "code": code,
            "name": desc,
        })

    return records[:count]


def _gen_branch_problems(rng: random.Random, count: int) -> list[dict]:
    """Generate conditional/branch problems."""
    records = []

    templates = [
        # max(a, b)
        ("max2", 2, lambda a, b: max(a, b),
         "if a >= b { return a; }\n    return b;"),
        # min(a, b)
        ("min2", 2, lambda a, b: min(a, b),
         "if a <= b { return a; }\n    return b;"),
        # abs(a)
        ("abs", 1, lambda a: abs(a),
         "if a >= 0 { return a; }\n    return 0 - a;"),
        # sign(a)
        ("sign", 1, lambda a: (1 if a > 0 else (-1 if a < 0 else 0)),
         "if a > 0 { return 1; }\n    if a < 0 { return -1; }\n    return 0;"),
        # abs_diff(a, b)
        ("abs_diff", 2, lambda a, b: abs(a - b),
         "if a >= b { return a - b; }\n    return b - a;"),
        # is_positive(a)
        ("is_positive", 1, lambda a: 1 if a > 0 else 0,
         "if a > 0 { return 1; }\n    return 0;"),
        # is_negative(a)
        ("is_negative", 1, lambda a: 1 if a < 0 else 0,
         "if a < 0 { return 1; }\n    return 0;"),
        # is_zero(a)
        ("is_zero", 1, lambda a: 1 if a == 0 else 0,
         "if a == 0 { return 1; }\n    return 0;"),
        # is_even(a)
        ("is_even", 1, lambda a: 1 if a % 2 == 0 else 0,
         "v0: i64 = a % 2;\n    if v0 == 0 { return 1; }\n    return 0;"),
        # positive_or_default(a, b)
        ("positive_or_default", 2, lambda a, b: a if a > 0 else b,
         "if a > 0 { return a; }\n    return b;"),
        # safe_div_or_neg1(a, b)
        ("safe_div_or_neg1", 2, lambda a, b: a // b if b != 0 else -1,
         "if b != 0 { return a / b; }\n    return -1;"),
        # max_pair_diff(a, b)
        ("max_pair_diff", 2, lambda a, b: abs(a - b),
         "if a >= b { return a - b; }\n    return b - a;"),
    ]

    while len(records) < count:
        name, n_args, fn, code_body = rng.choice(templates)

        examples = []
        seen = set()
        for _ in range(50):
            if len(examples) >= 8:
                break
            args = [rng.randint(-20, 30) for _ in range(n_args)]
            key = tuple(args)
            if key in seen:
                continue
            seen.add(key)
            result = eval_fn(fn, args)
            if result is not None:
                examples.append((args, result))

        if len(examples) < 4:
            continue

        outputs = set(t for _, t in examples)
        if len(outputs) <= 1:
            continue

        param_names = ["a", "b", "c", "d"][:n_args]
        params_sig = ", ".join(f"{n}: i64" for n in param_names)
        code = f"fn f({params_sig}) -> i64 {{\n    {code_body}\n}}\n"

        records.append({
            "io_pairs": [[list(inp), out] for inp, out in examples],
            "n_args": n_args,
            "method": "branch",
            "code": code,
            "name": name,
        })

    return records[:count]


def _gen_chained_branch_problems(rng: random.Random, count: int) -> list[dict]:
    """Generate chained branch problems (min3, max3, clamp)."""
    records = []

    templates = [
        # min3(a, b, c)
        ("min3", 3, lambda a, b, c: min(a, b, c),
         "v0: i64 = b;\n    if a <= b { v0 = a; }\n    result: i64 = c;\n    if v0 <= c { result = v0; }\n    return result;"),
        # max3(a, b, c)
        ("max3", 3, lambda a, b, c: max(a, b, c),
         "v0: i64 = b;\n    if a >= b { v0 = a; }\n    result: i64 = c;\n    if v0 >= c { result = v0; }\n    return result;"),
        # clamp(a, lo, hi)
        ("clamp", 3, lambda a, lo, hi: max(lo, min(a, hi)),
         "v0: i64 = a;\n    if a < b { v0 = b; }\n    result: i64 = v0;\n    if v0 > c { result = c; }\n    return result;"),
        # median3(a, b, c)
        ("median3", 3, lambda a, b, c: sorted([a, b, c])[1],
         "v0: i64 = b;\n    if a <= b { v0 = a; }\n    result: i64 = c;\n    if v0 >= c { result = v0; }\n    return result;"),
    ]

    while len(records) < count:
        name, n_args, fn, code_body = rng.choice(templates)

        examples = []
        seen = set()
        for _ in range(50):
            if len(examples) >= 8:
                break
            args = [rng.randint(-15, 25) for _ in range(n_args)]
            key = tuple(args)
            if key in seen:
                continue
            seen.add(key)
            result = eval_fn(fn, args)
            if result is not None:
                examples.append((args, result))

        if len(examples) < 4:
            continue

        outputs = set(t for _, t in examples)
        if len(outputs) <= 1:
            continue

        param_names = ["a", "b", "c", "d"][:n_args]
        params_sig = ", ".join(f"{n}: i64" for n in param_names)
        code = f"fn f({params_sig}) -> i64 {{\n    {code_body}\n}}\n"

        records.append({
            "io_pairs": [[list(inp), out] for inp, out in examples],
            "n_args": n_args,
            "method": "chained_branch",
            "code": code,
            "name": name,
        })

    return records[:count]


def _gen_loop_problems(rng: random.Random, count: int) -> list[dict]:
    """Generate loop-based problems (sum, factorial, fibonacci, gcd, etc.)."""
    records = []

    def sum_to_n(n):
        return n * (n + 1) // 2

    def factorial(n):
        if n < 0:
            return None
        r = 1
        for i in range(1, n + 1):
            r *= i
            if r > 10**9:
                return None
        return r

    def fibonacci(n):
        if n < 0:
            return None
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
            if a > 10**9:
                return None
        return a

    def gcd(a, b):
        a, b = abs(a), abs(b)
        while b:
            a, b = b, a % b
        return a

    def sum_squares(n):
        return sum(i * i for i in range(1, n + 1))

    def product_1_to_n(n):
        if n <= 0:
            return None
        r = 1
        for i in range(1, n + 1):
            r *= i
            if r > 10**9:
                return None
        return r

    def digit_sum(n):
        n = abs(n)
        s = 0
        while n > 0:
            s += n % 10
            n //= 10
        return s

    def digit_count(n):
        if n == 0:
            return 1
        n = abs(n)
        c = 0
        while n > 0:
            c += 1
            n //= 10
        return c

    def count_divisors(n):
        if n <= 0:
            return None
        c = 0
        for i in range(1, n + 1):
            if n % i == 0:
                c += 1
        return c

    def popcount(n):
        if n < 0:
            return None
        c = 0
        while n > 0:
            c += n & 1
            n >>= 1
        return c

    def power(base, exp):
        if exp < 0:
            return None
        r = 1
        for _ in range(exp):
            r *= base
            if abs(r) > 10**9:
                return None
        return r

    def collatz_steps(n):
        if n <= 0:
            return None
        steps = 0
        while n != 1:
            if n % 2 == 0:
                n //= 2
            else:
                n = 3 * n + 1
            steps += 1
            if steps > 1000:
                return None
        return steps

    templates = [
        # (name, n_args, fn, code_sketch, input_range)
        ("sum_to_n", 1, lambda a: sum_to_n(a) if a >= 0 else None,
         "acc: i64 = 0;\n    i: i64 = 1;\n    while i <= a {\n        acc = acc + i;\n        i = i + 1;\n    }\n    return acc;",
         (0, 30)),
        ("factorial", 1, factorial,
         "acc: i64 = 1;\n    i: i64 = 1;\n    while i <= a {\n        acc = acc * i;\n        i = i + 1;\n    }\n    return acc;",
         (0, 10)),
        ("fibonacci", 1, fibonacci,
         "a0: i64 = 0;\n    a1: i64 = 1;\n    i: i64 = 0;\n    while i < a {\n        t: i64 = a1;\n        a1 = a0 + a1;\n        a0 = t;\n        i = i + 1;\n    }\n    return a0;",
         (0, 20)),
        ("gcd", 2, lambda a, b: gcd(a, b) if a > 0 and b > 0 else None,
         "x: i64 = a;\n    y: i64 = b;\n    while y != 0 {\n        t: i64 = y;\n        y = x % y;\n        x = t;\n    }\n    return x;",
         (1, 50)),
        ("sum_squares", 1, lambda a: sum_squares(a) if a >= 0 else None,
         "acc: i64 = 0;\n    i: i64 = 1;\n    while i <= a {\n        acc = acc + i * i;\n        i = i + 1;\n    }\n    return acc;",
         (0, 20)),
        ("digit_sum", 1, lambda a: digit_sum(a) if a >= 0 else None,
         "n: i64 = a;\n    acc: i64 = 0;\n    while n > 0 {\n        acc = acc + n % 10;\n        n = n / 10;\n    }\n    return acc;",
         (0, 9999)),
        ("digit_count", 1, lambda a: digit_count(a) if a >= 0 else None,
         "n: i64 = a;\n    c: i64 = 0;\n    while n > 0 {\n        c = c + 1;\n        n = n / 10;\n    }\n    if c == 0 { return 1; }\n    return c;",
         (0, 99999)),
        ("count_divisors", 1, lambda a: count_divisors(a) if a is not None and a > 0 else None,
         "c: i64 = 0;\n    i: i64 = 1;\n    while i <= a {\n        if a % i == 0 { c = c + 1; }\n        i = i + 1;\n    }\n    return c;",
         (1, 30)),
        ("popcount", 1, lambda a: popcount(a) if a >= 0 else None,
         "n: i64 = a;\n    c: i64 = 0;\n    while n > 0 {\n        c = c + n % 2;\n        n = n / 2;\n    }\n    return c;",
         (0, 1023)),
        ("power", 2, lambda a, b: power(a, b),
         "r: i64 = 1;\n    i: i64 = 0;\n    while i < b {\n        r = r * a;\n        i = i + 1;\n    }\n    return r;",
         (0, 10)),
        ("collatz_steps", 1, lambda a: collatz_steps(a),
         "n: i64 = a;\n    s: i64 = 0;\n    while n != 1 {\n        if n % 2 == 0 { n = n / 2; } else { n = 3 * n + 1; }\n        s = s + 1;\n    }\n    return s;",
         (1, 50)),
        ("product_1_to_n", 1, product_1_to_n,
         "acc: i64 = 1;\n    i: i64 = 1;\n    while i <= a {\n        acc = acc * i;\n        i = i + 1;\n    }\n    return acc;",
         (1, 10)),
    ]

    while len(records) < count:
        name, n_args, fn, code_body, (lo, hi) = rng.choice(templates)

        examples = []
        seen = set()
        for _ in range(80):
            if len(examples) >= 8:
                break
            if n_args == 1:
                args = [rng.randint(lo, hi)]
            else:
                args = [rng.randint(lo, hi) for _ in range(n_args)]
            key = tuple(args)
            if key in seen:
                continue
            seen.add(key)
            result = eval_fn(fn, args)
            if result is not None:
                examples.append((args, result))

        if len(examples) < 4:
            continue

        outputs = set(t for _, t in examples)
        if len(outputs) <= 1:
            continue

        param_names = ["a", "b", "c", "d"][:n_args]
        params_sig = ", ".join(f"{n}: i64" for n in param_names)
        code = f"fn f({params_sig}) -> i64 {{\n    {code_body}\n}}\n"

        records.append({
            "io_pairs": [[list(inp), out] for inp, out in examples],
            "n_args": n_args,
            "method": "loop",
            "code": code,
            "name": name,
        })

    return records[:count]


# ─── Random expression generator (for diversity) ─────────────────────────────

def _random_simple_expr(n_args: int, rng: random.Random) -> tuple:
    """Generate a random simple expression and its type classification."""
    ops = ["+", "-", "*"]
    consts = [0, 1, -1, 2, -2, 3, 5, 7, 10, 100]

    def rand_operand():
        if rng.random() < 0.6 and n_args > 0:
            return ("arg", rng.randint(0, n_args - 1))
        else:
            return ("const", rng.choice(consts))

    def operand_str(op):
        if op[0] == "arg":
            return ["a", "b", "c", "d"][op[1]]
        return str(op[1])

    def operand_eval(op, args):
        if op[0] == "arg":
            return args[op[1]]
        return op[1]

    # Decide complexity
    complexity = rng.choice(["simple", "simple", "precomp", "precomp"])

    if complexity == "simple":
        # Single op: o1 OP o2
        op = rng.choice(ops)
        o1 = rand_operand()
        o2 = rand_operand()

        def fn(*args):
            a = operand_eval(o1, args)
            b = operand_eval(o2, args)
            if op == "+": return a + b
            if op == "-": return a - b
            if op == "*": return a * b
            return None

        code = f"return {operand_str(o1)} {op} {operand_str(o2)};"
        return fn, code, "expr"

    else:
        # Two ops: (o1 OP1 o2) OP2 o3
        op1 = rng.choice(ops)
        op2 = rng.choice(ops)
        o1 = rand_operand()
        o2 = rand_operand()
        o3 = rand_operand()

        def fn(*args):
            a = operand_eval(o1, args)
            b = operand_eval(o2, args)
            if op1 == "+": v = a + b
            elif op1 == "-": v = a - b
            elif op1 == "*": v = a * b
            else: return None
            c = operand_eval(o3, args)
            if op2 == "+": return v + c
            if op2 == "-": return v - c
            if op2 == "*": return v * c
            return None

        code = f"v0: i64 = {operand_str(o1)} {op1} {operand_str(o2)};\n    return v0 {op2} {operand_str(o3)};"
        return fn, code, "two_precomp" if rng.random() < 0.5 else "expr"


def _gen_random_expr_problems(rng: random.Random, count: int) -> list[dict]:
    """Generate random expression problems with auto-classified types."""
    records = []

    while len(records) < count:
        n_args = rng.randint(1, 3)
        fn, code_body, ptype = _random_simple_expr(n_args, rng)

        examples = []
        seen = set()
        for _ in range(50):
            if len(examples) >= 8:
                break
            args = [rng.randint(-15, 25) for _ in range(n_args)]
            key = tuple(args)
            if key in seen:
                continue
            seen.add(key)
            result = eval_fn(fn, args)
            if result is not None:
                examples.append((args, result))

        if len(examples) < 4:
            continue

        outputs = set(t for _, t in examples)
        if len(outputs) <= 1:
            continue

        param_names = ["a", "b", "c", "d"][:n_args]
        params_sig = ", ".join(f"{n}: i64" for n in param_names)
        code = f"fn f({params_sig}) -> i64 {{\n    {code_body}\n}}\n"

        records.append({
            "io_pairs": [[list(inp), out] for inp, out in examples],
            "n_args": n_args,
            "method": ptype,
            "code": code,
            "name": f"random_{ptype}_{len(records)}",
        })

    return records[:count]


# ─── Verification via mog_synth binary ────────────────────────────────────────

def verify_with_binary(record: dict) -> Optional[str]:
    """
    Run the mog_synth binary on a problem to verify solvability.
    Returns the method string if solved, None otherwise.
    """
    if not os.path.exists(MOG_SYNTH_BIN):
        return None

    # Build problem JSON
    io_pairs = record["io_pairs"]
    n_args = record["n_args"]
    examples = []
    for inp, out in io_pairs:
        examples.append({"inputs": inp, "expected": out})

    problem = {
        "name": record.get("name", "verify"),
        "examples": examples,
    }

    try:
        result = subprocess.run(
            [MOG_SYNTH_BIN, "--problem-json", json.dumps(problem)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            output = json.loads(result.stdout.strip())
            if output.get("success"):
                return output.get("method", "unknown")
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        pass

    return None


# ─── Data augmentation ────────────────────────────────────────────────────────

def augment_record(record: dict, rng: random.Random) -> dict:
    """Create an augmented copy by shuffling I/O pair order."""
    new_rec = dict(record)
    io_pairs = list(record["io_pairs"])
    rng.shuffle(io_pairs)
    new_rec["io_pairs"] = io_pairs
    new_rec["name"] = record.get("name", "aug") + "_aug"
    return new_rec


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate training data for expr meta-learner program-type classifier"
    )
    ap.add_argument("--out", type=str, default=None,
                    help="Output JSONL file (default: stdout)")
    ap.add_argument("--count", type=int, default=600,
                    help="Total number of records to generate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", action="store_true",
                    help="Verify solvability with mog_synth binary (slow)")
    ap.add_argument("--augment", type=int, default=0,
                    help="Number of augmented copies per record")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    total_target = args.count

    # Distribute records across program types
    # Roughly balanced with slight emphasis on expr/branch since they're most common
    n_expr = total_target * 25 // 100
    n_two_precomp = total_target * 15 // 100
    n_branch = total_target * 20 // 100
    n_chained = total_target * 10 // 100
    n_loop = total_target * 15 // 100
    n_random = total_target - n_expr - n_two_precomp - n_branch - n_chained - n_loop

    print(f"Generating training data (seed={args.seed})...", file=sys.stderr, flush=True)
    print(f"  expr: {n_expr}, two_precomp: {n_two_precomp}, branch: {n_branch}, "
          f"chained_branch: {n_chained}, loop: {n_loop}, random: {n_random}",
          file=sys.stderr, flush=True)

    records = []

    print("  Generating expr problems...", file=sys.stderr, flush=True)
    records.extend(_gen_expr_problems(rng, n_expr))

    print("  Generating two_precomp problems...", file=sys.stderr, flush=True)
    records.extend(_gen_two_precomp_problems(rng, n_two_precomp))

    print("  Generating branch problems...", file=sys.stderr, flush=True)
    records.extend(_gen_branch_problems(rng, n_branch))

    print("  Generating chained_branch problems...", file=sys.stderr, flush=True)
    records.extend(_gen_chained_branch_problems(rng, n_chained))

    print("  Generating loop problems...", file=sys.stderr, flush=True)
    records.extend(_gen_loop_problems(rng, n_loop))

    print("  Generating random expr problems...", file=sys.stderr, flush=True)
    records.extend(_gen_random_expr_problems(rng, n_random))

    # Shuffle all records
    rng.shuffle(records)

    # Optional verification
    if args.verify and os.path.exists(MOG_SYNTH_BIN):
        print(f"  Verifying {len(records)} records with mog_synth binary...",
              file=sys.stderr, flush=True)
        verified = 0
        for i, rec in enumerate(records):
            method = verify_with_binary(rec)
            if method:
                verified += 1
            if args.verbose and (i + 1) % 50 == 0:
                print(f"    verified {i + 1}/{len(records)} ({verified} solvable)",
                      file=sys.stderr, flush=True)
        print(f"  Verification: {verified}/{len(records)} solvable by binary",
              file=sys.stderr, flush=True)

    # Write output
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out = open(args.out, "w")
    else:
        out = sys.stdout

    for rec in records:
        out.write(json.dumps(rec) + "\n")
    out.flush()

    # Augmentation
    if args.augment > 0 and args.out:
        out.close()
        print(f"  Augmenting with {args.augment} copies per record...",
              file=sys.stderr, flush=True)
        with open(args.out) as f:
            orig_records = [json.loads(line) for line in f if line.strip()]
        with open(args.out, "a") as f:
            aug_count = 0
            for rec in orig_records:
                for _ in range(args.augment):
                    aug = augment_record(rec, rng)
                    f.write(json.dumps(aug) + "\n")
                    aug_count += 1
            print(f"  Augmented: {aug_count} additional records",
                  file=sys.stderr, flush=True)
    elif args.out:
        out.close()

    # Summary
    type_counts = {}
    for rec in records:
        m = rec["method"]
        type_counts[m] = type_counts.get(m, 0) + 1

    print(f"\nTotal: {len(records)} records", file=sys.stderr, flush=True)
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
