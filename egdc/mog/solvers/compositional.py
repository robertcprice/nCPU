"""Compositional program synthesis for Mog.

This is the next level beyond single-function discovery: building new programs
by composing already-discovered sub-programs.

Key idea: once the system discovers gcd, it can use gcd as a building block
to discover lcm, coprime_check, simplify_fraction, etc. The library of
capabilities grows from use.

Search hierarchy for composition:
1. Direct expression search (maybe the answer is just arithmetic)
2. Loop + sub-program call search (for i in range: call sub-program)
3. Expression involving sub-program (e.g., (a*b) / gcd(a,b))
4. Filter + reduce patterns using sub-programs
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from egdc.mog.solvers.program_search import (
    _py_eval_expr, _py_eval_cmp,
    _eval_code_on_examples,
    SearchResult,
)


@dataclass
class SubProgram:
    name: str
    arg_names: list[str]
    code: str
    # Python-callable reference for fast search evaluation
    py_fn: Any = None


@dataclass
class CompositionResult:
    success: bool
    code: str
    loss: float
    method: str
    metadata: dict[str, Any]


class CompositionalSolver:
    def __init__(self):
        self.library: dict[str, SubProgram] = {}
        self.solved_codes: dict[str, str] = {}  # name -> code, for auto-extraction

    def auto_extract_and_register(self):
        """Detect shared code across solved programs and register as sub-programs."""
        from egdc.mog.tools.auto_extract import SubProgramExtractor
        ext = SubProgramExtractor()
        frags = ext.find_shared_fragments(self.solved_codes)
        for i, frag in enumerate(frags[:5]):
            name = f"_extracted_{i}"
            if name not in self.library:
                self.library[name] = SubProgram(name, [], frag.code)

    def register_subprogram(self, name: str, arg_names: list[str], code: str, py_fn=None):
        """Register a discovered sub-program as a reusable building block."""
        if py_fn is None:
            py_fn = self._make_py_fn(name, arg_names, code)
        self.library[name] = SubProgram(name, arg_names, code, py_fn)

    def _make_py_fn(self, name: str, arg_names: list[str], code: str):
        """Create a Python callable from Mog code by using the interpreter."""
        from egdc.mog.lang import interpret

        def py_fn(*args):
            arg_strs = ", ".join(str(int(a)) for a in args)
            test = code + f"\nfn main() -> i64 {{ println_i64({name}({arg_strs})); return 0; }}"
            result = interpret(test)
            if result.success and result.output.strip():
                return float(result.output.strip().split("\n")[0])
            return None
        return py_fn

    def solve(self, fn_name: str, arg_names: list[str], examples,
              **kwargs) -> CompositionResult:
        """Solve by trying: direct search, then compositional search."""

        params = ", ".join(f"{a}: i64" for a in arg_names)

        # --- Phase 1: Direct expression with sub-program calls ---
        code, loss = self._expression_with_calls(fn_name, arg_names, examples)
        if loss < 1e-6:
            return CompositionResult(True, code, loss, "expression_composition", {})

        # --- Phase 2: Loop calling sub-program ---
        code, loss = self._loop_with_calls(fn_name, arg_names, examples)
        if loss < 1e-6:
            return CompositionResult(True, code, loss, "loop_composition", {})

        # --- Phase 3: Loop with body expression (no sub-program needed) ---
        code, loss = self._enriched_loop_search(fn_name, arg_names, examples)
        if loss < 1e-6:
            return CompositionResult(True, code, loss, "enriched_loop", {})

        # Return best found
        return CompositionResult(False, "", float("inf"), "failed", {})

    def _expression_with_calls(self, fn_name: str, arg_names: list[str],
                                examples) -> tuple[str, float]:
        """Search for expressions that combine args with sub-program calls.

        Candidate expressions:
        - (arg1 OP arg2) / sub(arg1, arg2)
        - sub(arg1, arg2) OP arg1
        - etc.
        """
        params = ", ".join(f"{a}: i64" for a in arg_names)

        # Build candidate atomic values
        atoms: list[tuple[str, Any]] = []  # (mog_expr, python_eval_fn)
        for a in arg_names:
            atoms.append((a, lambda env, n=a: env.get(n, 0)))
        for c in [0, 1, -1, 2]:
            atoms.append((str(c), lambda env, c=c: float(c)))

        # Add sub-program calls
        for sub in self.library.values():
            if len(sub.arg_names) == len(arg_names):
                call_args = ", ".join(arg_names)
                mog_call = f"{sub.name}({call_args})"
                atoms.append((mog_call, lambda env, s=sub: s.py_fn(*[env[a] for a in arg_names])))
            if len(sub.arg_names) == 1:
                for a in arg_names:
                    mog_call = f"{sub.name}({a})"
                    atoms.append((mog_call, lambda env, s=sub, n=a: s.py_fn(env[n])))

        best_loss = float("inf")
        best_code = ""

        # Try atom OP atom
        for mog1, fn1 in atoms:
            for mog2, fn2 in atoms:
                for op, py_op in [("+", lambda a, b: a + b), ("-", lambda a, b: a - b),
                                   ("*", lambda a, b: a * b),
                                   ("/", lambda a, b: int(a / b) if b != 0 else 99999)]:
                    loss = 0.0
                    for args, target in examples:
                        env = {n: float(v) for n, v in zip(arg_names, args)}
                        v1 = fn1(env)
                        v2 = fn2(env)
                        if v1 is None or v2 is None:
                            loss = float("inf")
                            break
                        pred = py_op(v1, v2)
                        loss += (pred - target) ** 2
                    loss /= max(len(examples), 1)
                    if loss < best_loss:
                        best_loss = loss
                        # Build code: include sub-program definitions + new function
                        sub_defs = "\n".join(s.code for s in self.library.values()
                                             if s.name in mog1 or s.name in mog2)
                        if sub_defs:
                            sub_defs += "\n\n"
                        best_code = (
                            f"{sub_defs}"
                            f"fn {fn_name}({params}) -> i64 {{\n"
                            f"    return ({mog1}) {op} ({mog2});\n"
                            f"}}\n"
                        )
                        if loss < 1e-6:
                            return best_code, best_loss

        # Try compound: (arg OP arg) OP atom — catches (a*b)/gcd(a,b)
        for a1 in arg_names:
            for a2 in arg_names:
                for op1, py_op1 in [("+", lambda a, b: a + b), ("-", lambda a, b: a - b),
                                     ("*", lambda a, b: a * b)]:
                    for mog_r, fn_r in atoms:
                        for op2, py_op2 in [("+", lambda a, b: a + b), ("-", lambda a, b: a - b),
                                             ("*", lambda a, b: a * b),
                                             ("/", lambda a, b: int(a / b) if b != 0 else 99999)]:
                            loss = 0.0
                            for args_vals, target in examples:
                                env = {n: float(v) for n, v in zip(arg_names, args_vals)}
                                v_left = py_op1(env[a1], env[a2])
                                v_right = fn_r(env)
                                if v_right is None:
                                    loss = float("inf")
                                    break
                                pred = py_op2(v_left, v_right)
                                loss += (pred - target) ** 2
                            loss /= max(len(examples), 1)
                            if loss < best_loss:
                                best_loss = loss
                                sub_defs = "\n".join(s.code for s in self.library.values()
                                                     if s.name in mog_r)
                                if sub_defs:
                                    sub_defs += "\n\n"
                                best_code = (
                                    f"{sub_defs}"
                                    f"fn {fn_name}({params}) -> i64 {{\n"
                                    f"    return ({a1} {op1} {a2}) {op2} ({mog_r});\n"
                                    f"}}\n"
                                )
                                if loss < 1e-6:
                                    return best_code, best_loss

        # Also try just returning a single atom
        for mog_expr, fn_eval in atoms:
            loss = 0.0
            for args, target in examples:
                env = {n: float(v) for n, v in zip(arg_names, args)}
                v = fn_eval(env)
                if v is None:
                    loss = float("inf")
                    break
                loss += (v - target) ** 2
            loss /= max(len(examples), 1)
            if loss < best_loss:
                best_loss = loss
                sub_defs = "\n".join(s.code for s in self.library.values()
                                     if s.name in mog_expr)
                if sub_defs:
                    sub_defs += "\n\n"
                best_code = f"{sub_defs}fn {fn_name}({params}) -> i64 {{\n    return {mog_expr};\n}}\n"
                if loss < 1e-6:
                    return best_code, best_loss

        return best_code, best_loss

    def _loop_with_calls(self, fn_name: str, arg_names: list[str],
                          examples) -> tuple[str, float]:
        """Search for loop programs that call sub-programs in the body.

        Pattern: acc = init; for i := start to bound { acc = acc + sub(i); }
        """
        params = ", ".join(f"{a}: i64" for a in arg_names)
        best_loss = float("inf")
        best_code = ""

        bounds = list(arg_names) + [f"{a} + 1" for a in arg_names]

        for sub in self.library.values():
            if len(sub.arg_names) != 1:
                continue
            for init in [0, 1]:
                for start in [0, 1, 2]:
                    for bound in bounds:
                        for body_op in ["+", "*"]:
                            loss = 0.0
                            for args, target in examples:
                                env = {n: float(v) for n, v in zip(arg_names, args)}
                                acc = float(init)
                                s = int(_py_eval_expr(str(start), env))
                                b = int(_py_eval_expr(bound, env))
                                for i in range(s, max(s, b)):
                                    sub_val = sub.py_fn(float(i))
                                    if sub_val is None:
                                        loss = float("inf")
                                        break
                                    if body_op == "+":
                                        acc = acc + sub_val
                                    elif body_op == "*":
                                        acc = acc * sub_val
                                if not math.isfinite(loss):
                                    break
                                diff = acc - target
                                if abs(diff) > 1e12:
                                    loss = float("inf")
                                    break
                                loss += diff ** 2
                            loss /= max(len(examples), 1)
                            if loss < best_loss:
                                best_loss = loss
                                sub_defs = sub.code + "\n\n"
                                best_code = (
                                    f"{sub_defs}"
                                    f"fn {fn_name}({params}) -> i64 {{\n"
                                    f"    acc: i64 = {init};\n"
                                    f"    i: i64 = {start};\n"
                                    f"    while i < ({bound}) {{\n"
                                    f"        acc = acc {body_op} {sub.name}(i);\n"
                                    f"        i = i + 1;\n"
                                    f"    }}\n"
                                    f"    return acc;\n"
                                    f"}}\n"
                                )
                                if loss < 1e-6:
                                    return best_code, best_loss
        return best_code, best_loss

    def _enriched_loop_search(self, fn_name: str, arg_names: list[str],
                               examples) -> tuple[str, float]:
        """Loop with richer body expressions: acc = acc + i*i, acc + i*2, etc."""
        params = ", ".join(f"{a}: i64" for a in arg_names)
        best_loss = float("inf")
        best_code = ""

        bounds = list(arg_names) + [f"{a} + 1" for a in arg_names]
        body_exprs = [
            ("i * i", lambda i: i * i),
            ("i * 2", lambda i: i * 2),
            ("i * i * i", lambda i: i * i * i),
            ("i", lambda i: i),
            ("1", lambda i: 1),
        ]

        for init in [0, 1]:
            for start in [0, 1, 2]:
                for bound in bounds:
                    for body_op in ["+", "*"]:
                        for expr_str, expr_fn in body_exprs:
                            loss = 0.0
                            for args, target in examples:
                                env = {n: float(v) for n, v in zip(arg_names, args)}
                                acc = float(init)
                                s = int(_py_eval_expr(str(start), env))
                                b = int(_py_eval_expr(bound, env))
                                for i in range(s, max(s, b)):
                                    val = expr_fn(float(i))
                                    if body_op == "+":
                                        acc = acc + val
                                    elif body_op == "*":
                                        if abs(acc) > 1e12:
                                            loss = float("inf")
                                            break
                                        acc = acc * val
                                if not math.isfinite(loss):
                                    break
                                diff = acc - target
                                if abs(diff) > 1e12:
                                    loss = float("inf")
                                    break
                                loss += diff ** 2
                            loss /= max(len(examples), 1)
                            if loss < best_loss:
                                best_loss = loss
                                best_code = (
                                    f"fn {fn_name}({params}) -> i64 {{\n"
                                    f"    acc: i64 = {init};\n"
                                    f"    i: i64 = {start};\n"
                                    f"    while i < ({bound}) {{\n"
                                    f"        acc = acc {body_op} ({expr_str});\n"
                                    f"        i = i + 1;\n"
                                    f"    }}\n"
                                    f"    return acc;\n"
                                    f"}}\n"
                                )
                                if loss < 1e-6:
                                    return best_code, best_loss
        return best_code, best_loss
