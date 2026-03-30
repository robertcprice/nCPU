"""Differentiable execution for a practical Mog subset.

This is a first real soft-execution layer for Mog programs. It is intentionally
subset-oriented rather than pretending to support the whole language at once.

Current focus:
- arithmetic / comparisons / casts
- variable bindings and assignments
- function calls
- arrays as Python lists of tensors
- structs as dict-like objects of tensors / concrete values
- if-expressions and many if-statement patterns
- while / for loops with bounded hard iteration over concrete ranges
- Result / Optional and match on those values

Important limitation:
- This is differentiable through numeric computation, not yet through arbitrary
  code-token selection the way the nCPU ISA engine does. That would require a
  much larger soft parser / soft AST layer. This module is the execution core
  that can score concrete parsed Mog programs and support execution-guided
  reranking / reward shaping now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import copy
import math

import torch

from egdc.mog_lang.lexer import lex
from egdc.mog_lang.parser import parse
from egdc.mog_lang import ast_nodes as ast
from egdc.mog_lang.evaluator import Environment, MogStruct, MogResult, MogOptional, MogClosure


TensorLike = torch.Tensor | int | float | bool


@dataclass
class SoftExecResult:
    success: bool
    output: str = ""
    return_value: Optional[torch.Tensor] = None
    error: Optional[str] = None
    metadata: dict[str, Any] | None = None


class SoftReturn(Exception):
    def __init__(self, value: Any):
        self.value = value


class SoftBreak(Exception):
    pass


class SoftContinue(Exception):
    pass


class DifferentiableMogExecutor:
    """Differentiable executor for a benchmark-safe Mog subset."""

    def __init__(
        self,
        device: str | torch.device = "cpu",
        compare_scale: float = 1.0,
        max_loop_iters: int = 256,
    ):
        self.device = torch.device(device)
        self.compare_scale = compare_scale
        self.max_loop_iters = max_loop_iters
        self.output: list[str] = []
        self.fn_defs: dict[str, ast.FnDecl] = {}
        self.struct_defs: dict[str, ast.StructDecl] = {}
        self.global_env = Environment()
        self._register_builtins()

    def _register_builtins(self):
        self.global_env.define("abs", lambda x: torch.abs(self._to_tensor(x)))
        self.global_env.define("sqrt", lambda x: torch.sqrt(torch.clamp(self._to_tensor(x), min=0.0)))
        self.global_env.define("pow", lambda a, b: torch.pow(self._to_tensor(a), self._to_tensor(b)))
        self.global_env.define("min", lambda a, b: torch.minimum(self._to_tensor(a), self._to_tensor(b)))
        self.global_env.define("max", lambda a, b: torch.maximum(self._to_tensor(a), self._to_tensor(b)))
        self.global_env.define("str", lambda x: str(int(self._scalar(x))))
        self.global_env.define("len", lambda x: len(x) if isinstance(x, (list, str, dict)) else 0)
        self.global_env.define("println_i64", self._builtin_println_i64)
        self.global_env.define("print_f64", self._builtin_print_f64)
        self.global_env.define("println", self._builtin_println)
        self.global_env.define("print", self._builtin_print)
        self.global_env.define("print_string", self._builtin_print_string)
        self.global_env.define("PI", torch.tensor(math.pi, dtype=torch.float32, device=self.device))
        self.global_env.define("E", torch.tensor(math.e, dtype=torch.float32, device=self.device))

    def _builtin_println_i64(self, x: Any):
        self.output.append(str(int(round(self._scalar(x)))))
        return torch.tensor(0.0, device=self.device)

    def _builtin_print_f64(self, x: Any):
        self.output.append(f"{float(self._scalar(x)):.7f}")
        return torch.tensor(0.0, device=self.device)

    def _builtin_println(self, *args: Any):
        self.output.append(" ".join(str(a) for a in args))
        return torch.tensor(0.0, device=self.device)

    def _builtin_print(self, x: Any):
        self.output.append(str(int(round(self._scalar(x)))))
        return torch.tensor(0.0, device=self.device)

    def _builtin_print_string(self, x: Any):
        if self.output:
            self.output[-1] += str(x)
        else:
            self.output.append(str(x))
        return torch.tensor(0.0, device=self.device)

    def _to_tensor(self, x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        if isinstance(x, bool):
            return torch.tensor(1.0 if x else 0.0, dtype=torch.float32, device=self.device)
        if isinstance(x, int):
            return torch.tensor(float(x), dtype=torch.float32, device=self.device)
        if isinstance(x, float):
            return torch.tensor(float(x), dtype=torch.float32, device=self.device)
        raise TypeError(f"cannot convert {type(x).__name__} to tensor")

    def _scalar(self, x: Any) -> float:
        if isinstance(x, torch.Tensor):
            return float(x.detach().item())
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, bool):
            return 1.0 if x else 0.0
        raise TypeError(f"cannot scalarize {type(x).__name__}")

    def _prob(self, x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            if x.ndim == 0:
                return torch.clamp(x, 0.0, 1.0)
            raise TypeError("expected scalar probability tensor")
        if isinstance(x, bool):
            return torch.tensor(1.0 if x else 0.0, dtype=torch.float32, device=self.device)
        if isinstance(x, (int, float)):
            return torch.tensor(1.0 if x else 0.0, dtype=torch.float32, device=self.device)
        return torch.tensor(1.0 if x else 0.0, dtype=torch.float32, device=self.device)

    def parse_program(self, code: str) -> ast.Program:
        return parse(lex(code))

    def load_program(self, code: str):
        program = self.parse_program(code)
        self.output = []
        self.fn_defs = {}
        self.struct_defs = {}
        self.global_env = Environment()
        self._register_builtins()
        for decl in program.declarations:
            if isinstance(decl, ast.FnDecl):
                self.fn_defs[decl.name] = decl
                self.global_env.define(decl.name, decl)
            elif isinstance(decl, ast.StructDecl):
                self.struct_defs[decl.name] = decl
                self.global_env.define(decl.name, decl)
        return program

    def run_main(self, code: str) -> SoftExecResult:
        try:
            self.load_program(code)
            if "main" not in self.fn_defs:
                return SoftExecResult(False, error="no main() function")
            ret = self._call_fn(self.fn_defs["main"], [], self.global_env)
            ret_t = self._to_tensor(ret) if not isinstance(ret, torch.Tensor) else ret
            return SoftExecResult(True, output="\n".join(self.output), return_value=ret_t)
        except Exception as e:
            return SoftExecResult(False, output="\n".join(self.output), error=str(e))

    def evaluate_function(self, code: str, fn_name: str, args: list[Any]) -> SoftExecResult:
        try:
            self.load_program(code)
            if fn_name not in self.fn_defs:
                return SoftExecResult(False, error=f"function {fn_name} not found")
            ret = self._call_fn(self.fn_defs[fn_name], args, self.global_env)
            ret_t = self._to_tensor(ret) if not isinstance(ret, torch.Tensor) and isinstance(ret, (int, float, bool)) else ret
            return SoftExecResult(True, output="\n".join(self.output), return_value=ret_t)
        except Exception as e:
            return SoftExecResult(False, output="\n".join(self.output), error=str(e))

    def compute_problem_loss(self, code: str, fn_name: str, test_cases: list[tuple[tuple[Any, ...], float]]) -> torch.Tensor:
        losses = []
        for args, expected in test_cases:
            result = self.evaluate_function(code, fn_name, list(args))
            if not result.success or result.return_value is None:
                losses.append(torch.tensor(100.0, device=self.device))
                continue
            pred = self._to_tensor(result.return_value)
            target = self._to_tensor(expected)
            losses.append((pred - target) ** 2)
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)

    def _call_fn(self, fn: ast.FnDecl, args: list[Any], env: Environment) -> Any:
        local = env.child()
        for i, param in enumerate(fn.params):
            if i < len(args):
                local.define(param.name, args[i])
            elif param.default is not None:
                local.define(param.name, self._eval_expr(param.default, env))
            else:
                local.define(param.name, torch.tensor(0.0, device=self.device))
        try:
            self._exec_block(fn.body, local)
            return torch.tensor(0.0, device=self.device)
        except SoftReturn as r:
            return r.value

    def _call_closure(self, closure: MogClosure, args: list[Any]) -> Any:
        local = closure.env.child()
        for i, param in enumerate(closure.params):
            if i < len(args):
                local.define(param.name, args[i])
            elif param.default is not None:
                local.define(param.name, self._eval_expr(param.default, closure.env))
            else:
                local.define(param.name, torch.tensor(0.0, device=self.device))
        if isinstance(closure.body, ast.Block):
            try:
                self._exec_block(closure.body, local)
                return torch.tensor(0.0, device=self.device)
            except SoftReturn as r:
                return r.value
        return self._eval_expr(closure.body, local)

    def _exec_block(self, block: ast.Block, env: Environment):
        for stmt in block.stmts:
            self._exec_stmt(stmt, env)

    def _exec_stmt(self, stmt: Any, env: Environment):
        if isinstance(stmt, ast.VarDecl):
            env.define(stmt.name, self._eval_expr(stmt.value, env))
            return
        if isinstance(stmt, ast.Assignment):
            value = self._eval_expr(stmt.value, env)
            self._assign(stmt.target, value, env)
            return
        if isinstance(stmt, ast.ReturnStmt):
            value = self._eval_expr(stmt.value, env) if stmt.value is not None else torch.tensor(0.0, device=self.device)
            raise SoftReturn(value)
        if isinstance(stmt, ast.ExprStmt):
            self._eval_expr(stmt.expr, env)
            return
        if isinstance(stmt, ast.IfStmt):
            self._exec_if(stmt, env)
            return
        if isinstance(stmt, ast.WhileStmt):
            self._exec_while(stmt, env)
            return
        if isinstance(stmt, ast.ForToStmt):
            self._exec_for_to(stmt, env)
            return
        if isinstance(stmt, ast.ForInRangeStmt):
            self._exec_for_in_range(stmt, env)
            return
        if isinstance(stmt, ast.ForInStmt):
            self._exec_for_in(stmt, env)
            return
        if isinstance(stmt, ast.BreakStmt):
            raise SoftBreak()
        if isinstance(stmt, ast.ContinueStmt):
            raise SoftContinue()
        if isinstance(stmt, ast.FnDecl):
            self.fn_defs[stmt.name] = stmt
            env.define(stmt.name, stmt)
            return
        raise TypeError(f"unsupported statement {type(stmt).__name__}")

    def _assign(self, target: Any, value: Any, env: Environment):
        if isinstance(target, ast.Ident):
            env.set(target.name, value)
            return
        if isinstance(target, ast.FieldAccess):
            obj = self._eval_expr(target.obj, env)
            if isinstance(obj, MogStruct):
                obj.fields[target.field] = value
                return
            raise TypeError("field assignment on non-struct")
        if isinstance(target, ast.IndexAccess):
            obj = self._eval_expr(target.obj, env)
            idx = int(round(self._scalar(self._eval_expr(target.index, env))))
            obj[idx] = value
            return
        raise TypeError(f"unsupported assignment target {type(target).__name__}")

    def _exec_if(self, stmt: ast.IfStmt, env: Environment):
        cond = self._eval_expr(stmt.condition, env)
        # Use hard branch choice for now, while preserving soft condition values in expressions.
        if self._scalar(cond) > 0.5:
            self._exec_block(stmt.then_block, env.child())
        elif stmt.else_block is not None:
            if isinstance(stmt.else_block, ast.Block):
                self._exec_block(stmt.else_block, env.child())
            else:
                self._exec_if(stmt.else_block, env)

    def _exec_while(self, stmt: ast.WhileStmt, env: Environment):
        steps = 0
        while self._scalar(self._eval_expr(stmt.condition, env)) > 0.5:
            steps += 1
            if steps > self.max_loop_iters:
                raise RuntimeError("while loop exceeded max_loop_iters")
            try:
                self._exec_block(stmt.body, env.child())
            except SoftBreak:
                break
            except SoftContinue:
                continue

    def _exec_for_to(self, stmt: ast.ForToStmt, env: Environment):
        start = int(round(self._scalar(self._eval_expr(stmt.start, env))))
        end = int(round(self._scalar(self._eval_expr(stmt.end, env))))
        local = env.child()
        for i in range(start, end):
            local.define(stmt.var_name, torch.tensor(float(i), device=self.device))
            try:
                self._exec_block(stmt.body, local)
            except SoftBreak:
                break
            except SoftContinue:
                continue

    def _exec_for_in_range(self, stmt: ast.ForInRangeStmt, env: Environment):
        start = int(round(self._scalar(self._eval_expr(stmt.start, env))))
        end = int(round(self._scalar(self._eval_expr(stmt.end, env))))
        local = env.child()
        for i in range(start, end):
            local.define(stmt.var_name, torch.tensor(float(i), device=self.device))
            try:
                self._exec_block(stmt.body, local)
            except SoftBreak:
                break
            except SoftContinue:
                continue

    def _exec_for_in(self, stmt: ast.ForInStmt, env: Environment):
        iterable = self._eval_expr(stmt.iterable, env)
        local = env.child()
        if isinstance(iterable, list):
            for idx, item in enumerate(iterable):
                if stmt.index_name:
                    local.define(stmt.index_name, torch.tensor(float(idx), device=self.device))
                local.define(stmt.var_name, item)
                try:
                    self._exec_block(stmt.body, local)
                except SoftBreak:
                    break
                except SoftContinue:
                    continue
            return
        if isinstance(iterable, dict):
            for k, v in iterable.items():
                if stmt.index_name:
                    local.define(stmt.index_name, k)
                    local.define(stmt.var_name, v)
                else:
                    local.define(stmt.var_name, k)
                try:
                    self._exec_block(stmt.body, local)
                except SoftBreak:
                    break
                except SoftContinue:
                    continue
            return
        raise TypeError(f"cannot iterate over {type(iterable).__name__}")

    def _eval_expr(self, expr: Any, env: Environment) -> Any:
        if isinstance(expr, ast.IntLit):
            return torch.tensor(float(expr.value), dtype=torch.float32, device=self.device)
        if isinstance(expr, ast.FloatLit):
            return torch.tensor(float(expr.value), dtype=torch.float32, device=self.device)
        if isinstance(expr, ast.BoolLit):
            return torch.tensor(1.0 if expr.value else 0.0, dtype=torch.float32, device=self.device)
        if isinstance(expr, ast.StringLit):
            return expr.value
        if isinstance(expr, ast.FStringLit):
            return expr.raw
        if isinstance(expr, ast.NoneLit):
            return MogOptional(False, None)
        if isinstance(expr, ast.Ident):
            return env.get(expr.name)
        if isinstance(expr, ast.UnaryOp):
            val = self._eval_expr(expr.operand, env)
            if expr.op == "-":
                return -self._to_tensor(val)
            if expr.op == "!":
                return 1.0 - self._prob(val)
            raise ValueError(expr.op)
        if isinstance(expr, ast.BinOp):
            return self._eval_binop(expr, env)
        if isinstance(expr, ast.Call):
            return self._eval_call(expr, env)
        if isinstance(expr, ast.FieldAccess):
            obj = self._eval_expr(expr.obj, env)
            if isinstance(obj, MogStruct):
                return obj.fields[expr.field]
            if isinstance(obj, (list, str, dict)) and expr.field == "len":
                return torch.tensor(float(len(obj)), dtype=torch.float32, device=self.device)
            raise TypeError(f"field access on {type(obj).__name__}")
        if isinstance(expr, ast.IndexAccess):
            obj = self._eval_expr(expr.obj, env)
            idx = int(round(self._scalar(self._eval_expr(expr.index, env))))
            return obj[idx]
        if isinstance(expr, ast.ArrayLit):
            return [self._eval_expr(e, env) for e in expr.elements]
        if isinstance(expr, ast.ArrayFill):
            value = self._eval_expr(expr.value, env)
            count = int(round(self._scalar(self._eval_expr(expr.count, env))))
            return [copy.deepcopy(value) for _ in range(count)]
        if isinstance(expr, ast.MapLit):
            return {self._eval_expr(k, env): self._eval_expr(v, env) for k, v in expr.pairs}
        if isinstance(expr, ast.StructConstruct):
            return MogStruct(expr.name, {k: self._eval_expr(v, env) for k, v in expr.fields})
        if isinstance(expr, ast.OkExpr):
            return MogResult(True, self._eval_expr(expr.value, env))
        if isinstance(expr, ast.ErrExpr):
            return MogResult(False, self._eval_expr(expr.value, env))
        if isinstance(expr, ast.SomeExpr):
            return MogOptional(True, self._eval_expr(expr.value, env))
        if isinstance(expr, ast.IfExpr):
            cond = self._prob(self._eval_expr(expr.condition, env))
            then_val = self._eval_expr(expr.then_expr, env)
            else_val = self._eval_expr(expr.else_expr, env)
            if isinstance(then_val, (torch.Tensor, int, float, bool)) and isinstance(else_val, (torch.Tensor, int, float, bool)):
                return cond * self._to_tensor(then_val) + (1.0 - cond) * self._to_tensor(else_val)
            return then_val if self._scalar(cond) > 0.5 else else_val
        if isinstance(expr, ast.MatchExpr):
            subject = self._eval_expr(expr.subject, env)
            for arm in expr.arms:
                child = env.child()
                if self._match_pattern(arm.pattern, subject, child):
                    if isinstance(arm.body, ast.Block):
                        self._exec_block(arm.body, child)
                        return torch.tensor(0.0, device=self.device)
                    return self._eval_expr(arm.body, child)
            raise RuntimeError("non-exhaustive match")
        if isinstance(expr, ast.CastExpr):
            val = self._eval_expr(expr.expr, env)
            if expr.target_type in {"i64", "i32", "int", "u64", "u32"}:
                return torch.round(self._to_tensor(val))
            if expr.target_type in {"f64", "f32", "float", "f16", "bf16"}:
                return self._to_tensor(val)
            return val
        if isinstance(expr, ast.PropagateExpr):
            val = self._eval_expr(expr.expr, env)
            if isinstance(val, MogResult):
                return val.value if val.is_ok else torch.tensor(-1e3, device=self.device)
            if isinstance(val, MogOptional):
                return val.value if val.is_some else torch.tensor(-1e3, device=self.device)
            return val
        if isinstance(expr, ast.ClosureLit):
            return MogClosure(expr.params, expr.body, env, expr.return_type)
        raise TypeError(f"unsupported expression {type(expr).__name__}")

    def _eval_binop(self, expr: ast.BinOp, env: Environment) -> Any:
        left = self._eval_expr(expr.left, env)
        right = self._eval_expr(expr.right, env)
        if expr.op == "+":
            if isinstance(left, str) or isinstance(right, str):
                return str(left) + str(right)
            return self._to_tensor(left) + self._to_tensor(right)
        if expr.op == "-":
            return self._to_tensor(left) - self._to_tensor(right)
        if expr.op == "*":
            return self._to_tensor(left) * self._to_tensor(right)
        if expr.op == "/":
            denom = self._to_tensor(right)
            return self._to_tensor(left) / torch.where(torch.abs(denom) < 1e-6, torch.ones_like(denom), denom)
        if expr.op == "%":
            l = torch.round(self._to_tensor(left))
            r = torch.round(self._to_tensor(right))
            r_safe = torch.where(torch.abs(r) < 1e-6, torch.ones_like(r), r)
            return torch.remainder(l, r_safe)
        if expr.op == "**":
            return torch.pow(self._to_tensor(left), self._to_tensor(right))
        if expr.op == "==":
            diff = self._to_tensor(left) - self._to_tensor(right)
            return torch.exp(-(diff * diff) / (2 * self.compare_scale * self.compare_scale))
        if expr.op == "!=":
            return 1.0 - self._eval_binop(ast.BinOp("==", expr.left, expr.right), env)
        if expr.op == "<":
            return torch.sigmoid((self._to_tensor(right) - self._to_tensor(left)) / self.compare_scale)
        if expr.op == ">":
            return torch.sigmoid((self._to_tensor(left) - self._to_tensor(right)) / self.compare_scale)
        if expr.op == "<=":
            return torch.maximum(
                self._eval_binop(ast.BinOp("<", expr.left, expr.right), env),
                self._eval_binop(ast.BinOp("==", expr.left, expr.right), env),
            )
        if expr.op == ">=":
            return torch.maximum(
                self._eval_binop(ast.BinOp(">", expr.left, expr.right), env),
                self._eval_binop(ast.BinOp("==", expr.left, expr.right), env),
            )
        if expr.op in ("&&", "and"):
            return self._prob(left) * self._prob(right)
        if expr.op in ("||", "or"):
            return 1.0 - ((1.0 - self._prob(left)) * (1.0 - self._prob(right)))
        if expr.op == "&":
            return torch.round(self._to_tensor(left)).to(torch.int64).bitwise_and(torch.round(self._to_tensor(right)).to(torch.int64)).to(torch.float32)
        if expr.op == "|":
            return torch.round(self._to_tensor(left)).to(torch.int64).bitwise_or(torch.round(self._to_tensor(right)).to(torch.int64)).to(torch.float32)
        if expr.op == "^":
            return torch.round(self._to_tensor(left)).to(torch.int64).bitwise_xor(torch.round(self._to_tensor(right)).to(torch.int64)).to(torch.float32)
        if expr.op == "<<":
            return torch.round(self._to_tensor(left)).to(torch.int64).bitwise_left_shift(torch.round(self._to_tensor(right)).to(torch.int64)).to(torch.float32)
        if expr.op == ">>":
            return torch.round(self._to_tensor(left)).to(torch.int64).bitwise_right_shift(torch.round(self._to_tensor(right)).to(torch.int64)).to(torch.float32)
        raise ValueError(f"unknown operator {expr.op}")

    def _eval_call(self, expr: ast.Call, env: Environment) -> Any:
        args = [self._eval_expr(a, env) for a in expr.args]
        if isinstance(expr.func, ast.FieldAccess):
            obj = self._eval_expr(expr.func.obj, env)
            method = expr.func.field
            return self._call_method(obj, method, args, env)
        fn = self._eval_expr(expr.func, env)
        if callable(fn):
            return fn(*args)
        if isinstance(fn, ast.FnDecl):
            return self._call_fn(fn, args, self.global_env)
        if isinstance(fn, MogClosure):
            return self._call_closure(fn, args)
        raise TypeError(f"cannot call {type(fn).__name__}")

    def _call_method(self, obj: Any, method: str, args: list[Any], env: Environment) -> Any:
        if isinstance(obj, list):
            if method == "push":
                obj.append(args[0])
                return torch.tensor(0.0, device=self.device)
            if method == "pop":
                return obj.pop()
            if method == "map":
                fn = args[0]
                return [self._call_closure(fn, [item]) if isinstance(fn, MogClosure) else self._call_fn(fn, [item], self.global_env) for item in obj]
            if method == "sort":
                return sorted(obj, key=lambda x: self._scalar(x))
            raise TypeError(f"unknown array method {method}")
        if isinstance(obj, str):
            if method == "trim": return obj.strip()
            if method == "upper": return obj.upper()
            if method == "lower": return obj.lower()
            if method == "split":
                if args:
                    sep = args[0]
                    if sep == "":
                        return list(obj)
                    return obj.split(sep)
                return obj.split()
            if method == "contains": return torch.tensor(1.0 if args[0] in obj else 0.0, device=self.device)
            if method == "starts_with": return torch.tensor(1.0 if obj.startswith(args[0]) else 0.0, device=self.device)
            if method == "ends_with": return torch.tensor(1.0 if obj.endswith(args[0]) else 0.0, device=self.device)
            if method == "replace": return obj.replace(args[0], args[1])
            raise TypeError(f"unknown string method {method}")
        if isinstance(obj, dict):
            if method == "has": return torch.tensor(1.0 if args[0] in obj else 0.0, device=self.device)
            if method == "keys": return list(obj.keys())
            if method == "values": return list(obj.values())
            raise TypeError(f"unknown map method {method}")
        raise TypeError(f"cannot call {method} on {type(obj).__name__}")

    def _match_pattern(self, pattern: Any, value: Any, env: Environment) -> bool:
        if isinstance(pattern, ast.WildcardPattern):
            return True
        if isinstance(pattern, ast.LitPattern):
            lit = pattern.value
            if isinstance(lit, ast.IntLit):
                return abs(self._scalar(value) - lit.value) < 1e-6
            if isinstance(lit, ast.FloatLit):
                return abs(self._scalar(value) - lit.value) < 1e-6
            if isinstance(lit, ast.StringLit):
                return value == lit.value
            if isinstance(lit, ast.BoolLit):
                return (self._scalar(value) > 0.5) == lit.value
            return False
        if isinstance(pattern, ast.OkPattern):
            if isinstance(value, MogResult) and value.is_ok:
                env.define(pattern.binding, value.value)
                return True
            return False
        if isinstance(pattern, ast.ErrPattern):
            if isinstance(value, MogResult) and not value.is_ok:
                env.define(pattern.binding, value.value)
                return True
            return False
        if isinstance(pattern, ast.SomePattern):
            if isinstance(value, MogOptional) and value.is_some:
                env.define(pattern.binding, value.value)
                return True
            return False
        if isinstance(pattern, ast.NonePattern):
            return isinstance(value, MogOptional) and not value.is_some
        if isinstance(pattern, ast.IdentPattern):
            env.define(pattern.name, value)
            return True
        return False


def make_numeric_testcases(problem) -> list[tuple[tuple[Any, ...], float]]:
    out = []
    for args, expected in problem.test_cases:
        try:
            expected_val = float(expected)
        except ValueError:
            continue
        out.append((args, expected_val))
    return out


if __name__ == "__main__":
    demo = """
fn factorial(n: i64) -> i64 {
    if n <= 1 {
        return 1;
    }
    return n * factorial(n - 1);
}

fn main() -> i64 {
    println_i64(factorial(6));
    return 0;
}
"""
    ex = DifferentiableMogExecutor()
    res = ex.run_main(demo)
    print("success:", res.success)
    print("output:", res.output)
