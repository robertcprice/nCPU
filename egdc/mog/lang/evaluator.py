"""Mog tree-walk interpreter (evaluator)."""
from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from . import ast_nodes as ast


# --- Runtime values ---

class MogError(Exception):
    pass

class ReturnSignal(Exception):
    def __init__(self, value: Any):
        self.value = value

class BreakSignal(Exception):
    pass

class ContinueSignal(Exception):
    pass

class PropagateSignal(Exception):
    """Raised by ? operator when Result is err or Optional is none."""
    def __init__(self, value: Any):
        self.value = value

@dataclass
class MogResult:
    is_ok: bool
    value: Any

@dataclass
class MogOptional:
    is_some: bool
    value: Any

@dataclass
class MogStruct:
    name: str
    fields: dict[str, Any]

@dataclass
class MogClosure:
    params: list[ast.Param]
    body: Any  # Block or Expr
    env: Environment
    return_type: Optional[str] = None


# --- Environment ---

class Environment:
    def __init__(self, parent: Optional[Environment] = None):
        self.bindings: dict[str, Any] = {}
        self.parent = parent

    def get(self, name: str) -> Any:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.get(name)
        raise MogError(f"undefined variable '{name}'")

    def set(self, name: str, value: Any):
        """Update existing binding (walks up scope chain)."""
        if name in self.bindings:
            self.bindings[name] = value
            return
        if self.parent:
            self.parent.set(name, value)
            return
        raise MogError(f"undefined variable '{name}' for assignment")

    def define(self, name: str, value: Any):
        """Create new binding in current scope."""
        self.bindings[name] = value

    def child(self) -> Environment:
        return Environment(parent=self)


# --- Evaluator ---

class Evaluator:
    MAX_RECURSION = 500
    MAX_LOOP_ITERS = 100_000

    def __init__(self):
        self.output: list[str] = []
        self.call_depth = 0
        self.global_env = Environment()
        self.struct_defs: dict[str, ast.StructDecl] = {}
        self.fn_defs: dict[str, ast.FnDecl] = {}
        self.input_queue: list[str] = []
        self._register_builtins()

    def _register_builtins(self):
        env = self.global_env
        # Print functions
        env.define("println", lambda *a: self._builtin_println(*a))
        env.define("println_i64", lambda x: self._builtin_println_i64(x))
        env.define("print_f64", lambda x: self._builtin_print_f64(x))
        env.define("print", lambda x: self._builtin_print(x))
        env.define("print_string", lambda x: self._builtin_print_string(x))
        # Conversion
        env.define("str", lambda x: str(int(x)) if isinstance(x, (int, float)) else str(x))
        env.define("len", lambda x: len(x) if isinstance(x, (str, list, dict)) else (x.fields.get("len", 0) if isinstance(x, MogStruct) else 0))
        env.define("int_from_string", lambda s: MogResult(True, int(s)) if s.lstrip("-").isdigit() else MogResult(False, f"cannot parse '{s}' as int"))
        env.define("parse_float", lambda s: self._builtin_parse_float(s))
        # Math
        env.define("abs", lambda x: abs(x))
        env.define("sqrt", lambda x: math.sqrt(x))
        env.define("pow", lambda a, b: a ** b)
        env.define("sin", lambda x: math.sin(x))
        env.define("cos", lambda x: math.cos(x))
        env.define("tan", lambda x: math.tan(x))
        env.define("asin", lambda x: math.asin(x))
        env.define("acos", lambda x: math.acos(x))
        env.define("atan2", lambda y, x: math.atan2(y, x))
        env.define("exp", lambda x: math.exp(x))
        env.define("log", lambda x: math.log(x))
        env.define("log2", lambda x: math.log2(x))
        env.define("floor", lambda x: int(math.floor(x)))
        env.define("ceil", lambda x: int(math.ceil(x)))
        env.define("round", lambda x: int(round(x)))
        env.define("min", lambda a, b: min(a, b))
        env.define("max", lambda a, b: max(a, b))
        # I/O — read functions pull from input_queue (set externally)
        env.define("read_i64", lambda: self._builtin_read_i64())
        env.define("read_string", lambda: self._builtin_read_string())
        env.define("read_line", lambda: self._builtin_read_string())
        env.define("has_input", lambda: self._builtin_has_input())
        # Constants
        env.define("PI", math.pi)
        env.define("E", math.e)

    def _builtin_println(self, *args):
        text = " ".join(str(a) for a in args)
        self.output.append(text)

    def _builtin_println_i64(self, x):
        self.output.append(str(int(x)))

    def _builtin_print_f64(self, x):
        # Match mogc format: 7 decimal places with trailing zeros
        self.output.append(f"{float(x):.7f}")

    def _builtin_print(self, x):
        self.output.append(str(int(x)))

    def _builtin_print_string(self, x):
        # print_string doesn't add newline, but we track lines
        if self.output:
            self.output[-1] += str(x)
        else:
            self.output.append(str(x))

    def _builtin_read_i64(self):
        if self.input_queue:
            val = self.input_queue.pop(0)
            try:
                return int(val)
            except ValueError:
                raise RuntimeError(f"read_i64: cannot parse '{val}' as integer")
        raise RuntimeError("read_i64: no input available")

    def _builtin_read_string(self):
        if self.input_queue:
            return self.input_queue.pop(0)
        raise RuntimeError("read_string: no input available")

    def _builtin_has_input(self):
        return 1 if self.input_queue else 0

    def _builtin_parse_float(self, s):
        try:
            return MogResult(True, float(s))
        except ValueError:
            return MogResult(False, f"cannot parse '{s}' as float")

    # --- Main entry ---

    def run(self, program: ast.Program) -> Any:
        """Execute a Mog program. Returns the main() return value."""
        # First pass: collect all fn and struct declarations
        for decl in program.declarations:
            if isinstance(decl, ast.FnDecl):
                self.fn_defs[decl.name] = decl
                self.global_env.define(decl.name, decl)
            elif isinstance(decl, ast.StructDecl):
                self.struct_defs[decl.name] = decl
                self.global_env.define(decl.name, decl)
            elif isinstance(decl, ast.RequiresDecl):
                pass  # Skip capabilities
            elif isinstance(decl, ast.TypeAlias):
                pass  # Skip type aliases

        # Run main()
        if "main" not in self.fn_defs:
            raise MogError("no main() function defined")
        return self._call_fn(self.fn_defs["main"], [], self.global_env)

    def _call_fn(self, fn: ast.FnDecl, args: list[Any], env: Environment) -> Any:
        self.call_depth += 1
        if self.call_depth > self.MAX_RECURSION:
            raise MogError(f"maximum recursion depth ({self.MAX_RECURSION}) exceeded")
        try:
            local = env.child()
            for i, param in enumerate(fn.params):
                if i < len(args):
                    local.define(param.name, args[i])
                elif param.default is not None:
                    local.define(param.name, self._eval_expr(param.default, env))
                else:
                    local.define(param.name, 0)
            try:
                self._exec_block(fn.body, local)
                return 0  # implicit return
            except ReturnSignal as r:
                return r.value
        finally:
            self.call_depth -= 1

    def _call_closure(self, closure: MogClosure, args: list[Any]) -> Any:
        self.call_depth += 1
        if self.call_depth > self.MAX_RECURSION:
            raise MogError(f"maximum recursion depth ({self.MAX_RECURSION}) exceeded")
        try:
            local = closure.env.child()
            for i, param in enumerate(closure.params):
                if i < len(args):
                    local.define(param.name, args[i])
                elif param.default is not None:
                    local.define(param.name, self._eval_expr(param.default, closure.env))
                else:
                    local.define(param.name, 0)
            if isinstance(closure.body, ast.Block):
                try:
                    self._exec_block(closure.body, local)
                    # If block has a single expression statement with no semicolon,
                    # treat it as an implicit return (closure shorthand)
                    if len(closure.body.stmts) == 1 and isinstance(closure.body.stmts[0], ast.ExprStmt):
                        return self._eval_expr(closure.body.stmts[0].expr, local)
                    return 0
                except ReturnSignal as r:
                    return r.value
            else:
                return self._eval_expr(closure.body, local)
        finally:
            self.call_depth -= 1

    # --- Statement execution ---

    def _exec_block(self, block: ast.Block, env: Environment):
        for stmt in block.stmts:
            self._exec_stmt(stmt, env)

    def _exec_stmt(self, stmt: Any, env: Environment):
        if isinstance(stmt, ast.VarDecl):
            val = self._eval_expr(stmt.value, env)
            env.define(stmt.name, val)
        elif isinstance(stmt, ast.Assignment):
            val = self._eval_expr(stmt.value, env)
            self._assign(stmt.target, val, env)
        elif isinstance(stmt, ast.ReturnStmt):
            val = self._eval_expr(stmt.value, env) if stmt.value else None
            raise ReturnSignal(val)
        elif isinstance(stmt, ast.IfStmt):
            self._exec_if(stmt, env)
        elif isinstance(stmt, ast.WhileStmt):
            self._exec_while(stmt, env)
        elif isinstance(stmt, ast.ForToStmt):
            self._exec_for_to(stmt, env)
        elif isinstance(stmt, ast.ForInRangeStmt):
            self._exec_for_in_range(stmt, env)
        elif isinstance(stmt, ast.ForInStmt):
            self._exec_for_in(stmt, env)
        elif isinstance(stmt, ast.BreakStmt):
            raise BreakSignal()
        elif isinstance(stmt, ast.ContinueStmt):
            raise ContinueSignal()
        elif isinstance(stmt, ast.ExprStmt):
            self._eval_expr(stmt.expr, env)
        elif isinstance(stmt, ast.FnDecl):
            env.define(stmt.name, stmt)
            self.fn_defs[stmt.name] = stmt
        elif isinstance(stmt, ast.StructDecl):
            self.struct_defs[stmt.name] = stmt
            env.define(stmt.name, stmt)
        elif isinstance(stmt, ast.Block):
            self._exec_block(stmt, env.child())
        else:
            raise MogError(f"unknown statement type: {type(stmt).__name__}")

    def _assign(self, target: Any, value: Any, env: Environment):
        if isinstance(target, ast.Ident):
            env.set(target.name, value)
        elif isinstance(target, ast.FieldAccess):
            obj = self._eval_expr(target.obj, env)
            if isinstance(obj, MogStruct):
                obj.fields[target.field] = value
            else:
                raise MogError(f"cannot set field on {type(obj)}")
        elif isinstance(target, ast.IndexAccess):
            obj = self._eval_expr(target.obj, env)
            idx = self._eval_expr(target.index, env)
            if isinstance(obj, list):
                obj[int(idx)] = value
            elif isinstance(obj, dict):
                obj[idx] = value
            else:
                raise MogError(f"cannot index-assign {type(obj)}")
        else:
            raise MogError(f"invalid assignment target: {type(target).__name__}")

    def _exec_if(self, stmt: ast.IfStmt, env: Environment):
        cond = self._eval_expr(stmt.condition, env)
        if self._truthy(cond):
            self._exec_block(stmt.then_block, env.child())
        elif stmt.else_block is not None:
            if isinstance(stmt.else_block, ast.IfStmt):
                self._exec_if(stmt.else_block, env)
            elif isinstance(stmt.else_block, ast.Block):
                self._exec_block(stmt.else_block, env.child())

    def _exec_while(self, stmt: ast.WhileStmt, env: Environment):
        iters = 0
        while self._truthy(self._eval_expr(stmt.condition, env)):
            iters += 1
            if iters > self.MAX_LOOP_ITERS:
                raise MogError(f"loop exceeded {self.MAX_LOOP_ITERS} iterations")
            try:
                self._exec_block(stmt.body, env.child())
            except BreakSignal:
                break
            except ContinueSignal:
                continue

    def _exec_for_to(self, stmt: ast.ForToStmt, env: Environment):
        start = int(self._eval_expr(stmt.start, env))
        end = int(self._eval_expr(stmt.end, env))
        local = env.child()
        for i in range(start, end):
            local.define(stmt.var_name, i)
            try:
                self._exec_block(stmt.body, local)
            except BreakSignal:
                break
            except ContinueSignal:
                continue

    def _exec_for_in_range(self, stmt: ast.ForInRangeStmt, env: Environment):
        start = int(self._eval_expr(stmt.start, env))
        end = int(self._eval_expr(stmt.end, env))
        local = env.child()
        for i in range(start, end):
            local.define(stmt.var_name, i)
            try:
                self._exec_block(stmt.body, local)
            except BreakSignal:
                break
            except ContinueSignal:
                continue

    def _exec_for_in(self, stmt: ast.ForInStmt, env: Environment):
        iterable = self._eval_expr(stmt.iterable, env)
        local = env.child()
        if isinstance(iterable, list):
            for i, item in enumerate(iterable):
                if stmt.index_name:
                    local.define(stmt.index_name, i)
                local.define(stmt.var_name, item)
                try:
                    self._exec_block(stmt.body, local)
                except BreakSignal:
                    break
                except ContinueSignal:
                    continue
        elif isinstance(iterable, dict):
            for k, v in iterable.items():
                if stmt.index_name:
                    local.define(stmt.index_name, k)
                    local.define(stmt.var_name, v)
                else:
                    local.define(stmt.var_name, k)
                try:
                    self._exec_block(stmt.body, local)
                except BreakSignal:
                    break
                except ContinueSignal:
                    continue
        else:
            raise MogError(f"cannot iterate over {type(iterable)}")

    # --- Expression evaluation ---

    def _eval_expr(self, expr: Any, env: Environment) -> Any:
        if isinstance(expr, ast.IntLit):
            return expr.value
        if isinstance(expr, ast.FloatLit):
            return expr.value
        if isinstance(expr, ast.StringLit):
            return expr.value
        if isinstance(expr, ast.FStringLit):
            return self._eval_fstring(expr.raw, env)
        if isinstance(expr, ast.BoolLit):
            return expr.value
        if isinstance(expr, ast.NoneLit):
            return MogOptional(False, None)
        if isinstance(expr, ast.Ident):
            return env.get(expr.name)
        if isinstance(expr, ast.BinOp):
            return self._eval_binop(expr, env)
        if isinstance(expr, ast.UnaryOp):
            return self._eval_unary(expr, env)
        if isinstance(expr, ast.Call):
            return self._eval_call(expr, env)
        if isinstance(expr, ast.FieldAccess):
            return self._eval_field_access(expr, env)
        if isinstance(expr, ast.IndexAccess):
            return self._eval_index_access(expr, env)
        if isinstance(expr, ast.ArrayLit):
            return [self._eval_expr(e, env) for e in expr.elements]
        if isinstance(expr, ast.ArrayFill):
            val = self._eval_expr(expr.value, env)
            count = int(self._eval_expr(expr.count, env))
            return [val] * count
        if isinstance(expr, ast.MapLit):
            return {self._eval_expr(k, env): self._eval_expr(v, env) for k, v in expr.pairs}
        if isinstance(expr, ast.StructConstruct):
            return self._eval_struct_construct(expr, env)
        if isinstance(expr, ast.OkExpr):
            return MogResult(True, self._eval_expr(expr.value, env))
        if isinstance(expr, ast.ErrExpr):
            return MogResult(False, self._eval_expr(expr.value, env))
        if isinstance(expr, ast.SomeExpr):
            return MogOptional(True, self._eval_expr(expr.value, env))
        if isinstance(expr, ast.IfExpr):
            cond = self._eval_expr(expr.condition, env)
            return self._eval_expr(expr.then_expr if self._truthy(cond) else expr.else_expr, env)
        if isinstance(expr, ast.MatchExpr):
            return self._eval_match(expr, env)
        if isinstance(expr, ast.CastExpr):
            return self._eval_cast(expr, env)
        if isinstance(expr, ast.PropagateExpr):
            return self._eval_propagate(expr, env)
        if isinstance(expr, ast.ClosureLit):
            return MogClosure(expr.params, expr.body, env, expr.return_type)
        if isinstance(expr, ast.RangeExpr):
            return (self._eval_expr(expr.start, env), self._eval_expr(expr.end, env))
        raise MogError(f"unknown expression type: {type(expr).__name__}")

    def _eval_fstring(self, template: str, env: Environment) -> str:
        """Evaluate an f-string template."""
        from .lexer import lex
        from .parser import parse as parse_prog

        result = []
        i = 0
        while i < len(template):
            if template[i] == "{" and i + 1 < len(template) and template[i + 1] == "{":
                result.append("{")
                i += 2
            elif template[i] == "{":
                # Find matching }
                depth = 1
                j = i + 1
                while j < len(template) and depth > 0:
                    if template[j] == "{":
                        depth += 1
                    elif template[j] == "}":
                        depth -= 1
                    j += 1
                expr_str = template[i + 1:j - 1]
                # Parse and evaluate the expression
                tokens = lex(expr_str + ";")
                # Quick hack: wrap in a dummy to parse as expression
                # Just parse the expression directly
                from .parser import Parser
                p = Parser(tokens)
                val = self._eval_expr(p._parse_expr(), env)
                if isinstance(val, float):
                    if val == int(val):
                        result.append(str(int(val)))
                    else:
                        result.append(str(val))
                elif isinstance(val, bool):
                    result.append("true" if val else "false")
                else:
                    result.append(str(val))
                i = j
            else:
                result.append(template[i])
                i += 1
        return "".join(result)

    def _eval_binop(self, expr: ast.BinOp, env: Environment) -> Any:
        left = self._eval_expr(expr.left, env)
        # Short-circuit for logical ops
        if expr.op in ("&&", "and"):
            if not self._truthy(left):
                return left
            return self._eval_expr(expr.right, env)
        if expr.op in ("||", "or"):
            if self._truthy(left):
                return left
            return self._eval_expr(expr.right, env)

        right = self._eval_expr(expr.right, env)

        # String concatenation
        if expr.op == "+" and isinstance(left, str):
            return left + str(right)

        # Range
        if expr.op == "..":
            return range(int(left), int(right))

        # Arithmetic
        if expr.op == "+": return left + right
        if expr.op == "-": return left - right
        if expr.op == "*": return left * right
        if expr.op == "/":
            if isinstance(left, int) and isinstance(right, int):
                if right == 0:
                    raise MogError("division by zero")
                return left // right
            return left / right
        if expr.op == "%": return left % right
        if expr.op == "**": return left ** right

        # Comparison
        if expr.op == "==": return left == right
        if expr.op == "!=": return left != right
        if expr.op == "<": return left < right
        if expr.op == ">": return left > right
        if expr.op == "<=": return left <= right
        if expr.op == ">=": return left >= right

        # Bitwise
        if expr.op == "&": return int(left) & int(right)
        if expr.op == "|": return int(left) | int(right)
        if expr.op == "^": return int(left) ^ int(right)
        if expr.op == "<<": return int(left) << int(right)
        if expr.op == ">>": return int(left) >> int(right)

        raise MogError(f"unknown operator '{expr.op}'")

    def _eval_unary(self, expr: ast.UnaryOp, env: Environment) -> Any:
        val = self._eval_expr(expr.operand, env)
        if expr.op == "-":
            return -val
        if expr.op == "!":
            return not self._truthy(val)
        raise MogError(f"unknown unary operator '{expr.op}'")

    def _eval_call(self, expr: ast.Call, env: Environment) -> Any:
        args = [self._eval_expr(a, env) for a in expr.args]

        # Method calls: obj.method(args)
        if isinstance(expr.func, ast.FieldAccess):
            obj = self._eval_expr(expr.func.obj, env)
            method = expr.func.field
            return self._call_method(obj, method, args, env)

        func = self._eval_expr(expr.func, env)

        if callable(func):
            return func(*args)
        if isinstance(func, ast.FnDecl):
            return self._call_fn(func, args, self.global_env)
        if isinstance(func, MogClosure):
            return self._call_closure(func, args)
        raise MogError(f"cannot call {type(func).__name__}")

    def _call_method(self, obj: Any, method: str, args: list, env: Environment) -> Any:
        # Array methods
        if isinstance(obj, list):
            if method == "push":
                obj.append(args[0])
                return None
            if method == "pop":
                return obj.pop()
            if method == "map":
                fn = args[0]
                return [self._call_value(fn, [item], env) for item in obj]
            if method == "filter":
                fn = args[0]
                return [item for item in obj if self._truthy(self._call_value(fn, [item], env))]
            if method == "sort":
                if args:
                    import functools
                    fn = args[0]
                    return sorted(obj, key=functools.cmp_to_key(lambda a, b: self._call_value(fn, [a, b], env)))
                return sorted(obj)
            if method == "join":
                return args[0].join(str(x) for x in obj)
            if method == "len":
                return len(obj)
            raise MogError(f"unknown array method '{method}'")

        # String methods
        if isinstance(obj, str):
            if method == "upper": return obj.upper()
            if method == "lower": return obj.lower()
            if method == "trim": return obj.strip()
            if method == "split":
                if args:
                    sep = args[0]
                    if sep == "":
                        return list(obj)
                    return obj.split(sep)
                return obj.split()
            if method == "contains": return args[0] in obj
            if method == "starts_with": return obj.startswith(args[0])
            if method == "ends_with": return obj.endswith(args[0])
            if method == "replace": return obj.replace(args[0], args[1])
            raise MogError(f"unknown string method '{method}'")

        # Map methods
        if isinstance(obj, dict):
            if method == "has": return args[0] in obj
            if method == "keys": return list(obj.keys())
            if method == "values": return list(obj.values())
            raise MogError(f"unknown map method '{method}'")

        raise MogError(f"cannot call method '{method}' on {type(obj).__name__}")

    def _call_value(self, fn: Any, args: list, env: Environment) -> Any:
        if callable(fn):
            return fn(*args)
        if isinstance(fn, ast.FnDecl):
            return self._call_fn(fn, args, self.global_env)
        if isinstance(fn, MogClosure):
            return self._call_closure(fn, args)
        raise MogError(f"cannot call {type(fn).__name__}")

    def _eval_field_access(self, expr: ast.FieldAccess, env: Environment) -> Any:
        obj = self._eval_expr(expr.obj, env)
        if isinstance(obj, MogStruct):
            if expr.field in obj.fields:
                return obj.fields[expr.field]
            raise MogError(f"struct '{obj.name}' has no field '{expr.field}'")
        if isinstance(obj, (list, str, dict)):
            if expr.field == "len":
                return len(obj)
        raise MogError(f"cannot access field '{expr.field}' on {type(obj).__name__}")

    def _eval_index_access(self, expr: ast.IndexAccess, env: Environment) -> Any:
        obj = self._eval_expr(expr.obj, env)
        idx = self._eval_expr(expr.index, env)
        if isinstance(obj, list):
            return obj[int(idx)]
        if isinstance(obj, dict):
            return obj[idx]
        if isinstance(obj, str):
            return obj[int(idx)]
        raise MogError(f"cannot index {type(obj).__name__}")

    def _eval_struct_construct(self, expr: ast.StructConstruct, env: Environment) -> MogStruct:
        fields = {}
        for fname, fexpr in expr.fields:
            fields[fname] = self._eval_expr(fexpr, env)
        return MogStruct(expr.name, fields)

    def _eval_match(self, expr: ast.MatchExpr, env: Environment) -> Any:
        subject = self._eval_expr(expr.subject, env)
        for arm in expr.arms:
            bound_env = env.child()
            if self._match_pattern(arm.pattern, subject, bound_env):
                if isinstance(arm.body, ast.Block):
                    try:
                        self._exec_block(arm.body, bound_env)
                        return None
                    except ReturnSignal as r:
                        raise  # propagate returns
                else:
                    return self._eval_expr(arm.body, bound_env)
        raise MogError("non-exhaustive match")

    def _match_pattern(self, pattern: Any, value: Any, env: Environment) -> bool:
        if isinstance(pattern, ast.WildcardPattern):
            return True
        if isinstance(pattern, ast.LitPattern):
            lit = pattern.value
            if isinstance(lit, ast.IntLit):
                return value == lit.value
            if isinstance(lit, ast.FloatLit):
                return value == lit.value
            if isinstance(lit, ast.StringLit):
                return value == lit.value
            if isinstance(lit, ast.BoolLit):
                return value == lit.value
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
            if isinstance(value, MogOptional) and not value.is_some:
                return True
            return False
        if isinstance(pattern, ast.IdentPattern):
            env.define(pattern.name, value)
            return True
        return False

    def _eval_cast(self, expr: ast.CastExpr, env: Environment) -> Any:
        val = self._eval_expr(expr.expr, env)
        t = expr.target_type
        if t in ("i64", "i32", "i16", "i8", "u64", "u32", "u16", "u8", "int"):
            return int(val)
        if t in ("f64", "f32", "f16", "bf16", "float"):
            return float(val)
        if t == "string":
            return str(val)
        return val

    def _eval_propagate(self, expr: ast.PropagateExpr, env: Environment) -> Any:
        val = self._eval_expr(expr.expr, env)
        if isinstance(val, MogResult):
            if val.is_ok:
                return val.value
            raise PropagateSignal(val)
        if isinstance(val, MogOptional):
            if val.is_some:
                return val.value
            raise PropagateSignal(val)
        raise MogError("? operator used on non-Result/Optional value")

    def _truthy(self, val: Any) -> bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, int):
            return val != 0
        if isinstance(val, float):
            return val != 0.0
        if isinstance(val, str):
            return len(val) > 0
        if val is None:
            return False
        if isinstance(val, MogOptional):
            return val.is_some
        return True
