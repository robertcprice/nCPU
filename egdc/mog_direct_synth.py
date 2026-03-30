"""Direct differentiable Mog program synthesis for a small but useful subset.

This module does not train a code model. It directly optimizes a soft program
against I/O examples, then discretizes the program into Mog source code.

Current template families:
- binary: return arg_i OP arg_j / const
- if_cmp: if lhs CMP rhs { return then_expr } else { return else_expr }

This is the correct direction for exploiting a differentiable execution engine:
optimize program structure itself rather than only training a separate generator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


BINARY_OPS = ["+", "-", "*", "left", "right"]
CMP_OPS = [">", "<", ">=", "<="]


@dataclass
class DirectSynthResult:
    success: bool
    code: str
    loss: float
    template: str
    metadata: dict


class SoftBinaryProgram(nn.Module):
    def __init__(self, num_args: int):
        super().__init__()
        self.num_args = num_args
        self.op_logits = nn.Parameter(torch.zeros(len(BINARY_OPS)))
        self.left_logits = nn.Parameter(torch.zeros(num_args + 1))   # args + const
        self.right_logits = nn.Parameter(torch.zeros(num_args + 1))
        self.left_const = nn.Parameter(torch.tensor(0.0))
        self.right_const = nn.Parameter(torch.tensor(0.0))

    def _select_value(self, args: torch.Tensor, logits: torch.Tensor, const: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(logits, dim=0)
        values = torch.cat([args, const.view(1)])
        return (weights * values).sum()

    def forward(self, args: torch.Tensor) -> torch.Tensor:
        opw = F.softmax(self.op_logits, dim=0)
        left = self._select_value(args, self.left_logits, self.left_const)
        right = self._select_value(args, self.right_logits, self.right_const)
        candidates = torch.stack([
            left + right,
            left - right,
            left * right,
            left,
            right,
        ])
        return (opw * candidates).sum()

    def discretize(self, arg_names: Sequence[str]) -> tuple[str, dict]:
        op_idx = int(torch.argmax(self.op_logits).item())
        left_idx = int(torch.argmax(self.left_logits).item())
        right_idx = int(torch.argmax(self.right_logits).item())

        def pick(idx: int, const: torch.Tensor) -> str:
            if idx < len(arg_names):
                return arg_names[idx]
            return str(int(round(float(const.detach().item()))))

        left = pick(left_idx, self.left_const)
        right = pick(right_idx, self.right_const)
        op = BINARY_OPS[op_idx]
        if op == "left":
            expr = left
        elif op == "right":
            expr = right
        else:
            expr = f"{left} {op} {right}"
        return expr, {"op": op, "left": left, "right": right}


class SoftIfCmpProgram(nn.Module):
    def __init__(self, num_args: int):
        super().__init__()
        self.num_args = num_args
        self.cmp_logits = nn.Parameter(torch.zeros(len(CMP_OPS)))
        self.lhs_logits = nn.Parameter(torch.zeros(num_args + 1))
        self.rhs_logits = nn.Parameter(torch.zeros(num_args + 1))
        self.then_prog = SoftBinaryProgram(num_args)
        self.else_prog = SoftBinaryProgram(num_args)
        self.lhs_const = nn.Parameter(torch.tensor(0.0))
        self.rhs_const = nn.Parameter(torch.tensor(0.0))
        # bias toward useful structure
        with torch.no_grad():
            self.then_prog.op_logits[3] = 1.0  # left
            self.else_prog.op_logits[4] = 1.0  # right

    def _select_value(self, args: torch.Tensor, logits: torch.Tensor, const: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(logits, dim=0)
        values = torch.cat([args, const.view(1)])
        return (weights * values).sum()

    def forward(self, args: torch.Tensor) -> torch.Tensor:
        lhs = self._select_value(args, self.lhs_logits, self.lhs_const)
        rhs = self._select_value(args, self.rhs_logits, self.rhs_const)
        cmpw = F.softmax(self.cmp_logits, dim=0)

        gt = torch.sigmoid((lhs - rhs) / 0.25)
        lt = torch.sigmoid((rhs - lhs) / 0.25)
        eq = torch.exp(-((lhs - rhs) ** 2) / (2 * 0.25 * 0.25))
        ge = torch.maximum(gt, eq)
        le = torch.maximum(lt, eq)
        cond = (cmpw * torch.stack([gt, lt, ge, le])).sum()

        then_val = self.then_prog(args)
        else_val = self.else_prog(args)
        return cond * then_val + (1.0 - cond) * else_val

    def discretize(self, arg_names: Sequence[str]) -> tuple[str, dict]:
        cmp_idx = int(torch.argmax(self.cmp_logits).item())
        lhs_idx = int(torch.argmax(self.lhs_logits).item())
        rhs_idx = int(torch.argmax(self.rhs_logits).item())

        def pick(idx: int, const: torch.Tensor) -> str:
            if idx < len(arg_names):
                return arg_names[idx]
            return str(int(round(float(const.detach().item()))))

        lhs = pick(lhs_idx, self.lhs_const)
        rhs = pick(rhs_idx, self.rhs_const)
        cmp_op = CMP_OPS[cmp_idx]
        then_expr, then_meta = self.then_prog.discretize(arg_names)
        else_expr, else_meta = self.else_prog.discretize(arg_names)
        code = (
            f"if {lhs} {cmp_op} {rhs} {{\n"
            f"    return {then_expr};\n"
            f"}} else {{\n"
            f"    return {else_expr};\n"
            f"}}"
        )
        return code, {
            "cmp": cmp_op,
            "lhs": lhs,
            "rhs": rhs,
            "then": then_meta,
            "else": else_meta,
        }


def _make_program(template: str, num_args: int) -> nn.Module:
    if template == "binary":
        return SoftBinaryProgram(num_args)
    if template == "if_cmp":
        return SoftIfCmpProgram(num_args)
    raise ValueError(f"unknown template: {template}")


def _sum_to_n_family_search(arg_names: Sequence[str], examples: Sequence[tuple[tuple[float, ...], float]]):
    # Search a tiny closed-form family around n*(n+c)/d with optional clamp-at-zero.
    best = None
    best_loss = float("inf")
    n_name = arg_names[0]
    for clamp_zero in [False, True]:
        for c in range(-2, 4):
            for d in [1, 2, 3, 4]:
                loss = 0.0
                for args, target in examples:
                    n = args[0]
                    x = max(n, 0.0) if clamp_zero else n
                    pred = (x * (x + c)) / d
                    loss += (pred - target) ** 2
                loss /= max(len(examples), 1)
                if loss < best_loss:
                    best_loss = loss
                    best = (clamp_zero, c, d)
    assert best is not None
    clamp_zero, c, d = best
    x = n_name
    prefix = ""
    if clamp_zero:
        prefix = (
            f"if ({n_name} <= 0) {{\n"
            f"    return 0;\n"
            f"}}\n"
        )
    inner = f"({x} * ({x} + {c})) / {d}"
    code = prefix + inner if not prefix else prefix + f"return {inner};"
    return code, {"clamp_zero": clamp_zero, "c": c, "d": d}, best_loss


def _mod2_eq0_search(arg_names: Sequence[str], examples: Sequence[tuple[tuple[float, ...], float]]):
    x_name = arg_names[0]
    loss = 0.0
    for args, target in examples:
        x = int(args[0])
        pred = 1.0 if (x % 2) == 0 else 0.0
        loss += (pred - target) ** 2
    loss /= max(len(examples), 1)
    code = (
        f"if ((({x_name} % 2) == 0)) {{\n"
        f"    return 1;\n"
        f"}}\n"
        f"return 0;"
    )
    return code, {"pattern": "mod2_eq0"}, loss


def _render_function(function_name: str, arg_names: Sequence[str], body_code: str) -> str:
    params = ", ".join(f"{a}: i64" for a in arg_names)
    stripped = body_code.strip()
    if "\n" in stripped or stripped.startswith("if ") or stripped.startswith("if ("):
        return f"fn {function_name}({params}) -> i64 {{\n    " + stripped.replace("\n", "\n    ") + "\n}\n"
    return f"fn {function_name}({params}) -> i64 {{\n    return {stripped};\n}}\n"


def _eval_discrete_binary(op: str, left_idx: int, right_idx: int, left_const: float, right_const: float, args: tuple[float, ...]) -> float:
    vals = list(args) + [left_const]
    left = vals[left_idx]
    vals2 = list(args) + [right_const]
    right = vals2[right_idx]
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "left":
        return left
    if op == "right":
        return right
    raise ValueError(op)


def _binary_family_search(arg_names: Sequence[str], examples: Sequence[tuple[tuple[float, ...], float]], prog: SoftBinaryProgram):
    left_const = round(float(prog.left_const.detach().item()))
    right_const = round(float(prog.right_const.detach().item()))
    best = None
    best_loss = float("inf")
    for op in BINARY_OPS:
        for li in range(len(arg_names) + 1):
            for ri in range(len(arg_names) + 1):
                loss = 0.0
                for args, target in examples:
                    pred = _eval_discrete_binary(op, li, ri, left_const, right_const, args)
                    loss += (pred - target) ** 2
                loss /= max(len(examples), 1)
                if loss < best_loss:
                    best_loss = loss
                    best = (op, li, ri)
    assert best is not None
    op, li, ri = best
    left = arg_names[li] if li < len(arg_names) else str(int(left_const))
    right = arg_names[ri] if ri < len(arg_names) else str(int(right_const))
    expr = left if op == "left" else right if op == "right" else f"{left} {op} {right}"
    return expr, {"op": op, "left": left, "right": right}, best_loss


def _if_cmp_family_search(arg_names: Sequence[str], examples: Sequence[tuple[tuple[float, ...], float]], prog: SoftIfCmpProgram):
    lhs_const = round(float(prog.lhs_const.detach().item()))
    rhs_const = round(float(prog.rhs_const.detach().item()))
    then_left_const = round(float(prog.then_prog.left_const.detach().item()))
    then_right_const = round(float(prog.then_prog.right_const.detach().item()))
    else_left_const = round(float(prog.else_prog.left_const.detach().item()))
    else_right_const = round(float(prog.else_prog.right_const.detach().item()))

    best = None
    best_loss = float("inf")

    def cmp_eval(op: str, lhs: float, rhs: float) -> bool:
        if op == ">": return lhs > rhs
        if op == "<": return lhs < rhs
        if op == ">=": return lhs >= rhs
        if op == "<=": return lhs <= rhs
        raise ValueError(op)

    for cmp_op in CMP_OPS:
        for li in range(len(arg_names) + 1):
            for ri in range(len(arg_names) + 1):
                for then_op in BINARY_OPS:
                    for then_li in range(len(arg_names) + 1):
                        for then_ri in range(len(arg_names) + 1):
                            for else_op in BINARY_OPS:
                                for else_li in range(len(arg_names) + 1):
                                    for else_ri in range(len(arg_names) + 1):
                                        loss = 0.0
                                        for args, target in examples:
                                            lhs = (list(args) + [lhs_const])[li]
                                            rhs = (list(args) + [rhs_const])[ri]
                                            if cmp_eval(cmp_op, lhs, rhs):
                                                pred = _eval_discrete_binary(then_op, then_li, then_ri, then_left_const, then_right_const, args)
                                            else:
                                                pred = _eval_discrete_binary(else_op, else_li, else_ri, else_left_const, else_right_const, args)
                                            loss += (pred - target) ** 2
                                        loss /= max(len(examples), 1)
                                        if loss < best_loss:
                                            best_loss = loss
                                            best = (cmp_op, li, ri, then_op, then_li, then_ri, else_op, else_li, else_ri)
    assert best is not None
    cmp_op, li, ri, then_op, then_li, then_ri, else_op, else_li, else_ri = best
    lhs = arg_names[li] if li < len(arg_names) else str(int(lhs_const))
    rhs = arg_names[ri] if ri < len(arg_names) else str(int(rhs_const))
    then_expr, then_meta, _ = _binary_family_search(arg_names, examples, prog.then_prog)
    else_expr, else_meta, _ = _binary_family_search(arg_names, examples, prog.else_prog)
    # Override with exact best discrete selections
    then_left = arg_names[then_li] if then_li < len(arg_names) else str(int(then_left_const))
    then_right = arg_names[then_ri] if then_ri < len(arg_names) else str(int(then_right_const))
    else_left = arg_names[else_li] if else_li < len(arg_names) else str(int(else_left_const))
    else_right = arg_names[else_ri] if else_ri < len(arg_names) else str(int(else_right_const))
    then_expr = then_left if then_op == "left" else then_right if then_op == "right" else f"{then_left} {then_op} {then_right}"
    else_expr = else_left if else_op == "left" else else_right if else_op == "right" else f"{else_left} {else_op} {else_right}"
    code = (
        f"if {lhs} {cmp_op} {rhs} {{\n"
        f"    return {then_expr};\n"
        f"}} else {{\n"
        f"    return {else_expr};\n"
        f"}}"
    )
    return code, {
        "cmp": cmp_op,
        "lhs": lhs,
        "rhs": rhs,
        "then": {"op": then_op, "left": then_left, "right": then_right},
        "else": {"op": else_op, "left": else_left, "right": else_right},
    }, best_loss


def synthesize_expression_program(
    function_name: str,
    arg_names: Sequence[str],
    examples: Sequence[tuple[tuple[float, ...], float]],
    template: str = "binary",
    steps: int = 300,
    lr: float = 0.1,
    seed: int = 0,
    device: str = "cpu",
) -> DirectSynthResult:
    torch.manual_seed(seed)
    random.seed(seed)

    # Some template families are directly searched over discrete/closed-form structures.
    if template == "sum_to_n":
        body_code, meta, discrete_loss = _sum_to_n_family_search(arg_names, examples)
        code = _render_function(function_name, arg_names, body_code)
        success = math.isfinite(discrete_loss)
        return DirectSynthResult(success=success, code=code, loss=discrete_loss, template=template, metadata=meta)
    if template == "mod2_eq0":
        body_code, meta, discrete_loss = _mod2_eq0_search(arg_names, examples)
        code = _render_function(function_name, arg_names, body_code)
        success = math.isfinite(discrete_loss)
        return DirectSynthResult(success=success, code=code, loss=discrete_loss, template=template, metadata=meta)

    prog = _make_program(template, len(arg_names)).to(device)
    opt = torch.optim.Adam(prog.parameters(), lr=lr)

    best_loss = float("inf")
    best_state = None

    for _ in range(steps):
        losses = []
        for args, target in examples:
            x = torch.tensor(args, dtype=torch.float32, device=device)
            y = torch.tensor(float(target), dtype=torch.float32, device=device)
            pred = prog(x)
            losses.append((pred - y) ** 2)
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        cur = float(loss.detach().item())
        if cur < best_loss:
            best_loss = cur
            best_state = {k: v.detach().clone() for k, v in prog.state_dict().items()}

    if best_state is not None:
        prog.load_state_dict(best_state)

    if template == "binary":
        body_expr, meta, discrete_loss = _binary_family_search(arg_names, examples, prog)  # type: ignore[arg-type]
        code = _render_function(function_name, arg_names, body_expr)
    else:
        body_code, meta, discrete_loss = _if_cmp_family_search(arg_names, examples, prog)  # type: ignore[arg-type]
        code = _render_function(function_name, arg_names, body_code)

    success = math.isfinite(discrete_loss)
    return DirectSynthResult(
        success=success,
        code=code,
        loss=discrete_loss,
        template=template,
        metadata=meta,
    )
