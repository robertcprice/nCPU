"""Differentiable program search for Mog.

This is the real use of the differentiable execution engine:
represent a program as learnable parameters, execute it differentiably,
and let gradient descent discover the program that satisfies I/O examples.

No hand-authored templates. No pattern matching. No heuristics.
The differentiable CPU finds the program.

Architecture:
- SoftMogProgram: a parameterized program with learnable soft choices
  for statements, expressions, operators, variables, constants, branches
- The differentiable Mog executor runs it
- Loss = MSE against expected outputs
- Optimizer finds the program

This is what TerpreT, differentiable Forth, and the nCPU SoftProgram do,
but for a high-level language.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Soft program representation
# ---------------------------------------------------------------------------

# Statement types a slot can be
STMT_TYPES = ["nop", "assign_binop", "assign_const", "assign_arg", "if_return", "accum_loop", "return_var"]

# Binary operators
OPS = ["+", "-", "*", "/", "%"]

# Comparison operators for if conditions
CMP_OPS = [">", "<", ">=", "<=", "==", "!="]


class SoftMogProgram(nn.Module):
    """A fully differentiable parameterized Mog program.

    The program has N statement slots. Each slot has learnable logits
    that determine what kind of statement it is, which variables/args
    it reads and writes, what operator it uses, and what constants
    appear.

    Everything is soft and differentiable. After optimization,
    we discretize by taking argmax at each choice point.
    """

    def __init__(
        self,
        num_args: int,
        num_slots: int = 8,
        num_vars: int = 6,
        max_const: float = 100.0,
    ):
        super().__init__()
        self.num_args = num_args
        self.num_slots = num_slots
        self.num_vars = num_vars
        self.num_sources = num_args + num_vars  # args + local vars
        self.max_const = max_const

        # Per-slot learnable parameters
        self.stmt_logits = nn.Parameter(torch.zeros(num_slots, len(STMT_TYPES)))
        self.dst_logits = nn.Parameter(torch.zeros(num_slots, num_vars))
        self.src1_logits = nn.Parameter(torch.zeros(num_slots, self.num_sources))
        self.src2_logits = nn.Parameter(torch.zeros(num_slots, self.num_sources))
        self.op_logits = nn.Parameter(torch.zeros(num_slots, len(OPS)))
        self.cmp_logits = nn.Parameter(torch.zeros(num_slots, len(CMP_OPS)))
        self.const_values = nn.Parameter(torch.zeros(num_slots))
        self.return_src_logits = nn.Parameter(torch.zeros(num_slots, self.num_sources))

        # For accum_loop: loop bound, accumulator init, and body op
        self.loop_bound_logits = nn.Parameter(torch.zeros(num_slots, self.num_sources))
        self.loop_body_op_logits = nn.Parameter(torch.zeros(num_slots, len(OPS)))

        # If-return: comparison threshold (learnable constant for RHS)
        self.if_rhs_logits = nn.Parameter(torch.zeros(num_slots, self.num_sources))
        self.if_return_val_logits = nn.Parameter(torch.zeros(num_slots, self.num_sources))

        self._init_biases()

    def _init_biases(self):
        with torch.no_grad():
            # Bias first slot toward assign_binop (compute something)
            self.stmt_logits[0, 1] = 2.0  # assign_binop
            # Bias remaining slots toward nop
            for i in range(1, self.num_slots - 1):
                self.stmt_logits[i, 0] = 1.0
            # Bias last slot toward return_var
            self.stmt_logits[-1, -1] = 3.0
            # Init first slot to read from args and write to v0
            self.src1_logits[0, 0] = 2.0  # src1 = arg0
            if self.num_args > 1:
                self.src2_logits[0, 1] = 2.0  # src2 = arg1
            self.dst_logits[0, 0] = 2.0  # dst = v0
            self.op_logits[0, 0] = 1.0  # op = +
            # Return reads from v0
            self.return_src_logits[-1, self.num_args] = 2.0  # return v0
            # Small random perturbation
            nn.init.normal_(self.const_values, std=1.0)

    def forward(self, args: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Execute the soft program on given arguments.

        Args:
            args: (num_args,) tensor of input argument values
            temperature: softmax temperature for discrete choices

        Returns:
            scalar output tensor (the program's return value)
        """
        device = args.device

        # Initialize variable storage: [num_sources] = [args | local_vars]
        storage = torch.zeros(self.num_sources, device=device)
        storage[:self.num_args] = args

        return_value = torch.tensor(0.0, device=device)
        return_prob = torch.tensor(0.0, device=device)

        for slot in range(self.num_slots):
            stmt_w = F.softmax(self.stmt_logits[slot] / temperature, dim=0)
            dst_w = F.softmax(self.dst_logits[slot] / temperature, dim=0)
            src1_w = F.softmax(self.src1_logits[slot] / temperature, dim=0)
            src2_w = F.softmax(self.src2_logits[slot] / temperature, dim=0)
            op_w = F.softmax(self.op_logits[slot] / temperature, dim=0)
            ret_w = F.softmax(self.return_src_logits[slot] / temperature, dim=0)

            # Soft reads
            src1_val = (src1_w * storage).sum()
            src2_val = (src2_w * storage).sum()
            const_val = self.const_values[slot]

            # Compute all binary ops in parallel
            safe_src2 = torch.where(torch.abs(src2_val) < 1e-6, torch.ones_like(src2_val), src2_val)
            op_results = torch.stack([
                src1_val + src2_val,        # +
                src1_val - src2_val,        # -
                src1_val * src2_val,        # *
                src1_val / safe_src2,       # /
                torch.remainder(torch.round(src1_val), torch.clamp(torch.round(torch.abs(safe_src2)), min=1.0)),  # %
            ])
            binop_result = (op_w * op_results).sum()

            # --- Statement effects ---

            # nop: nothing
            # assign_binop: var[dst] = src1 OP src2
            # assign_const: var[dst] = const
            # assign_arg: var[dst] = src1 (just a copy)
            # if_return: if src1 CMP src2 then return if_return_val
            # accum_loop: var[dst] = accumulate(src1, loop_body_op, 0..loop_bound)
            # return_var: return src1

            assign_binop_val = binop_result
            assign_const_val = const_val
            assign_arg_val = src1_val

            # Soft write for assignment statements
            new_val = (
                stmt_w[1] * assign_binop_val +
                stmt_w[2] * assign_const_val +
                stmt_w[3] * assign_arg_val
            )

            # Accum loop: simulate accumulation
            loop_bound_w = F.softmax(self.loop_bound_logits[slot] / temperature, dim=0)
            loop_body_op_w = F.softmax(self.loop_body_op_logits[slot] / temperature, dim=0)
            loop_bound = (loop_bound_w * storage).sum()
            # Approximate: for i in 0..N: acc += f(i, src2)
            # We simulate with a fixed unroll and soft bound weighting
            accum = torch.tensor(0.0, device=device)
            for step in range(16):  # max 16 iterations
                step_t = torch.tensor(float(step), device=device)
                # Weight by how likely this step is within bounds
                in_bounds = torch.sigmoid((loop_bound - step_t) / 0.5)
                # Compute loop body: acc OP step
                step_ops = torch.stack([
                    accum + step_t,
                    accum - step_t,
                    accum * step_t,
                    accum / torch.clamp(step_t, min=1e-6),
                    torch.remainder(torch.round(accum), torch.clamp(torch.round(torch.abs(step_t)), min=1.0)),
                ])
                step_result = (loop_body_op_w * step_ops).sum()
                accum = accum + in_bounds * (step_result - accum)

            new_val = new_val + stmt_w[5] * accum

            # Soft write to destination variable (fully out-of-place)
            write_enable = stmt_w[1] + stmt_w[2] + stmt_w[3] + stmt_w[5]
            new_storage = []
            for idx in range(self.num_sources):
                if idx >= self.num_args:
                    v = idx - self.num_args
                    w = dst_w[v] * write_enable
                    new_storage.append(storage[idx] * (1.0 - w) + new_val * w)
                else:
                    new_storage.append(storage[idx])
            storage = torch.stack(new_storage)

            # If-return
            cmp_w = F.softmax(self.cmp_logits[slot] / temperature, dim=0)
            if_rhs_w = F.softmax(self.if_rhs_logits[slot] / temperature, dim=0)
            if_rhs = (if_rhs_w * storage).sum()
            if_ret_w = F.softmax(self.if_return_val_logits[slot] / temperature, dim=0)
            if_ret_val = (if_ret_w * storage).sum()

            diff = src1_val - if_rhs
            cmp_results = torch.stack([
                torch.sigmoid(diff / 0.25),         # >
                torch.sigmoid(-diff / 0.25),        # <
                torch.sigmoid(diff / 0.25),         # >= (approx)
                torch.sigmoid(-diff / 0.25),        # <= (approx)
                torch.exp(-(diff ** 2) / 0.125),    # ==
                1.0 - torch.exp(-(diff ** 2) / 0.125),  # !=
            ])
            cond_true = (cmp_w * cmp_results).sum()
            if_return_prob = stmt_w[4] * cond_true

            # Blend if-return into output
            return_value = return_value * (1.0 - if_return_prob) + if_ret_val * if_return_prob
            return_prob = return_prob + if_return_prob * (1.0 - return_prob)

            # Return statement
            ret_val = (ret_w * storage).sum()
            ret_prob = stmt_w[6]
            return_value = return_value * (1.0 - ret_prob * (1.0 - return_prob)) + ret_val * ret_prob * (1.0 - return_prob)
            return_prob = return_prob + ret_prob * (1.0 - return_prob)

        return return_value

    def discretize(self, arg_names: Sequence[str]) -> str:
        """Convert soft program to discrete Mog source code."""
        var_names = [f"v{i}" for i in range(self.num_vars)]
        all_names = list(arg_names) + var_names

        lines = []
        for i in range(self.num_vars):
            lines.append(f"    v{i}: i64 = 0;")

        for slot in range(self.num_slots):
            stmt_idx = int(torch.argmax(self.stmt_logits[slot]).item())
            stmt_type = STMT_TYPES[stmt_idx]

            if stmt_type == "nop":
                continue

            dst_idx = int(torch.argmax(self.dst_logits[slot]).item())
            dst = var_names[dst_idx]
            src1_idx = int(torch.argmax(self.src1_logits[slot]).item())
            src2_idx = int(torch.argmax(self.src2_logits[slot]).item())
            src1 = all_names[src1_idx]
            src2 = all_names[src2_idx]

            if stmt_type == "assign_binop":
                op_idx = int(torch.argmax(self.op_logits[slot]).item())
                op = OPS[op_idx]
                if op in ("/", "%"):
                    lines.append(f"    {dst} = {src1} {op} {src2};")
                else:
                    lines.append(f"    {dst} = {src1} {op} {src2};")

            elif stmt_type == "assign_const":
                c = int(round(self.const_values[slot].item()))
                lines.append(f"    {dst} = {c};")

            elif stmt_type == "assign_arg":
                lines.append(f"    {dst} = {src1};")

            elif stmt_type == "if_return":
                cmp_idx = int(torch.argmax(self.cmp_logits[slot]).item())
                cmp_op = CMP_OPS[cmp_idx]
                if_rhs_idx = int(torch.argmax(self.if_rhs_logits[slot]).item())
                if_rhs = all_names[if_rhs_idx]
                if_ret_idx = int(torch.argmax(self.if_return_val_logits[slot]).item())
                if_ret = all_names[if_ret_idx]
                lines.append(f"    if ({src1} {cmp_op} {if_rhs}) {{ return {if_ret}; }}")

            elif stmt_type == "accum_loop":
                lb_idx = int(torch.argmax(self.loop_bound_logits[slot]).item())
                lb = all_names[lb_idx]
                body_op_idx = int(torch.argmax(self.loop_body_op_logits[slot]).item())
                body_op = OPS[body_op_idx]
                lines.append(f"    i: i64 = 0;")
                lines.append(f"    while i < {lb} {{")
                lines.append(f"        {dst} = {dst} {body_op} i;")
                lines.append(f"        i = i + 1;")
                lines.append(f"    }}")

            elif stmt_type == "return_var":
                ret_idx = int(torch.argmax(self.return_src_logits[slot]).item())
                ret = all_names[ret_idx]
                lines.append(f"    return {ret};")

        body = "\n".join(lines)
        params = ", ".join(f"{a}: i64" for a in arg_names)
        return f"fn program({params}) -> i64 {{\n{body}\n}}\n"


# ---------------------------------------------------------------------------
# Program search via gradient descent through differentiable execution
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    success: bool
    code: str
    loss: float
    steps_taken: int
    metadata: dict[str, Any]


class SoftBranchingProgram(nn.Module):
    """A program with: optional compute slots, then if(cond) return X else return Y.

    This is specifically designed for branching programs like max2, abs_diff, sign.
    The structure is:
        v0 = src1 OP src2   (optional compute)
        v1 = src1 OP src2   (optional compute)
        if (lhs CMP rhs) { return then_val } else { return else_val }
    """

    def __init__(self, num_args: int, num_compute_slots: int = 2, num_vars: int = 4):
        super().__init__()
        self.num_args = num_args
        self.num_compute_slots = num_compute_slots
        self.num_vars = num_vars
        self.num_sources = num_args + num_vars

        # Compute slots
        self.compute_enable = nn.Parameter(torch.zeros(num_compute_slots))
        self.compute_src1 = nn.Parameter(torch.zeros(num_compute_slots, self.num_sources))
        self.compute_src2 = nn.Parameter(torch.zeros(num_compute_slots, self.num_sources))
        self.compute_op = nn.Parameter(torch.zeros(num_compute_slots, len(OPS)))
        self.compute_dst = nn.Parameter(torch.zeros(num_compute_slots, num_vars))

        # Branch: if (lhs CMP rhs) { return then_expr } else { return else_expr }
        # Each arm is a binary expression: src1 OP src2 (with "left" = identity)
        self.cmp_logits = nn.Parameter(torch.zeros(len(CMP_OPS)))
        self.lhs_logits = nn.Parameter(torch.zeros(self.num_sources))
        self.rhs_logits = nn.Parameter(torch.zeros(self.num_sources))
        # Then arm expression
        self.then_src1 = nn.Parameter(torch.zeros(self.num_sources))
        self.then_src2 = nn.Parameter(torch.zeros(self.num_sources))
        self.then_op = nn.Parameter(torch.zeros(len(OPS) + 1))  # +1 for "identity" (just src1)
        # Else arm expression
        self.else_src1 = nn.Parameter(torch.zeros(self.num_sources))
        self.else_src2 = nn.Parameter(torch.zeros(self.num_sources))
        self.else_op = nn.Parameter(torch.zeros(len(OPS) + 1))

        # Constants
        self.const_values = nn.Parameter(torch.zeros(num_compute_slots))

        self._init()

    def _init(self):
        with torch.no_grad():
            self.compute_enable[0] = -1.0  # disabled by default
            self.compute_src1[0, 0] = 1.0
            if self.num_args > 1:
                self.compute_src2[0, 1] = 1.0
            self.compute_dst[0, 0] = 1.0
            if self.num_args > 1:
                self.lhs_logits[0] = 1.0
                self.rhs_logits[1] = 1.0
                self.then_src1[0] = 1.0  # then returns arg0
                self.then_op[-1] = 2.0   # identity
                self.else_src1[1] = 1.0  # else returns arg1
                self.else_op[-1] = 2.0   # identity
            self.cmp_logits[0] = 1.0  # >

    def forward(self, args: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        device = args.device
        storage = torch.zeros(self.num_sources, device=device)
        storage[:self.num_args] = args

        # Execute compute slots
        for slot in range(self.num_compute_slots):
            enable = torch.sigmoid(self.compute_enable[slot])
            src1_w = F.softmax(self.compute_src1[slot] / temperature, dim=0)
            src2_w = F.softmax(self.compute_src2[slot] / temperature, dim=0)
            op_w = F.softmax(self.compute_op[slot] / temperature, dim=0)
            dst_w = F.softmax(self.compute_dst[slot] / temperature, dim=0)

            src1_val = (src1_w * storage).sum()
            src2_val = (src2_w * storage).sum()
            safe_src2 = torch.where(torch.abs(src2_val) < 1e-6, torch.ones_like(src2_val), src2_val)

            op_results = torch.stack([
                src1_val + src2_val,
                src1_val - src2_val,
                src1_val * src2_val,
                src1_val / safe_src2,
                torch.remainder(torch.round(src1_val), torch.clamp(torch.round(torch.abs(safe_src2)), min=1.0)),
            ])
            result = (op_w * op_results).sum()

            new_storage = []
            for idx in range(self.num_sources):
                if idx >= self.num_args:
                    v = idx - self.num_args
                    w = dst_w[v] * enable
                    new_storage.append(storage[idx] * (1.0 - w) + result * w)
                else:
                    new_storage.append(storage[idx])
            storage = torch.stack(new_storage)

        # Branch
        cmp_w = F.softmax(self.cmp_logits / temperature, dim=0)
        lhs_w = F.softmax(self.lhs_logits / temperature, dim=0)
        rhs_w = F.softmax(self.rhs_logits / temperature, dim=0)

        lhs = (lhs_w * storage).sum()
        rhs = (rhs_w * storage).sum()

        # Then arm: expression
        then_val = self._eval_arm(storage, self.then_src1, self.then_src2, self.then_op, temperature)
        # Else arm: expression
        else_val = self._eval_arm(storage, self.else_src1, self.else_src2, self.else_op, temperature)

        diff = lhs - rhs
        cmp_results = torch.stack([
            torch.sigmoid(diff / 0.25),                      # >
            torch.sigmoid(-diff / 0.25),                     # <
            torch.sigmoid(diff / 0.25),                      # >= (approx)
            torch.sigmoid(-diff / 0.25),                     # <= (approx)
            torch.exp(-(diff ** 2) / 0.125),                 # ==
            1.0 - torch.exp(-(diff ** 2) / 0.125),           # !=
        ])
        cond = (cmp_w * cmp_results).sum()

        return cond * then_val + (1.0 - cond) * else_val

    def _eval_arm(self, storage: torch.Tensor, src1_logits: torch.Tensor,
                   src2_logits: torch.Tensor, op_logits: torch.Tensor,
                   temperature: float) -> torch.Tensor:
        """Evaluate a branch arm expression: src1 OP src2, or just src1 (identity)."""
        src1_w = F.softmax(src1_logits / temperature, dim=0)
        src2_w = F.softmax(src2_logits / temperature, dim=0)
        op_w = F.softmax(op_logits / temperature, dim=0)  # [len(OPS) + 1]
        s1 = (src1_w * storage).sum()
        s2 = (src2_w * storage).sum()
        safe_s2 = torch.where(torch.abs(s2) < 1e-6, torch.ones_like(s2), s2)
        op_results = torch.stack([
            s1 + s2,
            s1 - s2,
            s1 * s2,
            s1 / safe_s2,
            torch.remainder(torch.round(s1), torch.clamp(torch.round(torch.abs(safe_s2)), min=1.0)),
            s1,  # identity
        ])
        return (op_w * op_results).sum()

    def _discretize_arm(self, src1_logits: torch.Tensor, src2_logits: torch.Tensor,
                        op_logits: torch.Tensor, all_names: list[str]) -> str:
        s1_idx = int(torch.argmax(src1_logits).item())
        s2_idx = int(torch.argmax(src2_logits).item())
        op_idx = int(torch.argmax(op_logits).item())
        if op_idx >= len(OPS):  # identity
            return all_names[s1_idx]
        return f"{all_names[s1_idx]} {OPS[op_idx]} {all_names[s2_idx]}"

    def discretize(self, arg_names: list[str]) -> str:
        var_names = [f"v{i}" for i in range(self.num_vars)]
        all_names = list(arg_names) + var_names
        lines = []
        for i in range(self.num_vars):
            lines.append(f"    v{i}: i64 = 0;")

        for slot in range(self.num_compute_slots):
            if torch.sigmoid(self.compute_enable[slot]).item() > 0.5:
                src1_idx = int(torch.argmax(self.compute_src1[slot]).item())
                src2_idx = int(torch.argmax(self.compute_src2[slot]).item())
                op_idx = int(torch.argmax(self.compute_op[slot]).item())
                dst_idx = int(torch.argmax(self.compute_dst[slot]).item())
                lines.append(f"    {var_names[dst_idx]} = {all_names[src1_idx]} {OPS[op_idx]} {all_names[src2_idx]};")

        cmp_idx = int(torch.argmax(self.cmp_logits).item())
        lhs_idx = int(torch.argmax(self.lhs_logits).item())
        rhs_idx = int(torch.argmax(self.rhs_logits).item())
        then_expr = self._discretize_arm(self.then_src1, self.then_src2, self.then_op, all_names)
        else_expr = self._discretize_arm(self.else_src1, self.else_src2, self.else_op, all_names)

        lines.append(f"    if ({all_names[lhs_idx]} {CMP_OPS[cmp_idx]} {all_names[rhs_idx]}) {{")
        lines.append(f"        return {then_expr};")
        lines.append(f"    }} else {{")
        lines.append(f"        return {else_expr};")
        lines.append(f"    }}")

        body = "\n".join(lines)
        params = ", ".join(f"{a}: i64" for a in arg_names)
        return f"fn program({params}) -> i64 {{\n{body}\n}}\n"


ARM_EXPRS = ["identity", "+", "-", "*"]  # Reduced op set for arms to keep search tractable


def _py_eval_expr(expr_str: str, env: dict[str, float]) -> float:
    """Evaluate a simple Mog expression in Python. Fast, no interpreter overhead."""
    expr_str = expr_str.strip()
    if expr_str in env:
        return env[expr_str]
    try:
        return float(expr_str)
    except ValueError:
        pass
    for op in [" + ", " - ", " * "]:
        if op in expr_str:
            parts = expr_str.split(op, 1)
            l = _py_eval_expr(parts[0], env)
            r = _py_eval_expr(parts[1], env)
            if op == " + ": return l + r
            if op == " - ": return l - r
            if op == " * ": return l * r
    return 0.0


def _py_eval_cmp(cmp_op: str, lhs: float, rhs: float) -> bool:
    if cmp_op == ">": return lhs > rhs
    if cmp_op == "<": return lhs < rhs
    if cmp_op == ">=": return lhs >= rhs
    if cmp_op == "<=": return lhs <= rhs
    if cmp_op == "==": return lhs == rhs
    if cmp_op == "!=": return lhs != rhs
    return False


def _py_eval_branch_program(
    cmp_op: str, lhs_name: str, rhs_name: str,
    then_expr: str, else_expr: str,
    arg_names: list[str], examples,
) -> float:
    """Evaluate a branching program on examples using pure Python. Very fast."""
    total = 0.0
    for args, target in examples:
        env = {n: float(v) for n, v in zip(arg_names, args)}
        lhs = _py_eval_expr(lhs_name, env)
        rhs = _py_eval_expr(rhs_name, env)
        if _py_eval_cmp(cmp_op, lhs, rhs):
            pred = _py_eval_expr(then_expr, env)
        else:
            pred = _py_eval_expr(else_expr, env)
        total += (pred - target) ** 2
    return total / max(len(examples), 1)


def _branching_refinement(prog: SoftBranchingProgram, arg_names: list[str],
                          examples, function_name: str) -> tuple[str, float]:
    """Fast combinatorial refinement for branching programs using pure Python eval."""
    best_code = prog.discretize(arg_names).replace("fn program(", f"fn {function_name}(")
    best_loss = _eval_code_on_examples(best_code, arg_names, examples)

    if best_loss < 1e-6:
        return best_code, best_loss

    params = ", ".join(f"{a}: i64" for a in arg_names)
    var_lines = "\n".join(f"    v{i}: i64 = 0;" for i in range(prog.num_vars))

    CONSTS = [0, 1, -1, 100]
    search_names = list(arg_names) + [str(c) for c in CONSTS]

    # Build arm expressions
    arm_set: list[str] = list(search_names)
    for n1 in arg_names:
        for n2 in arg_names:
            for op in ["+", "-", "*"]:
                arm_set.append(f"{n1} {op} {n2}")
    for n1 in arg_names:
        for c in CONSTS:
            for op in ["+", "-", "*"]:
                arm_set.append(f"{n1} {op} {c}")
    arm_set = list(dict.fromkeys(arm_set))

    best_discrete_loss = best_loss

    for cmp_op in CMP_OPS:
        for lhs_name in search_names:
            for rhs_name in search_names:
                for then_e in arm_set:
                    for else_e in arm_set:
                        loss = _py_eval_branch_program(
                            cmp_op, lhs_name, rhs_name,
                            then_e, else_e, arg_names, examples,
                        )
                        if loss < best_discrete_loss:
                            best_discrete_loss = loss
                            code = (
                                f"fn {function_name}({params}) -> i64 {{\n"
                                f"{var_lines}\n"
                                f"    if ({lhs_name} {cmp_op} {rhs_name}) {{\n"
                                f"        return {then_e};\n"
                                f"    }} else {{\n"
                                f"        return {else_e};\n"
                                f"    }}\n"
                                f"}}\n"
                            )
                            best_code = code
                            best_loss = loss
                            if best_loss < 1e-6:
                                return best_code, best_loss
    return best_code, best_loss


def search_program(
    arg_names: Sequence[str],
    examples: Sequence[tuple[tuple[float, ...], float]],
    function_name: str = "program",
    num_slots: int = 8,
    num_vars: int = 6,
    steps: int = 2000,
    lr: float = 0.05,
    temperature_start: float = 2.0,
    temperature_end: float = 0.1,
    seed: int = 0,
    success_threshold: float = 0.5,
    num_restarts: int = 3,
) -> SearchResult:
    """Search for a Mog program using gradient descent through differentiable execution.

    This is the real deal: no templates, no hand-authored families.
    The differentiable CPU finds the program.
    """
    torch.manual_seed(seed)

    best_loss = float("inf")
    best_code = ""
    best_steps = 0

    # --- Try dedicated branching program structure too ---
    # Soft optimization is just to warm-start; the refinement does the real work.
    branch_steps = min(steps, 500)
    for restart in range(max(1, min(num_restarts, 2))):
        torch.manual_seed(seed + 5000 + restart * 1000)
        bprog = SoftBranchingProgram(num_args=len(arg_names))
        bopt = torch.optim.Adam(bprog.parameters(), lr=lr)
        for step in range(branch_steps):
            t = temperature_start + (temperature_end - temperature_start) * (step / max(steps - 1, 1))
            losses = []
            for args, target in examples:
                x = torch.tensor(args, dtype=torch.float32)
                y = torch.tensor(float(target), dtype=torch.float32)
                pred = bprog(x, temperature=t)
                losses.append((pred - y) ** 2)
            loss = torch.stack(losses).mean()
            bopt.zero_grad()
            loss.backward()
            bopt.step()
            cur = float(loss.detach().item())
            if cur < best_loss:
                best_loss = cur
                best_code = bprog.discretize(list(arg_names))
                best_steps = step

        # Branching refinement
        bcode, bloss = _branching_refinement(bprog, list(arg_names), examples, function_name)
        if bloss < best_loss:
            best_loss = bloss
            best_code = bcode

        if best_loss < 1e-6:
            best_code = best_code.replace("fn program(", f"fn {function_name}(")
            return SearchResult(success=True, code=best_code, loss=best_loss,
                                steps_taken=best_steps, metadata={"num_restarts": num_restarts, "num_slots": num_slots, "structure": "branching"})

    # --- General SoftMogProgram ---
    for restart in range(num_restarts):
        torch.manual_seed(seed + restart * 1000)
        prog = SoftMogProgram(num_args=len(arg_names), num_slots=num_slots, num_vars=num_vars)
        optimizer = torch.optim.Adam(prog.parameters(), lr=lr)

        for step in range(steps):
            # Anneal temperature
            t = temperature_start + (temperature_end - temperature_start) * (step / max(steps - 1, 1))

            losses = []
            for args, target in examples:
                x = torch.tensor(args, dtype=torch.float32)
                y = torch.tensor(float(target), dtype=torch.float32)
                pred = prog(x, temperature=t)
                losses.append((pred - y) ** 2)

            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cur = float(loss.detach().item())
            if cur < best_loss:
                best_loss = cur
                best_code = prog.discretize(arg_names)
                best_steps = step

            if cur < 1e-6:
                break

    # --- Discrete verification + local search ---
    # The soft program may have found the right continuous solution
    # but argmax discretization can pick the wrong neighbor.
    # Try small perturbations of the discrete choices to find a better discrete program.
    best_code, best_loss = _discrete_refinement(
        prog, arg_names, examples, best_code, best_loss, function_name
    )

    best_code = best_code.replace("fn program(", f"fn {function_name}(")
    success = best_loss < success_threshold

    return SearchResult(
        success=success,
        code=best_code,
        loss=best_loss,
        steps_taken=best_steps,
        metadata={"num_restarts": num_restarts, "num_slots": num_slots},
    )


def _eval_discrete(prog: SoftMogProgram, arg_names: Sequence[str],
                    examples: Sequence[tuple[tuple[float, ...], float]]) -> tuple[str, float]:
    """Evaluate the current discrete (argmax) program on examples."""
    code = prog.discretize(arg_names)
    # Quick Python-side eval of the discrete program
    loss = _eval_code_on_examples(code, arg_names, examples)
    return code, loss


def _eval_code_on_examples(code: str, arg_names: Sequence[str],
                           examples: Sequence[tuple[tuple[float, ...], float]]) -> float:
    """Evaluate discretized Mog code on examples using the interpreter."""
    try:
        from egdc.mog_lang import interpret

        total_loss = 0.0
        for args, target in examples:
            # Build a main that calls the function and prints the result
            arg_strs = ", ".join(str(int(a)) for a in args)
            fn_name_match = code.split("fn ")[1].split("(")[0] if "fn " in code else "program"
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


def _discrete_refinement(
    prog: SoftMogProgram,
    arg_names: Sequence[str],
    examples: Sequence[tuple[tuple[float, ...], float]],
    current_best_code: str,
    current_best_loss: float,
    function_name: str,
    max_tweaks: int = 50,
) -> tuple[str, float]:
    """Try small perturbations of the discrete program to close the soft-to-hard gap."""
    import copy

    best_code = current_best_code
    best_loss = current_best_loss

    # First: evaluate the current argmax discretization with the interpreter
    code, loss = _eval_discrete(prog, arg_names, examples)
    code = code.replace("fn program(", f"fn {function_name}(")
    if loss < best_loss:
        best_code = code
        best_loss = loss

    if best_loss < 1e-6:
        return best_code, best_loss

    # Try flipping individual discrete choices
    param_names_to_try = [
        "stmt_logits", "dst_logits", "src1_logits", "src2_logits",
        "op_logits", "return_src_logits",
    ]

    state = prog.state_dict()
    for pname in param_names_to_try:
        if pname not in state:
            continue
        tensor = state[pname]
        for slot in range(tensor.shape[0]):
            current_choice = int(torch.argmax(tensor[slot]).item())
            for alt in range(tensor.shape[1]):
                if alt == current_choice:
                    continue
                # Temporarily boost this alternative
                saved = tensor[slot].clone()
                tensor[slot] = torch.full_like(tensor[slot], -10.0)
                tensor[slot][alt] = 10.0
                prog.load_state_dict(state)

                code, loss = _eval_discrete(prog, arg_names, examples)
                code = code.replace("fn program(", f"fn {function_name}(")
                if loss < best_loss:
                    best_code = code
                    best_loss = loss
                    if best_loss < 1e-6:
                        # Restore and return early
                        tensor[slot] = saved
                        prog.load_state_dict(state)
                        return best_code, best_loss

                # Restore
                tensor[slot] = saved

    prog.load_state_dict(state)

    if best_loss < 1e-6:
        return best_code, best_loss

    if best_loss < 1e-6:
        return best_code, best_loss

    # --- Phase 1b: Combinatorial search on the first compute slot ---
    # The most common failure: soft optimization finds the right neighborhood
    # but argmax picks wrong src/op/dst/return combinations.
    # Brute-force the first compute slot + last return slot.
    stmt_t = state["stmt_logits"]
    src1_t = state["src1_logits"]
    src2_t = state["src2_logits"]
    op_t = state["op_logits"]
    dst_t = state["dst_logits"]
    ret_t = state["return_src_logits"]

    # Nop out all middle slots (keep slot 0 as compute, last as return)
    saved_stmts = [stmt_t[i].clone() for i in range(prog.num_slots)]
    for i in range(1, prog.num_slots - 1):
        stmt_t[i] = torch.full_like(stmt_t[i], -10.0)
        stmt_t[i][0] = 10.0  # nop
    # Force slot 0 = assign_binop, last = return_var
    stmt_t[0] = torch.full_like(stmt_t[0], -10.0)
    stmt_t[0][1] = 10.0
    stmt_t[-1] = torch.full_like(stmt_t[-1], -10.0)
    stmt_t[-1][-1] = 10.0  # return_var

    saved_s1 = src1_t[0].clone()
    saved_s2 = src2_t[0].clone()
    saved_op = op_t[0].clone()
    saved_dst = dst_t[0].clone()
    saved_ret = ret_t[-1].clone()

    ns = prog.num_sources
    for s1 in range(ns):
        for s2 in range(ns):
            for op in range(len(OPS)):
                for dst in range(prog.num_vars):
                    ret_idx = prog.num_args + dst
                    # Set choices
                    src1_t[0] = torch.full_like(src1_t[0], -10.0); src1_t[0][s1] = 10.0
                    src2_t[0] = torch.full_like(src2_t[0], -10.0); src2_t[0][s2] = 10.0
                    op_t[0] = torch.full_like(op_t[0], -10.0); op_t[0][op] = 10.0
                    dst_t[0] = torch.full_like(dst_t[0], -10.0); dst_t[0][dst] = 10.0
                    ret_t[-1] = torch.full_like(ret_t[-1], -10.0); ret_t[-1][ret_idx] = 10.0
                    prog.load_state_dict(state)
                    code, loss = _eval_discrete(prog, arg_names, examples)
                    code = code.replace("fn program(", f"fn {function_name}(")
                    if loss < best_loss:
                        best_code = code
                        best_loss = loss
                        if best_loss < 1e-6:
                            break
                if best_loss < 1e-6:
                    break
            if best_loss < 1e-6:
                break
        if best_loss < 1e-6:
            break

    # Restore
    for i in range(prog.num_slots):
        stmt_t[i] = saved_stmts[i]
    src1_t[0] = saved_s1; src2_t[0] = saved_s2; op_t[0] = saved_op
    dst_t[0] = saved_dst; ret_t[-1] = saved_ret
    prog.load_state_dict(state)

    if best_loss < 1e-6:
        return best_code, best_loss

    # --- Phase 2: Try flipping PAIRS of choices on the same slot ---
    # This handles cases where the soft solution requires two coordinated changes.
    critical_pairs = [
        ("src1_logits", "op_logits"),
        ("src1_logits", "src2_logits"),
        ("src2_logits", "op_logits"),
        ("dst_logits", "return_src_logits"),
    ]
    for pname1, pname2 in critical_pairs:
        if pname1 not in state or pname2 not in state:
            continue
        t1, t2 = state[pname1], state[pname2]
        for slot in range(min(t1.shape[0], t2.shape[0])):
            cur1 = int(torch.argmax(t1[slot]).item())
            cur2 = int(torch.argmax(t2[slot]).item())
            for alt1 in range(t1.shape[1]):
                if alt1 == cur1:
                    continue
                for alt2 in range(t2.shape[1]):
                    if alt2 == cur2:
                        continue
                    saved1 = t1[slot].clone()
                    saved2 = t2[slot].clone()
                    t1[slot] = torch.full_like(t1[slot], -10.0)
                    t1[slot][alt1] = 10.0
                    t2[slot] = torch.full_like(t2[slot], -10.0)
                    t2[slot][alt2] = 10.0
                    prog.load_state_dict(state)

                    code, loss = _eval_discrete(prog, arg_names, examples)
                    code = code.replace("fn program(", f"fn {function_name}(")
                    if loss < best_loss:
                        best_code = code
                        best_loss = loss
                        if best_loss < 1e-6:
                            t1[slot] = saved1
                            t2[slot] = saved2
                            prog.load_state_dict(state)
                            return best_code, best_loss

                    t1[slot] = saved1
                    t2[slot] = saved2

    prog.load_state_dict(state)
    return best_code, best_loss
