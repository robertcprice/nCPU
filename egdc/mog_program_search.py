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

    # Rename function
    best_code = best_code.replace("fn program(", f"fn {function_name}(")
    success = best_loss < success_threshold

    return SearchResult(
        success=success,
        code=best_code,
        loss=best_loss,
        steps_taken=best_steps,
        metadata={"num_restarts": num_restarts, "num_slots": num_slots},
    )
