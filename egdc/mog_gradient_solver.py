"""Gradient-first program solver.

This replaces ALL discrete enumeration with gradient descent.
Every structural choice uses Gumbel-softmax straight-through estimators
that anneal from soft (exploration) to hard (discrete program).

The solver trains multiple differentiable program structures in parallel
and picks the one with lowest loss after annealing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from egdc.mog_gumbel import gumbel_softmax, gumbel_read, gumbel_op, gumbel_cmp
from egdc.mog_soft_programs import (
    OPS, CMP_OPS,
    SoftLoopProgram, SoftMultiBranchProgram, SoftInteractiveProgram,
)


# ---------------------------------------------------------------------------
# Gumbel-annealed versions of all program structures
# ---------------------------------------------------------------------------

class GumbelArithmeticProgram(nn.Module):
    """return src1 OP src2, all Gumbel-discretized."""

    def __init__(self, num_args: int, num_consts: int = 5):
        super().__init__()
        self.num_args = num_args
        self.num_sources = num_args + num_consts
        self.const_values = nn.Parameter(torch.tensor([0.0, 1.0, -1.0, 2.0, 100.0]))
        self.src1_logits = nn.Parameter(torch.zeros(self.num_sources))
        self.src2_logits = nn.Parameter(torch.zeros(self.num_sources))
        self.op_logits = nn.Parameter(torch.zeros(len(OPS)))

        with torch.no_grad():
            if num_args > 0: self.src1_logits[0] = 1.0
            if num_args > 1: self.src2_logits[1] = 1.0
            self.op_logits[0] = 1.0  # +

    def forward(self, args: torch.Tensor, temperature: float = 1.0,
                hard: bool = False) -> torch.Tensor:
        storage = torch.cat([args, self.const_values])
        s1 = gumbel_read(storage, self.src1_logits, temperature, hard)
        s2 = gumbel_read(storage, self.src2_logits, temperature, hard)
        return gumbel_op(s1, s2, self.op_logits, temperature, hard)


class GumbelBranchProgram(nn.Module):
    """if (lhs CMP rhs) return then_expr else return else_expr, Gumbel-discretized."""

    def __init__(self, num_args: int, num_consts: int = 5):
        super().__init__()
        ns = num_args + num_consts
        self.num_args = num_args
        self.ns = ns
        self.const_values = nn.Parameter(torch.tensor([0.0, 1.0, -1.0, 2.0, 100.0]))
        # Condition
        self.cmp_logits = nn.Parameter(torch.zeros(len(CMP_OPS)))
        self.lhs_logits = nn.Parameter(torch.zeros(ns))
        self.rhs_logits = nn.Parameter(torch.zeros(ns))
        # Then arm: src1 OP src2
        self.then_s1 = nn.Parameter(torch.zeros(ns))
        self.then_s2 = nn.Parameter(torch.zeros(ns))
        self.then_op = nn.Parameter(torch.zeros(len(OPS) + 1))  # +1 identity
        # Else arm
        self.else_s1 = nn.Parameter(torch.zeros(ns))
        self.else_s2 = nn.Parameter(torch.zeros(ns))
        self.else_op = nn.Parameter(torch.zeros(len(OPS) + 1))

        with torch.no_grad():
            if num_args >= 2:
                self.lhs_logits[0] = 1.0; self.rhs_logits[1] = 1.0
                # Key: init then and else arms DIFFERENTLY
                self.then_s1[0] = 1.0; self.then_s2[1] = 1.0; self.then_op[1] = 1.0  # a-b
                self.else_s1[1] = 1.0; self.else_s2[0] = 1.0; self.else_op[1] = 1.0  # b-a
            self.cmp_logits[0] = 1.0

    def _eval_arm(self, storage, s1_logits, s2_logits, op_logits, temperature, hard):
        s1 = gumbel_read(storage, s1_logits, temperature, hard)
        s2 = gumbel_read(storage, s2_logits, temperature, hard)
        safe_s2 = torch.where(torch.abs(s2) < 1e-6, torch.ones_like(s2), s2)
        ops = torch.stack([
            s1 + s2, s1 - s2, s1 * s2, s1 / safe_s2,
            torch.remainder(torch.round(s1), torch.clamp(torch.round(torch.abs(safe_s2)), min=1.0)),
            s1,  # identity
        ])
        w = gumbel_softmax(op_logits, temperature, hard)
        return (w * ops).sum()

    def forward(self, args: torch.Tensor, temperature: float = 1.0,
                hard: bool = False) -> torch.Tensor:
        storage = torch.cat([args, self.const_values])
        lhs = gumbel_read(storage, self.lhs_logits, temperature, hard)
        rhs = gumbel_read(storage, self.rhs_logits, temperature, hard)
        cond = gumbel_cmp(lhs, rhs, self.cmp_logits, temperature, hard)
        then_val = self._eval_arm(storage, self.then_s1, self.then_s2, self.then_op, temperature, hard)
        else_val = self._eval_arm(storage, self.else_s1, self.else_s2, self.else_op, temperature, hard)
        return cond * then_val + (1.0 - cond) * else_val


class GumbelMultiBranchProgram(nn.Module):
    """Sequential if-return branches with Gumbel discretization."""

    def __init__(self, num_args: int, num_branches: int = 3, num_consts: int = 5):
        super().__init__()
        ns = num_args + num_consts
        self.ns = ns
        self.num_args = num_args
        self.num_branches = num_branches
        self.const_values = nn.Parameter(torch.tensor([0.0, 1.0, -1.0, 2.0, 100.0]))

        self.cmp_logits = nn.ParameterList([nn.Parameter(torch.zeros(len(CMP_OPS))) for _ in range(num_branches)])
        self.lhs_logits = nn.ParameterList([nn.Parameter(torch.zeros(ns)) for _ in range(num_branches)])
        self.rhs_logits = nn.ParameterList([nn.Parameter(torch.zeros(ns)) for _ in range(num_branches)])
        self.ret_logits = nn.ParameterList([nn.Parameter(torch.zeros(ns)) for _ in range(num_branches)])
        self.default_logits = nn.Parameter(torch.zeros(ns))

        with torch.no_grad():
            if num_args > 0:
                self.lhs_logits[0][0] = 1.0
                self.rhs_logits[0][num_args] = 1.0  # const 0
            self.cmp_logits[0][0] = 1.0  # >

    def forward(self, args: torch.Tensor, temperature: float = 1.0,
                hard: bool = False) -> torch.Tensor:
        storage = torch.cat([args, self.const_values])
        ret_val = torch.tensor(0.0)
        remaining = torch.tensor(1.0)

        for b in range(self.num_branches):
            lhs = gumbel_read(storage, self.lhs_logits[b], temperature, hard)
            rhs = gumbel_read(storage, self.rhs_logits[b], temperature, hard)
            cond = gumbel_cmp(lhs, rhs, self.cmp_logits[b], temperature, hard)
            val = gumbel_read(storage, self.ret_logits[b], temperature, hard)
            fire = cond * remaining
            ret_val = ret_val + fire * val
            remaining = remaining * (1.0 - cond)

        default = gumbel_read(storage, self.default_logits, temperature, hard)
        ret_val = ret_val + remaining * default
        return ret_val


# ---------------------------------------------------------------------------
# Unified gradient solver
# ---------------------------------------------------------------------------

@dataclass
class GradientSolveResult:
    success: bool
    code: str
    loss: float
    structure: str
    metadata: dict[str, Any]


def _train_gumbel(prog: nn.Module, examples, steps: int, lr: float,
                  seed: int, num_restarts: int) -> tuple[nn.Module, float]:
    """Train with Gumbel-softmax annealing. Returns best program and loss."""
    best_loss = float("inf")
    best_state = None

    def _make_fresh(template):
        if isinstance(template, GumbelArithmeticProgram):
            return GumbelArithmeticProgram(template.num_args)
        if isinstance(template, GumbelBranchProgram):
            return GumbelBranchProgram(template.num_args)
        if isinstance(template, GumbelMultiBranchProgram):
            return GumbelMultiBranchProgram(template.num_args, template.num_branches)
        if isinstance(template, SoftLoopProgram):
            return SoftLoopProgram(template.num_args)
        if isinstance(template, SoftMultiBranchProgram):
            return SoftMultiBranchProgram(template.num_args, template.num_branches)
        if isinstance(template, SoftInteractiveProgram):
            return SoftInteractiveProgram()
        return template

    for restart in range(num_restarts):
        torch.manual_seed(seed + restart * 1000)
        p = _make_fresh(prog)

        opt = torch.optim.Adam(p.parameters(), lr=lr)
        for step in range(steps):
            frac = step / max(steps - 1, 1)
            temp = 2.0 * (1.0 - frac) + 0.05 * frac
            # Use straight-through in last 20% of training
            hard = frac > 0.8

            losses = []
            cond_values = []
            for args, target in examples:
                x = torch.tensor(args, dtype=torch.float32)
                y = torch.tensor(float(target), dtype=torch.float32)
                if hasattr(p, 'forward') and 'hard' in p.forward.__code__.co_varnames:
                    pred = p(x, temperature=temp, hard=hard)
                else:
                    pred = p(x, temperature=temp)
                losses.append((pred - y) ** 2)
                # Track condition values for branch programs
                if isinstance(p, (GumbelBranchProgram,)) and hasattr(p, 'lhs_logits'):
                    storage = torch.cat([x, p.const_values])
                    from egdc.mog_gumbel import gumbel_read, gumbel_cmp
                    lhs = gumbel_read(storage, p.lhs_logits, temp, False)
                    rhs = gumbel_read(storage, p.rhs_logits, temp, False)
                    cond = gumbel_cmp(lhs, rhs, p.cmp_logits, temp, False)
                    cond_values.append(cond)
            loss = torch.stack(losses).mean()
            # Regularize: encourage condition to be neither always-true nor always-false
            if cond_values and frac < 0.7:
                cond_t = torch.stack(cond_values)
                mean_cond = cond_t.mean()
                diversity_penalty = -0.1 * (mean_cond * (1.0 - mean_cond))
                loss = loss + diversity_penalty
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(p.parameters(), 5.0)
            opt.step()

            cur = float(loss.item())
            if cur < best_loss:
                best_loss = cur
                best_state = {k: v.detach().clone() for k, v in p.state_dict().items()}

        if best_loss < 1e-6:
            break

    if best_state is not None:
        prog.load_state_dict(best_state)
    return prog, best_loss


def gradient_solve(
    arg_names: Sequence[str],
    examples: Sequence[tuple[tuple[float, ...], float]],
    function_name: str = "program",
    steps: int = 1500,
    lr: float = 0.03,
    num_restarts: int = 3,
    seed: int = 42,
) -> GradientSolveResult:
    """Solve a program synthesis problem using ONLY gradient descent.

    Trains multiple differentiable program structures in parallel
    and picks the one with lowest loss.
    """
    n = len(arg_names)

    # All candidate structures — trained by gradients, not enumeration.
    # Train all in parallel (conceptually), pick the one whose DISCRETE program
    # scores best, not the one with lowest soft loss.
    # Fast structures first, expensive loop last
    candidates = [
        ("arithmetic", GumbelArithmeticProgram(n), min(steps, 600), lr, min(num_restarts, 2)),
        ("branch", GumbelBranchProgram(n), min(steps, 1200), lr, min(num_restarts, 3)),
        ("multi_branch", GumbelMultiBranchProgram(n, num_branches=3), min(steps, 1200), lr, min(num_restarts, 3)),
        ("loop", SoftLoopProgram(n), min(steps, 1500), 0.03, max(1, num_restarts // 2)),
    ]

    best_loss = float("inf")
    best_code = ""
    best_structure = ""

    from egdc.mog_program_search import _eval_code_on_examples

    for structure_name, prog, s, l, nr in candidates:
        trained, soft_loss = _train_gumbel(prog, examples, s, l, seed, nr)
        code = _discretize(trained, arg_names, function_name)
        # Evaluate the DISCRETE program, not the soft loss
        discrete_loss = _eval_code_on_examples(code, list(arg_names), examples)
        if discrete_loss < best_loss:
            best_loss = discrete_loss
            best_structure = structure_name
            best_code = code
        # Only stop if discrete loss is perfect
        if best_loss < 1e-6:
            break

    return GradientSolveResult(
        success=best_loss < 2.0,
        code=best_code,
        loss=best_loss,
        structure=best_structure,
        metadata={"steps": steps, "num_restarts": num_restarts},
    )


def _discretize(prog: nn.Module, arg_names: Sequence[str], fn_name: str) -> str:
    """Convert a trained soft program to Mog source code via argmax."""
    params_str = ", ".join(f"{a}: i64" for a in arg_names)
    all_names = list(arg_names) + ["0", "1", "-1", "2", "100"]

    if isinstance(prog, GumbelArithmeticProgram):
        s1 = all_names[int(torch.argmax(prog.src1_logits).item())]
        s2 = all_names[int(torch.argmax(prog.src2_logits).item())]
        op = OPS[int(torch.argmax(prog.op_logits).item())]
        return f"fn {fn_name}({params_str}) -> i64 {{\n    return {s1} {op} {s2};\n}}\n"

    if isinstance(prog, GumbelBranchProgram):
        lhs = all_names[int(torch.argmax(prog.lhs_logits).item())]
        rhs = all_names[int(torch.argmax(prog.rhs_logits).item())]
        cmp = CMP_OPS[int(torch.argmax(prog.cmp_logits).item())]

        def arm_str(s1l, s2l, opl):
            s1 = all_names[int(torch.argmax(s1l).item())]
            s2 = all_names[int(torch.argmax(s2l).item())]
            oi = int(torch.argmax(opl).item())
            if oi >= len(OPS):
                return s1
            return f"{s1} {OPS[oi]} {s2}"

        then_e = arm_str(prog.then_s1, prog.then_s2, prog.then_op)
        else_e = arm_str(prog.else_s1, prog.else_s2, prog.else_op)
        return (
            f"fn {fn_name}({params_str}) -> i64 {{\n"
            f"    if ({lhs} {cmp} {rhs}) {{\n"
            f"        return {then_e};\n"
            f"    }} else {{\n"
            f"        return {else_e};\n"
            f"    }}\n"
            f"}}\n"
        )

    if isinstance(prog, GumbelMultiBranchProgram):
        lines = []
        for b in range(prog.num_branches):
            lhs = all_names[int(torch.argmax(prog.lhs_logits[b]).item())]
            rhs = all_names[int(torch.argmax(prog.rhs_logits[b]).item())]
            cmp = CMP_OPS[int(torch.argmax(prog.cmp_logits[b]).item())]
            ret = all_names[int(torch.argmax(prog.ret_logits[b]).item())]
            lines.append(f"    if ({lhs} {cmp} {rhs}) {{ return {ret}; }}")
        default = all_names[int(torch.argmax(prog.default_logits).item())]
        lines.append(f"    return {default};")
        body = "\n".join(lines)
        return f"fn {fn_name}({params_str}) -> i64 {{\n{body}\n}}\n"

    if isinstance(prog, SoftLoopProgram):
        # Discretize the loop
        all_names_l = list(arg_names) + [f"c{i}" for i in range(prog.num_vars)]
        # Bound
        bound_idx = int(torch.argmax(prog.bound_src).item())
        bound_name = all_names_l[bound_idx] if bound_idx < len(all_names_l) else "0"
        offset = int(round(prog.bound_offset.item()))
        bound_expr = f"{bound_name} + {offset}" if offset != 0 else bound_name
        # Body op
        body_op_idx = int(torch.argmax(prog.body_acc_op).item())
        body_op = OPS[body_op_idx] if body_op_idx < len(OPS) else "+"
        # Body i-expression
        i_expr_idx = int(torch.argmax(prog.body_i_expr_weights).item())
        i_exprs = ["i", "i * i", "1", str(int(round(prog.const_values[0].item())))]
        i_expr = i_exprs[i_expr_idx] if i_expr_idx < len(i_exprs) else "i"
        # Init
        init_idx = int(torch.argmax(prog.init_logits).item())
        init_val = all_names_l[init_idx] if init_idx < len(all_names_l) else "0"
        # Generate code
        return (
            f"fn {fn_name}({params_str}) -> i64 {{\n"
            f"    acc: i64 = {init_val};\n"
            f"    i: i64 = 0;\n"
            f"    while i < ({bound_expr}) {{\n"
            f"        acc = acc {body_op} ({i_expr});\n"
            f"        i = i + 1;\n"
            f"    }}\n"
            f"    return acc;\n"
            f"}}\n"
        )

    # Fallback: return placeholder
    return f"fn {fn_name}({params_str}) -> i64 {{ return 0; }}\n"
