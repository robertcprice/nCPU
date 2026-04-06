"""Two-phase gradient-based branch solver.

Fixes the arm collapse problem by training in two phases:
Phase 1: Learn the branch CONDITION by gradient descent
Phase 2: Train each arm INDEPENDENTLY on its subset of examples

Also supports learnable constants (not fixed [0,1,-1,2,100]).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from egdc.mog_gumbel import gumbel_softmax, gumbel_read, gumbel_op, gumbel_cmp
from egdc.mog_soft_programs import OPS, CMP_OPS


@dataclass
class TwoPhaseResult:
    success: bool
    code: str
    loss: float
    metadata: dict[str, Any]


class SoftCondition(nn.Module):
    """Learnable branch condition: lhs CMP rhs."""

    def __init__(self, num_sources: int):
        super().__init__()
        self.lhs = nn.Parameter(torch.zeros(num_sources))
        self.rhs = nn.Parameter(torch.zeros(num_sources))
        self.cmp = nn.Parameter(torch.zeros(len(CMP_OPS)))

    def forward(self, storage: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        lhs = gumbel_read(storage, self.lhs, temperature)
        rhs = gumbel_read(storage, self.rhs, temperature)
        return gumbel_cmp(lhs, rhs, self.cmp, temperature)


class SoftArm(nn.Module):
    """Learnable expression: src1 OP src2 (with identity option)."""

    def __init__(self, num_sources: int):
        super().__init__()
        self.src1 = nn.Parameter(torch.zeros(num_sources))
        self.src2 = nn.Parameter(torch.zeros(num_sources))
        self.op = nn.Parameter(torch.zeros(len(OPS) + 1))  # +1 identity

    def forward(self, storage: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        s1 = gumbel_read(storage, self.src1, temperature)
        s2 = gumbel_read(storage, self.src2, temperature)
        safe_s2 = torch.where(torch.abs(s2) < 1e-6, torch.ones_like(s2), s2)
        ops = torch.stack([
            s1 + s2, s1 - s2, s1 * s2, s1 / safe_s2,
            torch.remainder(torch.round(s1), torch.clamp(torch.round(torch.abs(safe_s2)), min=1.0)),
            s1,  # identity
        ])
        w = gumbel_softmax(self.op, temperature)
        return (w * ops).sum()


def _make_storage(args: torch.Tensor, consts: torch.Tensor) -> torch.Tensor:
    return torch.cat([args, consts])


def _discretize_arm(arm: SoftArm, all_names: list[str]) -> str:
    s1 = all_names[int(torch.argmax(arm.src1).item())]
    s2 = all_names[int(torch.argmax(arm.src2).item())]
    oi = int(torch.argmax(arm.op).item())
    if oi >= len(OPS):
        return s1
    return f"{s1} {OPS[oi]} {s2}"


def _mine_constants(arg_names: list[str], examples) -> list[float]:
    """Extract candidate constants from I/O examples.

    Looks at: output values, input values, differences between inputs,
    and common thresholds. Returns a deduplicated sorted list.
    """
    consts = set()
    # Always include these
    consts.update([0.0, 1.0, -1.0, 2.0])

    for args, target in examples:
        # Output values are strong constant candidates
        consts.add(float(target))
        # Input values
        for a in args:
            consts.add(float(a))
        # Differences between pairs of inputs
        for i in range(len(args)):
            for j in range(i + 1, len(args)):
                consts.add(abs(float(args[i]) - float(args[j])))
        # Input-output differences
        for a in args:
            consts.add(abs(float(a) - float(target)))

    # Deduplicate, sort, limit to manageable size
    consts = sorted(consts)
    # Keep the most useful ones (small values + values from data)
    if len(consts) > 10:
        # Prioritize: 0, 1, -1, then values from outputs, then inputs
        outputs = set(float(t) for _, t in examples)
        priority = [c for c in consts if c in outputs or abs(c) <= 2]
        rest = [c for c in consts if c not in priority]
        consts = priority + rest[:10 - len(priority)]
    return consts[:10]


def two_phase_branch_solve(
    arg_names: list[str],
    examples: Sequence[tuple[tuple[float, ...], float]],
    function_name: str,
    num_consts: int = 6,
    phase1_steps: int = 800,
    phase2_steps: int = 600,
    lr: float = 0.03,
    num_restarts: int = 5,
    seed: int = 0,
) -> TwoPhaseResult:
    """Two-phase branch solver with data-mined constants."""

    # Mine constants from examples
    mined = _mine_constants(arg_names, examples)

    num_args = len(arg_names)
    num_consts = len(mined)
    ns = num_args + num_consts
    params_str = ", ".join(f"{a}: i64" for a in arg_names)

    best_loss = float("inf")
    best_code = ""

    # Generate diverse condition inits: try arg vs each mined constant, with each comparison
    init_conditions = []
    for lhs_idx in range(min(num_args, 2)):  # try each arg as LHS
        for rhs_idx in range(num_args, ns):  # try each constant as RHS
            for cmp_idx in [0, 1, 4]:  # >, <, ==
                init_conditions.append((lhs_idx, rhs_idx, cmp_idx))
    if num_args >= 2:
        # Also try arg vs arg
        init_conditions.append((0, 1, 0))  # a > b
        init_conditions.append((1, 0, 0))  # b > a

    for restart in range(num_restarts):
        torch.manual_seed(seed + restart * 1000)
        consts = nn.Parameter(torch.tensor(mined, dtype=torch.float32))

        cond = SoftCondition(ns)
        then_rough = SoftArm(ns)
        else_rough = SoftArm(ns)

        # Pick a specific condition init for this restart
        with torch.no_grad():
            if restart < len(init_conditions):
                li, ri, ci = init_conditions[restart]
                cond.lhs[li] = 3.0; cond.rhs[ri] = 3.0; cond.cmp[ci] = 3.0
            elif num_args >= 2:
                cond.lhs[0] = 2.0; cond.rhs[1] = 2.0; cond.cmp[0] = 2.0
            else:
                cond.lhs[0] = 2.0; cond.rhs[num_args] = 2.0; cond.cmp[0] = 2.0
            # Init arms differently
            if num_args >= 2:
                then_rough.src1[0] = 2.0; then_rough.src2[1] = 2.0; then_rough.op[1] = 1.0
                else_rough.src1[1] = 2.0; else_rough.src2[0] = 2.0; else_rough.op[1] = 1.0

        all_p1 = list(cond.parameters()) + list(then_rough.parameters()) + list(else_rough.parameters()) + [consts]
        cond_opt = torch.optim.Adam(all_p1, lr=lr)
        for step in range(phase1_steps):
            t = 2.0 + (0.1 - 2.0) * (step / max(phase1_steps - 1, 1))
            losses = []
            cond_probs = []
            for args, target in examples:
                x = torch.tensor(args, dtype=torch.float32)
                storage = _make_storage(x, consts)
                c = cond(storage, t)
                cond_probs.append(c)
                # Full if/else prediction
                pred = c * then_rough(storage, t) + (1.0 - c) * else_rough(storage, t)
                losses.append((pred - target) ** 2)
            loss = torch.stack(losses).mean()
            # Strong diversity: condition MUST split examples
            cp = torch.stack(cond_probs)
            diversity = -0.5 * (cp.mean() * (1.0 - cp.mean()))
            loss = loss + diversity
            cond_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_p1, 5.0)
            cond_opt.step()

        # Partition examples by the learned condition
        then_examples = []
        else_examples = []
        with torch.no_grad():
            for args, target in examples:
                x = torch.tensor(args, dtype=torch.float32)
                storage = _make_storage(x, consts)
                c = cond(storage, 0.01).item()
                if c > 0.5:
                    then_examples.append((args, target))
                else:
                    else_examples.append((args, target))

        if not then_examples or not else_examples:
            # Condition didn't split — try next restart
            continue

        # Phase 2: Train each arm independently on its subset
        then_arm = SoftArm(ns)
        else_arm = SoftArm(ns)

        # Init arms differently
        with torch.no_grad():
            if num_args >= 2:
                then_arm.src1[0] = 2.0; then_arm.src2[1] = 2.0; then_arm.op[1] = 1.0  # a-b
                else_arm.src1[1] = 2.0; else_arm.src2[0] = 2.0; else_arm.op[1] = 1.0  # b-a

        then_opt = torch.optim.Adam(list(then_arm.parameters()) + [consts], lr=lr)
        for step in range(phase2_steps):
            t = 1.5 + (0.05 - 1.5) * (step / max(phase2_steps - 1, 1))
            losses = []
            for args, target in then_examples:
                x = torch.tensor(args, dtype=torch.float32)
                storage = _make_storage(x, consts)
                pred = then_arm(storage, t)
                losses.append((pred - target) ** 2)
            if losses:
                loss = torch.stack(losses).mean()
                then_opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(then_arm.parameters(), 5.0)
                then_opt.step()

        else_opt = torch.optim.Adam(list(else_arm.parameters()) + [consts], lr=lr)
        for step in range(phase2_steps):
            t = 1.5 + (0.05 - 1.5) * (step / max(phase2_steps - 1, 1))
            losses = []
            for args, target in else_examples:
                x = torch.tensor(args, dtype=torch.float32)
                storage = _make_storage(x, consts)
                pred = else_arm(storage, t)
                losses.append((pred - target) ** 2)
            if losses:
                loss = torch.stack(losses).mean()
                else_opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(else_arm.parameters(), 5.0)
                else_opt.step()

        # Discretize
        const_vals = [int(round(c.item())) for c in consts]
        all_names = list(arg_names) + [str(c) for c in const_vals]

        lhs_name = all_names[int(torch.argmax(cond.lhs).item())]
        rhs_name = all_names[int(torch.argmax(cond.rhs).item())]
        cmp_op = CMP_OPS[int(torch.argmax(cond.cmp).item())]
        then_expr = _discretize_arm(then_arm, all_names)
        else_expr = _discretize_arm(else_arm, all_names)

        code = (
            f"fn {function_name}({params_str}) -> i64 {{\n"
            f"    if ({lhs_name} {cmp_op} {rhs_name}) {{\n"
            f"        return {then_expr};\n"
            f"    }} else {{\n"
            f"        return {else_expr};\n"
            f"    }}\n"
            f"}}\n"
        )

        # Evaluate discrete program
        from egdc.mog_program_search import _eval_code_on_examples
        dloss = _eval_code_on_examples(code, arg_names, examples)
        if dloss < best_loss:
            best_loss = dloss
            best_code = code
        if best_loss < 1e-6:
            break

    # Also try two sequential branches (for sign, clamp, safe_div patterns)
    if best_loss > 1e-6:
        code2, loss2 = _recursive_branch_solve(arg_names, examples, function_name,
                                                num_consts, phase1_steps, phase2_steps, lr, seed)
        if loss2 < best_loss:
            best_loss = loss2
            best_code = code2

    return TwoPhaseResult(best_loss < 2.0, best_code, best_loss, {})


def _recursive_branch_solve(arg_names, examples, function_name, num_consts,
                             steps, arm_steps, lr, seed, max_depth=2):
    """Recursive two-phase: learn one branch, then recurse on remaining examples."""
    from egdc.mog_program_search import _eval_code_on_examples

    mined = _mine_constants(arg_names, examples)
    num_args = len(arg_names)
    num_consts = len(mined)
    ns = num_args + num_consts
    params_str = ", ".join(f"{a}: i64" for a in arg_names)

    best_loss = float("inf")
    best_code = ""

    # Build init conditions for recursive solver too
    rec_inits = []
    for lhs_idx in range(min(num_args, 2)):
        for rhs_idx in range(num_args, ns):
            for cmp_idx in [0, 1, 4]:  # >, <, ==
                rec_inits.append((lhs_idx, rhs_idx, cmp_idx))

    num_rec_restarts = min(len(rec_inits), 15)
    for restart in range(num_rec_restarts):
        torch.manual_seed(seed + 5000 + restart * 1000)
        branches = []

        consts = nn.Parameter(torch.tensor(mined, dtype=torch.float32))
        remaining = list(examples)

        for depth in range(max_depth):
            if len(remaining) <= 1:
                break

            cond = SoftCondition(ns)
            then_arm = SoftArm(ns)
            else_arm = SoftArm(ns)

            # Pick init: systematic across restarts and depths
            with torch.no_grad():
                cond.lhs.zero_(); cond.rhs.zero_(); cond.cmp.zero_()
                idx = (restart * max_depth + depth) % max(len(rec_inits), 1)
                if idx < len(rec_inits):
                    li, ri, ci = rec_inits[idx]
                    cond.lhs[li] = 3.0; cond.rhs[ri] = 3.0; cond.cmp[ci] = 3.0
                else:
                    cond.lhs[0] = 2.0; cond.rhs[num_args] = 2.0; cond.cmp[0] = 2.0

            all_p = list(cond.parameters()) + list(then_arm.parameters()) + list(else_arm.parameters()) + [consts]
            opt = torch.optim.Adam(all_p, lr=lr)

            for step in range(steps):
                t = 2.0 + (0.1 - 2.0) * (step / max(steps - 1, 1))
                losses = []
                cond_probs = []
                for args, target in remaining:
                    x = torch.tensor(args, dtype=torch.float32)
                    storage = _make_storage(x, consts)
                    c = cond(storage, t)
                    cond_probs.append(c)
                    pred = c * then_arm(storage, t) + (1.0 - c) * else_arm(storage, t)
                    losses.append((pred - target) ** 2)
                loss = torch.stack(losses).mean()
                cp = torch.stack(cond_probs)
                diversity = -0.5 * (cp.mean() * (1.0 - cp.mean()))
                loss = loss + diversity
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(all_p, 5.0)
                opt.step()

            # Partition by learned condition
            then_ex = []
            else_ex = []
            with torch.no_grad():
                for args, target in remaining:
                    x = torch.tensor(args, dtype=torch.float32)
                    storage = _make_storage(x, consts)
                    if cond(storage, 0.01).item() > 0.5:
                        then_ex.append((args, target))
                    else:
                        else_ex.append((args, target))

            if not then_ex or not else_ex:
                break  # condition didn't split, stop recursing

            # Train then-arm independently on its subset
            then_opt = torch.optim.Adam(list(then_arm.parameters()) + [consts], lr=lr)
            for step in range(arm_steps):
                t = 1.5 + (0.05 - 1.5) * (step / max(arm_steps - 1, 1))
                losses = []
                for args, target in then_ex:
                    x = torch.tensor(args, dtype=torch.float32)
                    storage = _make_storage(x, consts)
                    losses.append((then_arm(storage, t) - target) ** 2)
                loss = torch.stack(losses).mean()
                then_opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(then_arm.parameters(), 5.0)
                then_opt.step()

            # Record this branch
            const_vals = [int(round(c.item())) for c in consts]
            all_names = list(arg_names) + [str(c) for c in const_vals]
            l = all_names[int(torch.argmax(cond.lhs).item())]
            r = all_names[int(torch.argmax(cond.rhs).item())]
            c_op = CMP_OPS[int(torch.argmax(cond.cmp).item())]
            e = _discretize_arm(then_arm, all_names)
            branches.append(f"    if ({l} {c_op} {r}) {{ return {e}; }}")

            remaining = else_ex

        # Default: train an arm on whatever's left
        if remaining:
            default_arm = SoftArm(ns)
            d_opt = torch.optim.Adam(list(default_arm.parameters()) + [consts], lr=lr)
            for step in range(arm_steps):
                t = 1.5 + (0.05 - 1.5) * (step / max(arm_steps - 1, 1))
                losses = []
                for args, target in remaining:
                    x = torch.tensor(args, dtype=torch.float32)
                    storage = _make_storage(x, consts)
                    losses.append((default_arm(storage, t) - target) ** 2)
                loss = torch.stack(losses).mean()
                d_opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(default_arm.parameters(), 5.0)
                d_opt.step()

            const_vals = [int(round(c.item())) for c in consts]
            all_names = list(arg_names) + [str(c) for c in const_vals]
            d_expr = _discretize_arm(default_arm, all_names)
            branches.append(f"    return {d_expr};")
        else:
            branches.append(f"    return 0;")

        code = f"fn {function_name}({params_str}) -> i64 {{\n" + "\n".join(branches) + "\n}\n"
        dloss = _eval_code_on_examples(code, arg_names, examples)
        if dloss < best_loss:
            best_loss = dloss
            best_code = code
        if best_loss < 1e-6:
            break

    return best_code, best_loss
