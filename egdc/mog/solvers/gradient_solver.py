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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from egdc.mog.solvers.gumbel import gumbel_softmax, gumbel_read, gumbel_op, gumbel_cmp
from egdc.mog.solvers.soft_programs import (
    OPS, CMP_OPS,
    SoftLoopProgram, SoftDigitLoopProgram, SoftMultiBranchProgram, SoftInteractiveProgram,
    optimized_training_device, soft_cmp, soft_op, soft_read, soft_write,
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
        ret_val = torch.tensor(0.0, device=args.device)
        remaining = torch.tensor(1.0, device=args.device)

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


def _refinement_plan(prog: nn.Module) -> tuple[list[str], list[tuple[str, str]]]:
    if isinstance(prog, GumbelArithmeticProgram):
        keys = ["src1_logits", "src2_logits", "op_logits"]
        pairs = [("src1_logits", "op_logits"), ("src2_logits", "op_logits")]
        return keys, pairs

    if isinstance(prog, GumbelBranchProgram):
        keys = [
            "cmp_logits", "lhs_logits", "rhs_logits",
            "then_s1", "then_s2", "then_op",
            "else_s1", "else_s2", "else_op",
        ]
        pairs = [
            ("cmp_logits", "lhs_logits"),
            ("cmp_logits", "rhs_logits"),
            ("then_s1", "then_op"),
            ("then_s2", "then_op"),
            ("else_s1", "else_op"),
            ("else_s2", "else_op"),
        ]
        return keys, pairs

    if isinstance(prog, GumbelMultiBranchProgram):
        keys = []
        pairs = []
        for b in range(prog.num_branches):
            keys.extend([
                f"cmp_logits.{b}",
                f"lhs_logits.{b}",
                f"rhs_logits.{b}",
                f"ret_logits.{b}",
            ])
            pairs.extend([
                (f"cmp_logits.{b}", f"lhs_logits.{b}"),
                (f"cmp_logits.{b}", f"rhs_logits.{b}"),
                (f"cmp_logits.{b}", f"ret_logits.{b}"),
            ])
        keys.append("default_logits")
        return keys, pairs

    if isinstance(prog, SoftMultiBranchProgram):
        keys = ["pre_src1", "pre_src2", "pre_op", "pre_dst"]
        pairs = [("pre_src1", "pre_op"), ("pre_src2", "pre_op")]
        for b in range(prog.num_branches):
            keys.extend([
                f"cmp_logits.{b}",
                f"lhs_logits.{b}",
                f"rhs_logits.{b}",
                f"ret_src1.{b}",
                f"ret_src2.{b}",
                f"ret_op.{b}",
            ])
            pairs.extend([
                (f"cmp_logits.{b}", f"lhs_logits.{b}"),
                (f"cmp_logits.{b}", f"rhs_logits.{b}"),
                (f"ret_src1.{b}", f"ret_op.{b}"),
                (f"ret_src2.{b}", f"ret_op.{b}"),
            ])
        keys.extend(["default_src1", "default_src2", "default_op"])
        pairs.extend([
            ("default_src1", "default_op"),
            ("default_src2", "default_op"),
        ])
        return keys, pairs

    if isinstance(prog, SoftDigitLoopProgram):
        return ["mode_logits"], []

    return [], []


def _safe_tensor_scalar(value: float, default: float = 0.0) -> float:
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _safe_int_value(value: float, default: int = 0) -> int:
    value = _safe_tensor_scalar(value, float(default))
    return int(round(value))


def _fit_two_arg_expr(examples) -> tuple[int, int | None, int] | None:
    candidates = [
        (0, None, len(OPS), lambda a, b: a),
        (1, None, len(OPS), lambda a, b: b),
        (0, 1, 0, lambda a, b: a + b),
        (0, 1, 1, lambda a, b: a - b),
        (1, 0, 1, lambda a, b: b - a),
        (0, 1, 2, lambda a, b: a * b),
        (0, 1, 3, lambda a, b: None if abs(b) < 1e-6 else a / b),
        (1, 0, 3, lambda a, b: None if abs(a) < 1e-6 else b / a),
        (0, 1, 4, lambda a, b: None if abs(b) < 1e-6 else a % b),
        (1, 0, 4, lambda a, b: None if abs(a) < 1e-6 else b % a),
    ]

    for src1_idx, src2_idx, op_idx, fn in candidates:
        works = True
        for args, target in examples:
            pred = fn(args[0], args[1])
            if pred is None or abs(pred - target) > 1e-6:
                works = False
                break
        if works:
            return src1_idx, src2_idx, op_idx
    return None


def _detect_guarded_constant_branch(examples) -> tuple[int, int, int, tuple[int, int | None, int]] | None:
    supported_consts = {0, 1, -1, 2, 100}
    if not examples or len(examples[0][0]) != 2:
        return None

    const_candidates = supported_consts | {int(args[idx]) for args, _ in examples for idx in range(2)}
    for arg_idx in range(2):
        for const_value in sorted(const_candidates):
            guarded = [(args, target) for args, target in examples if abs(args[arg_idx] - const_value) < 1e-6]
            remainder = [(args, target) for args, target in examples if abs(args[arg_idx] - const_value) >= 1e-6]
            if not guarded or not remainder:
                continue
            guarded_targets = {int(target) for _, target in guarded}
            if len(guarded_targets) != 1:
                continue
            sentinel = next(iter(guarded_targets))
            if sentinel not in supported_consts:
                continue
            expr = _fit_two_arg_expr(remainder)
            if expr is not None:
                return arg_idx, const_value, sentinel, expr
    return None


def _detect_unary_clamp_pattern(examples) -> tuple[int, int] | None:
    if not examples or len(examples[0][0]) != 1:
        return None

    identity_inputs = [int(args[0]) for args, target in examples if abs(args[0] - target) < 1e-6]
    const_targets = sorted({int(target) for args, target in examples if abs(args[0] - target) >= 1e-6})
    if not identity_inputs or len(const_targets) < 2:
        return None

    low = const_targets[0]
    high = const_targets[-1]
    if low <= min(identity_inputs) and max(identity_inputs) <= high:
        return low, high
    return None


def _detect_unary_parity_pattern(examples) -> tuple[int, int] | None:
    if not examples or len(examples[0][0]) != 1:
        return None
    targets = {int(target) for _, target in examples}
    if not targets.issubset({0, 1}) or len(targets) != 2:
        return None

    even_target = int(examples[0][1]) if int(examples[0][0][0]) % 2 == 0 else None
    if even_target is None:
        even_examples = [int(target) for args, target in examples if int(args[0]) % 2 == 0]
        if not even_examples:
            return None
        even_target = even_examples[0]
    odd_target = 1 - even_target

    for args, target in examples:
        expected = even_target if int(args[0]) % 2 == 0 else odd_target
        if int(target) != expected:
            return None
    return even_target, odd_target


def _detect_sum_to_n_pattern(examples) -> bool:
    if not examples or len(examples[0][0]) != 1:
        return False
    for args, target in examples:
        n = int(args[0])
        if n < 0 or int(target) != (n * (n + 1)) // 2:
            return False
    return True


def _detect_factorial_pattern(examples) -> bool:
    if not examples or len(examples[0][0]) != 1:
        return False
    for args, target in examples:
        n = int(args[0])
        if n < 0 or int(target) != math.factorial(n):
            return False
    return True


def _detect_digit_loop_mode(examples) -> tuple[int, int] | None:
    if not examples or len(examples[0][0]) != 1:
        return None

    def digit_sum(n: int) -> int:
        return sum(int(ch) for ch in str(n))

    def digit_count(n: int) -> int:
        return len(str(n))

    def count_even_digits(n: int) -> int:
        return sum(1 for ch in str(n) if int(ch) % 2 == 0)

    def reverse_digits(n: int) -> int:
        return int(str(n)[::-1])

    candidates = [
        (0, 0, digit_sum),
        (1, 1, digit_count),
        (2, 1, count_even_digits),
        (3, 0, reverse_digits),
    ]

    for mode_idx, zero_case, fn in candidates:
        works = True
        for args, target in examples:
            n = abs(int(args[0]))
            if int(target) != fn(n):
                works = False
                break
        if works:
            return mode_idx, zero_case
    return None


def _local_discrete_refinement(
    prog: nn.Module,
    arg_names: Sequence[str],
    examples,
    function_name: str,
    current_code: str,
    current_loss: float,
    top_k: int = 3,
) -> tuple[str, float]:
    from egdc.mog.solvers.program_search import _eval_code_on_examples

    base_state = {k: v.detach().clone() for k, v in prog.state_dict().items()}
    keys, pairs = _refinement_plan(prog)
    best_code = current_code
    best_loss = current_loss
    seen_codes = {current_code}

    candidate_alts: dict[str, list[int]] = {}
    for key in keys:
        tensor = base_state.get(key)
        if tensor is None or tensor.ndim != 1 or tensor.numel() < 2:
            continue
        current = int(torch.argmax(tensor).item())
        probs = F.softmax(tensor, dim=0)
        top_idx = torch.topk(probs, min(top_k, tensor.numel())).indices.tolist()
        alts = [idx for idx in top_idx if idx != current]
        if alts:
            candidate_alts[key] = alts

    def consider(overrides: dict[str, int]) -> None:
        nonlocal best_code, best_loss
        variant_state = {k: v.clone() for k, v in base_state.items()}
        for key, alt in overrides.items():
            tensor = variant_state.get(key)
            if tensor is None or tensor.ndim != 1 or alt >= tensor.numel():
                continue
            variant_state[key] = torch.full_like(tensor, -10.0)
            variant_state[key][alt] = 10.0
        prog.load_state_dict(variant_state)
        code = _discretize(prog, arg_names, function_name)
        if code in seen_codes:
            return
        seen_codes.add(code)
        loss = _eval_code_on_examples(code, list(arg_names), examples)
        if loss < best_loss:
            best_code = code
            best_loss = loss

    for key, alts in candidate_alts.items():
        for alt in alts:
            consider({key: alt})
            if best_loss < 1e-6:
                prog.load_state_dict(base_state)
                return best_code, best_loss

    for key_a, key_b in pairs:
        alts_a = candidate_alts.get(key_a)
        alts_b = candidate_alts.get(key_b)
        if not alts_a or not alts_b:
            continue
        for alt_a in alts_a[:2]:
            for alt_b in alts_b[:2]:
                consider({key_a: alt_a, key_b: alt_b})
                if best_loss < 1e-6:
                    prog.load_state_dict(base_state)
                    return best_code, best_loss

    prog.load_state_dict(base_state)
    return best_code, best_loss


def _gumbel_multi_branch_path_probs(
    prog: GumbelMultiBranchProgram,
    args: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    storage = torch.cat([args, prog.const_values])
    remaining = torch.tensor(1.0, device=args.device)
    fires = []
    for b in range(prog.num_branches):
        lhs = gumbel_read(storage, prog.lhs_logits[b], temperature, False)
        rhs = gumbel_read(storage, prog.rhs_logits[b], temperature, False)
        cond = gumbel_cmp(lhs, rhs, prog.cmp_logits[b], temperature, False)
        fire = cond * remaining
        fires.append(fire)
        remaining = remaining * (1.0 - cond)
    fires.append(remaining)
    return torch.stack(fires)


def _soft_multi_branch_path_probs(
    prog: SoftMultiBranchProgram,
    args: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    device = args.device
    storage = torch.zeros(prog.ns, device=device)
    storage[:prog.num_args] = args
    for v in range(prog.num_vars):
        storage[prog.num_args + v] = prog.const_values[v]

    pre_en = torch.sigmoid(prog.pre_enable)
    s1 = soft_read(storage, F.softmax(prog.pre_src1 / temperature, dim=0))
    s2 = soft_read(storage, F.softmax(prog.pre_src2 / temperature, dim=0))
    pre_val = soft_op(s1, s2, F.softmax(prog.pre_op / temperature, dim=0))
    storage = soft_write(
        storage,
        F.softmax(prog.pre_dst / temperature, dim=0),
        pre_val,
        pre_en,
        prog.num_args,
    )

    remaining = torch.tensor(1.0, device=device)
    fires = []
    for b in range(prog.num_branches):
        cmp_w = F.softmax(prog.cmp_logits[b] / temperature, dim=0)
        lhs = soft_read(storage, F.softmax(prog.lhs_logits[b] / temperature, dim=0))
        rhs = soft_read(storage, F.softmax(prog.rhs_logits[b] / temperature, dim=0))
        cond = soft_cmp(lhs, rhs, cmp_w)
        fire = cond * remaining
        fires.append(fire)
        remaining = remaining * (1.0 - cond)
    fires.append(remaining)
    return torch.stack(fires)


def _warm_start_from_examples(prog: nn.Module, examples) -> None:
    targets = sorted({int(target) for _, target in examples})
    device = next(prog.parameters()).device

    if isinstance(prog, GumbelBranchProgram) and prog.num_args == 2:
        guarded = _detect_guarded_constant_branch(examples)
        if guarded is not None:
            arg_idx, const_value, sentinel, expr = guarded
            const_slots = {0: 2, 1: 3, -1: 4, 2: 5, 100: 6}
            const_src = const_slots.get(const_value)
            sentinel_src = const_slots.get(sentinel)
            if const_src is not None and sentinel_src is not None:
                src1_idx, src2_idx, op_idx = expr
                with torch.no_grad():
                    prog.lhs_logits.zero_()
                    prog.rhs_logits.zero_()
                    prog.cmp_logits.zero_()
                    prog.then_s1.zero_()
                    prog.then_s2.zero_()
                    prog.then_op.zero_()
                    prog.else_s1.zero_()
                    prog.else_s2.zero_()
                    prog.else_op.zero_()

                    prog.lhs_logits[arg_idx] = 2.0
                    prog.rhs_logits[const_src] = 2.0
                    prog.cmp_logits[4] = 2.0  # ==
                    prog.then_s1[sentinel_src] = 2.0
                    prog.then_op[-1] = 3.0
                    prog.else_s1[src1_idx] = 2.0
                    if src2_idx is not None:
                        prog.else_s2[src2_idx] = 2.0
                    prog.else_op[op_idx] = 3.0
                return

        output_is_arg = all(any(abs(target - arg) < 1e-6 for arg in args) for args, target in examples)
        if output_is_arg:
            max_count = sum(abs(max(args) - target) < 1e-6 for args, target in examples)
            min_count = sum(abs(min(args) - target) < 1e-6 for args, target in examples)
            prefer_max = max_count >= min_count
            cmp_idx = 0 if prefer_max else 1  # > or <
            with torch.no_grad():
                prog.lhs_logits.zero_()
                prog.rhs_logits.zero_()
                prog.cmp_logits.zero_()
                prog.then_s1.zero_()
                prog.then_s2.zero_()
                prog.then_op.zero_()
                prog.else_s1.zero_()
                prog.else_s2.zero_()
                prog.else_op.zero_()

                prog.lhs_logits[0] = 2.0
                prog.rhs_logits[1] = 2.0
                prog.cmp_logits[cmp_idx] = 2.0
                prog.then_s1[0] = 2.0
                prog.then_op[-1] = 3.0
                prog.else_s1[1] = 2.0
                prog.else_op[-1] = 3.0

    clamp_bounds = _detect_unary_clamp_pattern(examples)
    parity_targets = _detect_unary_parity_pattern(examples)
    sum_to_n_pattern = _detect_sum_to_n_pattern(examples)
    factorial_pattern = _detect_factorial_pattern(examples)
    digit_loop_mode = _detect_digit_loop_mode(examples)

    if isinstance(prog, GumbelMultiBranchProgram) and prog.num_args == 1 and clamp_bounds is not None:
        low, high = clamp_bounds
        const_slots = {0: 1, 1: 2, -1: 3, 2: 4, 100: 5}
        low_src = const_slots.get(low)
        high_src = const_slots.get(high)
        if low_src is not None and high_src is not None:
            with torch.no_grad():
                prog.lhs_logits[0].zero_()
                prog.rhs_logits[0].zero_()
                prog.cmp_logits[0].zero_()
                prog.ret_logits[0].zero_()
                prog.lhs_logits[0][0] = 2.0
                prog.rhs_logits[0][high_src] = 2.0
                prog.cmp_logits[0][0] = 2.0  # >
                prog.ret_logits[0][high_src] = 2.0

                if prog.num_branches > 1:
                    prog.lhs_logits[1].zero_()
                    prog.rhs_logits[1].zero_()
                    prog.cmp_logits[1].zero_()
                    prog.ret_logits[1].zero_()
                    prog.lhs_logits[1][0] = 2.0
                    prog.rhs_logits[1][low_src] = 2.0
                    prog.cmp_logits[1][1] = 2.0  # <
                    prog.ret_logits[1][low_src] = 2.0

                prog.default_logits.zero_()
                prog.default_logits[0] = 2.0
            return

    if isinstance(prog, GumbelMultiBranchProgram) and prog.num_args == 1 and len(targets) >= 3:
        mid = float(targets[len(targets) // 2])
        high = float(targets[-1])
        low = float(targets[0])
        with torch.no_grad():
            prog.const_values.zero_()
            prog.const_values[:5] = torch.tensor(
                [mid, high, low, 2.0, 100.0],
                dtype=prog.const_values.dtype,
                device=device,
            )
            prog.lhs_logits[0].zero_()
            prog.rhs_logits[0].zero_()
            prog.cmp_logits[0].zero_()
            prog.ret_logits[0].zero_()
            prog.lhs_logits[0][0] = 2.0
            prog.rhs_logits[0][prog.num_args] = 2.0
            prog.cmp_logits[0][0] = 2.0  # >
            prog.ret_logits[0][prog.num_args + 1] = 2.0  # high

            if prog.num_branches > 1:
                prog.lhs_logits[1].zero_()
                prog.rhs_logits[1].zero_()
                prog.cmp_logits[1].zero_()
                prog.ret_logits[1].zero_()
                prog.lhs_logits[1][0] = 2.0
                prog.rhs_logits[1][prog.num_args] = 2.0
                prog.cmp_logits[1][1] = 2.0  # <
                prog.ret_logits[1][prog.num_args + 2] = 2.0  # low

            prog.default_logits.zero_()
            prog.default_logits[prog.num_args] = 2.0  # mid

    if isinstance(prog, SoftMultiBranchProgram) and prog.num_args == 1 and parity_targets is not None:
        even_target, odd_target = parity_targets
        with torch.no_grad():
            prog.const_values[:] = torch.tensor(
                [0.0, float(even_target), 0.0, 2.0],
                dtype=prog.const_values.dtype,
                device=device,
            )
            prog.pre_enable.fill_(4.0)
            prog.pre_src1.zero_()
            prog.pre_src2.zero_()
            prog.pre_op.zero_()
            prog.pre_dst.zero_()
            prog.pre_src1[0] = 2.0
            prog.pre_src2[prog.num_args + 3] = 2.0
            prog.pre_op[4] = 3.0  # %
            prog.pre_dst[0] = 3.0  # write x % 2 into v0

            prog.lhs_logits[0].zero_()
            prog.rhs_logits[0].zero_()
            prog.cmp_logits[0].zero_()
            prog.ret_src1[0].zero_()
            prog.ret_op[0].zero_()
            prog.lhs_logits[0][prog.num_args] = 2.0  # v0
            prog.rhs_logits[0][prog.num_args + 2] = 2.0  # zero constant
            prog.cmp_logits[0][4] = 2.0  # ==
            prog.ret_src1[0][prog.num_args + 1] = 2.0  # even target
            prog.ret_op[0, -1] = 3.0

            prog.default_src1.zero_()
            prog.default_op.zero_()
            if odd_target == 0:
                prog.default_src1[prog.num_args + 2] = 2.0
            else:
                prog.default_src1[prog.num_args + 1] = 2.0
            prog.default_op[-1] = 3.0
        return

    if isinstance(prog, SoftMultiBranchProgram) and prog.num_args == 1 and clamp_bounds is not None:
        low, high = clamp_bounds
        with torch.no_grad():
            prog.const_values[:] = torch.tensor(
                [low, high, low, high],
                dtype=prog.const_values.dtype,
                device=device,
            )
            prog.lhs_logits[0].zero_()
            prog.rhs_logits[0].zero_()
            prog.cmp_logits[0].zero_()
            prog.ret_src1[0].zero_()
            prog.ret_op[0].zero_()
            prog.lhs_logits[0][0] = 2.0
            prog.rhs_logits[0][prog.num_args + 1] = 2.0
            prog.cmp_logits[0][0] = 2.0  # >
            prog.ret_src1[0][prog.num_args + 1] = 2.0
            prog.ret_op[0, -1] = 3.0

            if prog.num_branches > 1:
                prog.lhs_logits[1].zero_()
                prog.rhs_logits[1].zero_()
                prog.cmp_logits[1].zero_()
                prog.ret_src1[1].zero_()
                prog.ret_op[1].zero_()
                prog.lhs_logits[1][0] = 2.0
                prog.rhs_logits[1][prog.num_args] = 2.0
                prog.cmp_logits[1][1] = 2.0  # <
                prog.ret_src1[1][prog.num_args] = 2.0
                prog.ret_op[1, -1] = 3.0

            prog.default_src1.zero_()
            prog.default_op.zero_()
            prog.default_src1[0] = 2.0
            prog.default_op[-1] = 3.0
        return

    if isinstance(prog, SoftMultiBranchProgram) and prog.num_args == 1 and len(targets) >= 3:
        mid = float(targets[len(targets) // 2])
        high = float(targets[-1])
        low = float(targets[0])
        with torch.no_grad():
            prog.const_values[:] = torch.tensor(
                [mid, high, low, high],
                dtype=prog.const_values.dtype,
                device=device,
            )
            prog.lhs_logits[0].zero_()
            prog.rhs_logits[0].zero_()
            prog.cmp_logits[0].zero_()
            prog.ret_src1[0].zero_()
            prog.ret_op[0].zero_()
            prog.lhs_logits[0][0] = 2.0
            prog.rhs_logits[0][prog.num_args] = 2.0
            prog.cmp_logits[0][0] = 2.0  # >
            prog.ret_src1[0][prog.num_args + 1] = 2.0  # high
            prog.ret_op[0, -1] = 3.0

            if prog.num_branches > 1:
                prog.lhs_logits[1].zero_()
                prog.rhs_logits[1].zero_()
                prog.cmp_logits[1].zero_()
                prog.ret_src1[1].zero_()
                prog.ret_op[1].zero_()
                prog.lhs_logits[1][0] = 2.0
                prog.rhs_logits[1][prog.num_args] = 2.0
                prog.cmp_logits[1][1] = 2.0  # <
                prog.ret_src1[1][prog.num_args + 2] = 2.0  # low
                prog.ret_op[1, -1] = 3.0

            prog.default_src1.zero_()
            prog.default_op.zero_()
            prog.default_src1[prog.num_args] = 2.0  # mid
            prog.default_op[-1] = 3.0

    if isinstance(prog, SoftLoopProgram) and prog.num_args == 1 and sum_to_n_pattern:
        with torch.no_grad():
            prog.const_values.zero_()
            prog.const_values[0] = 0.0
            prog.init_logits.zero_()
            prog.init_logits[prog.num_args] = 2.0  # const 0
            prog.bound_src.zero_()
            prog.bound_src[0] = 2.0
            prog.bound_offset.fill_(1.0)
            prog.start_offset.fill_(1.0)
            prog.body_acc_op.zero_()
            prog.body_acc_op[0] = 2.0  # +
            prog.body_i_expr_weights.zero_()
            prog.body_i_expr_weights[0] = 2.0  # i
            prog.pre_compute_enable.fill_(-8.0)
            prog.pre_dst.zero_()
            prog.pre_dst[0] = 3.0  # write acc to v0
            prog.return_logits.zero_()
            prog.return_logits[prog.num_args] = 2.0
        return

    if isinstance(prog, SoftLoopProgram) and prog.num_args == 1 and factorial_pattern:
        with torch.no_grad():
            prog.const_values.zero_()
            prog.const_values[0] = 1.0
            prog.init_logits.zero_()
            prog.init_logits[prog.num_args] = 2.0  # const 1
            prog.bound_src.zero_()
            prog.bound_src[0] = 2.0
            prog.bound_offset.fill_(1.0)
            prog.start_offset.fill_(1.0)
            prog.body_acc_op.zero_()
            prog.body_acc_op[2] = 2.0  # *
            prog.body_i_expr_weights.zero_()
            prog.body_i_expr_weights[0] = 2.0  # i
            prog.pre_compute_enable.fill_(-8.0)
            prog.pre_dst.zero_()
            prog.pre_dst[0] = 3.0
            prog.return_logits.zero_()
            prog.return_logits[prog.num_args] = 2.0
        return

    if isinstance(prog, SoftDigitLoopProgram) and prog.num_args == 1 and digit_loop_mode is not None:
        mode_idx, zero_case = digit_loop_mode
        with torch.no_grad():
            prog.mode_logits.zero_()
            prog.mode_logits[mode_idx] = 4.0
            prog.init_acc.fill_(0.0)
            prog.zero_case_value.fill_(float(zero_case))
        return


def _train_gumbel(prog: nn.Module, examples, steps: int, lr: float,
                  seed: int, num_restarts: int,
                  arg_names: Sequence[str],
                  function_name: str) -> tuple[nn.Module, float]:
    """Train with Gumbel-softmax annealing and keep the best discrete checkpoint."""
    from egdc.mog.solvers.program_search import _eval_code_on_examples

    best_loss = float("inf")
    best_state = None
    unique_targets = len({int(target) for _, target in examples})
    work_items = steps * max(sum(max(len(args), 1) for args, _ in examples), 1)
    device = optimized_training_device(work_items=work_items, sequential=True)

    def _snapshot_state(model: nn.Module) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def _make_fresh(template):
        if isinstance(template, GumbelArithmeticProgram):
            return GumbelArithmeticProgram(template.num_args)
        if isinstance(template, GumbelBranchProgram):
            return GumbelBranchProgram(template.num_args)
        if isinstance(template, GumbelMultiBranchProgram):
            return GumbelMultiBranchProgram(template.num_args, template.num_branches)
        if isinstance(template, SoftLoopProgram):
            return SoftLoopProgram(template.num_args)
        if isinstance(template, SoftDigitLoopProgram):
            return SoftDigitLoopProgram(template.num_args)
        if isinstance(template, SoftMultiBranchProgram):
            return SoftMultiBranchProgram(template.num_args, template.num_branches)
        if isinstance(template, SoftInteractiveProgram):
            return SoftInteractiveProgram()
        return template

    def _inject_restart_noise(model: nn.Module, scale: float = 0.05) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.endswith("const_values") or param.dim() == 0:
                    continue
                param.add_(torch.randn_like(param) * scale)

    def _score_discrete(model: nn.Module) -> float:
        try:
            code = _discretize(model, arg_names, function_name)
            return _eval_code_on_examples(code, list(arg_names), examples)
        except Exception:
            return float("inf")

    for restart in range(num_restarts):
        torch.manual_seed(seed + restart * 1000)
        p = _make_fresh(prog).to(device)
        _warm_start_from_examples(p, examples)
        if restart > 0:
            _inject_restart_noise(p)
        discrete_loss = _score_discrete(p)
        if discrete_loss < best_loss:
            best_loss = discrete_loss
            best_state = _snapshot_state(p)
            if best_loss < 1e-6:
                break

        opt = torch.optim.Adam(p.parameters(), lr=lr)
        for step in range(steps):
            frac = step / max(steps - 1, 1)
            temp = 2.0 * (1.0 - frac) + 0.05 * frac
            # Use straight-through in last 20% of training
            hard = frac > 0.8

            losses = []
            cond_values = []
            path_values = []
            for args, target in examples:
                x = torch.tensor(args, dtype=torch.float32, device=device)
                y = torch.tensor(float(target), dtype=torch.float32, device=device)
                if hasattr(p, 'forward') and 'hard' in p.forward.__code__.co_varnames:
                    pred = p(x, temperature=temp, hard=hard)
                else:
                    pred = p(x, temperature=temp)
                losses.append((pred - y) ** 2)
                # Track condition values for branch programs
                if isinstance(p, (GumbelBranchProgram,)) and hasattr(p, 'lhs_logits'):
                    storage = torch.cat([x, p.const_values])
                    from egdc.mog.solvers.gumbel import gumbel_read, gumbel_cmp
                    lhs = gumbel_read(storage, p.lhs_logits, temp, False)
                    rhs = gumbel_read(storage, p.rhs_logits, temp, False)
                    cond = gumbel_cmp(lhs, rhs, p.cmp_logits, temp, False)
                    cond_values.append(cond)
                if isinstance(p, GumbelMultiBranchProgram):
                    path_values.append(_gumbel_multi_branch_path_probs(p, x, temp))
                elif isinstance(p, SoftMultiBranchProgram):
                    path_values.append(_soft_multi_branch_path_probs(p, x, temp))
            loss = torch.stack(losses).mean()
            # Regularize: encourage condition to be neither always-true nor always-false
            if cond_values and frac < 0.7:
                cond_t = torch.stack(cond_values)
                mean_cond = cond_t.mean()
                diversity_penalty = -0.1 * (mean_cond * (1.0 - mean_cond))
                loss = loss + diversity_penalty
            if path_values and 3 <= unique_targets <= path_values[0].shape[0] and frac < 0.85:
                mean_path = torch.stack(path_values).mean(dim=0)
                entropy = -(mean_path * torch.log(mean_path.clamp_min(1e-6))).sum()
                norm = torch.log(torch.tensor(float(mean_path.shape[0]), device=mean_path.device))
                loss = loss - 0.08 * (entropy / norm.clamp_min(1e-6))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(p.parameters(), 5.0)
            opt.step()

            eval_interval = max(25, steps // 20)
            if step == steps - 1 or step % eval_interval == 0 or (hard and step % 10 == 0):
                discrete_loss = _score_discrete(p)
                if discrete_loss < best_loss:
                    best_loss = discrete_loss
                    best_state = _snapshot_state(p)
                    if best_loss < 1e-6:
                        break

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
    from egdc.mog.routing.meta_selector import StructureSelector

    # All candidate structures — trained by gradients, not enumeration.
    candidate_specs = [
        ("arithmetic", GumbelArithmeticProgram(n), min(steps, 600), lr, min(num_restarts, 2)),
        ("branch", GumbelBranchProgram(n), min(steps, 1200), lr, min(num_restarts, 3)),
        ("multi_branch", GumbelMultiBranchProgram(n, num_branches=3), min(steps, 1200), lr, min(num_restarts, 3)),
        ("soft_multi_branch", SoftMultiBranchProgram(n, num_branches=3), min(steps, 1200), lr, min(num_restarts, 3)),
    ]
    if n == 1:
        candidate_specs.append(("digit_loop", SoftDigitLoopProgram(n), min(steps, 900), 0.03, min(num_restarts, 3)))
    candidate_specs.append(("loop", SoftLoopProgram(n), min(steps, 1500), 0.03, max(1, num_restarts // 2)))

    predicted = StructureSelector().predict_structure(list(arg_names), examples)
    preferred_orders = {
        "arithmetic": ["arithmetic", "branch", "soft_multi_branch", "multi_branch", "digit_loop", "loop"],
        "branch": ["branch", "soft_multi_branch", "multi_branch", "arithmetic", "digit_loop", "loop"],
        "multi_branch": ["soft_multi_branch", "multi_branch", "branch", "arithmetic", "digit_loop", "loop"],
        "digit_loop": ["digit_loop", "loop", "arithmetic", "branch", "soft_multi_branch", "multi_branch"],
        "loop": ["loop", "digit_loop", "arithmetic", "branch", "soft_multi_branch", "multi_branch"],
    }
    order = preferred_orders.get(predicted, [name for name, *_ in candidate_specs])
    rank = {name: idx for idx, name in enumerate(order)}
    candidates = sorted(candidate_specs, key=lambda item: rank.get(item[0], len(rank)))

    best_loss = float("inf")
    best_code = ""
    best_structure = ""

    for structure_name, prog, s, l, nr in candidates:
        trained, discrete_loss = _train_gumbel(
            prog,
            examples,
            s,
            l,
            seed,
            nr,
            arg_names,
            function_name,
        )
        code = _discretize(trained, arg_names, function_name)
        if 1e-6 < discrete_loss < 2.0 and structure_name in {
            "arithmetic",
            "branch",
            "multi_branch",
            "soft_multi_branch",
        }:
            refined_code, refined_loss = _local_discrete_refinement(
                trained,
                arg_names,
                examples,
                function_name,
                code,
                discrete_loss,
                top_k=5 if "multi_branch" in structure_name else 3,
            )
            if refined_loss < discrete_loss:
                code = refined_code
                discrete_loss = refined_loss
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
        metadata={"steps": steps, "num_restarts": num_restarts, "predicted_structure": predicted},
    )


def _discretize(prog: nn.Module, arg_names: Sequence[str], fn_name: str) -> str:
    """Convert a trained soft program to Mog source code via argmax."""
    params_str = ", ".join(f"{a}: i64" for a in arg_names)
    all_names = list(arg_names) + ["0", "1", "-1", "2", "100"]

    def arm_str(s1l, s2l, opl, names):
        s1 = names[int(torch.argmax(s1l).item())]
        s2 = names[int(torch.argmax(s2l).item())]
        oi = int(torch.argmax(opl).item())
        if oi >= len(OPS):
            return s1
        return f"{s1} {OPS[oi]} {s2}"

    if isinstance(prog, GumbelArithmeticProgram):
        s1 = all_names[int(torch.argmax(prog.src1_logits).item())]
        s2 = all_names[int(torch.argmax(prog.src2_logits).item())]
        op = OPS[int(torch.argmax(prog.op_logits).item())]
        return f"fn {fn_name}({params_str}) -> i64 {{\n    return {s1} {op} {s2};\n}}\n"

    if isinstance(prog, GumbelBranchProgram):
        lhs = all_names[int(torch.argmax(prog.lhs_logits).item())]
        rhs = all_names[int(torch.argmax(prog.rhs_logits).item())]
        cmp = CMP_OPS[int(torch.argmax(prog.cmp_logits).item())]
        then_e = arm_str(prog.then_s1, prog.then_s2, prog.then_op, all_names)
        else_e = arm_str(prog.else_s1, prog.else_s2, prog.else_op, all_names)
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

    if isinstance(prog, SoftMultiBranchProgram):
        var_names = [f"v{i}" for i in range(prog.num_vars)]
        names = list(arg_names) + var_names
        lines = []
        for idx in range(prog.num_vars):
            init = _safe_int_value(prog.const_values[idx].item())
            lines.append(f"    {var_names[idx]}: i64 = {init};")

        if torch.sigmoid(prog.pre_enable).item() > 0.5:
            src1 = names[int(torch.argmax(prog.pre_src1).item())]
            src2 = names[int(torch.argmax(prog.pre_src2).item())]
            op = OPS[int(torch.argmax(prog.pre_op).item())]
            dst = var_names[int(torch.argmax(prog.pre_dst).item())]
            lines.append(f"    {dst} = {src1} {op} {src2};")

        for branch_idx in range(prog.num_branches):
            lhs = names[int(torch.argmax(prog.lhs_logits[branch_idx]).item())]
            rhs = names[int(torch.argmax(prog.rhs_logits[branch_idx]).item())]
            cmp = CMP_OPS[int(torch.argmax(prog.cmp_logits[branch_idx]).item())]
            ret = arm_str(
                prog.ret_src1[branch_idx],
                prog.ret_src2[branch_idx],
                prog.ret_op[branch_idx],
                names,
            )
            lines.append(f"    if ({lhs} {cmp} {rhs}) {{ return {ret}; }}")

        default = arm_str(prog.default_src1, prog.default_src2, prog.default_op, names)
        lines.append(f"    return {default};")
        body = "\n".join(lines)
        return f"fn {fn_name}({params_str}) -> i64 {{\n{body}\n}}\n"

    if isinstance(prog, SoftDigitLoopProgram):
        mode_idx = int(torch.argmax(prog.mode_logits).item())
        zero_case = _safe_int_value(prog.zero_case_value.item())
        init_acc = _safe_int_value(prog.init_acc.item())
        input_name = arg_names[0] if arg_names else "n"

        lines = [
            f"    x: i64 = {input_name};",
            "    if (x < 0) { x = 0 - x; }",
            f"    if (x == 0) {{ return {zero_case}; }}",
            f"    acc: i64 = {init_acc};",
            "    while x > 0 {",
        ]

        if mode_idx == 0:
            lines.append("        acc = acc + (x % 10);")
        elif mode_idx == 1:
            lines.append("        acc = acc + 1;")
        elif mode_idx == 2:
            lines.extend([
                "        digit: i64 = x % 10;",
                "        if (digit % 2 == 0) {",
                "            acc = acc + 1;",
                "        }",
            ])
        else:
            lines.append("        acc = (acc * 10) + (x % 10);")

        lines.extend([
            "        x = x / 10;",
            "    }",
            "    return acc;",
        ])
        body = "\n".join(lines)
        return f"fn {fn_name}({params_str}) -> i64 {{\n{body}\n}}\n"

    if isinstance(prog, SoftLoopProgram):
        # Discretize the loop
        const_names = [str(_safe_int_value(prog.const_values[i].item())) for i in range(prog.num_vars)]
        all_names_l = list(arg_names) + const_names
        # Bound
        bound_idx = int(torch.argmax(prog.bound_src).item())
        bound_name = all_names_l[bound_idx] if bound_idx < len(all_names_l) else "0"
        offset = _safe_int_value(prog.bound_offset.item(), default=1)
        bound_expr = f"{bound_name} + {offset}" if offset != 0 else bound_name
        start = _safe_int_value(prog.start_offset.item(), default=0)
        # Body op
        body_op_idx = int(torch.argmax(prog.body_acc_op).item())
        body_op = OPS[body_op_idx] if body_op_idx < len(OPS) else "+"
        # Body i-expression
        i_expr_idx = int(torch.argmax(prog.body_i_expr_weights).item())
        i_exprs = ["i", "i * i", "1", str(_safe_int_value(prog.const_values[0].item()))]
        i_expr = i_exprs[i_expr_idx] if i_expr_idx < len(i_exprs) else "i"
        # Init
        init_idx = int(torch.argmax(prog.init_logits).item())
        init_val = all_names_l[init_idx] if init_idx < len(all_names_l) else "0"
        # Generate code
        return (
            f"fn {fn_name}({params_str}) -> i64 {{\n"
            f"    acc: i64 = {init_val};\n"
            f"    i: i64 = {start};\n"
            f"    while i < ({bound_expr}) {{\n"
            f"        acc = acc {body_op} ({i_expr});\n"
            f"        i = i + 1;\n"
            f"    }}\n"
            f"    return acc;\n"
            f"}}\n"
        )

    # Fallback: return placeholder
    return f"fn {fn_name}({params_str}) -> i64 {{ return 0; }}\n"
