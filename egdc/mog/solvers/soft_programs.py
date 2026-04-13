"""Fully differentiable program structures for Mog.

Every program structure — loops, multi-branch, composition — is a soft
parameterized module where ALL choices are learned by gradient descent
through differentiable execution. No discrete enumeration.

This is the actual differentiable CPU.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_PREFERRED_TORCH_DEVICE: torch.device | None = None


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

OPS = ["+", "-", "*", "/", "%"]
CMP_OPS = [">", "<", ">=", "<=", "==", "!="]


def preferred_torch_device() -> torch.device:
    global _PREFERRED_TORCH_DEVICE
    if _PREFERRED_TORCH_DEVICE is None:
        if torch.cuda.is_available():
            _PREFERRED_TORCH_DEVICE = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _PREFERRED_TORCH_DEVICE = torch.device("mps")
        else:
            _PREFERRED_TORCH_DEVICE = torch.device("cpu")
    return _PREFERRED_TORCH_DEVICE


def optimized_training_device(
    *,
    work_items: int,
    sequential: bool = False,
) -> torch.device:
    device = preferred_torch_device()
    if device.type == "cpu":
        return device
    if device.type == "mps":
        threshold = 32768 if sequential else 8192
    else:
        threshold = 8192 if sequential else 2048
    if work_items < threshold:
        return torch.device("cpu")
    return device


def _scalar_tensor(value: float, *, device: torch.device) -> torch.Tensor:
    return torch.tensor(float(value), dtype=torch.float32, device=device)


def soft_op(a: torch.Tensor, b: torch.Tensor, op_weights: torch.Tensor) -> torch.Tensor:
    """Soft-select over binary operations."""
    safe_b = torch.where(torch.abs(b) < 1e-6, torch.ones_like(b), b)
    results = torch.stack([
        a + b,
        a - b,
        a * b,
        a / safe_b,
        torch.remainder(torch.round(a), torch.clamp(torch.round(torch.abs(safe_b)), min=1.0)),
    ])
    return (op_weights * results).sum()


def soft_cmp(a: torch.Tensor, b: torch.Tensor, cmp_weights: torch.Tensor) -> torch.Tensor:
    """Soft comparison returning a probability in [0, 1]."""
    diff = a - b
    results = torch.stack([
        torch.sigmoid(diff / 0.25),                       # >
        torch.sigmoid(-diff / 0.25),                      # <
        torch.sigmoid(diff / 0.25),                       # >=
        torch.sigmoid(-diff / 0.25),                      # <=
        torch.exp(-(diff ** 2) / 0.125),                  # ==
        1.0 - torch.exp(-(diff ** 2) / 0.125),            # !=
    ])
    return (cmp_weights * results).sum()


def soft_read(storage: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Soft attention read from a storage vector."""
    return (weights * storage).sum()


def soft_write(storage: torch.Tensor, dst_weights: torch.Tensor,
               value: torch.Tensor, enable: torch.Tensor,
               num_args: int) -> torch.Tensor:
    """Soft write to storage variables (not args)."""
    new = []
    for idx in range(storage.shape[0]):
        if idx < num_args:
            new.append(storage[idx])
        else:
            v = idx - num_args
            w = dst_weights[v] * enable
            new.append(storage[idx] * (1.0 - w) + value * w)
    return torch.stack(new)


# ---------------------------------------------------------------------------
# SoftLoopProgram: differentiable loop discovery
# ---------------------------------------------------------------------------

class SoftLoopProgram(nn.Module):
    """Fully differentiable loop program.

    Structure:
        acc = soft_init
        for i = soft_start to soft_bound(args):
            acc = soft_body(acc, i, args)
        return soft_return(acc, args)

    Every choice is a learnable parameter:
    - init value
    - start value
    - bound (soft function of args)
    - body operation (soft combination of acc OP f(i))
    - return expression

    The loop is unrolled to MAX_ITER steps, each gated by a soft bound check.
    """

    MAX_ITER = 32

    def __init__(self, num_args: int, num_vars: int = 4):
        super().__init__()
        self.num_args = num_args
        self.num_vars = num_vars
        self.num_sources = num_args + num_vars

        # Accumulator init: soft read from sources (includes learnable constants)
        self.init_logits = nn.Parameter(torch.zeros(self.num_sources))
        self.const_values = nn.Parameter(torch.zeros(num_vars))

        # Loop bound: soft function of args → scalar
        self.bound_src = nn.Parameter(torch.zeros(self.num_sources))
        self.bound_offset = nn.Parameter(torch.tensor(1.0))  # bound = src + offset
        self.start_offset = nn.Parameter(torch.tensor(0.0))

        # Body: acc = acc BODY_OP f(i)
        # f(i) is a soft combination of: i, i*i, 1, args[k]
        self.body_acc_op = nn.Parameter(torch.zeros(len(OPS)))
        self.body_i_expr_weights = nn.Parameter(torch.zeros(4))  # i, i*i, 1, const

        # Pre-loop compute: optional v0 = src1 OP src2
        self.pre_compute_enable = nn.Parameter(torch.tensor(-4.0))
        self.pre_src1 = nn.Parameter(torch.zeros(self.num_sources))
        self.pre_src2 = nn.Parameter(torch.zeros(self.num_sources))
        self.pre_op = nn.Parameter(torch.zeros(len(OPS)))
        self.pre_dst = nn.Parameter(torch.zeros(num_vars))

        # Return: soft read from storage after loop
        self.return_logits = nn.Parameter(torch.zeros(self.num_sources))

        self._init()

    def _init(self):
        with torch.no_grad():
            # Default: acc=0, loop from 0 to arg0+1, body: acc = acc + i
            self.const_values[0] = 0.0
            self.init_logits[self.num_args] = 2.0  # v0 (const 0)
            if self.num_args > 0:
                self.bound_src[0] = 2.0  # bound = arg0
            self.bound_offset.fill_(1.0)
            self.start_offset.fill_(0.0)
            self.body_acc_op[0] = 2.0  # +
            self.body_i_expr_weights[0] = 2.0  # f(i) = i
            self.return_logits[self.num_args] = 2.0  # return v0 (acc)

    def forward(self, args: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        device = args.device
        storage = torch.zeros(self.num_sources, device=device)
        storage[:self.num_args] = args
        for v in range(self.num_vars):
            storage[self.num_args + v] = self.const_values[v]

        # Optional pre-compute
        pre_en = torch.sigmoid(self.pre_compute_enable)
        if pre_en.item() > 0.01:
            s1 = soft_read(storage, F.softmax(self.pre_src1 / temperature, dim=0))
            s2 = soft_read(storage, F.softmax(self.pre_src2 / temperature, dim=0))
            pre_val = soft_op(s1, s2, F.softmax(self.pre_op / temperature, dim=0))
            storage = soft_write(storage, F.softmax(self.pre_dst / temperature, dim=0),
                                  pre_val, pre_en, self.num_args)

        # Loop
        bound_w = F.softmax(self.bound_src / temperature, dim=0)
        bound = soft_read(storage, bound_w) + self.bound_offset

        acc_op_w = F.softmax(self.body_acc_op / temperature, dim=0)
        i_expr_w = F.softmax(self.body_i_expr_weights / temperature, dim=0)

        # Init accumulator
        init_w = F.softmax(self.init_logits / temperature, dim=0)
        acc = soft_read(storage, init_w)
        start = self.start_offset

        for step in range(self.MAX_ITER):
            i_val = torch.tensor(float(step), device=device) + start
            in_bounds = torch.sigmoid((bound - i_val - 0.5) / 0.3)

            # f(i) = weighted combination of [i, i*i, 1, const]
            i_exprs = torch.stack([
                i_val,
                i_val * i_val,
                torch.ones_like(i_val),
                self.const_values[0].to(device),
            ])
            f_i = (i_expr_w * i_exprs).sum()

            # new_acc = acc OP f_i
            new_acc = soft_op(acc, f_i, acc_op_w)
            acc = acc + in_bounds * (new_acc - acc)

        # Write acc back to storage for return
        storage = soft_write(storage, F.softmax(self.pre_dst / temperature, dim=0),
                              acc, torch.tensor(1.0, device=device), self.num_args)

        ret_w = F.softmax(self.return_logits / temperature, dim=0)
        return soft_read(storage, ret_w)


# ---------------------------------------------------------------------------
# SoftDigitLoopProgram: differentiable digit-state loops
# ---------------------------------------------------------------------------

class SoftDigitLoopProgram(nn.Module):
    """Differentiable unary digit-processing loop.

    Maintains two pieces of state:
    - x: the remaining absolute-value digits of the input
    - acc: an accumulator updated from the current least-significant digit

    The learned mode selects one of several digit-processing recurrences:
    - acc + digit
    - acc + 1
    - acc + is_even(digit)
    - acc * 10 + digit
    """

    MAX_DIGITS = 20

    def __init__(self, num_args: int):
        super().__init__()
        self.num_args = num_args
        self.mode_logits = nn.Parameter(torch.zeros(4))
        self.init_acc = nn.Parameter(torch.tensor(0.0))
        self.zero_case_value = nn.Parameter(torch.tensor(0.0))

        with torch.no_grad():
            self.mode_logits[0] = 2.0  # digit sum

    def forward(self, args: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        device = args.device
        source = args[0] if self.num_args > 0 else torch.tensor(0.0, device=device)
        x0 = torch.abs(torch.round(source))
        x = x0.clone()
        acc = self.init_acc.to(device)
        mode_w = F.softmax(self.mode_logits / temperature, dim=0)

        for _ in range(self.MAX_DIGITS):
            active = torch.sigmoid((x - 0.5) / 0.05)
            digit = torch.remainder(x, 10.0)
            digit_mod_2 = torch.remainder(torch.round(digit), 2.0)
            is_even_digit = torch.exp(-(digit_mod_2 ** 2) / 0.125)

            candidates = torch.stack([
                acc + digit,
                acc + 1.0,
                acc + is_even_digit,
                acc * 10.0 + digit,
            ])
            updated_acc = (mode_w * candidates).sum()
            acc = acc + active * (updated_acc - acc)

            next_x = torch.floor(x / 10.0)
            x = x + active * (next_x - x)

        is_zero = torch.sigmoid((0.5 - x0) / 0.05)
        return (1.0 - is_zero) * acc + is_zero * self.zero_case_value.to(device)


# ---------------------------------------------------------------------------
# SoftMultiBranchProgram: differentiable multi-branch discovery
# ---------------------------------------------------------------------------

class SoftMultiBranchProgram(nn.Module):
    """Fully differentiable multi-branch program.

    Structure:
        v0 = src1 OP src2  (optional pre-compute)
        if (lhs1 CMP1 rhs1) return expr1;
        if (lhs2 CMP2 rhs2) return expr2;
        return default_expr;

    Every comparison, source, operator, and return value is a soft parameter.
    """

    def __init__(self, num_args: int, num_branches: int = 3, num_vars: int = 4):
        super().__init__()
        self.num_args = num_args
        self.num_branches = num_branches
        self.num_vars = num_vars
        self.ns = num_args + num_vars

        self.const_values = nn.Parameter(torch.zeros(num_vars))

        # Pre-compute slots
        self.pre_enable = nn.Parameter(torch.tensor(-1.0))
        self.pre_src1 = nn.Parameter(torch.zeros(self.ns))
        self.pre_src2 = nn.Parameter(torch.zeros(self.ns))
        self.pre_op = nn.Parameter(torch.zeros(len(OPS)))
        self.pre_dst = nn.Parameter(torch.zeros(num_vars))

        # Branch conditions
        self.cmp_logits = nn.Parameter(torch.zeros(num_branches, len(CMP_OPS)))
        self.lhs_logits = nn.Parameter(torch.zeros(num_branches, self.ns))
        self.rhs_logits = nn.Parameter(torch.zeros(num_branches, self.ns))

        # Branch return expressions: src1 OP src2, with identity option
        self.ret_src1 = nn.Parameter(torch.zeros(num_branches, self.ns))
        self.ret_src2 = nn.Parameter(torch.zeros(num_branches, self.ns))
        self.ret_op = nn.Parameter(torch.zeros(num_branches, len(OPS) + 1))  # +1 identity

        # Default return
        self.default_src1 = nn.Parameter(torch.zeros(self.ns))
        self.default_src2 = nn.Parameter(torch.zeros(self.ns))
        self.default_op = nn.Parameter(torch.zeros(len(OPS) + 1))

        self._init()

    def _init(self):
        with torch.no_grad():
            # Default constants: 0, 1, -1, 100
            self.const_values[0] = 0.0
            self.const_values[1] = 1.0
            self.const_values[2] = -1.0
            self.const_values[3] = 100.0

            # First branch: compare arg0 vs const0 (0), return const1 (1)
            if self.num_args > 0:
                self.lhs_logits[0, 0] = 1.0
                self.rhs_logits[0, self.num_args] = 1.0  # const 0
                self.cmp_logits[0, 0] = 1.0  # >
                self.ret_src1[0, self.num_args + 1] = 1.0  # const 1
                self.ret_op[0, -1] = 2.0  # identity

            # Default: return const 0
            self.default_src1[self.num_args] = 1.0
            self.default_op[-1] = 2.0

    def _eval_arm(self, storage, src1_logits, src2_logits, op_logits, temperature):
        src1_w = F.softmax(src1_logits / temperature, dim=0)
        src2_w = F.softmax(src2_logits / temperature, dim=0)
        op_w = F.softmax(op_logits / temperature, dim=0)
        s1 = soft_read(storage, src1_w)
        s2 = soft_read(storage, src2_w)
        safe_s2 = torch.where(torch.abs(s2) < 1e-6, torch.ones_like(s2), s2)
        ops = torch.stack([
            s1 + s2, s1 - s2, s1 * s2, s1 / safe_s2,
            torch.remainder(torch.round(s1), torch.clamp(torch.round(torch.abs(safe_s2)), min=1.0)),
            s1,  # identity
        ])
        return (op_w * ops).sum()

    def forward(self, args: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        device = args.device
        storage = torch.zeros(self.ns, device=device)
        storage[:self.num_args] = args
        for v in range(self.num_vars):
            storage[self.num_args + v] = self.const_values[v]

        # Pre-compute
        pre_en = torch.sigmoid(self.pre_enable)
        s1 = soft_read(storage, F.softmax(self.pre_src1 / temperature, dim=0))
        s2 = soft_read(storage, F.softmax(self.pre_src2 / temperature, dim=0))
        pre_val = soft_op(s1, s2, F.softmax(self.pre_op / temperature, dim=0))
        storage = soft_write(storage, F.softmax(self.pre_dst / temperature, dim=0),
                              pre_val, pre_en, self.num_args)

        # Sequential branches
        return_value = torch.tensor(0.0, device=device)
        remaining_prob = torch.tensor(1.0, device=device)

        for b in range(self.num_branches):
            cmp_w = F.softmax(self.cmp_logits[b] / temperature, dim=0)
            lhs = soft_read(storage, F.softmax(self.lhs_logits[b] / temperature, dim=0))
            rhs = soft_read(storage, F.softmax(self.rhs_logits[b] / temperature, dim=0))
            cond = soft_cmp(lhs, rhs, cmp_w)

            ret_val = self._eval_arm(storage, self.ret_src1[b], self.ret_src2[b],
                                      self.ret_op[b], temperature)

            # This branch fires with probability cond * remaining_prob
            fire_prob = cond * remaining_prob
            return_value = return_value + fire_prob * ret_val
            remaining_prob = remaining_prob * (1.0 - cond)

        # Default
        default_val = self._eval_arm(storage, self.default_src1, self.default_src2,
                                      self.default_op, temperature)
        return_value = return_value + remaining_prob * default_val

        return return_value


# ---------------------------------------------------------------------------
# SoftInteractiveProgram: differentiable state-update discovery
# ---------------------------------------------------------------------------

class SoftInteractiveProgram(nn.Module):
    """Differentiable state update function: state' = f(state, input).

    Each candidate update is computed independently so gradients don't
    interfere between modes (e.g., state+input vs state+1).
    """

    NUM_CANDIDATES = 9

    def __init__(self):
        super().__init__()
        # Selection over independent candidate outputs
        self.mode_logits = nn.Parameter(torch.zeros(self.NUM_CANDIDATES))
        # Learnable constants for some candidates
        self.const_a = nn.Parameter(torch.tensor(1.0))
        self.const_b = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        state: torch.Tensor,
        inp: torch.Tensor,
        *,
        is_first: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if is_first is None:
            is_first = torch.tensor(0.0, dtype=state.dtype, device=state.device)
        # Each candidate is computed independently — no gradient cross-talk
        candidates = torch.stack([
            state + inp,                    # 0: running sum
            state - inp,                    # 1: running difference
            state * inp,                    # 2: running product
            inp,                            # 3: passthrough
            state + self.const_a,           # 4: counter (state + 1)
            inp * self.const_a,             # 5: scale input
            self.const_b,                   # 6: constant output
            is_first * inp + (1.0 - is_first) * torch.maximum(state, inp),
                                            # 7: running max
            is_first * inp + (1.0 - is_first) * torch.minimum(state, inp),
                                            # 8: running min
        ])
        mode_w = F.softmax(self.mode_logits / temperature, dim=0)
        return (mode_w * candidates).sum()


# ---------------------------------------------------------------------------
# SoftInteractivePairProgram: differentiable pairwise reduction discovery
# ---------------------------------------------------------------------------

class SoftInteractivePairProgram(nn.Module):
    """Differentiable two-input reducer used by buffered interactive streams."""

    NUM_CANDIDATES = 9

    def __init__(self):
        super().__init__()
        self.mode_logits = nn.Parameter(torch.zeros(self.NUM_CANDIDATES))

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        candidates = torch.stack([
            a + b,                  # 0: pair add
            a - b,                  # 1: pair sub
            b - a,                  # 2: reversed pair sub
            a * b,                  # 3: pair mul
            torch.maximum(a, b),    # 4: pair max
            torch.minimum(a, b),    # 5: pair min
            torch.abs(a - b),       # 6: pair abs diff
            a,                      # 7: first item
            b,                      # 8: second item
        ])
        mode_w = F.softmax(self.mode_logits / temperature, dim=0)
        return (mode_w * candidates).sum()


# ---------------------------------------------------------------------------
# SoftInteractiveStateEmitProgram: differentiable stateful sparse emission
# ---------------------------------------------------------------------------

class SoftInteractiveStateEmitProgram(nn.Module):
    """Differentiable state update plus sparse emission over a stream."""

    NUM_UPDATE_CANDIDATES = 9
    NUM_EMIT_CANDIDATES = 8

    def __init__(self):
        super().__init__()
        self.update_logits = nn.Parameter(torch.zeros(self.NUM_UPDATE_CANDIDATES))
        self.emit_logits = nn.Parameter(torch.zeros(self.NUM_EMIT_CANDIDATES))
        self.const_a = nn.Parameter(torch.tensor(1.0))
        self.const_b = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        state: torch.Tensor,
        inp: torch.Tensor,
        *,
        is_first: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if is_first is None:
            is_first = torch.tensor(0.0, dtype=state.dtype, device=state.device)

        update_candidates = torch.stack([
            state + inp,                    # 0: running sum
            state - inp,                    # 1: running difference
            state * inp,                    # 2: running product
            inp,                            # 3: passthrough
            state + self.const_a,           # 4: counter (state + 1)
            inp * self.const_a,             # 5: scale input
            self.const_b,                   # 6: constant output
            is_first * inp + (1.0 - is_first) * torch.maximum(state, inp),
                                            # 7: running max
            is_first * inp + (1.0 - is_first) * torch.minimum(state, inp),
                                            # 8: running min
        ])
        update_w = F.softmax(self.update_logits / temperature, dim=0)
        new_state = (update_w * update_candidates).sum()

        delta = new_state - state
        changed = 1.0 - torch.exp(-(delta ** 2) / 0.125)
        increased = torch.sigmoid(delta / 0.25)
        decreased = torch.sigmoid(-delta / 0.25)
        emit_candidates = torch.stack([
            torch.ones_like(new_state),                             # 0: always
            is_first + (1.0 - is_first) * changed,                 # 1: first or changed
            is_first + (1.0 - is_first) * increased,               # 2: first or increased
            is_first + (1.0 - is_first) * decreased,               # 3: first or decreased
            torch.sigmoid(inp / 0.25),                             # 4: input > 0
            torch.sigmoid(new_state / 0.25),                       # 5: state > 0
            is_first,                                              # 6: first only
            torch.zeros_like(new_state),                           # 7: never
        ])
        emit_w = F.softmax(self.emit_logits / temperature, dim=0)
        emit_prob = (emit_w * emit_candidates).sum()
        emit_prob = torch.clamp(emit_prob, 1e-6, 1.0 - 1e-6)
        return new_state, emit_prob


# ---------------------------------------------------------------------------
# SoftInteractiveTwoRegisterProgram: differentiable two-register dense stream
# ---------------------------------------------------------------------------

class SoftInteractiveTwoRegisterProgram(nn.Module):
    """Differentiable dense transducer with two persistent registers."""

    NUM_A_CANDIDATES = 8
    NUM_B_CANDIDATES = 6
    NUM_OUTPUT_CANDIDATES = 5

    def __init__(self):
        super().__init__()
        self.a_logits = nn.Parameter(torch.zeros(self.NUM_A_CANDIDATES))
        self.b_logits = nn.Parameter(torch.zeros(self.NUM_B_CANDIDATES))
        self.out_logits = nn.Parameter(torch.zeros(self.NUM_OUTPUT_CANDIDATES))
        self.const_a = nn.Parameter(torch.tensor(1.0))
        self.const_b = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        reg_a: torch.Tensor,
        reg_b: torch.Tensor,
        inp: torch.Tensor,
        *,
        is_first: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if is_first is None:
            is_first = torch.tensor(0.0, dtype=reg_a.dtype, device=reg_a.device)

        a_candidates = torch.stack([
            reg_a + inp,                                             # 0: accumulate input
            reg_a - inp,                                             # 1: subtract input
            inp,                                                     # 2: passthrough input
            is_first * inp + (1.0 - is_first) * torch.maximum(reg_a, inp),
                                                                     # 3: running max
            is_first * inp + (1.0 - is_first) * torch.minimum(reg_a, inp),
                                                                     # 4: running min
            reg_a + self.const_a,                                    # 5: add constant
            torch.clamp(reg_a + inp, min=inp),                       # 6: Kadane step (max subarray)
            torch.clamp(reg_a + inp, max=inp),                       # 7: anti-Kadane step (min subarray)
        ])
        a_w = F.softmax(self.a_logits / temperature, dim=0)
        next_a = (a_w * a_candidates).sum()

        b_candidates = torch.stack([
            reg_b,                                                   # 0: keep register
            reg_b + self.const_b,                                    # 1: add constant
            inp,                                                     # 2: passthrough input
            reg_b + inp,                                             # 3: accumulate input
            is_first * next_a + (1.0 - is_first) * torch.clamp(reg_b, min=next_a),
                                                                     # 4: global max of reg_a, init on first step
            is_first * next_a + (1.0 - is_first) * torch.clamp(reg_b, max=next_a),
                                                                     # 5: global min of reg_a, init on first step
        ])
        b_w = F.softmax(self.b_logits / temperature, dim=0)
        next_b = (b_w * b_candidates).sum()

        safe_div = next_a / torch.where(
            torch.abs(next_b) < 1e-3,
            torch.ones_like(next_b),
            next_b,
        )
        out_candidates = torch.stack([
            next_a,                  # 0
            next_b,                  # 1
            next_a + next_b,         # 2
            next_a - next_b,         # 3
            safe_div,                # 4
        ])
        out_w = F.softmax(self.out_logits / temperature, dim=0)
        output = (out_w * out_candidates).sum()
        return next_a, next_b, output


# ---------------------------------------------------------------------------
# SoftInteractiveTwoRegisterEmitProgram: differentiable sparse two-register stream
# ---------------------------------------------------------------------------

class SoftInteractiveTwoRegisterEmitProgram(nn.Module):
    """Differentiable sparse transducer with two persistent registers."""

    NUM_A_CANDIDATES = 8
    NUM_B_CANDIDATES = 6
    NUM_OUTPUT_CANDIDATES = 5
    NUM_EMIT_CANDIDATES = 26

    def __init__(self):
        super().__init__()
        self.a_logits = nn.Parameter(torch.zeros(self.NUM_A_CANDIDATES))
        self.b_logits = nn.Parameter(torch.zeros(self.NUM_B_CANDIDATES))
        self.out_logits = nn.Parameter(torch.zeros(self.NUM_OUTPUT_CANDIDATES))
        self.emit_logits = nn.Parameter(torch.zeros(self.NUM_EMIT_CANDIDATES))
        self.const_a = nn.Parameter(torch.tensor(1.0))
        self.const_b = nn.Parameter(torch.tensor(1.0))
        self.emit_threshold = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        reg_a: torch.Tensor,
        reg_b: torch.Tensor,
        inp: torch.Tensor,
        *,
        is_first: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if is_first is None:
            is_first = torch.tensor(0.0, dtype=reg_a.dtype, device=reg_a.device)

        a_candidates = torch.stack([
            reg_a + inp,                                             # 0: accumulate input
            reg_a - inp,                                             # 1: subtract input
            inp,                                                     # 2: passthrough input
            is_first * inp + (1.0 - is_first) * torch.maximum(reg_a, inp),
                                                                     # 3: running max
            is_first * inp + (1.0 - is_first) * torch.minimum(reg_a, inp),
                                                                     # 4: running min
            reg_a + self.const_a,                                    # 5: add constant
            torch.clamp(reg_a + inp, min=inp),                       # 6: Kadane step (max subarray)
            torch.clamp(reg_a + inp, max=inp),                       # 7: anti-Kadane step (min subarray)
        ])
        a_w = F.softmax(self.a_logits / temperature, dim=0)
        next_a = (a_w * a_candidates).sum()

        b_candidates = torch.stack([
            reg_b,                                                   # 0: keep register
            reg_b + self.const_b,                                    # 1: add constant
            inp,                                                     # 2: passthrough input
            reg_b + inp,                                             # 3: accumulate input
            is_first * next_a + (1.0 - is_first) * torch.clamp(reg_b, min=next_a),
                                                                     # 4: global max of reg_a, init on first step
            is_first * next_a + (1.0 - is_first) * torch.clamp(reg_b, max=next_a),
                                                                     # 5: global min of reg_a, init on first step
        ])
        b_w = F.softmax(self.b_logits / temperature, dim=0)
        next_b = (b_w * b_candidates).sum()

        safe_div = next_a / torch.where(
            torch.abs(next_b) < 1e-3,
            torch.ones_like(next_b),
            next_b,
        )
        safe_old_div = reg_a / torch.where(
            torch.abs(reg_b) < 1e-3,
            torch.ones_like(reg_b),
            reg_b,
        )
        out_candidates = torch.stack([
            next_a,                  # 0
            next_b,                  # 1
            next_a + next_b,         # 2
            next_a - next_b,         # 3
            safe_div,                # 4
        ])
        out_w = F.softmax(self.out_logits / temperature, dim=0)
        output = (out_w * out_candidates).sum()
        old_out_candidates = torch.stack([
            reg_a,                   # 0
            reg_b,                   # 1
            reg_a + reg_b,           # 2
            reg_a - reg_b,           # 3
            safe_old_div,            # 4
        ])
        old_output = (out_w * old_out_candidates).sum()
        delta_output = output - old_output
        output_changed = 1.0 - torch.exp(-(delta_output ** 2) / 0.125)
        output_increased = torch.sigmoid(delta_output / 0.25)
        output_decreased = torch.sigmoid(-delta_output / 0.25)
        output_above_threshold = torch.sigmoid(
            (output - self.emit_threshold) / 0.25
        )
        prev_above_threshold = torch.sigmoid(
            (old_output - self.emit_threshold) / 0.25
        )
        crosses_above_threshold = (
            (1.0 - is_first)
            * output_above_threshold
            * (1.0 - prev_above_threshold)
        )
        crosses_below_threshold = (
            (1.0 - is_first)
            * prev_above_threshold
            * (1.0 - output_above_threshold)
        )
        reg_a_gt_reg_b = torch.sigmoid((next_a - next_b) / 0.25)
        reg_a_lt_reg_b = torch.sigmoid((next_b - next_a) / 0.25)
        output_gt_reg_b = torch.sigmoid((output - next_b) / 0.25)
        output_lt_reg_b = torch.sigmoid((next_b - output) / 0.25)
        prev_reg_a_gt_reg_b = torch.sigmoid((reg_a - reg_b) / 0.25)
        prev_reg_a_lt_reg_b = torch.sigmoid((reg_b - reg_a) / 0.25)
        prev_output_gt_reg_b = torch.sigmoid((old_output - reg_b) / 0.25)
        prev_output_lt_reg_b = torch.sigmoid((reg_b - old_output) / 0.25)
        reg_a_crosses_above_reg_b = (
            (1.0 - is_first)
            * reg_a_gt_reg_b
            * (1.0 - prev_reg_a_gt_reg_b)
        )
        reg_a_crosses_below_reg_b = (
            (1.0 - is_first)
            * reg_a_lt_reg_b
            * (1.0 - prev_reg_a_lt_reg_b)
        )
        output_crosses_above_reg_b = (
            (1.0 - is_first)
            * output_gt_reg_b
            * (1.0 - prev_output_gt_reg_b)
        )
        output_crosses_below_reg_b = (
            (1.0 - is_first)
            * output_lt_reg_b
            * (1.0 - prev_output_lt_reg_b)
        )
        reg_diff_above_threshold = torch.sigmoid(
            ((next_a - next_b) - self.emit_threshold) / 0.25
        )
        prev_reg_diff_above_threshold = torch.sigmoid(
            ((reg_a - reg_b) - self.emit_threshold) / 0.25
        )
        reg_diff_crosses_above_threshold = (
            (1.0 - is_first)
            * reg_diff_above_threshold
            * (1.0 - prev_reg_diff_above_threshold)
        )
        output_diff_above_threshold = torch.sigmoid(
            ((output - next_b) - self.emit_threshold) / 0.25
        )
        prev_output_diff_above_threshold = torch.sigmoid(
            ((old_output - reg_b) - self.emit_threshold) / 0.25
        )
        output_diff_crosses_above_threshold = (
            (1.0 - is_first)
            * output_diff_above_threshold
            * (1.0 - prev_output_diff_above_threshold)
        )

        emit_candidates = torch.stack([
            torch.ones_like(output),          # 0: always
            torch.sigmoid(inp / 0.25),        # 1: input > 0
            torch.sigmoid(-inp / 0.25),       # 2: input < 0
            torch.sigmoid(output / 0.25),     # 3: output > 0
            torch.sigmoid(next_a / 0.25),     # 4: reg_a > 0
            torch.sigmoid(next_b / 0.25),     # 5: reg_b > 0
            is_first,                         # 6: first only
            torch.zeros_like(output),         # 7: never
            is_first + (1.0 - is_first) * output_changed,   # 8: first or output changed
            is_first + (1.0 - is_first) * output_increased, # 9: first or output increased
            is_first + (1.0 - is_first) * output_decreased, # 10: first or output decreased
            output_above_threshold,           # 11: output > threshold
            crosses_above_threshold,          # 12: output crosses above threshold
            crosses_below_threshold,          # 13: output crosses below threshold
            reg_a_gt_reg_b,                   # 14: reg_a > reg_b
            reg_a_lt_reg_b,                   # 15: reg_a < reg_b
            output_gt_reg_b,                  # 16: output > reg_b
            output_lt_reg_b,                  # 17: output < reg_b
            reg_a_crosses_above_reg_b,        # 18: reg_a crosses above reg_b
            reg_a_crosses_below_reg_b,        # 19: reg_a crosses below reg_b
            output_crosses_above_reg_b,       # 20: output crosses above reg_b
            output_crosses_below_reg_b,       # 21: output crosses below reg_b
            reg_diff_above_threshold,         # 22: reg_a - reg_b > threshold
            reg_diff_crosses_above_threshold, # 23: reg_a - reg_b crosses above threshold
            output_diff_above_threshold,      # 24: output - reg_b > threshold
            output_diff_crosses_above_threshold,
                                               # 25: output - reg_b crosses above threshold
        ])
        emit_w = F.softmax(self.emit_logits / temperature, dim=0)
        emit_prob = (emit_w * emit_candidates).sum()
        emit_prob = torch.clamp(emit_prob, 1e-6, 1.0 - 1e-6)
        return next_a, next_b, output, emit_prob


# ---------------------------------------------------------------------------
# SoftLatent programs: vectorized differentiable recursive ambiguity solvers
# ---------------------------------------------------------------------------

class SoftLatentOutputProgram(nn.Module):
    """Differentiable latent output rediscovery from (reg_a, reg_b)."""

    NUM_OUTPUT_CANDIDATES = 5

    def __init__(self):
        super().__init__()
        self.out_logits = nn.Parameter(torch.zeros(self.NUM_OUTPUT_CANDIDATES))

    def forward(self, features: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        reg_a = features[..., 0]
        reg_b = features[..., 1]
        safe_div = reg_a / torch.where(
            torch.abs(reg_b) < 1e-3,
            torch.ones_like(reg_b),
            reg_b,
        )
        candidates = torch.stack([
            reg_a,
            reg_b,
            reg_a + reg_b,
            reg_a - reg_b,
            safe_div,
        ], dim=-1)
        out_w = F.softmax(self.out_logits / temperature, dim=0)
        return (candidates * out_w).sum(dim=-1)


class SoftLatentEmitProgram(nn.Module):
    """Differentiable emit rediscovery from latent stream features."""

    NUM_EMIT_CANDIDATES = 26

    def __init__(self):
        super().__init__()
        self.emit_logits = nn.Parameter(torch.zeros(self.NUM_EMIT_CANDIDATES))
        self.emit_threshold = nn.Parameter(torch.tensor(0.0))

    def forward(self, features: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        inp = features[..., 0]
        reg_a = features[..., 1]
        reg_b = features[..., 2]
        output = features[..., 3]
        prev_output = features[..., 4]
        prev_reg_a = features[..., 5]
        prev_reg_b = features[..., 6]
        is_first = features[..., 7]

        delta_output = output - prev_output
        output_changed = 1.0 - torch.exp(-(delta_output ** 2) / 0.125)
        output_increased = torch.sigmoid(delta_output / 0.25)
        output_decreased = torch.sigmoid(-delta_output / 0.25)
        output_above_threshold = torch.sigmoid(
            (output - self.emit_threshold) / 0.25
        )
        prev_above_threshold = torch.sigmoid(
            (prev_output - self.emit_threshold) / 0.25
        )
        crosses_above_threshold = (
            (1.0 - is_first)
            * output_above_threshold
            * (1.0 - prev_above_threshold)
        )
        crosses_below_threshold = (
            (1.0 - is_first)
            * prev_above_threshold
            * (1.0 - output_above_threshold)
        )
        reg_a_gt_reg_b = torch.sigmoid((reg_a - reg_b) / 0.25)
        reg_a_lt_reg_b = torch.sigmoid((reg_b - reg_a) / 0.25)
        output_gt_reg_b = torch.sigmoid((output - reg_b) / 0.25)
        output_lt_reg_b = torch.sigmoid((reg_b - output) / 0.25)
        prev_reg_a_gt_reg_b = torch.sigmoid((prev_reg_a - prev_reg_b) / 0.25)
        prev_reg_a_lt_reg_b = torch.sigmoid((prev_reg_b - prev_reg_a) / 0.25)
        prev_output_gt_reg_b = torch.sigmoid((prev_output - prev_reg_b) / 0.25)
        prev_output_lt_reg_b = torch.sigmoid((prev_reg_b - prev_output) / 0.25)
        reg_a_crosses_above_reg_b = (
            (1.0 - is_first)
            * reg_a_gt_reg_b
            * (1.0 - prev_reg_a_gt_reg_b)
        )
        reg_a_crosses_below_reg_b = (
            (1.0 - is_first)
            * reg_a_lt_reg_b
            * (1.0 - prev_reg_a_lt_reg_b)
        )
        output_crosses_above_reg_b = (
            (1.0 - is_first)
            * output_gt_reg_b
            * (1.0 - prev_output_gt_reg_b)
        )
        output_crosses_below_reg_b = (
            (1.0 - is_first)
            * output_lt_reg_b
            * (1.0 - prev_output_lt_reg_b)
        )
        reg_diff_above_threshold = torch.sigmoid(
            ((reg_a - reg_b) - self.emit_threshold) / 0.25
        )
        prev_reg_diff_above_threshold = torch.sigmoid(
            ((prev_reg_a - prev_reg_b) - self.emit_threshold) / 0.25
        )
        reg_diff_crosses_above_threshold = (
            (1.0 - is_first)
            * reg_diff_above_threshold
            * (1.0 - prev_reg_diff_above_threshold)
        )
        output_diff_above_threshold = torch.sigmoid(
            ((output - reg_b) - self.emit_threshold) / 0.25
        )
        prev_output_diff_above_threshold = torch.sigmoid(
            ((prev_output - prev_reg_b) - self.emit_threshold) / 0.25
        )
        output_diff_crosses_above_threshold = (
            (1.0 - is_first)
            * output_diff_above_threshold
            * (1.0 - prev_output_diff_above_threshold)
        )

        emit_candidates = torch.stack([
            torch.ones_like(output),
            torch.sigmoid(inp / 0.25),
            torch.sigmoid(-inp / 0.25),
            torch.sigmoid(output / 0.25),
            torch.sigmoid(reg_a / 0.25),
            torch.sigmoid(reg_b / 0.25),
            is_first,
            torch.zeros_like(output),
            is_first + (1.0 - is_first) * output_changed,
            is_first + (1.0 - is_first) * output_increased,
            is_first + (1.0 - is_first) * output_decreased,
            output_above_threshold,
            crosses_above_threshold,
            crosses_below_threshold,
            reg_a_gt_reg_b,
            reg_a_lt_reg_b,
            output_gt_reg_b,
            output_lt_reg_b,
            reg_a_crosses_above_reg_b,
            reg_a_crosses_below_reg_b,
            output_crosses_above_reg_b,
            output_crosses_below_reg_b,
            reg_diff_above_threshold,
            reg_diff_crosses_above_threshold,
            output_diff_above_threshold,
            output_diff_crosses_above_threshold,
        ], dim=-1)
        emit_w = F.softmax(self.emit_logits / temperature, dim=0)
        emit_prob = (emit_candidates * emit_w).sum(dim=-1)
        return torch.clamp(emit_prob, 1e-6, 1.0 - 1e-6)


# ---------------------------------------------------------------------------
# SoftInteractiveFilterProgram: differentiable subsequence emission discovery
# ---------------------------------------------------------------------------

class SoftInteractiveFilterProgram(nn.Module):
    """Differentiable passthrough filter over an input stream.

    Learns an emit predicate over each input item. When discretized, the
    program emits the input itself if the predicate holds, otherwise it skips
    that item and produces no output.
    """

    NUM_CONDITIONS = 7

    def __init__(self):
        super().__init__()
        self.condition_logits = nn.Parameter(torch.zeros(self.NUM_CONDITIONS))

    def forward(self, inp: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        abs_rounded = torch.round(torch.abs(inp))
        parity = torch.remainder(abs_rounded, 2.0)
        is_even = torch.exp(-(parity ** 2) / 0.125)
        is_zero = torch.exp(-(inp ** 2) / 0.125)
        conditions = torch.stack([
            torch.sigmoid(inp / 0.25),         # 0: x > 0
            torch.sigmoid(-inp / 0.25),        # 1: x < 0
            is_zero,                           # 2: x == 0
            is_even,                           # 3: x % 2 == 0
            1.0 - is_even,                     # 4: x % 2 != 0
            torch.ones_like(inp),              # 5: always emit
            torch.zeros_like(inp),             # 6: never emit
        ])
        cond_w = F.softmax(self.condition_logits / temperature, dim=0)
        return (cond_w * conditions).sum()


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_soft_program(
    prog: nn.Module,
    examples: Sequence[tuple[tuple[float, ...], float]],
    steps: int = 1000,
    lr: float = 0.05,
    temp_start: float = 2.0,
    temp_end: float = 0.1,
    seed: int = 0,
) -> float:
    """Train a soft program on I/O examples. Returns best loss."""
    torch.manual_seed(seed)
    work_items = steps * max(sum(max(len(args), 1) for args, _ in examples), 1)
    device = optimized_training_device(work_items=work_items, sequential=True)
    prog = prog.to(device)
    opt = torch.optim.Adam(prog.parameters(), lr=lr)
    best = float("inf")
    for step in range(steps):
        t = temp_start + (temp_end - temp_start) * (step / max(steps - 1, 1))
        losses = []
        for args, target in examples:
            x = torch.tensor(args, dtype=torch.float32, device=device)
            y = _scalar_tensor(float(target), device=device)
            pred = prog(x, temperature=t)
            losses.append((pred - y) ** 2)
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        cur = float(loss.item())
        if cur < best:
            best = cur
        if cur < 1e-8:
            break
    return best


def _train_vectorized_program(
    prog: nn.Module,
    xs: torch.Tensor,
    ys: torch.Tensor,
    *,
    steps: int = 1000,
    lr: float = 0.05,
    temp_start: float = 2.0,
    temp_end: float = 0.1,
    seed: int = 0,
    sequential: bool = False,
) -> float:
    torch.manual_seed(seed)
    work_items = steps * max(xs.shape[0] * max(xs.shape[-1], 1), 1)
    device = optimized_training_device(work_items=work_items, sequential=sequential)
    prog = prog.to(device)
    xs = xs.to(device)
    ys = ys.to(device)
    opt = torch.optim.Adam(prog.parameters(), lr=lr)
    best = float("inf")
    for step in range(steps):
        t = temp_start + (temp_end - temp_start) * (step / max(steps - 1, 1))
        pred = prog(xs, temperature=t)
        loss = ((pred - ys) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        cur = float(loss.item())
        if cur < best:
            best = cur
        if cur < 1e-8:
            break
    return best


def train_interactive_program(
    prog: SoftInteractiveProgram,
    traces: list[list[tuple[int, int]]],
    steps: int = 1000,
    lr: float = 0.05,
    temp_start: float = 2.0,
    temp_end: float = 0.1,
    seed: int = 0,
) -> float:
    """Train an interactive program on I/O traces."""
    torch.manual_seed(seed)
    work_items = steps * max(sum(len(trace) for trace in traces), 1)
    device = optimized_training_device(work_items=work_items, sequential=True)
    prog = prog.to(device)
    opt = torch.optim.Adam(prog.parameters(), lr=lr)
    best = float("inf")
    for step in range(steps):
        t = temp_start + (temp_end - temp_start) * (step / max(steps - 1, 1))
        losses = []
        for trace in traces:
            state = _scalar_tensor(0.0, device=device)
            for idx, (inp, expected) in enumerate(trace):
                state = prog(
                    state,
                    _scalar_tensor(float(inp), device=device),
                    is_first=_scalar_tensor(1.0 if idx == 0 else 0.0, device=device),
                    temperature=t,
                )
                losses.append((state - _scalar_tensor(expected, device=device)) ** 2)
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        cur = float(loss.item())
        if cur < best:
            best = cur
        if cur < 1e-8:
            break
    return best


def train_interactive_pair_program(
    prog: SoftInteractivePairProgram,
    traces: list[list[tuple[int, int, int]]],
    steps: int = 1000,
    lr: float = 0.05,
    temp_start: float = 2.0,
    temp_end: float = 0.1,
    seed: int = 0,
) -> float:
    """Train a buffered pairwise reducer on grouped interactive traces."""
    torch.manual_seed(seed)
    work_items = steps * max(sum(len(trace) for trace in traces), 1)
    device = optimized_training_device(work_items=work_items, sequential=True)
    prog = prog.to(device)
    opt = torch.optim.Adam(prog.parameters(), lr=lr)
    best = float("inf")
    for step in range(steps):
        t = temp_start + (temp_end - temp_start) * (step / max(steps - 1, 1))
        losses = []
        for trace in traces:
            for a, b, expected in trace:
                pred = prog(
                    _scalar_tensor(float(a), device=device),
                    _scalar_tensor(float(b), device=device),
                    temperature=t,
                )
                losses.append((pred - _scalar_tensor(expected, device=device)) ** 2)
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        cur = float(loss.item())
        if cur < best:
            best = cur
        if cur < 1e-8:
            break
    return best


def _softmin(values: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    return -temperature * torch.logsumexp(-values / temperature, dim=0)


def train_interactive_state_emit_program(
    prog: SoftInteractiveStateEmitProgram,
    traces: list[tuple[list[int], list[int]]],
    steps: int = 1000,
    lr: float = 0.05,
    temp_start: float = 2.0,
    temp_end: float = 0.1,
    seed: int = 0,
) -> float:
    """Train a sparse stateful transducer with differentiable subsequence alignment."""
    torch.manual_seed(seed)
    total_events = sum(len(inputs) * max(len(expected_outputs), 1) for inputs, expected_outputs in traces)
    device = optimized_training_device(work_items=steps * max(total_events, 1), sequential=True)
    prog = prog.to(device)
    opt = torch.optim.Adam(prog.parameters(), lr=lr)
    best = float("inf")
    inf = 1e6
    for step in range(steps):
        t = temp_start + (temp_end - temp_start) * (step / max(steps - 1, 1))
        losses = []
        for inputs, expected_outputs in traces:
            state = _scalar_tensor(0.0, device=device)
            dp = [_scalar_tensor(0.0, device=device)] + [
                _scalar_tensor(inf, device=device) for _ in expected_outputs
            ]
            for idx, inp in enumerate(inputs):
                new_state, emit_prob = prog(
                    state,
                    _scalar_tensor(float(inp), device=device),
                    is_first=_scalar_tensor(1.0 if idx == 0 else 0.0, device=device),
                    temperature=t,
                )
                skip_cost = -torch.log(torch.clamp(1.0 - emit_prob, min=1e-6))
                emit_base_cost = -torch.log(torch.clamp(emit_prob, min=1e-6))
                next_dp = [dp[0] + skip_cost]
                for j, expected in enumerate(expected_outputs, start=1):
                    target = _scalar_tensor(float(expected), device=device)
                    emit_cost = emit_base_cost + (new_state - target) ** 2
                    next_dp.append(
                        _softmin(torch.stack([
                            dp[j] + skip_cost,
                            dp[j - 1] + emit_cost,
                        ]))
                    )
                dp = next_dp
                state = new_state
            losses.append(dp[len(expected_outputs)])
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        cur = float(loss.item())
        if cur < best:
            best = cur
        if cur < 1e-8:
            break
    return best


def train_interactive_two_register_program(
    prog: SoftInteractiveTwoRegisterProgram,
    traces: list[list[tuple[int, int]]],
    steps: int = 1000,
    lr: float = 0.05,
    temp_start: float = 2.0,
    temp_end: float = 0.1,
    seed: int = 0,
) -> float:
    """Train a dense two-register transducer on 1:1 interactive traces."""
    torch.manual_seed(seed)
    work_items = steps * max(sum(len(trace) for trace in traces), 1)
    device = optimized_training_device(work_items=work_items, sequential=True)
    prog = prog.to(device)
    opt = torch.optim.Adam(prog.parameters(), lr=lr)
    best = float("inf")
    for step in range(steps):
        t = temp_start + (temp_end - temp_start) * (step / max(steps - 1, 1))
        losses = []
        for trace in traces:
            reg_a = _scalar_tensor(0.0, device=device)
            reg_b = _scalar_tensor(0.0, device=device)
            for idx, (inp, expected) in enumerate(trace):
                reg_a, reg_b, pred = prog(
                    reg_a,
                    reg_b,
                    _scalar_tensor(float(inp), device=device),
                    is_first=_scalar_tensor(1.0 if idx == 0 else 0.0, device=device),
                    temperature=t,
                )
                losses.append((pred - _scalar_tensor(expected, device=device)) ** 2)
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        cur = float(loss.item())
        if cur < best:
            best = cur
        if cur < 1e-8:
            break
    return best


def train_interactive_two_register_emit_program(
    prog: SoftInteractiveTwoRegisterEmitProgram,
    traces: list[tuple[list[int], list[int]]],
    steps: int = 1000,
    lr: float = 0.05,
    temp_start: float = 2.0,
    temp_end: float = 0.1,
    seed: int = 0,
) -> float:
    """Train a sparse two-register transducer with differentiable subsequence alignment."""
    torch.manual_seed(seed)
    total_events = sum(len(inputs) * max(len(expected_outputs), 1) for inputs, expected_outputs in traces)
    device = optimized_training_device(work_items=steps * max(total_events, 1), sequential=True)
    prog = prog.to(device)
    opt = torch.optim.Adam(prog.parameters(), lr=lr)
    best = float("inf")
    inf = 1e6
    for step in range(steps):
        t = temp_start + (temp_end - temp_start) * (step / max(steps - 1, 1))
        losses = []
        for inputs, expected_outputs in traces:
            reg_a = _scalar_tensor(0.0, device=device)
            reg_b = _scalar_tensor(0.0, device=device)
            dp = [_scalar_tensor(0.0, device=device)] + [
                _scalar_tensor(inf, device=device) for _ in expected_outputs
            ]
            for idx, inp in enumerate(inputs):
                reg_a, reg_b, output, emit_prob = prog(
                    reg_a,
                    reg_b,
                    _scalar_tensor(float(inp), device=device),
                    is_first=_scalar_tensor(1.0 if idx == 0 else 0.0, device=device),
                    temperature=t,
                )
                skip_cost = -torch.log(torch.clamp(1.0 - emit_prob, min=1e-6))
                emit_base_cost = -torch.log(torch.clamp(emit_prob, min=1e-6))
                next_dp = [dp[0] + skip_cost]
                for j, expected in enumerate(expected_outputs, start=1):
                    target = _scalar_tensor(float(expected), device=device)
                    emit_cost = emit_base_cost + (output - target) ** 2
                    next_dp.append(
                        _softmin(torch.stack([
                            dp[j] + skip_cost,
                            dp[j - 1] + emit_cost,
                        ]))
                    )
                dp = next_dp
            losses.append(dp[len(expected_outputs)])
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        cur = float(loss.item())
        if cur < best:
            best = cur
        if cur < 1e-8:
            break
    return best


def train_interactive_filter_program(
    prog: SoftInteractiveFilterProgram,
    inputs: list[list[int]],
    emit_targets: list[list[int]],
    steps: int = 1000,
    lr: float = 0.05,
    temp_start: float = 2.0,
    temp_end: float = 0.1,
    seed: int = 0,
) -> float:
    """Train a passthrough interactive filter on per-input emit targets."""
    torch.manual_seed(seed)
    work_items = steps * max(sum(len(stream) for stream in inputs), 1)
    device = optimized_training_device(work_items=work_items, sequential=True)
    prog = prog.to(device)
    opt = torch.optim.Adam(prog.parameters(), lr=lr)
    best = float("inf")
    for step in range(steps):
        t = temp_start + (temp_end - temp_start) * (step / max(steps - 1, 1))
        losses = []
        for stream, targets in zip(inputs, emit_targets, strict=True):
            for inp, target in zip(stream, targets, strict=True):
                emit_prob = prog(_scalar_tensor(float(inp), device=device), temperature=t)
                losses.append((emit_prob - _scalar_tensor(float(target), device=device)) ** 2)
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        cur = float(loss.item())
        if cur < best:
            best = cur
        if cur < 1e-8:
            break
    return best


def train_latent_output_program(
    prog: SoftLatentOutputProgram,
    examples: Sequence[tuple[tuple[float, float], float]],
    steps: int = 1000,
    lr: float = 0.05,
    temp_start: float = 2.0,
    temp_end: float = 0.1,
    seed: int = 0,
) -> float:
    xs = torch.tensor([list(args) for args, _ in examples], dtype=torch.float32)
    ys = torch.tensor([float(target) for _, target in examples], dtype=torch.float32)
    return _train_vectorized_program(
        prog,
        xs,
        ys,
        steps=steps,
        lr=lr,
        temp_start=temp_start,
        temp_end=temp_end,
        seed=seed,
    )


def train_latent_emit_program(
    prog: SoftLatentEmitProgram,
    examples: Sequence[tuple[tuple[float, ...], float]],
    steps: int = 1000,
    lr: float = 0.05,
    temp_start: float = 2.0,
    temp_end: float = 0.1,
    seed: int = 0,
) -> float:
    xs = torch.tensor([list(args) for args, _ in examples], dtype=torch.float32)
    ys = torch.tensor([float(target) for _, target in examples], dtype=torch.float32)
    return _train_vectorized_program(
        prog,
        xs,
        ys,
        steps=steps,
        lr=lr,
        temp_start=temp_start,
        temp_end=temp_end,
        seed=seed,
    )
