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


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

OPS = ["+", "-", "*", "/", "%"]
CMP_OPS = [">", "<", ">=", "<=", "==", "!="]


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

        # Body: acc = acc BODY_OP f(i)
        # f(i) is a soft combination of: i, i*i, 1, args[k]
        self.body_acc_op = nn.Parameter(torch.zeros(len(OPS)))
        self.body_i_expr_weights = nn.Parameter(torch.zeros(4))  # i, i*i, 1, const

        # Pre-loop compute: optional v0 = src1 OP src2
        self.pre_compute_enable = nn.Parameter(torch.tensor(0.0))
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

        for step in range(self.MAX_ITER):
            i_val = torch.tensor(float(step), device=device)
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

    NUM_CANDIDATES = 7

    def __init__(self):
        super().__init__()
        # Selection over independent candidate outputs
        self.mode_logits = nn.Parameter(torch.zeros(self.NUM_CANDIDATES))
        # Learnable constants for some candidates
        self.const_a = nn.Parameter(torch.tensor(1.0))
        self.const_b = nn.Parameter(torch.tensor(0.0))

    def forward(self, state: torch.Tensor, inp: torch.Tensor,
                temperature: float = 1.0) -> torch.Tensor:
        # Each candidate is computed independently — no gradient cross-talk
        candidates = torch.stack([
            state + inp,                    # 0: running sum
            state - inp,                    # 1: running difference
            state * inp,                    # 2: running product
            inp,                            # 3: passthrough
            state + self.const_a,           # 4: counter (state + 1)
            inp * self.const_a,             # 5: scale input
            self.const_b,                   # 6: constant output
        ])
        mode_w = F.softmax(self.mode_logits / temperature, dim=0)
        return (mode_w * candidates).sum()


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
    opt = torch.optim.Adam(prog.parameters(), lr=lr)
    best = float("inf")
    for step in range(steps):
        t = temp_start + (temp_end - temp_start) * (step / max(steps - 1, 1))
        losses = []
        for args, target in examples:
            x = torch.tensor(args, dtype=torch.float32)
            y = torch.tensor(float(target), dtype=torch.float32)
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
    opt = torch.optim.Adam(prog.parameters(), lr=lr)
    best = float("inf")
    for step in range(steps):
        t = temp_start + (temp_end - temp_start) * (step / max(steps - 1, 1))
        losses = []
        for trace in traces:
            state = torch.tensor(0.0)
            for inp, expected in trace:
                state = prog(state, torch.tensor(float(inp)), temperature=t)
                losses.append((state - expected) ** 2)
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
