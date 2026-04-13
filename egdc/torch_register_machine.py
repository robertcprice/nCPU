#!/usr/bin/env python3
"""
torch_register_machine.py — PyTorch differentiable register machine.

The Rust FD-gradient register machine has 600+ params and needs 1200 forward
passes per optimization step. PyTorch autograd does 1 forward + 1 backward = 2.
That's a 600x speedup. This makes the universal register machine actually viable.

Architecture:
  - N registers = n_args + 6 constants + 6 scratch
  - K instruction steps (default 8)
  - Each step: soft-select op/src1/src2/dst + conditional gate
  - 10 ops: +, -, *, /, %, |a-b|, max, min, neg, identity
  - Soft write: reg[i] = dst_w[i] * result + (1-dst_w[i]) * reg[i]
  - Fully differentiable via PyTorch autograd

After training, exports params to JSON for Rust discretization, or discretizes
directly to Mog code in Python.

Usage:
    from egdc.torch_register_machine import TorchRegisterMachine, train_rm

    rm = TorchRegisterMachine(n_args=2, n_steps=8)
    examples = [([3, 4], 7), ([1, 2], 3), ([0, 0], 0)]
    success, code = train_rm(rm, examples, fn_name="add", n_epochs=2000)
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Constants ────────────────────────────────────────────────────────────────

N_OPS = 10          # +, -, *, /, %, |a-b|, max, min, neg, identity
N_CONSTS = 6        # [0, 1, -1, 2, -2, 10]
N_SCRATCH = 6
N_CMPS = 6          # <, <=, ==, >=, >, !=
CONSTS = [0.0, 1.0, -1.0, 2.0, -2.0, 10.0]
OP_NAMES = ["+", "-", "*", "/", "%", "abs_diff", "max", "min", "neg", "id"]
CMP_NAMES = ["<", "<=", "==", ">=", ">", "!="]


# ─── Differentiable operations ────────────────────────────────────────────────

def soft_op(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted mix of 10 operations. Fully differentiable."""
    safe_b = torch.where(b.abs() < 1e-6, torch.ones_like(b), b)
    ops = torch.stack([
        a + b,                          # 0: +
        a - b,                          # 1: -
        a * b,                          # 2: *
        a / safe_b,                     # 3: /
        a - (a / safe_b).trunc() * safe_b,  # 4: %
        (a - b).abs(),                  # 5: |a-b|
        torch.maximum(a, b),            # 6: max
        torch.minimum(a, b),            # 7: min
        -a,                             # 8: neg
        a,                              # 9: identity
    ])
    return (weights * ops).sum()


def soft_cmp(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor, temp: float) -> torch.Tensor:
    """Soft comparison returning [0,1]. Differentiable."""
    d = a - b
    t = max(temp, 0.5)
    gauss_var = max(t * t * 0.5, 0.125)
    results = torch.stack([
        torch.sigmoid(-d / t),              # <
        torch.sigmoid(-d / t),              # <=
        torch.exp(-d * d / gauss_var),      # ==
        torch.sigmoid(d / t),               # >=
        torch.sigmoid(d / t),               # >
        1.0 - torch.exp(-d * d / gauss_var),  # !=
    ])
    return (weights * results).sum()


# ─── Register Machine ────────────────────────────────────────────────────────

class TorchRegisterMachine(nn.Module):
    """Universal differentiable register machine with PyTorch autograd."""

    def __init__(self, n_args: int, n_steps: int = 8, n_scratch: int = N_SCRATCH):
        super().__init__()
        self.n_args = n_args
        self.n_steps = n_steps
        self.n_scratch = n_scratch
        self.n_regs = n_args + N_CONSTS + n_scratch

        nr = self.n_regs

        # Per-step parameters: op(10) + src1(nr) + src2(nr) + dst(nr) + gate_cmp(6) + gate_s1(nr) + gate_s2(nr)
        self.step_params = nn.ParameterList([
            nn.Parameter(torch.randn(N_OPS + 5 * nr + N_CMPS) * 0.1)
            for _ in range(n_steps)
        ])

        # Return logits
        self.ret_logits = nn.Parameter(torch.zeros(nr))

        # Learnable constants (initialized to standard values)
        self.consts = nn.Parameter(torch.tensor(CONSTS, dtype=torch.float32))

        # Initialize: bias steps toward identity (nop)
        for sp in self.step_params:
            sp.data[9] = 2.0  # op = identity
            sp.data[N_OPS + 3 * nr + 2] = 2.0  # gate_cmp = == (always true)

    def forward(self, inputs: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        """Forward pass. inputs shape: (n_args,). Returns scalar.

        Uses Gumbel-softmax with straight-through estimator when temp < 0.5
        to force discrete-like behavior that matches discretization.
        """
        nr = self.n_regs

        # Initialize register file
        regs = torch.zeros(nr)
        regs[:self.n_args] = inputs
        regs[self.n_args:self.n_args + N_CONSTS] = self.consts

        def _gumbel_sm(logits: torch.Tensor) -> torch.Tensor:
            """Gumbel-softmax with straight-through when temp is low."""
            soft = F.softmax(logits / max(temp, 0.05), dim=0)
            if temp < 0.5 and self.training:
                # Straight-through: use hard argmax in forward, soft in backward
                hard = torch.zeros_like(soft)
                hard[soft.argmax()] = 1.0
                return hard - soft.detach() + soft  # STE trick
            return soft

        # Execute instruction sequence
        for step_idx in range(self.n_steps):
            sp = self.step_params[step_idx]

            op_w = _gumbel_sm(sp[:N_OPS])
            s1_w = _gumbel_sm(sp[N_OPS:N_OPS + nr])
            s2_w = _gumbel_sm(sp[N_OPS + nr:N_OPS + 2*nr])
            dst_w = _gumbel_sm(sp[N_OPS + 2*nr:N_OPS + 3*nr])
            gate_cmp_w = _gumbel_sm(sp[N_OPS + 3*nr:N_OPS + 3*nr + N_CMPS])
            gate_s1_w = _gumbel_sm(sp[N_OPS + 3*nr + N_CMPS:N_OPS + 4*nr + N_CMPS])
            gate_s2_w = _gumbel_sm(sp[N_OPS + 4*nr + N_CMPS:N_OPS + 5*nr + N_CMPS])

            # Read operands
            v1 = (regs * s1_w).sum()
            v2 = (regs * s2_w).sum()

            # Compute
            result = soft_op(v1, v2, op_w)

            # Gate
            g_lhs = (regs * gate_s1_w).sum()
            g_rhs = (regs * gate_s2_w).sum()
            gate = soft_cmp(g_lhs, g_rhs, gate_cmp_w, temp)

            # Soft write
            regs = dst_w * gate * result + (1.0 - dst_w * gate) * regs

        # Return
        ret_w = _gumbel_sm(self.ret_logits)
        return (regs * ret_w).sum()

    def forward_batch(self, batch_inputs: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        """Forward on a batch. batch_inputs: (B, n_args). Returns (B,)."""
        return torch.stack([self.forward(inp, temp) for inp in batch_inputs])

    def discretize(self, fn_name: str, param_names: list[str] | None = None) -> str:
        """Convert trained soft program to discrete Mog code."""
        nr = self.n_regs
        if param_names is None:
            param_names = ["a", "b", "c", "d", "e", "f"][:self.n_args]

        consts = [int(round(c.item())) for c in self.consts]
        reg_names = list(param_names) + [str(c) for c in consts] + [f"r{i}" for i in range(self.n_scratch)]

        sig = ", ".join(f"{n}: i64" for n in param_names)
        lines = [f"fn {fn_name}({sig}) -> i64 {{"]

        scratch_declared = set()

        for step_idx in range(self.n_steps):
            sp = self.step_params[step_idx].data

            op_i = sp[:N_OPS].argmax().item()
            s1_i = sp[N_OPS:N_OPS + nr].argmax().item()
            s2_i = sp[N_OPS + nr:N_OPS + 2*nr].argmax().item()
            dst_i = sp[N_OPS + 2*nr:N_OPS + 3*nr].argmax().item()
            gate_cmp_i = sp[N_OPS + 3*nr:N_OPS + 3*nr + N_CMPS].argmax().item()
            gate_s1_i = sp[N_OPS + 3*nr + N_CMPS:N_OPS + 4*nr + N_CMPS].argmax().item()
            gate_s2_i = sp[N_OPS + 4*nr + N_CMPS:N_OPS + 5*nr + N_CMPS].argmax().item()

            # Skip nops
            if op_i == 9 and dst_i == s1_i:
                continue
            # Skip writes to non-scratch
            if dst_i < self.n_args + N_CONSTS:
                continue

            dst = reg_names[dst_i]
            s1 = reg_names[s1_i]
            s2 = reg_names[s2_i]

            # Build expression
            if op_i <= 4:
                expr = f"{s1} {OP_NAMES[op_i]} {s2}"
            elif op_i == 5:  # abs_diff
                expr = f"if {s1} > {s2} {{ {s1} - {s2} }} else {{ {s2} - {s1} }}"
            elif op_i == 6:  # max
                expr = f"if {s1} > {s2} {{ {s1} }} else {{ {s2} }}"
            elif op_i == 7:  # min
                expr = f"if {s1} < {s2} {{ {s1} }} else {{ {s2} }}"
            elif op_i == 8:  # neg
                expr = f"0 - {s1}"
            else:  # identity
                expr = s1

            # Gated?
            is_gated = gate_s1_i != gate_s2_i or gate_cmp_i != 2
            decl = ": i64 " if dst not in scratch_declared else " "
            scratch_declared.add(dst)

            if is_gated:
                gs1, gs2 = reg_names[gate_s1_i], reg_names[gate_s2_i]
                lines.append(f"    if {gs1} {CMP_NAMES[gate_cmp_i]} {gs2} {{ {dst}{decl}= {expr}; }}")
            else:
                lines.append(f"    {dst}{decl}= {expr};")

        ret_i = self.ret_logits.data.argmax().item()
        lines.append(f"    return {reg_names[ret_i]};")
        lines.append("}")
        return "\n".join(lines) + "\n"


# ─── Training ─────────────────────────────────────────────────────────────────

def train_rm(
    rm: TorchRegisterMachine,
    examples: list[tuple[list[int | float], int | float]],
    fn_name: str = "f",
    n_epochs: int = 3000,
    lr: float = 0.01,
    temp_schedule: tuple[float, float] = (2.0, 0.05),
    verify_fn: Any = None,
    verbose: bool = False,
) -> tuple[bool, str]:
    """Train the register machine on I/O examples.

    Returns (solved, code) where solved=True if discretized code passes verification.
    """
    optimizer = torch.optim.Adam(rm.parameters(), lr=lr)
    inputs_t = torch.tensor([[float(x) for x in ex[0]] for ex in examples], dtype=torch.float32)
    targets_t = torch.tensor([float(ex[1]) for ex in examples], dtype=torch.float32)
    n = len(examples)

    best_loss = float("inf")
    best_state = None
    t0 = time.time()

    for epoch in range(n_epochs):
        # Temperature annealing
        progress = epoch / max(n_epochs - 1, 1)
        temp = temp_schedule[0] * (1 - progress) + temp_schedule[1] * progress

        # Forward
        preds = rm.forward_batch(inputs_t, temp)
        mse_loss = F.mse_loss(preds, targets_t)

        # Entropy regularization: encourage sharp (one-hot-like) distributions
        entropy_penalty = 0.0
        if progress > 0.3:  # Start after 30% of training
            for sp in rm.step_params:
                nr = rm.n_regs
                for logits_slice in [sp[:N_OPS], sp[N_OPS:N_OPS+nr], sp[N_OPS+nr:N_OPS+2*nr], sp[N_OPS+2*nr:N_OPS+3*nr]]:
                    probs = F.softmax(logits_slice / max(temp, 0.1), dim=0)
                    entropy_penalty += -(probs * (probs + 1e-8).log()).sum()
            # Scale: small penalty, grows with progress
            entropy_penalty *= 0.01 * progress

        loss = mse_loss + entropy_penalty

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 5.0)
        optimizer.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.clone() for k, v in rm.state_dict().items()}

        # Periodic discretization check
        if loss_val < 1.0 or epoch % 200 == 199:
            code = rm.discretize(fn_name)
            if verify_fn and verify_fn(code):
                if verbose:
                    print(f"  [epoch {epoch}] loss={loss_val:.4f} SOLVED in {time.time()-t0:.1f}s")
                return True, code

        if verbose and epoch % 500 == 0:
            print(f"  [epoch {epoch}] loss={loss_val:.4f} temp={temp:.2f}")

        # Early stopping
        if epoch == n_epochs // 4:
            loss_at_25 = best_loss
        if epoch == n_epochs // 2 and best_loss > loss_at_25 * 0.95:
            break

    # Final check with best params
    if best_state:
        rm.load_state_dict(best_state)
    code = rm.discretize(fn_name)
    if verify_fn and verify_fn(code):
        return True, code

    return False, code


# ─── Mog verification ────────────────────────────────────────────────────────

def make_mog_verifier(examples: list[tuple], signature: str, mog_binary: str | None = None) -> Any:
    """Create a verification function that checks Mog code against examples."""
    if mog_binary is None:
        candidates = [
            Path(__file__).parent.parent / "mog_synth" / "target" / "release" / "mog_synth",
            Path(__file__).parent.parent / "mog_synth" / "target" / "debug" / "mog_synth",
        ]
        for c in candidates:
            if c.exists():
                mog_binary = str(c)
                break

    if not mog_binary:
        return None

    def verify(code: str) -> bool:
        """Verify by sending to mog_synth --problem-json and checking the code against examples."""
        # Direct Python verification: parse the Mog code and execute mentally
        # Actually, just call mog_synth with the code as reference and check
        fn_name_v = code.split("(")[0].replace("fn ", "").strip()
        problem_json = {
            "name": fn_name_v,
            "signature": signature or f"fn {fn_name_v}(a: i64) -> i64",
            "examples": [{"inputs": list(a) if isinstance(a, (list,tuple)) else [a], "expected": int(e)} for a, e in examples],
            "holdouts": [],
        }
        # Hack: set reference_code to the candidate and solve — if template_reference works, it's valid
        # But we can't set reference_code via JSON. Instead, verify by executing each example.
        try:
            # Use Python-based verification: parse and execute the Mog-like code
            return _verify_mog_python(code, examples)
        except Exception:
            return False

    return verify


def _verify_mog_python(code: str, examples: list[tuple]) -> bool:
    """Quick Python-based verification of simple Mog expressions.

    Handles: return a OP b, if/else patterns, variable assignments.
    Falls back to False if the code is too complex to parse.
    """
    import re
    # Extract function body
    m = re.search(r'fn\s+\w+\([^)]*\)\s*->\s*i64\s*\{(.*)\}', code, re.DOTALL)
    if not m:
        return False
    body = m.group(1).strip()

    # Build a Python function from the Mog code
    fn_match = re.match(r'fn\s+(\w+)\(([^)]*)\)', code)
    if not fn_match:
        return False
    fn_name = fn_match.group(1)
    params = [p.strip().split(":")[0].strip() for p in fn_match.group(2).split(",") if p.strip()]

    # Convert Mog to Python
    py_body = body
    # Replace Mog-style variable declarations
    py_body = re.sub(r'(\w+)\s*:\s*i64\s*=', r'\1 =', py_body)
    py_body = re.sub(r'(\w+)\s*:=', r'\1 =', py_body)
    # Mog uses { } for blocks, Python uses :
    # Simple approach: just try to eval each statement
    py_lines = []
    for line in py_body.split(";"):
        line = line.strip().rstrip(";").strip()
        if not line or line == "{" or line == "}":
            continue
        # Handle return
        if line.startswith("return "):
            py_lines.append(line)
        # Handle if/else (simple single-statement)
        elif line.startswith("if "):
            # Convert: if a < b { return b; } -> if a < b: return b
            line = line.replace("{", ":").replace("}", "").strip()
            if line.endswith(":"):
                continue  # incomplete
            py_lines.append(line)
        else:
            py_lines.append(line)

    # Build function string
    py_func = f"def {fn_name}({', '.join(params)}):\n"
    for line in py_lines:
        py_func += f"    {line}\n"

    # Execute and test
    try:
        ns = {}
        exec(compile(py_func, "<mog>", "exec"), ns)
        fn = ns[fn_name]
        for args, expected in examples:
            if isinstance(args, (list, tuple)):
                result = fn(*args)
            else:
                result = fn(args)
            if int(result) != int(expected):
                return False
        return True
    except Exception:
        return False


# ─── High-level synthesis ─────────────────────────────────────────────────────

def synthesize_with_torch_rm(
    fn_name: str,
    examples: list[tuple[list, int]],
    signature: str | None = None,
    n_steps: int = 8,
    n_epochs: int = 3000,
    n_restarts: int = 5,
    verbose: bool = False,
) -> tuple[bool, str, float]:
    """Synthesize a program using the PyTorch register machine.

    Returns (solved, code, time_s).
    """
    n_args = len(examples[0][0]) if isinstance(examples[0][0], (list, tuple)) else 1
    # Normalize examples
    norm_examples = []
    for args, expected in examples:
        if isinstance(args, (list, tuple)):
            norm_examples.append((list(args), expected))
        else:
            norm_examples.append(([args], expected))

    verifier = make_mog_verifier(norm_examples, signature or f"fn {fn_name}(a: i64) -> i64")

    t0 = time.time()
    for restart in range(n_restarts):
        rm = TorchRegisterMachine(n_args=n_args, n_steps=n_steps)
        # Add random perturbation for restarts > 0
        if restart > 0:
            with torch.no_grad():
                for p in rm.parameters():
                    p.add_(torch.randn_like(p) * 0.5)

        solved, code = train_rm(
            rm, norm_examples, fn_name=fn_name,
            n_epochs=n_epochs, verify_fn=verifier, verbose=verbose,
        )
        if solved:
            return True, code, time.time() - t0

    return False, code, time.time() - t0


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Demo: synthesize add_two
    print("PyTorch Register Machine — Demo")
    print("=" * 50)

    demos = [
        ("add_two", [([1, 2], 3), ([3, 4], 7), ([0, 0], 0), ([-1, 5], 4), ([10, 20], 30)]),
        ("square", [([2], 4), ([3], 9), ([0], 0), ([-3], 9), ([5], 25), ([1], 1)]),
        ("abs_diff", [([5, 3], 2), ([3, 5], 2), ([0, 0], 0), ([-2, 3], 5), ([10, 7], 3)]),
        ("triple", [([1], 3), ([2], 6), ([0], 0), ([-1], -3), ([5], 15), ([10], 30)]),
        ("max2", [([3, 5], 5), ([5, 3], 5), ([0, 0], 0), ([-1, -3], -1), ([10, 10], 10)]),
    ]

    for fn_name, examples in demos:
        print(f"\n--- {fn_name} ---")
        solved, code, elapsed = synthesize_with_torch_rm(
            fn_name, examples, n_steps=6, n_epochs=2000, n_restarts=3, verbose=True,
        )
        print(f"{'SOLVED' if solved else 'FAILED'} in {elapsed:.1f}s")
        if code:
            print(code)
