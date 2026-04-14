#!/usr/bin/env python3
"""
warm_start_synth.py — Meta-learner guided warm-start synthesis.

Uses the trained program-type classifier to predict which synthesis type
will work, then provides intelligent initialization for that type.

This is the bridge between the meta-learner (predicts type) and the
differentiable synthesizer (finds exact params within that type).

Architecture:
  1. Meta-learner predicts: "this is a loop/branch/expr problem"
  2. Type-specific warm starter generates good initial params
  3. Rust synthesizer runs with those initial params (fewer restarts needed)
  4. Result: faster convergence, fewer wasted cycles

Also includes a PyTorch-based nested loop synthesizer for O(n²) programs
that the Rust synthesizer can't handle yet.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Meta-learner integration ────────────────────────────────────────────────

def load_meta_learner():
    """Load the trained program-type classifier."""
    model_path = Path(__file__).parent.parent / "mog_synth" / "models" / "expr_type_classifier.pt"
    if not model_path.exists():
        return None
    sys.path.insert(0, str(Path(__file__).parent.parent / "mog_synth" / "scripts"))
    from train_expr_metalearner import load_model
    return load_model(str(model_path))


def predict_type(meta_learner, examples: list[tuple]) -> tuple[str, dict[str, float]]:
    """Predict program type from I/O examples."""
    n_args = len(examples[0][0]) if isinstance(examples[0][0], (list, tuple)) else 1
    io_pairs = []
    for args, exp in examples[:8]:
        if isinstance(args, (list, tuple)):
            io_pairs.append([list(args), exp])
        else:
            io_pairs.append([[args], exp])
    return meta_learner.predict(io_pairs, n_args)


# ─── Nested Loop Synthesizer (PyTorch) ───────────────────────────────────────
#
# For O(n²) programs like bubble_sort_count, insertion_sort_shifts, etc.
# Structure:
#   acc = init
#   for i in 0..n:
#     for j in 0..bound(i, n):
#       val = arr[soft_idx(i, j)]
#       cmp_val = arr[soft_idx2(i, j)]
#       gate = soft_cmp(val, cmp_val)
#       acc += gate * rhs(val, cmp_val, acc)
#   return acc

class SoftNestedLoop(nn.Module):
    """Differentiable nested loop for O(n²) array programs."""

    def __init__(self, n_scalar: int = 0, max_arr: int = 16):
        super().__init__()
        self.n_scalar = n_scalar
        self.max_arr = max_arr

        # Init accumulator from pool [0, 1, -1, arr[0], arr_len, scalar_args...]
        init_pool = 5 + n_scalar
        self.init_logits = nn.Parameter(torch.zeros(init_pool))
        self.init_logits.data[0] = 2.0  # bias toward 0

        # Inner loop bound: j < i, j < n, j < n-i, etc.
        # Parameterized as: bound = soft_select(i, n, n-i, i+1)
        self.bound_logits = nn.Parameter(torch.zeros(4))
        self.bound_logits.data[1] = 2.0  # bias toward n

        # Array index for inner read: soft_select(i, j, i+j, n-1-j)
        self.idx1_logits = nn.Parameter(torch.zeros(4))
        self.idx1_logits.data[1] = 2.0  # bias toward j

        # Second array index (for comparison): soft_select(j, j+1, i, 0)
        self.idx2_logits = nn.Parameter(torch.zeros(4))
        self.idx2_logits.data[1] = 2.0  # bias toward j+1

        # Comparison: <, <=, ==, >=, >, !=
        self.cmp_logits = nn.Parameter(torch.zeros(6))

        # Accumulator update: op(acc, rhs) where rhs from pool
        self.op_logits = nn.Parameter(torch.zeros(5))  # +, -, *, /, %
        self.op_logits.data[0] = 2.0  # bias toward +

        # RHS source: 1, val, cmp_val, val-cmp_val, acc
        self.rhs_logits = nn.Parameter(torch.zeros(5))
        self.rhs_logits.data[0] = 2.0  # bias toward 1 (counting)

        # Return source: acc, arr_len, 0
        self.ret_logits = nn.Parameter(torch.zeros(3))
        self.ret_logits.data[0] = 2.0  # bias toward acc

        # Learnable constants
        self.consts = nn.Parameter(torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0]))

    def soft_array_read(self, arr: torch.Tensor, idx: torch.Tensor, arr_len: float) -> torch.Tensor:
        """Differentiable array read at soft index."""
        positions = torch.arange(self.max_arr, dtype=torch.float32)
        weights = torch.exp(-(positions - idx) ** 2 / 0.5)
        # Mask out-of-bounds
        in_bounds = torch.sigmoid((arr_len - positions - 0.5) / 0.3)
        weights = weights * in_bounds
        weights = weights / (weights.sum() + 1e-8)
        return (arr * weights).sum()

    def forward(self, arr: torch.Tensor, arr_len: float, scalar_args: list[float], temp: float = 1.0) -> torch.Tensor:
        """Execute nested loop on array."""
        n = arr_len

        # Init
        init_pool = torch.stack([
            self.consts[0],  # 0
            self.consts[1],  # 1
            self.consts[2],  # -1
            arr[0] if arr.numel() > 0 else torch.tensor(0.0),
            torch.tensor(n),
        ] + [torch.tensor(s) for s in scalar_args[:self.n_scalar]])
        init_w = F.softmax(self.init_logits / temp, dim=0)
        acc = (init_pool * init_w).sum()

        for i_step in range(self.max_arr):
            i = float(i_step)
            i_in = torch.sigmoid(torch.tensor((n - i - 0.5) / 0.3))
            if i_in.item() < 0.01:
                break

            # Inner loop bound
            bounds = torch.stack([
                torch.tensor(i),
                torch.tensor(n),
                torch.tensor(n - i),
                torch.tensor(i + 1.0),
            ])
            bound_w = F.softmax(self.bound_logits / temp, dim=0)
            inner_bound = (bounds * bound_w).sum()

            for j_step in range(self.max_arr):
                j = float(j_step)
                j_in = torch.sigmoid((inner_bound - j - 0.5) / 0.3) * i_in
                if j_in.item() < 0.01:
                    break

                # Array indices
                indices1 = torch.stack([
                    torch.tensor(i), torch.tensor(j),
                    torch.tensor(i + j), torch.tensor(max(0, n - 1 - j)),
                ])
                idx1_w = F.softmax(self.idx1_logits / temp, dim=0)
                idx1 = (indices1 * idx1_w).sum()

                indices2 = torch.stack([
                    torch.tensor(j), torch.tensor(min(j + 1, n - 1)),
                    torch.tensor(i), torch.tensor(0.0),
                ])
                idx2_w = F.softmax(self.idx2_logits / temp, dim=0)
                idx2 = (indices2 * idx2_w).sum()

                val = self.soft_array_read(arr, idx1, n)
                cmp_val = self.soft_array_read(arr, idx2, n)

                # Comparison gate
                d = val - cmp_val
                t_cmp = max(temp, 0.5)
                gv = max(t_cmp * t_cmp * 0.5, 0.125)
                cmp_results = torch.stack([
                    torch.sigmoid(-d / t_cmp),
                    torch.sigmoid(-d / t_cmp),
                    torch.exp(-d * d / gv),
                    torch.sigmoid(d / t_cmp),
                    torch.sigmoid(d / t_cmp),
                    1.0 - torch.exp(-d * d / gv),
                ])
                cmp_w = F.softmax(self.cmp_logits / temp, dim=0)
                gate = (cmp_results * cmp_w).sum()

                # RHS
                rhs_pool = torch.stack([
                    self.consts[1],  # 1
                    val,
                    cmp_val,
                    val - cmp_val,
                    acc,
                ])
                rhs_w = F.softmax(self.rhs_logits / temp, dim=0)
                rhs = (rhs_pool * rhs_w).sum()

                # Op
                safe_rhs = torch.where(rhs.abs() < 1e-6, torch.ones_like(rhs), rhs)
                op_results = torch.stack([
                    acc + rhs,
                    acc - rhs,
                    acc * rhs,
                    acc / safe_rhs,
                    acc - (acc / safe_rhs).trunc() * safe_rhs,
                ])
                op_w = F.softmax(self.op_logits / temp, dim=0)
                new_acc = (op_results * op_w).sum()

                acc = j_in * gate * new_acc + (1.0 - j_in * gate) * acc

        # Return
        ret_pool = torch.stack([acc, torch.tensor(n), self.consts[0]])
        ret_w = F.softmax(self.ret_logits / temp, dim=0)
        return (ret_pool * ret_w).sum()


def train_nested_loop(
    examples: list[tuple[list[int], int]],
    fn_name: str = "f",
    n_epochs: int = 2000,
    n_restarts: int = 3,
    verbose: bool = False,
) -> tuple[bool, str, float]:
    """Train a SoftNestedLoop on array I/O examples.

    Examples should be [(arr, expected), ...] where arr is list[int].
    """
    t0 = time.time()

    for restart in range(n_restarts):
        model = SoftNestedLoop()
        if restart > 0:
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 0.3)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        best_loss = float("inf")

        for epoch in range(n_epochs):
            progress = epoch / max(n_epochs - 1, 1)
            temp = 2.0 * (1 - progress) + 0.1 * progress

            total_loss = torch.tensor(0.0)
            for arr_list, expected in examples:
                arr = torch.zeros(16)
                for k, v in enumerate(arr_list[:16]):
                    arr[k] = float(v)
                pred = model(arr, float(len(arr_list)), [], temp)
                diff = pred - float(expected)
                total_loss = total_loss + diff * diff

            loss = total_loss / len(examples)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()

            if verbose and epoch % 500 == 0:
                print(f"  [restart {restart}, epoch {epoch}] loss={loss.item():.4f} temp={temp:.2f}")

            # Early termination
            if epoch == n_epochs // 4:
                loss_25 = best_loss
            if epoch == n_epochs // 2 and best_loss > loss_25 * 0.9:
                break

        if verbose:
            print(f"  restart {restart}: best_loss={best_loss:.4f}")

    elapsed = time.time() - t0
    # TODO: discretize to Mog code
    return False, f"// nested loop training: best_loss={best_loss:.4f}", elapsed


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Warm-Start Synthesis + Nested Loop Demo")
    print("=" * 50)

    # Test meta-learner
    ml = load_meta_learner()
    if ml:
        print("\nMeta-learner predictions:")
        test_cases = [
            ("a+b", [([1, 2], 3), ([3, 4], 7), ([0, 0], 0)]),
            ("factorial", [([1], 1), ([3], 6), ([5], 120), ([7], 5040)]),
            ("clamp", [([5, -3, 10], 5), ([-5, -3, 10], -3), ([15, -3, 10], 10)]),
            ("gcd", [([12, 8], 4), ([7, 3], 1), ([100, 75], 25)]),
        ]
        for name, examples in test_cases:
            pred_type, probs = predict_type(ml, examples)
            conf = max(probs.values())
            print(f"  {name:15} → {pred_type:15} ({conf:.0%})")

    # Test nested loop on bubble_sort_count
    print("\nNested loop training (bubble_sort_count):")
    examples = [
        ([3, 1, 2], 1),      # 1 swap
        ([2, 1], 1),          # 1 swap
        ([1, 2, 3], 0),       # already sorted
        ([3, 2, 1], 3),       # 3 swaps
        ([4, 3, 2, 1], 6),    # 6 swaps
        ([1], 0),             # trivial
    ]
    solved, code, t = train_nested_loop(examples, "bubble_sort_count", n_epochs=1000, verbose=True)
    print(f"  Result: {code} ({t:.1f}s)")
