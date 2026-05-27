#!/usr/bin/env python3
"""Train the Direct Neural ALU: single-pass MLP for 32-bit arithmetic.

Replaces 160 sequential MLP calls (Kogge-Stone CLA) with ONE forward pass.
Supports ADD, SUB, AND, OR, XOR on full 32-bit unsigned integers.

Training strategy:
  1. Curriculum: 8-bit -> 16-bit -> 32-bit values
  2. Hard-example mining: focus on long carry chains after initial convergence
  3. Per-bit accuracy tracking: monitor carry propagation learning
  4. Separate BCE loss for result bits and flags

Usage:
    python -m ncpu.neural.train_direct_alu --device mps --hidden 512 --n-blocks 4
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Inline model + utilities (avoid import issues during training) ──────────

OP_NAMES = ["add", "sub", "and", "or", "xor"]
NUM_OPS = len(OP_NAMES)
NUM_OP_CLASSES = 8  # 5 used + 3 reserved


def int_to_bits(x: torch.Tensor, n_bits: int = 32) -> torch.Tensor:
    """(N,) int64 -> (N, n_bits) float32, LSB first."""
    shifts = torch.arange(n_bits, dtype=torch.int64, device=x.device)
    return ((x.unsqueeze(1) >> shifts.unsqueeze(0)) & 1).float()


def bits_to_int(bits: torch.Tensor) -> torch.Tensor:
    """(N, n_bits) float32 -> (N,) int64 unsigned."""
    n_bits = bits.shape[1]
    weights = (1 << torch.arange(n_bits, dtype=torch.int64, device=bits.device))
    return ((bits > 0.5).long() * weights).sum(dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class DirectNeuralALU(nn.Module):
    def __init__(self, hidden: int = 512, n_blocks: int = 4):
        super().__init__()
        self.hidden = hidden
        self.n_blocks = n_blocks
        self.input_proj = nn.Linear(72, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden) for _ in range(n_blocks)])
        self.result_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 32),
        )
        self.flags_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, 4),
        )

    def forward(self, a_bits, b_bits, op_code):
        x = torch.cat([a_bits, b_bits, op_code], dim=1)
        h = F.gelu(self.input_proj(x))
        for block in self.blocks:
            h = block(h)
        return torch.cat([self.result_head(h), self.flags_head(h)], dim=1)


# ─── Data generation ─────────────────────────────────────────────────────────


def compute_flags(a: torch.Tensor, b: torch.Tensor, result: torch.Tensor,
                  op_idx: torch.Tensor) -> torch.Tensor:
    """Compute ARM64-style N, Z, C, V flags.

    Args:
        a, b: (N,) int64, unsigned 32-bit values
        result: (N,) int64, unsigned 32-bit result
        op_idx: (N,) int64, operation index

    Returns:
        (N, 4) float32: [N, Z, C, V]
    """
    N_flag = ((result >> 31) & 1).float()
    Z_flag = (result == 0).float()

    # Carry flag: for ADD, carry out of bit 31; for SUB, NOT borrow
    # For logical ops, C is typically 0
    is_add = (op_idx == 0).long()
    is_sub = (op_idx == 1).long()

    # ADD carry: (a + b) > 0xFFFFFFFF
    add_sum = a + b
    add_carry = (add_sum > 0xFFFFFFFF).float()

    # SUB carry (ARM: carry = NOT borrow): a >= b
    sub_carry = (a >= b).float()

    C_flag = add_carry * is_add.float() + sub_carry * is_sub.float()

    # Overflow flag (signed): sign(a) == sign(b) != sign(result) for ADD
    # For SUB: sign(a) != sign(b) and sign(result) != sign(a)
    a_sign = (a >> 31) & 1
    b_sign = (b >> 31) & 1
    r_sign = (result >> 31) & 1

    add_overflow = ((a_sign == b_sign) & (a_sign != r_sign)).float()
    sub_overflow = ((a_sign != b_sign) & (r_sign != a_sign)).float()

    V_flag = add_overflow * is_add.float() + sub_overflow * is_sub.float()

    return torch.stack([N_flag, Z_flag, C_flag, V_flag], dim=1)


def generate_batch(
    batch_size: int,
    max_bits: int = 32,
    device: torch.device = torch.device("cpu"),
    hard_fraction: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a training batch.

    Args:
        batch_size: number of samples
        max_bits: maximum bit width for values (8, 16, or 32 for curriculum)
        device: target device
        hard_fraction: fraction of batch devoted to hard carry-chain examples

    Returns:
        a_bits (batch, 32), b_bits (batch, 32), op_onehot (batch, 8),
        result_bits (batch, 32), flags (batch, 4)
    """
    max_val = min(2**max_bits, 2**32)

    n_hard = int(batch_size * hard_fraction)
    n_random = batch_size - n_hard

    # Random values
    a_rand = torch.randint(0, max_val, (n_random,), dtype=torch.int64)
    b_rand = torch.randint(0, max_val, (n_random,), dtype=torch.int64)

    if n_hard > 0:
        # Hard examples: long carry chains
        a_hard, b_hard = _generate_hard_examples(n_hard, max_bits)
        a = torch.cat([a_rand, a_hard])
        b = torch.cat([b_rand, b_hard])
    else:
        a = a_rand
        b = b_rand

    op_idx = torch.randint(0, NUM_OPS, (batch_size,), dtype=torch.int64)

    # Compute ground truth results (vectorized per op)
    result = torch.zeros(batch_size, dtype=torch.int64)
    mask32 = 0xFFFFFFFF

    for op_i, op_name in enumerate(OP_NAMES):
        mask = op_idx == op_i
        if not mask.any():
            continue
        am, bm = a[mask], b[mask]
        if op_name == "add":
            result[mask] = (am + bm) & mask32
        elif op_name == "sub":
            result[mask] = (am - bm) & mask32
        elif op_name == "and":
            result[mask] = am & bm
        elif op_name == "or":
            result[mask] = am | bm
        elif op_name == "xor":
            result[mask] = am ^ bm

    # Compute flags
    flags = compute_flags(a, b, result, op_idx)

    # Convert to bit vectors
    a_bits = int_to_bits(a, 32).to(device)
    b_bits = int_to_bits(b, 32).to(device)
    r_bits = int_to_bits(result, 32).to(device)
    op_onehot = F.one_hot(op_idx, num_classes=NUM_OP_CLASSES).float().to(device)
    flags = flags.to(device)

    return a_bits, b_bits, op_onehot, r_bits, flags


def _generate_hard_examples(n: int, max_bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate examples with long carry chains for ADD/SUB."""
    max_val = min(2**max_bits, 2**32)
    a = torch.zeros(n, dtype=torch.int64)
    b = torch.zeros(n, dtype=torch.int64)

    quarter = n // 4

    # Pattern 1: 0xFFF...F + small  (max carry propagation)
    ones = min(max_val - 1, 0xFFFFFFFF)
    a[:quarter] = ones
    b[:quarter] = torch.randint(1, min(256, max_val), (quarter,), dtype=torch.int64)

    # Pattern 2: 0x7FFF...F + 1..N  (carry into sign bit)
    half_ones = min(max_val // 2 - 1, 0x7FFFFFFF)
    a[quarter:2*quarter] = half_ones
    b[quarter:2*quarter] = torch.randint(1, min(256, max_val), (quarter,), dtype=torch.int64)

    # Pattern 3: alternating bits  (e.g. 0xAAAAAAAA + 0x55555555)
    idx = 2 * quarter
    end = 3 * quarter
    if max_bits >= 32:
        a[idx:end] = 0xAAAAAAAA
        b[idx:end] = 0x55555555
    else:
        mask = (1 << max_bits) - 1
        a[idx:end] = 0xAAAAAAAA & mask
        b[idx:end] = 0x55555555 & mask

    # Pattern 4: powers of two boundaries
    rest = n - 3 * quarter
    for i in range(rest):
        bit_pos = torch.randint(0, max_bits, (1,)).item()
        a[3*quarter + i] = (1 << bit_pos) - 1
        b[3*quarter + i] = 1

    return a, b


# ─── Exhaustive 8-bit validation ─────────────────────────────────────────────


def validate_exhaustive_8bit(
    model: DirectNeuralALU, device: torch.device, verbose: bool = True
) -> dict:
    """Test all 256x256 input pairs for all 5 ops (327,680 total tests).

    Returns dict with per-op accuracy and overall accuracy.
    """
    model.eval()
    results = {}
    total_correct = 0
    total_tests = 0

    all_a = torch.arange(256, dtype=torch.int64)
    all_b = torch.arange(256, dtype=torch.int64)
    # Cartesian product: 65536 pairs
    grid_a = all_a.repeat_interleave(256)
    grid_b = all_b.repeat(256)

    a_bits = int_to_bits(grid_a, 32).to(device)
    b_bits = int_to_bits(grid_b, 32).to(device)

    with torch.no_grad():
        for op_i, op_name in enumerate(OP_NAMES):
            op_oh = F.one_hot(
                torch.full((65536,), op_i, dtype=torch.int64, device=device),
                num_classes=NUM_OP_CLASSES,
            ).float()

            # Compute ground truth
            if op_name == "add":
                gt = (grid_a + grid_b) & 0xFFFFFFFF
            elif op_name == "sub":
                gt = (grid_a - grid_b) & 0xFFFFFFFF
            elif op_name == "and":
                gt = grid_a & grid_b
            elif op_name == "or":
                gt = grid_a | grid_b
            elif op_name == "xor":
                gt = grid_a ^ grid_b

            gt_bits = int_to_bits(gt, 32).to(device)

            # Forward pass (in chunks to fit in memory)
            chunk = 16384
            pred_bits_list = []
            for start in range(0, 65536, chunk):
                end = min(start + chunk, 65536)
                out = model(a_bits[start:end], b_bits[start:end], op_oh[start:end])
                pred_bits_list.append((out[:, :32] > 0.0).float())

            pred_bits = torch.cat(pred_bits_list, dim=0)
            pred_int = bits_to_int(pred_bits.cpu())
            gt_int = gt

            correct = (pred_int == gt_int).sum().item()
            acc = correct / 65536
            results[op_name] = acc
            total_correct += correct
            total_tests += 65536

            if verbose:
                print(f"  8-bit exhaustive {op_name.upper():3s}: {correct}/65536 = {acc*100:.2f}%")

    overall = total_correct / total_tests
    results["overall"] = overall
    if verbose:
        print(f"  8-bit exhaustive OVERALL: {total_correct}/{total_tests} = {overall*100:.2f}%")

    return results


def validate_random_32bit(
    model: DirectNeuralALU,
    device: torch.device,
    n_samples: int = 100_000,
    verbose: bool = True,
) -> dict:
    """Validate on random 32-bit pairs. Returns per-op and overall accuracy."""
    model.eval()
    results = {}
    total_correct = 0
    total_tests = 0

    a = torch.randint(0, 2**32, (n_samples,), dtype=torch.int64)
    b = torch.randint(0, 2**32, (n_samples,), dtype=torch.int64)

    a_bits = int_to_bits(a, 32).to(device)
    b_bits = int_to_bits(b, 32).to(device)

    chunk = 16384

    with torch.no_grad():
        for op_i, op_name in enumerate(OP_NAMES):
            op_oh = F.one_hot(
                torch.full((n_samples,), op_i, dtype=torch.int64, device=device),
                num_classes=NUM_OP_CLASSES,
            ).float()

            if op_name == "add":
                gt = (a + b) & 0xFFFFFFFF
            elif op_name == "sub":
                gt = (a - b) & 0xFFFFFFFF
            elif op_name == "and":
                gt = a & b
            elif op_name == "or":
                gt = a | b
            elif op_name == "xor":
                gt = a ^ b

            pred_bits_list = []
            for start in range(0, n_samples, chunk):
                end = min(start + chunk, n_samples)
                out = model(a_bits[start:end], b_bits[start:end], op_oh[start:end])
                pred_bits_list.append((out[:, :32] > 0.0).float())

            pred_bits = torch.cat(pred_bits_list, dim=0)
            pred_int = bits_to_int(pred_bits.cpu())

            correct = (pred_int == gt).sum().item()
            acc = correct / n_samples
            results[op_name] = acc
            total_correct += correct
            total_tests += n_samples

            if verbose:
                wrong = n_samples - correct
                print(f"  32-bit random {op_name.upper():3s}: {correct}/{n_samples} = {acc*100:.4f}%"
                      f" ({wrong} wrong)")

    overall = total_correct / total_tests
    results["overall"] = overall
    if verbose:
        print(f"  32-bit random OVERALL: {total_correct}/{total_tests} = {overall*100:.4f}%")

    return results


def per_bit_accuracy(
    model: DirectNeuralALU, device: torch.device, n_samples: int = 50_000
) -> torch.Tensor:
    """Compute accuracy for each of the 32 output bits across all ops.

    Returns: (32,) float tensor of per-bit accuracy.
    """
    model.eval()
    correct_per_bit = torch.zeros(32)
    total = 0

    a = torch.randint(0, 2**32, (n_samples,), dtype=torch.int64)
    b = torch.randint(0, 2**32, (n_samples,), dtype=torch.int64)
    op_idx = torch.randint(0, NUM_OPS, (n_samples,), dtype=torch.int64)

    result = torch.zeros(n_samples, dtype=torch.int64)
    mask32 = 0xFFFFFFFF
    for oi, name in enumerate(OP_NAMES):
        m = op_idx == oi
        if not m.any():
            continue
        am, bm = a[m], b[m]
        if name == "add":   result[m] = (am + bm) & mask32
        elif name == "sub": result[m] = (am - bm) & mask32
        elif name == "and": result[m] = am & bm
        elif name == "or":  result[m] = am | bm
        elif name == "xor": result[m] = am ^ bm

    gt_bits = int_to_bits(result, 32)
    a_bits = int_to_bits(a, 32).to(device)
    b_bits = int_to_bits(b, 32).to(device)
    op_oh = F.one_hot(op_idx.to(device), num_classes=NUM_OP_CLASSES).float()

    chunk = 16384
    with torch.no_grad():
        for start in range(0, n_samples, chunk):
            end = min(start + chunk, n_samples)
            out = model(a_bits[start:end], b_bits[start:end], op_oh[start:end])
            pred = (out[:, :32] > 0.0).float().cpu()
            gt_chunk = gt_bits[start:end]
            correct_per_bit += (pred == gt_chunk).float().sum(dim=0)
            total += end - start

    return correct_per_bit / total


# ─── Training loop ────────────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Architecture: DirectNeuralALU(hidden={args.hidden}, n_blocks={args.n_blocks})")

    model = DirectNeuralALU(hidden=args.hidden, n_blocks=args.n_blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Resume from checkpoint if requested
    start_step = 0
    if args.resume:
        ckpt_path = Path(args.save_dir) / "direct_alu.pt"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state["model_state_dict"])
            start_step = state.get("step", 0)
            print(f"Resumed from checkpoint at step {start_step}")
            print(f"  Previous 32-bit acc: {state.get('val32_overall', 0)*100:.4f}%")
        else:
            print(f"WARNING: --resume specified but no checkpoint at {ckpt_path}")

    lr = args.lr
    if args.refine:
        lr = args.lr * 0.1  # Lower LR for refinement
        print(f"Refinement mode: lr={lr:.1e}, skip curriculum, 32-bit + hard examples only")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=lr * 0.01
    )

    # Curriculum schedule: (step_threshold, max_bits, hard_fraction)
    if args.refine:
        # Refinement: skip curriculum, go straight to 32-bit with heavy hard mining
        curriculum = [
            (0, 32, 0.4),   # 40% hard examples from the start
        ]
    else:
        curriculum = [
            (0, 8, 0.0),           # Phase 1: 8-bit only
            (2000, 12, 0.0),       # Phase 2: 12-bit
            (5000, 16, 0.0),       # Phase 3: 16-bit
            (10000, 24, 0.0),      # Phase 4: 24-bit
            (15000, 32, 0.0),      # Phase 5: full 32-bit
            (25000, 32, 0.2),      # Phase 6: 32-bit + 20% hard examples
            (40000, 32, 0.3),      # Phase 7: 32-bit + 30% hard examples
        ]

    best_acc_32 = 0.0
    perfect_streak = 0
    save_path = Path(args.save_dir) / "direct_alu.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for up to {args.max_steps} steps, batch_size={args.batch_size}")
    print(f"Save path: {save_path}")
    print("=" * 80)

    t0 = time.time()
    running_loss = 0.0
    running_bit_acc = 0.0
    running_flag_acc = 0.0

    for step in range(1, args.max_steps + 1):
        model.train()

        # Determine curriculum phase
        max_bits = 8
        hard_frac = 0.0
        for thresh, mb, hf in curriculum:
            if step >= thresh:
                max_bits, hard_frac = mb, hf

        # Generate batch
        a_bits, b_bits, op_oh, gt_bits, gt_flags = generate_batch(
            args.batch_size, max_bits=max_bits, device=device, hard_fraction=hard_frac
        )

        # Forward
        out = model(a_bits, b_bits, op_oh)
        result_logits = out[:, :32]
        flag_logits = out[:, 32:]

        # Loss: BCE on result bits + flags
        loss_result = F.binary_cross_entropy_with_logits(result_logits, gt_bits)
        loss_flags = F.binary_cross_entropy_with_logits(flag_logits, gt_flags)
        loss = loss_result + args.flag_weight * loss_flags

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Track metrics
        with torch.no_grad():
            pred_bits = (result_logits > 0.0).float()
            bit_acc = (pred_bits == gt_bits).float().mean().item()
            flag_acc = ((flag_logits > 0.0).float() == gt_flags).float().mean().item()

        running_loss += loss.item()
        running_bit_acc += bit_acc
        running_flag_acc += flag_acc

        # Logging
        if step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            avg_bit = running_bit_acc / args.log_every
            avg_flag = running_flag_acc / args.log_every
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[{step:6d}/{args.max_steps}]  loss={avg_loss:.4f}  "
                f"bit_acc={avg_bit*100:.2f}%  flag_acc={avg_flag*100:.2f}%  "
                f"bits={max_bits}b  hard={hard_frac:.0%}  lr={lr_now:.2e}  "
                f"t={elapsed:.0f}s"
            )
            running_loss = 0.0
            running_bit_acc = 0.0
            running_flag_acc = 0.0

        # Validation
        if step % args.val_every == 0:
            print(f"\n--- Validation at step {step} ---")

            # 8-bit exhaustive
            val8 = validate_exhaustive_8bit(model, device, verbose=True)

            # 32-bit random
            val32 = validate_random_32bit(model, device, n_samples=100_000, verbose=True)

            # Per-bit accuracy
            pba = per_bit_accuracy(model, device, n_samples=50_000)
            lowest_bits = pba.argsort()[:5]
            print(f"  Lowest per-bit acc: "
                  + ", ".join(f"bit{i}={pba[i]*100:.1f}%" for i in lowest_bits))

            # Save best
            if val32["overall"] > best_acc_32:
                best_acc_32 = val32["overall"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "step": step,
                        "hidden": args.hidden,
                        "n_blocks": args.n_blocks,
                        "val8_overall": val8["overall"],
                        "val32_overall": val32["overall"],
                        "per_op_32": {k: v for k, v in val32.items() if k != "overall"},
                    },
                    save_path,
                )
                print(f"  ** Saved new best: 32-bit acc = {best_acc_32*100:.4f}%")

            # Early stopping check
            if val8["overall"] >= 1.0 and val32["overall"] >= 0.999:
                perfect_streak += 1
                print(f"  Perfect streak: {perfect_streak}/{args.early_stop_patience}")
                if perfect_streak >= args.early_stop_patience:
                    print(f"\nEarly stop: {args.early_stop_patience} consecutive near-perfect validations.")
                    break
            else:
                perfect_streak = 0

            print("---\n")

    # Final validation
    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"Training complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Best 32-bit accuracy: {best_acc_32*100:.4f}%")

    # Reload best and do final exhaustive validation
    if save_path.exists():
        state = torch.load(save_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        print(f"\nLoaded best checkpoint (step {state['step']})")

    print("\n--- Final 8-bit exhaustive validation ---")
    validate_exhaustive_8bit(model, device, verbose=True)

    print("\n--- Final 32-bit random validation (500K samples) ---")
    validate_random_32bit(model, device, n_samples=500_000, verbose=True)

    print(f"\n--- Final per-bit accuracy (100K samples) ---")
    pba = per_bit_accuracy(model, device, n_samples=100_000)
    for i in range(32):
        bar = "#" * int(pba[i] * 50)
        print(f"  bit[{i:2d}] {pba[i]*100:6.2f}% {bar}")

    print(f"\nModel saved to: {save_path}")
    print(f"Parameters: {n_params:,}")


def main():
    parser = argparse.ArgumentParser(description="Train Direct Neural ALU")
    parser.add_argument("--device", default="mps", help="Device (mps, cuda, cpu)")
    parser.add_argument("--hidden", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--n-blocks", type=int, default=4, help="Number of residual blocks")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=200_000, help="Max training steps")
    parser.add_argument("--flag-weight", type=float, default=0.5, help="Weight for flag loss")
    parser.add_argument("--log-every", type=int, default=200, help="Log every N steps")
    parser.add_argument("--val-every", type=int, default=5000, help="Validate every N steps")
    parser.add_argument("--early-stop-patience", type=int, default=5,
                        help="Stop after N consecutive near-perfect validations")
    parser.add_argument("--save-dir", default="models/alu", help="Save directory")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--refine", action="store_true",
                        help="Refinement mode: skip curriculum, low LR, heavy hard mining")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
