#!/usr/bin/env python3
"""
🧠 FusedALU Curriculum Training
================================

Uses curriculum learning to progressively train harder operations:

Phase 1: Bitwise ops (AND, OR, XOR) - Each bit independent
Phase 2: Arithmetic (ADD, SUB) - Require carry chain learning
Phase 3: Shifts (LSL, LSR, ROR, ASR) - Positional understanding
Phase 4: Multiplication (MUL) - Full 64-bit multiply

This approach lets the model learn simpler patterns first,
building the foundation for harder operations.

Usage:
    python train_fused_alu_curriculum.py --epochs 50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import os
import sys
import time
import argparse
import random
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

# ============================================================
# DEVICE SETUP
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 Using CUDA: {torch.cuda.get_device_name()}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # H200/Hopper optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    cc = torch.cuda.get_device_capability()
    if cc[0] >= 9:
        print(f"   ✅ Hopper architecture detected (SM {cc[0]}{cc[1]})")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🍎 Using MPS")
else:
    device = torch.device("cpu")
    print("💻 Using CPU")


# ============================================================
# CURRICULUM CONFIGURATION
# ============================================================

@dataclass
class CurriculumPhase:
    """Defines a curriculum training phase."""
    name: str
    ops: List[int]
    epochs: int
    lr: float
    target_accuracy: float
    warmup_epochs: int = 2

CURRICULUM = [
    # Phase 1: Bitwise operations (easiest - each bit independent)
    CurriculumPhase(
        name="Phase 1: Bitwise",
        ops=[2, 3, 4],  # AND, OR, XOR
        epochs=5,
        lr=1e-3,
        target_accuracy=0.99,
        warmup_epochs=1
    ),

    # Phase 2: Add bitwise to maintain, introduce arithmetic
    CurriculumPhase(
        name="Phase 2: Arithmetic",
        ops=[0, 1, 2, 3, 4],  # ADD, SUB + bitwise
        epochs=10,
        lr=5e-4,
        target_accuracy=0.98,
        warmup_epochs=2
    ),

    # Phase 3: Introduce shifts (keep previous ops)
    CurriculumPhase(
        name="Phase 3: Shifts",
        ops=[0, 1, 2, 3, 4, 5, 6, 7, 8],  # All except MUL
        epochs=15,
        lr=3e-4,
        target_accuracy=0.97,
        warmup_epochs=2
    ),

    # Phase 4: Full curriculum with MUL
    CurriculumPhase(
        name="Phase 4: Multiplication",
        ops=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # All ops
        epochs=20,
        lr=1e-4,
        target_accuracy=0.95,
        warmup_epochs=3
    ),
]


# ============================================================
# FUSED ALU MODEL
# ============================================================

class FusedALU(nn.Module):
    """
    Unified neural ALU that handles ALL arithmetic operations.

    Operations:
        0 = ADD, 1 = SUB, 2 = AND, 3 = OR, 4 = XOR,
        5 = LSL, 6 = LSR, 7 = ROR, 8 = ASR, 9 = MUL
    """

    def __init__(self, bit_width: int = 64, d_model: int = 256, nhead: int = 8, num_layers: int = 4):
        super().__init__()
        self.bit_width = bit_width
        self.d_model = d_model

        # Operation embedding (16 ops max)
        self.op_embed = nn.Embedding(16, d_model)

        # Bit position embedding
        self.pos_embed = nn.Embedding(bit_width, d_model)

        # Operand projections
        self.operand_a_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )

        self.operand_b_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )

        # Transformer for bit interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection (no sigmoid - use BCEWithLogitsLoss)
        self.result_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        self.register_buffer('pos_indices', torch.arange(bit_width))

    def forward(self, op: torch.Tensor, a_bits: torch.Tensor, b_bits: torch.Tensor) -> torch.Tensor:
        batch_size = op.shape[0]

        # Embeddings
        op_emb = self.op_embed(op)
        pos_emb = self.pos_embed(self.pos_indices)

        # Project operand bits
        a_proj = self.operand_a_proj(a_bits.unsqueeze(-1))
        b_proj = self.operand_b_proj(b_bits.unsqueeze(-1))

        # Combine
        combined = a_proj + b_proj + pos_emb.unsqueeze(0) + op_emb.unsqueeze(1)

        # Transform
        transformed = self.transformer(combined)

        # Output
        result = self.result_proj(transformed).squeeze(-1)
        return result


# ============================================================
# DATA GENERATION
# ============================================================

MASK64 = (1 << 64) - 1
OP_NAMES = ['ADD', 'SUB', 'AND', 'OR', 'XOR', 'LSL', 'LSR', 'ROR', 'ASR', 'MUL']

def compute_classical(op: int, a: int, b: int) -> int:
    """Ground truth computation."""
    a = a & MASK64
    b = b & MASK64

    if op == 0:  # ADD
        return (a + b) & MASK64
    elif op == 1:  # SUB
        return (a - b) & MASK64
    elif op == 2:  # AND
        return a & b
    elif op == 3:  # OR
        return a | b
    elif op == 4:  # XOR
        return a ^ b
    elif op == 5:  # LSL
        return (a << (b & 63)) & MASK64
    elif op == 6:  # LSR
        return a >> (b & 63)
    elif op == 7:  # ROR
        s = b & 63
        return ((a >> s) | (a << (64 - s))) & MASK64
    elif op == 8:  # ASR
        s = b & 63
        sign = (a >> 63) & 1
        result = a >> s
        if sign and s > 0:
            result |= (MASK64 << (64 - s)) & MASK64
        return result
    elif op == 9:  # MUL
        return (a * b) & MASK64
    return 0


def int_to_bits(values: torch.Tensor, bits: int = 64) -> torch.Tensor:
    """Convert integers to bit tensors."""
    batch = values.shape[0]
    result = torch.zeros(batch, bits, device=values.device, dtype=torch.float32)
    for b in range(bits):
        result[:, b] = ((values >> b) & 1).float()
    return result


def generate_batch(batch_size: int, ops: List[int], difficulty: str = "full") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate training batch with curriculum-aware difficulty."""

    # Random operations from allowed set
    op_codes = torch.tensor([random.choice(ops) for _ in range(batch_size)],
                           dtype=torch.long, device=device)

    # Generate operands based on difficulty
    if difficulty == "easy":
        # Small values - easier for learning
        a_vals = torch.randint(0, 2**16, (batch_size,), dtype=torch.long, device=device)
        b_vals = torch.randint(0, 2**16, (batch_size,), dtype=torch.long, device=device)
    elif difficulty == "medium":
        # Mix of small and medium values
        a_vals = torch.randint(0, 2**32, (batch_size,), dtype=torch.long, device=device)
        b_vals = torch.randint(0, 2**32, (batch_size,), dtype=torch.long, device=device)
    else:  # full
        # Full 64-bit range
        a_high = torch.randint(0, 2**32, (batch_size,), dtype=torch.long, device=device)
        a_low = torch.randint(0, 2**32, (batch_size,), dtype=torch.long, device=device)
        a_vals = (a_high << 32) | a_low

        b_high = torch.randint(0, 2**32, (batch_size,), dtype=torch.long, device=device)
        b_low = torch.randint(0, 2**32, (batch_size,), dtype=torch.long, device=device)
        b_vals = (b_high << 32) | b_low

    # Fix b for shift operations (0-63)
    shift_mask = (op_codes >= 5) & (op_codes <= 8)
    if shift_mask.any():
        b_vals[shift_mask] = torch.randint(0, 64, (shift_mask.sum().item(),),
                                           dtype=torch.long, device=device)

    # Compute ground truth (CPU for arbitrary precision)
    ops_cpu = op_codes.cpu().tolist()
    a_cpu = a_vals.cpu().tolist()
    b_cpu = b_vals.cpu().tolist()

    results = []
    for i in range(batch_size):
        a = a_cpu[i] if a_cpu[i] >= 0 else a_cpu[i] + (1 << 64)
        b = b_cpu[i] if b_cpu[i] >= 0 else b_cpu[i] + (1 << 64)
        r = compute_classical(ops_cpu[i], a, b)
        if r >= (1 << 63):
            r = r - (1 << 64)
        results.append(r)

    results = torch.tensor(results, dtype=torch.long, device=device)

    # Convert to bits
    a_bits = int_to_bits(a_vals)
    b_bits = int_to_bits(b_vals)
    r_bits = int_to_bits(results)

    return op_codes, a_bits, b_bits, r_bits


# ============================================================
# CURRICULUM TRAINING
# ============================================================

def train_phase(
    model: FusedALU,
    phase: CurriculumPhase,
    batch_size: int,
    samples_per_epoch: int,
    save_path: str,
    best_accuracy: float
) -> Tuple[FusedALU, float]:
    """Train a single curriculum phase."""

    print(f"\n{'='*70}")
    print(f"📚 {phase.name}")
    print(f"{'='*70}")
    print(f"   Operations: {[OP_NAMES[i] for i in phase.ops]}")
    print(f"   Epochs: {phase.epochs}")
    print(f"   Learning rate: {phase.lr}")
    print(f"   Target accuracy: {phase.target_accuracy*100:.0f}%")

    # Optimizer with phase-specific learning rate
    optimizer = optim.AdamW(model.parameters(), lr=phase.lr, weight_decay=0.01)

    # Warmup + cosine decay scheduler
    total_steps = phase.epochs * (samples_per_epoch // batch_size)
    warmup_steps = phase.warmup_epochs * (samples_per_epoch // batch_size)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    # Difficulty progression within phase
    difficulties = ["easy"] * 2 + ["medium"] * 3 + ["full"] * (phase.epochs - 5)
    if len(difficulties) < phase.epochs:
        difficulties.extend(["full"] * (phase.epochs - len(difficulties)))

    for epoch in range(phase.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        num_batches = samples_per_epoch // batch_size

        # Get difficulty for this epoch
        difficulty = difficulties[min(epoch, len(difficulties)-1)]

        for batch_idx in range(num_batches):
            # Generate batch with curriculum
            ops_t, a_bits, b_bits, targets = generate_batch(batch_size, phase.ops, difficulty)

            # Forward with mixed precision
            with autocast(dtype=torch.bfloat16):
                outputs = model(ops_t, a_bits, b_bits)
                loss = criterion(outputs, targets)

            # Backward
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 20 == 0:
                print(f"\r   Epoch {epoch+1}/{phase.epochs} | "
                      f"Batch {batch_idx+1}/{num_batches} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Diff: {difficulty} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}", end="", flush=True)

        # Validation
        model.eval()
        with torch.no_grad():
            # Test each op separately
            op_accuracies = {}
            for op in phase.ops:
                val_ops = torch.full((1000,), op, dtype=torch.long, device=device)

                # Full difficulty for validation
                _, val_a, val_b, val_targets = generate_batch(1000, [op], "full")
                val_ops_actual = torch.full_like(val_ops, op)

                with autocast(dtype=torch.bfloat16):
                    val_logits = model(val_ops_actual, val_a, val_b)
                    val_outputs = torch.sigmoid(val_logits)

                predictions = (val_outputs > 0.5).float()
                bit_acc = (predictions == val_targets).float().mean().item()
                exact_match = (predictions == val_targets).all(dim=1).float().mean().item()
                op_accuracies[OP_NAMES[op]] = (bit_acc, exact_match)

            # Overall
            avg_bit_acc = sum(v[0] for v in op_accuracies.values()) / len(op_accuracies)
            avg_exact = sum(v[1] for v in op_accuracies.values()) / len(op_accuracies)

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / num_batches

        print(f"\r   Epoch {epoch+1}/{phase.epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Bit: {avg_bit_acc*100:.2f}% | "
              f"Exact: {avg_exact*100:.2f}% | "
              f"Time: {epoch_time:.1f}s")

        # Per-op breakdown
        for op_name, (bit_acc, exact) in op_accuracies.items():
            status = "✅" if exact >= 0.99 else "⚠️" if exact >= 0.90 else "❌"
            print(f"      {status} {op_name}: {exact*100:.1f}%")

        # Save if best
        if avg_bit_acc > best_accuracy:
            best_accuracy = avg_bit_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'bit_accuracy': avg_bit_acc,
                'exact_match': avg_exact,
                'phase': phase.name,
                'epoch': epoch,
                'op_accuracies': op_accuracies
            }, save_path)
            print(f"      💾 Saved (best: {avg_bit_acc*100:.2f}%)")

        # Early exit if target reached
        if avg_exact >= phase.target_accuracy:
            print(f"   ✅ Target accuracy reached! Moving to next phase.")
            break

    return model, best_accuracy


def train_curriculum(
    epochs_multiplier: float = 1.0,
    batch_size: int = 16384,
    samples_per_epoch: int = 1_000_000,
    save_path: str = "models/final/fused_alu_curriculum.pt",
    resume_from: str = None
):
    """Run full curriculum training."""

    print("=" * 70)
    print("🧠 FUSED ALU CURRICULUM TRAINING")
    print("=" * 70)

    # Create or load model
    if resume_from and os.path.exists(resume_from):
        print(f"\n📂 Resuming from {resume_from}")
        model = FusedALU().to(device)
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)

        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        best_accuracy = checkpoint.get('bit_accuracy', 0)
        print(f"   Loaded with accuracy: {best_accuracy*100:.2f}%")
    else:
        model = FusedALU().to(device)
        best_accuracy = 0

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model: {total_params:,} parameters")

    # Compile for H200
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("   ✅ torch.compile() enabled")
        except Exception as e:
            print(f"   ⚠️ torch.compile() failed: {e}")

    print(f"\n📚 CURRICULUM PHASES:")
    for i, phase in enumerate(CURRICULUM):
        phase.epochs = int(phase.epochs * epochs_multiplier)
        print(f"   {i+1}. {phase.name}: {[OP_NAMES[j] for j in phase.ops]}")

    # Run curriculum
    for phase in CURRICULUM:
        model, best_accuracy = train_phase(
            model=model,
            phase=phase,
            batch_size=batch_size,
            samples_per_epoch=samples_per_epoch,
            save_path=save_path,
            best_accuracy=best_accuracy
        )

    print("\n" + "=" * 70)
    print("📊 CURRICULUM TRAINING COMPLETE")
    print("=" * 70)
    print(f"   Best bit accuracy: {best_accuracy*100:.2f}%")
    print(f"   Model saved: {save_path}")

    return model


# ============================================================
# MAIN
# ============================================================

import math

def main():
    parser = argparse.ArgumentParser(description="FusedALU Curriculum Training")
    parser.add_argument("--epochs-mult", type=float, default=1.0,
                       help="Multiply phase epochs by this factor")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--output", type=str, default="models/final/fused_alu_curriculum.pt")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")

    args = parser.parse_args()

    train_curriculum(
        epochs_multiplier=args.epochs_mult,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples,
        save_path=args.output,
        resume_from=args.resume
    )


if __name__ == "__main__":
    main()
