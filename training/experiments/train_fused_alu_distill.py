#!/usr/bin/env python3
"""
🧠 FusedALU Knowledge Distillation Training
=============================================

Uses pre-trained individual ALU models as TEACHERS to train the
unified FusedALU STUDENT model.

This is much faster and more accurate than training from scratch because:
1. Teachers already learned the operations at 100% accuracy
2. Student learns from soft targets (probability distributions)
3. Knowledge is "distilled" into a single efficient model

Usage:
    # On H200 GPU via SSH:
    ssh -p 18277 root@ssh2.vast.ai -L 8080:localhost:8080

    # Run training:
    python train_fused_alu_distill.py --epochs 20 --batch-size 4096
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import time
import argparse
import random
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 Using CUDA: {torch.cuda.get_device_name()}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🍎 Using Apple MPS")
else:
    device = torch.device("cpu")
    print("💻 Using CPU")

# Import FusedALU
from neural_cpu_fused import FusedALU


# ============================================================
# TEACHER MODEL LOADER
# ============================================================

class TeacherEnsemble:
    """
    Loads and manages pre-trained teacher models.

    Maps operation codes to their corresponding teacher models.
    """

    # Operation code mapping
    OP_MAP = {
        0: 'ADD',   # ADD_64bit_100pct.pt
        1: 'SUB',   # Will use ADD with negation trick
        2: 'AND',   # AND_64bit_100pct.pt
        3: 'OR',    # OR_64bit_100pct.pt
        4: 'XOR',   # Will derive from AND/OR
        5: 'LSL',   # LSL_64bit_exact_100pct.pt
        6: 'LSR',   # LSR_64bit_exact_100pct.pt
        7: 'ROR',   # ROL_64bit_exact_100pct.pt (reversed)
        8: 'ASR',   # ASR_64bit_exact_100pct.pt
        9: 'MUL',   # MUL_64bit_100pct.pt
    }

    def __init__(self, models_dir: str = "models/final"):
        self.models_dir = Path(models_dir)
        self.teachers: Dict[int, nn.Module] = {}
        self.available_ops: List[int] = []

        self._load_teachers()

    def _load_teachers(self):
        """Load all available teacher models."""
        print("\n📚 Loading teacher models...")

        # Model file mapping
        model_files = {
            0: "ADD_64bit_100pct.pt",
            2: "AND_64bit_100pct.pt",
            3: "OR_64bit_100pct.pt",
            5: "LSL_64bit_exact_100pct.pt",
            6: "LSR_64bit_exact_100pct.pt",
            7: "ROL_64bit_exact_100pct.pt",
            8: "ASR_64bit_exact_100pct.pt",
            9: "MUL_64bit_100pct.pt",
        }

        for op_code, filename in model_files.items():
            path = self.models_dir / filename
            if path.exists():
                try:
                    checkpoint = torch.load(path, map_location=device, weights_only=False)

                    # Extract model - handle different checkpoint formats
                    if isinstance(checkpoint, dict):
                        if 'model_state_dict' in checkpoint:
                            # Need to instantiate model first
                            model = self._create_model_for_op(op_code)
                            if model:
                                model.load_state_dict(checkpoint['model_state_dict'])
                                model = model.to(device)
                                model.eval()
                                self.teachers[op_code] = model
                                self.available_ops.append(op_code)
                                print(f"   ✅ {self.OP_MAP.get(op_code, op_code)}: {filename}")
                        elif 'model' in checkpoint:
                            model = checkpoint['model'].to(device)
                            model.eval()
                            self.teachers[op_code] = model
                            self.available_ops.append(op_code)
                            print(f"   ✅ {self.OP_MAP.get(op_code, op_code)}: {filename}")
                    elif isinstance(checkpoint, nn.Module):
                        model = checkpoint.to(device)
                        model.eval()
                        self.teachers[op_code] = model
                        self.available_ops.append(op_code)
                        print(f"   ✅ {self.OP_MAP.get(op_code, op_code)}: {filename}")
                except Exception as e:
                    print(f"   ⚠️ {self.OP_MAP.get(op_code, op_code)}: Failed ({e})")
            else:
                print(f"   ❌ {self.OP_MAP.get(op_code, op_code)}: Not found")

        print(f"\n   Loaded {len(self.teachers)}/{len(model_files)} teachers")

    def _create_model_for_op(self, op_code: int) -> Optional[nn.Module]:
        """Create an empty model instance for loading weights."""
        # Try to import the model class from neural_cpu
        try:
            from neural_cpu import (
                NeuralBinaryOp,
                NeuralShift,
                NeuralMultiplier
            )

            if op_code in [0, 2, 3, 4]:  # ADD, AND, OR, XOR
                return NeuralBinaryOp(bit_width=64)
            elif op_code in [5, 6, 7, 8]:  # Shifts
                return NeuralShift(bit_width=64)
            elif op_code == 9:  # MUL
                return NeuralMultiplier(bit_width=64)
        except ImportError:
            pass

        return None

    def get_teacher_output(
        self,
        op_code: int,
        a_bits: torch.Tensor,
        b_bits: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Get teacher's output for an operation."""
        if op_code not in self.teachers:
            return None

        teacher = self.teachers[op_code]
        with torch.no_grad():
            return teacher(a_bits, b_bits)

    def has_teacher(self, op_code: int) -> bool:
        """Check if a teacher exists for the operation."""
        return op_code in self.teachers


# ============================================================
# DATA GENERATION
# ============================================================

def generate_training_batch(
    batch_size: int,
    ops: List[int],
    use_edge_cases: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a batch of training data with ground truth.

    Returns: (op_codes, a_bits, b_bits, result_bits)
    """
    MASK64 = (1 << 64) - 1

    op_codes = []
    a_values = []
    b_values = []
    results = []

    def compute_classical(op, a, b):
        """Classical computation (ground truth)."""
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
            shift = b & 63
            return ((a >> shift) | (a << (64 - shift))) & MASK64
        elif op == 8:  # ASR
            shift = b & 63
            sign = (a >> 63) & 1
            result = a >> shift
            if sign:
                result |= (MASK64 << (64 - shift)) & MASK64
            return result
        elif op == 9:  # MUL
            return (a * b) & MASK64
        elif op == 12:  # SDIV
            if b == 0:
                return 0
            a_s = a if a < (1 << 63) else a - (1 << 64)
            b_s = b if b < (1 << 63) else b - (1 << 64)
            return int(a_s / b_s) & MASK64
        elif op == 13:  # UDIV
            return a // b if b != 0 else 0
        return 0

    # Edge cases to include
    edge_values = [0, 1, 2, MASK64, MASK64 - 1, 1 << 63, (1 << 63) - 1, 0xFF, 0xFFFF]

    for _ in range(batch_size):
        op = random.choice(ops)

        # Generate operands
        if use_edge_cases and random.random() < 0.1:
            a = random.choice(edge_values)
            b = random.choice(edge_values)
        elif op in [5, 6, 7, 8]:  # Shift operations
            a = random.randint(0, MASK64)
            b = random.randint(0, 63)
        elif op in [12, 13]:  # Division
            a = random.randint(0, MASK64)
            b = random.randint(1, MASK64)
        else:
            a = random.randint(0, MASK64)
            b = random.randint(0, MASK64)

        result = compute_classical(op, a, b)

        op_codes.append(op)
        a_values.append(a)
        b_values.append(b)
        results.append(result)

    # Convert to bit tensors
    def to_bits(values, bits=64):
        tensor = torch.zeros(len(values), bits, device=device)
        for i, v in enumerate(values):
            for j in range(bits):
                tensor[i, j] = (v >> j) & 1
        return tensor

    return (
        torch.tensor(op_codes, dtype=torch.long, device=device),
        to_bits(a_values),
        to_bits(b_values),
        to_bits(results)
    )


# ============================================================
# KNOWLEDGE DISTILLATION LOSS
# ============================================================

class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.

    L = α * L_hard + (1 - α) * L_soft

    Where:
    - L_hard: BCE loss with ground truth
    - L_soft: KL divergence with teacher outputs (soft targets)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 2.0
    ):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.bce = nn.BCELoss()

    def forward(
        self,
        student_output: torch.Tensor,
        ground_truth: torch.Tensor,
        teacher_output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute distillation loss.

        Args:
            student_output: Student model output [batch, 64]
            ground_truth: Ground truth bits [batch, 64]
            teacher_output: Teacher model output [batch, 64] (optional)
        """
        # Hard loss (ground truth)
        hard_loss = self.bce(student_output, ground_truth)

        if teacher_output is None:
            return hard_loss

        # Soft loss (teacher knowledge)
        # Apply temperature softening
        student_soft = torch.sigmoid(
            torch.logit(student_output.clamp(1e-7, 1 - 1e-7)) / self.temperature
        )
        teacher_soft = torch.sigmoid(
            torch.logit(teacher_output.clamp(1e-7, 1 - 1e-7)) / self.temperature
        )

        soft_loss = F.mse_loss(student_soft, teacher_soft)

        # Combined loss
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss


# ============================================================
# TRAINING LOOP
# ============================================================

def train_with_distillation(
    epochs: int = 20,
    batch_size: int = 4096,
    learning_rate: float = 1e-4,
    samples_per_epoch: int = 1_000_000,
    save_path: str = "models/final/fused_alu_distilled.pt",
    use_teachers: bool = True
):
    """
    Train FusedALU with knowledge distillation.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        samples_per_epoch: Training samples per epoch
        save_path: Path to save model
        use_teachers: Whether to use teacher models for distillation
    """
    print("=" * 70)
    print("🧠 FUSED ALU KNOWLEDGE DISTILLATION TRAINING")
    print("=" * 70)

    # Load teacher models
    teachers = TeacherEnsemble() if use_teachers else None

    # Determine which operations to train
    ops_to_train = list(range(10))  # 0-9: ADD, SUB, AND, OR, XOR, LSL, LSR, ROR, ASR, MUL

    print(f"\n📊 Training Configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Samples/epoch: {samples_per_epoch:,}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Operations: {ops_to_train}")
    print(f"   Device: {device}")
    if teachers:
        print(f"   Teachers available: {teachers.available_ops}")

    # Create student model
    student = FusedALU().to(device)
    total_params = sum(p.numel() for p in student.parameters())
    print(f"   Student parameters: {total_params:,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2
    )

    # Loss function
    criterion = DistillationLoss(alpha=0.7, temperature=2.0)

    # Training tracking
    best_accuracy = 0
    history = []

    print("\n" + "=" * 70)
    print("🚀 Starting training...")
    print("=" * 70)

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training phase
        student.train()
        total_loss = 0
        num_batches = samples_per_epoch // batch_size

        for batch_idx in range(num_batches):
            # Generate batch
            ops, a_bits, b_bits, targets = generate_training_batch(
                batch_size, ops_to_train
            )

            # Forward pass
            outputs = student(ops, a_bits, b_bits)

            # Get teacher outputs for distillation
            teacher_outputs = None
            if teachers:
                teacher_outputs = torch.zeros_like(outputs)
                for i, op in enumerate(ops.tolist()):
                    if teachers.has_teacher(op):
                        t_out = teachers.get_teacher_output(
                            op,
                            a_bits[i:i+1],
                            b_bits[i:i+1]
                        )
                        if t_out is not None:
                            teacher_outputs[i] = t_out.squeeze(0)
                    else:
                        teacher_outputs[i] = targets[i]  # Use ground truth

            # Compute loss
            loss = criterion(outputs, targets, teacher_outputs)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            # Progress indicator
            if (batch_idx + 1) % 50 == 0:
                print(f"\r   Epoch {epoch+1}/{epochs} | "
                      f"Batch {batch_idx+1}/{num_batches} | "
                      f"Loss: {loss.item():.4f}", end="")

        avg_loss = total_loss / num_batches
        scheduler.step()

        # Validation phase
        student.eval()
        with torch.no_grad():
            val_ops, val_a, val_b, val_targets = generate_training_batch(
                10000, ops_to_train
            )

            val_outputs = student(val_ops, val_a, val_b)

            # Bit-wise accuracy
            predictions = (val_outputs > 0.5).float()
            accuracy = (predictions == val_targets).float().mean().item()

            # Per-bit accuracy (should be very high)
            per_bit_acc = (predictions == val_targets).float().mean(dim=0)

            # Count exact matches
            exact_matches = (predictions == val_targets).all(dim=1).float().mean().item()

        epoch_time = time.time() - epoch_start

        print(f"\r   Epoch {epoch+1}/{epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Bit Acc: {accuracy*100:.2f}% | "
              f"Exact: {exact_matches*100:.2f}% | "
              f"Time: {epoch_time:.1f}s")

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': student.state_dict(),
                'accuracy': accuracy,
                'exact_match': exact_matches,
                'epoch': epoch,
                'config': {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate
                }
            }, save_path)
            print(f"   💾 Saved (best accuracy: {accuracy*100:.2f}%)")

        history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'accuracy': accuracy,
            'exact_match': exact_matches
        })

    # Final summary
    print("\n" + "=" * 70)
    print("📊 TRAINING SUMMARY")
    print("=" * 70)
    print(f"   Best bit accuracy: {best_accuracy*100:.2f}%")
    print(f"   Model saved: {save_path}")
    print(f"   Total epochs: {epochs}")

    return student, history


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train FusedALU with Knowledge Distillation")

    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4096,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--samples", type=int, default=1_000_000,
                       help="Samples per epoch")
    parser.add_argument("--output", type=str, default="models/final/fused_alu_distilled.pt",
                       help="Output model path")
    parser.add_argument("--no-teachers", action="store_true",
                       help="Train without teacher models")

    args = parser.parse_args()

    train_with_distillation(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        samples_per_epoch=args.samples,
        save_path=args.output,
        use_teachers=not args.no_teachers
    )


if __name__ == "__main__":
    main()
