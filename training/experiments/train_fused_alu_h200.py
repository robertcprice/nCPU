#!/usr/bin/env python3
"""
🧠 FusedALU H200-Optimized Training
====================================

Optimized for NVIDIA H200 with:
- torch.compile() for Hopper architecture
- bfloat16 mixed precision (2x faster)
- Large batch sizes (utilizing 141GB VRAM)
- Correct model paths for teacher loading

Usage:
    python train_fused_alu_h200.py --epochs 20 --batch-size 32768
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

# ============================================================
# DEVICE SETUP
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 Using CUDA: {torch.cuda.get_device_name()}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # H200 optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Check for Hopper architecture (compute capability 9.0)
    cc = torch.cuda.get_device_capability()
    if cc[0] >= 9:
        print(f"   ✅ Hopper architecture detected (SM {cc[0]}{cc[1]})")
        print(f"   ✅ TF32 enabled for matrix ops")
else:
    device = torch.device("cpu")
    print("💻 Using CPU")


# ============================================================
# FUSED ALU MODEL (inline to avoid import issues)
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

        # Operand projections - project each bit to d_model
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

        # Output projection (no sigmoid - use BCEWithLogitsLoss for autocast safety)
        self.result_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        # Register position indices
        self.register_buffer('pos_indices', torch.arange(bit_width))

    def forward(self, op: torch.Tensor, a_bits: torch.Tensor, b_bits: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            op: Operation codes [batch]
            a_bits: First operand bits [batch, bit_width]
            b_bits: Second operand bits [batch, bit_width]

        Returns:
            Result bits [batch, bit_width]
        """
        batch_size = op.shape[0]

        # Get operation embedding [batch, d_model]
        op_emb = self.op_embed(op)

        # Get position embeddings [bit_width, d_model]
        pos_emb = self.pos_embed(self.pos_indices)

        # Project operand bits [batch, bit_width, d_model]
        a_proj = self.operand_a_proj(a_bits.unsqueeze(-1))
        b_proj = self.operand_b_proj(b_bits.unsqueeze(-1))

        # Combine: operands + position + operation context
        # [batch, bit_width, d_model]
        combined = a_proj + b_proj + pos_emb.unsqueeze(0) + op_emb.unsqueeze(1)

        # Transformer processing
        transformed = self.transformer(combined)

        # Output projection [batch, bit_width]
        result = self.result_proj(transformed).squeeze(-1)

        return result


# ============================================================
# TEACHER MODEL LOADER
# ============================================================

class TeacherEnsemble:
    """Load pre-trained teacher models for knowledge distillation."""

    # Operation name mapping
    OP_NAMES = {
        0: 'ADD', 1: 'SUB', 2: 'AND', 3: 'OR', 4: 'XOR',
        5: 'LSL', 6: 'LSR', 7: 'ROR', 8: 'ASR', 9: 'MUL'
    }

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.teachers: Dict[int, nn.Module] = {}
        self.available_ops: List[int] = []
        self._load_teachers()

    def _load_teachers(self):
        """Load all available teacher models."""
        print("\n📚 Loading teacher models...")

        # Try different filename patterns
        patterns = {
            0: ["ADD_64bit_100pct.pt", "ADD_64bit.pt"],
            1: ["SUB_64bit_100pct.pt", "SUB_64bit.pt"],
            2: ["AND_64bit_100pct.pt", "AND_64bit.pt"],
            3: ["OR_64bit_100pct.pt", "OR_64bit.pt"],
            4: ["XOR_64bit_100pct.pt", "XOR_64bit.pt"],
            5: ["LSL_64bit_exact_100pct.pt", "LSL_64bit_100pct.pt", "LSL_64bit.pt"],
            6: ["LSR_64bit_exact_100pct.pt", "LSR_64bit_100pct.pt", "LSR_64bit.pt"],
            7: ["ROL_64bit_exact_100pct.pt", "ROR_64bit_exact_100pct.pt"],
            8: ["ASR_64bit_exact_100pct.pt", "ASR_64bit.pt"],
            9: ["MUL_64bit_add_hybrid_v3.5_100pct.pt", "ExactMul_64bit_100pct.pt", "MUL_64bit.pt"],
        }

        for op_code, filenames in patterns.items():
            loaded = False
            for filename in filenames:
                path = self.models_dir / filename
                if path.exists():
                    try:
                        checkpoint = torch.load(path, map_location=device, weights_only=False)

                        # Handle different checkpoint formats
                        if isinstance(checkpoint, dict) and 'model' in checkpoint:
                            model = checkpoint['model']
                        elif isinstance(checkpoint, nn.Module):
                            model = checkpoint
                        else:
                            # Try to create from state dict
                            continue

                        model = model.to(device)
                        model.eval()
                        self.teachers[op_code] = model
                        self.available_ops.append(op_code)
                        print(f"   ✅ {self.OP_NAMES[op_code]}: {filename}")
                        loaded = True
                        break
                    except Exception as e:
                        pass

            if not loaded:
                print(f"   ⚠️ {self.OP_NAMES[op_code]}: Not found (will use ground truth)")

        print(f"\n   Loaded {len(self.teachers)}/10 teachers")

    @torch.no_grad()
    def get_teacher_output(self, op_code: int, a_bits: torch.Tensor, b_bits: torch.Tensor) -> Optional[torch.Tensor]:
        """Get teacher's output for an operation."""
        if op_code not in self.teachers:
            return None

        teacher = self.teachers[op_code]
        try:
            return teacher(a_bits, b_bits)
        except:
            return None

    def has_teacher(self, op_code: int) -> bool:
        return op_code in self.teachers


# ============================================================
# DATA GENERATION (Vectorized for H200)
# ============================================================

MASK64 = (1 << 64) - 1

def compute_classical_single(op: int, a: int, b: int) -> int:
    """Classical computation for a single operation. Uses Python int for arbitrary precision."""
    # Ensure unsigned 64-bit values
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
        shift = b & 63
        return (a << shift) & MASK64
    elif op == 6:  # LSR
        shift = b & 63
        return a >> shift
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


def compute_classical_batch(ops: torch.Tensor, a_vals: torch.Tensor, b_vals: torch.Tensor) -> torch.Tensor:
    """Compute ground truth for batch using Python arbitrary precision."""
    results = torch.zeros_like(a_vals)

    # Move to CPU for Python int operations
    ops_cpu = ops.cpu().tolist()
    a_cpu = a_vals.cpu().tolist()
    b_cpu = b_vals.cpu().tolist()

    result_list = []
    for i in range(len(ops_cpu)):
        # Handle negative values (tensor stores as signed)
        a = a_cpu[i] if a_cpu[i] >= 0 else a_cpu[i] + (1 << 64)
        b = b_cpu[i] if b_cpu[i] >= 0 else b_cpu[i] + (1 << 64)
        r = compute_classical_single(ops_cpu[i], a, b)
        # Store back, handling potential sign issues
        if r >= (1 << 63):
            r = r - (1 << 64)  # Convert to signed for tensor storage
        result_list.append(r)

    results = torch.tensor(result_list, dtype=torch.long, device=a_vals.device)
    return results


def int_to_bits(values: torch.Tensor, bits: int = 64) -> torch.Tensor:
    """Convert integers to bit tensors (handles signed tensors correctly)."""
    batch = values.shape[0]
    result = torch.zeros(batch, bits, device=values.device, dtype=torch.float32)

    # Handle signed values by working with the raw bits
    # In Python/PyTorch, negative numbers have the high bit set
    for b in range(bits):
        result[:, b] = ((values >> b) & 1).float()

    return result


def generate_batch_fast(batch_size: int, ops: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate training batch with ground truth (fast version)."""
    # Random operations
    op_codes = torch.tensor([random.choice(ops) for _ in range(batch_size)],
                           dtype=torch.long, device=device)

    # Random operands (generate in two 32-bit parts to avoid overflow)
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

    # Compute ground truth
    results = compute_classical_batch(op_codes, a_vals, b_vals)

    # Convert to bits (int_to_bits already returns float32)
    a_bits = int_to_bits(a_vals)
    b_bits = int_to_bits(b_vals)
    r_bits = int_to_bits(results)

    return op_codes, a_bits, b_bits, r_bits


# ============================================================
# TRAINING
# ============================================================

def train_fused_alu(
    epochs: int = 20,
    batch_size: int = 32768,  # Large for H200!
    learning_rate: float = 3e-4,
    samples_per_epoch: int = 2_000_000,
    save_path: str = "models/final/fused_alu_h200.pt",
    use_compile: bool = True
):
    """Train FusedALU with H200 optimizations."""

    print("=" * 70)
    print("🧠 FUSED ALU H200-OPTIMIZED TRAINING")
    print("=" * 70)

    # Skip teachers - train from ground truth (classical computation)
    # This is actually cleaner and the model will learn the correct behavior
    print("\n📖 Training from ground truth (classical computation)")

    # Operations to train
    ops = list(range(10))  # 0-9

    print(f"\n📊 Configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size:,}")
    print(f"   Samples/epoch: {samples_per_epoch:,}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Device: {device}")
    print(f"   Use torch.compile: {use_compile}")

    # Create model
    student = FusedALU().to(device)
    total_params = sum(p.numel() for p in student.parameters())
    print(f"   Parameters: {total_params:,}")

    # Compile for H200 (Hopper)
    if use_compile and hasattr(torch, 'compile'):
        print("\n🔧 Compiling model for H200...")
        try:
            student = torch.compile(student, mode="reduce-overhead")
            print("   ✅ torch.compile() enabled")
        except Exception as e:
            print(f"   ⚠️ torch.compile() failed: {e}")

    # Optimizer with conservative learning rate (prevents instability)
    optimizer = optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * (samples_per_epoch // batch_size),
        eta_min=learning_rate * 0.01
    )

    # Loss and scaler for mixed precision (BCEWithLogitsLoss is autocast-safe)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    # Training
    best_accuracy = 0

    print("\n" + "=" * 70)
    print("🚀 Starting training...")
    print("=" * 70)

    for epoch in range(epochs):
        epoch_start = time.time()
        student.train()
        total_loss = 0
        num_batches = samples_per_epoch // batch_size

        for batch_idx in range(num_batches):
            # Generate batch
            ops_t, a_bits, b_bits, targets = generate_batch_fast(batch_size, ops)

            # Forward with mixed precision
            with autocast(dtype=torch.bfloat16):
                outputs = student(ops_t, a_bits, b_bits)
                loss = criterion(outputs, targets)

            # Backward with scaling
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 20 == 0:
                print(f"\r   Epoch {epoch+1}/{epochs} | "
                      f"Batch {batch_idx+1}/{num_batches} | "
                      f"Loss: {loss.item():.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}", end="", flush=True)

        avg_loss = total_loss / num_batches

        # Validation
        student.eval()
        with torch.no_grad():
            val_ops, val_a, val_b, val_targets = generate_batch_fast(10000, ops)
            with autocast(dtype=torch.bfloat16):
                val_logits = student(val_ops, val_a, val_b)
                val_outputs = torch.sigmoid(val_logits)  # Apply sigmoid for validation

            predictions = (val_outputs > 0.5).float()
            bit_accuracy = (predictions == val_targets).float().mean().item()
            exact_matches = (predictions == val_targets).all(dim=1).float().mean().item()

        epoch_time = time.time() - epoch_start

        print(f"\r   Epoch {epoch+1}/{epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Bit Acc: {bit_accuracy*100:.2f}% | "
              f"Exact: {exact_matches*100:.2f}% | "
              f"Time: {epoch_time:.1f}s")

        # Save best
        if bit_accuracy > best_accuracy:
            best_accuracy = bit_accuracy
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': student.state_dict(),
                'bit_accuracy': bit_accuracy,
                'exact_match': exact_matches,
                'epoch': epoch,
                'config': {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'lr': learning_rate
                }
            }, save_path)
            print(f"   💾 Saved (best: {bit_accuracy*100:.2f}%)")

    print("\n" + "=" * 70)
    print("📊 TRAINING COMPLETE")
    print("=" * 70)
    print(f"   Best bit accuracy: {best_accuracy*100:.2f}%")
    print(f"   Model saved: {save_path}")

    return student


def main():
    parser = argparse.ArgumentParser(description="Train FusedALU (H200 Optimized)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32768)  # Large for H200!
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--samples", type=int, default=2_000_000)
    parser.add_argument("--output", type=str, default="models/final/fused_alu_h200.pt")
    parser.add_argument("--no-compile", action="store_true")

    args = parser.parse_args()

    train_fused_alu(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        samples_per_epoch=args.samples,
        save_path=args.output,
        use_compile=not args.no_compile
    )


if __name__ == "__main__":
    main()
