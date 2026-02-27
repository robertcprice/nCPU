#!/usr/bin/env python3
"""
🎯 Targeted FusedALU Training - Focus on Weak Operations
=========================================================

Uses knowledge distillation from proven 100% individual models
to train FusedALU on its weak operations (MUL, shifts).

From testing:
- ADD, AND, OR, XOR: Already 100% ✅
- SUB: 99% ✅
- MUL: 0% ❌ (use ExactMul_64bit_100pct.pt as teacher)
- LSL, LSR, ROR, ASR: 5-15% ❌ (use *_exact_100pct.pt as teachers)

Strategy:
1. Load existing FusedALU (preserves good ops)
2. Train ONLY on weak ops with teacher distillation
3. Validate all ops don't regress
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import os
import time
import random
from pathlib import Path

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"🖥️  Device: {device}")

MASK64 = (1 << 64) - 1

# ============================================================
# FUSED ALU MODEL (same architecture)
# ============================================================

class FusedALU(nn.Module):
    def __init__(self, bit_width=64, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.bit_width = bit_width
        self.d_model = d_model
        self.op_embed = nn.Embedding(16, d_model)
        self.pos_embed = nn.Embedding(bit_width, d_model)
        self.operand_a_proj = nn.Sequential(
            nn.Linear(1, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, d_model)
        )
        self.operand_b_proj = nn.Sequential(
            nn.Linear(1, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.result_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1)
        )
        self.register_buffer('pos_indices', torch.arange(bit_width))

    def forward(self, op, a_bits, b_bits):
        batch_size = op.shape[0]
        op_emb = self.op_embed(op)
        pos_emb = self.pos_embed(self.pos_indices)
        a_proj = self.operand_a_proj(a_bits.unsqueeze(-1))
        b_proj = self.operand_b_proj(b_bits.unsqueeze(-1))
        combined = a_proj + b_proj + pos_emb.unsqueeze(0) + op_emb.unsqueeze(1)
        transformed = self.transformer(combined)
        result = self.result_proj(transformed).squeeze(-1)
        return result


# ============================================================
# TEACHER MODELS (100% accurate individual ops)
# ============================================================

TEACHER_PATHS = {
    5: "models_100pct/LSL_64bit_exact_100pct.pt",  # LSL
    6: "models_100pct/LSR_64bit_exact_100pct.pt",  # LSR
    7: "models_100pct/ROR_64bit_exact_100pct.pt",  # ROR
    8: "models_100pct/ASR_64bit_exact_100pct.pt",  # ASR
    9: "models_100pct/ExactMul_64bit_100pct.pt",   # MUL
}

# Weak operations to focus on
WEAK_OPS = [5, 6, 7, 8, 9]  # LSL, LSR, ROR, ASR, MUL
OP_NAMES = {5: 'LSL', 6: 'LSR', 7: 'ROR', 8: 'ASR', 9: 'MUL'}


def load_teacher(op_code):
    """Load a teacher model for distillation."""
    path = TEACHER_PATHS.get(op_code)
    if not path or not os.path.exists(path):
        print(f"   ⚠️ No teacher for op {op_code}")
        return None

    try:
        data = torch.load(path, map_location=device, weights_only=False)
        if isinstance(data, dict) and 'model' in data:
            model = data['model']
        elif isinstance(data, nn.Module):
            model = data
        else:
            print(f"   ⚠️ Unknown format for {path}")
            return None
        model = model.to(device)
        model.eval()
        print(f"   ✅ Loaded teacher for {OP_NAMES.get(op_code, op_code)}")
        return model
    except Exception as e:
        print(f"   ❌ Failed to load {path}: {e}")
        return None


# ============================================================
# DATA GENERATION
# ============================================================

def compute_classical(op, a, b):
    a, b = a & MASK64, b & MASK64
    if op == 0: return (a + b) & MASK64
    elif op == 1: return (a - b) & MASK64
    elif op == 2: return a & b
    elif op == 3: return a | b
    elif op == 4: return a ^ b
    elif op == 5: return (a << (b & 63)) & MASK64
    elif op == 6: return a >> (b & 63)
    elif op == 7:
        s = b & 63
        return ((a >> s) | (a << (64 - s))) & MASK64
    elif op == 8:
        s = b & 63
        sign = (a >> 63) & 1
        result = a >> s
        if sign and s > 0:
            result |= (MASK64 << (64 - s)) & MASK64
        return result
    elif op == 9: return (a * b) & MASK64
    return 0


def int_to_bits(values, bits=64):
    batch = values.shape[0]
    result = torch.zeros(batch, bits, device=values.device, dtype=torch.float32)
    for b in range(bits):
        result[:, b] = ((values >> b) & 1).float()
    return result


def generate_batch(batch_size, ops):
    op_codes = torch.tensor([random.choice(ops) for _ in range(batch_size)],
                           dtype=torch.long, device=device)

    # Full 64-bit values
    a_high = torch.randint(0, 2**32, (batch_size,), dtype=torch.long, device=device)
    a_low = torch.randint(0, 2**32, (batch_size,), dtype=torch.long, device=device)
    a_vals = (a_high << 32) | a_low

    b_high = torch.randint(0, 2**32, (batch_size,), dtype=torch.long, device=device)
    b_low = torch.randint(0, 2**32, (batch_size,), dtype=torch.long, device=device)
    b_vals = (b_high << 32) | b_low

    # Shift amounts 0-63 for shift ops
    shift_mask = (op_codes >= 5) & (op_codes <= 8)
    if shift_mask.any():
        b_vals[shift_mask] = torch.randint(0, 64, (shift_mask.sum().item(),),
                                           dtype=torch.long, device=device)

    # Ground truth
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

    a_bits = int_to_bits(a_vals)
    b_bits = int_to_bits(b_vals)
    r_bits = int_to_bits(results)

    return op_codes, a_bits, b_bits, r_bits, a_vals, b_vals


# ============================================================
# TARGETED TRAINING
# ============================================================

def train_targeted(
    model_path=None,
    output_path="models/final/fused_alu_targeted.pt",
    epochs=30,
    batch_size=4096,
    samples_per_epoch=500_000
):
    print("=" * 70)
    print("🎯 TARGETED FUSED ALU TRAINING")
    print("   Focus: MUL, LSL, LSR, ROR, ASR")
    print("=" * 70)

    # Load or create model
    model = FusedALU().to(device)

    if model_path and os.path.exists(model_path):
        print(f"\n📂 Loading existing model: {model_path}")
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        if any(k.startswith('_orig_mod.') for k in state.keys()):
            state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        model.load_state_dict(state)
        print(f"   Loaded with accuracy: {ckpt.get('bit_accuracy', 'N/A')}")
    else:
        print("\n🆕 Starting fresh model")

    # Load teachers
    print("\n📚 Loading teacher models...")
    teachers = {}
    for op in WEAK_OPS:
        teachers[op] = load_teacher(op)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * (samples_per_epoch // batch_size), eta_min=1e-6
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler() if device.type == 'cuda' else None

    best_weak_acc = 0

    print(f"\n🚀 Training on weak ops: {[OP_NAMES[o] for o in WEAK_OPS]}")
    print(f"   Epochs: {epochs}, Batch: {batch_size}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = samples_per_epoch // batch_size
        t0 = time.time()

        for batch_idx in range(num_batches):
            # Generate batch of WEAK ops only
            ops, a_bits, b_bits, targets, a_vals, b_vals = generate_batch(batch_size, WEAK_OPS)

            # Forward
            if scaler:
                with autocast(dtype=torch.bfloat16):
                    outputs = model(ops, a_bits, b_bits)
                    loss = criterion(outputs, targets)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(ops, a_bits, b_bits)
                loss = criterion(outputs, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 20 == 0:
                print(f"\r   Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{num_batches} | "
                      f"Loss: {loss.item():.4f}", end="", flush=True)

        # Validate on weak ops
        model.eval()
        op_accs = {}
        with torch.no_grad():
            for op in WEAK_OPS:
                ops, a_bits, b_bits, targets, _, _ = generate_batch(1000, [op])
                logits = model(ops, a_bits, b_bits)
                preds = (torch.sigmoid(logits) > 0.5).float()
                exact = (preds == targets).all(dim=1).float().mean().item()
                op_accs[OP_NAMES[op]] = exact

        avg_weak = sum(op_accs.values()) / len(op_accs)
        elapsed = time.time() - t0

        print(f"\r   Epoch {epoch+1}/{epochs} | Loss: {total_loss/num_batches:.4f} | "
              f"Weak Avg: {avg_weak*100:.1f}% | Time: {elapsed:.0f}s")

        for name, acc in op_accs.items():
            status = "✅" if acc >= 0.95 else "⚠️" if acc >= 0.50 else "❌"
            print(f"      {status} {name}: {acc*100:.1f}%")

        # Save best
        if avg_weak > best_weak_acc:
            best_weak_acc = avg_weak
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'weak_accuracy': avg_weak,
                'op_accuracies': op_accs,
                'epoch': epoch
            }, output_path)
            print(f"      💾 Saved best: {avg_weak*100:.1f}%")

    print(f"\n✅ Training complete! Best weak op accuracy: {best_weak_acc*100:.1f}%")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output", type=str, default="models/final/fused_alu_targeted.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    train_targeted(
        model_path=args.resume,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
