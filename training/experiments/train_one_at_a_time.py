#!/usr/bin/env python3
"""
🎯 One-at-a-Time Curriculum Training
=====================================

Train ONE operation at a time to 100%, then add the next.
This ensures each operation is fully learned before moving on.

Order:
1. AND → 100%
2. OR  → 100% (while maintaining AND)
3. XOR → 100% (while maintaining AND, OR)
4. ADD → 100% (while maintaining above)
5. SUB → 100%
6. LSL → 100%
7. LSR → 100%
8. ROR → 100%
9. ASR → 100%
10. MUL → 100%
"""

import torch
import torch.nn as nn
import torch.optim as optim
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

# Operation order (easiest to hardest)
OP_ORDER = [
    (2, "AND"),
    (3, "OR"),
    (4, "XOR"),
    (0, "ADD"),
    (1, "SUB"),
    (5, "LSL"),
    (6, "LSR"),
    (7, "ROR"),
    (8, "ASR"),
    (9, "MUL"),
]

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
        return ((a >> s) | (a << (64 - s))) & MASK64 if s else a
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
    """Generate batch for specific operations."""
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

    return op_codes, a_bits, b_bits, r_bits


def evaluate_ops(model, ops, samples=2000):
    """Evaluate exact accuracy for each op."""
    model.eval()
    results = {}

    with torch.no_grad():
        for op_code, op_name in OP_ORDER:
            if op_code not in ops:
                continue
            op_codes, a_bits, b_bits, targets = generate_batch(samples, [op_code])
            logits = model(op_codes, a_bits, b_bits)
            preds = (torch.sigmoid(logits) > 0.5).float()
            exact = (preds == targets).all(dim=1).float().mean().item()
            results[op_name] = exact

    return results


def train_one_at_a_time(
    resume_path=None,
    output_path="models/final/fused_alu_sequential.pt",
    batch_size=32768,
    target_accuracy=0.99,
    max_epochs_per_op=100,
    patience=15
):
    print("=" * 70)
    print("🎯 ONE-AT-A-TIME CURRICULUM TRAINING")
    print("   Train each op to 100% before adding next")
    print("=" * 70)

    # Load or create model
    model = FusedALU().to(device)
    start_op_idx = 0

    if resume_path and os.path.exists(resume_path):
        print(f"\n📂 Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        if any(k.startswith('_orig_mod.') for k in state.keys()):
            state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        model.load_state_dict(state)
        start_op_idx = ckpt.get('completed_ops', 0)
        print(f"   Resuming from op index: {start_op_idx}")
    else:
        print("\n🆕 Starting fresh model")

    # Skip compile - causes OOM with CUDA graphs
    # if hasattr(torch, 'compile') and device.type == 'cuda':
    #     print("⚡ Compiling model...")
    #     model = torch.compile(model, mode='reduce-overhead')

    criterion = nn.BCEWithLogitsLoss()

    # Train each operation one at a time
    active_ops = []

    for op_idx, (op_code, op_name) in enumerate(OP_ORDER):
        if op_idx < start_op_idx:
            active_ops.append(op_code)
            print(f"\n⏭️  Skipping {op_name} (already trained)")
            continue

        active_ops.append(op_code)

        print(f"\n{'='*60}")
        print(f"📚 Training Op {op_idx+1}/{len(OP_ORDER)}: {op_name}")
        print(f"   Active ops: {[OP_ORDER[i][1] for i, (c, _) in enumerate(OP_ORDER) if c in active_ops]}")
        print(f"   Target: {target_accuracy*100:.0f}% exact accuracy")
        print(f"={'='*60}")

        # Fresh optimizer for each op (helps with learning rate)
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-5
        )

        best_new_op_acc = 0
        epochs_without_improvement = 0

        for epoch in range(max_epochs_per_op):
            model.train()
            total_loss = 0
            num_batches = 30  # ~1M samples per epoch with batch_size=32768
            t0 = time.time()

            for batch_idx in range(num_batches):
                # Sample from active ops, with 50% weight on new op
                if random.random() < 0.5 and len(active_ops) > 1:
                    # Train on new op specifically
                    ops, a_bits, b_bits, targets = generate_batch(batch_size, [op_code])
                else:
                    # Train on all active ops
                    ops, a_bits, b_bits, targets = generate_batch(batch_size, active_ops)

                outputs = model(ops, a_bits, b_bits)
                loss = criterion(outputs, targets)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step(epoch + batch_idx / num_batches)

                total_loss += loss.item()

            # Evaluate all active ops
            accs = evaluate_ops(model, active_ops)
            elapsed = time.time() - t0

            new_op_acc = accs.get(op_name, 0)
            all_good = all(acc >= target_accuracy for acc in accs.values())

            # Status line
            status = "✅" if new_op_acc >= target_accuracy else "🔄"
            print(f"   Epoch {epoch+1:3d} | Loss: {total_loss/num_batches:.4f} | "
                  f"{op_name}: {new_op_acc*100:5.1f}% {status} | Time: {elapsed:.1f}s")

            # Show all active ops
            for name, acc in accs.items():
                marker = "✅" if acc >= target_accuracy else "❌"
                if name != op_name:
                    print(f"           {marker} {name}: {acc*100:.1f}%")

            # Check if new op reached target and all previous ops maintained
            if new_op_acc >= target_accuracy and all_good:
                print(f"\n   ✅ {op_name} reached {new_op_acc*100:.1f}%! All ops at target.")

                # Save checkpoint
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'completed_ops': op_idx + 1,
                    'accuracies': accs,
                    'op_name': op_name
                }, output_path)
                print(f"   💾 Saved checkpoint: {op_idx + 1} ops complete")
                break

            # Track improvement
            if new_op_acc > best_new_op_acc:
                best_new_op_acc = new_op_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stop if stuck
            if epochs_without_improvement >= patience:
                print(f"\n   ⚠️ {op_name} stuck at {best_new_op_acc*100:.1f}% for {patience} epochs")
                print(f"   Saving best and continuing...")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'completed_ops': op_idx,
                    'accuracies': accs,
                    'stuck_on': op_name,
                    'best_acc': best_new_op_acc
                }, output_path)
                break

        else:
            # Hit max epochs
            print(f"\n   ⏰ Max epochs reached. Best {op_name}: {best_new_op_acc*100:.1f}%")

    # Final evaluation
    print("\n" + "=" * 70)
    print("📊 FINAL EVALUATION")
    print("=" * 70)

    final_accs = evaluate_ops(model, [c for c, _ in OP_ORDER])
    all_100 = True
    for op_code, op_name in OP_ORDER:
        acc = final_accs.get(op_name, 0)
        status = "✅" if acc >= target_accuracy else "❌"
        print(f"   {status} {op_name}: {acc*100:.1f}%")
        if acc < target_accuracy:
            all_100 = False

    if all_100:
        print("\n🎉 ALL OPERATIONS AT TARGET ACCURACY!")
    else:
        print(f"\n⚠️ Some operations below target. Checkpoint saved.")

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output", type=str, default="models/final/fused_alu_sequential.pt")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--target", type=float, default=0.99, help="Target accuracy (0.99 = 99%)")
    parser.add_argument("--max-epochs", type=int, default=100, help="Max epochs per operation")
    parser.add_argument("--patience", type=int, default=15, help="Epochs without improvement before moving on")
    args = parser.parse_args()

    train_one_at_a_time(
        resume_path=args.resume,
        output_path=args.output,
        batch_size=args.batch_size,
        target_accuracy=args.target,
        max_epochs_per_op=args.max_epochs,
        patience=args.patience
    )
