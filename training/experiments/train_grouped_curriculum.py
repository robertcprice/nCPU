#!/usr/bin/env python3
"""
🎯 Grouped Curriculum Training
==============================

Fix for catastrophic forgetting by training operations in SEMANTIC GROUPS.

Groups (train all ops in a group together):
1. Bitwise: AND, OR, XOR
2. Arithmetic: ADD, SUB
3. Left Shifts: LSL, LSR
4. Right Shifts: ROR, ASR
5. Multiplication: MUL

Within each group, ALL ops train in parallel (no forgetting).
Between groups, we use REHEARSAL (include samples from previous groups).
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

# Operation groups (semantic clustering)
OP_GROUPS = [
    {
        "name": "Bitwise",
        "ops": [(2, "AND"), (3, "OR"), (4, "XOR")],
        "description": "Simple per-bit operations"
    },
    {
        "name": "Arithmetic",
        "ops": [(0, "ADD"), (1, "SUB")],
        "description": "Addition and subtraction with carry"
    },
    {
        "name": "Left Shifts",
        "ops": [(5, "LSL"), (6, "LSR")],
        "description": "Left shift and logical right shift"
    },
    {
        "name": "Right Shifts",
        "ops": [(7, "ROR"), (8, "ASR")],
        "description": "Rotate and arithmetic right shift"
    },
    {
        "name": "Multiplication",
        "ops": [(9, "MUL")],
        "description": "Multiplication (hardest operation)"
    }
]

ALL_OPS = [(code, name) for group in OP_GROUPS for code, name in group["ops"]]


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
        for op_code, op_name in ALL_OPS:
            if op_code not in ops:
                results[op_name] = None
                continue
            op_codes, a_bits, b_bits, targets = generate_batch(samples, [op_code])
            logits = model(op_codes, a_bits, b_bits)
            preds = (torch.sigmoid(logits) > 0.5).float()
            exact = (preds == targets).all(dim=1).float().mean().item()
            results[op_name] = exact

    return results


def train_grouped_curriculum(
    resume_path=None,
    output_path="models/final/fused_alu_grouped.pt",
    batch_size=32768,
    target_accuracy=0.99,
    max_epochs_per_group=80,
    patience=20,
    rehearsal_ratio=0.3  # 30% of samples from previous groups
):
    print("=" * 70)
    print("🎯 GROUPED CURRICULUM TRAINING")
    print("   Train operations in semantic groups to prevent forgetting")
    print("=" * 70)

    # Load or create model
    model = FusedALU().to(device)
    start_group_idx = 0
    learned_ops = set()

    if resume_path and os.path.exists(resume_path):
        print(f"\n📂 Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        if any(k.startswith('_orig_mod.') for k in state.keys()):
            state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        model.load_state_dict(state)
        start_group_idx = ckpt.get('completed_groups', 0)
        learned_ops = set(ckpt.get('learned_ops', []))
        print(f"   Resuming from group {start_group_idx}")
        print(f"   Already learned: {[OP_GROUPS[i]['name'] for i in range(start_group_idx)]}")
    else:
        print("\n🆕 Starting fresh model")

    criterion = nn.BCEWithLogitsLoss()

    # Train each group
    for group_idx, group in enumerate(OP_GROUPS):
        if group_idx < start_group_idx:
            for code, name in group["ops"]:
                learned_ops.add(code)
            continue

        group_ops = [code for code, _ in group["ops"]]
        group_op_names = [name for _, name in group["ops"]]

        print(f"\n{'='*70}")
        print(f"📚 Group {group_idx+1}/{len(OP_GROUPS)}: {group['name']}")
        print(f"   Operations: {', '.join(group_op_names)}")
        print(f"   Description: {group['description']}")
        print(f"   Target: {target_accuracy*100:.0f}% exact accuracy")
        print(f"{'='*70}")

        # Fresh optimizer for each group
        optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=15, T_mult=2, eta_min=5e-6
        )

        best_acc = 0
        epochs_without_improvement = 0

        for epoch in range(max_epochs_per_group):
            model.train()
            total_loss = 0
            num_batches = 30
            t0 = time.time()

            for batch_idx in range(num_batches):
                # REHEARSAL: Mix current group with previous groups
                if learned_ops and random.random() < rehearsal_ratio:
                    # Sample from learned ops (rehearsal)
                    ops, a_bits, b_bits, targets = generate_batch(
                        batch_size // 2, list(learned_ops)
                    )
                    # Sample from current group
                    ops2, a_bits2, b_bits2, targets2 = generate_batch(
                        batch_size - batch_size // 2, group_ops
                    )
                    # Concatenate
                    ops = torch.cat([ops, ops2])
                    a_bits = torch.cat([a_bits, a_bits2])
                    b_bits = torch.cat([b_bits, b_bits2])
                    targets = torch.cat([targets, targets2])
                else:
                    # Train on current group
                    ops, a_bits, b_bits, targets = generate_batch(batch_size, group_ops)

                outputs = model(ops, a_bits, b_bits)
                loss = criterion(outputs, targets)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step(epoch + batch_idx / num_batches)

                total_loss += loss.item()

            # Evaluate current group + learned ops
            all_active_ops = list(learned_ops) + group_ops
            accs = evaluate_ops(model, all_active_ops)
            elapsed = time.time() - t0

            # Check current group ops
            group_accs = [accs[name] for name in group_op_names if accs[name] is not None]
            avg_group_acc = sum(group_accs) / len(group_accs) if group_accs else 0

            # Check if all current group ops at target
            current_group_good = all(accs.get(name, 0) >= target_accuracy for name in group_op_names)

            # Check if learned ops maintained
            learned_ops_good = True
            for code in learned_ops:
                name = next(n for c, n in ALL_OPS if c == code)
                if accs.get(name, 0) < target_accuracy * 0.95:  # Allow 5% degradation
                    learned_ops_good = False
                    break

            # Status line
            status = "✅" if avg_group_acc >= target_accuracy else "🔄"
            print(f"   Epoch {epoch+1:3d} | Loss: {total_loss/num_batches:.4f} | "
                  f"{group['name']}: {avg_group_acc*100:5.1f}% {status} | Time: {elapsed:.1f}s")

            # Show all ops in group
            for name in group_op_names:
                acc = accs.get(name, 0)
                marker = "✅" if acc >= target_accuracy else "❌"
                print(f"           {marker} {name}: {acc*100:.1f}%")

            # Show learned ops status
            if learned_ops:
                print(f"           📚 Previously learned:")
                for code in learned_ops:
                    name = next(n for c, n in ALL_OPS if c == code)
                    acc = accs.get(name, 0)
                    marker = "✅" if acc >= target_accuracy * 0.95 else "⚠️"
                    print(f"              {marker} {name}: {acc*100:.1f}%")

            # Check completion
            if current_group_good and learned_ops_good:
                print(f"\n   ✅ {group['name']} complete! All ops at target.")

                # Add to learned ops
                for code, name in group["ops"]:
                    learned_ops.add(code)

                # Save checkpoint
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'completed_groups': group_idx + 1,
                    'learned_ops': list(learned_ops),
                    'accuracies': accs,
                    'group_name': group['name']
                }, output_path)
                print(f"   💾 Saved checkpoint: {group_idx + 1} groups complete")
                break

            # Track improvement
            if avg_group_acc > best_acc:
                best_acc = avg_group_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stop if stuck
            if epochs_without_improvement >= patience:
                print(f"\n   ⚠️ {group['name']} stuck at {best_acc*100:.1f}% for {patience} epochs")
                print(f"   Saving best and continuing...")

                # Still add to learned ops
                for code, name in group["ops"]:
                    learned_ops.add(code)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'completed_groups': group_idx + 1,
                    'learned_ops': list(learned_ops),
                    'accuracies': accs,
                    'stuck_on': group['name'],
                    'best_acc': best_acc
                }, output_path)
                break

        else:
            # Hit max epochs
            print(f"\n   ⏰ Max epochs reached. Best {group['name']}: {best_acc*100:.1f}%")
            # Still add to learned ops
            for code, name in group["ops"]:
                learned_ops.add(code)

    # Final evaluation
    print("\n" + "=" * 70)
    print("📊 FINAL EVALUATION")
    print("=" * 70)

    final_accs = evaluate_ops(model, [c for c, _ in ALL_OPS])
    all_good = True
    for op_code, op_name in ALL_OPS:
        acc = final_accs.get(op_name, 0)
        status = "✅" if acc >= target_accuracy else "❌"
        print(f"   {status} {op_name}: {acc*100:.1f}%")
        if acc < target_accuracy:
            all_good = False

    if all_good:
        print("\n🎉 ALL OPERATIONS AT TARGET ACCURACY!")
    else:
        print(f"\n⚠️ Some operations below target. Checkpoint saved.")

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output", type=str, default="models/final/fused_alu_grouped.pt")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--target", type=float, default=0.99, help="Target accuracy (0.99 = 99%)")
    parser.add_argument("--max-epochs", type=int, default=80, help="Max epochs per group")
    parser.add_argument("--patience", type=int, default=20, help="Epochs without improvement before moving on")
    parser.add_argument("--rehearsal", type=float, default=0.3, help="Rehearsal ratio (0.0-1.0)")
    args = parser.parse_args()

    train_grouped_curriculum(
        resume_path=args.resume,
        output_path=args.output,
        batch_size=args.batch_size,
        target_accuracy=args.target,
        max_epochs_per_group=args.max_epochs,
        patience=args.patience,
        rehearsal_ratio=args.rehearsal
    )
