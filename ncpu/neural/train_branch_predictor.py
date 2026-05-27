#!/usr/bin/env python3
"""Train the Neural Branch Predictor.

Generates synthetic branch scenarios with known outcomes and trains the model
to predict taken/not-taken from flags + condition code + context.

Usage:
    python ncpu/neural/train_branch_predictor.py
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.neural.neural_branch_predictor import NeuralBranchPredictor


def eval_condition(cond, n, z, c, v):
    """Evaluate ARM64 condition code against NZCV flags."""
    if cond == 0: return z            # EQ
    if cond == 1: return not z        # NE
    if cond == 2: return c            # CS
    if cond == 3: return not c        # CC
    if cond == 4: return n            # MI
    if cond == 5: return not n        # PL
    if cond == 6: return v            # VS
    if cond == 7: return not v        # VC
    if cond == 8: return c and not z  # HI
    if cond == 9: return not c or z   # LS
    if cond == 10: return n == v      # GE
    if cond == 11: return n != v      # LT
    if cond == 12: return not z and (n == v)  # GT
    if cond == 13: return z or (n != v)       # LE
    if cond == 14: return True        # AL
    return False                      # NV


def generate_data(n_samples=30000):
    """Generate branch prediction training data."""
    torch.manual_seed(42)

    conds = torch.randint(0, 15, (n_samples,))  # condition codes
    # Flags: mix of random and structured (loop-like patterns)
    flags = torch.zeros(n_samples, 4)
    flag_histories = torch.zeros(n_samples, 4, 4)
    is_backward = torch.zeros(n_samples)
    counter_hints = torch.zeros(n_samples)
    taken = torch.zeros(n_samples)

    for i in range(n_samples):
        cc = conds[i].item()

        if i < n_samples // 3:
            # Random flags
            f = (torch.rand(4) > 0.5).float()
        elif i < 2 * n_samples // 3:
            # Loop-like: counter decrementing, Z=0 until last iteration
            counter = torch.randint(1, 1000, (1,)).item()
            iter_num = torch.randint(0, counter + 1, (1,)).item()
            f = torch.tensor([0.0, 1.0 if iter_num == counter else 0.0, 1.0, 0.0])
            is_backward[i] = 1.0
            counter_hints[i] = max(0, (counter - iter_num)) / max(counter, 1)
        else:
            # Comparison-like: random signed comparison results
            a = torch.randint(-100, 100, (1,)).item()
            b = torch.randint(-100, 100, (1,)).item()
            result = a - b
            f = torch.tensor([
                1.0 if result < 0 else 0.0,   # N
                1.0 if result == 0 else 0.0,   # Z
                1.0 if a >= b else 0.0,        # C (unsigned)
                0.0,                            # V (simplified)
            ])

        flags[i] = f

        # Generate flag history (4 previous states)
        for h in range(4):
            flag_histories[i, h] = (torch.rand(4) > 0.5).float()
        flag_histories[i, -1] = f  # most recent = current

        # Ground truth
        n_f, z_f, c_f, v_f = f[0].item() > 0.5, f[1].item() > 0.5, f[2].item() > 0.5, f[3].item() > 0.5
        taken[i] = 1.0 if eval_condition(cc, n_f, z_f, c_f, v_f) else 0.0

    return conds, flags, flag_histories, is_backward, counter_hints, taken


def train():
    print("Generating training data...")
    conds, flags, flag_histories, is_backward, counter_hints, taken = generate_data(30000)
    n = len(conds)
    n_train = int(n * 0.9)
    print(f"  {n} samples, {taken.sum().item():.0f} taken ({100*taken.mean().item():.1f}%)")

    model = NeuralBranchPredictor()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.BCELoss()
    batch_sz = 256
    best_acc = 0.0

    for epoch in range(60):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_sz):
            idx = perm[i:i+batch_sz]
            preds = []
            for j in idx:
                model._flag_history = flag_histories[j]
                p = model(conds[j].item(), flags[j], is_backward[j].item() > 0.5,
                         counter_hints[j].item())
                preds.append(p)
            pred_t = torch.stack(preds)
            loss = criterion(pred_t, taken[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for j in range(n_train, min(n_train + 500, n)):
                model._flag_history = flag_histories[j]
                p = model(conds[j].item(), flags[j], is_backward[j].item() > 0.5,
                         counter_hints[j].item())
                pred = p.item() > 0.5
                gt = taken[j].item() > 0.5
                if pred == gt:
                    correct += 1
                total += 1

        acc = correct / total
        if acc > best_acc:
            best_acc = acc
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={total_loss/n_batches:.4f} acc={100*acc:.1f}%")
        if acc >= 0.99:
            print(f"\n  99%+ accuracy at epoch {epoch+1}!")
            break

    save_path = PROJECT_ROOT / "models" / "alu" / "branch_predictor.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    size_kb = save_path.stat().st_size / 1024
    print(f"\nSaved to {save_path} ({size_kb:.1f} KB)")
    print(f"Best accuracy: {100*best_acc:.1f}%")


if __name__ == "__main__":
    train()
