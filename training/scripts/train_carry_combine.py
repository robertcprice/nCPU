#!/usr/bin/env python3
"""Train the carry-combine neural network for parallel-prefix (Kogge-Stone) addition.

The carry-combine operator computes:
    G_out = G_i | (P_i & G_j)
    P_out = P_i & P_j

where G = generate, P = propagate. This has only 2^4 = 16 input combinations,
so training to 100% accuracy is trivial and takes seconds.

Usage:
    python3 training/train_carry_combine.py

Output:
    models/alu/carry_combine.pt
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SAVE_PATH = PROJECT_ROOT / "models" / "alu" / "carry_combine.pt"


class NeuralCarryCombine(nn.Module):
    """Neural carry-combine operator for parallel-prefix addition.

    Computes (G_out, P_out) = (G_i | (P_i & G_j), P_i & P_j)
    Trained on all 2^4 = 16 input combinations to 100% accuracy.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, gp_pairs: torch.Tensor) -> torch.Tensor:
        """gp_pairs: [batch, 4] -> [batch, 2] (G_out, P_out)"""
        return torch.sigmoid(self.net(gp_pairs))


def generate_truth_table() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate all 16 input/output pairs for carry-combine.

    Input: [G_i, P_i, G_j, P_j] — 4 binary values
    Output: [G_out, P_out] — 2 binary values
        G_out = G_i | (P_i & G_j)
        P_out = P_i & P_j
    """
    inputs = []
    outputs = []

    for g_i in [0.0, 1.0]:
        for p_i in [0.0, 1.0]:
            for g_j in [0.0, 1.0]:
                for p_j in [0.0, 1.0]:
                    g_out = max(g_i, p_i * g_j)  # G_i | (P_i & G_j)
                    p_out = p_i * p_j             # P_i & P_j
                    inputs.append([g_i, p_i, g_j, p_j])
                    outputs.append([g_out, p_out])

    return torch.tensor(inputs), torch.tensor(outputs)


def train():
    print("=" * 60)
    print("  Training Carry-Combine Network")
    print("  Truth table: 16 entries (2^4 inputs)")
    print("=" * 60)

    model = NeuralCarryCombine(hidden_dim=64)
    inputs, targets = generate_truth_table()

    print(f"\n  Architecture: Linear(4,64) -> ReLU -> Linear(64,32) -> ReLU -> Linear(32,2)")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Training samples: {len(inputs)}")

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    best_acc = 0.0
    for epoch in range(2000):
        model.train()
        out = model(inputs)
        loss = criterion(out, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check accuracy
        with torch.no_grad():
            preds = (out > 0.5).float()
            correct = (preds == targets).all(dim=1).sum().item()
            acc = correct / len(targets)

        if acc > best_acc:
            best_acc = acc

        if (epoch + 1) % 100 == 0 or acc == 1.0:
            print(f"  Epoch {epoch+1:>4d}: loss={loss.item():.6f}  accuracy={acc*100:.1f}%")

        if acc == 1.0:
            print(f"\n  100% accuracy achieved at epoch {epoch+1}!")
            break
    else:
        print(f"\n  WARNING: Did not reach 100% accuracy (best: {best_acc*100:.1f}%)")

    # Verify exhaustively
    model.eval()
    with torch.no_grad():
        out = model(inputs)
        preds = (out > 0.5).float()
        all_correct = (preds == targets).all().item()

    print(f"\n  Exhaustive verification: {'PASS' if all_correct else 'FAIL'}")

    if all_correct:
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  Saved to {SAVE_PATH}")
        print(f"  File size: {SAVE_PATH.stat().st_size:,} bytes")
    else:
        print("  NOT saving — model did not achieve 100% accuracy")
        return 1

    # Print truth table for verification
    print(f"\n  Truth Table Verification:")
    print(f"  {'G_i':>3} {'P_i':>3} {'G_j':>3} {'P_j':>3} | {'G_out':>5} {'P_out':>5} | {'Pred_G':>6} {'Pred_P':>6}")
    print(f"  {'-'*3:>3} {'-'*3:>3} {'-'*3:>3} {'-'*3:>3} | {'-'*5:>5} {'-'*5:>5} | {'-'*6:>6} {'-'*6:>6}")
    with torch.no_grad():
        out = model(inputs)
        for i in range(16):
            gi, pi, gj, pj = inputs[i].tolist()
            go, po = targets[i].tolist()
            pg, pp = (out[i] > 0.5).float().tolist()
            match = "OK" if pg == go and pp == po else "FAIL"
            print(f"  {gi:>3.0f} {pi:>3.0f} {gj:>3.0f} {pj:>3.0f} | {go:>5.0f} {po:>5.0f} | {pg:>6.0f} {pp:>6.0f}  {match}")

    print(f"\n{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(train())
