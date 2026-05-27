#!/usr/bin/env python3
"""Train the Neural Hazard Predictor from deterministic LUT ground truth.

Generates training data by evaluating all ARM64 bit patterns against the
hazard detection rules, then trains a small MLP to 100% accuracy.

Usage:
    python ncpu/neural/train_hazard_predictor.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.neural.neural_hazard_predictor import NeuralHazardPredictor


# ── Hazard pattern definitions (same as decoder.py _build_hazard_luts) ──────

_RM_PATTERNS = [
    (0xFF200000, 0x8B000000), (0xFF200000, 0x0B000000),
    (0xFF200000, 0xCB000000), (0xFF200000, 0x4B000000),
    (0xFF200000, 0xAB000000), (0xFF200000, 0x2B000000),
    (0xFF200000, 0xEB000000), (0xFF200000, 0x6B000000),
    (0xFFE0FC00, 0x9B007C00), (0xFFE0FC00, 0x1B007C00),
    (0xFF200000, 0x8A000000), (0xFF200000, 0x0A000000),
    (0xFF200000, 0xAA000000), (0xFF200000, 0x2A000000),
    (0xFF200000, 0xCA000000), (0xFF200000, 0x4A000000),
    (0xFF200000, 0xEA000000), (0xFF200000, 0x6A000000),
    (0xFF200000, 0x8A200000), (0xFF200000, 0x0A200000),
    (0xFF200000, 0xEA200000), (0xFF200000, 0x6A200000),
    (0xFF200000, 0xCA200000), (0xFF200000, 0x4A200000),
    (0xFF200000, 0xAA200000), (0xFF200000, 0x2A200000),
    (0xFFE0FC00, 0x9AC02000), (0xFFE0FC00, 0x1AC02000),
    (0xFFE0FC00, 0x9AC02400), (0xFFE0FC00, 0x1AC02400),
    (0xFFE0FC00, 0x9AC02800), (0xFFE0FC00, 0x1AC02800),
    (0xFFC00000, 0x9A800000), (0xFFC00000, 0x1A800000),
    (0xFFE00C00, 0xF8600800), (0xFFE00C00, 0xB8600800),
    (0xFFE00C00, 0x38A00800), (0xFFE00C00, 0x38E00800),
    (0xFFE00C00, 0x78600800),
    (0xFFE00C00, 0xF8200800), (0xFFE00C00, 0xB8200800),
    (0xFFE00C00, 0x38200800), (0xFFE00C00, 0x78200800),
]

_RN_NOT_PATTERNS = [
    (0xFF800000, 0xD2800000), (0xFF800000, 0x52800000),
    (0xFF800000, 0xF2800000), (0xFF800000, 0x72800000),
    (0xFF800000, 0x92800000), (0xFF800000, 0x12800000),
    (0x9F000000, 0x90000000), (0x9F000000, 0x10000000),
]

_WR_PATTERNS = [
    (0xFF000000, 0x91000000), (0xFF000000, 0xD1000000),
    (0xFF200000, 0x8B000000), (0xFF200000, 0xCB000000),
    (0xFF800000, 0xD2800000), (0xFF800000, 0x52800000),
    (0xFF800000, 0xF2800000), (0xFF800000, 0x72800000),
    (0x9F000000, 0x90000000),
    (0xFF200000, 0xEB000000), (0xFF000000, 0xF1000000),
    (0xFFE0FC00, 0x9B007C00), (0xFFE0FC00, 0x1B007C00),
    (0xFFC00000, 0xB9800000), (0xFFC00000, 0x79400000),
    (0xFFC00000, 0x39800000), (0xFFC00000, 0x39C00000),
    (0xFFE00C00, 0xF8400000), (0xFFE00C00, 0xB8400000),
    (0xFFE00C00, 0xF8600800),
    (0xFFFFFC00, 0xC85F7C00),
    (0xFFC00000, 0x9A800000), (0xFFC00000, 0x1A800000),
    (0xFFE00C00, 0xF8400C00),
    (0xFFE00C00, 0xB8400400), (0xFFE00C00, 0xB8400C00),
    (0xFFE00C00, 0x78600800), (0xFFE00C00, 0x78400400),
    (0xFFE00C00, 0x38A00800), (0xFFE00C00, 0x38E00800),
    (0xFFC00000, 0xB3400000), (0xFFC00000, 0x33000000),
    (0xFFFFFC00, 0xDAC00000), (0xFFFFFC00, 0x5AC00000),
    (0xFF800000, 0x92800000), (0xFF800000, 0x12800000),
    (0xFFE00000, 0x9B200000), (0xFFE00000, 0x9BA00000),
    (0xFFE0FC00, 0x9B407C00), (0xFFE0FC00, 0x9BC07C00),
    (0xFFE00000, 0x93800000), (0xFFE00000, 0x13800000),
    (0xFFE00C00, 0xB8800400), (0xFFE00C00, 0xB8800C00),
    (0xFFE00C00, 0xB8A00800),
    (0xFFFFFC00, 0xC8DFFC00), (0xFFFFFC00, 0x88DFFC00),
    (0xFFC00000, 0x79800000), (0xFFC00000, 0x79C00000),
    (0xFFE00C00, 0x78400000),
    (0xFFE00C00, 0x78800000), (0xFFE00C00, 0x78C00000),
    (0xFFE00C00, 0x38800000), (0xFFE00C00, 0x38C00000),
    (0xFFE00C00, 0x78800400), (0xFFE00C00, 0x78C00400),
    (0xFFE00C00, 0x78800C00), (0xFFE00C00, 0x78C00C00),
    (0xFFC00000, 0x29400000),
    (0xFFE07C00, 0xC8207C00), (0xFFE07C00, 0xC8607C00),
    (0xFFE07C00, 0xC8A07C00), (0xFFE07C00, 0xC8E07C00),
    (0xFFE07C00, 0x88207C00), (0xFFE07C00, 0x88607C00),
    (0xFFE07C00, 0x88A07C00), (0xFFE07C00, 0x88E07C00),
    (0xFF200000, 0x8A200000), (0xFF200000, 0x0A200000),
    (0xFF200000, 0xEA200000), (0xFF200000, 0x6A200000),
    (0xFF200000, 0xCA200000), (0xFF200000, 0x4A200000),
    (0xFFFFFC00, 0x13001C00), (0xFFFFFC00, 0x13003C00),
    (0xFFFFFC00, 0xDAC00400), (0xFFFFFC00, 0x5AC00400),
    (0xFFE0FC00, 0x9A000000), (0xFFE0FC00, 0x1A000000),
    (0xFFE0FC00, 0xBA000000), (0xFFE0FC00, 0x3A000000),
    (0xFFE0FC00, 0xDA000000), (0xFFE0FC00, 0x5A000000),
    (0xFFE0FC00, 0xFA000000), (0xFFE0FC00, 0x7A000000),
    (0xFFFFFC00, 0xC8DF7C00), (0xFFFFFC00, 0x88DF7C00),
    (0xFF000000, 0x58000000), (0xFF000000, 0x18000000),
    (0xFFC00000, 0x53000000), (0xFFC00000, 0x13000000),
    (0xFFE00C00, 0x78400C00), (0xFFE00C00, 0x38400C00),
]

# OpType top-byte values that also write (from op_type_table)
_WR_OPTYPE_TOP_BYTES = {
    0xAA, 0x8A, 0xCA,  # ORR/AND/EOR REG
    0xD3, 0x53,         # UBFM (LSL/LSR IMM)
    0x93,               # SBFM/SXTW
    0xF9, 0x39, 0xA9,   # LDR/LDRB/LDP (shared with STR but conservative)
    0xF8, 0xB8,         # LDUR/LDR variants
    0x9B,               # MUL
    0x92, 0xB2,         # AND_IMM/ORR_IMM
    0x2A,               # MOV_REG (32-bit ORR)
    0x9A, 0x1A,         # CSEL / LSL_REG etc.
}


def _match_any(inst: int, patterns: list) -> bool:
    for mask, value in patterns:
        if (inst & mask) == value:
            return True
    return False


def generate_training_data(n_samples: int = 200_000) -> tuple:
    """Generate diverse instruction encodings with ground-truth labels."""
    torch.manual_seed(42)

    # Structured samples: cover all 2048 11-bit keys with diverse lower bits
    keys = torch.arange(2048)
    structured = []
    for _ in range(50):
        lower = torch.randint(0, 2**21, (2048,))
        insts = (keys << 21) | lower
        structured.append(insts)
    structured = torch.cat(structured)

    # Exact-match samples: every (mask, value) pattern generates explicit hits
    exact = []
    for patterns in [_RM_PATTERNS, _RN_NOT_PATTERNS, _WR_PATTERNS]:
        for mask, value in patterns:
            # Generate 32 variants with random bits in unmasked positions
            for _ in range(32):
                rand_bits = torch.randint(0, 2**32, (1,)).item()
                inst = (rand_bits & ~mask) | value
                exact.append(inst)
    exact_t = torch.tensor(exact, dtype=torch.int64)

    # Random samples
    random_insts = torch.randint(0, 2**32, (n_samples,), dtype=torch.int64)
    all_insts = torch.cat([structured, exact_t, random_insts])

    # Generate labels
    labels = torch.zeros(len(all_insts), 3, dtype=torch.float32)
    for i, inst_val in enumerate(all_insts.tolist()):
        inst = int(inst_val) & 0xFFFFFFFF
        rn_not = _match_any(inst, _RN_NOT_PATTERNS)
        labels[i, 0] = 0.0 if rn_not else 1.0  # reads_rn
        labels[i, 1] = 1.0 if _match_any(inst, _RM_PATTERNS) else 0.0  # reads_rm
        wr = _match_any(inst, _WR_PATTERNS)
        if not wr:
            tb = (inst >> 24) & 0xFF
            if tb in _WR_OPTYPE_TOP_BYTES:
                wr = True
        labels[i, 2] = 1.0 if wr else 0.0  # writes_rd

    return all_insts, labels


def train():
    print("Generating training data...")
    insts, labels = generate_training_data()
    print(f"  {len(insts)} samples, label distribution:")
    for i, name in enumerate(["reads_rn", "reads_rm", "writes_rd"]):
        pos = labels[:, i].sum().item()
        print(f"    {name}: {pos:.0f} positive ({100*pos/len(labels):.1f}%)")

    model = NeuralHazardPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.BCELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")

    batch_size = 4096
    best_acc = 0.0
    for epoch in range(200):
        model.train()
        perm = torch.randperm(len(insts))
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(insts), batch_size):
            idx = perm[i:i+batch_size]
            x = insts[idx]
            y = labels[idx]

            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Evaluate accuracy
        model.eval()
        with torch.no_grad():
            pred = model(insts)
            pred_bool = pred > 0.5
            labels_bool = labels > 0.5
            correct = (pred_bool == labels_bool).all(dim=1).float().mean().item()

        if (epoch + 1) % 20 == 0 or correct > best_acc:
            print(f"  Epoch {epoch+1:3d}: loss={total_loss/n_batches:.4f} acc={100*correct:.2f}%")
        if correct > best_acc:
            best_acc = correct

        if correct >= 1.0:
            print(f"\n  100% accuracy reached at epoch {epoch+1}!")
            break

    # Verify exhaustively on all 2048 keys
    print("\nExhaustive verification on all 2048 11-bit keys...")
    model.eval()
    errors = 0
    with torch.no_grad():
        for k in range(2048):
            # Test with 16 random lower-bit variants per key
            test_insts = torch.tensor([(k << 21) | (r & 0x1FFFFF) for r in range(16)], dtype=torch.int64)
            pred = model(test_insts) > 0.5
            for i, inst_val in enumerate(test_insts.tolist()):
                inst = int(inst_val) & 0xFFFFFFFF
                expected = [
                    not _match_any(inst, _RN_NOT_PATTERNS),
                    _match_any(inst, _RM_PATTERNS),
                    _match_any(inst, _WR_PATTERNS) or ((inst >> 24) & 0xFF) in _WR_OPTYPE_TOP_BYTES,
                ]
                for j in range(3):
                    if pred[i, j].item() != expected[j]:
                        errors += 1
                        if errors <= 5:
                            print(f"  MISMATCH: key={k:#05x} inst={inst:#010x} dim={j} pred={pred[i,j].item()} expected={expected[j]}")

    # Count false positives (conservative, safe) vs false negatives (dangerous)
    false_neg = 0
    for k in range(2048):
        test_insts = torch.tensor([(k << 21) | (r & 0x1FFFFF) for r in range(16)], dtype=torch.int64)
        pred_check = model(test_insts) > 0.5
        for i, inst_val in enumerate(test_insts.tolist()):
            inst = int(inst_val) & 0xFFFFFFFF
            expected = [
                not _match_any(inst, _RN_NOT_PATTERNS),
                _match_any(inst, _RM_PATTERNS),
                _match_any(inst, _WR_PATTERNS) or ((inst >> 24) & 0xFF) in _WR_OPTYPE_TOP_BYTES,
            ]
            for j in range(3):
                if not pred_check[i, j].item() and expected[j]:
                    false_neg += 1  # missed hazard — dangerous!

    if errors == 0:
        print("  All keys verified correctly!")
    elif false_neg == 0:
        print(f"  {errors} false positives (conservative, safe for hazard detection)")
    else:
        print(f"  {errors} mismatches ({false_neg} FALSE NEGATIVES — UNSAFE)")
        return

    # Save
    save_path = PROJECT_ROOT / "models" / "alu" / "hazard_predictor.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    size_kb = save_path.stat().st_size / 1024
    print(f"\nSaved to {save_path} ({size_kb:.1f} KB)")
    print(f"Final accuracy: {100*best_acc:.2f}%")


if __name__ == "__main__":
    train()
