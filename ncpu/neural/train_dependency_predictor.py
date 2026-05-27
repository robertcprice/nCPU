#!/usr/bin/env python3
"""Train the Neural Dependency Graph Predictor.

Fully vectorized data generation + batched training. Generates instruction
sequences, computes ground-truth hazard matrices via tensor ops, trains
a Conv1d predictor to predict first_hazard from register fields.

Usage:
    python ncpu/neural/train_dependency_predictor.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.neural.neural_dependency_predictor import NeuralDependencyPredictor


def build_hazard_luts():
    """Build 2048-entry LUTs for reads_rm, rn_not_reg, writes_rd (same as decoder.py)."""
    _rm_patterns = [
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
    _rn_not_patterns = [
        (0xFF800000, 0xD2800000), (0xFF800000, 0x52800000),
        (0xFF800000, 0xF2800000), (0xFF800000, 0x72800000),
        (0xFF800000, 0x92800000), (0xFF800000, 0x12800000),
        (0x9F000000, 0x90000000), (0x9F000000, 0x10000000),
    ]
    _wr_patterns = [
        (0xFF000000, 0x91000000), (0xFF000000, 0xD1000000),
        (0xFF200000, 0x8B000000), (0xFF200000, 0xCB000000),
        (0xFF800000, 0xD2800000), (0xFF800000, 0x52800000),
        (0xFF800000, 0xF2800000), (0xFF800000, 0x72800000),
        (0x9F000000, 0x90000000),
        (0xFF200000, 0xEB000000), (0xFF000000, 0xF1000000),
        (0xFFE0FC00, 0x9B007C00), (0xFFE0FC00, 0x1B007C00),
        (0xFFC00000, 0x9A800000), (0xFFC00000, 0x1A800000),
        (0xFF800000, 0x92800000), (0xFF800000, 0x12800000),
        (0xFFE0FC00, 0x9A000000), (0xFFE0FC00, 0x1A000000),
        (0xFFE0FC00, 0xBA000000), (0xFFE0FC00, 0x3A000000),
        (0xFFE0FC00, 0xDA000000), (0xFFE0FC00, 0x5A000000),
        (0xFFE0FC00, 0xFA000000), (0xFFE0FC00, 0x7A000000),
        (0xFF000000, 0x58000000), (0xFF000000, 0x18000000),
    ]
    _wr_top_bytes = {0xAA,0x8A,0xCA,0xD3,0x53,0x93,0xF9,0x39,
                     0xA9,0xF8,0xB8,0x9B,0x92,0xB2,0x2A,0x9A,0x1A}

    def _match(k, patterns):
        for mask, value in patterns:
            km = (mask >> 21) & 0x7FF
            kv = (value >> 21) & 0x7FF
            if (k & km) == kv:
                return True
        return False

    rm_lut = torch.zeros(2048, dtype=torch.bool)
    rn_not_lut = torch.zeros(2048, dtype=torch.bool)
    wr_lut = torch.zeros(2048, dtype=torch.bool)

    wr_optype_keys = set()
    for tb in _wr_top_bytes:
        for low3 in range(8):
            wr_optype_keys.add((tb << 3) | low3)

    for k in range(2048):
        rm_lut[k] = _match(k, _rm_patterns)
        rn_not_lut[k] = _match(k, _rn_not_patterns)
        wr_lut[k] = _match(k, _wr_patterns) or (k in wr_optype_keys)

    return rm_lut, rn_not_lut, wr_lut


def generate_training_data_vectorized(n_samples: int, B: int = 8):
    """Fully vectorized training data generation — ~1000x faster than per-sample."""
    torch.manual_seed(42)
    print("  Building hazard LUTs...")
    rm_lut, rn_not_lut, wr_lut = build_hazard_luts()

    # Generate random register assignments [N, B]
    rds = torch.randint(0, 32, (n_samples, B), dtype=torch.int64)
    rns = torch.randint(0, 32, (n_samples, B), dtype=torch.int64)
    rms = torch.randint(0, 32, (n_samples, B), dtype=torch.int64)

    # Generate random 11-bit hazard keys to determine instruction type
    # Mix of common ARM64 instruction types
    common_keys = torch.tensor([
        0x8B000000 >> 21,  # ADD REG 64
        0x91000000 >> 21,  # ADD IMM 64
        0xCB000000 >> 21,  # SUB REG 64
        0xD2800000 >> 21,  # MOVZ 64
        0xF1000000 >> 21,  # SUBS IMM 64
        0xEB000000 >> 21,  # SUBS REG 64
        0x8A000000 >> 21,  # AND REG 64
        0xAA000000 >> 21,  # ORR REG 64
        0x9B007C00 >> 21,  # MUL 64 (approx key)
        0x00000000 >> 21,  # NOP
        0xF9000000 >> 21,  # STR IMM 64
        0xB9000000 >> 21,  # STR IMM 32
        0xD1000000 >> 21,  # SUB IMM 64
        0x52800000 >> 21,  # MOVZ 32
    ], dtype=torch.int64)

    key_indices = torch.randint(0, len(common_keys), (n_samples, B))
    hk = common_keys[key_indices]  # [N, B]

    # Lookup hazard properties from LUTs
    reads_rm = rm_lut[hk]       # [N, B] bool
    reads_rn = ~rn_not_lut[hk]  # [N, B] bool
    writes_rd = wr_lut[hk]      # [N, B] bool

    # For MOVZ-like instructions (rn_not), randomize the rns field
    # (simulates the imm16 encoding in rns bits)
    movz_mask = rn_not_lut[hk]
    rns = torch.where(movz_mask, torch.randint(0, 32, rns.shape, dtype=torch.int64), rns)

    # Compute per-position hazard labels using the deterministic matrix algorithm
    # Vectorized: for each sample, build [B, B] hazard matrix
    print("  Computing ground-truth hazard matrices...")
    hazard_labels = torch.zeros(n_samples, B, dtype=torch.float32)

    # Process in chunks for memory efficiency
    chunk = 5000
    for c in range(0, n_samples, chunk):
        end = min(c + chunk, n_samples)
        n_chunk = end - c

        _rds = rds[c:end]       # [chunk, B]
        _rns = rns[c:end]
        _rms = rms[c:end]
        _rm = reads_rm[c:end]
        _rn = reads_rn[c:end]
        _wr = writes_rd[c:end]

        # Build [chunk, B, B] hazard matrices via broadcasting
        rds_col = _rds.unsqueeze(1)       # [chunk, 1, B] — writer destinations
        rns_row = _rns.unsqueeze(2)       # [chunk, B, 1] — reader rn
        rms_row = _rms.unsqueeze(2)       # [chunk, B, 1] — reader rm

        # hazard[n, i, j] = inst i reads what inst j writes
        hz_rn = (rds_col == rns_row) & _rn.unsqueeze(2)   # [chunk, B, B]
        hz_rm = (rds_col == rms_row) & _rm.unsqueeze(2)    # [chunk, B, B]
        hz_any = hz_rn | hz_rm

        # Lower triangular: only j < i
        lower_tri = torch.tril(torch.ones(B, B, dtype=torch.bool), diagonal=-1)
        hz_valid = hz_any & lower_tri.unsqueeze(0)

        # Filter by writes_rd and rd != 31
        hz_valid = hz_valid & _wr.unsqueeze(1) & (_rds != 31).unsqueeze(1)

        # Per-position: does any earlier instruction cause a hazard?
        has_hazard = hz_valid.any(dim=2)  # [chunk, B]
        hazard_labels[c:end] = has_hazard.float()

    return {
        'rds': rds, 'rns': rns, 'rms': rms,
        'reads_rm': reads_rm, 'writes_rd': writes_rd,
        'hazard_labels': hazard_labels,
    }


def train():
    B = 8
    N = 100000  # 100K samples, vectorized generation
    print(f"Generating {N} training samples (batch_size={B})...")
    data = generate_training_data_vectorized(N, B)
    labels = data['hazard_labels']
    n = len(labels)
    n_train = int(n * 0.9)

    has_any = labels.any(dim=1).sum().item()
    print(f"  {n} samples, {has_any} ({100*has_any/n:.1f}%) have hazards")
    print(f"  {labels.sum().item():.0f} total hazard positions")

    model = NeuralDependencyPredictor(max_batch=64, d_model=64, n_heads=4, reg_dim=16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    pos_weight = torch.tensor([8.0])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    BIG = torch.tensor(B * 2, dtype=torch.int64)
    batch_sz = 1024
    best_acc = 0.0

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400)
    for epoch in range(400):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_sz):
            idx = perm[i:i+batch_sz]

            # Fully batched forward: [batch_sz, B] inputs
            all_logits = model.forward_logits(
                data['rds'][idx], data['rns'][idx], data['rms'][idx],
                data['reads_rm'][idx], data['writes_rd'][idx])  # [batch_sz, B]
            all_labels = labels[idx]
            loss = criterion(all_logits, all_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Evaluate: batched first_hazard comparison
        model.eval()
        with torch.no_grad():
            eval_logits = model.forward_logits(
                data['rds'][n_train:], data['rns'][n_train:], data['rms'][n_train:],
                data['reads_rm'][n_train:], data['writes_rd'][n_train:])  # [n_eval, B]
            eval_preds = torch.sigmoid(eval_logits) > 0.5  # [n_eval, B]
            eval_labels_b = labels[n_train:] > 0.5

            # first_hazard for predictions: first True position per sample
            pred_fh = torch.where(eval_preds, torch.arange(B).unsqueeze(0), B * 2)
            pred_first = pred_fh.min(dim=1).values  # [n_eval]
            # first_hazard for ground truth
            gt_fh = torch.where(eval_labels_b, torch.arange(B).unsqueeze(0), B * 2)
            gt_first = gt_fh.min(dim=1).values
            # Match: same first_hazard OR both >= B (no hazard)
            match = (pred_first == gt_first) | ((pred_first >= B) & (gt_first >= B))
            acc = match.float().mean().item()
        scheduler.step()
        if acc > best_acc:
            best_acc = acc
            # Save best model
            _save = PROJECT_ROOT / "models" / "alu" / "dependency_predictor.pt"
            _save.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), _save)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={total_loss/n_batches:.4f} first_hazard_acc={100*acc:.1f}% lr={optimizer.param_groups[0]['lr']:.5f}")
        if acc >= 0.98:
            print(f"\n  98%+ accuracy at epoch {epoch+1}!")
            break

    save_path = PROJECT_ROOT / "models" / "alu" / "dependency_predictor.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    size_kb = save_path.stat().st_size / 1024
    print(f"\nSaved to {save_path} ({size_kb:.1f} KB)")
    print(f"Best accuracy: {100*best_acc:.1f}%")


if __name__ == "__main__":
    train()
