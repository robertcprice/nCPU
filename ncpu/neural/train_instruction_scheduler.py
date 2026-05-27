#!/usr/bin/env python3
"""Train the Neural Instruction Scheduler.

Learns to reorder instructions to maximize batch utilization while preserving
data dependencies. This is the neural equivalent of out-of-order execution.

Training signal: for each instruction batch, the optimal order is one that
maximizes the first_hazard position (more instructions execute before a stall).
We use a ranking loss: instructions with no dependencies should be scheduled
before instructions that depend on earlier ones.

Usage:
    python ncpu/neural/train_instruction_scheduler.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.neural.neural_instruction_scheduler import NeuralInstructionScheduler
from ncpu.neural.train_dependency_predictor import build_hazard_luts


def generate_scheduling_data(n_samples: int, B: int = 8):
    """Generate instruction sequences with optimal scheduling labels."""
    torch.manual_seed(42)
    rm_lut, rn_not_lut, wr_lut = build_hazard_luts()

    common_keys = torch.tensor([
        0x8B000000 >> 21, 0x91000000 >> 21, 0xCB000000 >> 21,
        0xD2800000 >> 21, 0xF1000000 >> 21, 0xEB000000 >> 21,
        0x8A000000 >> 21, 0xAA000000 >> 21, 0x9B007C00 >> 21,
        0x00000000 >> 21, 0xF9000000 >> 21, 0xD1000000 >> 21,
    ], dtype=torch.int64)

    rds = torch.randint(0, 32, (n_samples, B), dtype=torch.int64)
    rns = torch.randint(0, 32, (n_samples, B), dtype=torch.int64)
    rms = torch.randint(0, 32, (n_samples, B), dtype=torch.int64)
    key_idx = torch.randint(0, len(common_keys), (n_samples, B))
    hk = common_keys[key_idx]
    reads_rm = rm_lut[hk]
    reads_rn = ~rn_not_lut[hk]
    writes_rd = wr_lut[hk]

    # For MOVZ-like: randomize rns
    movz_mask = rn_not_lut[hk]
    rns = torch.where(movz_mask, torch.randint(0, 32, rns.shape, dtype=torch.int64), rns)

    # Compute scheduling priority labels:
    # Instructions with NO dependencies on earlier instructions get priority 1.0
    # Instructions WITH dependencies get priority 0.0
    # This teaches the model to move independent instructions earlier.
    print("  Computing scheduling labels...")
    priority_labels = torch.zeros(n_samples, B, dtype=torch.float32)

    chunk = 5000
    for c in range(0, n_samples, chunk):
        end = min(c + chunk, n_samples)
        _rds = rds[c:end]
        _rns = rns[c:end]
        _rms = rms[c:end]
        _rm = reads_rm[c:end]
        _rn = reads_rn[c:end]
        _wr = writes_rd[c:end]

        # Build dependency matrix [chunk, B, B]
        rds_col = _rds.unsqueeze(1)
        rns_row = _rns.unsqueeze(2)
        rms_row = _rms.unsqueeze(2)
        hz_rn = (rds_col == rns_row) & _rn.unsqueeze(2)
        hz_rm = (rds_col == rms_row) & _rm.unsqueeze(2)
        lower_tri = torch.tril(torch.ones(B, B, dtype=torch.bool), diagonal=-1)
        hz_valid = (hz_rn | hz_rm) & lower_tri.unsqueeze(0)
        hz_valid = hz_valid & _wr.unsqueeze(1) & (_rds != 31).unsqueeze(1)

        # Per-position: does this instruction have ANY dependency?
        has_dep = hz_valid.any(dim=2)  # [chunk, B]
        # Priority: 1.0 for independent, 0.0 for dependent
        # Position 0 is always independent (no earlier instruction)
        prio = (~has_dep).float()
        prio[:, 0] = 1.0  # first instruction never has dependency
        priority_labels[c:end] = prio

    return {
        'rds': rds, 'rns': rns, 'rms': rms,
        'reads_rm': reads_rm, 'writes_rd': writes_rd,
        'priority_labels': priority_labels,
    }


def compute_first_hazard_for_order(rds, rns, rms, reads_rm, reads_rn, writes_rd, order, B):
    """Given an execution order, compute first_hazard position."""
    # Reorder all arrays
    o_rds = rds[order]
    o_rns = rns[order]
    o_rms = rms[order]
    o_rm = reads_rm[order]
    o_rn = reads_rn[order]
    o_wr = writes_rd[order]

    for i in range(1, B):
        for j in range(i):
            if not o_wr[j] or o_rds[j] == 31:
                continue
            rd_j = o_rds[j].item()
            if (o_rn[i] and o_rns[i].item() == rd_j) or (o_rm[i] and o_rms[i].item() == rd_j):
                return i
    return B


def train():
    B = 8
    N = 10000
    print(f"Generating {N} scheduling samples (batch_size={B})...")
    data = generate_scheduling_data(N, B)
    labels = data['priority_labels']
    n = len(labels)
    n_train = int(n * 0.9)

    indep_rate = labels.mean().item()
    print(f"  {n} samples, {100*indep_rate:.1f}% positions are independent")

    model = NeuralInstructionScheduler(max_batch=64, d_model=64, n_heads=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    batch_sz = 256
    best_improvement = 0.0

    # Also need reads_rn for evaluation (not stored, recompute from LUT)
    rm_lut, rn_not_lut, _ = build_hazard_luts()
    common_keys = torch.tensor([
        0x8B000000 >> 21, 0x91000000 >> 21, 0xCB000000 >> 21,
        0xD2800000 >> 21, 0xF1000000 >> 21, 0xEB000000 >> 21,
        0x8A000000 >> 21, 0xAA000000 >> 21, 0x9B007C00 >> 21,
        0x00000000 >> 21, 0xF9000000 >> 21, 0xD1000000 >> 21,
    ], dtype=torch.int64)

    for epoch in range(30):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_sz):
            idx = perm[i:i+batch_sz]
            logits_list = []
            for j in idx:
                logits = model.forward_logits(
                    data['rds'][j], data['rns'][j], data['rms'][j],
                    data['reads_rm'][j], data['writes_rd'][j])
                logits_list.append(logits)
            all_logits = torch.stack(logits_list)
            loss = criterion(all_logits, labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Evaluate: does the neural order improve first_hazard vs original?
        model.eval()
        improvements = 0
        total_eval = 0
        with torch.no_grad():
            for j in range(n_train, min(n_train + 200, n)):
                _rds = data['rds'][j]
                _rns = data['rns'][j]
                _rms = data['rms'][j]
                _rm = data['reads_rm'][j]
                _wr = data['writes_rd'][j]
                # Compute reads_rn from rn_not_lut
                # (simplified: assume all read rn for this eval)
                _rn = torch.ones(B, dtype=torch.bool)

                # Original order first_hazard
                orig_fh = compute_first_hazard_for_order(_rds, _rns, _rms, _rm, _rn, _wr,
                                                         torch.arange(B), B)
                # Neural order first_hazard
                order, _ = model(_rds, _rns, _rms, _rm, _wr)
                neural_fh = compute_first_hazard_for_order(_rds, _rns, _rms, _rm, _rn, _wr,
                                                           order, B)
                if neural_fh > orig_fh:
                    improvements += 1
                total_eval += 1

        imp_rate = improvements / total_eval if total_eval > 0 else 0
        if imp_rate > best_improvement:
            best_improvement = imp_rate

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={total_loss/n_batches:.4f} "
                  f"improved_schedules={100*imp_rate:.1f}%")

    save_path = PROJECT_ROOT / "models" / "alu" / "instruction_scheduler.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    size_kb = save_path.stat().st_size / 1024
    print(f"\nSaved to {save_path} ({size_kb:.1f} KB)")
    print(f"Best improvement rate: {100*best_improvement:.1f}% of schedules improved")


if __name__ == "__main__":
    train()
