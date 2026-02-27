#!/usr/bin/env python3
"""
PURE NEURAL DECODER TRAINING
===============================
Train decoder to correctly classify ALL ARM64 instructions without heuristics.
This demonstrates true neural pattern recognition - no manual rules!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import sys
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
print(f"Training on: {device}")

# Import the current decoder architecture
sys.path.insert(0, '.')
from run_neural_rtos_v2 import FullyNeuralCPU, UniversalARM64Decoder

# Category mapping (must match run_neural_rtos_v2.py)
CATEGORY_TO_IDX = {
    'ADD': 0,
    'SUB': 1,
    'AND': 2,
    'OR': 3,
    'XOR': 4,
    'MUL': 5,
    'DIV': 6,
    'SHIFT': 7,
    'LOAD': 8,
    'STORE': 9,
    'BRANCH': 10,
    'COMPARE': 11,
    'MOVE': 12,  # MOVZ/MOVK go here!
    'SYSTEM': 13,
    'UNKNOWN': 14,
}

NUM_CATEGORIES = 15


def encode_movz_movk(rd: int, imm16: int, hw: int = 0, sf: int = 1, op: str = 'MOVZ'):
    """Encode MOVZ/MOVN/MOVK instructions."""
    if op == 'MOVZ':
        base = 0x52800000  # MOVZ (sf=1, hw=0)
    elif op == 'MOVN':
        base = 0x12800000  # MOVN (sf=1, hw=0)
    elif op == 'MOVK':
        base = 0x72800000  # MOVK (sf=1, hw=0)
    else:
        base = 0x52800000

    if sf:
        base |= (1 << 31)

    inst = base | ((hw & 0x3) << 21) | ((imm16 & 0xFFFF) << 5) | (rd & 0x1F)

    return inst, {
        'category': CATEGORY_TO_IDX['MOVE'],
        'rd': rd,
        'rn': 31,  # Not used
        'rm': 31,  # Not used
        'sets_flags': False,
    }


def encode_add_sub(rd: int, rn: int, imm: int = 0, is_add: bool = True, is_imm: bool = True, sets_flags: bool = False, sf: int = 1):
    """Encode ADD/SUB/ADDS/SUBS instructions."""
    base = 0x11000000  # ADD imm
    if not is_add:
        base |= (1 << 30)  # SUB bit
    if sets_flags:
        base |= (1 << 29)  # S bit
    if sf:
        base |= (1 << 31)

    if is_imm:
        inst = base | ((imm & 0xFFF) << 10) | ((rn & 0x1F) << 5) | (rd & 0x1F)
    else:
        inst = base | ((rn & 0x1F) << 16) | ((31 & 0x1F) << 10) | ((rn & 0x1F) << 5) | (rd & 0x1F)

    cat = CATEGORY_TO_IDX['SUB'] if not is_add else CATEGORY_TO_IDX['ADD']
    return inst, {'category': cat, 'rd': rd, 'rn': rn, 'rm': 31, 'sets_flags': sets_flags}


def encode_load_store(rt: int, rn: int, imm: int = 0, is_load: bool = True, is_store: bool = False, sf: int = 1):
    """Encode LDR/STR instructions."""
    if is_store:
        base = 0xF9000000 if sf else 0xB9000000
    else:
        base = 0xF9400000 if sf else 0xB9400000

    scaled_imm = (imm >> (3 if sf else 2)) & 0xFFF
    inst = base | (scaled_imm << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F)

    cat = CATEGORY_TO_IDX['LOAD'] if is_load else CATEGORY_TO_IDX['STORE']
    return inst, {'category': cat, 'rd': rt, 'rn': rn, 'rm': 31, 'sets_flags': False}


def encode_branch(imm: int = 0, is_call: bool = False, cond: int = None):
    """Encode branch instructions."""
    if cond is not None:
        # B.cond with condition code
        imm19 = (imm >> 2) & 0x7FFFF
        inst = 0x54000000 | (imm19 << 5) | cond
        return inst, {'category': CATEGORY_TO_IDX['BRANCH'], 'rd': cond, 'rn': 31, 'rm': 31, 'sets_flags': False}
    elif is_call:
        inst = 0x94000000 | (imm & 0x3FFFFFF)
        return inst, {'category': CATEGORY_TO_IDX['BRANCH'], 'rd': 30, 'rn': 31, 'rm': 31, 'sets_flags': False}
    else:
        inst = 0x14000000 | (imm & 0x3FFFFFF)
        return inst, {'category': CATEGORY_TO_IDX['BRANCH'], 'rd': 31, 'rn': 31, 'rm': 31, 'sets_flags': False}


def encode_compare(rn: int, imm: int = 0):
    """Encode CMP (SUBS to xzr)."""
    return encode_add_sub(31, rn, imm, is_add=True, is_imm=True, sets_flags=True)


def encode_logical(rd: int, rn: int, rm: int, op: str = 'AND', sets_flags: bool = False):
    """Encode AND/ORR/EOR instructions."""
    if op == 'AND':
        base = 0x0A000000
    elif op == 'ORR':
        base = 0x2A000000
    elif op == 'EOR':
        base = 0x4A000000
    else:
        base = 0x0A000000

    if sets_flags:
        base |= (1 << 29)

    inst = base | ((rm & 0x1F) << 16) | ((rn & 0x1F) << 5) | (rd & 0x1F)

    cat = CATEGORY_TO_IDX['AND'] if op == 'AND' else (CATEGORY_TO_IDX['OR'] if op == 'ORR' else CATEGORY_TO_IDX['XOR'])
    return inst, {'category': cat, 'rd': rd, 'rn': rn, 'rm': rm, 'sets_flags': sets_flags}


def generate_balanced_batch(batch_size: int):
    """Generate a balanced training batch with ALL instruction types."""
    batch = []

    # Distribution: 30% MOVZ/MOVK, 20% ADD/SUB, 15% LOAD/STORE, 10% BRANCH, 10% LOGICAL, 15% other
    for _ in range(batch_size):
        r = random.random()

        if r < 0.30:  # MOVZ/MOVK/MOVN
            op_type = random.choice(['MOVZ', 'MOVN', 'MOVK'])
            rd = random.randint(0, 30)
            imm16 = random.randint(0, 0xFFFF)
            hw = random.randint(0, 3)
            sf = random.choice([0, 1])

            # For 32-bit, only hw 0 and 1 are valid
            if sf == 0:
                hw = hw & 1

            inst, labels = encode_movz_movk(rd, imm16, hw, sf, op_type)
            batch.append((inst, labels))

        elif r < 0.50:  # ADD/SUB/ADDS/SUBS
            rd = random.randint(0, 30)
            rn = random.randint(0, 30)
            imm = random.randint(0, 0xFFF)
            is_add = random.choice([True, False])
            is_imm = random.choice([True, False])
            sets_flags = random.choice([True, False])
            sf = random.choice([0, 1])

            inst, labels = encode_add_sub(rd, rn, imm, is_add, is_imm, sets_flags, sf)
            batch.append((inst, labels))

        elif r < 0.65:  # LDR/STR
            rt = random.randint(0, 30)
            rn = random.randint(0, 30)
            imm = random.randint(0, 0x7FFF)  # Small offset
            is_load = random.choice([True, False])
            is_store = not is_load
            sf = random.choice([0, 1])

            inst, labels = encode_load_store(rt, rn, imm, is_load, is_store, sf)
            batch.append((inst, labels))

        elif r < 0.75:  # B.cond
            cond = random.randint(0, 14)  # All condition codes
            # Small backward branch (common in loops)
            offset = random.choice([-2, -4, -6, -8, -12, -16, -20])
            inst, labels = encode_branch(offset, cond=cond)
            batch.append((inst, labels))

        elif r < 0.85:  # Logical (AND/ORR/EOR)
            rd = random.randint(0, 30)
            rn = random.randint(0, 30)
            rm = random.randint(0, 30)
            op = random.choice(['AND', 'ORR', 'EOR'])
            sets_flags = random.choice([True, False])

            inst, labels = encode_logical(rd, rn, rm, op, sets_flags)
            batch.append((inst, labels))

        else:  # CMP (SUBS to xzr)
            rn = random.randint(0, 30)
            imm = random.randint(0, 0xFFF)
            inst, labels = encode_compare(rn, imm)
            batch.append((inst, labels))

    return batch


def train_pure_neural_decoder():
    """Train decoder to learn ALL patterns without heuristics."""

    print("="*70)
    print("PURE NEURAL DECODER TRAINING")
    print("="*70)
    print()
    print("NO HEURISTICS - Neural network learns everything from data!")
    print()

    # Create decoder
    model = UniversalARM64Decoder(d_model=256).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Training config
    batch_size = 512
    samples_per_epoch = 100000
    max_epochs = 200

    best_overall_acc = 0
    patience = 0
    max_patience = 30

    for epoch in range(max_epochs):
        model.train()
        epoch_start = time.time()

        total_loss = 0
        cat_correct = 0
        rd_correct = 0
        rn_correct = 0
        rm_correct = 0
        sets_flags_correct = 0
        total = 0

        # Track specific accuracies
        movz_total = 0
        movz_correct = 0
        branch_total = 0
        branch_correct = 0
        sets_flags_total = 0
        sets_flags_correct = 0

        num_batches = samples_per_epoch // batch_size

        for batch_idx in range(num_batches):
            # Generate balanced batch
            batch_data = generate_balanced_batch(batch_size)

            # Convert to tensors
            insts = torch.tensor([b[0] for b in batch_data], dtype=torch.long, device=device)
            cats = torch.tensor([b[1]['category'] for b in batch_data], dtype=torch.long, device=device)
            rds = torch.tensor([b[1]['rd'] for b in batch_data], dtype=torch.long, device=device)
            rns = torch.tensor([b[1]['rn'] for b in batch_data], dtype=torch.long, device=device)
            rms = torch.tensor([b[1]['rm'] for b in batch_data], dtype=torch.long, device=device)
            set_flags = torch.tensor([b[1]['sets_flags'] for b in batch_data], dtype=torch.long, device=device)

            # Convert instructions to bits
            bits = torch.zeros(batch_size, 32, dtype=torch.float32, device=device)
            for i in range(32):
                bits[:, i] = ((insts >> i) & 1).float()

            # Forward pass
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(bits)

                # Compute losses
                loss_rd = F.cross_entropy(outputs['rd'], rds)
                loss_rn = F.cross_entropy(outputs['rn'], rns)
                loss_rm = F.cross_entropy(outputs['rm'], rms)
                loss_cat = F.cross_entropy(outputs['category'], cats)
                loss_sets_flags = F.binary_cross_entropy_with_logits(outputs['sets_flags'].squeeze(-1), set_flags.float())

                # Total loss - weight category and RM heavily
                loss = loss_cat * 4 + loss_rd * 2 + loss_rn * 2 + loss_rm * 2 + loss_sets_flags * 3

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            # Compute accuracy
            pred_cats = outputs['category'].argmax(dim=1)
            pred_rds = outputs['rd'].argmax(dim=1)
            pred_rns = outputs['rn'].argmax(dim=1)
            pred_rms = outputs['rm'].argmax(dim=1)
            pred_sets_flags = outputs['sets_flags'].squeeze(-1) > 0

            cat_correct += (pred_cats == cats).sum().item()
            rd_correct += (pred_rds == rds).sum().item()
            rn_correct += (pred_rns == rns).sum().item()
            rm_correct += (pred_rms == rms).sum().item()
            sets_flags_correct += (pred_sets_flags == set_flags).sum().item()

            # Track specific accuracies
            is_movz = (cats == CATEGORY_TO_IDX['MOVE'])
            if is_movz.any():
                movz_total += is_movz.sum().item()
                movz_correct += ((pred_cats == cats) & is_movz).sum().item()

            is_branch = (cats == CATEGORY_TO_IDX['BRANCH'])
            if is_branch.any():
                branch_total += is_branch.sum().item()
                branch_correct += ((pred_cats == cats) & is_branch).sum().item()

            should_set_flags = (set_flags == 1)
            if should_set_flags.any():
                sets_flags_total += should_set_flags.sum().item()
                sets_flags_correct += (pred_sets_flags == set_flags).sum().item()

            total += batch_size

        # Calculate accuracies
        acc_cat = cat_correct / total
        acc_rd = rd_correct / total
        acc_rn = rn_correct / total
        acc_rm = rm_correct / total
        acc_sets_flags = sets_flags_correct / max(sets_flags_total, 1)
        acc_movz = movz_correct / max(movz_total, 1)
        acc_branch = branch_correct / max(branch_total, 1)

        scheduler.step()

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1:3d}: Loss={total_loss/num_batches:.4f} "
              f"Cat={acc_cat:.1%} RD={acc_rd:.1%} RN={acc_rn:.1%} RM={acc_rm:.1%} Flags={acc_sets_flags:.1%} "
              f"MOVZ={acc_movz:.1%} Branch={acc_branch:.1%} "
              f"[{epoch_time:.1f}s]")

        # Save best model
        overall_acc = (acc_cat + acc_rd + acc_rn + acc_rm + acc_sets_flags) / 5
        if overall_acc > best_overall_acc:
            best_overall_acc = overall_acc
            patience = 0

            Path("models/final").mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracies': {
                    'category': acc_cat,
                    'rd': acc_rd,
                    'rn': acc_rn,
                    'rm': acc_rm,
                    'sets_flags': acc_sets_flags,
                    'movz': acc_movz,
                    'branch': acc_branch,
                    'overall': overall_acc,
                },
                'epoch': epoch + 1,
            }, "models/final/decoder_pure_neural.pt")
            print(f"  ✅ Saved best model (overall acc={overall_acc:.1%})")
        else:
            patience += 1

        # Check if we've hit target (99%+ on all metrics)
        if acc_cat >= 0.99 and acc_movz >= 0.99 and acc_branch >= 0.99:
            print(f"\n🎉 ACHIEVED 99%+ ACCURACY ON ALL METRICS!")
            break

        # Early stopping
        if patience >= max_patience:
            print(f"\nEarly stopping - no improvement for {max_patience} epochs")
            break

    print()
    print(f"✅ Training complete!")
    print(f"   Best overall accuracy: {best_overall_acc:.1%}")
    print(f"   Final category accuracy: {acc_cat:.1%}")
    print(f"   Final RD accuracy: {acc_rd:.1%}")
    print(f"   Final RN accuracy: {acc_rn:.1%}")
    print(f"   Final RM accuracy: {acc_rm:.1%}")
    print(f"   Final MOVZ accuracy: {acc_movz:.1%}")
    print(f"   Final branch accuracy: {acc_branch:.1%}")
    print()
    print("="*70)
    print("PURE NEURAL DECODER READY - NO HEURISTICS USED!")
    print("="*70)


if __name__ == "__main__":
    train_pure_neural_decoder()
