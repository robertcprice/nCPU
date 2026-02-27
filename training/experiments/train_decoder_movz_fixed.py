#!/usr/bin/env python3
"""
TRAIN DECODER WITH MOVZ/MOVK EMPHASIS
========================================

This script trains the ARM64 decoder with heavy emphasis on MOVZ/MOVK
instructions to fix the misclassification bug.

Target: MOVZ/MOVK accuracy > 99%
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
from run_neural_rtos_v2 import OpCategory, UniversalARM64Decoder

# Map category names to indices
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

IDX_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_IDX.items()}

NUM_CATEGORIES = len(OpCategory)


def encode_movz_movk(op: str, rd: int, imm16: int, hw: int = 0, sf: int = 1) -> tuple:
    """Encode MOVZ/MOVN/MOVK instructions."""
    # MOVZ: sf|10|100101|hw|imm16|Rd
    # MOVN: sf|00|100101|hw|imm16|Rd
    # MOVK: sf|11|100101|hw|imm16|Rd
    if op == 'MOVZ':
        base = 0x52800000
    elif op == 'MOVN':
        base = 0x12800000
    elif op == 'MOVK':
        base = 0x72800000
    else:
        base = 0x52800000

    if sf:
        base |= (1 << 31)

    inst = base | ((hw & 0x3) << 21) | ((imm16 & 0xFFFF) << 5) | (rd & 0x1F)

    # Category 12 = MOVE
    return inst, {
        'category': CATEGORY_TO_IDX['MOVE'],
        'rd': rd,
        'rn': 31,  # Not used
        'rm': 31,  # Not used
        'op': 0,   # Not used
    }


def encode_add_sub(rd: int, rn: int, imm: int = 0, is_add: bool = True, is_imm: bool = True, sf: int = 1) -> tuple:
    """Encode ADD/SUB instructions."""
    base = 0x11000000
    if not is_add:
        base |= (1 << 30)  # SUB bit
    if sf:
        base |= (1 << 31)

    if is_imm:
        inst = base | ((imm & 0xFFF) << 10) | ((rn & 0x1F) << 5) | (rd & 0x1F)
    else:
        inst = base | ((rn & 0x1F) << 16) | ((31 & 0x1F) << 10) | ((rn & 0x1F) << 5) | (rd & 0x1F)

    cat = CATEGORY_TO_IDX['ADD'] if is_add else CATEGORY_TO_IDX['SUB']
    return inst, {'category': cat, 'rd': rd, 'rn': rn, 'rm': 31, 'op': 0}


def encode_load_store(rt: int, rn: int, imm: int = 0, is_load: bool = True, is_store: bool = False, sf: int = 1) -> tuple:
    """Encode LDR/STR instructions."""
    if is_store:
        base = 0xF9000000 if sf else 0xB9000000
    else:
        base = 0xF9400000 if sf else 0xB9400000

    scaled_imm = (imm >> (3 if sf else 2)) & 0xFFF
    inst = base | (scaled_imm << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F)

    cat = CATEGORY_TO_IDX['LOAD'] if is_load else CATEGORY_TO_IDX['STORE']
    return inst, {'category': cat, 'rd': rt, 'rn': rn, 'rm': 31, 'op': 0}


def encode_branch(imm: int = 0, is_call: bool = False) -> tuple:
    """Encode branch instructions."""
    if is_call:
        inst = 0x94000000 | (imm & 0x3FFFFFF)
    else:
        inst = 0x14000000 | (imm & 0x3FFFFFF)

    return inst, {'category': CATEGORY_TO_IDX['BRANCH'], 'rd': 31, 'rn': 31, 'rm': 31, 'op': 0}


def encode_compare(rn: int, imm: int = 0, is_imm: bool = True) -> tuple:
    """Encode CMP instructions."""
    return encode_add_sub(31, rn, imm, is_add=True, is_imm=is_imm)


def generate_movz_heavy_batch(batch_size: int, movz_ratio: float = 0.4) -> list:
    """Generate a training batch with MOVZ emphasis."""
    batch = []

    for _ in range(batch_size):
        # Decide if this sample should be MOVZ/MOVK
        if random.random() < movz_ratio:
            # MOVZ/MOVK instruction
            op = random.choice(['MOVZ', 'MOVN', 'MOVK'])
            rd = random.randint(0, 30)
            imm16 = random.randint(0, 0xFFFF)
            hw = random.randint(0, 3)
            sf = random.choice([0, 1])

            # For 64-bit, allow all 4 hw values
            # For 32-bit, only hw 0 and 1 are valid
            if sf == 0:
                hw = hw & 1

            inst, labels = encode_movz_movk(op, rd, imm16, hw, sf)
            batch.append((inst, labels))
        else:
            # Other instruction (ADD, SUB, LDR, STR, B, CMP)
            inst_type = random.choice(['ADD', 'SUB', 'LDR', 'STR', 'B', 'CMP'])
            rd = random.randint(0, 30)
            rn = random.randint(0, 30)
            imm = random.randint(0, 0xFFF)

            if inst_type == 'ADD':
                inst, labels = encode_add_sub(rd, rn, imm, is_add=True)
            elif inst_type == 'SUB':
                inst, labels = encode_add_sub(rd, rn, imm, is_add=False)
            elif inst_type == 'LDR':
                inst, labels = encode_load_store(rd, rn, imm, is_load=True)
            elif inst_type == 'STR':
                inst, labels = encode_load_store(rd, rn, imm, is_load=False, is_store=True)
            elif inst_type == 'B':
                inst, labels = encode_branch(imm=random.randint(0, 0xFFFFF))
            elif inst_type == 'CMP':
                inst, labels = encode_compare(rn, imm)

            batch.append((inst, labels))

    return batch


def train_decoder_with_movz_emphasis():
    """Train decoder with heavy MOVZ emphasis."""

    print("="*70)
    print(" TRAINING DECODER WITH MOVZ/MOVK EMPHASIS")
    print("="*70)
    print()

    # Create decoder
    model = UniversalARM64Decoder(d_model=256).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    print(f"Target: MOVZ accuracy > 99%")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Training config
    batch_size = 512
    samples_per_epoch = 100000
    max_epochs = 100
    movz_ratio = 0.5  # 50% MOVZ/MOVK!

    best_movz_acc = 0
    patience = 0
    max_patience = 15

    for epoch in range(max_epochs):
        model.train()
        epoch_start = time.time()

        total_loss = 0
        cat_correct = 0
        rd_correct = 0
        rn_correct = 0
        movz_correct = 0
        total = 0
        movz_total = 0

        num_batches = samples_per_epoch // batch_size

        for batch_idx in range(num_batches):
            # Generate MOVZ-heavy batch
            batch_data = generate_movz_heavy_batch(batch_size, movz_ratio=movz_ratio)

            # Convert to tensors
            insts = torch.tensor([b[0] for b in batch_data], dtype=torch.long, device=device)
            cats = torch.tensor([b[1]['category'] for b in batch_data], dtype=torch.long, device=device)
            rds = torch.tensor([b[1]['rd'] for b in batch_data], dtype=torch.long, device=device)
            rns = torch.tensor([b[1]['rn'] for b in batch_data], dtype=torch.long, device=device)

            # Convert instructions to bits (as float for MPS compatibility)
            bits = torch.zeros(batch_size, 32, dtype=torch.float32, device=device)
            for i in range(32):
                bits[:, i] = ((insts >> i) & 1).float()

            # Forward pass
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(bits)

                # Compute losses for each output
                loss_rd = F.cross_entropy(outputs['rd'], rds)
                loss_rn = F.cross_entropy(outputs['rn'], rns)
                loss_cat = F.cross_entropy(outputs['category'], cats)

                # Total loss - weight category heavily for MOVZ accuracy
                loss = loss_cat * 3 + loss_rd + loss_rn

            # Backward
            optimizer.zero_grad()
            loss_cat.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss_cat.item()

            # Compute accuracy
            pred_cats = outputs['category'].argmax(dim=1)
            cat_correct += (pred_cats == cats).sum().item()
            rd_correct += (outputs['rd'].argmax(dim=1) == rds).sum().item()
            rn_correct += (outputs['rn'].argmax(dim=1) == rns).sum().item()

            # MOVZ-specific accuracy
            movz_mask = (cats == CATEGORY_TO_IDX['MOVE'])
            if movz_mask.any():
                movz_total += movz_mask.sum().item()
                movz_correct += ((pred_cats == cats) & movz_mask).sum().item()

            total += batch_size

        # Calculate accuracies
        acc_cat = cat_correct / total
        acc_rd = rd_correct / total
        acc_rn = rn_correct / total
        acc_movz = movz_correct / movz_total if movz_total > 0 else 0

        scheduler.step()

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1:3d}: Loss={total_loss/num_batches:.4f} "
              f"Cat={acc_cat:.1%} MOVZ={acc_movz:.1%} ({movz_correct}/{movz_total}) "
              f"[{epoch_time:.1f}s]")

        # Save best model
        if acc_movz > best_movz_acc:
            best_movz_acc = acc_movz
            patience = 0

            Path("models/final").mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': {
                    'category': acc_cat,
                    'movz': acc_movz,
                },
                'epoch': epoch + 1,
            }, "models/final/decoder_movz_fixed.pt")
            print(f"  ✅ Saved best model (MOVZ acc={acc_movz:.1%})")
        else:
            patience += 1

        # Check if we've hit target
        if acc_movz >= 0.99:
            print(f"\n🎉 ACHIEVED 99%+ MOVZ ACCURACY!")
            break

        # Early stopping
        if patience >= max_patience:
            print(f"\nEarly stopping - no improvement for {max_patience} epochs")
            break

    print()
    print(f"✅ Training complete!")
    print(f"   Best MOVZ accuracy: {best_movz_acc:.1%}")
    print(f"   Final category accuracy: {acc_cat:.1%}")
    print()
    print("="*70)


if __name__ == "__main__":
    import sys
    train_decoder_with_movz_emphasis()
