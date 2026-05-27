#!/usr/bin/env python3
"""Train the Neural Loop Detector for gpu_only vectorizer.

Generates synthetic ARM64 loop patterns with known types (countdown, count-up,
mem_fill, none) and trains the NeuralLoopDetector to classify them and identify
the counter register.

Usage:
    python ncpu/neural/train_loop_detector.py
"""

import sys
import struct
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.neural.cpu.extractors import NeuralLoopDetector


def _encode_add_reg(rd, rn, rm):
    return 0x8B000000 | (rm << 16) | (rn << 5) | rd

def _encode_sub_imm(rd, rn, imm):
    return 0xD1000000 | ((imm & 0xFFF) << 10) | (rn << 5) | rd

def _encode_subs_imm(rd, rn, imm):
    return 0xF1000000 | ((imm & 0xFFF) << 10) | (rn << 5) | rd

def _encode_add_imm(rd, rn, imm):
    return 0x91000000 | ((imm & 0xFFF) << 10) | (rn << 5) | rd

def _encode_movz(rd, imm):
    return 0xD2800000 | ((imm & 0xFFFF) << 5) | rd

def _encode_mul(rd, rn, rm):
    return 0x9B007C00 | (rm << 16) | (rn << 5) | rd

def _encode_strb(rt, rn, imm):
    return 0x39000000 | ((imm & 0xFFF) << 10) | (rn << 5) | rt

def _encode_and_reg(rd, rn, rm):
    return 0x8A000000 | (rm << 16) | (rn << 5) | rd

def _encode_orr_reg(rd, rn, rm):
    return 0xAA000000 | (rm << 16) | (rn << 5) | rd

def _encode_nop():
    return 0xD503201F  # NOP


def inst_to_bits(inst: int, device='cpu') -> torch.Tensor:
    """Convert 32-bit instruction to [32] float bit tensor."""
    return torch.tensor([(inst >> b) & 1 for b in range(32)], dtype=torch.float32, device=device)


def generate_training_data(n_samples: int = 20000, max_body: int = 8):
    """Generate synthetic loops with known types and counter registers."""
    torch.manual_seed(42)

    all_body_bits = []    # [N, max_body_len, 32]
    all_reg_values = []   # [N, 32]
    all_types = []        # [N] int (0=none, 1=count_up, 2=countdown, 3=mem_fill)
    all_counters = []     # [N] int (0-31)

    for _ in range(n_samples):
        loop_type = torch.randint(0, 4, (1,)).item()
        body_insts = []
        counter_reg = torch.randint(0, 8, (1,)).item()  # X0-X7 as counters
        acc_reg = (counter_reg + 1) % 8
        step_reg = (counter_reg + 2) % 8
        base_reg = (counter_reg + 3) % 8
        counter_val = torch.randint(10, 5000, (1,)).item()

        reg_values = torch.zeros(32, dtype=torch.float32)
        reg_values[counter_reg] = counter_val
        reg_values[step_reg] = torch.randint(1, 10, (1,)).item()

        if loop_type == 0:  # none — random non-loop instructions
            n_insts = torch.randint(2, max_body + 1, (1,)).item()
            for _ in range(n_insts):
                body_insts.append(_encode_nop())
            counter_reg = 0  # doesn't matter

        elif loop_type == 1:  # count_up: ADD acc, acc, step; ADD counter, counter, #1; SUBS X?, counter, limit
            body_insts.append(_encode_add_reg(acc_reg, acc_reg, step_reg))
            body_insts.append(_encode_add_imm(counter_reg, counter_reg, 1))
            body_insts.append(_encode_subs_imm(counter_reg, counter_reg, 1))
            reg_values[acc_reg] = 0

        elif loop_type == 2:  # countdown: ADD acc, acc, step; SUBS counter, counter, #1
            n_body = torch.randint(1, 4, (1,)).item()
            for _ in range(n_body):
                body_insts.append(_encode_add_reg(acc_reg, acc_reg, step_reg))
            body_insts.append(_encode_subs_imm(counter_reg, counter_reg, 1))
            reg_values[acc_reg] = 0

        elif loop_type == 3:  # mem_fill: STRB value, [base]; ADD base, base, #1; SUBS counter, counter, #1
            val_reg = (counter_reg + 4) % 8
            reg_values[val_reg] = torch.randint(0, 256, (1,)).item()
            reg_values[base_reg] = torch.randint(0x10000, 0x50000, (1,)).item()
            body_insts.append(_encode_strb(val_reg, base_reg, 0))
            body_insts.append(_encode_add_imm(base_reg, base_reg, 1))
            body_insts.append(_encode_subs_imm(counter_reg, counter_reg, 1))

        # Add some random extra instructions for variety
        if loop_type > 0 and torch.rand(1).item() > 0.5:
            extra = torch.randint(0, 3, (1,)).item()
            for _ in range(extra):
                r = torch.randint(8, 16, (1,)).item()
                body_insts.insert(-1, _encode_add_imm(r, r, 1))

        # Convert to body_bits [max_body_len, 32]
        body_bits = torch.zeros(32, 32, dtype=torch.float32)  # max_body_len=32 (model default)
        for i, inst in enumerate(body_insts[:32]):
            body_bits[i] = inst_to_bits(inst)

        all_body_bits.append(body_bits)
        all_reg_values.append(reg_values)
        all_types.append(loop_type)
        all_counters.append(counter_reg)

    return {
        'body_bits': torch.stack(all_body_bits),
        'reg_values': torch.stack(all_reg_values),
        'types': torch.tensor(all_types, dtype=torch.long),
        'counters': torch.tensor(all_counters, dtype=torch.long),
    }


def train():
    print("Generating training data...")
    data = generate_training_data(n_samples=10000)
    n = len(data['types'])
    n_train = int(n * 0.9)
    print(f"  {n} samples")
    for t in range(4):
        cnt = (data['types'] == t).sum().item()
        print(f"    type {t}: {cnt} ({100*cnt/n:.1f}%)")

    model = NeuralLoopDetector(max_body_len=32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    type_criterion = nn.CrossEntropyLoss()
    counter_criterion = nn.CrossEntropyLoss()
    batch_sz = 128
    best_type_acc = 0.0
    best_counter_acc = 0.0

    for epoch in range(100):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_sz):
            idx = perm[i:i+batch_sz]
            losses = []
            for j in idx:
                type_logits, counter_probs, _ = model(data['body_bits'][j], data['reg_values'][j])
                t_loss = type_criterion(type_logits.unsqueeze(0), data['types'][j:j+1])
                # Only train counter for non-None types
                if data['types'][j] > 0:
                    c_loss = counter_criterion(counter_probs.unsqueeze(0), data['counters'][j:j+1])
                    losses.append(t_loss + c_loss)
                else:
                    losses.append(t_loss)

            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Evaluate (subsample for speed)
        model.eval()
        type_correct = 0
        counter_correct = 0
        counter_total = 0
        total = 0
        eval_indices = list(range(n_train, min(n_train + 500, n)))
        with torch.no_grad():
            for j in eval_indices:
                type_logits, counter_probs, _ = model(data['body_bits'][j], data['reg_values'][j])
                pred_type = type_logits.argmax().item()
                gt_type = data['types'][j].item()
                if pred_type == gt_type:
                    type_correct += 1
                total += 1
                if gt_type > 0:
                    pred_counter = counter_probs.argmax().item()
                    gt_counter = data['counters'][j].item()
                    if pred_counter == gt_counter:
                        counter_correct += 1
                    counter_total += 1

        t_acc = type_correct / total if total > 0 else 0
        c_acc = counter_correct / counter_total if counter_total > 0 else 0
        if t_acc > best_type_acc:
            best_type_acc = t_acc
        if c_acc > best_counter_acc:
            best_counter_acc = c_acc

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={total_loss/n_batches:.4f} type_acc={100*t_acc:.1f}% counter_acc={100*c_acc:.1f}%")

        if t_acc >= 0.99 and c_acc >= 0.95:
            print(f"\n  Target accuracy reached at epoch {epoch+1}!")
            break

    save_path = PROJECT_ROOT / "ncpu" / "neural" / "loop_detector_fast.pt"
    torch.save(model.state_dict(), save_path)
    size_kb = save_path.stat().st_size / 1024
    print(f"\nSaved to {save_path} ({size_kb:.1f} KB)")
    print(f"Best type accuracy: {100*best_type_acc:.1f}%")
    print(f"Best counter accuracy: {100*best_counter_acc:.1f}%")


if __name__ == "__main__":
    train()
