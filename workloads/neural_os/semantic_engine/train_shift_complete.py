#!/usr/bin/env python3
"""
Complete Shift Operations Training

Trains ALL shift operations to 100% accuracy:
- LSL (Logical Shift Left) ✅ Already done
- LSR (Logical Shift Right) ✅ Already done
- ASR (Arithmetic Shift Right) - Sign bit preservation
- ROL (Rotate Left) - Circular shift
- ROR (Rotate Right) - Circular shift

Uses the same architecture that achieved 100% on LSL/LSR.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import os
import sys
import argparse

try:
    sys.stdout.reconfigure(line_buffering=True)
except:
    pass


class ShiftNet(nn.Module):
    """
    Universal shift network - works for all shift types.
    Same architecture that achieved 100% on LSL/LSR.
    """
    def __init__(self, bits=64, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.bits = bits

        # Input: value bits + shift amount (6 bits for 0-63)
        self.input_proj = nn.Linear(bits + 6, d_model)

        # Position embedding for output bits
        self.pos_embed = nn.Parameter(torch.randn(1, bits, d_model) * 0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output = nn.Linear(d_model, 1)

    def forward(self, value_bits, shift_bits):
        batch = value_bits.shape[0]

        # Broadcast input to each output position
        combined = torch.cat([value_bits, shift_bits], dim=-1)  # [batch, bits+6]
        x = combined.unsqueeze(1).expand(-1, self.bits, -1)  # [batch, bits, bits+6]

        x = self.input_proj(x)  # [batch, bits, d_model]
        x = x + self.pos_embed

        x = self.transformer(x)
        out = torch.sigmoid(self.output(x).squeeze(-1))
        return out


def generate_shift_data(batch_size, bits, device, op='LSL'):
    """
    Generate shift operation data.

    Operations:
    - LSL: output[i] = input[i-shift] if i >= shift else 0
    - LSR: output[i] = input[i+shift] if i+shift < bits else 0
    - ASR: output[i] = input[i+shift] if i+shift < bits else sign_bit
    - ROL: output[i] = input[(i-shift) % bits] (circular left)
    - ROR: output[i] = input[(i+shift) % bits] (circular right)
    """
    # Random input bits
    a_bits = torch.randint(0, 2, (batch_size, bits), device=device).float()

    # Random shift amounts (0 to bits-1)
    shifts = torch.randint(0, bits, (batch_size,), device=device)
    shift_bits = ((shifts.unsqueeze(1) >> torch.arange(6, device=device)) & 1).float()

    # Position indices
    positions = torch.arange(bits, device=device).unsqueeze(0).expand(batch_size, -1)

    if op == 'LSL':
        # Logical Shift Left: zeros fill from right
        src_positions = positions - shifts.unsqueeze(1)
        valid_mask = src_positions >= 0
        src_positions = src_positions.clamp(0, bits-1)
        gathered = torch.gather(a_bits, 1, src_positions)
        result = gathered * valid_mask.float()

    elif op == 'LSR':
        # Logical Shift Right: zeros fill from left
        src_positions = positions + shifts.unsqueeze(1)
        valid_mask = src_positions < bits
        src_positions = src_positions.clamp(0, bits-1)
        gathered = torch.gather(a_bits, 1, src_positions)
        result = gathered * valid_mask.float()

    elif op == 'ASR':
        # Arithmetic Shift Right: sign bit fills from left
        sign_bit = a_bits[:, bits-1:bits]  # MSB is sign bit
        src_positions = positions + shifts.unsqueeze(1)
        valid_mask = src_positions < bits
        src_positions = src_positions.clamp(0, bits-1)
        gathered = torch.gather(a_bits, 1, src_positions)
        # Where invalid, use sign bit
        result = gathered * valid_mask.float() + sign_bit.expand(-1, bits) * (~valid_mask).float()

    elif op == 'ROL':
        # Rotate Left: circular shift left
        src_positions = (positions - shifts.unsqueeze(1)) % bits
        result = torch.gather(a_bits, 1, src_positions)

    elif op == 'ROR':
        # Rotate Right: circular shift right
        src_positions = (positions + shifts.unsqueeze(1)) % bits
        result = torch.gather(a_bits, 1, src_positions)

    else:
        raise ValueError(f"Unknown operation: {op}")

    return a_bits, shift_bits, result


def train_shift(device, save_dir, bits=64, op='LSL', epochs=1000, batch_size=4096):
    """Train a shift operation to 100%."""
    print(f"\n{'='*70}")
    print(f"TRAINING {op} ({bits}-bit)")
    print("="*70)

    model = ShiftNet(bits=bits, d_model=256, nhead=8, num_layers=4).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)
    criterion = nn.BCELoss()

    best_acc = 0
    consecutive_100 = 0

    for epoch in range(epochs):
        model.train()

        a_bits, shift_bits, target = generate_shift_data(batch_size, bits, device, op)

        optimizer.zero_grad()
        output = model(a_bits, shift_bits)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        pred = (output > 0.5).float()
        correct = (pred == target).all(dim=1).float().mean().item() * 100

        if epoch % 50 == 0 or correct > best_acc:
            print(f"  Epoch {epoch:4d}: loss={loss.item():.4f}, acc={correct:.2f}%")

        if correct > best_acc:
            best_acc = correct
            torch.save(model.state_dict(), f"{save_dir}/{op}_{bits}bit_checkpoint.pt")

            if correct >= 100.0:
                consecutive_100 += 1
                if consecutive_100 >= 3:
                    print(f"  ✅ {op} {bits}-bit: 100% x3 - DONE!")
                    torch.save(model.state_dict(), f"{save_dir}/{op}_{bits}bit_100pct.pt")
                    return True
            else:
                consecutive_100 = 0

    print(f"  Final: {best_acc:.2f}%")
    return best_acc >= 100.0


def train_all_shifts(device, save_dir, bits=64):
    """Train all shift operations."""
    operations = ['LSL', 'LSR', 'ASR', 'ROL', 'ROR']
    results = {}

    for op in operations:
        # Check if already exists
        existing_path = f"{save_dir}/{op}_{bits}bit_100pct.pt"
        if os.path.exists(existing_path):
            print(f"\n✅ {op} already trained: {existing_path}")
            results[op] = True
            continue

        # Also check models_100pct directory
        alt_path = f"../models_100pct/{op}_{bits}bit_100pct.pt"
        if os.path.exists(alt_path):
            print(f"\n✅ {op} already exists: {alt_path}")
            results[op] = True
            continue

        success = train_shift(device, save_dir, bits, op)
        results[op] = success

    # Summary
    print(f"\n{'='*70}")
    print("SHIFT OPERATIONS SUMMARY")
    print("="*70)
    for op, success in results.items():
        status = "✅ 100%" if success else "❌ Needs work"
        print(f"  {op}: {status}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--bits', type=int, default=64)
    parser.add_argument('--ops', nargs='+', default=['ASR', 'ROL', 'ROR'],
                        help='Operations to train (default: ASR ROL ROR)')
    parser.add_argument('--all', action='store_true', help='Train all shift operations')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.all:
        train_all_shifts(args.device, args.save_dir, args.bits)
    else:
        for op in args.ops:
            train_shift(args.device, args.save_dir, args.bits, op)


if __name__ == '__main__':
    main()
