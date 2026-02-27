#!/usr/bin/env python3
"""
DIV Training with CosineAnnealingWarmRestarts (CAWR) - Conservative LR.

Key insight: Hard cases (remaining 40%) need MULTIPLE LR cycles to learn.
CAWR provides automatic periodic LR boosts without manual intervention.

IMPORTANT: Using max_lr=2e-4 (not 4e-4) to prevent catastrophic forgetting.
4e-4 restart caused crash from 64% → 0.35%.

Schedule: T_0=1500, T_mult=2
- Cycle 1: epochs 0-1500 (LR: 2e-4 → 5e-6 → 2e-4)
- Cycle 2: epochs 1500-4500 (LR: 2e-4 → 5e-6 → 2e-4)
- Cycle 3: epochs 4500-10500 (LR: 2e-4 → 5e-6 → 2e-4)
- etc.
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


class DirectParallelDivNet(nn.Module):
    """Same architecture as V2."""
    def __init__(self, max_bits=64, d_model=384, nhead=16, num_layers=8):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.pos_embed = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, dividend_bits, divisor_bits):
        batch = dividend_bits.shape[0]
        bits = dividend_bits.shape[1]

        x = torch.stack([dividend_bits, divisor_bits], dim=-1)
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :bits, :]
        x = self.transformer(x)
        output = self.output_head(x).squeeze(-1)

        return output


def generate_div_data(batch_size, bits, device):
    """Generate standard division data."""
    max_divisor = 2 ** (bits // 2)
    divisor = torch.randint(1, max(2, max_divisor), (batch_size,), device=device)

    max_dividend = 2 ** bits
    dividend = torch.randint(0, max_dividend, (batch_size,), device=device)

    quotient = dividend // divisor

    dividend_bits = ((dividend.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    divisor_bits = ((divisor.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    quotient_bits = ((quotient.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()

    return dividend_bits, divisor_bits, quotient_bits


def train_cawr(model, device, save_dir, start_bits=20, max_bits=64,
               batch_size=4096, accum_steps=4):
    """
    Training with CosineAnnealingWarmRestarts for multiple LR cycles.

    Key advantage: Multiple chances to learn hard cases.
    """
    print("\n" + "=" * 70)
    print(f"DIV TRAINING WITH COSINE ANNEALING WARM RESTARTS")
    print("=" * 70)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation steps: {accum_steps}")
    print(f"Effective batch size: {batch_size * accum_steps}")

    # Lower LR to prevent catastrophic forgetting on restarts
    # 4e-4 caused crash from 64% → 0.35%, using 2e-4 instead
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

    # Longer epochs for CAWR - needs multiple cycles
    full_curriculum = [
        (20, 25000), (24, 30000), (28, 35000), (32, 40000),
        (40, 50000), (48, 60000), (56, 70000), (64, 100000)
    ]

    curriculum = [(b, e) for (b, e) in full_curriculum if b >= start_bits and b <= max_bits]

    print(f"Curriculum: {curriculum}")

    for bits, max_epochs in curriculum:
        print(f"\n{'=' * 70}")
        print(f"Training: {bits}-bit DIV (CAWR)")
        print(f"{'=' * 70}")

        # CosineAnnealingWarmRestarts - multiple cycles!
        # T_0=1500: first cycle length
        # T_mult=2: each subsequent cycle is 2x longer
        # eta_min=5e-6: floor LR (proportional to max_lr=2e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=1500,
            T_mult=2,
            eta_min=5e-6
        )

        best_acc = 0
        consecutive_100 = 0
        cycle_num = 0
        prev_lr = 2e-4  # Track LR to detect actual restarts

        for epoch in range(max_epochs):
            model.train()

            # Gradient accumulation
            optimizer.zero_grad()
            total_loss = 0
            total_correct = 0

            for accum_step in range(accum_steps):
                dividend_bits, divisor_bits, target = generate_div_data(
                    batch_size, bits, device
                )

                with autocast('cuda', dtype=torch.bfloat16):
                    output = model(dividend_bits, divisor_bits)
                    loss = criterion(output[:, :bits], target[:, :bits])
                    loss = loss / accum_steps

                scaler.scale(loss).backward()
                total_loss += loss.item() * accum_steps

                with torch.no_grad():
                    pred = (torch.sigmoid(output[:, :bits]) > 0.5).float()
                    total_correct += (pred == target[:, :bits]).all(dim=1).float().sum().item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            avg_loss = total_loss / accum_steps
            correct = total_correct / (batch_size * accum_steps) * 100
            lr = optimizer.param_groups[0]['lr']

            # Detect warm restart (LR jumps up significantly from previous)
            if lr > prev_lr * 1.5 and epoch > 100:
                cycle_num += 1
                print(f"    🔄 WARM RESTART #{cycle_num} at epoch {epoch} (LR: {prev_lr:.2e} → {lr:.2e})")
            prev_lr = lr

            if epoch % 100 == 0 or correct > best_acc:
                print(f"    Epoch {epoch:5d}: loss={avg_loss:.4f}, acc={correct:.2f}%, lr={lr:.2e}")
                sys.stdout.flush()

            if correct > best_acc:
                best_acc = correct
                torch.save(model.state_dict(), f"{save_dir}/DIV_{bits}bit_cawr_ckpt.pt")

            if correct >= 100.0:
                consecutive_100 += 1
                if consecutive_100 >= 3:
                    print(f"    ✅ {bits}-bit: 100% x3 - ADVANCING!")
                    torch.save(model.state_dict(), f"{save_dir}/DIV_{bits}bit_cawr_100pct.pt")
                    break
            else:
                consecutive_100 = 0

        print(f"    Level complete: best={best_acc:.2f}% after {cycle_num} warm restarts")

    print("\n" + "=" * 70)
    print("CAWR TRAINING COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--bits', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--accum-steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--start-bits', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = DirectParallelDivNet(max_bits=args.bits, d_model=384, nhead=16, num_layers=8)

    print(f"Loading checkpoint: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    print("Checkpoint loaded successfully!")

    model = model.to(args.device)

    print(f"Model: {model.__class__.__name__}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_cawr(model, args.device, args.save_dir,
               start_bits=args.start_bits, max_bits=args.bits,
               batch_size=args.batch_size, accum_steps=args.accum_steps)


if __name__ == '__main__':
    main()
