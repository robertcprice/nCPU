#!/usr/bin/env python3
"""
Resume DIV v2 from 16-bit checkpoint with STANDARD data (no easy mode).

Hypothesis: v2 failed at 20-bit because "easy" data doesn't transfer well,
not because of model capacity.
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
    """Same architecture as v2 that achieved 16-bit 100%."""
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


def generate_standard_div_data(batch_size, bits, device):
    """
    Generate STANDARD division data - NO easy mode.
    This is the key change from the original training.
    """
    max_divisor = 2 ** (bits // 2)
    divisor = torch.randint(1, max(2, max_divisor), (batch_size,), device=device)

    max_dividend = 2 ** bits
    dividend = torch.randint(0, max_dividend, (batch_size,), device=device)

    quotient = dividend // divisor

    dividend_bits = ((dividend.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    divisor_bits = ((divisor.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    quotient_bits = ((quotient.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()

    return dividend_bits, divisor_bits, quotient_bits


def train_from_checkpoint(model, device, save_dir, start_bits=20, max_bits=64, batch_size=4096):
    print("\n" + "=" * 70)
    print(f"RESUME DIV v2 FROM {start_bits}-BIT WITH STANDARD DATA")
    print("=" * 70)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}")
    print(f"Data mode: STANDARD (no easy mode)")

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

    # Continue curriculum from start_bits
    full_curriculum = [
        (20, 20000), (24, 25000), (28, 30000), (32, 35000),
        (40, 45000), (48, 55000), (56, 65000), (64, 80000)
    ]

    curriculum = [(b, e) for (b, e) in full_curriculum if b >= start_bits and b <= max_bits]

    print(f"Curriculum: {curriculum}")

    for bits, max_epochs in curriculum:
        print(f"\n{'=' * 70}")
        print(f"Training: {bits}-bit DIV (STANDARD DATA)")
        print(f"{'=' * 70}")

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=2000, T_mult=2, eta_min=1e-6
        )
        best_acc = 0
        consecutive_100 = 0

        for epoch in range(max_epochs):
            model.train()

            # ALWAYS use standard data - no easy mode!
            dividend_bits, divisor_bits, target = generate_standard_div_data(
                batch_size, bits, device
            )

            optimizer.zero_grad()

            with autocast('cuda', dtype=torch.bfloat16):
                output = model(dividend_bits, divisor_bits)
                loss = criterion(output[:, :bits], target[:, :bits])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                pred = (torch.sigmoid(output[:, :bits]) > 0.5).float()
                correct = (pred == target[:, :bits]).all(dim=1).float().mean().item() * 100

            if epoch % 200 == 0 or correct > best_acc:
                print(f"    Epoch {epoch:5d}: loss={loss.item():.4f}, acc={correct:.2f}%")
                sys.stdout.flush()

            if correct > best_acc:
                best_acc = correct
                torch.save(model.state_dict(), f"{save_dir}/DIV_{bits}bit_v2_resume_ckpt.pt")

            if correct >= 100.0:
                consecutive_100 += 1
                if consecutive_100 >= 3:
                    print(f"    ✅ {bits}-bit: 100% x3 - ADVANCING!")
                    torch.save(model.state_dict(), f"{save_dir}/DIV_{bits}bit_v2_resume_100pct.pt")
                    break
            else:
                consecutive_100 = 0

        print(f"    Level complete: best={best_acc:.2f}%")

    print("\n" + "=" * 70)
    print("RESUME TRAINING COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--bits', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to 16-bit checkpoint')
    parser.add_argument('--start-bits', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Create model with SAME architecture as v2
    model = DirectParallelDivNet(max_bits=args.bits, d_model=384, nhead=16, num_layers=8)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    print("Checkpoint loaded successfully!")

    model = model.to(args.device)

    print(f"Model: {model.__class__.__name__}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_from_checkpoint(model, args.device, args.save_dir,
                          start_bits=args.start_bits, max_bits=args.bits,
                          batch_size=args.batch_size)


if __name__ == '__main__':
    main()
