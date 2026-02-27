#!/usr/bin/env python3
"""
LARGE DirectParallelDivNet - More capacity for hard division.

Hypothesis: The 67% ceiling might be a capacity issue, not architecture.
Try: d_model=512, num_layers=12, nhead=16 (31M+ params)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import os
import sys
import argparse

try:
    sys.stdout.reconfigure(line_buffering=True)
except:
    pass


class LargeParallelDivNet(nn.Module):
    """Large capacity parallel division network."""
    def __init__(self, max_bits=64, d_model=512, nhead=16, num_layers=12):
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
    """Generate division data."""
    max_divisor = 2 ** (bits // 2)
    divisor = torch.randint(1, max(2, max_divisor), (batch_size,), device=device)

    max_dividend = 2 ** bits
    dividend = torch.randint(0, max_dividend, (batch_size,), device=device)

    quotient = dividend // divisor

    dividend_bits = ((dividend.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    divisor_bits = ((divisor.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    quotient_bits = ((quotient.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()

    return dividend_bits, divisor_bits, quotient_bits


def train_large(model, device, save_dir, start_bits=8, max_bits=64,
                batch_size=2048, accum_steps=8):
    """Train large model with curriculum."""
    print("\n" + "=" * 70)
    print("LARGE PARALLEL DIVISION TRAINING")
    print("=" * 70)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size} x {accum_steps} = {batch_size * accum_steps}")

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

    curriculum = [
        (8, 3000), (12, 5000), (16, 8000), (20, 15000),
        (24, 20000), (28, 25000), (32, 35000),
        (40, 45000), (48, 55000), (56, 65000), (64, 80000)
    ]

    curriculum = [(b, e) for (b, e) in curriculum if b >= start_bits and b <= max_bits]
    print(f"Curriculum: {curriculum}")

    for bits, max_epochs in curriculum:
        print(f"\n{'=' * 70}")
        print(f"Training: {bits}-bit DIV (LARGE)")
        print(f"{'=' * 70}")

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1500, T_mult=2, eta_min=5e-6
        )

        best_acc = 0
        consecutive_100 = 0

        for epoch in range(max_epochs):
            model.train()
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

            if epoch % 100 == 0 or correct > best_acc:
                lr = optimizer.param_groups[0]['lr']
                print(f"    Epoch {epoch:5d}: loss={avg_loss:.4f}, acc={correct:.2f}%, lr={lr:.2e}")
                sys.stdout.flush()

            if correct > best_acc:
                best_acc = correct
                torch.save(model.state_dict(), f"{save_dir}/DIV_{bits}bit_large_ckpt.pt")

            if correct >= 100.0:
                consecutive_100 += 1
                if consecutive_100 >= 3:
                    print(f"    ✅ {bits}-bit: 100% x3 - ADVANCING!")
                    torch.save(model.state_dict(), f"{save_dir}/DIV_{bits}bit_large_100pct.pt")
                    break
            else:
                consecutive_100 = 0

        print(f"    Level complete: best={best_acc:.2f}%")

    print("\n" + "=" * 70)
    print("LARGE TRAINING COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--bits', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--accum-steps', type=int, default=8)
    parser.add_argument('--start-bits', type=int, default=8)
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = LargeParallelDivNet(
        max_bits=args.bits,
        d_model=args.d_model,
        nhead=16,
        num_layers=args.num_layers
    )

    model = model.to(args.device)

    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"d_model={args.d_model}, layers={args.num_layers}")

    train_large(model, args.device, args.save_dir,
                start_bits=args.start_bits, max_bits=args.bits,
                batch_size=args.batch_size, accum_steps=args.accum_steps)


if __name__ == '__main__':
    main()
