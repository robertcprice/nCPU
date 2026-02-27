#!/usr/bin/env python3
"""
PARALLEL DIV v3 - Enhanced capacity for larger bit widths

Key changes from v2:
1. Larger model (d_model=512, num_layers=10)
2. Use "standard" data from the start (not "easy")
3. Lower learning rate for stability
4. Better weight initialization
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


class LargeParallelDivNet(nn.Module):
    """
    Large capacity parallel DIV for 64-bit operations.

    Key: More capacity to handle the complexity of 64-bit division.
    """
    def __init__(self, max_bits=64, d_model=512, nhead=16, num_layers=10):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # Input: 2 bits per position (dividend bit, divisor bit)
        self.input_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        # Main transformer - NO CAUSAL MASK
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, dividend_bits, divisor_bits):
        batch = dividend_bits.shape[0]
        bits = dividend_bits.shape[1]

        # Stack inputs
        x = torch.stack([dividend_bits, divisor_bits], dim=-1)  # [batch, bits, 2]

        # Project
        x = self.input_proj(x)  # [batch, bits, d_model]
        x = x + self.pos_embed[:, :bits, :]

        # Transform (NO causal mask!)
        x = self.transformer(x)

        # Output
        output = self.output_head(x).squeeze(-1)

        return output


def generate_div_data(batch_size, bits, device):
    """
    Generate standard division data (not easy mode).

    Random dividends and divisors where divisor is constrained
    to ensure quotient fits in bit range.
    """
    # Constrain divisor to ensure quotient fits
    max_divisor = 2 ** (bits // 2)
    divisor = torch.randint(1, max(2, max_divisor), (batch_size,), device=device)

    max_dividend = 2 ** bits
    dividend = torch.randint(0, max_dividend, (batch_size,), device=device)

    quotient = dividend // divisor

    # Convert to bits
    dividend_bits = ((dividend.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    divisor_bits = ((divisor.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    quotient_bits = ((quotient.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()

    return dividend_bits, divisor_bits, quotient_bits


def train_div(model, device, save_dir, max_bits=64, batch_size=4096, model_name="parallel_div_v3"):
    print("\n" + "=" * 70)
    print(f"PARALLEL DIV v3 TRAINING: {model.__class__.__name__}")
    print("=" * 70)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}")

    # Lower learning rate for stability
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

    # Progressive curriculum with more epochs
    curriculum = [
        (8, 5000), (12, 8000), (16, 12000), (20, 15000),
        (24, 20000), (28, 25000), (32, 30000),
        (40, 40000), (48, 50000), (56, 60000), (64, 80000)
    ]

    curriculum = [(b, e) for (b, e) in curriculum if b <= max_bits]

    print(f"Curriculum: {curriculum}")

    for bits, max_epochs in curriculum:
        print(f"\n{'=' * 70}")
        print(f"Training: {bits}-bit DIV")
        print(f"{'=' * 70}")

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=2000, T_mult=2, eta_min=1e-6
        )
        best_acc = 0
        consecutive_100 = 0

        for epoch in range(max_epochs):
            model.train()

            # Always use standard data
            dividend_bits, divisor_bits, target = generate_div_data(
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
                torch.save(model.state_dict(), f"{save_dir}/DIV_{bits}bit_{model_name}_ckpt.pt")

            if correct >= 100.0:
                consecutive_100 += 1
                if consecutive_100 >= 3:
                    print(f"    ✅ {bits}-bit: 100% x3 - ADVANCING!")
                    torch.save(model.state_dict(), f"{save_dir}/DIV_{bits}bit_{model_name}_100pct.pt")
                    break
            else:
                consecutive_100 = 0

        print(f"    Level complete: best={best_acc:.2f}%")

    print("\n" + "=" * 70)
    print("PARALLEL DIV v3 TRAINING COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--bits', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--start-bits', type=int, default=8, help='Start from this bit width')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Larger model for better capacity
    model = LargeParallelDivNet(max_bits=args.bits, d_model=512, nhead=16, num_layers=10)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=args.device))

    model = model.to(args.device)

    print(f"Model: {model.__class__.__name__}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_div(model, args.device, args.save_dir,
              max_bits=args.bits, batch_size=args.batch_size, model_name="parallel_v3")


if __name__ == '__main__':
    main()
