#!/usr/bin/env python3
"""
ACCELERATED DIV Training v2 - Conservative but faster.

Key changes:
1. Larger effective batch size via gradient accumulation (16384)
2. Conservative OneCycleLR (4e-4 max, not 1e-3 which caused instability)
3. 5% warmup for faster peak reach
4. Mixed precision throughout
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


def train_accelerated(model, device, save_dir, start_bits=20, max_bits=64,
                      batch_size=4096, accum_steps=4):
    """
    Accelerated training with gradient accumulation and OneCycleLR.

    Effective batch size = batch_size * accum_steps = 4096 * 4 = 16384
    """
    print("\n" + "=" * 70)
    print(f"ACCELERATED DIV TRAINING")
    print("=" * 70)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation steps: {accum_steps}")
    print(f"Effective batch size: {batch_size * accum_steps}")

    # More conservative LR - OneCycleLR at 1e-3 caused instability
    optimizer = optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

    # Continue curriculum from start_bits
    full_curriculum = [
        (20, 15000), (24, 20000), (28, 25000), (32, 30000),
        (40, 40000), (48, 50000), (56, 60000), (64, 80000)
    ]

    curriculum = [(b, e) for (b, e) in full_curriculum if b >= start_bits and b <= max_bits]

    print(f"Curriculum: {curriculum}")

    for bits, max_epochs in curriculum:
        print(f"\n{'=' * 70}")
        print(f"Training: {bits}-bit DIV (ACCELERATED)")
        print(f"{'=' * 70}")

        # OneCycleLR with conservative max_lr to prevent instability
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=4e-4,  # Conservative: 1e-3 caused instability
            total_steps=max_epochs,
            pct_start=0.05,  # 5% warmup (faster to reach peak)
            anneal_strategy='cos',
            div_factor=4,  # start_lr = max_lr / 4 = 1e-4
            final_div_factor=40  # end_lr = max_lr / 40 = 1e-5
        )

        best_acc = 0
        consecutive_100 = 0

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
                    loss = loss / accum_steps  # Normalize for accumulation

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
                lr = scheduler.get_last_lr()[0]
                print(f"    Epoch {epoch:5d}: loss={avg_loss:.4f}, acc={correct:.2f}%, lr={lr:.2e}")
                sys.stdout.flush()

            if correct > best_acc:
                best_acc = correct
                torch.save(model.state_dict(), f"{save_dir}/DIV_{bits}bit_accelerated_ckpt.pt")

            if correct >= 100.0:
                consecutive_100 += 1
                if consecutive_100 >= 3:
                    print(f"    ✅ {bits}-bit: 100% x3 - ADVANCING!")
                    torch.save(model.state_dict(), f"{save_dir}/DIV_{bits}bit_accelerated_100pct.pt")
                    break
            else:
                consecutive_100 = 0

        print(f"    Level complete: best={best_acc:.2f}%")

    print("\n" + "=" * 70)
    print("ACCELERATED TRAINING COMPLETE!")
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

    train_accelerated(model, args.device, args.save_dir,
                      start_bits=args.start_bits, max_bits=args.bits,
                      batch_size=args.batch_size, accum_steps=args.accum_steps)


if __name__ == '__main__':
    main()
