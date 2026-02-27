#!/usr/bin/env python3
"""
DIVISION via RECIPROCAL + MULTIPLICATION

Research shows division is the hardest arithmetic operation for neural networks.
Best approach (NMRU): Learn reciprocal, then multiply.

DIV(a, b) = MUL(a, RECIPROCAL(b))

We have a 100% accurate MUL model!
Now we just need to train a RECIPROCAL model.

For integer division:
- quotient = dividend // divisor
- We can reformulate: learn to output quotient bits directly given (dividend, divisor)
- OR learn reciprocal in fixed-point and multiply

This script tries the direct quotient approach with a specialized architecture
inspired by the NMRU research.
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


class NeuralReciprocalUnit(nn.Module):
    """
    Neural Reciprocal Unit (NRU) inspired by research.

    For integer division, we compute reciprocal in fixed-point format,
    then multiply by dividend and extract integer part.

    Key insight: 1/b ≈ (2^N / b) in fixed-point with N fractional bits
    Then: a/b = (a * (2^N / b)) >> N
    """
    def __init__(self, max_bits=64, d_model=384, nhead=16, num_layers=6):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # Encode divisor
        self.divisor_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        # Transformer to learn reciprocal pattern
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output: reciprocal in fixed-point (same bit width)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, divisor_bits):
        """
        Compute reciprocal representation.

        Args:
            divisor_bits: [batch, bits] - divisor in bit representation

        Returns:
            reciprocal_logits: [batch, bits] - reciprocal in fixed-point
        """
        batch = divisor_bits.shape[0]
        bits = divisor_bits.shape[1]

        # Embed divisor bits
        x = self.divisor_embed(divisor_bits.unsqueeze(-1))
        x = x + self.pos_embedding[:, :bits, :]

        # Transform
        x = self.transformer(x)

        # Output reciprocal bits
        output = self.output_head(x).squeeze(-1)

        return output


class DirectQuotientPredictor(nn.Module):
    """
    Directly predict quotient bits from dividend and divisor.

    Uses cross-attention between dividend and divisor, similar to
    how our MUL model uses outer product.

    Key insight for division:
    - quotient bit k is 1 if (remaining_dividend) >= (divisor << k)
    - This is a comparison operation that can be learned
    """
    def __init__(self, max_bits=64, d_model=384, nhead=16, num_layers=8):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # Embed dividend and divisor with position information
        self.dividend_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.divisor_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Position embeddings
        self.pos_dividend = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)
        self.pos_divisor = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        # Cross-attention layers (dividend attends to divisor)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
            for _ in range(num_layers // 2)
        ])

        # Self-attention layers
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True,
                activation='gelu'
            )
            for _ in range(num_layers // 2)
        ])

        # Layer norms for cross attention
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers // 2)
        ])

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        # Causal mask (quotient bits from MSB to LSB have dependencies)
        # Actually for division, we go from MSB to LSB
        causal = torch.triu(torch.ones(max_bits, max_bits), diagonal=1).bool()
        self.register_buffer('causal_mask', causal)

    def forward(self, dividend_bits, divisor_bits):
        batch = dividend_bits.shape[0]
        bits = dividend_bits.shape[1]

        # Embed
        div_emb = self.dividend_embed(dividend_bits.unsqueeze(-1))
        dis_emb = self.divisor_embed(divisor_bits.unsqueeze(-1))

        # Add positions
        div_emb = div_emb + self.pos_dividend[:, :bits, :]
        dis_emb = dis_emb + self.pos_divisor[:, :bits, :]

        # Interleaved cross-attention and self-attention
        x = div_emb
        for cross_attn, cross_norm, self_attn in zip(
            self.cross_attn_layers, self.cross_norms, self.self_attn_layers
        ):
            # Cross-attention: x attends to divisor
            cross_out, _ = cross_attn(x, dis_emb, dis_emb)
            x = cross_norm(x + cross_out)

            # Self-attention
            x = self_attn(x)

        # Output quotient bits
        output = self.output_head(x).squeeze(-1)

        return output


def generate_div_data(batch_size, bits, device):
    """Generate division training data with careful range control."""
    # For N-bit output quotient:
    # - Divisor: random number from 1 to 2^(bits//2) to ensure quotient fits
    # - Dividend: random N-bit number

    max_divisor = 2 ** (bits // 2)
    divisor = torch.randint(1, max(2, max_divisor), (batch_size,), device=device)

    max_dividend = 2 ** bits
    dividend = torch.randint(0, max_dividend, (batch_size,), device=device)

    # Compute quotient (integer division)
    quotient = dividend // divisor

    # Convert to bits
    dividend_bits = ((dividend.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    divisor_bits = ((divisor.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    quotient_bits = ((quotient.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()

    return dividend_bits, divisor_bits, quotient_bits


def generate_easy_div_data(batch_size, bits, device):
    """
    Generate easier division data for initial learning.

    Start with exact divisions (no remainder) to help model learn pattern.
    """
    max_divisor = 2 ** (bits // 4)  # Smaller divisors
    max_quotient = 2 ** (bits // 2)  # Ensure quotient fits

    divisor = torch.randint(1, max(2, max_divisor), (batch_size,), device=device)
    quotient = torch.randint(0, max(1, max_quotient), (batch_size,), device=device)

    # Compute dividend = quotient * divisor (exact division!)
    dividend = quotient * divisor

    # Clip to bits range
    dividend = dividend & ((1 << bits) - 1)
    quotient = dividend // divisor  # Recalculate in case of overflow

    # Convert to bits
    dividend_bits = ((dividend.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    divisor_bits = ((divisor.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    quotient_bits = ((quotient.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()

    return dividend_bits, divisor_bits, quotient_bits


def train_div_direct(model, device, save_dir, max_bits=64, batch_size=4096, model_name="div_direct"):
    print("\n" + "=" * 70)
    print(f"DIRECT QUOTIENT PREDICTION: {model.__class__.__name__}")
    print("=" * 70)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}")

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

    # Extended curriculum with MORE epochs for division
    curriculum = [
        (8, 5000), (12, 8000), (16, 12000), (20, 15000),
        (24, 18000), (28, 22000), (32, 25000),
        (40, 30000), (48, 35000), (56, 40000), (64, 50000)
    ]

    curriculum = [(b, e) for (b, e) in curriculum if b <= max_bits]

    print(f"Curriculum: {curriculum}")

    for bits, max_epochs in curriculum:
        print(f"\n{'=' * 70}")
        print(f"Training: {bits}-bit DIV")
        print(f"{'=' * 70}")

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
        best_acc = 0
        consecutive_100 = 0

        # Start with easy data (exact divisions), transition to hard
        easy_epochs = max_epochs // 4

        for epoch in range(max_epochs):
            model.train()

            # Use easier data initially, then transition
            if epoch < easy_epochs:
                dividend_bits, divisor_bits, target = generate_easy_div_data(batch_size, bits, device)
            else:
                dividend_bits, divisor_bits, target = generate_div_data(batch_size, bits, device)

            if bits < max_bits:
                dividend_bits = F.pad(dividend_bits, (0, max_bits - bits))
                divisor_bits = F.pad(divisor_bits, (0, max_bits - bits))
                target = F.pad(target, (0, max_bits - bits))

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

            if epoch % 100 == 0 or correct > best_acc:
                data_type = "easy" if epoch < easy_epochs else "full"
                print(f"    Epoch {epoch:4d}: loss={loss.item():.4f}, acc={correct:.2f}% ({data_type})")

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
    print("DIVISION TRAINING COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--bits', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--model', choices=['nru', 'direct'], default='direct')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.model == 'nru':
        model = NeuralReciprocalUnit(max_bits=args.bits, d_model=384, nhead=16, num_layers=6)
        model_name = "div_nru"
    else:
        model = DirectQuotientPredictor(max_bits=args.bits, d_model=384, nhead=16, num_layers=8)
        model_name = "div_direct"

    model = model.to(args.device)

    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_div_direct(model, args.device, args.save_dir,
                     max_bits=args.bits, batch_size=args.batch_size, model_name=model_name)


if __name__ == '__main__':
    main()
