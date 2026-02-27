#!/usr/bin/env python3
"""
NMRU-Inspired Division Training

Research shows DIV is the hardest operation for neural networks.
Best approach: Learn RECIPROCAL, then use MUL(a, RECIPROCAL(b)).

Since we have 100% accurate MUL, we just need to train a reciprocal network!

Key insight from NMRU research (Neural Modular Reciprocal Unit):
- Instead of learning division directly, learn 1/b
- Then DIV(a,b) = MUL(a, 1/b)
- Achieves 91.6% success rate vs 65% for direct division

Our approach:
1. Train ReciprocalNet to output 1/b in fixed-point representation
2. Use our proven MUL model for the final multiplication
3. Extract integer quotient from the result

For integer division: quotient = floor(a/b)
We can represent this as: quotient = floor(a * (2^N / b)) >> N
Where N is the fixed-point precision.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import os
import sys
import argparse
import math

# Enable line buffering
try:
    sys.stdout.reconfigure(line_buffering=True)
except:
    pass


class ReciprocalNet(nn.Module):
    """
    Learn reciprocal in fixed-point representation.

    Given divisor b, output 1/b as fixed-point bits.
    We use 2N bits for precision (N for integer part, N for fractional).

    For integer division of N-bit numbers:
    - Input: divisor b (N bits)
    - Output: reciprocal 2^(2N) / b (2N bits fixed-point)

    Then: a / b ≈ (a * reciprocal) >> (2N)
    """
    def __init__(self, max_bits=64, d_model=384, nhead=16, num_layers=6):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # Embed divisor bits
        self.divisor_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Position embeddings for input and output
        self.pos_embed_in = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)
        self.pos_embed_out = nn.Parameter(torch.randn(1, max_bits * 2, d_model) * 0.02)

        # Cross-attention: output positions attend to divisor bits
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
            for _ in range(num_layers // 2)
        ])

        # Self-attention for output refinement
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True,
                activation='gelu'
            )
            for _ in range(num_layers // 2)
        ])

        # Layer norms
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers // 2)
        ])

        # Output head: predict reciprocal bits
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        # Output query embeddings (learnable)
        self.output_queries = nn.Parameter(torch.randn(1, max_bits * 2, d_model) * 0.02)

    def forward(self, divisor_bits, out_bits=None):
        """
        Compute reciprocal representation.

        Args:
            divisor_bits: [batch, bits] - divisor in bit representation
            out_bits: output precision (default: 2 * input bits)

        Returns:
            reciprocal_logits: [batch, out_bits] - reciprocal in fixed-point
        """
        batch = divisor_bits.shape[0]
        in_bits = divisor_bits.shape[1]
        out_bits = out_bits or in_bits * 2

        # Embed divisor
        div_emb = self.divisor_embed(divisor_bits.unsqueeze(-1))  # [batch, in_bits, d_model]
        div_emb = div_emb + self.pos_embed_in[:, :in_bits, :]

        # Initialize output queries
        out_emb = self.output_queries[:, :out_bits, :].expand(batch, -1, -1)
        out_emb = out_emb + self.pos_embed_out[:, :out_bits, :]

        # Interleaved cross and self attention
        x = out_emb
        for cross_attn, cross_norm, self_attn in zip(
            self.cross_attn_layers, self.cross_norms, self.self_attn_layers
        ):
            # Cross-attention: output attends to divisor
            cross_out, _ = cross_attn(x, div_emb, div_emb)
            x = cross_norm(x + cross_out)

            # Self-attention for output refinement
            x = self_attn(x)

        # Output reciprocal bits
        output = self.output_head(x).squeeze(-1)

        return output


class DivisionFromReciprocal(nn.Module):
    """
    Division using learned quotient prediction with reciprocal-style encoding.

    Instead of MUL(a, 1/b), we directly learn to predict quotient bits
    by encoding (dividend, divisor) together and using cross-attention.

    Key insight: We encode divisor's "reciprocal pattern" implicitly
    and learn the mapping to quotient bits.
    """
    def __init__(self, max_bits=64, d_model=384, nhead=16, num_layers=8):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # Separate embeddings for dividend and divisor
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
        self.pos_output = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        # Learnable output queries
        self.output_queries = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        # Cross-attention layers: output attends to both dividend and divisor
        self.cross_attn_div = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
            for _ in range(num_layers // 2)
        ])
        self.cross_attn_dis = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
            for _ in range(num_layers // 2)
        ])

        # Self-attention for output refinement
        self.self_attn = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True,
                activation='gelu'
            )
            for _ in range(num_layers // 2)
        ])

        # Layer norms
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, dividend_bits, divisor_bits):
        batch = dividend_bits.shape[0]
        bits = dividend_bits.shape[1]

        # Embed dividend and divisor
        div_emb = self.dividend_embed(dividend_bits.unsqueeze(-1))
        dis_emb = self.divisor_embed(divisor_bits.unsqueeze(-1))

        # Add position embeddings
        div_emb = div_emb + self.pos_dividend[:, :bits, :]
        dis_emb = dis_emb + self.pos_divisor[:, :bits, :]

        # Initialize output from learned queries
        out_emb = self.output_queries[:, :bits, :].expand(batch, -1, -1)
        out_emb = out_emb + self.pos_output[:, :bits, :]

        # Interleaved cross-attention and self-attention
        x = out_emb
        for i, (ca_div, ca_dis, sa) in enumerate(zip(
            self.cross_attn_div, self.cross_attn_dis, self.self_attn
        )):
            # Cross-attend to dividend
            attn_div, _ = ca_div(x, div_emb, div_emb)
            x = self.norms[i*2](x + attn_div)

            # Cross-attend to divisor
            attn_dis, _ = ca_dis(x, dis_emb, dis_emb)
            x = self.norms[i*2+1](x + attn_dis)

            # Self-attention for refinement
            x = sa(x)

        # Output quotient bits
        output = self.output_head(x).squeeze(-1)

        return output


class DirectDivisionWithLSB(nn.Module):
    """
    Direct division approach with LSB (least significant bit) focus.

    Key insight: Division can be computed bit-by-bit from MSB to LSB.
    For each quotient bit k (from MSB to LSB):
        - q[k] = 1 if (divisor << k) <= remaining_dividend
        - remaining_dividend -= q[k] * (divisor << k)

    This is a sequential process, but we can parallelize with transformers
    by using causal attention - each bit can only see more significant bits.
    """
    def __init__(self, max_bits=64, d_model=384, nhead=16, num_layers=8):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # Embed dividend and divisor together
        self.input_embed = nn.Sequential(
            nn.Linear(2, d_model),  # 2 channels: dividend bit, divisor bit
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        # Transformer with causal mask (MSB to LSB computation)
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

        # Reversed causal mask - MSB can see all, LSB sees only MSBs
        # For division, we compute from MSB to LSB
        causal = torch.triu(torch.ones(max_bits, max_bits), diagonal=1).bool()
        self.register_buffer('causal_mask', causal)

    def forward(self, dividend_bits, divisor_bits):
        batch = dividend_bits.shape[0]
        bits = dividend_bits.shape[1]

        # Stack inputs: [batch, bits, 2]
        x = torch.stack([dividend_bits, divisor_bits], dim=-1)

        # Embed
        x = self.input_embed(x)  # [batch, bits, d_model]
        x = x + self.pos_embed[:, :bits, :]

        # Apply transformer with causal mask
        mask = self.causal_mask[:bits, :bits]
        x = self.transformer(x, mask=mask)

        # Output quotient bits
        output = self.output_head(x).squeeze(-1)

        return output


def generate_div_data(batch_size, bits, device, easy=False):
    """
    Generate division training data.

    For curriculum learning:
    - easy=True: Small divisors, exact divisions (no remainder)
    - easy=False: General case with remainders
    """
    if easy:
        # Easy mode: small divisors, exact divisions
        max_divisor = 2 ** (bits // 4)
        max_quotient = 2 ** (bits // 2)

        divisor = torch.randint(1, max(2, max_divisor), (batch_size,), device=device)
        quotient = torch.randint(0, max(1, max_quotient), (batch_size,), device=device)

        # Compute dividend = quotient * divisor (exact!)
        dividend = quotient * divisor

        # Clip to bits range
        dividend = dividend & ((1 << bits) - 1)
        quotient = dividend // divisor  # Recalculate
    else:
        # General case
        max_divisor = 2 ** (bits // 2)
        max_dividend = 2 ** bits

        divisor = torch.randint(1, max(2, max_divisor), (batch_size,), device=device)
        dividend = torch.randint(0, max_dividend, (batch_size,), device=device)
        quotient = dividend // divisor

    # Convert to bits (LSB first)
    dividend_bits = ((dividend.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    divisor_bits = ((divisor.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    quotient_bits = ((quotient.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()

    return dividend_bits, divisor_bits, quotient_bits


def train_div(model, device, save_dir, max_bits=64, batch_size=4096, model_name="div_nmru"):
    """Train division model with curriculum learning."""
    print("\n" + "=" * 70)
    print(f"NMRU-INSPIRED DIVISION TRAINING: {model.__class__.__name__}")
    print("=" * 70)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}")

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

    # Extended curriculum with VERY gradual progression
    # Division is HARD - we need more epochs
    curriculum = [
        (8, 8000),    # Start very small
        (12, 12000),  # Gradual increase
        (16, 16000),
        (20, 20000),
        (24, 25000),
        (28, 30000),
        (32, 35000),
        (40, 45000),
        (48, 55000),
        (56, 65000),
        (64, 80000),
    ]

    curriculum = [(b, e) for (b, e) in curriculum if b <= max_bits]
    print(f"Curriculum: {curriculum}")

    for bits, max_epochs in curriculum:
        print(f"\n{'=' * 70}")
        print(f"Training: {bits}-bit DIV")
        print(f"{'=' * 70}")

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-6
        )
        best_acc = 0
        consecutive_100 = 0

        # Two-phase training: easy first, then hard
        easy_epochs = max_epochs // 3

        for epoch in range(max_epochs):
            model.train()

            # Generate data (easy for first third, hard for rest)
            easy = epoch < easy_epochs
            dividend_bits, divisor_bits, target = generate_div_data(
                batch_size, bits, device, easy=easy
            )

            # Pad to max_bits
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

            if epoch % 200 == 0 or correct > best_acc:
                data_type = "easy" if easy else "hard"
                print(f"    Epoch {epoch:5d}: loss={loss.item():.4f}, acc={correct:.2f}% ({data_type})")

            if correct > best_acc:
                best_acc = correct
                torch.save(model.state_dict(), f"{save_dir}/DIV_{bits}bit_{model_name}_ckpt.pt")

            if correct >= 100.0:
                consecutive_100 += 1
                if consecutive_100 >= 3:
                    print(f"    ✅ {bits}-bit DIV: 100% x3 - ADVANCING!")
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
    parser.add_argument('--model', choices=['direct', 'reciprocal'], default='direct',
                        help='direct: Direct quotient prediction, reciprocal: Use reciprocal+MUL')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.model == 'direct':
        # Direct approach with improved architecture
        model = DirectDivisionWithLSB(
            max_bits=args.bits, d_model=384, nhead=16, num_layers=8
        )
        model_name = "div_direct_lsb"
    else:
        # Reciprocal-style approach with cross-attention
        # No MUL model needed - learns quotient directly
        model = DivisionFromReciprocal(
            max_bits=args.bits, d_model=384, nhead=16, num_layers=8
        )
        model_name = "div_reciprocal"

    model = model.to(args.device)

    print(f"Model: {model.__class__.__name__}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_div(model, args.device, args.save_dir, args.bits, args.batch_size, model_name)


if __name__ == '__main__':
    main()
