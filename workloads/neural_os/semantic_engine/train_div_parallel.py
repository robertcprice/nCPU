#!/usr/bin/env python3
"""
PURE PARALLEL DIV using transformer architecture.

Division is fundamentally different from multiplication:
- MUL: a × b = result (combine)
- DIV: a ÷ b = quotient (decompose, find q where q*b ≤ a < (q+1)*b)

Key insight: Division quotient bit k is 1 if:
  (divisor << k) ≤ remaining_dividend

We can encode both dividend and divisor together and let the transformer
learn the comparison and subtraction patterns.

Architecture approach:
1. Encode dividend and divisor as bit sequences
2. Use transformer to learn the division algorithm
3. Output quotient bits directly
4. Curriculum learning from 8-bit to 64-bit
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import os
import sys
import argparse

# Enable line buffering for real-time output
try:
    sys.stdout.reconfigure(line_buffering=True)
except:
    pass  # Python < 3.7 compatibility


class DivisionTransformer(nn.Module):
    """
    Transformer-based division model.

    Input: dividend bits + divisor bits (concatenated)
    Output: quotient bits

    The transformer learns to compute each quotient bit by attending
    to the relevant parts of dividend and divisor.
    """
    def __init__(self, max_bits=64, d_model=384, nhead=16, num_layers=7):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # Input projection: 2 channels (dividend bit, divisor bit) -> d_model
        self.input_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        # Transformer encoder
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

        # Causal mask - for division, higher bits may depend on lower bit results
        causal = torch.triu(torch.ones(max_bits, max_bits), diagonal=1).bool()
        self.register_buffer('causal_mask', causal)

    def forward(self, dividend_bits, divisor_bits):
        """
        Forward pass.

        Args:
            dividend_bits: [batch, bits] - dividend in bit representation
            divisor_bits: [batch, bits] - divisor in bit representation

        Returns:
            quotient_logits: [batch, bits] - logits for quotient bits
        """
        batch = dividend_bits.shape[0]
        bits = dividend_bits.shape[1]

        # Stack dividend and divisor as 2-channel input
        # Shape: [batch, bits, 2]
        x = torch.stack([dividend_bits, divisor_bits], dim=-1)

        # Project to d_model
        x = self.input_proj(x)

        # Add positional embedding
        x = x + self.pos_embedding[:, :bits, :]

        # Apply transformer with causal mask
        mask = self.causal_mask[:bits, :bits]
        x = self.transformer(x, mask=mask)

        # Output quotient bits
        output = self.output_head(x).squeeze(-1)

        return output


class DivisionTransformerV2(nn.Module):
    """
    V2: Enhanced architecture with explicit dividend/divisor encoding
    and cross-attention between them.
    """
    def __init__(self, max_bits=64, d_model=384, nhead=16, num_layers=7):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # Separate embeddings for dividend and divisor
        self.dividend_embed = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

        self.divisor_embed = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Position embeddings
        self.pos_dividend = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)
        self.pos_divisor = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        # Cross-attention: dividend attends to divisor
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=0.1, batch_first=True
        )

        # Self-attention transformer
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

        # Causal mask for quotient prediction (MSB to LSB dependency)
        causal = torch.triu(torch.ones(max_bits, max_bits), diagonal=1).bool()
        self.register_buffer('causal_mask', causal)

    def forward(self, dividend_bits, divisor_bits):
        batch = dividend_bits.shape[0]
        bits = dividend_bits.shape[1]

        # Embed dividend and divisor separately
        div_emb = self.dividend_embed(dividend_bits.unsqueeze(-1))
        dis_emb = self.divisor_embed(divisor_bits.unsqueeze(-1))

        # Add position embeddings
        div_emb = div_emb + self.pos_dividend[:, :bits, :]
        dis_emb = dis_emb + self.pos_divisor[:, :bits, :]

        # Cross-attention: dividend attends to divisor
        cross_out, _ = self.cross_attn(div_emb, dis_emb, dis_emb)

        # Combine with residual
        x = div_emb + cross_out

        # Self-attention with causal mask
        mask = self.causal_mask[:bits, :bits]
        x = self.transformer(x, mask=mask)

        # Output quotient bits
        output = self.output_head(x).squeeze(-1)

        return output


def generate_div_data(batch_size, bits, device):
    """
    Generate division training data.

    For N-bit division:
    - Dividend: random N-bit number
    - Divisor: random N//2-bit number (to ensure quotient fits in N bits)
    - Quotient: dividend // divisor
    """
    # Divisor should be non-zero and smaller to get meaningful quotients
    max_divisor = 2 ** (bits // 2)
    divisor = torch.randint(1, max_divisor, (batch_size,), device=device)

    # Dividend can be full range
    max_dividend = 2 ** bits
    dividend = torch.randint(0, max_dividend, (batch_size,), device=device)

    # Compute quotient
    quotient = dividend // divisor

    # Convert to bits
    dividend_bits = ((dividend.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    divisor_bits = ((divisor.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    quotient_bits = ((quotient.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()

    return dividend_bits, divisor_bits, quotient_bits


def train_div(model, device, save_dir, max_bits=64, batch_size=4096, model_name="div_v1"):
    print("\n" + "=" * 70)
    print(f"DIVISION TRAINING: {model.__class__.__name__}")
    print("=" * 70)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}")

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

    # Curriculum: start small, grow to max_bits
    curriculum = [
        (8, 3000), (16, 5000), (24, 8000), (32, 12000),
        (40, 15000), (48, 20000), (56, 25000), (64, 30000)
    ]

    # Filter to max_bits
    curriculum = [(b, e) for (b, e) in curriculum if b <= max_bits]

    print(f"Curriculum: {curriculum}")

    for bits, max_epochs in curriculum:
        print(f"\n{'=' * 70}")
        print(f"Training: {bits}-bit DIV")
        print(f"{'=' * 70}")

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
        best_acc = 0
        consecutive_100 = 0

        for epoch in range(max_epochs):
            model.train()

            dividend_bits, divisor_bits, target = generate_div_data(batch_size, bits, device)

            # Pad to max_bits if needed
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
                print(f"    Epoch {epoch:4d}: loss={loss.item():.4f}, acc={correct:.2f}%")

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
    parser.add_argument('--model', choices=['v1', 'v2'], default='v2', help='Model version')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Create model based on version
    if args.model == 'v1':
        model = DivisionTransformer(max_bits=args.bits, d_model=384, nhead=16, num_layers=7)
        model_name = "div_v1"
    else:
        model = DivisionTransformerV2(max_bits=args.bits, d_model=384, nhead=16, num_layers=7)
        model_name = "div_v2"

    model = model.to(args.device)

    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_div(model, args.device, args.save_dir,
              max_bits=args.bits, batch_size=args.batch_size, model_name=model_name)


if __name__ == '__main__':
    main()
