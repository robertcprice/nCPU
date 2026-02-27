#!/usr/bin/env python3
"""
LONG DIVISION Neural Network

Division is the hardest operation. This approach mimics ACTUAL long division:

For each quotient bit from MSB to LSB:
1. Check: Is (current_partial_quotient + 2^k) * divisor <= dividend?
2. If yes: quotient[k] = 1
3. If no: quotient[k] = 0

This is autoregressive prediction - each bit depends on:
- The dividend
- The divisor
- The quotient bits we've already decided (higher positions)

Key insight: This is a SEQUENTIAL COMPARISON problem, not a single-shot prediction.
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


class LongDivisionNet(nn.Module):
    """
    Long Division Network - Autoregressive quotient prediction.

    Key architecture choices:
    1. Causal mask: Each quotient bit only sees higher-order bits
    2. Cross-attention: Dividend/divisor inform each quotient decision
    3. Comparison-focused: Network learns to compare (q*b) vs a

    For quotient bit k, the network learns:
    - Is (accumulated_quotient + 2^k) * divisor <= dividend?
    """
    def __init__(self, max_bits=64, d_model=384, nhead=16, num_layers=8):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # Embed dividend bits
        self.dividend_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Embed divisor bits
        self.divisor_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Position embeddings
        self.pos_dividend = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)
        self.pos_divisor = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)
        self.pos_quotient = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        # Quotient bit queries - learnable queries for each quotient position
        # These will cross-attend to dividend/divisor and self-attend causally
        self.quotient_queries = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        # Cross-attention: quotient queries attend to dividend
        self.cross_attn_dividend = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
            for _ in range(num_layers)
        ])

        # Cross-attention: quotient queries attend to divisor
        self.cross_attn_divisor = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
            for _ in range(num_layers)
        ])

        # Self-attention with CAUSAL mask (quotient bit k can only see bits > k)
        # Note: We go from MSB (bit max_bits-1) to LSB (bit 0)
        # So bit k can see bits k+1, k+2, ... max_bits-1
        self.self_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
            for _ in range(num_layers)
        ])

        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers * 4)  # 4 norms per layer
        ])

        # FFN per layer
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(0.1),
            )
            for _ in range(num_layers)
        ])

        # Output head - predict probability for each quotient bit
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        # Register causal mask
        # For autoregressive from MSB to LSB: bit k can see bits j where j > k
        # This means we mask positions j <= k (lower triangle including diagonal)
        causal = torch.triu(torch.ones(max_bits, max_bits), diagonal=1).bool()
        self.register_buffer('causal_mask', ~causal)  # Mask where True = masked out

    def forward(self, dividend_bits, divisor_bits):
        """
        Forward pass.

        Args:
            dividend_bits: [batch, bits] - dividend as bit vector (LSB first)
            divisor_bits: [batch, bits] - divisor as bit vector (LSB first)

        Returns:
            quotient_logits: [batch, bits] - logits for each quotient bit
        """
        batch = dividend_bits.shape[0]
        bits = dividend_bits.shape[1]

        # Embed dividend and divisor
        div_emb = self.dividend_embed(dividend_bits.unsqueeze(-1))  # [batch, bits, d_model]
        dis_emb = self.divisor_embed(divisor_bits.unsqueeze(-1))    # [batch, bits, d_model]

        # Add position embeddings
        div_emb = div_emb + self.pos_dividend[:, :bits, :]
        dis_emb = dis_emb + self.pos_divisor[:, :bits, :]

        # Initialize quotient queries
        q = self.quotient_queries[:, :bits, :].expand(batch, -1, -1)
        q = q + self.pos_quotient[:, :bits, :]

        # Get causal mask for this bit width
        mask = self.causal_mask[:bits, :bits]

        # Process through layers
        norm_idx = 0
        for i in range(len(self.cross_attn_dividend)):
            # Cross-attention to dividend
            cross_div, _ = self.cross_attn_dividend[i](q, div_emb, div_emb)
            q = self.norms[norm_idx](q + cross_div)
            norm_idx += 1

            # Cross-attention to divisor
            cross_dis, _ = self.cross_attn_divisor[i](q, dis_emb, dis_emb)
            q = self.norms[norm_idx](q + cross_dis)
            norm_idx += 1

            # Causal self-attention (quotient bits attending to each other)
            self_out, _ = self.self_attn[i](q, q, q, attn_mask=mask)
            q = self.norms[norm_idx](q + self_out)
            norm_idx += 1

            # FFN
            q = self.norms[norm_idx](q + self.ffn[i](q))
            norm_idx += 1

        # Output quotient bit predictions
        output = self.output_head(q).squeeze(-1)

        return output


class LongDivisionNetV2(nn.Module):
    """
    V2: Simpler architecture with explicit remainder tracking.

    Key insight: Long division is really about tracking remainder:
    - remainder[k] = remainder[k+1] * 2 + dividend[k]
    - quotient[k] = 1 if remainder[k] >= divisor else 0
    - if quotient[k] == 1: remainder[k] -= divisor

    This version embeds the algorithm more directly.
    """
    def __init__(self, max_bits=64, d_model=384, nhead=16, num_layers=6):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # Input: dividend bits + divisor bits + position encoding
        # For each quotient position, we concatenate all input info
        self.input_proj = nn.Linear(max_bits * 2 + max_bits, d_model)  # dividend + divisor + position

        # Learnable position encoding for quotient positions
        self.pos_embed = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        # Main transformer with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        # Causal mask (MSB to LSB processing)
        causal = torch.triu(torch.ones(max_bits, max_bits), diagonal=1).bool()
        self.register_buffer('causal_mask', ~causal)

    def forward(self, dividend_bits, divisor_bits):
        batch = dividend_bits.shape[0]
        bits = dividend_bits.shape[1]

        # Create position encoding (one-hot for each quotient bit position)
        pos_onehot = torch.eye(self.max_bits, device=dividend_bits.device)[:bits]
        pos_onehot = pos_onehot.unsqueeze(0).expand(batch, -1, -1)  # [batch, bits, max_bits]

        # Expand dividend and divisor for each quotient position
        div_expanded = dividend_bits.unsqueeze(1).expand(-1, bits, -1)  # [batch, bits, bits]
        dis_expanded = divisor_bits.unsqueeze(1).expand(-1, bits, -1)   # [batch, bits, bits]

        # Pad to max_bits if needed
        if bits < self.max_bits:
            div_expanded = F.pad(div_expanded, (0, self.max_bits - bits))
            dis_expanded = F.pad(dis_expanded, (0, self.max_bits - bits))

        # Concatenate all inputs
        x = torch.cat([div_expanded, dis_expanded, pos_onehot], dim=-1)  # [batch, bits, max_bits*2 + max_bits]

        # Project
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :bits, :]

        # Transform with causal mask
        mask = self.causal_mask[:bits, :bits]
        x = self.transformer(x, mask=mask)

        # Output
        output = self.output_head(x).squeeze(-1)

        return output


class ComparisonDivNet(nn.Module):
    """
    Division as a series of comparisons.

    For each quotient bit k (from MSB to LSB):
    - Need to determine if (partial_quotient + 2^k) * divisor <= dividend
    - This is a COMPARISON operation

    Architecture: Learn to compare shifted divisor against dividend minus accumulated product.
    """
    def __init__(self, max_bits=64, d_model=384, nhead=16, num_layers=6):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # Embed both operands together for comparison
        self.operand_embed = nn.Sequential(
            nn.Linear(2, d_model // 2),  # dividend bit, divisor bit
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        # Shift amount embedding (for the comparison at each quotient bit)
        self.shift_embed = nn.Embedding(max_bits, d_model)

        # Main transformer - processes comparison at each shift level
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Comparison head - produces comparison result for each quotient bit
        self.comparison_head = nn.Sequential(
            nn.Linear(d_model * max_bits, d_model),
            nn.GELU(),
            nn.Linear(d_model, max_bits)
        )

    def forward(self, dividend_bits, divisor_bits):
        batch = dividend_bits.shape[0]
        bits = dividend_bits.shape[1]
        device = dividend_bits.device

        # Stack operands: [batch, bits, 2]
        operands = torch.stack([dividend_bits, divisor_bits], dim=-1)

        # Embed: [batch, bits, d_model]
        x = self.operand_embed(operands)
        x = x + self.pos_embed[:, :bits, :]

        # Process
        x = self.transformer(x)

        # Flatten and produce quotient bits
        x_flat = x.reshape(batch, -1)

        # Pad if needed
        if bits < self.max_bits:
            x_flat = F.pad(x_flat, (0, self.d_model * (self.max_bits - bits)))

        output = self.comparison_head(x_flat)

        return output[:, :bits]


def generate_div_data(batch_size, bits, device, easy=False):
    """
    Generate division training data.

    For N-bit division:
    - Dividend: random N-bit number
    - Divisor: random number (constrained to avoid overflow)
    - Quotient: dividend // divisor

    Easy mode: exact divisions (dividend = quotient * divisor)
    """
    if easy:
        # Generate exact divisions for easier learning
        max_divisor = 2 ** (bits // 3)  # Smaller divisors
        max_quotient = 2 ** (bits // 2)  # Ensure quotient fits

        divisor = torch.randint(1, max(2, max_divisor), (batch_size,), device=device)
        quotient = torch.randint(0, max(1, max_quotient), (batch_size,), device=device)

        # Compute dividend = quotient * divisor (exact!)
        dividend = quotient * divisor

        # Clip to bits range
        dividend = dividend & ((1 << bits) - 1)
        quotient = dividend // divisor  # Recalculate in case of overflow
    else:
        # Regular random divisions
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


def train_div(model, device, save_dir, max_bits=64, batch_size=4096, model_name="longdiv"):
    print("\n" + "=" * 70)
    print(f"LONG DIVISION TRAINING: {model.__class__.__name__}")
    print("=" * 70)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}")

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

    # Extended curriculum with more epochs for difficult division
    curriculum = [
        (8, 10000), (12, 15000), (16, 20000), (20, 25000),
        (24, 30000), (28, 35000), (32, 40000),
        (40, 50000), (48, 60000), (56, 70000), (64, 80000)
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

        # Start with easy data (exact divisions), transition to full
        easy_epochs = max_epochs // 3

        for epoch in range(max_epochs):
            model.train()

            # Curriculum within curriculum: start easy
            use_easy = epoch < easy_epochs
            dividend_bits, divisor_bits, target = generate_div_data(
                batch_size, bits, device, easy=use_easy
            )

            # Pad to max_bits if model requires fixed size
            if hasattr(model, 'max_bits') and bits < model.max_bits:
                dividend_bits = F.pad(dividend_bits, (0, model.max_bits - bits))
                divisor_bits = F.pad(divisor_bits, (0, model.max_bits - bits))
                target = F.pad(target, (0, model.max_bits - bits))

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
                mode = "easy" if use_easy else "full"
                print(f"    Epoch {epoch:5d}: loss={loss.item():.4f}, acc={correct:.2f}% ({mode})")

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
    print("LONG DIVISION TRAINING COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--bits', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--model', choices=['v1', 'v2', 'compare'], default='v1',
                        help='v1=LongDivisionNet, v2=Simpler, compare=ComparisonDivNet')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.model == 'v1':
        model = LongDivisionNet(max_bits=args.bits, d_model=384, nhead=16, num_layers=6)
        model_name = "longdiv_v1"
    elif args.model == 'v2':
        model = LongDivisionNetV2(max_bits=args.bits, d_model=384, nhead=16, num_layers=6)
        model_name = "longdiv_v2"
    else:
        model = ComparisonDivNet(max_bits=args.bits, d_model=384, nhead=16, num_layers=6)
        model_name = "longdiv_compare"

    model = model.to(args.device)

    print(f"Model: {model.__class__.__name__}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_div(model, args.device, args.save_dir,
              max_bits=args.bits, batch_size=args.batch_size, model_name=model_name)


if __name__ == '__main__':
    main()
