#!/usr/bin/env python3
"""
PARALLEL DIV - Inspired by our 100% MUL architecture

Key insight from MUL success:
- Outer product computes ALL partial products in parallel
- Causal transformer handles carry propagation
- NO sequential dependencies in the core computation

For DIV, we need similar parallelism:
- Each quotient bit is DETERMINED by global (dividend, divisor) relationship
- No need for sequential bit-by-bit computation
- Model should learn to extract quotient bits simultaneously

Architecture approaches:
1. Cross-Product Attention: dividend bits × divisor bits → quotient pattern
2. Comparison Matrix: Like outer product but for >= comparisons
3. Direct Parallel: Same structure as MUL, different operation

Division is fundamentally: Find q such that q*b ≤ a < (q+1)*b
This is a COMPARISON problem, not an accumulation problem.
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


class ParallelDivNet(nn.Module):
    """
    Parallel Division Network - NO causal masking, processes all bits simultaneously.

    Key architectural choices:
    1. Cross-product attention between dividend and divisor bits
    2. Position-aware processing (bit position matters for division)
    3. No sequential dependencies - all quotient bits computed in parallel

    This mirrors the MUL architecture that achieved 100% accuracy.
    """
    def __init__(self, max_bits=64, d_model=384, nhead=16, num_layers=6):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # Embed dividend and divisor with position info
        self.dividend_embed = nn.Sequential(
            nn.Linear(1 + max_bits, d_model),  # bit value + position one-hot
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.divisor_embed = nn.Sequential(
            nn.Linear(1 + max_bits, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Cross-attention: each quotient position attends to dividend and divisor
        self.cross_attn_div = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
            for _ in range(num_layers)
        ])

        self.cross_attn_dis = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
            for _ in range(num_layers)
        ])

        # Self-attention among quotient bits (NO causal mask - parallel!)
        self.self_attn = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])

        # Quotient position queries
        self.quotient_queries = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, dividend_bits, divisor_bits):
        batch = dividend_bits.shape[0]
        bits = dividend_bits.shape[1]
        device = dividend_bits.device

        # Create position one-hot encoding
        pos_onehot = torch.eye(self.max_bits, device=device)[:bits]  # [bits, max_bits]
        pos_onehot = pos_onehot.unsqueeze(0).expand(batch, -1, -1)   # [batch, bits, max_bits]

        # Combine bits with position info
        div_input = torch.cat([dividend_bits.unsqueeze(-1), pos_onehot], dim=-1)
        dis_input = torch.cat([divisor_bits.unsqueeze(-1), pos_onehot], dim=-1)

        # Embed
        div_emb = self.dividend_embed(div_input)  # [batch, bits, d_model]
        dis_emb = self.divisor_embed(dis_input)   # [batch, bits, d_model]

        # Initialize quotient queries
        q = self.quotient_queries[:, :bits, :].expand(batch, -1, -1)

        # Process through layers (NO causal masking!)
        for i in range(len(self.cross_attn_div)):
            # Cross-attend to dividend
            cross_div, _ = self.cross_attn_div[i](q, div_emb, div_emb)
            q = q + cross_div

            # Cross-attend to divisor
            cross_dis, _ = self.cross_attn_dis[i](q, dis_emb, dis_emb)
            q = q + cross_dis

            # Self-attention among quotient bits (parallel)
            q = self.self_attn[i](q)

        # Output
        output = self.output_head(q).squeeze(-1)

        return output


class OuterProductDivNet(nn.Module):
    """
    Division using Outer Product approach (inspired by MUL success).

    For MUL: outer(a, b) creates partial products matrix, sum anti-diagonals
    For DIV: outer(dividend, divisor) creates comparison matrix

    Key insight: quotient bit k indicates whether 2^k * divisor "fits into" dividend
    This can be seen as a series of comparisons at different scales.
    """
    def __init__(self, max_bits=64, d_model=384, nhead=16, num_layers=6):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # Create interaction matrix between dividend and divisor
        # Similar to partial products but for comparison
        self.interaction_proj = nn.Sequential(
            nn.Linear(4, d_model // 4),  # [div_bit, dis_bit, div_pos, dis_pos]
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 2),
        )

        # Position embeddings for the 2D interaction matrix
        self.pos_embed_2d = nn.Parameter(torch.randn(1, max_bits * max_bits, d_model // 2) * 0.02)

        # Transformer to process interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Reduce to quotient bits
        self.quotient_queries = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        self.reduce_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, dividend_bits, divisor_bits):
        batch = dividend_bits.shape[0]
        bits = dividend_bits.shape[1]
        device = dividend_bits.device

        # Create outer product of positions
        div_pos = torch.arange(bits, device=device).float() / bits
        dis_pos = torch.arange(bits, device=device).float() / bits

        # Create 2D grid of interactions
        # [batch, bits, bits, 4] -> [div_bit, dis_bit, div_pos, dis_pos]
        div_grid = dividend_bits.unsqueeze(2).expand(-1, -1, bits)  # [batch, bits, bits]
        dis_grid = divisor_bits.unsqueeze(1).expand(-1, bits, -1)   # [batch, bits, bits]
        div_pos_grid = div_pos.unsqueeze(1).expand(bits, bits).unsqueeze(0).expand(batch, -1, -1)
        dis_pos_grid = dis_pos.unsqueeze(0).expand(bits, bits).unsqueeze(0).expand(batch, -1, -1)

        interactions = torch.stack([div_grid, dis_grid, div_pos_grid, dis_pos_grid], dim=-1)
        # [batch, bits, bits, 4]

        # Flatten and project
        interactions_flat = interactions.reshape(batch, bits * bits, 4)
        x = self.interaction_proj(interactions_flat)  # [batch, bits*bits, d_model//2]

        # Add position embedding
        x = torch.cat([x, self.pos_embed_2d[:, :bits*bits, :]], dim=-1)  # [batch, bits*bits, d_model]

        # Transform
        x = self.transformer(x)

        # Reduce to quotient bits via attention
        q = self.quotient_queries[:, :bits, :].expand(batch, -1, -1)
        reduced, _ = self.reduce_attn(q, x, x)

        # Output
        output = self.output_head(reduced).squeeze(-1)

        return output


class ComparisonMatrixDivNet(nn.Module):
    """
    Division via Comparison Matrix.

    Key insight: For each quotient bit position k, we're essentially asking:
    "Is dividend >= (current_partial * divisor + 2^k * divisor)?"

    But we can reformulate this as learning a comparison function:
    comparison_matrix[i][j] tells us about bit i of dividend vs bit j of divisor
    at various shift levels.

    This is like the partial products matrix but for comparisons.
    """
    def __init__(self, max_bits=64, d_model=384, nhead=16, num_layers=6):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # Input: concatenated dividend and divisor with position
        self.input_proj = nn.Linear(max_bits * 2, d_model)

        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        # Main transformer (NO causal mask)
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

    def forward(self, dividend_bits, divisor_bits):
        batch = dividend_bits.shape[0]
        bits = dividend_bits.shape[1]
        device = dividend_bits.device

        # Pad if needed
        if bits < self.max_bits:
            dividend_bits = F.pad(dividend_bits, (0, self.max_bits - bits))
            divisor_bits = F.pad(divisor_bits, (0, self.max_bits - bits))

        # Concatenate for each output position
        combined = torch.cat([dividend_bits, divisor_bits], dim=-1)  # [batch, max_bits*2]
        combined = combined.unsqueeze(1).expand(-1, bits, -1)        # [batch, bits, max_bits*2]

        # Project
        x = self.input_proj(combined)  # [batch, bits, d_model]
        x = x + self.pos_embed[:, :bits, :]

        # Transform (NO causal mask - parallel processing)
        x = self.transformer(x)

        # Output
        output = self.output_head(x).squeeze(-1)

        return output[:, :bits] if bits < self.max_bits else output


class DirectParallelDivNet(nn.Module):
    """
    Simplest parallel DIV - same structure as our successful MUL model.

    Key: Just concatenate dividend and divisor, let transformer figure it out.
    Works because transformer can learn arbitrary functions given enough capacity.
    """
    def __init__(self, max_bits=64, d_model=384, nhead=16, num_layers=8):
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


def generate_div_data(batch_size, bits, device, mode='standard'):
    """
    Generate division training data.

    Modes:
    - 'easy': Exact divisions (dividend = quotient * divisor)
    - 'standard': Random divisions with constrained divisor
    - 'hard': Full range divisions
    """
    if mode == 'easy':
        # Exact divisions for easier initial learning
        max_divisor = 2 ** (bits // 3)
        max_quotient = 2 ** (bits // 2)

        divisor = torch.randint(1, max(2, max_divisor), (batch_size,), device=device)
        quotient = torch.randint(0, max(1, max_quotient), (batch_size,), device=device)
        dividend = quotient * divisor
        dividend = dividend & ((1 << bits) - 1)
        quotient = dividend // divisor

    elif mode == 'standard':
        # Standard: divisor up to half bit-width to keep quotient in range
        max_divisor = 2 ** (bits // 2)
        divisor = torch.randint(1, max(2, max_divisor), (batch_size,), device=device)
        max_dividend = 2 ** bits
        dividend = torch.randint(0, max_dividend, (batch_size,), device=device)
        quotient = dividend // divisor

    else:  # hard
        # Full range - quotient may overflow
        max_val = 2 ** bits
        dividend = torch.randint(0, max_val, (batch_size,), device=device)
        divisor = torch.randint(1, max_val, (batch_size,), device=device)
        quotient = dividend // divisor
        quotient = quotient & ((1 << bits) - 1)  # Clip quotient to bits

    # Convert to bits
    dividend_bits = ((dividend.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    divisor_bits = ((divisor.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    quotient_bits = ((quotient.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()

    return dividend_bits, divisor_bits, quotient_bits


def train_div(model, device, save_dir, max_bits=64, batch_size=4096, model_name="parallel_div"):
    print("\n" + "=" * 70)
    print(f"PARALLEL DIV TRAINING: {model.__class__.__name__}")
    print("=" * 70)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}")

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

    # Progressive curriculum
    curriculum = [
        (8, 8000), (12, 12000), (16, 16000), (20, 20000),
        (24, 25000), (28, 30000), (32, 35000),
        (40, 45000), (48, 55000), (56, 65000), (64, 80000)
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

        # Data mode curriculum
        easy_epochs = max_epochs // 3

        for epoch in range(max_epochs):
            model.train()

            # Progressive difficulty
            if epoch < easy_epochs:
                mode = 'easy'
            elif epoch < 2 * easy_epochs:
                mode = 'standard'
            else:
                mode = 'standard'  # Don't go to 'hard' yet

            dividend_bits, divisor_bits, target = generate_div_data(
                batch_size, bits, device, mode=mode
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
                print(f"    Epoch {epoch:5d}: loss={loss.item():.4f}, acc={correct:.2f}% ({mode})")
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
    print("PARALLEL DIV TRAINING COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--bits', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--model', choices=['parallel', 'outer', 'compare', 'direct'], default='direct',
                        help='parallel=ParallelDivNet, outer=OuterProductDivNet, compare=ComparisonMatrixDivNet, direct=DirectParallelDivNet')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.model == 'parallel':
        model = ParallelDivNet(max_bits=args.bits, d_model=384, nhead=16, num_layers=6)
        model_name = "parallel_v1"
    elif args.model == 'outer':
        model = OuterProductDivNet(max_bits=args.bits, d_model=384, nhead=16, num_layers=6)
        model_name = "outer_v1"
    elif args.model == 'compare':
        model = ComparisonMatrixDivNet(max_bits=args.bits, d_model=384, nhead=16, num_layers=6)
        model_name = "compare_v1"
    else:
        model = DirectParallelDivNet(max_bits=args.bits, d_model=384, nhead=16, num_layers=8)
        model_name = "direct_v1"

    model = model.to(args.device)

    print(f"Model: {model.__class__.__name__}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_div(model, args.device, args.save_dir,
              max_bits=args.bits, batch_size=args.batch_size, model_name=model_name)


if __name__ == '__main__':
    main()
