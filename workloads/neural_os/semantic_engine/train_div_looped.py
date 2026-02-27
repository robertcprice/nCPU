#!/usr/bin/env python3
"""
LOOPED TRANSFORMER for Division - Breaking the 67% barrier.

KEY INSIGHT: Division requires ITERATION. Standard transformers do one pass.
Looped transformers apply the SAME layers multiple times, giving the model
the computational depth needed for division.

Research shows:
- Bounded-depth transformers are limited to TC^0 complexity
- Division requires TC^1 (sequential) computation
- Looping gives effective depth = num_layers * num_loops

This should break through to 90%+ accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import os
import sys
import argparse
import math

try:
    sys.stdout.reconfigure(line_buffering=True)
except:
    pass


class LoopedDivisionTransformer(nn.Module):
    """
    Looped Transformer for Division.

    Key difference: We apply the transformer block MULTIPLE times (num_loops).
    Each loop gets a learned "loop embedding" so the model knows which iteration it's on.

    Effective depth = num_layers * num_loops
    """
    def __init__(self, max_bits=64, d_model=384, nhead=16, num_layers=4, num_loops=8):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model
        self.num_loops = num_loops

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        # Loop embedding - tells model which iteration we're on
        self.loop_embed = nn.Parameter(torch.randn(num_loops, 1, d_model) * 0.02)

        # Shared transformer block (applied num_loops times)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer norm between loops
        self.loop_norm = nn.LayerNorm(d_model)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        print(f"LoopedDivisionTransformer: {num_layers} layers x {num_loops} loops = {num_layers * num_loops} effective depth")

    def forward(self, dividend_bits, divisor_bits):
        batch = dividend_bits.shape[0]
        bits = dividend_bits.shape[1]

        # Stack inputs
        x = torch.stack([dividend_bits, divisor_bits], dim=-1)
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :bits, :]

        # LOOPED COMPUTATION - the key innovation
        for loop_idx in range(self.num_loops):
            # Add loop embedding so model knows which iteration
            x = x + self.loop_embed[loop_idx]

            # Apply shared transformer
            x = self.transformer(x)

            # Normalize between loops for stability
            x = self.loop_norm(x)

        # Output
        output = self.output_head(x).squeeze(-1)
        return output


class RecurrentDivisionTransformer(nn.Module):
    """
    Alternative: Recurrent refinement approach.

    Each loop refines the previous prediction, similar to iterative algorithms.
    Uses residual connections between loops.
    """
    def __init__(self, max_bits=64, d_model=384, nhead=16, num_layers=4, num_loops=8):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model
        self.num_loops = num_loops

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        # Loop embedding
        self.loop_embed = nn.Embedding(num_loops, d_model)

        # Shared refinement block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.refine_block = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Gating for residual
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(d_model)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        print(f"RecurrentDivisionTransformer: {num_layers} layers x {num_loops} refinement loops")

    def forward(self, dividend_bits, divisor_bits):
        batch = dividend_bits.shape[0]
        bits = dividend_bits.shape[1]

        # Initial encoding
        x = torch.stack([dividend_bits, divisor_bits], dim=-1)
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :bits, :]

        # Iterative refinement
        state = x
        for loop_idx in range(self.num_loops):
            # Add loop information
            loop_emb = self.loop_embed(torch.tensor(loop_idx, device=x.device))
            h = state + loop_emb

            # Refine
            refined = self.refine_block(h)

            # Gated residual update
            gate = self.gate(torch.cat([state, refined], dim=-1))
            state = gate * refined + (1 - gate) * state
            state = self.norm(state)

        output = self.output_head(state).squeeze(-1)
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


def train_looped(model, device, save_dir, start_bits=8, max_bits=64,
                 batch_size=4096, accum_steps=4):
    """Train looped transformer with curriculum."""
    print("\n" + "=" * 70)
    print("LOOPED TRANSFORMER DIVISION TRAINING")
    print("=" * 70)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size} x {accum_steps} = {batch_size * accum_steps}")

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

    # Start fresh with curriculum - looped model should learn faster
    curriculum = [
        (8, 5000), (12, 8000), (16, 12000), (20, 20000),
        (24, 25000), (28, 30000), (32, 40000),
        (40, 50000), (48, 60000), (56, 70000), (64, 100000)
    ]

    curriculum = [(b, e) for (b, e) in curriculum if b >= start_bits and b <= max_bits]
    print(f"Curriculum: {curriculum}")

    for bits, max_epochs in curriculum:
        print(f"\n{'=' * 70}")
        print(f"Training: {bits}-bit DIV (LOOPED)")
        print(f"{'=' * 70}")

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=2000, T_mult=2, eta_min=1e-5
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
                torch.save(model.state_dict(), f"{save_dir}/DIV_{bits}bit_looped_ckpt.pt")

            if correct >= 100.0:
                consecutive_100 += 1
                if consecutive_100 >= 3:
                    print(f"    ✅ {bits}-bit: 100% x3 - ADVANCING!")
                    torch.save(model.state_dict(), f"{save_dir}/DIV_{bits}bit_looped_100pct.pt")
                    break
            else:
                consecutive_100 = 0

        print(f"    Level complete: best={best_acc:.2f}%")

    print("\n" + "=" * 70)
    print("LOOPED TRAINING COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--bits', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--accum-steps', type=int, default=4)
    parser.add_argument('--num-loops', type=int, default=8, help='Number of transformer loops')
    parser.add_argument('--num-layers', type=int, default=4, help='Layers per loop')
    parser.add_argument('--start-bits', type=int, default=8)
    parser.add_argument('--model-type', choices=['looped', 'recurrent'], default='looped')
    parser.add_argument('--checkpoint', type=str, default=None, help='Optional checkpoint to load')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Create model
    if args.model_type == 'looped':
        model = LoopedDivisionTransformer(
            max_bits=args.bits,
            d_model=384,
            nhead=16,
            num_layers=args.num_layers,
            num_loops=args.num_loops
        )
    else:
        model = RecurrentDivisionTransformer(
            max_bits=args.bits,
            d_model=384,
            nhead=16,
            num_layers=args.num_layers,
            num_loops=args.num_loops
        )

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        print("Checkpoint loaded!")

    model = model.to(args.device)

    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Effective depth: {args.num_layers} x {args.num_loops} = {args.num_layers * args.num_loops}")

    train_looped(model, args.device, args.save_dir,
                 start_bits=args.start_bits, max_bits=args.bits,
                 batch_size=args.batch_size, accum_steps=args.accum_steps)


if __name__ == '__main__':
    main()
