#!/usr/bin/env python3
"""
PARALLEL MUL - True parallel bit multiplication

NO SEQUENTIAL LOOPS! All bit positions processed simultaneously.

Key insight: Use matrix operations that handle all 64 bits in ONE forward pass.

Architecture options:
1. Direct parallel: Concatenate inputs → parallel MLP → all output bits
2. Transformer: Self-attention lets all bits interact simultaneously
3. Parallel partial products: Generate all partials via matrix ops, sum via learned parallel reducer
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import os
import sys
import argparse

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)


class ParallelMulNet(nn.Module):
    """
    Fully parallel multiplication - NO LOOPS in forward pass.

    Architecture:
    1. Embed both input bit vectors
    2. Cross-attention between a and b (parallel over all positions)
    3. Parallel output projection

    All operations are batched matrix multiplies - true parallelism!
    """
    def __init__(self, max_bits=64, d_model=256, n_heads=8, n_layers=4):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # Bit embeddings (parallel: processes all bits at once)
        self.bit_embed = nn.Linear(1, d_model)

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        # Input projections (parallel over all positions)
        self.proj_a = nn.Linear(d_model, d_model)
        self.proj_b = nn.Linear(d_model, d_model)

        # Transformer layers for bit interaction (all parallel via attention)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model * 2,  # Concatenated a and b
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            for _ in range(n_layers)
        ])

        # Parallel output projection (all bits at once)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, a_bits, b_bits):
        """
        Fully parallel forward pass - NO LOOPS!

        a_bits, b_bits: [batch, bits]
        returns: [batch, bits] logits for product
        """
        batch = a_bits.shape[0]
        bits = a_bits.shape[1]

        # Embed bits: [batch, bits] -> [batch, bits, d_model]
        # This is PARALLEL - one matrix multiply for all bits
        a_emb = self.bit_embed(a_bits.unsqueeze(-1)) + self.pos_embed[:, :bits]
        b_emb = self.bit_embed(b_bits.unsqueeze(-1)) + self.pos_embed[:, :bits]

        # Project (parallel)
        a_proj = self.proj_a(a_emb)  # [batch, bits, d_model]
        b_proj = self.proj_b(b_emb)  # [batch, bits, d_model]

        # Concatenate a and b features for each position
        # [batch, bits, d_model*2]
        combined = torch.cat([a_proj, b_proj], dim=-1)

        # Transformer layers (attention is parallel over positions!)
        for layer in self.layers:
            combined = layer(combined)

        # Output projection (parallel over all positions)
        # [batch, bits, 1] -> [batch, bits]
        output = self.output_proj(combined).squeeze(-1)

        return output


class ParallelMulNetV2(nn.Module):
    """
    Alternative: Parallel partial product generation + parallel summation.

    Key insight: a × b = Σ (a × b[i]) << i

    But instead of looping, we:
    1. Generate ALL masked versions of a in parallel: a * b[0], a * b[1], ..., a * b[63]
    2. Apply position-dependent transformations in parallel
    3. Sum via learned parallel reduction

    NO PYTHON LOOPS - all matrix operations!
    """
    def __init__(self, max_bits=64, hidden_dim=512):
        super().__init__()
        self.max_bits = max_bits

        # Parallel partial product transformer
        # Input: [batch, bits] a_bits and [batch, bits] b_bits
        # Generate [batch, bits, bits] partial products via broadcasting

        # Position-aware transformation (applied in parallel)
        self.position_transform = nn.Sequential(
            nn.Linear(max_bits * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_bits)
        )

        # Parallel reducer: takes all partial info and outputs result
        self.reducer = nn.Sequential(
            nn.Linear(max_bits * max_bits, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_bits)
        )

    def forward(self, a_bits, b_bits):
        """
        Fully parallel forward pass.
        """
        batch = a_bits.shape[0]
        bits = a_bits.shape[1]
        device = a_bits.device

        # Generate ALL partial products in parallel via broadcasting
        # a_bits: [batch, bits] -> [batch, bits, 1]
        # b_bits: [batch, bits] -> [batch, 1, bits]
        # Product: [batch, bits, bits] - all a[i] * b[j] combinations
        a_expanded = a_bits.unsqueeze(2)  # [batch, bits, 1]
        b_expanded = b_bits.unsqueeze(1)  # [batch, 1, bits]

        # Outer product: [batch, bits, bits]
        # partial[i,j] = a[i] * b[j]
        partials = a_expanded * b_expanded

        # Flatten and process through reducer (all parallel)
        flat = partials.view(batch, -1)  # [batch, bits*bits]

        output = self.reducer(flat)

        return output


class ParallelMulNetV3(nn.Module):
    """
    V3: Parallel convolution-based approach.

    Insight: Multiplication is like convolution of bit patterns!
    Use 1D convolutions which are inherently parallel.
    """
    def __init__(self, max_bits=64, hidden_channels=256):
        super().__init__()
        self.max_bits = max_bits

        # Embed inputs to channels
        self.embed_a = nn.Conv1d(1, hidden_channels, kernel_size=1)
        self.embed_b = nn.Conv1d(1, hidden_channels, kernel_size=1)

        # Cross-correlation layers (parallel via convolution)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels * 2, hidden_channels * 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_channels * 2, hidden_channels, kernel_size=5, padding=2),
            nn.GELU(),
        )

        # Global context (parallel pooling + broadcast)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_proj = nn.Linear(hidden_channels, hidden_channels)

        # Output projection
        self.output = nn.Conv1d(hidden_channels * 2, 1, kernel_size=1)

    def forward(self, a_bits, b_bits):
        batch = a_bits.shape[0]
        bits = a_bits.shape[1]

        # [batch, bits] -> [batch, 1, bits] for conv1d
        a = a_bits.unsqueeze(1)
        b = b_bits.unsqueeze(1)

        # Embed (parallel)
        a_emb = self.embed_a(a)  # [batch, channels, bits]
        b_emb = self.embed_b(b)

        # Concatenate along channel dim
        combined = torch.cat([a_emb, b_emb], dim=1)  # [batch, channels*2, bits]

        # Convolutional processing (parallel)
        features = self.conv_layers(combined)  # [batch, channels, bits]

        # Add global context (parallel)
        global_feat = self.global_pool(features)  # [batch, channels, 1]
        global_feat = self.global_proj(global_feat.squeeze(-1)).unsqueeze(-1)  # [batch, channels, 1]
        global_feat = global_feat.expand(-1, -1, bits)  # [batch, channels, bits]

        # Combine local and global
        combined_feat = torch.cat([features, global_feat], dim=1)  # [batch, channels*2, bits]

        # Output (parallel)
        output = self.output(combined_feat).squeeze(1)  # [batch, bits]

        return output


def generate_mul_data(batch_size, bits, device):
    """Generate multiplication training data."""
    max_val = 2 ** (bits // 2)
    a = torch.randint(0, max_val, (batch_size,), device=device)
    b = torch.randint(0, max_val, (batch_size,), device=device)
    result = a * b

    a_bits = ((a.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    b_bits = ((b.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    result_bits = ((result.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()

    return a_bits, b_bits, result_bits


def train(model, device, save_dir, max_bits=64, batch_size=2048, model_name="parallel"):
    print("\n" + "=" * 70)
    print(f"⚡ PARALLEL MUL TRAINING: {model.__class__.__name__}")
    print("=" * 70)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}")
    print("NO SEQUENTIAL LOOPS - True parallel computation!")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

    curriculum = [(8, 3000), (16, 4000), (32, 5000), (64, 6000)]
    curriculum = [(b, e) for (b, e) in curriculum if b <= max_bits]

    for bits, max_epochs in curriculum:
        print(f"\n{'=' * 70}")
        print(f"📊 Training: {bits}-bit MUL (PARALLEL)")
        print(f"{'=' * 70}")

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
        best_acc = 0
        consecutive_100 = 0

        for epoch in range(max_epochs):
            model.train()

            a_bits, b_bits, target = generate_mul_data(batch_size, bits, device)

            if bits < max_bits:
                a_bits = F.pad(a_bits, (0, max_bits - bits))
                b_bits = F.pad(b_bits, (0, max_bits - bits))
                target = F.pad(target, (0, max_bits - bits))

            optimizer.zero_grad()

            with autocast('cuda', dtype=torch.bfloat16):
                output = model(a_bits, b_bits)
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
                torch.save(model.state_dict(), f"{save_dir}/MUL_{bits}bit_{model_name}_ckpt.pt")

            if correct >= 100.0:
                consecutive_100 += 1
                if consecutive_100 >= 3:
                    print(f"    ✅ {bits}-bit: 100% x3 - ADVANCING!")
                    torch.save(model.state_dict(), f"{save_dir}/MUL_{bits}bit_{model_name}_100pct.pt")
                    break
            else:
                consecutive_100 = 0

        print(f"    Level complete: best={best_acc:.2f}%")

    print("\n" + "=" * 70)
    print("🎉 PARALLEL MUL TRAINING COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--bits', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--model', default='v2', choices=['transformer', 'v2', 'conv'])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.model == 'transformer':
        model = ParallelMulNet(max_bits=args.bits, d_model=256, n_heads=8, n_layers=4)
        model_name = "transformer"
    elif args.model == 'v2':
        model = ParallelMulNetV2(max_bits=args.bits, hidden_dim=512)
        model_name = "v2"
    else:
        model = ParallelMulNetV3(max_bits=args.bits, hidden_channels=256)
        model_name = "conv"

    model = model.to(args.device)

    train(model, args.device, args.save_dir, max_bits=args.bits,
          batch_size=args.batch_size, model_name=model_name)


if __name__ == '__main__':
    main()
