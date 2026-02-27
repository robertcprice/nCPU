#!/usr/bin/env python3
"""
APPROACH 3: 256×256 Lookup + Neural Combination

Split numbers into 8-bit chunks, use learned embeddings for chunk products,
then learn to combine them correctly.

For 64-bit: a = a7|a6|a5|a4|a3|a2|a1|a0 (8 chunks of 8 bits each)
            b = b7|b6|b5|b4|b3|b2|b1|b0

Product = Σ (ai × bj) << (8*(i+j)) for all i,j pairs

This reduces the problem to:
1. 8-bit × 8-bit multiplications (256×256 = 65536 possible results)
2. Combining partial products with correct shifts

The key insight: 8-bit MUL can be learned as a lookup table!
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


class ChunkMulLookup(nn.Module):
    """
    Learned lookup for 8-bit × 8-bit multiplication.
    Uses embeddings to represent each possible 8-bit value.
    OPTIMIZED: Supports batched multi-pair forward pass.
    """
    def __init__(self, chunk_bits=8, embed_dim=64):
        super().__init__()
        self.chunk_bits = chunk_bits
        self.num_values = 2 ** chunk_bits  # 256
        self.embed_dim = embed_dim

        # Embeddings for each possible input value
        self.embed_a = nn.Embedding(self.num_values, embed_dim)
        self.embed_b = nn.Embedding(self.num_values, embed_dim)

        # MLP to compute product from embeddings
        self.mul_net = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, chunk_bits * 2)  # Output is 16 bits (max product of 8-bit × 8-bit)
        )

    def forward(self, a_chunk, b_chunk):
        """
        a_chunk, b_chunk: [batch] or [batch, num_pairs] integers 0-255
        returns: [batch, 16] or [batch, num_pairs, 16] bits of product
        """
        emb_a = self.embed_a(a_chunk)  # [batch, embed_dim] or [batch, num_pairs, embed_dim]
        emb_b = self.embed_b(b_chunk)
        combined = torch.cat([emb_a, emb_b], dim=-1)  # concat on last dim
        return self.mul_net(combined)


class LookupMulNet(nn.Module):
    """
    64-bit multiplication using chunk-based lookup.

    Architecture:
    1. Split inputs into 8-bit chunks
    2. Compute all chunk products using learned lookup
    3. Combine with learned shift-and-add network
    """
    def __init__(self, max_bits=64, chunk_bits=8, hidden_dim=512):
        super().__init__()
        self.max_bits = max_bits
        self.chunk_bits = chunk_bits
        self.num_chunks = max_bits // chunk_bits  # 8 chunks for 64-bit

        # Chunk multiplication lookup
        self.chunk_mul = ChunkMulLookup(chunk_bits=chunk_bits, embed_dim=64)

        # Combiner: takes all chunk products and combines them
        # Number of chunk products: num_chunks × num_chunks = 64 for 64-bit
        # Each product is 16 bits
        num_products = self.num_chunks * self.num_chunks
        product_bits = chunk_bits * 2

        self.combiner = nn.Sequential(
            nn.Linear(num_products * product_bits, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_bits)
        )

    def forward(self, a_bits, b_bits):
        batch = a_bits.shape[0]
        bits = a_bits.shape[1]
        device = a_bits.device

        # Convert bits to chunks (8-bit integers) - VECTORIZED
        powers = 2 ** torch.arange(self.chunk_bits, device=device)
        a_chunks = []
        b_chunks = []
        for i in range(self.num_chunks):
            start = i * self.chunk_bits
            end = start + self.chunk_bits
            if end <= bits:
                chunk_a = (a_bits[:, start:end] * powers).sum(dim=1).long()
                chunk_b = (b_bits[:, start:end] * powers).sum(dim=1).long()
            else:
                chunk_a = torch.zeros(batch, dtype=torch.long, device=device)
                chunk_b = torch.zeros(batch, dtype=torch.long, device=device)
            a_chunks.append(chunk_a)
            b_chunks.append(chunk_b)

        # Stack chunks: [batch, num_chunks]
        a_stack = torch.stack(a_chunks, dim=1)  # [batch, 8]
        b_stack = torch.stack(b_chunks, dim=1)  # [batch, 8]

        # Create all pairs - BATCHED instead of 64 sequential calls
        # a_expanded: [batch, 64] where we repeat each a_chunk 8 times
        # b_expanded: [batch, 64] where we tile all b_chunks 8 times
        a_expanded = a_stack.repeat_interleave(self.num_chunks, dim=1)  # [batch, 64]
        b_expanded = b_stack.repeat(1, self.num_chunks)  # [batch, 64]

        # Single batched forward pass for ALL 64 chunk products!
        all_products = self.chunk_mul(a_expanded, b_expanded)  # [batch, 64, 16]
        all_products = all_products.view(batch, -1)  # [batch, 64*16 = 1024]

        # Combine to get final result
        output = self.combiner(all_products)

        return output


class SimpleLookupMul(nn.Module):
    """
    Simpler version: direct embedding lookup for the whole multiplication.
    Only works for small bit widths (8-bit max due to memory).
    """
    def __init__(self, max_bits=8, embed_dim=128):
        super().__init__()
        self.max_bits = max_bits
        self.num_values = 2 ** max_bits

        # Direct product embedding (only feasible for small bits)
        if max_bits <= 8:
            # For 8-bit: 256 × 256 = 65536 possible products
            self.product_embed = nn.Embedding(self.num_values * self.num_values, embed_dim)
            self.output_net = nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, max_bits * 2)  # Output up to 16 bits
            )
        else:
            raise ValueError(f"SimpleLookupMul only works for max_bits <= 8, got {max_bits}")

    def forward(self, a_bits, b_bits):
        batch = a_bits.shape[0]
        bits = min(a_bits.shape[1], self.max_bits)
        device = a_bits.device

        # Convert to integers
        a_int = (a_bits[:, :bits] * (2 ** torch.arange(bits, device=device))).sum(dim=1).long()
        b_int = (b_bits[:, :bits] * (2 ** torch.arange(bits, device=device))).sum(dim=1).long()

        # Combined index
        idx = a_int * self.num_values + b_int

        # Lookup and transform
        emb = self.product_embed(idx)
        output = self.output_net(emb)

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


def train(model, device, save_dir, max_bits=64, batch_size=2048):
    print("\n" + "=" * 70)
    print("🎯 APPROACH 3: LOOKUP + NEURAL COMBINATION MUL")
    print("=" * 70)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

    # For SimpleLookupMul, only train on small bits
    if isinstance(model, SimpleLookupMul):
        curriculum = [(8, 2000)]
    else:
        curriculum = [(8, 2000), (16, 3000), (32, 4000), (64, 5000)]
        curriculum = [(b, e) for (b, e) in curriculum if b <= max_bits]

    for bits, max_epochs in curriculum:
        print(f"\n{'=' * 70}")
        print(f"📊 Training: {bits}-bit MUL")
        print(f"{'=' * 70}")

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
        best_acc = 0
        consecutive_100 = 0

        for epoch in range(max_epochs):
            model.train()

            a_bits, b_bits, target = generate_mul_data(batch_size, bits, device)

            if bits < max_bits and not isinstance(model, SimpleLookupMul):
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
                torch.save(model.state_dict(), f"{save_dir}/MUL_{bits}bit_lookup_ckpt.pt")

            if correct >= 100.0:
                consecutive_100 += 1
                if consecutive_100 >= 3:
                    print(f"    ✅ {bits}-bit: 100% x3 - ADVANCING!")
                    torch.save(model.state_dict(), f"{save_dir}/MUL_{bits}bit_lookup_100pct.pt")
                    break
            else:
                consecutive_100 = 0

        print(f"    Level complete: best={best_acc:.2f}%")

    print("\n" + "=" * 70)
    print("🎉 LOOKUP MUL TRAINING COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--bits', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--model', default='chunk', choices=['chunk', 'simple'])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.model == 'chunk':
        model = LookupMulNet(max_bits=args.bits, chunk_bits=8, hidden_dim=512)
    else:
        model = SimpleLookupMul(max_bits=8, embed_dim=128)

    model = model.to(args.device)

    train(model, args.device, args.save_dir, max_bits=args.bits, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
