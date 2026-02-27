#!/usr/bin/env python3
"""
PURE PARALLEL MULTIPLICATION - Zero loops in forward pass!

Architecture (from deep research):
1. Outer product: a ⊗ b gives all 4096 partial products in ONE matmul
2. Anti-diagonal extraction via precomputed sparse matrix multiply
3. Learned carry propagation via MLP (no sequential recurrence!)

Key insight: Carry propagation can be LEARNED in parallel, not computed sequentially,
if we give the network enough capacity and the right inductive bias.
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


class PureParallelMul(nn.Module):
    """
    Fully parallel 64-bit multiplication - NO LOOPS!

    Forward pass operations:
    1. einsum('bi,bj->bij') - outer product (parallel)
    2. matmul with sparse extract matrix - diagonal extraction (parallel)
    3. MLP layers - carry propagation (parallel)
    """
    def __init__(self, max_bits=64, hidden_dim=512):
        super().__init__()
        self.max_bits = max_bits
        out_bits = max_bits  # Result fits in max_bits for half-width inputs

        # Build anti-diagonal extraction matrix [out_bits, max_bits*max_bits]
        # This converts the 2D partial product matrix to 1D diagonal sums
        extract_mat = torch.zeros(out_bits, max_bits * max_bits)
        for k in range(out_bits):
            for i in range(max_bits):
                j = k - i
                if 0 <= j < max_bits:
                    extract_mat[k, i * max_bits + j] = 1.0
        self.register_buffer('extract', extract_mat)

        # Deep MLP for learning carry propagation pattern
        # Key: must have enough depth and width to learn 64-bit carry chains
        self.carry_net = nn.Sequential(
            nn.Linear(out_bits, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Position encoding for bits (helps with carry structure)
        pos = torch.arange(out_bits).float() / out_bits
        self.register_buffer('pos_encoding', pos)

        # Final output combining diagonal sums + carry info
        self.output = nn.Sequential(
            nn.Linear(hidden_dim + out_bits + out_bits, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_bits)
        )

    def forward(self, a_bits, b_bits):
        batch = a_bits.shape[0]
        bits = a_bits.shape[1]

        # STEP 1 (PARALLEL): Outer product - all partial products at once!
        # pp[b,i,j] = a[b,i] * b[b,j]
        pp = torch.einsum('bi,bj->bij', a_bits, b_bits)  # [batch, bits, bits]
        pp_flat = pp.reshape(batch, -1)  # [batch, bits*bits]

        # STEP 2 (PARALLEL): Extract anti-diagonal sums via matrix multiply
        # diag_sums[k] = sum of pp[i,j] where i+j=k
        extract = self.extract[:bits, :bits*bits]  # Trim to actual size
        diag_sums = torch.matmul(pp_flat, extract.T)  # [batch, bits]

        # STEP 3 (PARALLEL): Learn carry propagation
        carry_features = self.carry_net(diag_sums)  # [batch, hidden_dim]

        # Add positional encoding (helps network understand bit positions)
        pos = self.pos_encoding[:bits].unsqueeze(0).expand(batch, -1)

        # STEP 4 (PARALLEL): Combine and predict all output bits
        combined = torch.cat([carry_features, diag_sums, pos], dim=1)
        output = self.output(combined)

        return output


class PureParallelMulV2(nn.Module):
    """
    V2: Adds residual connections and more structure.

    Key improvements:
    - Residual connections help gradient flow for deep carry learning
    - Separate processing for low and high bits (different carry patterns)
    - Multi-scale feature extraction
    """
    def __init__(self, max_bits=64, hidden_dim=512):
        super().__init__()
        self.max_bits = max_bits

        # Anti-diagonal extraction matrix
        extract_mat = torch.zeros(max_bits, max_bits * max_bits)
        for k in range(max_bits):
            for i in range(max_bits):
                j = k - i
                if 0 <= j < max_bits:
                    extract_mat[k, i * max_bits + j] = 1.0
        self.register_buffer('extract', extract_mat)

        # Input encoding
        self.input_proj = nn.Linear(max_bits, hidden_dim)

        # Residual blocks for carry learning
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(6)
        ])

        # Multi-scale processing (different receptive fields for carries)
        self.scale1 = nn.Linear(hidden_dim, hidden_dim // 4)
        self.scale2 = nn.Linear(hidden_dim, hidden_dim // 4)
        self.scale4 = nn.Linear(hidden_dim, hidden_dim // 4)
        self.scale8 = nn.Linear(hidden_dim, hidden_dim // 4)

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_bits)
        )

    def forward(self, a_bits, b_bits):
        batch = a_bits.shape[0]
        bits = a_bits.shape[1]

        # PARALLEL: Outer product
        pp = torch.einsum('bi,bj->bij', a_bits, b_bits)
        pp_flat = pp.reshape(batch, -1)

        # PARALLEL: Extract diagonal sums
        extract = self.extract[:bits, :bits*bits]
        diag_sums = torch.matmul(pp_flat, extract.T)

        # PARALLEL: Project to hidden dim
        x = self.input_proj(F.pad(diag_sums, (0, self.max_bits - bits)))

        # PARALLEL: Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # PARALLEL: Multi-scale features (captures different carry distances)
        s1 = self.scale1(x)
        s2 = self.scale2(F.avg_pool1d(x.unsqueeze(1), 2, 2, 0).squeeze(1).repeat(1, 2)[:, :x.shape[1]]) if x.shape[1] > 1 else self.scale2(x)
        s4 = self.scale4(x)  # Simplified
        s8 = self.scale8(x)  # Simplified

        multi = torch.cat([s1, s2, s4, s8], dim=1)

        # PARALLEL: Output
        output = self.output(multi)

        return output[:, :bits]


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        return x + self.net(x)


class PureParallelMulV3(nn.Module):
    """
    V3: Most aggressive - uses attention-style mixing between bit positions.

    Key insight: Carry propagation is about bits "communicating" with each other.
    Self-attention (without loops) naturally handles this!
    """
    def __init__(self, max_bits=64, d_model=128, n_heads=8):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # Anti-diagonal extraction
        extract_mat = torch.zeros(max_bits, max_bits * max_bits)
        for k in range(max_bits):
            for i in range(max_bits):
                j = k - i
                if 0 <= j < max_bits:
                    extract_mat[k, i * max_bits + j] = 1.0
        self.register_buffer('extract', extract_mat)

        # Project diagonal sums to embedding space
        self.proj_in = nn.Linear(1, d_model)

        # Position encoding
        self.pos_embed = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        # Self-attention layers (ALL PARALLEL via matrix operations)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(4)
        ])
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            )
            for _ in range(4)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(8)
        ])

        # Causal mask for carry propagation (bit k can see bits 0..k-1)
        causal = torch.triu(torch.ones(max_bits, max_bits), diagonal=1).bool()
        self.register_buffer('causal_mask', causal)

        # Output projection
        self.proj_out = nn.Linear(d_model, 1)

    def forward(self, a_bits, b_bits):
        batch = a_bits.shape[0]
        bits = a_bits.shape[1]

        # PARALLEL: Outer product
        pp = torch.einsum('bi,bj->bij', a_bits, b_bits)
        pp_flat = pp.reshape(batch, -1)

        # PARALLEL: Extract diagonal sums
        extract = self.extract[:bits, :bits*bits]
        diag_sums = torch.matmul(pp_flat, extract.T)  # [batch, bits]

        # PARALLEL: Project each position to embedding
        x = self.proj_in(diag_sums.unsqueeze(-1))  # [batch, bits, d_model]
        x = x + self.pos_embed[:, :bits, :]

        # PARALLEL: Self-attention with causal mask
        mask = self.causal_mask[:bits, :bits]
        for i in range(4):
            # Self-attention (parallel over all positions!)
            attn_out, _ = self.attn_layers[i](x, x, x, attn_mask=mask)
            x = self.norms[2*i](x + attn_out)
            x = self.norms[2*i+1](x + self.ff_layers[i](x))

        # PARALLEL: Project to output
        output = self.proj_out(x).squeeze(-1)  # [batch, bits]

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


def train(model, device, save_dir, max_bits=64, batch_size=2048, model_name="pure"):
    print("\n" + "=" * 70)
    print(f"⚡ PURE PARALLEL MUL: {model.__class__.__name__}")
    print("=" * 70)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}")
    print("ZERO LOOPS in forward pass - True parallel computation!")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

    curriculum = [(8, 3000), (16, 4000), (32, 5000), (64, 6000)]
    curriculum = [(b, e) for (b, e) in curriculum if b <= max_bits]

    for bits, max_epochs in curriculum:
        print(f"\n{'=' * 70}")
        print(f"📊 Training: {bits}-bit MUL (PURE PARALLEL)")
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
    print("🎉 PURE PARALLEL MUL TRAINING COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--bits', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--model', default='v1', choices=['v1', 'v2', 'v3'])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.model == 'v1':
        model = PureParallelMul(max_bits=args.bits, hidden_dim=512)
        model_name = "pure_v1"
    elif args.model == 'v2':
        model = PureParallelMulV2(max_bits=args.bits, hidden_dim=512)
        model_name = "pure_v2"
    else:
        model = PureParallelMulV3(max_bits=args.bits, d_model=128, n_heads=8)
        model_name = "pure_v3"

    model = model.to(args.device)

    train(model, args.device, args.save_dir, max_bits=args.bits,
          batch_size=args.batch_size, model_name=model_name)


if __name__ == '__main__':
    main()
