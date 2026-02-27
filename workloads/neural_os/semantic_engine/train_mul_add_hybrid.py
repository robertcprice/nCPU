#!/usr/bin/env python3
"""
PURE PARALLEL MUL using ADD model's carry propagation!

Key insight:
- ADD model already learned carry propagation across 64 bits (100% accurate!)
- Multiplication = parallel partial products + parallel diagonal sums + carry propagation
- We can use ADD model's transformer for the carry part!

Architecture (ZERO LOOPS in forward pass):
1. Outer product: a ⊗ b = all 4096 partial products in ONE einsum
2. Anti-diagonal extraction via precomputed sparse matrix multiply
3. ADD model's transformer for carry propagation (pre-trained!)
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


class AddCarryTransformer(nn.Module):
    """
    The EXACT architecture that achieved 100% on ADD.
    We load pre-trained weights for carry propagation.
    """
    def __init__(self, max_bits=64, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # Input projection (we'll adapt this for MUL)
        self.input_proj = nn.Linear(2, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.carry_head = nn.Linear(d_model, 1)


class PureParallelMulWithADD(nn.Module):
    """
    Pure parallel MUL using pre-trained ADD model for carry propagation.

    Forward pass - ZERO LOOPS:
    1. einsum('bi,bj->bij') - outer product (parallel)
    2. matmul with sparse extract matrix - diagonal sums (parallel)
    3. ADD model transformer - carry propagation (parallel, pre-trained!)
    4. Final output projection (parallel)
    """
    def __init__(self, max_bits=64, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # Build anti-diagonal extraction matrix [out_bits, max_bits*max_bits]
        # This converts 2D partial products to 1D diagonal sums in ONE matmul
        extract_mat = torch.zeros(max_bits, max_bits * max_bits)
        for k in range(max_bits):
            for i in range(max_bits):
                j = k - i
                if 0 <= j < max_bits:
                    extract_mat[k, i * max_bits + j] = 1.0
        self.register_buffer('extract', extract_mat)

        # Project diagonal sums to transformer input
        # Diagonal sums can be > 1 (e.g., up to 64 for bit 63)
        # We need to encode both the count AND whether it produces a carry
        self.diag_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )

        # Position embedding (same as ADD)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        # Transformer for carry propagation (same architecture as ADD)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head - predict final bit values
        self.output_head = nn.Linear(d_model, 1)

        # Register causal mask buffer
        causal = torch.triu(torch.ones(max_bits, max_bits), diagonal=1).bool()
        self.register_buffer('causal_mask', causal)

    def forward(self, a_bits, b_bits):
        batch = a_bits.shape[0]
        bits = a_bits.shape[1]
        device = a_bits.device

        # STEP 1 (PARALLEL): Outer product - ALL partial products at once!
        # pp[b,i,j] = a[b,i] * b[b,j]
        pp = torch.einsum('bi,bj->bij', a_bits, b_bits)  # [batch, bits, bits]
        pp_flat = pp.reshape(batch, -1)  # [batch, bits*bits]

        # STEP 2 (PARALLEL): Extract anti-diagonal sums via matrix multiply
        # diag_sums[k] = sum of pp[i,j] where i+j=k
        extract = self.extract[:bits, :bits*bits]  # Trim to actual size
        diag_sums = torch.matmul(pp_flat, extract.T)  # [batch, bits]

        # STEP 3 (PARALLEL): Project diagonal sums to transformer dim
        x = self.diag_proj(diag_sums.unsqueeze(-1))  # [batch, bits, d_model]
        x = x + self.pos_embedding[:, :bits, :]

        # STEP 4 (PARALLEL): Transformer with causal mask (same as ADD!)
        mask = self.causal_mask[:bits, :bits]
        x = self.transformer(x, mask=mask)

        # STEP 5 (PARALLEL): Output projection
        output = self.output_head(x).squeeze(-1)  # [batch, bits]

        return output

    def load_add_weights(self, add_checkpoint_path, strict=False):
        """Load pre-trained ADD model weights for transformer layers."""
        add_state = torch.load(add_checkpoint_path, map_location='cpu')
        own_state = self.state_dict()

        loaded = 0
        for name, param in add_state.items():
            # Match transformer and position embedding
            if 'transformer' in name or 'pos_embedding' in name:
                if name in own_state and own_state[name].shape == param.shape:
                    own_state[name] = param
                    loaded += 1
                    print(f"  Loaded: {name}")

        self.load_state_dict(own_state, strict=False)
        print(f"  Total loaded from ADD: {loaded} parameters")


class PureParallelMulWithADDv2(nn.Module):
    """
    V2: More expressive projection + deeper transformer.

    Key improvements:
    - Better encoding of diagonal sums (multi-bit counts)
    - Residual connections around projection
    - Option for larger model
    """
    def __init__(self, max_bits=64, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # Build anti-diagonal extraction matrix
        extract_mat = torch.zeros(max_bits, max_bits * max_bits)
        for k in range(max_bits):
            for i in range(max_bits):
                j = k - i
                if 0 <= j < max_bits:
                    extract_mat[k, i * max_bits + j] = 1.0
        self.register_buffer('extract', extract_mat)

        # Enhanced diagonal sum projection
        # The sum at position k can be 0 to k+1, representing that many 1-bits
        # This is like a multi-bit adder result!
        self.diag_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        # Transformer for carry propagation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head with more capacity
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        # Causal mask
        causal = torch.triu(torch.ones(max_bits, max_bits), diagonal=1).bool()
        self.register_buffer('causal_mask', causal)

    def forward(self, a_bits, b_bits):
        batch = a_bits.shape[0]
        bits = a_bits.shape[1]

        # PARALLEL: Outer product
        pp = torch.einsum('bi,bj->bij', a_bits, b_bits)
        pp_flat = pp.reshape(batch, -1)

        # PARALLEL: Anti-diagonal extraction
        extract = self.extract[:bits, :bits*bits]
        diag_sums = torch.matmul(pp_flat, extract.T)

        # PARALLEL: Project to transformer input
        # Normalize sums by max possible value at each position
        max_contributors = torch.arange(1, bits + 1, device=a_bits.device).float()
        max_contributors = torch.minimum(max_contributors, torch.tensor(bits, device=a_bits.device).float() - max_contributors + 1)
        max_contributors = torch.clamp(max_contributors, min=1)
        normalized_sums = diag_sums / max_contributors.unsqueeze(0)

        x = self.diag_proj(normalized_sums.unsqueeze(-1))
        x = x + self.pos_embedding[:, :bits, :]

        # PARALLEL: Transformer
        mask = self.causal_mask[:bits, :bits]
        x = self.transformer(x, mask=mask)

        # PARALLEL: Output
        output = self.output_head(x).squeeze(-1)

        return output


def generate_mul_data(batch_size, bits, device):
    """Generate multiplication training data."""
    max_val = 2 ** (bits // 2)  # Keep inputs small so result fits in bits
    a = torch.randint(0, max_val, (batch_size,), device=device)
    b = torch.randint(0, max_val, (batch_size,), device=device)
    result = a * b

    a_bits = ((a.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    b_bits = ((b.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    result_bits = ((result.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()

    return a_bits, b_bits, result_bits


def train(model, device, save_dir, add_model_path=None, max_bits=64, batch_size=2048, model_name="add_hybrid"):
    print("\n" + "=" * 70)
    print(f"PURE PARALLEL MUL + ADD CARRY: {model.__class__.__name__}")
    print("=" * 70)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}")
    print("ZERO LOOPS in forward pass - True parallel + ADD knowledge!")

    # Load ADD model weights if available
    if add_model_path and os.path.exists(add_model_path):
        print(f"\nLoading ADD model from: {add_model_path}")
        if hasattr(model, 'load_add_weights'):
            model.load_add_weights(add_model_path)
        else:
            # Try direct state dict load for compatible layers
            add_state = torch.load(add_model_path, map_location=device)
            own_state = model.state_dict()
            loaded = 0
            for name, param in add_state.items():
                if name in own_state and own_state[name].shape == param.shape:
                    own_state[name] = param
                    loaded += 1
            if loaded > 0:
                model.load_state_dict(own_state, strict=False)
                print(f"  Direct loaded: {loaded} parameters")

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

    # Progressive curriculum: +8 bits each level for smooth learning
    # INCREASED epochs for higher bit widths - they need more time!
    curriculum = [
        (8, 3000), (16, 3000), (24, 6000), (32, 8000),
        (40, 10000), (48, 12000), (56, 14000), (64, 16000)
    ]
    curriculum = [(b, e) for (b, e) in curriculum if b <= max_bits]

    for bits, max_epochs in curriculum:
        print(f"\n{'=' * 70}")
        print(f"Training: {bits}-bit MUL (PURE PARALLEL + ADD)")
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
    print("PURE PARALLEL MUL + ADD TRAINING COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--bits', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--model', default='v1', choices=['v1', 'v2', 'v3', 'v3.5', 'v4'])
    parser.add_argument('--add-model', default=None, help='Path to trained ADD model')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Find ADD model if not specified
    if args.add_model is None:
        candidates = [
            f"{args.save_dir}/ADD_64bit_100pct.pt",
            f"{args.save_dir}/ADD_64bit_checkpoint.pt",
        ]
        for c in candidates:
            if os.path.exists(c):
                args.add_model = c
                print(f"Found ADD model: {c}")
                break

    if args.model == 'v1':
        model = PureParallelMulWithADD(max_bits=args.bits, d_model=64, nhead=4, num_layers=3)
        model_name = "add_hybrid_v1"
    elif args.model == 'v2':
        model = PureParallelMulWithADDv2(max_bits=args.bits, d_model=128, nhead=8, num_layers=4)
        model_name = "add_hybrid_v2"
    elif args.model == 'v3':  # higher capacity for longer carry chains (~5M params)
        model = PureParallelMulWithADDv2(max_bits=args.bits, d_model=256, nhead=16, num_layers=6)
        model_name = "add_hybrid_v3"
    elif args.model == 'v3.5':  # sweet spot capacity (~10M params)
        model = PureParallelMulWithADDv2(max_bits=args.bits, d_model=384, nhead=16, num_layers=7)
        model_name = "add_hybrid_v3.5"
    else:  # v4 - MASSIVE capacity (25M+ params)
        model = PureParallelMulWithADDv2(max_bits=args.bits, d_model=512, nhead=16, num_layers=8)
        model_name = "add_hybrid_v4"

    model = model.to(args.device)

    train(model, args.device, args.save_dir, add_model_path=args.add_model,
          max_bits=args.bits, batch_size=args.batch_size, model_name=model_name)


if __name__ == '__main__':
    main()
