#!/usr/bin/env python3
"""
APPROACH 1: Hybrid Frozen Components MUL

Uses pre-trained LSL model as a frozen component.
MUL = Σ (a << i) × b[i] for all bit positions i

The shift operations are done by our trained LSL model (frozen).
Only the combination/accumulation is learned.
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


class FrozenLSLModel(nn.Module):
    """Load and freeze our trained LSL model."""
    def __init__(self, checkpoint_path, max_bits=64, hidden_dim=768):  # Match checkpoint!
        super().__init__()
        self.max_bits = max_bits

        # EXACT architecture from checkpoint (no LayerNorm!)
        # shift_decoder: Linear(64,768) → ReLU → Linear(768,768) → ReLU → Linear(768,64)
        self.shift_decoder = nn.Sequential(
            nn.Linear(max_bits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_bits)
        )

        # index_net: Linear(128,768) → ReLU → Linear(768,768) → ReLU → Linear(768,64)
        self.index_net = nn.Sequential(
            nn.Linear(max_bits * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_bits)
        )

        # validity_net: Linear(128,384) → ReLU → Linear(384,1)
        self.validity_net = nn.Sequential(
            nn.Linear(max_bits * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.register_buffer('temperature', torch.tensor(0.01))

        # Load checkpoint if exists
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location='cpu')
            self.load_state_dict(state, strict=False)
            print(f"✅ Loaded LSL checkpoint: {checkpoint_path}")

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, a_bits, shift_amount):
        """Perform LSL: a << shift_amount"""
        batch = a_bits.shape[0]
        bits = a_bits.shape[1]
        device = a_bits.device

        # Create shift_bits from integer shift_amount
        shift_bits = ((shift_amount.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()

        # Decode shift
        shift_logits = self.shift_decoder(shift_bits)[:, :bits]
        shift_soft = F.one_hot(shift_logits.argmax(dim=-1), bits).float()

        outputs = []
        for i in range(bits):
            out_pos = torch.zeros(batch, self.max_bits, device=device)
            out_pos[:, i] = 1.0
            combined = torch.cat([out_pos[:, :bits], shift_soft], dim=1)

            idx_logits = self.index_net(combined)[:, :bits]
            idx_soft = F.one_hot(idx_logits.argmax(dim=-1), bits).float()

            pointed_value = (idx_soft * a_bits).sum(dim=1)
            valid_logit = self.validity_net(combined).squeeze(-1)

            output = pointed_value * torch.sigmoid(valid_logit)
            outputs.append(output)

        return torch.stack(outputs, dim=1)


class HybridMulNet(nn.Module):
    """
    MUL using frozen LSL + learned accumulator.

    For each bit i of b:
        partial[i] = LSL(a, i) * b[i]
    result = learned_sum(partial[0], partial[1], ...)
    """
    def __init__(self, lsl_checkpoint, max_bits=64, hidden_dim=512):
        super().__init__()
        self.max_bits = max_bits

        # Frozen LSL model (always use 768 to match checkpoint!)
        self.lsl = FrozenLSLModel(lsl_checkpoint, max_bits, hidden_dim=768)

        # Learned accumulator: takes all partial products and outputs sum
        # Input: bits * bits (all partial products flattened)
        # Output: bits (the product)
        self.accumulator = nn.Sequential(
            nn.Linear(max_bits * max_bits, hidden_dim * 2),
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

        # Generate all partial products using frozen LSL
        partial_products = []
        for i in range(bits):
            shift_amount = torch.full((batch,), i, device=device, dtype=torch.long)
            shifted = self.lsl(a_bits, shift_amount)  # a << i
            masked = shifted * b_bits[:, i:i+1]  # (a << i) * b[i]
            partial_products.append(masked)

        # Stack and flatten: [batch, bits, bits] -> [batch, bits*bits]
        all_partials = torch.stack(partial_products, dim=1)  # [batch, bits, bits]
        flattened = all_partials.view(batch, -1)  # [batch, bits*bits]

        # Learned accumulation
        logits = self.accumulator(flattened)
        return logits


def generate_mul_data(batch_size, bits, device):
    """Generate multiplication training data."""
    # Use smaller numbers to avoid overflow
    max_val = 2 ** (bits // 2)
    a = torch.randint(0, max_val, (batch_size,), device=device)
    b = torch.randint(0, max_val, (batch_size,), device=device)
    result = a * b

    # Convert to bits
    a_bits = ((a.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    b_bits = ((b.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    result_bits = ((result.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()

    return a_bits, b_bits, result_bits


def train(model, device, save_dir, max_bits=64, batch_size=1024):
    print("\n" + "=" * 70)
    print("🔧 APPROACH 1: HYBRID MUL (Frozen LSL + Learned Accumulator)")
    print("=" * 70)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Batch size: {batch_size}")

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

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

            if bits < max_bits:
                a_bits = F.pad(a_bits, (0, max_bits - bits))
                b_bits = F.pad(b_bits, (0, max_bits - bits))
                target = F.pad(target, (0, max_bits - bits))

            optimizer.zero_grad()

            with autocast('cuda', dtype=torch.bfloat16):
                logits = model(a_bits, b_bits)
                loss = criterion(logits[:, :bits], target[:, :bits])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                pred = (torch.sigmoid(logits[:, :bits]) > 0.5).float()
                correct = (pred == target[:, :bits]).all(dim=1).float().mean().item() * 100

            if epoch % 100 == 0 or correct > best_acc:
                print(f"    Epoch {epoch:4d}: loss={loss.item():.4f}, acc={correct:.2f}%")

            if correct > best_acc:
                best_acc = correct
                torch.save(model.state_dict(), f"{save_dir}/MUL_{bits}bit_hybrid_ckpt.pt")

            if correct >= 100.0:
                consecutive_100 += 1
                if consecutive_100 >= 3:
                    print(f"    ✅ {bits}-bit: 100% x3 - ADVANCING!")
                    torch.save(model.state_dict(), f"{save_dir}/MUL_{bits}bit_hybrid_100pct.pt")
                    break
            else:
                consecutive_100 = 0

        print(f"    Level complete: best={best_acc:.2f}%")

    print("\n" + "=" * 70)
    print("🎉 HYBRID MUL TRAINING COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--bits', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lsl-checkpoint', default='models/LSL_64bit_exact_100pct.pt')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = HybridMulNet(args.lsl_checkpoint, max_bits=args.bits, hidden_dim=512)
    model = model.to(args.device)

    train(model, args.device, args.save_dir, max_bits=args.bits, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
