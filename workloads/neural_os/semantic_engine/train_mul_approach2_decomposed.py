#!/usr/bin/env python3
"""
APPROACH 2: Decomposed Neural MUL

Structure is built-in (shift-and-add), but everything is learned from scratch.
Uses explicit supervision on intermediate steps.

MUL = Σ (a << i) × b[i]

Components (all learned):
1. Shift generator: learns to compute a << i for each i
2. Bit masker: learns to multiply by b[i]
3. Parallel adder: learns to sum all partial products
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


class DecomposedMulNet(nn.Module):
    """
    Decomposed multiplication with supervised intermediate steps.

    Architecture mirrors the shift-and-add algorithm:
    1. Generate all shifts: a<<0, a<<1, ..., a<<(bits-1)
    2. Mask each shift by corresponding bit of b
    3. Sum all masked partial products

    Each step is supervised during training.
    """
    def __init__(self, max_bits=64, hidden_dim=256):
        super().__init__()
        self.max_bits = max_bits
        self.hidden_dim = hidden_dim

        # Shift generator: for each position i, generate a << i
        # Input: a_bits + position_one_hot
        # Output: shifted bits
        self.shift_net = nn.Sequential(
            nn.Linear(max_bits + max_bits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_bits)
        )

        # Partial product summer: sum all partial products
        # Uses tree-like structure for better gradient flow
        # Input: all partial products [bits, bits]
        # Output: sum [bits]
        self.sum_layers = nn.ModuleList()
        current_inputs = max_bits
        while current_inputs > 1:
            next_inputs = (current_inputs + 1) // 2
            self.sum_layers.append(nn.Sequential(
                nn.Linear(max_bits * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, max_bits)
            ))
            current_inputs = next_inputs

        # Final output layer
        self.output_net = nn.Sequential(
            nn.Linear(max_bits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_bits)
        )

        self.register_buffer('temperature', torch.tensor(1.0))

    def set_temperature(self, temp):
        self.temperature.fill_(temp)

    def forward(self, a_bits, b_bits, return_aux=False):
        batch = a_bits.shape[0]
        bits = a_bits.shape[1]
        device = a_bits.device

        # Generate all partial products
        partial_products = []
        shift_logits_all = []

        for i in range(bits):
            # One-hot position encoding
            pos_one_hot = torch.zeros(batch, self.max_bits, device=device)
            pos_one_hot[:, i] = 1.0

            # Generate a << i
            shift_input = torch.cat([a_bits, pos_one_hot], dim=1)
            shift_logits = self.shift_net(shift_input)[:, :bits]
            shift_logits_all.append(shift_logits)

            # Soft or hard selection
            if self.training:
                shifted = torch.sigmoid(shift_logits / self.temperature)
            else:
                shifted = (shift_logits > 0).float()

            # Mask by b[i]
            partial = shifted * b_bits[:, i:i+1]
            partial_products.append(partial)

        # Stack partial products: [batch, bits, bits]
        partials = torch.stack(partial_products, dim=1)

        # Tree-like summation
        current = partials
        for layer in self.sum_layers:
            n = current.shape[1]
            if n == 1:
                current = current.squeeze(1)
                break

            # Pair up and sum
            new_partials = []
            for j in range(0, n, 2):
                if j + 1 < n:
                    pair = torch.cat([current[:, j], current[:, j+1]], dim=1)
                else:
                    pair = torch.cat([current[:, j], torch.zeros_like(current[:, j])], dim=1)
                summed = layer(pair)
                new_partials.append(summed)

            current = torch.stack(new_partials, dim=1)

        if len(current.shape) == 3:
            current = current.squeeze(1)

        # Final output
        output = self.output_net(current)

        if return_aux:
            return output, torch.stack(shift_logits_all, dim=1)
        return output


class SimplerDecomposedMul(nn.Module):
    """
    Simpler version: just learn the partial products and sum them.
    """
    def __init__(self, max_bits=64, hidden_dim=512):
        super().__init__()
        self.max_bits = max_bits

        # Learn each partial product: a << i for position i
        self.partial_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(max_bits, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, max_bits)
            )
            for _ in range(max_bits)
        ])

        # Learn the final summation
        self.sum_net = nn.Sequential(
            nn.Linear(max_bits * max_bits, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_bits)
        )

    def forward(self, a_bits, b_bits, return_aux=False):
        batch = a_bits.shape[0]
        bits = a_bits.shape[1]
        device = a_bits.device

        partial_products = []
        shift_logits_all = []
        for i in range(bits):
            # Generate a << i using learned network
            logits = self.partial_nets[i](a_bits)[:, :bits]
            shift_logits_all.append(logits)
            shifted = torch.sigmoid(logits)
            # Mask by b[i]
            partial = shifted * b_bits[:, i:i+1]
            partial_products.append(partial)

        # Concatenate all partials
        all_partials = torch.cat(partial_products, dim=1)  # [batch, bits*bits]

        # Learn the sum
        output = self.sum_net(all_partials)

        if return_aux:
            return output, torch.stack(shift_logits_all, dim=1)
        return output


def generate_mul_data(batch_size, bits, device):
    """Generate multiplication data with ground truth for supervision."""
    max_val = 2 ** (bits // 2)
    a = torch.randint(0, max_val, (batch_size,), device=device)
    b = torch.randint(0, max_val, (batch_size,), device=device)
    result = a * b

    a_bits = ((a.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    b_bits = ((b.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
    result_bits = ((result.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()

    # Ground truth for partial products: (a << i) for each i
    target_shifts = []
    for i in range(bits):
        shifted = a << i
        shifted_bits = ((shifted.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()
        target_shifts.append(shifted_bits)
    target_shifts = torch.stack(target_shifts, dim=1)  # [batch, bits, bits]

    return a_bits, b_bits, result_bits, target_shifts


def train(model, device, save_dir, max_bits=64, batch_size=2048):
    print("\n" + "=" * 70)
    print("🧠 APPROACH 2: DECOMPOSED NEURAL MUL")
    print("=" * 70)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scaler = GradScaler('cuda')

    output_loss_fn = nn.BCEWithLogitsLoss()
    shift_loss_fn = nn.BCEWithLogitsLoss()

    curriculum = [(8, 3000), (16, 4000), (32, 5000), (64, 6000)]
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

            if hasattr(model, 'set_temperature'):
                temp = max(0.1, 1.0 - epoch / 1000)
                model.set_temperature(temp)

            a_bits, b_bits, target, target_shifts = generate_mul_data(batch_size, bits, device)

            if bits < max_bits:
                a_bits = F.pad(a_bits, (0, max_bits - bits))
                b_bits = F.pad(b_bits, (0, max_bits - bits))
                target = F.pad(target, (0, max_bits - bits))

            optimizer.zero_grad()

            with autocast('cuda', dtype=torch.bfloat16):
                if hasattr(model, 'forward') and 'return_aux' in model.forward.__code__.co_varnames:
                    output, shift_logits = model(a_bits, b_bits, return_aux=True)

                    # Main output loss
                    main_loss = output_loss_fn(output[:, :bits], target[:, :bits])

                    # Shift supervision loss (for each position)
                    shift_loss = 0
                    for i in range(bits):
                        shift_loss += shift_loss_fn(shift_logits[:, i, :bits], target_shifts[:, i, :bits])
                    shift_loss /= bits

                    loss = main_loss + 0.3 * shift_loss
                else:
                    output = model(a_bits, b_bits)
                    loss = output_loss_fn(output[:, :bits], target[:, :bits])

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
                torch.save(model.state_dict(), f"{save_dir}/MUL_{bits}bit_decomposed_ckpt.pt")

            if correct >= 100.0:
                consecutive_100 += 1
                if consecutive_100 >= 3:
                    print(f"    ✅ {bits}-bit: 100% x3 - ADVANCING!")
                    torch.save(model.state_dict(), f"{save_dir}/MUL_{bits}bit_decomposed_100pct.pt")
                    break
            else:
                consecutive_100 = 0

        print(f"    Level complete: best={best_acc:.2f}%")

    print("\n" + "=" * 70)
    print("🎉 DECOMPOSED MUL TRAINING COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--bits', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--model', default='simpler', choices=['decomposed', 'simpler'])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.model == 'decomposed':
        model = DecomposedMulNet(max_bits=args.bits, hidden_dim=256)
    else:
        model = SimplerDecomposedMul(max_bits=args.bits, hidden_dim=512)

    model = model.to(args.device)

    train(model, args.device, args.save_dir, max_bits=args.bits, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
