#!/usr/bin/env python3
"""
EXACT ROL/ROR (Rotate) Training

ROL formula: output[i] = input[(i - shift) % bits]
ROR formula: output[i] = input[(i + shift) % bits]

Rotates are simpler than shifts - no validity check needed, bits just wrap around!
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


class DecomposedRotateNet(nn.Module):
    """
    Decomposed network for Rotate operations.

    Learns:
    1. Shift decoding (binary → one-hot)
    2. Index prediction (source position for each output, with wraparound)

    No validity check needed - all positions are valid!
    """
    def __init__(self, max_bits=64, hidden_dim=512):
        super().__init__()
        self.max_bits = max_bits

        # Shift decoder
        self.shift_decoder = nn.Sequential(
            nn.Linear(max_bits, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_bits)
        )

        # Index predictor (handles modular arithmetic)
        self.index_net = nn.Sequential(
            nn.Linear(max_bits * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_bits)
        )

        self.register_buffer('temperature', torch.tensor(1.0))

    def set_temperature(self, temp):
        self.temperature.fill_(temp)

    def forward(self, a_bits, shift_bits, return_aux=False):
        batch = a_bits.shape[0]
        bits = a_bits.shape[1]
        device = a_bits.device

        # Decode shift
        shift_logits = self.shift_decoder(shift_bits)[:, :bits]
        if self.training:
            shift_soft = F.softmax(shift_logits / self.temperature, dim=-1)
        else:
            shift_soft = F.one_hot(shift_logits.argmax(dim=-1), bits).float()

        outputs = []
        index_logits_list = []

        for i in range(bits):
            out_pos = torch.zeros(batch, self.max_bits, device=device)
            out_pos[:, i] = 1.0

            combined = torch.cat([out_pos[:, :bits], shift_soft], dim=1)

            # Predict source index
            idx_logits = self.index_net(combined)[:, :bits]
            index_logits_list.append(idx_logits)

            if self.training:
                idx_soft = F.softmax(idx_logits / self.temperature, dim=-1)
            else:
                idx_soft = F.one_hot(idx_logits.argmax(dim=-1), bits).float()

            # Gather value - no validity mask needed for rotates!
            output = (idx_soft * a_bits).sum(dim=1)
            outputs.append(output)

        result = torch.stack(outputs, dim=1)

        if return_aux:
            return (result, shift_logits, torch.stack(index_logits_list, dim=1))
        return result


def generate_rotate_data(batch_size, bits, device, max_shift, op='ROL'):
    """Generate rotate training data."""
    a_bits = torch.randint(0, 2, (batch_size, bits), device=device).float()
    shifts = torch.randint(0, max_shift + 1, (batch_size,), device=device)
    positions = torch.arange(bits, device=device).unsqueeze(0)

    if op == 'ROL':
        # ROL: output[i] = input[(i - shift) % bits]
        src_pos = (positions - shifts.unsqueeze(1)) % bits
    else:  # ROR
        # ROR: output[i] = input[(i + shift) % bits]
        src_pos = (positions + shifts.unsqueeze(1)) % bits

    result = torch.gather(a_bits, 1, src_pos)
    shift_bits = ((shifts.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()

    target_indices = src_pos
    target_shift = F.one_hot(shifts.clamp(0, bits - 1), bits).float()

    return a_bits, shift_bits, result, target_indices, target_shift


def train(model, device, save_dir, max_bits=64, batch_size=4096, op='ROL'):
    print("\n" + "=" * 70)
    print(f"🎯 EXACT {op} TRAINING")
    print("=" * 70)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scaler = GradScaler('cuda')

    output_loss_fn = nn.MSELoss()
    attn_loss_fn = nn.CrossEntropyLoss()

    curriculum = [(8, 7), (16, 15), (24, 23), (32, 31), (40, 39), (48, 47), (56, 55), (64, 63)]
    curriculum = [(b, s) for (b, s) in curriculum if b <= max_bits]

    # Check for checkpoints
    start_level = 0
    for i, (bits, _) in enumerate(curriculum):
        ckpt_path = f"{save_dir}/{op}_{bits}bit_exact_100pct.pt"
        if os.path.exists(ckpt_path):
            print(f"✅ Found 100% checkpoint: {ckpt_path}")
            start_level = i + 1
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            break

    if start_level > 0:
        print(f"🚀 Resuming from level {start_level + 1}/{len(curriculum)}")
        curriculum = curriculum[start_level:]

    for bits, max_shift in curriculum:
        print(f"\n{'=' * 70}")
        print(f"📊 Training: {bits}-bit, shifts 0-{max_shift}")
        print(f"{'=' * 70}")

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000, eta_min=1e-6)
        best_acc = 0
        consecutive_100 = 0

        for epoch in range(3000):
            model.train()
            temp = max(0.1, 1.0 - epoch / 1000)
            model.set_temperature(temp)

            a_bits, shift_bits, target, target_indices, target_shift = \
                generate_rotate_data(batch_size, bits, device, max_shift, op)

            if bits < max_bits:
                a_bits = F.pad(a_bits, (0, max_bits - bits))
                shift_bits = F.pad(shift_bits, (0, max_bits - bits))
                target = F.pad(target, (0, max_bits - bits))

            optimizer.zero_grad()

            with autocast('cuda', dtype=torch.bfloat16):
                output, shift_logits, idx_logits = model(a_bits, shift_bits, return_aux=True)

                main_loss = output_loss_fn(output[:, :bits], target[:, :bits])
                shift_loss = attn_loss_fn(shift_logits, target_shift[:, :bits].argmax(dim=1))

                idx_loss = 0
                for i in range(bits):
                    idx_loss += attn_loss_fn(idx_logits[:, i, :], target_indices[:, i])
                idx_loss /= bits

                loss = main_loss + 0.5 * shift_loss + 0.3 * idx_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                model.eval()
                model.set_temperature(0.01)
                eval_output = model(a_bits, shift_bits)
                pred = (eval_output[:, :bits] > 0.5).float()
                correct = (pred == target[:, :bits]).all(dim=1).float().mean().item() * 100

            if epoch % 100 == 0 or correct > best_acc:
                print(f"    Epoch {epoch:4d}: loss={loss.item():.4f}, acc={correct:.2f}%, temp={temp:.3f}")

            if correct > best_acc:
                best_acc = correct
                torch.save(model.state_dict(), f"{save_dir}/{op}_{bits}bit_exact_ckpt.pt")

            if correct >= 100.0:
                consecutive_100 += 1
                if consecutive_100 >= 3:
                    print(f"    ✅ {bits}-bit: 100% x3 - ADVANCING!")
                    torch.save(model.state_dict(), f"{save_dir}/{op}_{bits}bit_exact_100pct.pt")
                    break
            else:
                consecutive_100 = 0

        print(f"    Level complete: best={best_acc:.2f}%")

    print("\n" + "=" * 70)
    print(f"🎉 {op} TRAINING COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--bits', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--op', default='ROL', choices=['ROL', 'ROR'])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = DecomposedRotateNet(max_bits=args.bits, hidden_dim=512)
    model = model.to(args.device)

    train(model, args.device, args.save_dir, max_bits=args.bits, batch_size=args.batch_size, op=args.op)


if __name__ == '__main__':
    main()
