#!/usr/bin/env python3
"""
EXACT ASR (Arithmetic Shift Right) Training

ASR formula: output[i] = input[i + shift] if i + shift < bits else input[bits-1]
Unlike LSR which fills with zeros, ASR fills with the sign bit (MSB)
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


class DecomposedASRNet(nn.Module):
    """
    Decomposed network for Arithmetic Shift Right.

    Learns:
    1. Shift decoding (binary → one-hot)
    2. Index prediction (source position for each output)
    3. Fill prediction (use sign bit or normal index?)
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

        # Index predictor
        self.index_net = nn.Sequential(
            nn.Linear(max_bits * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_bits)
        )

        # Fill predictor: should this position use sign bit?
        self.fill_net = nn.Sequential(
            nn.Linear(max_bits * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
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

        # Get sign bit (MSB = last bit in our convention)
        sign_bit = a_bits[:, bits - 1:bits]  # [batch, 1]

        outputs = []
        index_logits_list = []
        fill_logits_list = []

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

            # Gather value from input
            pointed_value = (idx_soft * a_bits).sum(dim=1)

            # Predict if should use sign bit
            fill_logit = self.fill_net(combined).squeeze(-1)
            fill_logits_list.append(fill_logit)

            # Output: sign bit if fill, else pointed value
            use_sign = torch.sigmoid(fill_logit)
            output = pointed_value * (1 - use_sign) + sign_bit.squeeze(-1) * use_sign
            outputs.append(output)

        result = torch.stack(outputs, dim=1)

        if return_aux:
            return (result,
                    shift_logits,
                    torch.stack(index_logits_list, dim=1),
                    torch.stack(fill_logits_list, dim=1))
        return result


def generate_asr_data(batch_size, bits, device, max_shift):
    """Generate ASR training data."""
    a_bits = torch.randint(0, 2, (batch_size, bits), device=device).float()
    shifts = torch.randint(0, max_shift + 1, (batch_size,), device=device)
    positions = torch.arange(bits, device=device).unsqueeze(0)

    # ASR: output[i] = input[i + shift] if i + shift < bits else input[bits-1]
    src_pos = positions + shifts.unsqueeze(1)
    use_sign = src_pos >= bits  # Positions that should use sign bit
    src_pos_clamped = src_pos.clamp(0, bits - 1)

    # Get values from valid positions
    normal_values = torch.gather(a_bits, 1, src_pos_clamped)

    # Get sign bit (MSB)
    sign_bit = a_bits[:, bits - 1:bits].expand(-1, bits)

    # Result: normal value or sign bit
    result = torch.where(use_sign, sign_bit, normal_values)

    shift_bits = ((shifts.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()

    # Ground truth for supervision
    target_indices = src_pos_clamped
    target_fill = use_sign.float()
    target_shift = F.one_hot(shifts.clamp(0, bits - 1), bits).float()

    return a_bits, shift_bits, result, target_indices, target_fill, target_shift


def train(model, device, save_dir, max_bits=64, batch_size=4096):
    print("\n" + "=" * 70)
    print(f"🎯 EXACT ASR TRAINING")
    print("=" * 70)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scaler = GradScaler('cuda')

    output_loss_fn = nn.MSELoss()
    attn_loss_fn = nn.CrossEntropyLoss()
    fill_loss_fn = nn.BCEWithLogitsLoss()

    curriculum = [(8, 7), (16, 15), (24, 23), (32, 31), (40, 39), (48, 47), (56, 55), (64, 63)]
    curriculum = [(b, s) for (b, s) in curriculum if b <= max_bits]

    # Check for checkpoints
    start_level = 0
    for i, (bits, _) in enumerate(curriculum):
        ckpt_path = f"{save_dir}/ASR_{bits}bit_exact_100pct.pt"
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

            a_bits, shift_bits, target, target_indices, target_fill, target_shift = \
                generate_asr_data(batch_size, bits, device, max_shift)

            if bits < max_bits:
                a_bits = F.pad(a_bits, (0, max_bits - bits))
                shift_bits = F.pad(shift_bits, (0, max_bits - bits))
                target = F.pad(target, (0, max_bits - bits))
                target_fill = F.pad(target_fill, (0, max_bits - bits))

            optimizer.zero_grad()

            with autocast('cuda', dtype=torch.bfloat16):
                output, shift_logits, idx_logits, fill_logits = model(a_bits, shift_bits, return_aux=True)

                main_loss = output_loss_fn(output[:, :bits], target[:, :bits])
                shift_loss = attn_loss_fn(shift_logits, target_shift[:, :bits].argmax(dim=1))

                idx_loss = 0
                for i in range(bits):
                    idx_loss += attn_loss_fn(idx_logits[:, i, :], target_indices[:, i])
                idx_loss /= bits

                fill_loss = fill_loss_fn(fill_logits[:, :bits], target_fill[:, :bits])

                loss = main_loss + 0.5 * shift_loss + 0.3 * idx_loss + 0.2 * fill_loss

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
                torch.save(model.state_dict(), f"{save_dir}/ASR_{bits}bit_exact_ckpt.pt")

            if correct >= 100.0:
                consecutive_100 += 1
                if consecutive_100 >= 3:
                    print(f"    ✅ {bits}-bit: 100% x3 - ADVANCING!")
                    torch.save(model.state_dict(), f"{save_dir}/ASR_{bits}bit_exact_100pct.pt")
                    break
            else:
                consecutive_100 = 0

        print(f"    Level complete: best={best_acc:.2f}%")

    print("\n" + "=" * 70)
    print(f"🎉 ASR TRAINING COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--bits', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4096)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = DecomposedASRNet(max_bits=args.bits, hidden_dim=512)
    model = model.to(args.device)

    train(model, args.device, args.save_dir, max_bits=args.bits, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
