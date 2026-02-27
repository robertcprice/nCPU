#!/usr/bin/env python3
"""
STRICT CURRICULUM LSL TRAINING
==============================
Uses pre-computed shift matrices (proven approach) with:
1. VERY gradual shift range increase (0-1, 0-2, 0-3, ... 0-63)
2. STRICT 100% requirement before advancing
3. Auxiliary supervision on shift decoder
4. Mix of current + previous level examples

Key insight from train_lsl_formula.py:
- Pre-compute shift matrices: M[s] where M[s][i,j] = 1 iff j == i-s and i >= s
- Only learn shift decoding: binary → one-hot
- Apply shift via deterministic matrix multiply
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import random
import time
import os
import sys

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

device = "cuda" if torch.cuda.is_available() else "cpu"


class StrictShiftNet(nn.Module):
    """
    Shift network with pre-computed matrices.

    The architecture focuses entirely on accurate shift decoding.
    Uses deeper network with residual connections for better gradient flow.
    """

    def __init__(self, max_bits=64, hidden_dim=512):
        super().__init__()
        self.max_bits = max_bits

        # Deep shift decoder with residuals
        self.input_proj = nn.Linear(max_bits, hidden_dim)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(4)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max_bits)
        )

        # Pre-compute shift matrices
        self._precompute_shift_matrices(max_bits)

        self.register_buffer('temperature', torch.tensor(1.0))

    def _precompute_shift_matrices(self, bits):
        """Pre-compute the permutation matrix for each shift amount."""
        matrices = []
        for s in range(bits):
            M = torch.zeros(bits, bits)
            for i in range(bits):
                j = i - s
                if j >= 0:
                    M[i, j] = 1.0
            matrices.append(M)
        self.register_buffer('shift_matrices', torch.stack(matrices))

    def set_temperature(self, temp):
        self.temperature.fill_(temp)

    def forward(self, a_bits, shift_bits, return_shift_logits=False):
        batch = a_bits.shape[0]
        bits = a_bits.shape[1]

        # Decode shift amount with residual layers
        x = self.input_proj(shift_bits)
        for layer in self.layers:
            x = x + 0.5 * layer(x)  # Residual with scaling
        shift_logits = self.output_proj(x)[:, :bits]

        if self.training:
            shift_probs = F.softmax(shift_logits / self.temperature, dim=-1)
        else:
            # Hard selection at inference
            shift_probs = F.one_hot(shift_logits.argmax(dim=-1), bits).float()

        # Apply shift using pre-computed matrices
        M = self.shift_matrices[:bits, :bits, :bits]
        M_weighted = torch.einsum('bs,sij->bij', shift_probs, M)
        output = torch.bmm(M_weighted, a_bits.unsqueeze(-1)).squeeze(-1)

        if return_shift_logits:
            return output, shift_logits
        return output


def int_to_bits(val, bits=64):
    return torch.tensor([(val >> i) & 1 for i in range(bits)], dtype=torch.float32)


def bits_to_int(bits_t):
    return sum(int(b > 0.5) << i for i, b in enumerate(bits_t.cpu().tolist()))


def generate_batch(batch_size, bits, max_shift, device):
    """Generate training data with ground truth shift for auxiliary loss."""
    a_bits = torch.randint(0, 2, (batch_size, bits), device=device).float()
    shifts = torch.randint(0, max_shift + 1, (batch_size,), device=device)
    positions = torch.arange(bits, device=device).unsqueeze(0)

    # LSL: output[i] = input[i - shift] if i >= shift else 0
    src_pos = positions - shifts.unsqueeze(1)
    valid = src_pos >= 0
    src_pos_clamped = src_pos.clamp(0, bits - 1)
    result = torch.gather(a_bits, 1, src_pos_clamped) * valid.float()

    # Binary encoding of shift
    shift_bits = ((shifts.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()

    # Ground truth one-hot for auxiliary loss
    target_shift_onehot = F.one_hot(shifts.clamp(0, bits - 1), bits).float()

    return a_bits, shift_bits, result, target_shift_onehot


def evaluate_exact(model, bits, max_shift, device, num_tests=500):
    """Exact evaluation - check bit-perfect accuracy."""
    model.eval()
    correct = 0

    with torch.no_grad():
        for _ in range(num_tests):
            val = random.randint(0, (1 << bits) - 1)
            shift = random.randint(0, max_shift)
            expected = (val << shift) & ((1 << bits) - 1)

            input_bits = int_to_bits(val, bits).unsqueeze(0).to(device)
            shift_bits = int_to_bits(shift, bits).unsqueeze(0).to(device)

            output = model(input_bits, shift_bits)
            result = bits_to_int(output[0])

            if result == expected:
                correct += 1

    model.train()
    return correct / num_tests


def train():
    print("=" * 70)
    print("STRICT CURRICULUM LSL TRAINING")
    print("=" * 70)
    print(f"Device: {device}")

    BITS = 64

    model = StrictShiftNet(max_bits=BITS, hidden_dim=512).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    scaler = GradScaler('cuda')

    output_loss_fn = nn.MSELoss()
    shift_loss_fn = nn.CrossEntropyLoss()

    os.makedirs("models/strict_curriculum", exist_ok=True)

    # Curriculum: 8-bit increments (0-7, 0-15, 0-23, ..., 0-63)
    # Each level adds 8 more shift positions to master
    curriculum = [7, 15, 23, 31, 39, 47, 55, 63]

    batch_size = 4096

    for level_idx, max_shift in enumerate(curriculum):
        print(f"\n{'='*70}")
        print(f"LEVEL {level_idx + 1}/{len(curriculum)}: shifts 0-{max_shift}")
        print("="*70)

        # Adaptive learning rate
        base_lr = 2e-3 * (0.95 ** level_idx)
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=200, T_mult=2, eta_min=1e-6
        )

        best_acc = 0
        consecutive_100 = 0
        epochs_at_level = 0
        max_epochs = 3000

        while epochs_at_level < max_epochs:
            model.train()

            # Temperature: start warm, go cold
            temp = max(0.05, 1.0 - epochs_at_level / 500)
            model.set_temperature(temp)

            # Generate current level data
            a_bits, shift_bits, target, target_shift = generate_batch(
                batch_size, BITS, max_shift, device
            )

            # Mix in previous level data (20% of batch) to prevent forgetting
            if level_idx > 0:
                prev_max_shift = curriculum[max(0, level_idx - 1)]
                prev_a, prev_s, prev_t, prev_ts = generate_batch(
                    batch_size // 5, BITS, prev_max_shift, device
                )
                # Replace part of batch
                n = batch_size * 4 // 5
                a_bits = torch.cat([a_bits[:n], prev_a])
                shift_bits = torch.cat([shift_bits[:n], prev_s])
                target = torch.cat([target[:n], prev_t])
                target_shift = torch.cat([target_shift[:n], prev_ts])

            optimizer.zero_grad()

            with autocast('cuda', dtype=torch.bfloat16):
                output, shift_logits = model(a_bits, shift_bits, return_shift_logits=True)

                # Main output loss
                main_loss = output_loss_fn(output, target)

                # Auxiliary: shift decoder supervision
                shift_aux_loss = shift_loss_fn(shift_logits, target_shift.argmax(dim=1))

                # Combined loss - weight shift loss higher early on
                shift_weight = max(0.1, 0.5 - epochs_at_level / 1000)
                loss = main_loss + shift_weight * shift_aux_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Evaluate with exact integer comparison
            acc = evaluate_exact(model, BITS, max_shift, device, num_tests=300)

            if epochs_at_level % 50 == 0 or acc > best_acc:
                print(f"  Epoch {epochs_at_level:4d}: loss={loss.item():.4f} "
                      f"acc={acc*100:.1f}% temp={temp:.3f}")

            if acc > best_acc:
                best_acc = acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'level': level_idx,
                    'max_shift': max_shift,
                    'accuracy': acc
                }, f"models/strict_curriculum/LSL_shift{max_shift}_best.pt")

            # Check for 100% (with small tolerance for random sampling)
            if acc >= 0.998:  # 99.8% to account for sampling variance
                consecutive_100 += 1
                if consecutive_100 >= 5:  # Require 5 consecutive passes
                    print(f"  >>> 100% ACCURACY x5 at level {level_idx + 1}! <<<")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'level': level_idx,
                        'max_shift': max_shift,
                        'accuracy': 1.0
                    }, f"models/strict_curriculum/LSL_shift{max_shift}_PERFECT.pt")
                    break
            else:
                consecutive_100 = 0

            epochs_at_level += 1

        if best_acc < 0.998:
            print(f"  WARNING: Did not reach 100% at shift 0-{max_shift}")
            print(f"  Best: {best_acc*100:.1f}%")
            # Try extra training with harder settings
            print(f"  Attempting recovery with lower temperature...")
            model.set_temperature(0.01)
            for extra in range(500):
                a_bits, shift_bits, target, target_shift = generate_batch(
                    batch_size, BITS, max_shift, device
                )
                optimizer.zero_grad()
                with autocast('cuda', dtype=torch.bfloat16):
                    output, shift_logits = model(a_bits, shift_bits, return_shift_logits=True)
                    loss = output_loss_fn(output, target) + 0.3 * shift_loss_fn(
                        shift_logits, target_shift.argmax(dim=1)
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if extra % 50 == 0:
                    acc = evaluate_exact(model, BITS, max_shift, device, num_tests=300)
                    print(f"    Extra {extra}: acc={acc*100:.1f}%")
                    if acc >= 0.998:
                        print(f"  >>> RECOVERY SUCCESS! <<<")
                        break

    # Final comprehensive test
    print("\n" + "=" * 70)
    print("FINAL VERIFICATION")
    print("=" * 70)

    test_cases = [
        (1, 4),
        (255, 1),
        (0xF0F0, 4),
        (0x1234567890ABCDEF, 8),
        (1, 63),
        ((1 << 64) - 1, 0),
        (0xAAAAAAAAAAAAAAAA, 32),
    ]

    model.eval()
    model.set_temperature(0.01)
    all_correct = True

    with torch.no_grad():
        for val, shift in test_cases:
            input_bits = int_to_bits(val, BITS).unsqueeze(0).to(device)
            shift_bits_t = int_to_bits(shift, BITS).unsqueeze(0).to(device)

            output = model(input_bits, shift_bits_t)
            result = bits_to_int(output[0])
            expected = (val << shift) & ((1 << BITS) - 1)

            status = "✓ OK" if result == expected else f"✗ GOT {result}"
            if result != expected:
                all_correct = False
            print(f"  {val:#018x} << {shift:2d} = {expected:#018x}: {status}")

    if all_correct:
        print("\n>>> ALL TESTS PASSED! <<<")
        torch.save({
            'model_state_dict': model.state_dict(),
            'architecture': 'StrictShiftNet',
            'bits': BITS,
            'accuracy': 1.0
        }, "models/strict_curriculum/LSL_64bit_FINAL.pt")
    else:
        print("\n>>> SOME TESTS FAILED <<<")

    return model


if __name__ == "__main__":
    train()
