#!/usr/bin/env python3
"""
CURRICULUM LSL TRAINING v2
==========================
Uses HARD selection via Gumbel-Softmax for exact bit routing.

Key insight: LSL is a SELECTION problem, not a computation problem.
Each output bit selects exactly ONE input bit (or zero).

Soft selection (regular softmax) produces blended outputs that can't
achieve 100% accuracy. We need HARD selection.

Gumbel-Softmax trick:
- During training: differentiable approximation to argmax
- Temperature annealing: start soft, end hard
- At inference: use actual argmax for exact selection

Architecture v2:
1. Shift decoder: convert shift bits to one-hot (with Gumbel)
2. Position predictor: for each output bit, predict source position
3. Hard selection: use Gumbel-Softmax or straight-through estimator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


def gumbel_softmax(logits, temperature=1.0, hard=False):
    """
    Gumbel-Softmax for differentiable discrete selection.

    When hard=True, uses straight-through estimator:
    - Forward: one-hot
    - Backward: soft gradients
    """
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y_soft = F.softmax((logits + gumbels) / temperature, dim=-1)

    if hard:
        # Straight-through: one-hot in forward, soft gradients in backward
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


class HardLSLNet(nn.Module):
    """
    LSL network with hard selection.

    For each output bit position i:
    - If shift <= i: output[i] = input[i - shift]
    - Else: output[i] = 0

    We learn to:
    1. Decode shift amount accurately
    2. Compute source position for each output
    3. Select the correct input bit OR output zero
    """

    def __init__(self, max_bits=64, hidden_dim=256):
        super().__init__()
        self.max_bits = max_bits

        # Shift decoder: robust multi-layer decoder
        self.shift_decoder = nn.Sequential(
            nn.Linear(max_bits, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),  # ReLU often more stable for discrete tasks
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_bits)
        )

        # Position encodings (fixed, not learned)
        self.register_buffer('positions', torch.arange(max_bits).float())

        # For each output position, predict whether to output 0 or copy
        # This helps handle the "shift > position" case
        self.zero_predictor = nn.Sequential(
            nn.Linear(max_bits + max_bits, hidden_dim),  # shift + position
            nn.ReLU(),
            nn.Linear(hidden_dim, max_bits)  # per-position zero probability
        )

        # Temperature for Gumbel-Softmax (annealed during training)
        self.temperature = 1.0

    def forward(self, input_bits, shift_bits, hard=False):
        """
        Args:
            input_bits: [batch, max_bits]
            shift_bits: [batch, max_bits]
            hard: if True, use hard selection (for inference)
        """
        batch = input_bits.shape[0]

        # Decode shift amount
        shift_logits = self.shift_decoder(shift_bits)

        if hard or self.training:
            # Use Gumbel-Softmax for hard/semi-hard selection
            shift_probs = gumbel_softmax(
                shift_logits,
                temperature=self.temperature,
                hard=hard
            )
        else:
            shift_probs = F.softmax(shift_logits, dim=-1)

        # Convert to actual shift values (expected value)
        positions = self.positions.unsqueeze(0)  # [1, max_bits]
        shift_val = (shift_probs * positions).sum(dim=-1)  # [batch]

        # For each output position, compute source position
        # source[i] = i - shift (but we use the probabilistic version)
        # [batch, max_bits]
        source_positions = positions - shift_val.unsqueeze(-1)

        # Create routing: output[out_pos] should select input[out_pos - shift]
        # Build selection matrix
        # selector[b, out, in] = prob that output[out] selects input[in]

        # We use the fact that each output position has ONE source
        # source_positions[b, out] gives the (fractional) source index

        # For hard selection during inference
        if hard:
            # Round to nearest integer source
            src_idx = source_positions.round().long().clamp(0, self.max_bits - 1)
            # Gather from input
            gathered = torch.gather(input_bits, 1, src_idx)
            # Zero out positions where shift > position (source would be negative)
            valid = source_positions >= 0
            result = gathered * valid.float()
        else:
            # Soft selection using interpolation
            # This is differentiable
            src_floor = source_positions.floor().long().clamp(0, self.max_bits - 1)
            src_ceil = (src_floor + 1).clamp(0, self.max_bits - 1)
            frac = (source_positions - src_floor.float()).clamp(0, 1)

            val_floor = torch.gather(input_bits, 1, src_floor)
            val_ceil = torch.gather(input_bits, 1, src_ceil)

            # Interpolate
            gathered = val_floor * (1 - frac) + val_ceil * frac

            # Zero out invalid positions
            valid = source_positions >= -0.5  # Small margin for interpolation
            result = gathered * valid.float()

        return result

    def set_temperature(self, temp):
        """Set Gumbel-Softmax temperature."""
        self.temperature = temp


class PerBitLSLNet(nn.Module):
    """
    Alternative: predict each output bit independently.

    For each output bit position i, the network predicts:
    1. Which input bit to look at (i - shift)
    2. Whether to output that bit or 0

    This is more parameter-heavy but more robust.
    """

    def __init__(self, max_bits=64, hidden_dim=128):
        super().__init__()
        self.max_bits = max_bits

        # Shared shift encoder
        self.shift_encoder = nn.Sequential(
            nn.Linear(max_bits, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Per-position predictors
        # Each predicts: source_index (0 to max_bits-1) or "zero"
        self.pos_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(max_bits + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, max_bits + 1)  # +1 for "output zero" option
            )
            for _ in range(max_bits)
        ])

        self.temperature = 1.0

    def forward(self, input_bits, shift_bits, hard=False):
        batch = input_bits.shape[0]

        # Encode shift
        shift_enc = self.shift_encoder(shift_bits)

        results = []
        for out_pos in range(self.max_bits):
            # Predict source for this output position
            combined = torch.cat([input_bits, shift_enc], dim=-1)
            source_logits = self.pos_predictors[out_pos](combined)

            if hard:
                selection = gumbel_softmax(source_logits, self.temperature, hard=True)
            else:
                selection = F.softmax(source_logits / self.temperature, dim=-1)

            # selection[:, 0:max_bits] = prob of selecting each input bit
            # selection[:, max_bits] = prob of outputting zero

            # Extended input with zero option
            extended_input = torch.cat([
                input_bits,
                torch.zeros(batch, 1, device=input_bits.device)
            ], dim=-1)

            # Select
            out_bit = (selection * extended_input).sum(dim=-1)
            results.append(out_bit)

        return torch.stack(results, dim=-1)


def int_to_bits(val, bits=64):
    return torch.tensor([(val >> i) & 1 for i in range(bits)], dtype=torch.float32)


def bits_to_int(bits_t):
    return sum(int(b > 0.5) << i for i, b in enumerate(bits_t.cpu().tolist()))


def generate_curriculum_batch(batch_size, bits, max_shift, device):
    vals = [random.randint(0, (1 << bits) - 1) for _ in range(batch_size)]
    shifts = [random.randint(0, max_shift) for _ in range(batch_size)]

    inputs = []
    shift_bits = []
    targets = []

    for val, shift in zip(vals, shifts):
        result = (val << shift) & ((1 << bits) - 1)
        inputs.append(int_to_bits(val, bits))
        shift_bits.append(int_to_bits(shift, bits))
        targets.append(int_to_bits(result, bits))

    return (torch.stack(inputs).to(device),
            torch.stack(shift_bits).to(device),
            torch.stack(targets).to(device))


def evaluate_model(model, bits, max_shift, device, num_tests=200, hard=True):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(num_tests):
            val = random.randint(0, (1 << bits) - 1)
            shift = random.randint(0, max_shift)
            expected = (val << shift) & ((1 << bits) - 1)

            input_bits = int_to_bits(val, bits).unsqueeze(0).to(device)
            shift_bits_t = int_to_bits(shift, bits).unsqueeze(0).to(device)

            output = model(input_bits, shift_bits_t, hard=hard)
            result = bits_to_int(output[0])

            if result == expected:
                correct += 1
            total += 1

        # Edge cases
        edge_cases = [
            (0, 0),
            ((1 << bits) - 1, 0),
            (1, max_shift),
            ((1 << bits) - 1, max_shift),
        ]

        for val, shift in edge_cases:
            if shift <= max_shift:
                expected = (val << shift) & ((1 << bits) - 1)
                input_bits = int_to_bits(val, bits).unsqueeze(0).to(device)
                shift_bits_t = int_to_bits(shift, bits).unsqueeze(0).to(device)

                output = model(input_bits, shift_bits_t, hard=hard)
                result = bits_to_int(output[0])

                if result == expected:
                    correct += 1
                total += 1

    model.train()
    return correct / total


def train_curriculum():
    print("=" * 60)
    print("CURRICULUM LSL TRAINING v2 (Hard Selection)")
    print("=" * 60)
    print(f"Device: {device}")

    BITS = 64

    model = HardLSLNet(max_bits=BITS, hidden_dim=256).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    os.makedirs("models/curriculum", exist_ok=True)

    # Fine-grained curriculum
    curriculum = [0, 1, 2, 3, 5, 7, 11, 15, 23, 31, 47, 63]

    batch_size = 256

    for level_idx, max_shift in enumerate(curriculum):
        print(f"\n{'='*60}")
        print(f"LEVEL {level_idx + 1}: shifts 0-{max_shift}")
        print("=" * 60)

        # Anneal temperature: start soft, end hard
        initial_temp = max(2.0 - level_idx * 0.15, 0.5)
        model.set_temperature(initial_temp)

        # Fresh scheduler per level
        for param_group in optimizer.param_groups:
            param_group['lr'] = 3e-4 * (0.9 ** level_idx)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=100, T_mult=2
        )

        best_acc = 0
        best_hard_acc = 0

        for epoch in range(1500):
            model.train()
            total_loss = 0

            # Anneal temperature during epoch
            progress = epoch / 1500
            current_temp = initial_temp * (1 - 0.8 * progress) + 0.1
            model.set_temperature(current_temp)

            for _ in range(30):
                inputs, shifts, targets = generate_curriculum_batch(
                    batch_size, BITS, max_shift, device
                )

                # Mix in easier examples
                if level_idx > 0 and random.random() < 0.15:
                    prev_max = curriculum[max(0, level_idx - 2)]
                    easy_in, easy_sh, easy_tgt = generate_curriculum_batch(
                        batch_size // 4, BITS, prev_max, device
                    )
                    n = batch_size * 3 // 4
                    inputs = torch.cat([inputs[:n], easy_in])
                    shifts = torch.cat([shifts[:n], easy_sh])
                    targets = torch.cat([targets[:n], easy_tgt])

                optimizer.zero_grad()

                # Train with semi-hard selection
                output = model(inputs, shifts, hard=False)

                # BCE loss
                output_clamped = output.clamp(1e-7, 1 - 1e-7)
                loss = F.binary_cross_entropy(output_clamped, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()

            # Evaluate with hard selection (what matters for actual use)
            hard_acc = evaluate_model(model, BITS, max_shift, device, num_tests=150, hard=True)

            if epoch % 25 == 0 or hard_acc > best_hard_acc:
                print(f"  Epoch {epoch:4d}: loss={total_loss/30:.4f} hard_acc={hard_acc*100:.1f}% temp={current_temp:.2f}")

            if hard_acc > best_hard_acc:
                best_hard_acc = hard_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'level': level_idx,
                    'max_shift': max_shift,
                    'accuracy': hard_acc
                }, f"models/curriculum/LSL_v2_level{level_idx}.pt")

                if hard_acc >= 0.999:
                    print(f"  >>> 100% ACCURACY at level {level_idx + 1}! <<<")
                    break

        if best_hard_acc < 0.999:
            print(f"  Level {level_idx+1} best: {best_hard_acc*100:.1f}%")

    # Final test
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION")
    print("=" * 60)

    test_cases = [
        (1, 4),
        (255, 1),
        (0xF0F0, 4),
        (0x1234567890ABCDEF, 8),
        (1, 63),
    ]

    model.eval()
    all_correct = True
    with torch.no_grad():
        for val, shift in test_cases:
            input_bits = int_to_bits(val, BITS).unsqueeze(0).to(device)
            shift_bits = int_to_bits(shift, BITS).unsqueeze(0).to(device)

            output = model(input_bits, shift_bits, hard=True)
            result = bits_to_int(output[0])
            expected = (val << shift) & ((1 << BITS) - 1)

            status = "OK" if result == expected else f"GOT {result}"
            if result != expected:
                all_correct = False
            print(f"  {val} << {shift} = {expected}: {status}")

    if all_correct:
        print("\n>>> ALL TESTS PASSED! <<<")
        torch.save({
            'model_state_dict': model.state_dict(),
            'architecture': 'HardLSLNet',
            'bits': BITS,
            'accuracy': 1.0
        }, "models/curriculum/LSL_64bit_FINAL_v2.pt")


if __name__ == "__main__":
    train_curriculum()
