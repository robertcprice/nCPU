#!/usr/bin/env python3
"""
PARALLEL SHIFT - Permutation Matrix Approach
=============================================
Uses learned permutation matrices for O(1) parallel shifting.
NO LOOPS - single matrix multiply!

For LSL: output[i] = input[i - shift] (or 0 if out of bounds)
For LSR: output[i] = input[i + shift] (or 0 if out of bounds)

Architecture:
- Learn 64 permutation matrices (one per shift amount)
- Decode shift amount -> select permutation matrix
- Apply permutation via batched matrix multiply
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


class ParallelShiftNet(nn.Module):
    """Parallel shift using learned permutation matrices."""

    def __init__(self, max_bits=64, direction='left'):
        super().__init__()
        self.max_bits = max_bits
        self.direction = direction  # 'left' for LSL, 'right' for LSR

        # Learned permutation logits: [max_shift, max_bits, max_bits]
        # perm[s, i, j] = logit that output bit i comes from input bit j
        self.perm_logits = nn.Parameter(torch.zeros(max_bits, max_bits, max_bits))

        # Shift amount decoder: convert shift bits to one-hot
        self.shift_decoder = nn.Sequential(
            nn.Linear(max_bits, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, max_bits)
        )

        # Initialize with correct shift patterns (as starting point)
        self._init_shift_patterns()

    def _init_shift_patterns(self):
        """Initialize permutation matrices to approximate shifts."""
        with torch.no_grad():
            for s in range(self.max_bits):
                for i in range(self.max_bits):
                    if self.direction == 'left':
                        # LSL: output[i] comes from input[i - s]
                        src = i - s
                    else:
                        # LSR: output[i] comes from input[i + s]
                        src = i + s

                    if 0 <= src < self.max_bits:
                        self.perm_logits[s, i, src] = 10.0  # High logit
                    # Else: no source, output should be 0

    def forward(self, input_bits, shift_bits):
        """
        Parallel shift operation.

        Args:
            input_bits: [batch, max_bits] - value to shift
            shift_bits: [batch, max_bits] - shift amount as bits

        Returns:
            [batch, max_bits] - shifted result
        """
        batch = input_bits.shape[0]

        # Decode shift amount to soft one-hot
        shift_logits = self.shift_decoder(shift_bits)
        shift_probs = F.softmax(shift_logits, dim=-1)  # [batch, max_bits]

        # Combine permutation matrices weighted by shift probability
        # perms: [batch, max_bits, max_bits]
        # shift_probs: [batch, max_bits]
        # Want: combined[b, i, j] = sum_s(shift_probs[b, s] * perm[s, i, j])

        # Get soft permutation for each shift
        soft_perms = F.softmax(self.perm_logits, dim=-1)  # [max_bits, max_bits, max_bits]

        # Weighted combination
        combined = torch.einsum('bs,sij->bij', shift_probs, soft_perms)

        # Apply permutation: output = combined @ input
        result = torch.bmm(combined, input_bits.unsqueeze(-1)).squeeze(-1)

        return result


def int_to_bits(val, bits=64):
    return torch.tensor([(val >> i) & 1 for i in range(bits)], dtype=torch.float32)


def bits_to_int(bits_t):
    return sum(int(b > 0.5) << i for i, b in enumerate(bits_t.cpu().tolist()))


def generate_batch(batch_size, bits, direction, device):
    """Generate training batch."""
    inputs = []
    shifts = []
    targets = []

    max_val = (1 << bits) - 1

    for _ in range(batch_size):
        val = random.randint(0, max_val)
        shift = random.randint(0, bits - 1)

        if direction == 'left':
            result = (val << shift) & max_val
        else:
            result = val >> shift

        inputs.append(int_to_bits(val, bits))
        shifts.append(int_to_bits(shift, bits))
        targets.append(int_to_bits(result, bits))

    return (torch.stack(inputs).to(device),
            torch.stack(shifts).to(device),
            torch.stack(targets).to(device))


def train(direction='left'):
    op_name = 'LSL' if direction == 'left' else 'LSR'
    print("=" * 60)
    print(f"PARALLEL {op_name} TRAINING")
    print("=" * 60)
    print(f"Device: {device}")

    model = ParallelShiftNet(max_bits=64, direction=direction).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    os.makedirs("models/final", exist_ok=True)

    best_acc = 0
    batch_size = 512

    for epoch in range(200):
        model.train()
        total_loss = 0
        t0 = time.time()

        for _ in range(100):
            inputs, shifts, targets = generate_batch(batch_size, 64, direction, device)

            optimizer.zero_grad()
            pred = model(inputs, shifts)
            loss = F.binary_cross_entropy_with_logits(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Test
        model.eval()
        test_cases = [
            (1, 4),    # 1 << 4 = 16
            (255, 1),  # 255 << 1 = 510
            (0xF0F0, 4),
            (0x1234567890ABCDEF, 8),
            (1, 63),   # Edge case
        ]

        correct = 0
        with torch.no_grad():
            for val, shift in test_cases:
                input_bits = int_to_bits(val).unsqueeze(0).to(device)
                shift_bits = int_to_bits(shift).unsqueeze(0).to(device)
                result = model(input_bits, shift_bits)
                result_int = bits_to_int(result[0])

                if direction == 'left':
                    expected = (val << shift) & ((1 << 64) - 1)
                else:
                    expected = val >> shift

                if result_int == expected:
                    correct += 1

        acc = correct / len(test_cases)
        elapsed = time.time() - t0

        if epoch % 20 == 0 or acc > best_acc:
            print(f"Epoch {epoch+1}: loss={total_loss/100:.4f} acc={100*acc:.0f}% [{elapsed:.1f}s]")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "accuracy": acc,
                "op_name": op_name,
                "bits": 64,
                "architecture": "ParallelShiftNet"
            }, f"models/final/{op_name}_64bit_parallel.pt")
            print(f"  Saved (acc={100*acc:.0f}%)")

        if acc >= 1.0:
            print("100% ACCURACY!")
            break

    print(f"Best accuracy: {100*best_acc:.0f}%")

    # Final verification
    print(f"\nFinal verification ({op_name}):")
    model.eval()
    with torch.no_grad():
        for val, shift in test_cases:
            input_bits = int_to_bits(val).unsqueeze(0).to(device)
            shift_bits = int_to_bits(shift).unsqueeze(0).to(device)
            result = model(input_bits, shift_bits)
            result_int = bits_to_int(result[0])

            if direction == 'left':
                expected = (val << shift) & ((1 << 64) - 1)
            else:
                expected = val >> shift

            status = "OK" if result_int == expected else f"GOT {result_int}"
            print(f"  {val} {'<<' if direction == 'left' else '>>'} {shift} = {expected}: {status}")


if __name__ == "__main__":
    import sys
    direction = sys.argv[1] if len(sys.argv) > 1 else 'left'
    train(direction)
