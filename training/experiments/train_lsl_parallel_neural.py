#!/usr/bin/env python3
"""
PARALLEL NEURAL LSL - Based on proven DecomposedShiftNet
=========================================================
Parallelized version of the working train_lsl_exact.py approach.

Key insight: The DecomposedShiftNet achieves 100% but uses a Python loop.
This version parallelizes the computation for speed while keeping the
same proven architecture.

What the network learns (NO hardcoded matrices):
1. shift_decoder: binary shift → one-hot shift (learned)
2. index_net: (output_pos, shift) → source_index (learned)
3. validity_net: (output_pos, shift) → valid? (learned)

Auxiliary supervision guides learning but doesn't hardcode the operation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import random
import os
import sys

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

device = "cuda" if torch.cuda.is_available() else "cpu"


class ParallelDecomposedLSL(nn.Module):
    """
    Parallelized version of DecomposedShiftNet.

    All 64 output positions computed in ONE forward pass.
    Uses batch matrix operations instead of Python loops.
    """

    def __init__(self, max_bits=64, hidden_dim=512):
        super().__init__()
        self.max_bits = max_bits

        # Shift decoder: binary → one-hot shift amount (LEARNED)
        self.shift_decoder = nn.Sequential(
            nn.Linear(max_bits, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_bits)
        )

        # Index net: (output_pos + shift) → source_index (LEARNED)
        # This learns to compute "output_pos - shift"
        self.index_net = nn.Sequential(
            nn.Linear(max_bits * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_bits)
        )

        # Validity net: (output_pos + shift) → is_valid (LEARNED)
        # This learns "output_pos >= shift"
        self.validity_net = nn.Sequential(
            nn.Linear(max_bits * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Pre-compute position one-hots for efficiency
        self.register_buffer('pos_onehots', torch.eye(max_bits))
        self.register_buffer('temp', torch.tensor(1.0))

    def set_temperature(self, t):
        self.temp.fill_(t)

    def forward(self, a_bits, shift_bits, return_aux=False):
        """
        Fully parallel forward pass.

        Args:
            a_bits: [batch, bits] - input value
            shift_bits: [batch, bits] - shift amount in binary
        """
        batch = a_bits.shape[0]
        bits = a_bits.shape[1]

        # Step 1: Decode shift amount (LEARNED)
        shift_logits = self.shift_decoder(shift_bits)[:, :bits]  # [batch, bits]

        if self.training:
            shift_soft = F.softmax(shift_logits / self.temp, dim=-1)
        else:
            shift_soft = F.one_hot(shift_logits.argmax(dim=-1), bits).float()

        # Step 2: Create all position inputs in parallel
        # pos_onehots: [bits, bits] - identity matrix
        # We need [batch, bits, bits*2] where each position has (pos_onehot, shift_soft)
        pos_inputs = self.pos_onehots[:bits, :bits]  # [bits, bits]
        pos_inputs = pos_inputs.unsqueeze(0).expand(batch, -1, -1)  # [batch, bits, bits]

        # Expand shift_soft to match: [batch, bits, bits]
        shift_expanded = shift_soft.unsqueeze(1).expand(-1, bits, -1)  # [batch, bits, bits]

        # Combine: [batch, bits, bits*2]
        combined = torch.cat([pos_inputs, shift_expanded], dim=-1)

        # Step 3: Predict source indices for ALL positions in parallel (LEARNED)
        # Reshape for batch processing: [batch*bits, bits*2]
        combined_flat = combined.view(batch * bits, bits * 2)
        index_logits_flat = self.index_net(combined_flat)[:, :bits]  # [batch*bits, bits]
        index_logits = index_logits_flat.view(batch, bits, bits)  # [batch, out_pos, in_pos]

        if self.training:
            index_soft = F.softmax(index_logits / self.temp, dim=-1)
        else:
            index_soft = F.one_hot(index_logits.argmax(dim=-1), bits).float()

        # Step 4: Gather values using learned indices
        # a_bits: [batch, bits]
        # index_soft: [batch, out_pos, in_pos]
        # output[b, out_pos] = sum_in(index_soft[b, out_pos, in] * a_bits[b, in])
        gathered = torch.bmm(index_soft, a_bits.unsqueeze(-1)).squeeze(-1)  # [batch, bits]

        # Step 5: Predict validity for ALL positions in parallel (LEARNED)
        validity_logits_flat = self.validity_net(combined_flat)  # [batch*bits, 1]
        validity_logits = validity_logits_flat.view(batch, bits)  # [batch, bits]

        # Step 6: Final output = gathered * validity
        output = gathered * torch.sigmoid(validity_logits)

        if return_aux:
            return output, shift_logits, index_logits, validity_logits
        return output


def int_to_bits(val, bits=64):
    return torch.tensor([(val >> i) & 1 for i in range(bits)], dtype=torch.float32)


def bits_to_int(bits_t):
    return sum(int(b > 0.5) << i for i, b in enumerate(bits_t.cpu().tolist()))


def generate_batch(batch_size, bits, max_shift, device):
    """Generate training data with ground truth for auxiliary supervision."""
    a_bits = torch.randint(0, 2, (batch_size, bits), device=device).float()
    shifts = torch.randint(0, max_shift + 1, (batch_size,), device=device)
    positions = torch.arange(bits, device=device).unsqueeze(0)

    # Ground truth LSL
    src_pos = positions - shifts.unsqueeze(1)
    valid = src_pos >= 0
    src_pos_clamped = src_pos.clamp(0, bits - 1)
    result = torch.gather(a_bits, 1, src_pos_clamped) * valid.float()

    shift_bits = ((shifts.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()

    # Ground truth for auxiliary losses
    target_shift = F.one_hot(shifts.clamp(0, bits - 1), bits).float()
    target_source = src_pos_clamped  # [batch, bits]
    target_valid = valid.float()

    return a_bits, shift_bits, result, target_shift, target_source, target_valid


def evaluate_exact(model, bits, max_shift, device, num_tests=300):
    model.eval()
    correct = 0

    with torch.no_grad():
        for _ in range(num_tests):
            val = random.randint(0, (1 << bits) - 1)
            shift = random.randint(0, max_shift)
            expected = (val << shift) & ((1 << bits) - 1)

            input_bits = int_to_bits(val, bits).unsqueeze(0).to(device)
            shift_bits_t = int_to_bits(shift, bits).unsqueeze(0).to(device)

            output = model(input_bits, shift_bits_t)
            result = bits_to_int(output[0])

            if result == expected:
                correct += 1

    model.train()
    return correct / num_tests


def train():
    print("=" * 70)
    print("PARALLEL NEURAL LSL - No Hardcoded Matrices")
    print("=" * 70)
    print(f"Device: {device}")

    BITS = 64

    model = ParallelDecomposedLSL(max_bits=BITS, hidden_dim=768).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print("Architecture: shift_decoder + index_net + validity_net (all LEARNED)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scaler = GradScaler('cuda')

    output_loss_fn = nn.MSELoss()
    shift_loss_fn = nn.CrossEntropyLoss()
    index_loss_fn = nn.CrossEntropyLoss()
    valid_loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs("models/parallel_neural", exist_ok=True)

    # Curriculum: 8-bit increments
    curriculum = [7, 15, 23, 31, 39, 47, 55, 63]
    batch_size = 4096

    for level_idx, max_shift in enumerate(curriculum):
        print(f"\n{'='*70}")
        print(f"LEVEL {level_idx + 1}/{len(curriculum)}: shifts 0-{max_shift}")
        print("=" * 70)

        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-3 * (0.9 ** level_idx)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=3000, eta_min=1e-6
        )

        best_acc = 0
        consecutive_100 = 0
        max_epochs = 3000

        for epoch in range(max_epochs):
            model.train()
            temp = max(0.1, 1.0 - epoch / 1000)
            model.set_temperature(temp)

            # Generate data
            a_bits, shift_bits, target, target_shift, target_source, target_valid = \
                generate_batch(batch_size, BITS, max_shift, device)

            # Mix previous level
            if level_idx > 0:
                prev_max = curriculum[max(0, level_idx - 1)]
                prev_data = generate_batch(batch_size // 5, BITS, prev_max, device)
                n = batch_size * 4 // 5
                a_bits = torch.cat([a_bits[:n], prev_data[0]])
                shift_bits = torch.cat([shift_bits[:n], prev_data[1]])
                target = torch.cat([target[:n], prev_data[2]])
                target_shift = torch.cat([target_shift[:n], prev_data[3]])
                target_source = torch.cat([target_source[:n], prev_data[4]])
                target_valid = torch.cat([target_valid[:n], prev_data[5]])

            optimizer.zero_grad()

            with autocast('cuda', dtype=torch.bfloat16):
                output, shift_logits, index_logits, validity_logits = \
                    model(a_bits, shift_bits, return_aux=True)

                # Main output loss
                main_loss = output_loss_fn(output, target)

                # Auxiliary: shift decoder loss
                shift_loss = shift_loss_fn(shift_logits, target_shift.argmax(dim=1))

                # Auxiliary: index prediction loss (for each output position)
                # index_logits: [batch, out_pos, in_pos]
                # target_source: [batch, out_pos]
                index_loss = 0
                for i in range(BITS):
                    index_loss += index_loss_fn(index_logits[:, i, :], target_source[:, i])
                index_loss /= BITS

                # Auxiliary: validity loss
                valid_loss = valid_loss_fn(validity_logits, target_valid)

                # Combined - decrease aux weight over time
                aux_weight = max(0.1, 0.5 - epoch / 2000)
                loss = main_loss + aux_weight * (0.3 * shift_loss + 0.5 * index_loss + 0.2 * valid_loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            acc = evaluate_exact(model, BITS, max_shift, device, num_tests=200)

            if epoch % 100 == 0 or acc > best_acc:
                print(f"  Epoch {epoch:4d}: loss={loss.item():.4f} acc={acc*100:.1f}% temp={temp:.2f}")

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), f"models/parallel_neural/LSL_shift{max_shift}_best.pt")

            if acc >= 0.998:
                consecutive_100 += 1
                if consecutive_100 >= 3:
                    print(f"  >>> 100% at level {level_idx + 1}! <<<")
                    torch.save(model.state_dict(), f"models/parallel_neural/LSL_shift{max_shift}_100pct.pt")
                    break
            else:
                consecutive_100 = 0

        if best_acc < 0.998:
            print(f"  Level {level_idx+1} best: {best_acc*100:.1f}%")

    # Final verification
    print("\n" + "=" * 70)
    print("FINAL VERIFICATION")
    print("=" * 70)

    test_cases = [
        (1, 4), (255, 1), (0xF0F0, 4), (0x1234567890ABCDEF, 8),
        (1, 63), ((1 << 64) - 1, 0), (0xAAAAAAAAAAAAAAAA, 32),
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
            status = "OK" if result == expected else f"GOT {result}"
            if result != expected:
                all_correct = False
            print(f"  {val:#018x} << {shift:2d} = {expected:#018x}: {status}")

    if all_correct:
        print("\n>>> ALL TESTS PASSED! <<<")
        torch.save(model.state_dict(), "models/parallel_neural/LSL_64bit_FINAL.pt")


if __name__ == "__main__":
    train()
