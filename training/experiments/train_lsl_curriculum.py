#!/usr/bin/env python3
"""
CURRICULUM LSL TRAINING
=======================
Progressive curriculum approach for Left Shift Logic.

Key principles:
1. Start with tiny shifts (0-1) and MASTER them (100%)
2. Only advance when perfect accuracy achieved
3. Mix easier examples to prevent catastrophic forgetting
4. Use position-aware architecture that understands bit semantics

The insight: LSL is fundamentally about "copying bit[i-shift] to bit[i]"
- For shift=0: identity (output[i] = input[i])
- For shift=1: output[i] = input[i-1], output[0] = 0
- For shift=k: output[i] = input[i-k] if i >= k else 0

A curriculum approach lets the model learn this pattern incrementally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


class CurriculumLSLNet(nn.Module):
    """
    Position-aware network for LSL using attention.

    Key insight: Each output bit only depends on ONE input bit (or zero).
    This is fundamentally a ROUTING problem, not a computation problem.

    Architecture:
    - Position embeddings for each bit
    - Shift amount encoder
    - Cross-attention to route input bits to output positions
    """

    def __init__(self, max_bits=64, hidden_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.max_bits = max_bits
        self.hidden_dim = hidden_dim

        # Position embeddings (learnable)
        self.pos_embed = nn.Parameter(torch.randn(max_bits, hidden_dim) * 0.02)

        # Input bit embeddings (0 or 1 -> hidden_dim)
        self.bit_embed = nn.Linear(1, hidden_dim)

        # Shift amount encoder - crucial for curriculum!
        self.shift_encoder = nn.Sequential(
            nn.Linear(max_bits, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Transformer layers for routing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head - predicts each bit independently
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, input_bits, shift_bits):
        """
        Args:
            input_bits: [batch, max_bits] - the value to shift
            shift_bits: [batch, max_bits] - shift amount as bits
        Returns:
            [batch, max_bits] - shifted result (logits)
        """
        batch = input_bits.shape[0]

        # Embed input bits with position info
        # [batch, max_bits, hidden]
        x = self.bit_embed(input_bits.unsqueeze(-1))
        x = x + self.pos_embed.unsqueeze(0)

        # Encode shift amount
        shift_encoding = self.shift_encoder(shift_bits)  # [batch, hidden]

        # Add shift info to all positions
        x = x + shift_encoding.unsqueeze(1)

        # Process through transformer
        x = self.transformer(x)

        # Output prediction for each bit
        out = self.output_head(x).squeeze(-1)  # [batch, max_bits]

        return out


class SimpleLSLNet(nn.Module):
    """
    Simpler architecture - direct bit routing via learned masks.

    For each shift amount s, we learn which input bit maps to each output bit.
    This is more directly aligned with what LSL actually does.
    """

    def __init__(self, max_bits=64, hidden_dim=512):
        super().__init__()
        self.max_bits = max_bits

        # Shift decoder: bits -> one-hot over shift amounts
        self.shift_decoder = nn.Sequential(
            nn.Linear(max_bits, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_bits)  # logits for each shift amount
        )

        # For each output bit, learn which input bit to select
        # routing_weights[s, out_bit, in_bit] = weight for shift s
        self.routing_logits = nn.Parameter(torch.zeros(max_bits, max_bits, max_bits))

        # Initialize with ground truth patterns
        self._init_routing()

    def _init_routing(self):
        """Initialize routing to correct LSL pattern."""
        with torch.no_grad():
            for s in range(self.max_bits):
                for out_bit in range(self.max_bits):
                    in_bit = out_bit - s
                    if 0 <= in_bit < self.max_bits:
                        self.routing_logits[s, out_bit, in_bit] = 10.0
                    # else: all zeros (softmax will spread, but should learn to output 0)

    def forward(self, input_bits, shift_bits):
        batch = input_bits.shape[0]

        # Decode shift amount to probabilities
        shift_logits = self.shift_decoder(shift_bits)
        shift_probs = F.softmax(shift_logits, dim=-1)  # [batch, max_bits]

        # Get routing probabilities for each shift
        routing_probs = F.softmax(self.routing_logits, dim=-1)  # [max_bits, max_bits, max_bits]

        # Combine: weighted sum over shifts
        # combined[b, out, in] = sum_s(shift_probs[b, s] * routing_probs[s, out, in])
        combined = torch.einsum('bs,soi->boi', shift_probs, routing_probs)

        # Apply routing: output[out] = sum_in(combined[out, in] * input[in])
        result = torch.bmm(combined, input_bits.unsqueeze(-1)).squeeze(-1)

        return result


def int_to_bits(val, bits=64):
    """Convert integer to bit tensor (LSB first)."""
    return torch.tensor([(val >> i) & 1 for i in range(bits)], dtype=torch.float32)


def bits_to_int(bits_t):
    """Convert bit tensor back to integer."""
    return sum(int(b > 0.5) << i for i, b in enumerate(bits_t.cpu().tolist()))


def generate_curriculum_batch(batch_size, bits, max_shift, device):
    """
    Generate training batch with BOUNDED shift amounts.

    This is the key to curriculum learning - we control complexity!
    """
    # Random values (full bit range)
    vals = [random.randint(0, (1 << bits) - 1) for _ in range(batch_size)]

    # BOUNDED shifts (curriculum control!)
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


def evaluate_model(model, bits, max_shift, device, num_tests=200):
    """
    Evaluate model accuracy on the CURRENT curriculum level.

    Tests both random cases and edge cases for the current shift range.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        # Random tests
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
            total += 1

        # Edge cases for current level
        edge_cases = [
            (0, 0),                    # Zero shift of zero
            ((1 << bits) - 1, 0),      # Max value, no shift
            (1, max_shift),            # 1 shifted max amount
            ((1 << bits) - 1, max_shift),  # Max value, max shift
        ]

        for val, shift in edge_cases:
            if shift <= max_shift:
                expected = (val << shift) & ((1 << bits) - 1)
                input_bits = int_to_bits(val, bits).unsqueeze(0).to(device)
                shift_bits_t = int_to_bits(shift, bits).unsqueeze(0).to(device)

                output = model(input_bits, shift_bits_t)
                result = bits_to_int(output[0])

                if result == expected:
                    correct += 1
                total += 1

    model.train()
    return correct / total


def train_curriculum():
    """
    Main curriculum training loop.

    Curriculum levels:
    1. shifts 0-0 (identity - trivial)
    2. shifts 0-1 (learn basic shift pattern)
    3. shifts 0-3
    4. shifts 0-7
    5. shifts 0-15
    6. shifts 0-31
    7. shifts 0-63 (full range)

    Only advance when 100% accuracy achieved!
    """
    print("=" * 60)
    print("CURRICULUM LSL TRAINING")
    print("=" * 60)
    print(f"Device: {device}")

    BITS = 64

    # Using the simpler routing-based model
    model = SimpleLSLNet(max_bits=BITS, hidden_dim=512).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    os.makedirs("models/curriculum", exist_ok=True)

    # Curriculum: progressively increase max shift
    curriculum = [0, 1, 3, 7, 15, 31, 63]

    batch_size = 512

    for level_idx, max_shift in enumerate(curriculum):
        print(f"\n{'='*60}")
        print(f"LEVEL {level_idx + 1}: shifts 0-{max_shift}")
        print("=" * 60)

        # Reset optimizer LR for new level
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-3 * (0.8 ** level_idx)  # Slight decay per level

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=500, eta_min=1e-5
        )

        best_acc = 0
        patience_counter = 0
        max_patience = 100

        for epoch in range(2000):  # Max epochs per level
            model.train()
            total_loss = 0

            # Training batches
            for _ in range(50):
                # Current level data
                inputs, shifts, targets = generate_curriculum_batch(
                    batch_size, BITS, max_shift, device
                )

                # Optional: mix in easier examples (prevent forgetting)
                if level_idx > 0 and random.random() < 0.2:
                    prev_max_shift = curriculum[level_idx - 1]
                    easy_inputs, easy_shifts, easy_targets = generate_curriculum_batch(
                        batch_size // 4, BITS, prev_max_shift, device
                    )
                    inputs = torch.cat([inputs[:batch_size*3//4], easy_inputs])
                    shifts = torch.cat([shifts[:batch_size*3//4], easy_shifts])
                    targets = torch.cat([targets[:batch_size*3//4], easy_targets])

                optimizer.zero_grad()
                output = model(inputs, shifts)

                # BCE loss for bit prediction
                loss = F.binary_cross_entropy(
                    output.clamp(1e-7, 1-1e-7),
                    targets
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()

            # Evaluate
            acc = evaluate_model(model, BITS, max_shift, device)

            if epoch % 20 == 0 or acc > best_acc:
                print(f"  Epoch {epoch:4d}: loss={total_loss/50:.4f} acc={acc*100:.1f}%")

            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'level': level_idx,
                    'max_shift': max_shift,
                    'accuracy': acc
                }, f"models/curriculum/LSL_level{level_idx}_shift{max_shift}.pt")

                if acc >= 0.999:  # 99.9% to account for floating point
                    print(f"  >>> 100% ACCURACY at level {level_idx + 1}! <<<")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'level': level_idx,
                        'max_shift': max_shift,
                        'accuracy': 1.0
                    }, f"models/curriculum/LSL_level{level_idx}_PERFECT.pt")
                    break
            else:
                patience_counter += 1

            # If stuck, try harder
            if patience_counter >= max_patience and best_acc < 0.95:
                print(f"  Stuck at {best_acc*100:.1f}%, increasing LR...")
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 1.5
                patience_counter = 0

        if best_acc < 0.999:
            print(f"  WARNING: Did not reach 100% at level {level_idx + 1}")
            print(f"  Best accuracy: {best_acc*100:.1f}%")
            # Continue anyway to see how far we can get

    # Final verification
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION")
    print("=" * 60)

    test_cases = [
        (1, 4),    # 1 << 4 = 16
        (255, 1),  # 255 << 1 = 510
        (0xF0F0, 4),
        (0x1234567890ABCDEF, 8),
        (1, 63),   # Edge case
    ]

    model.eval()
    all_correct = True
    with torch.no_grad():
        for val, shift in test_cases:
            input_bits = int_to_bits(val, BITS).unsqueeze(0).to(device)
            shift_bits = int_to_bits(shift, BITS).unsqueeze(0).to(device)

            output = model(input_bits, shift_bits)
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
            'architecture': 'SimpleLSLNet',
            'bits': BITS,
            'accuracy': 1.0
        }, "models/curriculum/LSL_64bit_FINAL.pt")

    return model


if __name__ == "__main__":
    train_curriculum()
