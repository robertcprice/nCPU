#!/usr/bin/env python3
"""
FULLY NEURAL LSL TRAINING
=========================
NO pre-computed matrices. NO hardcoded operations.
The model must LEARN:
1. How to decode the shift amount from binary
2. How to route bits from input to output positions

Curriculum approach:
- Phase 1: Learn shift decoding (auxiliary task)
- Phase 2: Learn bit routing for small shifts (0-7)
- Phase 3: Incrementally expand shift range (8-bit increments)

Architecture: Attention-based routing where each output position
learns to attend to the correct input position.
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


class FullyNeuralLSL(nn.Module):
    """
    Fully learned LSL - no hardcoded shift matrices.

    The model learns:
    1. Position embeddings that encode bit positions
    2. Shift encoding that captures the shift amount
    3. Attention mechanism to route input[i-s] -> output[i]
    4. Zero-masking for positions where i < shift
    """

    def __init__(self, max_bits=64, d_model=256, n_heads=8, n_layers=4):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # Learnable position embeddings for input and output positions
        self.input_pos_embed = nn.Embedding(max_bits, d_model)
        self.output_pos_embed = nn.Embedding(max_bits, d_model)

        # Bit value embedding (0 or 1 -> d_model)
        self.bit_embed = nn.Linear(1, d_model)

        # Shift encoder - learns to process binary shift representation
        self.shift_encoder = nn.Sequential(
            nn.Linear(max_bits, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Cross-attention layers: output positions attend to input positions
        # This is where the model learns the routing!
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
            for _ in range(n_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers * 2)
        ])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            )
            for _ in range(n_layers)
        ])

        # Output head - predicts the bit value
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        # Validity predictor - learns when output should be 0 (i < shift)
        self.validity_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),  # output_pos + shift
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, input_bits, shift_bits, return_attention=False):
        """
        Args:
            input_bits: [batch, max_bits] - the value to shift
            shift_bits: [batch, max_bits] - shift amount in binary
        Returns:
            [batch, max_bits] - shifted result
        """
        batch = input_bits.shape[0]
        bits = self.max_bits
        device = input_bits.device

        # Create position indices
        positions = torch.arange(bits, device=device)

        # Embed input: bit values + position
        input_embedded = self.bit_embed(input_bits.unsqueeze(-1))  # [batch, bits, d_model]
        input_embedded = input_embedded + self.input_pos_embed(positions)

        # Embed output positions (queries)
        output_queries = self.output_pos_embed(positions).unsqueeze(0).expand(batch, -1, -1)

        # Encode shift amount
        shift_enc = self.shift_encoder(shift_bits)  # [batch, d_model]

        # Add shift encoding to both input and output representations
        input_embedded = input_embedded + shift_enc.unsqueeze(1)
        output_queries = output_queries + shift_enc.unsqueeze(1)

        # Cross-attention: output positions attend to input positions
        # This is where the model learns: "output[i] should look at input[i-shift]"
        attn_weights_all = []
        x = output_queries

        for i, (attn, ffn) in enumerate(zip(self.cross_attn_layers, self.ffn_layers)):
            # Cross attention
            attended, attn_weights = attn(
                x, input_embedded, input_embedded,
                need_weights=True
            )
            attn_weights_all.append(attn_weights)
            x = self.layer_norms[i*2](x + attended)

            # FFN
            x = self.layer_norms[i*2 + 1](x + ffn(x))

        # Predict bit values from attended representation
        bit_logits = self.output_head(x).squeeze(-1)  # [batch, bits]

        # Predict validity (should this position output 0?)
        validity_input = torch.cat([
            x,  # [batch, bits, d_model]
            shift_enc.unsqueeze(1).expand(-1, bits, -1)  # [batch, bits, d_model]
        ], dim=-1)
        validity_logits = self.validity_head(validity_input).squeeze(-1)  # [batch, bits]

        # Combine: output = sigmoid(bit_logits) * sigmoid(validity)
        output = torch.sigmoid(bit_logits) * torch.sigmoid(validity_logits)

        if return_attention:
            return output, attn_weights_all, validity_logits
        return output


class SimpleNeuralLSL(nn.Module):
    """
    Simpler fully-neural approach using direct position prediction.

    For each output position i, the network predicts:
    1. Which input position to copy from (learned routing)
    2. Whether the output should be valid (i >= shift)

    No hardcoded matrices - everything is learned!
    """

    def __init__(self, max_bits=64, hidden_dim=512):
        super().__init__()
        self.max_bits = max_bits

        # Shift encoder
        self.shift_encoder = nn.Sequential(
            nn.Linear(max_bits, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Position encoder (learns position semantics)
        self.pos_encoder = nn.Sequential(
            nn.Linear(max_bits, hidden_dim),  # one-hot position
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Source position predictor: given output_pos and shift, predict source_pos
        # This learns the "i - shift" computation!
        self.source_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_bits)  # logits over source positions
        )

        # Validity predictor: given output_pos and shift, is output_pos >= shift?
        self.validity_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.register_buffer('temperature', torch.tensor(1.0))

    def set_temperature(self, temp):
        self.temperature.fill_(temp)

    def forward(self, input_bits, shift_bits, return_aux=False):
        batch = input_bits.shape[0]
        bits = self.max_bits
        device = input_bits.device

        # Encode shift
        shift_enc = self.shift_encoder(shift_bits)  # [batch, hidden]

        outputs = []
        source_logits_all = []
        validity_logits_all = []

        for out_pos in range(bits):
            # One-hot encode output position
            out_pos_onehot = torch.zeros(batch, bits, device=device)
            out_pos_onehot[:, out_pos] = 1.0
            pos_enc = self.pos_encoder(out_pos_onehot)

            # Combine position and shift encodings
            combined = torch.cat([pos_enc, shift_enc], dim=-1)

            # Predict source position (learns "out_pos - shift")
            source_logits = self.source_predictor(combined)
            source_logits_all.append(source_logits)

            if self.training:
                source_probs = F.softmax(source_logits / self.temperature, dim=-1)
            else:
                # Hard selection at inference
                source_probs = F.one_hot(source_logits.argmax(dim=-1), bits).float()

            # Gather from input using learned routing
            gathered = (source_probs * input_bits).sum(dim=-1)

            # Predict validity (learns "out_pos >= shift")
            validity_logit = self.validity_predictor(combined).squeeze(-1)
            validity_logits_all.append(validity_logit)

            # Output = gathered value * validity
            output = gathered * torch.sigmoid(validity_logit)
            outputs.append(output)

        result = torch.stack(outputs, dim=-1)

        if return_aux:
            return result, torch.stack(source_logits_all, dim=1), torch.stack(validity_logits_all, dim=1)
        return result


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

    # Ground truth source positions (for auxiliary loss)
    target_source = src_pos_clamped  # [batch, bits]
    target_valid = valid.float()

    return a_bits, shift_bits, result, target_source, target_valid


def evaluate_exact(model, bits, max_shift, device, num_tests=300):
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
    print("FULLY NEURAL LSL TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print("NO hardcoded matrices - model learns routing!")

    BITS = 64

    model = SimpleNeuralLSL(max_bits=BITS, hidden_dim=512).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scaler = GradScaler('cuda')

    output_loss_fn = nn.MSELoss()
    source_loss_fn = nn.CrossEntropyLoss()
    valid_loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs("models/fully_neural", exist_ok=True)

    # Curriculum: 8-bit increments
    curriculum = [7, 15, 23, 31, 39, 47, 55, 63]

    batch_size = 2048  # Smaller batch for more complex model

    for level_idx, max_shift in enumerate(curriculum):
        print(f"\n{'='*70}")
        print(f"LEVEL {level_idx + 1}/{len(curriculum)}: shifts 0-{max_shift}")
        print("="*70)

        # Learning rate decay per level
        base_lr = 1e-3 * (0.9 ** level_idx)
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=300, T_mult=2, eta_min=1e-6
        )

        best_acc = 0
        consecutive_100 = 0
        epochs_at_level = 0
        max_epochs = 5000

        while epochs_at_level < max_epochs:
            model.train()

            # Temperature annealing
            temp = max(0.1, 1.0 - epochs_at_level / 1000)
            model.set_temperature(temp)

            # Generate data
            a_bits, shift_bits, target, target_source, target_valid = generate_batch(
                batch_size, BITS, max_shift, device
            )

            # Mix previous level data
            if level_idx > 0:
                prev_max = curriculum[max(0, level_idx - 1)]
                prev_a, prev_s, prev_t, prev_src, prev_v = generate_batch(
                    batch_size // 5, BITS, prev_max, device
                )
                n = batch_size * 4 // 5
                a_bits = torch.cat([a_bits[:n], prev_a])
                shift_bits = torch.cat([shift_bits[:n], prev_s])
                target = torch.cat([target[:n], prev_t])
                target_source = torch.cat([target_source[:n], prev_src])
                target_valid = torch.cat([target_valid[:n], prev_v])

            optimizer.zero_grad()

            with autocast('cuda', dtype=torch.bfloat16):
                output, source_logits, validity_logits = model(
                    a_bits, shift_bits, return_aux=True
                )

                # Main output loss
                main_loss = output_loss_fn(output, target)

                # Auxiliary: source position supervision
                # Teaches the model WHERE to look
                source_loss = 0
                for i in range(BITS):
                    source_loss += source_loss_fn(source_logits[:, i, :], target_source[:, i])
                source_loss /= BITS

                # Auxiliary: validity supervision
                # Teaches the model WHEN to output zero
                valid_loss = valid_loss_fn(validity_logits, target_valid)

                # Combined loss - heavier aux supervision early
                aux_weight = max(0.1, 0.5 - epochs_at_level / 2000)
                loss = main_loss + aux_weight * (source_loss + valid_loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Evaluate
            acc = evaluate_exact(model, BITS, max_shift, device, num_tests=200)

            if epochs_at_level % 100 == 0 or acc > best_acc:
                print(f"  Epoch {epochs_at_level:4d}: loss={loss.item():.4f} "
                      f"acc={acc*100:.1f}% temp={temp:.2f}")

            if acc > best_acc:
                best_acc = acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'level': level_idx,
                    'max_shift': max_shift,
                    'accuracy': acc
                }, f"models/fully_neural/LSL_shift{max_shift}_best.pt")

            if acc >= 0.998:
                consecutive_100 += 1
                if consecutive_100 >= 5:
                    print(f"  >>> 100% ACCURACY x5 at level {level_idx + 1}! <<<")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'level': level_idx,
                        'max_shift': max_shift,
                        'accuracy': 1.0
                    }, f"models/fully_neural/LSL_shift{max_shift}_PERFECT.pt")
                    break
            else:
                consecutive_100 = 0

            epochs_at_level += 1

        if best_acc < 0.998:
            print(f"  Level {level_idx+1} best: {best_acc*100:.1f}%")

    # Final verification
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
            'architecture': 'SimpleNeuralLSL',
            'bits': BITS,
            'accuracy': 1.0
        }, "models/fully_neural/LSL_64bit_FINAL.pt")

    return model


if __name__ == "__main__":
    train()
