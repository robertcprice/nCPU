#!/usr/bin/env python3
"""
TRULY NEURAL LSL - v2
=====================
Two-phase curriculum:
1. PHASE 1: Learn shift decoding (binary → integer/one-hot)
2. PHASE 2: Learn bit routing (source position selection)

NO hardcoded matrices. The model learns both operations.

Key insight: Use RELATIVE position encoding.
For LSL, attention from output[i] to input[j] should peak when j == i - shift.
We can learn this as: attention_score = f(i, j, shift) where f learns the pattern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import random
import os
import sys
import math

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

device = "cuda" if torch.cuda.is_available() else "cpu"


class NeuralLSLv2(nn.Module):
    """
    Two-component neural LSL:
    1. Shift decoder: learns binary → shift amount
    2. Routing network: learns to select correct source bit

    The routing uses learned position embeddings and attention.
    """

    def __init__(self, max_bits=64, d_model=256):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # ============================================
        # Component 1: Shift Decoder
        # Learns to convert binary representation to shift amount
        # ============================================
        self.shift_decoder = nn.Sequential(
            nn.Linear(max_bits, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, max_bits)  # Output: logits for each shift value
        )

        # ============================================
        # Component 2: Routing Network
        # Learns to compute source position for each output position
        # ============================================

        # Learnable position embeddings
        self.pos_embed = nn.Embedding(max_bits, d_model)

        # Routing attention: computes where each output position should look
        # Given (output_pos_embed, shift_embed), predict attention over input positions
        self.routing_query = nn.Linear(d_model * 2, d_model)
        self.routing_key = nn.Linear(d_model, d_model)

        # ============================================
        # Component 3: Validity Network
        # Learns when output should be zero (i < shift)
        # ============================================
        self.validity_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

        self.register_buffer('temperature', torch.tensor(1.0))

    def set_temperature(self, temp):
        self.temperature.fill_(temp)

    def decode_shift(self, shift_bits):
        """Decode binary shift to logits over shift amounts."""
        return self.shift_decoder(shift_bits)

    def forward(self, input_bits, shift_bits, return_aux=False):
        batch = input_bits.shape[0]
        bits = self.max_bits
        device = input_bits.device

        # Step 1: Decode shift amount
        shift_logits = self.decode_shift(shift_bits)

        if self.training:
            shift_probs = F.softmax(shift_logits / self.temperature, dim=-1)
        else:
            shift_probs = F.one_hot(shift_logits.argmax(dim=-1), bits).float()

        # Convert to embedding (weighted sum of position embeddings)
        pos_indices = torch.arange(bits, device=device)
        all_pos_embeds = self.pos_embed(pos_indices)  # [bits, d_model]
        shift_embed = torch.matmul(shift_probs, all_pos_embeds)  # [batch, d_model]

        # Step 2: For each output position, compute routing attention
        outputs = []
        attn_weights_list = []
        validity_logits_list = []

        for out_pos in range(bits):
            # Get output position embedding
            out_pos_embed = self.pos_embed(torch.tensor(out_pos, device=device))
            out_pos_embed = out_pos_embed.unsqueeze(0).expand(batch, -1)  # [batch, d_model]

            # Combine with shift embedding to form query
            query_input = torch.cat([out_pos_embed, shift_embed], dim=-1)  # [batch, d_model*2]
            query = self.routing_query(query_input)  # [batch, d_model]

            # Keys are all input position embeddings
            keys = self.routing_key(all_pos_embeds)  # [bits, d_model]

            # Compute attention: which input position should we look at?
            # This learns the pattern: attend to position (out_pos - shift)
            attn_logits = torch.matmul(query, keys.T) / math.sqrt(self.d_model)  # [batch, bits]

            if self.training:
                attn_weights = F.softmax(attn_logits / self.temperature, dim=-1)
            else:
                attn_weights = F.one_hot(attn_logits.argmax(dim=-1), bits).float()

            attn_weights_list.append(attn_weights)

            # Gather value from input using attention
            gathered = (attn_weights * input_bits).sum(dim=-1)  # [batch]

            # Step 3: Predict validity (should output be zero?)
            validity_input = torch.cat([out_pos_embed, shift_embed], dim=-1)
            validity_logit = self.validity_net(validity_input).squeeze(-1)  # [batch]
            validity_logits_list.append(validity_logit)

            # Final output
            output = gathered * torch.sigmoid(validity_logit)
            outputs.append(output)

        result = torch.stack(outputs, dim=-1)  # [batch, bits]

        if return_aux:
            return result, shift_logits, torch.stack(attn_weights_list, dim=1), torch.stack(validity_logits_list, dim=1)
        return result


def int_to_bits(val, bits=64):
    return torch.tensor([(val >> i) & 1 for i in range(bits)], dtype=torch.float32)


def bits_to_int(bits_t):
    return sum(int(b > 0.5) << i for i, b in enumerate(bits_t.cpu().tolist()))


def generate_batch(batch_size, bits, max_shift, device):
    """Generate data with ground truth for all components."""
    a_bits = torch.randint(0, 2, (batch_size, bits), device=device).float()
    shifts = torch.randint(0, max_shift + 1, (batch_size,), device=device)
    positions = torch.arange(bits, device=device).unsqueeze(0)

    src_pos = positions - shifts.unsqueeze(1)
    valid = src_pos >= 0
    src_pos_clamped = src_pos.clamp(0, bits - 1)
    result = torch.gather(a_bits, 1, src_pos_clamped) * valid.float()

    shift_bits = ((shifts.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()

    # Ground truth for supervision
    target_shift_idx = shifts  # [batch] - the shift amount as index
    target_source_idx = src_pos_clamped  # [batch, bits] - source index for each output
    target_valid = valid.float()  # [batch, bits] - validity mask

    return a_bits, shift_bits, result, target_shift_idx, target_source_idx, target_valid


def evaluate_exact(model, bits, max_shift, device, num_tests=200):
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
    print("TRULY NEURAL LSL v2 - Two Phase Curriculum")
    print("=" * 70)
    print(f"Device: {device}")

    BITS = 64

    model = NeuralLSLv2(max_bits=BITS, d_model=256).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # ==========================================
    # PHASE 1: Train Shift Decoder First
    # ==========================================
    print("\n" + "=" * 70)
    print("PHASE 1: Learning Shift Decoding")
    print("=" * 70)

    optimizer = torch.optim.AdamW(model.shift_decoder.parameters(), lr=2e-3)
    shift_loss_fn = nn.CrossEntropyLoss()

    # Train shift decoder to near-perfect accuracy
    batch_size = 4096
    for epoch in range(500):
        # Generate random shifts and their binary representations
        shifts = torch.randint(0, BITS, (batch_size,), device=device)
        shift_bits = ((shifts.unsqueeze(1) >> torch.arange(BITS, device=device)) & 1).float()

        optimizer.zero_grad()
        shift_logits = model.decode_shift(shift_bits)
        loss = shift_loss_fn(shift_logits, shifts)
        loss.backward()
        optimizer.step()

        # Check accuracy
        pred_shifts = shift_logits.argmax(dim=-1)
        acc = (pred_shifts == shifts).float().mean().item()

        if epoch % 50 == 0 or acc > 0.99:
            print(f"  Shift decoder epoch {epoch}: loss={loss.item():.4f} acc={acc*100:.1f}%")

        if acc > 0.999:
            print(f"  Shift decoder converged at epoch {epoch}!")
            break

    # ==========================================
    # PHASE 2: Train Full Model with Curriculum
    # ==========================================
    print("\n" + "=" * 70)
    print("PHASE 2: Learning Bit Routing with Curriculum")
    print("=" * 70)

    # Now train the full model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scaler = GradScaler('cuda')

    output_loss_fn = nn.MSELoss()
    shift_loss_fn = nn.CrossEntropyLoss()
    routing_loss_fn = nn.CrossEntropyLoss()
    valid_loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs("models/neural_v2", exist_ok=True)

    # Curriculum: 8-bit increments
    curriculum = [7, 15, 23, 31, 39, 47, 55, 63]
    batch_size = 2048

    for level_idx, max_shift in enumerate(curriculum):
        print(f"\n{'='*70}")
        print(f"LEVEL {level_idx + 1}/{len(curriculum)}: shifts 0-{max_shift}")
        print("="*70)

        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-3 * (0.9 ** level_idx)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=200, T_mult=2, eta_min=1e-6
        )

        best_acc = 0
        consecutive_100 = 0
        max_epochs = 3000

        for epoch in range(max_epochs):
            model.train()
            temp = max(0.1, 1.0 - epoch / 800)
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
                output, shift_logits, attn_weights, validity_logits = \
                    model(a_bits, shift_bits, return_aux=True)

                # Main output loss
                main_loss = output_loss_fn(output, target)

                # Shift decoder loss (keep it accurate)
                shift_loss = shift_loss_fn(shift_logits, target_shift)

                # Routing loss (teach correct attention patterns)
                routing_loss = 0
                for i in range(BITS):
                    routing_loss += routing_loss_fn(attn_weights[:, i, :], target_source[:, i])
                routing_loss /= BITS

                # Validity loss
                valid_loss = valid_loss_fn(validity_logits, target_valid)

                # Combined - reduce auxiliary weights over time
                aux_weight = max(0.05, 0.3 - epoch / 2000)
                loss = main_loss + aux_weight * (shift_loss + routing_loss + valid_loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            acc = evaluate_exact(model, BITS, max_shift, device, num_tests=150)

            if epoch % 100 == 0 or acc > best_acc:
                print(f"  Epoch {epoch:4d}: loss={loss.item():.4f} acc={acc*100:.1f}% temp={temp:.2f}")

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), f"models/neural_v2/LSL_shift{max_shift}_best.pt")

            if acc >= 0.998:
                consecutive_100 += 1
                if consecutive_100 >= 5:
                    print(f"  >>> 100% at level {level_idx + 1}! <<<")
                    torch.save(model.state_dict(), f"models/neural_v2/LSL_shift{max_shift}_PERFECT.pt")
                    break
            else:
                consecutive_100 = 0

        if best_acc < 0.998:
            print(f"  Level {level_idx+1} best: {best_acc*100:.1f}%")

    # Final test
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
        torch.save(model.state_dict(), "models/neural_v2/LSL_64bit_FINAL.pt")


if __name__ == "__main__":
    train()
