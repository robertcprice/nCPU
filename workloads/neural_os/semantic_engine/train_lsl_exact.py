#!/usr/bin/env python3
"""
EXACT LSL Training - Force exact attention patterns

Key insight: LSL is DETERMINISTIC - we know exactly where attention should go.
For output[i] with shift s: attend to input[i-s] with probability 1 (if valid).

This model uses:
1. Auxiliary supervision on attention distributions
2. Straight-through estimator for hard attention
3. Separate validity network
4. Temperature annealing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import os
import sys
import argparse
import math

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)


class ExactPointerNet(nn.Module):
    """
    Pointer Network with explicit attention supervision.

    For LSL by shift s: output[i] = input[i-s] if i >= s else 0

    Architecture:
    1. Shift decoder: binary → integer shift amount
    2. Index predictor: for each output position, predict source index
    3. Validity predictor: is i >= shift?
    4. Gather with straight-through estimator
    """
    def __init__(self, max_bits=64, d_model=256):
        super().__init__()
        self.max_bits = max_bits
        self.d_model = d_model

        # Shift decoder: learns to convert binary to integer conceptually
        # Output: embedding that encodes the shift amount
        self.shift_encoder = nn.Sequential(
            nn.Linear(max_bits, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Position embeddings (separate for input and output positions)
        self.out_pos_embed = nn.Embedding(max_bits, d_model)
        self.in_pos_embed = nn.Embedding(max_bits, d_model)

        # Index predictor: given output position and shift, predict attention logits
        self.index_predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, max_bits)
        )

        # Validity predictor: given output position and shift, is i >= shift?
        self.validity_predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

        # Temperature for attention (annealed during training)
        self.register_buffer('temperature', torch.tensor(1.0))

    def set_temperature(self, temp):
        self.temperature.fill_(temp)

    def forward(self, a_bits, shift_bits, return_aux=False):
        batch = a_bits.shape[0]
        bits = a_bits.shape[1]
        device = a_bits.device

        # Encode shift amount
        shift_enc = self.shift_encoder(shift_bits)  # [batch, d_model]

        # Get input position embeddings
        in_pos_ids = torch.arange(bits, device=device)
        in_pos_embs = self.in_pos_embed(in_pos_ids)  # [bits, d_model]

        outputs = []
        attention_logits = []
        validity_logits = []

        for i in range(bits):
            # Output position embedding
            out_pos_emb = self.out_pos_embed(torch.tensor(i, device=device))
            out_pos_emb = out_pos_emb.unsqueeze(0).expand(batch, -1)  # [batch, d_model]

            # Combine output position with shift encoding
            combined = torch.cat([out_pos_emb, shift_enc], dim=1)  # [batch, d_model*2]

            # Predict attention distribution over input positions
            attn_logits = self.index_predictor(combined)[:, :bits]  # [batch, bits]
            attention_logits.append(attn_logits)

            # Apply temperature-scaled softmax
            if self.training:
                # Soft attention during training for gradients
                attn = F.softmax(attn_logits / self.temperature, dim=-1)
            else:
                # Hard attention during inference
                attn = F.one_hot(attn_logits.argmax(dim=-1), bits).float()

            # Gather value using attention
            pointed_value = (attn * a_bits).sum(dim=1)  # [batch]

            # Predict validity
            valid_logit = self.validity_predictor(combined).squeeze(-1)  # [batch]
            validity_logits.append(valid_logit)

            # Final output: pointed_value * validity
            output = pointed_value * torch.sigmoid(valid_logit)
            outputs.append(output)

        result = torch.stack(outputs, dim=1)  # [batch, bits]

        if return_aux:
            return result, torch.stack(attention_logits, dim=1), torch.stack(validity_logits, dim=1)
        return result


class DecomposedShiftNet(nn.Module):
    """
    Fully decomposed shift network.

    Learns each component separately:
    1. Shift decoding (binary → one-hot shift amount)
    2. Per-position: compute (position - shift) mod bits
    3. Per-position: is position >= shift?
    4. Gather from input

    Uses supervision on intermediate representations.
    """
    def __init__(self, max_bits=64, hidden_dim=512):
        super().__init__()
        self.max_bits = max_bits

        # Shift decoder: binary → shift amount (as one-hot)
        self.shift_decoder = nn.Sequential(
            nn.Linear(max_bits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_bits)  # One-hot shift prediction
        )

        # Index computer: for each output position, compute source index
        # Input: output_position (one-hot) + decoded_shift
        self.index_net = nn.Sequential(
            nn.Linear(max_bits * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_bits)  # Source index (one-hot)
        )

        # Validity network: is output_position >= shift?
        self.validity_net = nn.Sequential(
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

        # Decode shift amount
        shift_logits = self.shift_decoder(shift_bits)[:, :bits]  # [batch, bits]
        if self.training:
            shift_soft = F.softmax(shift_logits / self.temperature, dim=-1)
        else:
            shift_soft = F.one_hot(shift_logits.argmax(dim=-1), bits).float()

        outputs = []
        index_logits_list = []
        validity_logits_list = []

        for i in range(bits):
            # One-hot output position
            out_pos = torch.zeros(batch, self.max_bits, device=device)
            out_pos[:, i] = 1.0

            # Combine with decoded shift
            combined = torch.cat([out_pos[:, :bits], shift_soft], dim=1)

            # Predict source index
            idx_logits = self.index_net(combined)[:, :bits]
            index_logits_list.append(idx_logits)

            if self.training:
                idx_soft = F.softmax(idx_logits / self.temperature, dim=-1)
            else:
                idx_soft = F.one_hot(idx_logits.argmax(dim=-1), bits).float()

            # Gather value
            pointed_value = (idx_soft * a_bits).sum(dim=1)

            # Predict validity
            valid_logit = self.validity_net(combined).squeeze(-1)
            validity_logits_list.append(valid_logit)

            output = pointed_value * torch.sigmoid(valid_logit)
            outputs.append(output)

        result = torch.stack(outputs, dim=1)

        if return_aux:
            return (result,
                    shift_logits,
                    torch.stack(index_logits_list, dim=1),
                    torch.stack(validity_logits_list, dim=1))
        return result


def generate_shift_data(batch_size, bits, device, max_shift, op='LSL'):
    """Generate training data with ground truth for supervision."""
    a_bits = torch.randint(0, 2, (batch_size, bits), device=device).float()
    shifts = torch.randint(0, max_shift + 1, (batch_size,), device=device)
    positions = torch.arange(bits, device=device).unsqueeze(0)

    if op == 'LSL':
        src_pos = positions - shifts.unsqueeze(1)
        valid = src_pos >= 0
    else:  # LSR
        src_pos = positions + shifts.unsqueeze(1)
        valid = src_pos < bits

    src_pos_clamped = src_pos.clamp(0, bits - 1)
    result = torch.gather(a_bits, 1, src_pos_clamped) * valid.float()
    shift_bits = ((shifts.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()

    # Ground truth for auxiliary supervision
    # Target attention: one-hot at src_pos (or uniform if invalid)
    target_indices = src_pos_clamped  # [batch, bits] - which input each output should attend to
    target_validity = valid.float()   # [batch, bits] - which outputs are valid

    # One-hot shift amount
    target_shift = F.one_hot(shifts.clamp(0, bits - 1), bits).float()

    return a_bits, shift_bits, result, target_indices, target_validity, target_shift


def train(model, device, save_dir, max_bits=64, batch_size=4096, op='LSL'):
    print("\n" + "=" * 70)
    print(f"🎯 EXACT LSL TRAINING - Supervised Attention")
    print("=" * 70)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scaler = GradScaler('cuda')

    # Loss functions
    output_loss_fn = nn.MSELoss()
    attn_loss_fn = nn.CrossEntropyLoss()
    valid_loss_fn = nn.BCEWithLogitsLoss()

    # Curriculum
    curriculum = [(8, 7), (16, 15), (24, 23), (32, 31), (40, 39), (48, 47), (56, 55), (64, 63)]
    curriculum = [(b, s) for (b, s) in curriculum if b <= max_bits]

    # Check for existing checkpoints and find starting point
    start_level = 0
    for i, (bits, _) in enumerate(curriculum):
        ckpt_path = f"{save_dir}/{op}_{bits}bit_exact_100pct.pt"
        if os.path.exists(ckpt_path):
            print(f"✅ Found 100% checkpoint for {bits}-bit: {ckpt_path}")
            start_level = i + 1
            # Load the latest checkpoint
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            # Try loading best checkpoint if no 100% exists
            best_ckpt = f"{save_dir}/{op}_{bits}bit_exact_ckpt.pt"
            if os.path.exists(best_ckpt) and start_level == i:
                print(f"📂 Loading best checkpoint for {bits}-bit: {best_ckpt}")
                model.load_state_dict(torch.load(best_ckpt, map_location=device))
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

            # Temperature annealing: start at 1.0, decay to 0.1
            temp = max(0.1, 1.0 - epoch / 1000)
            model.set_temperature(temp)

            # Generate data with ground truth
            a_bits, shift_bits, target, target_indices, target_validity, target_shift = \
                generate_shift_data(batch_size, bits, device, max_shift, op)

            # Pad to max_bits if needed
            if bits < max_bits:
                a_bits = F.pad(a_bits, (0, max_bits - bits))
                shift_bits = F.pad(shift_bits, (0, max_bits - bits))
                target = F.pad(target, (0, max_bits - bits))

            optimizer.zero_grad()

            with autocast('cuda', dtype=torch.bfloat16):
                if isinstance(model, DecomposedShiftNet):
                    output, shift_logits, idx_logits, valid_logits = model(a_bits, shift_bits, return_aux=True)

                    # Main output loss
                    main_loss = output_loss_fn(output[:, :bits], target[:, :bits])

                    # Shift decoding loss
                    shift_loss = attn_loss_fn(shift_logits, target_shift[:, :bits].argmax(dim=1))

                    # Index prediction loss (for each position)
                    idx_loss = 0
                    for i in range(bits):
                        idx_loss += attn_loss_fn(idx_logits[:, i, :], target_indices[:, i])
                    idx_loss /= bits

                    # Validity loss
                    valid_loss = valid_loss_fn(valid_logits[:, :bits], target_validity)

                    # Combined loss with weights
                    loss = main_loss + 0.5 * shift_loss + 0.3 * idx_loss + 0.2 * valid_loss

                else:  # ExactPointerNet
                    output, attn_logits, valid_logits = model(a_bits, shift_bits, return_aux=True)

                    # Main output loss
                    main_loss = output_loss_fn(output[:, :bits], target[:, :bits])

                    # Attention supervision loss
                    attn_loss = 0
                    for i in range(bits):
                        attn_loss += attn_loss_fn(attn_logits[:, i, :], target_indices[:, i])
                    attn_loss /= bits

                    # Validity loss
                    valid_loss = valid_loss_fn(valid_logits[:, :bits], target_validity)

                    # Combined loss
                    loss = main_loss + 0.5 * attn_loss + 0.3 * valid_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Accuracy: threshold at 0.5
            with torch.no_grad():
                model.eval()
                model.set_temperature(0.01)  # Nearly hard for eval
                eval_output = model(a_bits, shift_bits)
                pred = (eval_output[:, :bits] > 0.5).float()
                correct = (pred == target[:, :bits]).all(dim=1).float().mean().item() * 100

            if epoch % 100 == 0 or correct > best_acc:
                print(f"    Epoch {epoch:4d}: loss={loss.item():.4f}, acc={correct:.2f}%, temp={temp:.3f}")

            if correct > best_acc:
                best_acc = correct
                torch.save(model.state_dict(), f"{save_dir}/{op}_{bits}bit_exact_ckpt.pt")

            # Track consecutive 100% epochs (FIXED: moved outside best_acc check)
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
    print(f"🎉 EXACT LSL TRAINING COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--bits', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--model', default='exact', choices=['exact', 'decomposed'])
    parser.add_argument('--op', default='LSL', choices=['LSL', 'LSR'])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.model == 'exact':
        model = ExactPointerNet(max_bits=args.bits, d_model=384)
    else:  # decomposed
        model = DecomposedShiftNet(max_bits=args.bits, hidden_dim=768)

    model = model.to(args.device)

    train(model, args.device, args.save_dir,
          max_bits=args.bits, batch_size=args.batch_size, op=args.op)


if __name__ == '__main__':
    main()
