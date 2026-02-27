#!/usr/bin/env python3
"""
FORMULA-GUIDED LSL Training

Key insight: We KNOW the exact formula for LSL:
  output[i] = input[i - shift] if i >= shift else 0

This model learns to:
1. Decode shift from binary representation
2. Apply the formula using differentiable soft indexing

The trick is making indexing differentiable while staying close to hard selection.
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


class ShiftDecoderNet(nn.Module):
    """
    Learns to decode binary shift representation to usable form.

    The key insight: instead of trying to learn the whole algorithm,
    just learn the shift decoding. The rest is deterministic!
    """
    def __init__(self, max_bits=64, hidden_dim=256):
        super().__init__()
        self.max_bits = max_bits

        # Decode binary shift to a probability distribution over shift amounts
        self.decoder = nn.Sequential(
            nn.Linear(max_bits, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_bits)  # Logits for each possible shift
        )

        self.register_buffer('temperature', torch.tensor(1.0))

    def set_temperature(self, temp):
        self.temperature.fill_(temp)

    def forward(self, a_bits, shift_bits):
        batch = a_bits.shape[0]
        bits = a_bits.shape[1]
        device = a_bits.device

        # Decode shift to probability distribution
        shift_logits = self.decoder(shift_bits)[:, :bits]  # [batch, bits]

        if self.training:
            # Soft during training
            shift_probs = F.softmax(shift_logits / self.temperature, dim=-1)
        else:
            # Hard during inference
            shift_probs = F.one_hot(shift_logits.argmax(dim=-1), bits).float()

        # Now apply the LSL formula using soft indexing
        # For each output position i, we want input[i - shift]
        # We compute this as: sum over s of P(shift=s) * input[i-s]

        outputs = []
        for i in range(bits):
            # For output position i, compute weighted sum over possible shifts
            value = torch.zeros(batch, device=device)
            valid_prob = torch.zeros(batch, device=device)

            for s in range(bits):
                src_idx = i - s
                if src_idx >= 0:
                    # Valid source position
                    value += shift_probs[:, s] * a_bits[:, src_idx]
                    valid_prob += shift_probs[:, s]
                # If src_idx < 0, output should be 0, so we don't add anything

            outputs.append(value)

        return torch.stack(outputs, dim=1)


class FastShiftNet(nn.Module):
    """
    Optimized version using matrix operations instead of loops.

    Uses the fact that LSL can be represented as matrix multiplication
    with a shift-dependent permutation matrix.
    """
    def __init__(self, max_bits=64, hidden_dim=512):
        super().__init__()
        self.max_bits = max_bits

        # Decode shift to probability distribution
        self.shift_decoder = nn.Sequential(
            nn.Linear(max_bits, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_bits)
        )

        # Pre-compute shift matrices (lower triangular shift patterns)
        self._precompute_shift_matrices(max_bits)

        self.register_buffer('temperature', torch.tensor(1.0))

    def _precompute_shift_matrices(self, bits):
        """Pre-compute the permutation matrix for each shift amount."""
        # shift_matrices[s] is the matrix that implements LSL by s
        # M[i,j] = 1 if j == i - s and i >= s, else 0
        matrices = []
        for s in range(bits):
            M = torch.zeros(bits, bits)
            for i in range(bits):
                j = i - s
                if j >= 0:
                    M[i, j] = 1.0
            matrices.append(M)
        self.register_buffer('shift_matrices', torch.stack(matrices))  # [bits, bits, bits]

    def set_temperature(self, temp):
        self.temperature.fill_(temp)

    def forward(self, a_bits, shift_bits):
        batch = a_bits.shape[0]
        bits = a_bits.shape[1]
        device = a_bits.device

        # Decode shift amount
        shift_logits = self.shift_decoder(shift_bits)[:, :bits]

        if self.training:
            shift_probs = F.softmax(shift_logits / self.temperature, dim=-1)  # [batch, bits]
        else:
            shift_probs = F.one_hot(shift_logits.argmax(dim=-1), bits).float()

        # Get relevant shift matrices
        M = self.shift_matrices[:bits, :bits, :bits]  # [bits, bits, bits]

        # Weighted sum of shift matrices: [batch, bits, bits]
        # M_weighted[b, i, j] = sum_s shift_probs[b, s] * M[s, i, j]
        M_weighted = torch.einsum('bs,sij->bij', shift_probs, M)

        # Apply to input: output = M_weighted @ input
        output = torch.bmm(M_weighted, a_bits.unsqueeze(-1)).squeeze(-1)

        return output


class UltraFastShiftNet(nn.Module):
    """
    Even faster version with learned refinement.

    Architecture:
    1. Shift decoder (like above)
    2. Apply deterministic shift operation
    3. Optional learned refinement layer
    """
    def __init__(self, max_bits=64, hidden_dim=512, use_refinement=True):
        super().__init__()
        self.max_bits = max_bits
        self.use_refinement = use_refinement

        # Shift decoder
        self.shift_decoder = nn.Sequential(
            nn.Linear(max_bits, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_bits)
        )

        # Optional refinement (learns to correct errors)
        if use_refinement:
            self.refine = nn.Sequential(
                nn.Linear(max_bits * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, max_bits)
            )

        self._precompute_shift_matrices(max_bits)
        self.register_buffer('temperature', torch.tensor(1.0))

    def _precompute_shift_matrices(self, bits):
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

    def forward(self, a_bits, shift_bits):
        batch = a_bits.shape[0]
        bits = a_bits.shape[1]
        device = a_bits.device

        # Decode shift
        shift_logits = self.shift_decoder(shift_bits)[:, :bits]

        if self.training:
            shift_probs = F.softmax(shift_logits / self.temperature, dim=-1)
        else:
            shift_probs = F.one_hot(shift_logits.argmax(dim=-1), bits).float()

        # Weighted shift matrices
        M = self.shift_matrices[:bits, :bits, :bits]
        M_weighted = torch.einsum('bs,sij->bij', shift_probs, M)

        # Apply shift
        output = torch.bmm(M_weighted, a_bits.unsqueeze(-1)).squeeze(-1)

        # Optional refinement
        if self.use_refinement:
            refinement_input = torch.cat([output, shift_probs], dim=1)
            correction = torch.sigmoid(self.refine(refinement_input)[:, :bits])
            # Soft gating: output * correction + (1 - output) * (1 - correction)
            # This allows small corrections without destroying the main signal
            output = output + 0.1 * (correction - 0.5)
            output = output.clamp(0, 1)

        return output


def generate_shift_data(batch_size, bits, device, max_shift, op='LSL'):
    """Generate training data."""
    a_bits = torch.randint(0, 2, (batch_size, bits), device=device).float()
    shifts = torch.randint(0, max_shift + 1, (batch_size,), device=device)
    positions = torch.arange(bits, device=device).unsqueeze(0)

    if op == 'LSL':
        src_pos = positions - shifts.unsqueeze(1)
        valid = src_pos >= 0
    else:
        src_pos = positions + shifts.unsqueeze(1)
        valid = src_pos < bits

    src_pos = src_pos.clamp(0, bits - 1)
    result = torch.gather(a_bits, 1, src_pos) * valid.float()
    shift_bits = ((shifts.unsqueeze(1) >> torch.arange(bits, device=device)) & 1).float()

    # Also return ground truth shift for auxiliary loss
    target_shift = F.one_hot(shifts.clamp(0, bits - 1), bits).float()

    return a_bits, shift_bits, result, target_shift


def train(model, device, save_dir, max_bits=64, batch_size=8192, op='LSL'):
    print("\n" + "=" * 70)
    print(f"🎯 FORMULA-GUIDED {op} TRAINING")
    print("=" * 70)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}")

    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    scaler = GradScaler('cuda')

    output_loss_fn = nn.MSELoss()
    shift_loss_fn = nn.CrossEntropyLoss()

    curriculum = [(8, 7), (16, 15), (24, 23), (32, 31), (40, 39), (48, 47), (56, 55), (64, 63)]
    curriculum = [(b, s) for (b, s) in curriculum if b <= max_bits]

    for bits, max_shift in curriculum:
        print(f"\n{'=' * 70}")
        print(f"📊 Training: {bits}-bit, shifts 0-{max_shift}")
        print(f"{'=' * 70}")

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-6)
        best_acc = 0
        consecutive_100 = 0

        for epoch in range(2000):
            model.train()

            # Temperature annealing
            temp = max(0.05, 1.0 - epoch / 500)
            model.set_temperature(temp)

            a_bits, shift_bits, target, target_shift = generate_shift_data(
                batch_size, bits, device, max_shift, op)

            if bits < max_bits:
                a_bits = F.pad(a_bits, (0, max_bits - bits))
                shift_bits = F.pad(shift_bits, (0, max_bits - bits))
                target = F.pad(target, (0, max_bits - bits))
                target_shift = F.pad(target_shift, (0, max_bits - bits))

            optimizer.zero_grad()

            with autocast('cuda', dtype=torch.bfloat16):
                output = model(a_bits, shift_bits)
                loss = output_loss_fn(output[:, :bits], target[:, :bits])

                # Add shift decoder supervision if model has shift_decoder
                if hasattr(model, 'shift_decoder'):
                    shift_logits = model.shift_decoder(shift_bits)[:, :bits]
                    shift_aux_loss = shift_loss_fn(shift_logits, target_shift[:, :bits].argmax(dim=1))
                    loss = loss + 0.3 * shift_aux_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Evaluation
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
                torch.save(model.state_dict(), f"{save_dir}/{op}_{bits}bit_formula_ckpt.pt")

                if correct >= 100.0:
                    consecutive_100 += 1
                    if consecutive_100 >= 3:
                        print(f"    ✅ {bits}-bit: 100% x3 - ADVANCING!")
                        torch.save(model.state_dict(), f"{save_dir}/{op}_{bits}bit_formula_100pct.pt")
                        break
                else:
                    consecutive_100 = 0

        print(f"    Level complete: best={best_acc:.2f}%")

    print("\n" + "=" * 70)
    print(f"🎉 FORMULA-GUIDED {op} TRAINING COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--bits', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--model', default='fast', choices=['simple', 'fast', 'ultra'])
    parser.add_argument('--op', default='LSL', choices=['LSL', 'LSR'])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.model == 'simple':
        model = ShiftDecoderNet(max_bits=args.bits, hidden_dim=512)
    elif args.model == 'fast':
        model = FastShiftNet(max_bits=args.bits, hidden_dim=512)
    else:  # ultra
        model = UltraFastShiftNet(max_bits=args.bits, hidden_dim=512, use_refinement=True)

    model = model.to(args.device)

    train(model, args.device, args.save_dir,
          max_bits=args.bits, batch_size=args.batch_size, op=args.op)


if __name__ == '__main__':
    main()
