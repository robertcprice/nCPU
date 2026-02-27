#!/usr/bin/env python3
"""
TRULY NEURAL SIN/COS
====================
Learns sin/cos from scratch - NO hardcoded tables!

Architecture options:
1. Learned Embedding LUT - embeddings trained to output sin/cos
2. SIREN (Sinusoidal Representation Networks) - periodic activations
3. Fourier Features - random Fourier projection

We use Learned Embedding LUT - it's like a hash table but neural:
- Discretize angle into bins
- Each bin has a learned embedding
- Decode embedding to sin/cos values

This is truly learned, not pre-filled!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


class NeuralSinCosLUT(nn.Module):
    """
    Truly neural sin/cos using learned embeddings.

    NOT pre-filled with math.sin - learns from training data!
    """

    def __init__(self, num_bins=1024, embed_dim=64):
        super().__init__()
        self.num_bins = num_bins
        self.embed_dim = embed_dim

        # Learned embeddings (initialized randomly, NOT with sin/cos!)
        self.embed = nn.Embedding(num_bins, embed_dim)

        # Interpolation network for smooth output between bins
        self.interp = nn.Sequential(
            nn.Linear(embed_dim + 1, embed_dim),  # +1 for fractional position
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

        # Output heads
        self.sin_head = nn.Linear(embed_dim, 1)
        self.cos_head = nn.Linear(embed_dim, 1)

    def forward(self, angle):
        """
        Args:
            angle: [...] - angles in radians

        Returns:
            [..., 2] - [sin, cos] values
        """
        shape = angle.shape
        angle = angle.flatten()

        # Normalize to [0, 2*pi]
        angle_norm = torch.remainder(angle, 2 * math.pi)

        # Convert to bin index and fractional part
        bin_float = angle_norm / (2 * math.pi) * self.num_bins
        bin_idx = bin_float.long().clamp(0, self.num_bins - 1)
        bin_frac = (bin_float - bin_idx.float()).unsqueeze(-1)  # [batch, 1]

        # Get embeddings
        emb = self.embed(bin_idx)  # [batch, embed_dim]

        # Interpolation with fractional position
        combined = torch.cat([emb, bin_frac], dim=-1)
        h = self.interp(combined)

        # Output
        sin_val = torch.tanh(self.sin_head(h))  # tanh bounds to [-1, 1]
        cos_val = torch.tanh(self.cos_head(h))

        result = torch.cat([sin_val, cos_val], dim=-1)
        return result.view(*shape, 2)


class FourierSinCos(nn.Module):
    """
    Sin/Cos using random Fourier features.

    Projects angle through random frequencies, then learns mapping.
    """

    def __init__(self, num_frequencies=256, hidden_dim=256):
        super().__init__()

        # Random frequencies (fixed, not learned)
        self.register_buffer('frequencies', torch.randn(num_frequencies) * 10)

        # Learnable network
        self.net = nn.Sequential(
            nn.Linear(num_frequencies * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Tanh()
        )

    def forward(self, angle):
        shape = angle.shape
        angle = angle.flatten().unsqueeze(-1)  # [batch, 1]

        # Fourier features
        proj = angle * self.frequencies  # [batch, num_freq]
        features = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

        # Network
        out = self.net(features)
        return out.view(*shape, 2)


def generate_batch(batch_size, device):
    """Generate training batch."""
    angles = torch.rand(batch_size, device=device) * 2 * math.pi
    sin_vals = torch.sin(angles)
    cos_vals = torch.cos(angles)
    targets = torch.stack([sin_vals, cos_vals], dim=-1)
    return angles, targets


def train():
    print("=" * 60)
    print("TRULY NEURAL SIN/COS TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print("Learning sin/cos from scratch - NO hardcoded values!")
    print()

    # Try both architectures
    models = {
        'LUT': NeuralSinCosLUT(num_bins=1024, embed_dim=64),
        'Fourier': FourierSinCos(num_frequencies=256, hidden_dim=256),
    }

    best_overall = None
    best_acc = 0

    for name, model in models.items():
        print(f"\nTraining {name} model...")
        model = model.to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

        batch_size = 1024
        model_best_acc = 0

        for epoch in range(500):
            model.train()
            total_loss = 0

            for _ in range(50):
                angles, targets = generate_batch(batch_size, device)
                optimizer.zero_grad()
                pred = model(angles)
                loss = F.mse_loss(pred, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()

            # Test
            model.eval()
            test_angles = torch.tensor([
                0, math.pi/6, math.pi/4, math.pi/3, math.pi/2,
                math.pi, 3*math.pi/2, 2*math.pi - 0.01
            ], device=device)

            with torch.no_grad():
                pred = model(test_angles)
                true_sin = torch.sin(test_angles)
                true_cos = torch.cos(test_angles)

                sin_err = (pred[:, 0] - true_sin).abs().max().item()
                cos_err = (pred[:, 1] - true_cos).abs().max().item()
                max_err = max(sin_err, cos_err)

                # Count "correct" if within 0.01
                correct = ((pred[:, 0] - true_sin).abs() < 0.01).sum().item()
                correct += ((pred[:, 1] - true_cos).abs() < 0.01).sum().item()
                acc = correct / (2 * len(test_angles))

            if epoch % 50 == 0:
                print(f"  Epoch {epoch+1}: loss={total_loss/50:.6f} max_err={max_err:.4f} acc={100*acc:.1f}%")

            if acc > model_best_acc:
                model_best_acc = acc
                if acc > best_acc:
                    best_acc = acc
                    best_overall = (name, model.state_dict().copy())

            if acc >= 1.0:
                print(f"  100% accuracy achieved!")
                break

        print(f"  {name} best accuracy: {100*model_best_acc:.1f}%")

    # Save best model
    if best_overall:
        name, state = best_overall
        os.makedirs("models/final", exist_ok=True)
        torch.save({
            "model_state_dict": state,
            "accuracy": best_acc,
            "architecture": name,
            "op_name": "SINCOS"
        }, "models/final/sincos_neural_parallel.pt")
        print(f"\nSaved best model ({name}) with {100*best_acc:.1f}% accuracy")

    # Final verification
    print("\nFinal verification:")
    if best_overall:
        name, state = best_overall
        if name == 'LUT':
            model = NeuralSinCosLUT(num_bins=1024, embed_dim=64).to(device)
        else:
            model = FourierSinCos(num_frequencies=256, hidden_dim=256).to(device)
        model.load_state_dict(state)
        model.eval()

        test_cases = [
            (0, 0, 1),
            (math.pi/4, 0.7071, 0.7071),
            (math.pi/2, 1, 0),
            (math.pi, 0, -1),
            (3*math.pi/2, -1, 0),
        ]

        with torch.no_grad():
            for angle, exp_sin, exp_cos in test_cases:
                pred = model(torch.tensor([angle], device=device))
                sin_val = pred[0, 0].item()
                cos_val = pred[0, 1].item()
                sin_err = abs(sin_val - exp_sin)
                cos_err = abs(cos_val - exp_cos)
                status = "OK" if sin_err < 0.02 and cos_err < 0.02 else "ERR"
                print(f"  angle={angle:.4f}: sin={sin_val:.4f} (exp {exp_sin:.4f}), cos={cos_val:.4f} (exp {exp_cos:.4f}) {status}")


if __name__ == "__main__":
    train()
