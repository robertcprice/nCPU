#!/usr/bin/env python3
"""
NEURAL 19-BIT BRANCH OFFSET EXTRACTOR
=====================================

Trains a neural network to extract 19-bit branch offsets from:
- B.cond (conditional branch): offset at bits [23:5]
- CBZ/CBNZ (compare and branch): offset at bits [23:5]

This complements the 26-bit extractor used for B/BL instructions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import random
import time

# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Training on: {device}")


class NeuralBranch19Extractor(nn.Module):
    """
    Neural extractor for 19-bit branch offsets.

    Input: 32 instruction bits
    Output: 19 offset bits (from positions [23:5])

    Architecture: Transformer-based for learning bit relationships
    """

    def __init__(self, d_model: int = 128):
        super().__init__()

        # Embed each input bit
        self.bit_embed = nn.Linear(1, d_model)

        # Positional encoding for 32 bit positions
        self.pos_embed = nn.Parameter(torch.randn(32, d_model) * 0.02)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Output heads for 19 offset bits
        self.offset_head = nn.Linear(d_model, 19)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bits: [batch, 32] instruction bits
        Returns:
            offset_logits: [batch, 19] logits for offset bits
        """
        # Embed bits: [batch, 32, d_model]
        x = self.bit_embed(bits.unsqueeze(-1))
        x = x + self.pos_embed

        # Transform
        x = self.transformer(x)

        # Global pooling
        x = x.mean(dim=1)  # [batch, d_model]

        # Output offset bits
        offset_logits = self.offset_head(x)  # [batch, 19]

        return offset_logits


def generate_bcond(offset19: int, cond: int = 0) -> int:
    """Generate B.cond instruction: 0x54000000 | (imm19 << 5) | cond"""
    return 0x54000000 | ((offset19 & 0x7FFFF) << 5) | (cond & 0xF)


def generate_cbz(rt: int, offset19: int, sf: int = 1) -> int:
    """Generate CBZ instruction: sf=1 for 64-bit (0xB4), sf=0 for 32-bit (0x34)"""
    return (sf << 31) | (0x34 << 24) | ((offset19 & 0x7FFFF) << 5) | (rt & 0x1F)


def generate_cbnz(rt: int, offset19: int, sf: int = 1) -> int:
    """Generate CBNZ instruction: sf=1 for 64-bit (0xB5), sf=0 for 32-bit (0x35)"""
    return (sf << 31) | (0x35 << 24) | ((offset19 & 0x7FFFF) << 5) | (rt & 0x1F)


def inst_to_bits(inst: int) -> torch.Tensor:
    """Convert instruction to bit tensor"""
    return torch.tensor([float((inst >> i) & 1) for i in range(32)])


def generate_training_data(num_samples: int = 50000):
    """Generate training data for all 19-bit branch instructions"""
    data = []

    for _ in range(num_samples):
        # Random 19-bit offset (full range)
        offset19 = random.randint(0, 0x7FFFF)

        # Randomly choose instruction type
        inst_type = random.choice(['bcond', 'cbz', 'cbnz'])

        if inst_type == 'bcond':
            cond = random.randint(0, 15)
            inst = generate_bcond(offset19, cond)
        elif inst_type == 'cbz':
            rt = random.randint(0, 30)
            sf = random.choice([0, 1])
            inst = generate_cbz(rt, offset19, sf)
        else:  # cbnz
            rt = random.randint(0, 30)
            sf = random.choice([0, 1])
            inst = generate_cbnz(rt, offset19, sf)

        bits = inst_to_bits(inst)

        # Target: 19 offset bits
        offset_bits = torch.tensor([float((offset19 >> i) & 1) for i in range(19)])

        data.append((bits, offset_bits))

    return data


def train_extractor():
    """Train the 19-bit branch offset extractor"""
    print("=" * 70)
    print("   TRAINING NEURAL 19-BIT BRANCH OFFSET EXTRACTOR")
    print("=" * 70)

    # Generate data
    print("\n[1] Generating training data...")
    train_data = generate_training_data(50000)
    val_data = generate_training_data(5000)
    print(f"    Training samples: {len(train_data)}")
    print(f"    Validation samples: {len(val_data)}")

    # Create model
    model = NeuralBranch19Extractor(d_model=128).to(device)
    print(f"\n[2] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.BCEWithLogitsLoss()

    # Training
    print("\n[3] Training...")
    batch_size = 256
    best_accuracy = 0

    for epoch in range(50):
        model.train()
        random.shuffle(train_data)

        total_loss = 0
        num_batches = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            bits = torch.stack([b[0] for b in batch]).to(device)
            offset_targets = torch.stack([b[1] for b in batch]).to(device)

            optimizer.zero_grad()
            offset_logits = model(bits)

            loss = criterion(offset_logits, offset_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size]
                bits = torch.stack([b[0] for b in batch]).to(device)
                offset_targets = torch.stack([b[1] for b in batch]).to(device)

                offset_logits = model(bits)
                offset_preds = (offset_logits > 0).float()

                # Check if ALL 19 bits are correct
                correct += (offset_preds == offset_targets).all(dim=1).sum().item()
                total += len(batch)

        accuracy = correct / total * 100
        avg_loss = total_loss / num_batches

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Save best model
            Path('models/final').mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch
            }, 'models/final/neural_branch19_extractor.pt')

        if epoch % 5 == 0 or accuracy == 100:
            print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.2f}% (best={best_accuracy:.2f}%)")

        if accuracy == 100:
            print(f"\n    ✅ Perfect accuracy achieved at epoch {epoch}!")
            break

    print(f"\n[4] Final Results:")
    print(f"    Best accuracy: {best_accuracy:.2f}%")
    print(f"    Model saved to: models/final/neural_branch19_extractor.pt")

    # Verify on specific test cases
    print("\n[5] Verification tests:")
    model.eval()

    test_cases = [
        ("B.EQ +5", generate_bcond(5, 0), 5),
        ("B.NE -10", generate_bcond(0x7FFFF - 9, 1), 0x7FFFF - 9),
        ("CBZ X0, +100", generate_cbz(0, 100, 1), 100),
        ("CBNZ W5, +1000", generate_cbnz(5, 1000, 0), 1000),
        ("B.GT +0x3FFFF", generate_bcond(0x3FFFF, 12), 0x3FFFF),
    ]

    with torch.no_grad():
        for name, inst, expected_offset in test_cases:
            bits = inst_to_bits(inst).unsqueeze(0).to(device)
            offset_logits = model(bits)

            # Reconstruct offset
            offset_bits = (offset_logits[0] > 0).long()
            extracted = sum(int(offset_bits[i].item()) << i for i in range(19))

            status = "✅" if extracted == expected_offset else "❌"
            print(f"    {status} {name}: expected={expected_offset}, got={extracted}")

    print("=" * 70)
    return model


if __name__ == "__main__":
    train_extractor()
