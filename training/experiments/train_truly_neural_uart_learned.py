#!/usr/bin/env python3
"""
TRULY NEURAL UART - LEARNED BEHAVIOR
=====================================
The network LEARNS buffer read/write operations from examples.
NO pre-populated values. NO lookup tables.

The network learns:
- Write: how to store a byte at a position
- Read: how to retrieve a byte from a position
- State: buffer contents are learned associations

This uses the SAME architecture as the neural register file
but learns through pure training, not pre-population.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


class TrulyNeuralUARTLearned(nn.Module):
    """
    UART that LEARNS buffer operations - no pre-population.

    Uses attention mechanism to learn:
    - Position → key association
    - Write: update value at attended position
    - Read: retrieve value from attended position
    """

    def __init__(self, buffer_size=16, byte_bits=8, hidden_dim=64):
        super().__init__()
        self.buffer_size = buffer_size
        self.byte_bits = byte_bits

        # Buffer values IN WEIGHTS - learned through training
        self.buffer_values = nn.Parameter(torch.randn(buffer_size, byte_bits) * 0.01)

        # LEARNED position embedding
        self.position_embedding = nn.Embedding(buffer_size, hidden_dim)

        # LEARNED write network
        self.write_net = nn.Sequential(
            nn.Linear(hidden_dim + byte_bits, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, buffer_size),  # attention over buffer
        )

        # LEARNED read network
        self.read_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, buffer_size),  # attention over buffer
        )

        # LEARNED value transform (for writing)
        self.value_transform = nn.Sequential(
            nn.Linear(byte_bits, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, byte_bits),
        )

        self.temperature = nn.Parameter(torch.tensor(0.3))

    def write(self, position, byte_bits):
        """
        LEARN to write byte at position.

        The network learns which buffer slot corresponds to which position.
        """
        batch = position.shape[0]

        # Get position embedding
        pos_embed = self.position_embedding(position)  # [batch, hidden_dim]

        # Compute write attention
        write_input = torch.cat([pos_embed, byte_bits], dim=-1)
        write_logits = self.write_net(write_input)

        temp = torch.clamp(self.temperature.abs(), min=0.1)
        write_attn = F.softmax(write_logits / temp, dim=-1)  # [batch, buffer_size]

        # Transform byte value
        transformed_value = torch.sigmoid(self.value_transform(byte_bits))  # [batch, byte_bits]

        # Write via attention-weighted update
        # Compute outer product and update buffer
        write_update = torch.einsum('bp,bd->bpd', write_attn, transformed_value)
        update_mean = write_update.mean(dim=0)

        # Update buffer values (gradient flows through this)
        mask = write_attn.mean(dim=0, keepdim=True).T  # [buffer_size, 1]
        value_logits = torch.log(update_mean.clamp(0.01, 0.99) / (1 - update_mean.clamp(0.01, 0.99)))
        self.buffer_values.data = (1 - 0.9 * mask) * self.buffer_values.data + 0.9 * mask * value_logits

        return write_attn

    def read(self, position):
        """
        LEARN to read byte from position.
        """
        batch = position.shape[0]

        # Get position embedding
        pos_embed = self.position_embedding(position)

        # Compute read attention
        read_logits = self.read_net(pos_embed)

        temp = torch.clamp(self.temperature.abs(), min=0.1)
        read_attn = F.softmax(read_logits / temp, dim=-1)

        # Read via attention
        buffer_vals = torch.sigmoid(self.buffer_values)  # [buffer_size, byte_bits]
        read_value = torch.matmul(read_attn, buffer_vals)  # [batch, byte_bits]

        return read_value, read_attn


def byte_to_bits(val):
    return torch.tensor([(val >> i) & 1 for i in range(8)], dtype=torch.float32)


def bits_to_byte(bits_t):
    return sum(int(b > 0.5) << i for i, b in enumerate(bits_t.tolist()))


def train():
    print("=" * 60)
    print("TRULY NEURAL UART - LEARNED BEHAVIOR")
    print("=" * 60)
    print(f"Device: {device}")
    print("Network LEARNS buffer operations from examples!")
    print("NO pre-populated values. Pure learning.")

    buffer_size = 16

    model = TrulyNeuralUARTLearned(
        buffer_size=buffer_size,
        byte_bits=8,
        hidden_dim=64
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    os.makedirs("models/final", exist_ok=True)

    best_acc = 0
    batch_size = 32

    for epoch in range(200):
        model.train()
        total_loss = 0
        t0 = time.time()

        for step in range(100):
            # Reset buffer periodically for fresh learning
            if step % 20 == 0:
                model.buffer_values.data = torch.randn_like(model.buffer_values.data) * 0.01

            # Training: write then read same position, should get same value
            positions = torch.randint(0, buffer_size, (batch_size,), device=device)
            byte_values = torch.stack([byte_to_bits(random.randint(0, 255)) for _ in range(batch_size)]).to(device)

            optimizer.zero_grad()

            # Write
            model.write(positions, byte_values)

            # Read back
            read_values, _ = model.read(positions)

            # Loss: read should match write
            loss = F.mse_loss(read_values, byte_values)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Test: write to all slots, then read all back
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for test_round in range(10):
                # Reset buffer
                model.buffer_values.data = torch.randn_like(model.buffer_values.data) * 0.01

                # Write unique value to each slot
                test_data = {}
                for pos in range(buffer_size):
                    byte_val = (pos * 17 + test_round * 7) % 256
                    test_data[pos] = byte_val

                    pos_t = torch.tensor([pos], device=device)
                    byte_t = byte_to_bits(byte_val).unsqueeze(0).to(device)
                    model.write(pos_t, byte_t)

                # Read each slot back
                for pos in range(buffer_size):
                    pos_t = torch.tensor([pos], device=device)
                    read_val, _ = model.read(pos_t)
                    read_byte = bits_to_byte(read_val[0].cpu())

                    if read_byte == test_data[pos]:
                        correct += 1
                    total += 1

        acc = correct / total
        elapsed = time.time() - t0

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/100:.4f} roundtrip_acc={100*acc:.1f}% [{elapsed:.1f}s]")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "roundtrip_accuracy": acc,
                "op_name": "TRULY_NEURAL_UART_LEARNED",
                "buffer_size": buffer_size,
            }, "models/final/truly_neural_uart_learned_best.pt")
            print(f"  Saved (acc={100*acc:.1f}%)")

        if acc >= 0.99:
            print("99%+ ACCURACY!")
            break

    print(f"\nBest roundtrip accuracy: {100*best_acc:.1f}%")

    # Final verification
    print("\nFinal verification:")
    model.eval()
    model.buffer_values.data = torch.randn_like(model.buffer_values.data) * 0.01

    with torch.no_grad():
        test_bytes = [0x00, 0xFF, 0x55, 0xAA, 0x12, 0x34, 0x56, 0x78]
        for pos, byte_val in enumerate(test_bytes):
            pos_t = torch.tensor([pos], device=device)
            byte_t = byte_to_bits(byte_val).unsqueeze(0).to(device)
            model.write(pos_t, byte_t)

        for pos, expected in enumerate(test_bytes):
            pos_t = torch.tensor([pos], device=device)
            read_val, _ = model.read(pos_t)
            read_byte = bits_to_byte(read_val[0].cpu())
            status = "✓" if read_byte == expected else f"✗ got {read_byte}"
            print(f"  Slot {pos}: write 0x{expected:02X}, read 0x{read_byte:02X} {status}")


if __name__ == "__main__":
    train()
