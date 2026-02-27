#!/usr/bin/env python3
"""
TRULY NEURAL UART V3
====================
Simplified UART - exact same architecture as the truly neural register file!

The register file gets 100% accuracy. UART is just a smaller register file
for bytes instead of 64-bit values. We use the EXACT SAME CODE PATTERN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


class TrulyNeuralUARTV3(nn.Module):
    """
    UART V3 - EXACT architecture as truly neural register file.

    Just like register file:
    - Slots stored as nn.Parameter
    - Position keys for attention
    - Read/write via attention + Hebbian update
    """

    def __init__(self, buffer_size=16, byte_bits=8, key_dim=32):
        super().__init__()
        self.buffer_size = buffer_size
        self.byte_bits = byte_bits
        self.key_dim = key_dim

        # RX Buffer - SAME as register file values
        self.rx_values = nn.Parameter(torch.zeros(buffer_size, byte_bits))

        # Position keys - SAME as register file keys
        self.rx_keys = nn.Parameter(torch.randn(buffer_size, key_dim) * 0.1)

        # TX Buffer
        self.tx_values = nn.Parameter(torch.zeros(buffer_size, byte_bits))
        self.tx_keys = nn.Parameter(torch.randn(buffer_size, key_dim) * 0.1)

        # Query encoder for position lookup
        self.query_encoder = nn.Sequential(
            nn.Linear(buffer_size, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, key_dim),
        )

        self.temperature = nn.Parameter(torch.tensor(0.5))

    def write_rx(self, slot_onehot, byte_bits):
        """
        Write byte to RX buffer slot.

        Args:
            slot_onehot: [batch, buffer_size] - which slot to write
            byte_bits: [batch, byte_bits] - byte value
        """
        batch = slot_onehot.shape[0]

        # Create query from slot
        query = self.query_encoder(slot_onehot)

        # Key similarity
        key_sim = torch.matmul(query, self.rx_keys.T)

        # Direct slot matching (strong signal)
        slot_sim = torch.matmul(slot_onehot, torch.eye(self.buffer_size, device=slot_onehot.device))

        combined_sim = key_sim + 3.0 * slot_sim

        temp = torch.clamp(self.temperature.abs(), min=0.1)
        attention = F.softmax(combined_sim / temp, dim=-1)

        # Hebbian write: values += lr * attention.T @ byte_bits
        write_update = torch.einsum('bp,bd->pd', attention, byte_bits) / batch

        # Convert to logits and update
        byte_logits = torch.log(byte_bits.mean(dim=0).clamp(0.01, 0.99) / (1 - byte_bits.mean(dim=0).clamp(0.01, 0.99)))

        # Update only the attended slots
        mask = attention.mean(dim=0).unsqueeze(-1)
        self.rx_values.data = (1 - 0.8 * mask) * self.rx_values.data + 0.8 * mask * byte_logits.unsqueeze(0)

        # Update keys to match slots
        self.rx_keys.data = 0.9 * self.rx_keys.data + 0.1 * query.mean(dim=0, keepdim=True).expand(self.buffer_size, -1) * mask

    def read_rx(self, slot_onehot):
        """
        Read byte from RX buffer slot.

        Args:
            slot_onehot: [batch, buffer_size] - which slot to read

        Returns:
            byte_bits: [batch, byte_bits]
        """
        batch = slot_onehot.shape[0]

        # Create query from slot
        query = self.query_encoder(slot_onehot)

        # Key similarity
        key_sim = torch.matmul(query, self.rx_keys.T)

        # Direct slot matching
        slot_sim = torch.matmul(slot_onehot, torch.eye(self.buffer_size, device=slot_onehot.device))

        combined_sim = key_sim + 3.0 * slot_sim

        temp = torch.clamp(self.temperature.abs(), min=0.1)
        attention = F.softmax(combined_sim / temp, dim=-1)

        # Read via attention
        values = torch.sigmoid(self.rx_values)
        byte_bits = torch.matmul(attention, values)

        return byte_bits


def byte_to_bits(val):
    return torch.tensor([(val >> i) & 1 for i in range(8)], dtype=torch.float32)


def bits_to_byte(bits_t):
    return sum(int(b > 0.5) << i for i, b in enumerate(bits_t.tolist()))


def generate_batch(batch_size, buffer_size, device):
    """Generate training batch - slot + byte pairs."""
    slots = []
    bytes_data = []

    for _ in range(batch_size):
        slot = random.randint(0, buffer_size - 1)
        byte_val = random.randint(0, 255)

        slot_onehot = F.one_hot(torch.tensor(slot), num_classes=buffer_size).float()
        slots.append(slot_onehot)
        bytes_data.append(byte_to_bits(byte_val))

    return (
        torch.stack(slots).to(device),
        torch.stack(bytes_data).to(device),
    )


def train():
    print("=" * 60)
    print("TRULY NEURAL UART V3 TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print("EXACT same architecture as truly neural register file!")

    buffer_size = 16

    model = TrulyNeuralUARTV3(
        buffer_size=buffer_size,
        byte_bits=8,
        key_dim=32
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    os.makedirs("models/final", exist_ok=True)

    best_acc = 0
    batch_size = 64  # Smaller batch for cleaner updates

    for epoch in range(100):
        model.train()
        total_loss = 0
        t0 = time.time()

        for step in range(100):
            # Reset buffer periodically
            if step % 10 == 0:
                model.rx_values.data = torch.zeros_like(model.rx_values.data)

            # Generate single slot-byte pairs
            slots, byte_data = generate_batch(batch_size, buffer_size, device)

            optimizer.zero_grad()

            # Write to buffer
            model.write_rx(slots, byte_data)

            # Read back
            read_bytes = model.read_rx(slots)

            # Loss
            loss = F.mse_loss(read_bytes, byte_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Test - write to each slot, then read back
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for test_round in range(20):
                # Reset buffer
                model.rx_values.data = torch.zeros_like(model.rx_values.data)

                # Write specific value to each slot
                test_values = {}
                for slot in range(buffer_size):
                    byte_val = (slot * 17 + test_round * 3) % 256
                    test_values[slot] = byte_val

                    slot_onehot = F.one_hot(torch.tensor([slot]), num_classes=buffer_size).float().to(device)
                    byte_bits = byte_to_bits(byte_val).unsqueeze(0).to(device)
                    model.write_rx(slot_onehot, byte_bits)

                # Read back each slot
                for slot in range(buffer_size):
                    slot_onehot = F.one_hot(torch.tensor([slot]), num_classes=buffer_size).float().to(device)
                    read_bits = model.read_rx(slot_onehot)
                    read_byte = bits_to_byte(read_bits[0].cpu())

                    expected = test_values[slot]
                    if read_byte == expected:
                        correct += 1
                    total += 1

        acc = correct / total
        elapsed = time.time() - t0

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/100:.4f} slot_acc={100*acc:.1f}% [{elapsed:.1f}s]")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "slot_accuracy": acc,
                "op_name": "TRULY_NEURAL_UART_V3",
                "buffer_size": buffer_size,
            }, "models/final/truly_neural_uart_v3_best.pt")
            print(f"  Saved (acc={100*acc:.1f}%)")

        if acc >= 0.99:
            print("99%+ ACCURACY!")
            break

    print(f"\nBest slot accuracy: {100*best_acc:.1f}%")

    # Final verification
    print("\nFinal verification:")
    model.eval()
    model.rx_values.data = torch.zeros_like(model.rx_values.data)

    with torch.no_grad():
        test_bytes = [0x00, 0xFF, 0x55, 0xAA, 0x12, 0x34, 0x56, 0x78]
        for slot, byte_val in enumerate(test_bytes):
            slot_onehot = F.one_hot(torch.tensor([slot]), num_classes=buffer_size).float().to(device)
            byte_bits = byte_to_bits(byte_val).unsqueeze(0).to(device)
            model.write_rx(slot_onehot, byte_bits)

        for slot, expected in enumerate(test_bytes):
            slot_onehot = F.one_hot(torch.tensor([slot]), num_classes=buffer_size).float().to(device)
            read_bits = model.read_rx(slot_onehot)
            read_byte = bits_to_byte(read_bits[0].cpu())
            status = "✓" if read_byte == expected else f"✗ got {read_byte}"
            print(f"  Slot {slot}: write 0x{expected:02X}, read 0x{read_byte:02X} {status}")


if __name__ == "__main__":
    train()
