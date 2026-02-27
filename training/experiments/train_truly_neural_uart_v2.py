#!/usr/bin/env python3
"""
TRULY NEURAL UART V2
====================
UART buffer using the same attention-based approach as the register file.

Key insight: A FIFO buffer is just a register file with different access patterns.
We use the SAME architecture as the truly neural register file!

- Buffer slots stored as nn.Parameter
- Read/write via attention
- Position tracking via learned attention patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


class TrulyNeuralUARTV2(nn.Module):
    """
    UART V2 - same architecture as truly neural register file.

    Buffer operations via attention:
    - Write: position attention → update buffer slot
    - Read: position attention → retrieve buffer slot
    - Position tracking: learned via keys
    """

    def __init__(self, buffer_size=16, byte_bits=8, key_dim=32):
        super().__init__()
        self.buffer_size = buffer_size
        self.byte_bits = byte_bits
        self.key_dim = key_dim

        # === TX BUFFER IN WEIGHTS ===
        self.tx_buffer = nn.Parameter(torch.zeros(buffer_size, byte_bits))
        self.tx_keys = nn.Parameter(torch.randn(buffer_size, key_dim) * 0.1)

        # === RX BUFFER IN WEIGHTS ===
        self.rx_buffer = nn.Parameter(torch.zeros(buffer_size, byte_bits))
        self.rx_keys = nn.Parameter(torch.randn(buffer_size, key_dim) * 0.1)

        # Position encoder: position one-hot → key space
        self.pos_encoder = nn.Sequential(
            nn.Linear(buffer_size, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, key_dim),
        )

        # Byte encoder: byte bits → key space (for content-based lookup)
        self.byte_encoder = nn.Sequential(
            nn.Linear(byte_bits, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, key_dim),
        )

        self.temperature = nn.Parameter(torch.tensor(0.5))
        self.write_lr = nn.Parameter(torch.tensor(0.5))

    def write_tx(self, position, byte_bits):
        """
        Write byte to TX buffer at position.

        Args:
            position: [batch] - position index (0 to buffer_size-1)
            byte_bits: [batch, byte_bits] - byte to write

        Returns:
            success: [batch]
        """
        batch = position.shape[0]

        # Position one-hot
        pos_onehot = F.one_hot(position.long(), num_classes=self.buffer_size).float()

        # Create query from position
        query = self.pos_encoder(pos_onehot)

        # Attention over TX buffer positions
        key_sim = torch.matmul(query, self.tx_keys.T)
        temp = torch.clamp(self.temperature.abs(), min=0.1)
        attention = F.softmax(key_sim / temp, dim=-1)

        # Write via Hebbian update
        lr = torch.clamp(self.write_lr.abs(), 0.01, 1.0)
        write_update = torch.einsum('bp,bd->bpd', attention, byte_bits)
        write_update_mean = write_update.mean(dim=0)

        byte_logits = torch.log(write_update_mean.clamp(0.01, 0.99) / (1 - write_update_mean.clamp(0.01, 0.99)))
        mask = attention.mean(dim=0).unsqueeze(-1)
        self.tx_buffer.data = (1 - lr * mask) * self.tx_buffer.data + lr * mask * byte_logits

        return torch.ones(batch, device=position.device)

    def read_tx(self, position):
        """
        Read byte from TX buffer at position.

        Args:
            position: [batch] - position index

        Returns:
            byte_bits: [batch, byte_bits]
        """
        batch = position.shape[0]

        # Position one-hot
        pos_onehot = F.one_hot(position.long(), num_classes=self.buffer_size).float()

        # Create query
        query = self.pos_encoder(pos_onehot)

        # Attention over TX buffer
        key_sim = torch.matmul(query, self.tx_keys.T)
        temp = torch.clamp(self.temperature.abs(), min=0.1)
        attention = F.softmax(key_sim / temp, dim=-1)

        # Read via attention
        buffer_values = torch.sigmoid(self.tx_buffer)
        byte_bits = torch.matmul(attention, buffer_values)

        return byte_bits

    def write_rx(self, position, byte_bits):
        """Write byte to RX buffer at position."""
        batch = position.shape[0]

        pos_onehot = F.one_hot(position.long(), num_classes=self.buffer_size).float()
        query = self.pos_encoder(pos_onehot)

        key_sim = torch.matmul(query, self.rx_keys.T)
        temp = torch.clamp(self.temperature.abs(), min=0.1)
        attention = F.softmax(key_sim / temp, dim=-1)

        lr = torch.clamp(self.write_lr.abs(), 0.01, 1.0)
        write_update = torch.einsum('bp,bd->bpd', attention, byte_bits)
        write_update_mean = write_update.mean(dim=0)

        byte_logits = torch.log(write_update_mean.clamp(0.01, 0.99) / (1 - write_update_mean.clamp(0.01, 0.99)))
        mask = attention.mean(dim=0).unsqueeze(-1)
        self.rx_buffer.data = (1 - lr * mask) * self.rx_buffer.data + lr * mask * byte_logits

        return torch.ones(batch, device=position.device)

    def read_rx(self, position):
        """Read byte from RX buffer at position."""
        batch = position.shape[0]

        pos_onehot = F.one_hot(position.long(), num_classes=self.buffer_size).float()
        query = self.pos_encoder(pos_onehot)

        key_sim = torch.matmul(query, self.rx_keys.T)
        temp = torch.clamp(self.temperature.abs(), min=0.1)
        attention = F.softmax(key_sim / temp, dim=-1)

        buffer_values = torch.sigmoid(self.rx_buffer)
        byte_bits = torch.matmul(attention, buffer_values)

        return byte_bits


def byte_to_bits(val):
    return torch.tensor([(val >> i) & 1 for i in range(8)], dtype=torch.float32)


def bits_to_byte(bits_t):
    return sum(int(b > 0.5) << i for i, b in enumerate(bits_t.tolist()))


def generate_batch(batch_size, buffer_size, device):
    """Generate training batch for UART operations."""
    positions = []
    bytes_data = []

    for _ in range(batch_size):
        pos = random.randint(0, buffer_size - 1)
        byte_val = random.randint(0, 255)

        positions.append(pos)
        bytes_data.append(byte_to_bits(byte_val))

    return (
        torch.tensor(positions, device=device),
        torch.stack(bytes_data).to(device),
    )


def train():
    print("=" * 60)
    print("TRULY NEURAL UART V2 TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print("Buffer operations via attention (like register file)!")

    buffer_size = 16

    model = TrulyNeuralUARTV2(
        buffer_size=buffer_size,
        byte_bits=8,
        key_dim=32
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    os.makedirs("models/final", exist_ok=True)

    best_acc = 0
    batch_size = 256

    for epoch in range(100):
        model.train()
        total_loss = 0
        t0 = time.time()

        for step in range(100):
            # Reset buffers periodically
            if step % 20 == 0:
                model.tx_buffer.data = torch.zeros_like(model.tx_buffer.data)
                model.rx_buffer.data = torch.zeros_like(model.rx_buffer.data)

            positions, byte_data = generate_batch(batch_size, buffer_size, device)

            optimizer.zero_grad()

            # Write to RX buffer
            model.write_rx(positions, byte_data)

            # Read back from RX buffer
            read_bytes = model.read_rx(positions)

            # Loss: should read what we wrote
            loss = F.mse_loss(read_bytes, byte_data)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Test
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for test_round in range(20):
                # Reset buffers
                model.rx_buffer.data = torch.zeros_like(model.rx_buffer.data)

                # Write specific values to specific positions
                test_data = []
                for pos in range(buffer_size):
                    byte_val = (pos * 17 + test_round) % 256  # Deterministic test pattern
                    test_data.append((pos, byte_val))

                    # Write
                    pos_t = torch.tensor([pos], device=device)
                    byte_t = byte_to_bits(byte_val).unsqueeze(0).to(device)
                    model.write_rx(pos_t, byte_t)

                # Read back and verify
                for pos, expected_byte in test_data:
                    pos_t = torch.tensor([pos], device=device)
                    read_bits = model.read_rx(pos_t)
                    read_byte = bits_to_byte(read_bits[0].cpu())

                    if read_byte == expected_byte:
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
                "op_name": "TRULY_NEURAL_UART_V2",
                "buffer_size": buffer_size,
            }, "models/final/truly_neural_uart_v2_best.pt")
            print(f"  Saved (acc={100*acc:.1f}%)")

        if acc >= 0.99:
            print("99%+ ACCURACY!")
            break

    print(f"\nBest roundtrip accuracy: {100*best_acc:.1f}%")


if __name__ == "__main__":
    train()
