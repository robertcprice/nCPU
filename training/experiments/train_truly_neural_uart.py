#!/usr/bin/env python3
"""
TRULY NEURAL UART
=================
Serial I/O with ALL state stored IN THE NETWORK WEIGHTS.

Like the Truly Neural Register File:
- TX buffer stored as nn.Parameter
- RX buffer stored as nn.Parameter
- Status register stored as nn.Parameter
- Read/write logic is LEARNED, not hardcoded!

The UART learns:
1. How to enqueue bytes to TX buffer
2. How to dequeue bytes from RX buffer
3. How to maintain status flags
4. When to generate interrupts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


class TrulyNeuralUART(nn.Module):
    """
    UART where ALL state is stored IN the neural network weights.

    Architecture:
    - tx_buffer: nn.Parameter [buffer_size, 8] - TX FIFO IN weights
    - rx_buffer: nn.Parameter [buffer_size, 8] - RX FIFO IN weights
    - tx_head/tail: nn.Parameter - FIFO pointers IN weights
    - rx_head/tail: nn.Parameter - FIFO pointers IN weights
    - status_state: nn.Parameter [8] - status bits IN weights

    Operations via attention-based read/write.
    """

    def __init__(self, buffer_size=16, key_dim=64):
        super().__init__()
        self.buffer_size = buffer_size
        self.key_dim = key_dim

        # === ALL STATE IN WEIGHTS ===
        # TX buffer (16 entries x 8 bits each)
        self.tx_buffer = nn.Parameter(torch.zeros(buffer_size, 8))
        self.tx_write_ptr = nn.Parameter(torch.zeros(buffer_size))  # One-hot position
        self.tx_read_ptr = nn.Parameter(torch.zeros(buffer_size))

        # RX buffer (16 entries x 8 bits each)
        self.rx_buffer = nn.Parameter(torch.zeros(buffer_size, 8))
        self.rx_write_ptr = nn.Parameter(torch.zeros(buffer_size))
        self.rx_read_ptr = nn.Parameter(torch.zeros(buffer_size))

        # Status: [rx_empty, rx_full, tx_empty, tx_full, rx_ready, tx_ready, irq_rx, irq_tx]
        self.status_state = nn.Parameter(torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))

        # Position keys for attention
        self.position_keys = nn.Parameter(torch.randn(buffer_size, key_dim) * 0.1)

        # === LEARNED OPERATIONS ===
        # Write encoder: byte → buffer update
        self.tx_write_encoder = nn.Sequential(
            nn.Linear(8 + buffer_size, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, buffer_size),  # Attention over positions
        )

        # Read encoder: position → byte
        self.rx_read_encoder = nn.Sequential(
            nn.Linear(buffer_size, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, buffer_size),  # Attention over positions
        )

        # Pointer advance logic
        self.ptr_advance = nn.Sequential(
            nn.Linear(buffer_size * 2, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, buffer_size),  # New pointer position
        )

        # Status update logic
        self.status_encoder = nn.Sequential(
            nn.Linear(buffer_size * 4, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, 8),  # New status
        )

        # Temperature and learning rate
        self.temperature = nn.Parameter(torch.tensor(0.5))
        self.update_lr = nn.Parameter(torch.tensor(0.3))

    def write_tx(self, byte_bits):
        """
        Write byte to TX buffer.

        Args:
            byte_bits: [batch, 8] - byte to write as bits

        Returns:
            success: [batch] - whether write succeeded
            status: [batch, 8] - new status
        """
        batch = byte_bits.shape[0]

        # Get current write pointer
        write_ptr = F.softmax(self.tx_write_ptr, dim=-1).unsqueeze(0).expand(batch, -1)

        # Compute where to write (attention over positions)
        write_input = torch.cat([byte_bits, write_ptr], dim=-1)
        write_attention = self.tx_write_encoder(write_input)
        temp = torch.clamp(self.temperature.abs(), min=0.1)
        write_pos = F.softmax(write_attention / temp, dim=-1)  # [batch, buffer_size]

        # Write byte to buffer via outer product
        lr = torch.clamp(self.update_lr.abs(), 0.01, 1.0)
        write_update = torch.einsum('bp,bd->bpd', write_pos, byte_bits)  # [batch, buffer_size, 8]
        write_update_mean = write_update.mean(dim=0)

        # Update buffer
        byte_logits = torch.log(write_update_mean.clamp(0.01, 0.99) / (1 - write_update_mean.clamp(0.01, 0.99)))
        self.tx_buffer.data = (1 - lr * write_pos.mean(dim=0).unsqueeze(-1)) * self.tx_buffer.data + lr * byte_logits

        # Advance write pointer
        ptr_input = torch.cat([write_ptr, F.softmax(self.tx_read_ptr, dim=-1).unsqueeze(0).expand(batch, -1)], dim=-1)
        new_ptr_logits = self.ptr_advance(ptr_input)
        new_ptr = F.softmax(new_ptr_logits / temp, dim=-1)

        new_ptr_mean = new_ptr.mean(dim=0)
        ptr_logits = torch.log(new_ptr_mean.clamp(0.01, 0.99) / (1 - new_ptr_mean.clamp(0.01, 0.99)))
        self.tx_write_ptr.data = (1 - lr) * self.tx_write_ptr.data + lr * ptr_logits

        # Update status
        status_input = torch.cat([
            F.softmax(self.tx_write_ptr, dim=-1).unsqueeze(0).expand(batch, -1),
            F.softmax(self.tx_read_ptr, dim=-1).unsqueeze(0).expand(batch, -1),
            F.softmax(self.rx_write_ptr, dim=-1).unsqueeze(0).expand(batch, -1),
            F.softmax(self.rx_read_ptr, dim=-1).unsqueeze(0).expand(batch, -1),
        ], dim=-1)
        new_status = torch.sigmoid(self.status_encoder(status_input))

        # Success if buffer not full
        tx_full = torch.sigmoid(self.status_state[3]).unsqueeze(0).expand(batch)
        success = 1 - tx_full

        return success, new_status

    def read_rx(self):
        """
        Read byte from RX buffer.

        Returns:
            byte_bits: [8] - read byte as bits
            valid: [] - whether read was valid (buffer not empty)
        """
        # Get current read pointer
        read_ptr = F.softmax(self.rx_read_ptr, dim=-1)

        # Read via attention
        read_attention = self.rx_read_encoder(read_ptr.unsqueeze(0))
        temp = torch.clamp(self.temperature.abs(), min=0.1)
        read_pos = F.softmax(read_attention / temp, dim=-1).squeeze(0)

        # Read byte from buffer
        buffer_values = torch.sigmoid(self.rx_buffer)
        byte_bits = torch.matmul(read_pos, buffer_values)

        # Check if valid (not empty)
        rx_empty = torch.sigmoid(self.status_state[0])
        valid = 1 - rx_empty

        # Advance read pointer
        lr = torch.clamp(self.update_lr.abs(), 0.01, 1.0)
        ptr_input = torch.cat([read_ptr, F.softmax(self.rx_write_ptr, dim=-1)], dim=-1).unsqueeze(0)
        new_ptr_logits = self.ptr_advance(ptr_input).squeeze(0)
        new_ptr = F.softmax(new_ptr_logits / temp, dim=-1)

        # Only advance if valid read
        ptr_logits = torch.log(new_ptr.clamp(0.01, 0.99) / (1 - new_ptr.clamp(0.01, 0.99)))
        self.rx_read_ptr.data = (1 - lr * valid) * self.rx_read_ptr.data + lr * valid * ptr_logits

        return byte_bits, valid

    def receive_byte(self, byte_bits):
        """
        Receive byte into RX buffer (from external input).

        Args:
            byte_bits: [batch, 8] - received bytes

        Returns:
            success: [batch] - whether receive succeeded
        """
        batch = byte_bits.shape[0]

        # Get current write pointer
        write_ptr = F.softmax(self.rx_write_ptr, dim=-1).unsqueeze(0).expand(batch, -1)

        # Write to RX buffer
        lr = torch.clamp(self.update_lr.abs(), 0.01, 1.0)
        write_update = torch.einsum('bp,bd->bpd', write_ptr, byte_bits)
        write_update_mean = write_update.mean(dim=0)

        byte_logits = torch.log(write_update_mean.clamp(0.01, 0.99) / (1 - write_update_mean.clamp(0.01, 0.99)))
        self.rx_buffer.data = (1 - lr * write_ptr.mean(dim=0).unsqueeze(-1)) * self.rx_buffer.data + lr * byte_logits

        # Advance write pointer
        temp = torch.clamp(self.temperature.abs(), min=0.1)
        ptr_input = torch.cat([write_ptr, F.softmax(self.rx_read_ptr, dim=-1).unsqueeze(0).expand(batch, -1)], dim=-1)
        new_ptr_logits = self.ptr_advance(ptr_input)
        new_ptr = F.softmax(new_ptr_logits / temp, dim=-1)

        new_ptr_mean = new_ptr.mean(dim=0)
        ptr_logits = torch.log(new_ptr_mean.clamp(0.01, 0.99) / (1 - new_ptr_mean.clamp(0.01, 0.99)))
        self.rx_write_ptr.data = (1 - lr) * self.rx_write_ptr.data + lr * ptr_logits

        # Update status (mark RX not empty)
        self.status_state.data[0] = self.status_state.data[0] - lr  # rx_empty → 0
        self.status_state.data[4] = self.status_state.data[4] + lr  # rx_ready → 1

        rx_full = torch.sigmoid(self.status_state[1]).unsqueeze(0).expand(batch)
        success = 1 - rx_full

        return success

    def read_status(self):
        """Read status register."""
        return torch.sigmoid(self.status_state)


def byte_to_bits(val):
    """Convert byte to bit tensor."""
    return torch.tensor([(val >> i) & 1 for i in range(8)], dtype=torch.float32)


def bits_to_byte(bits_t):
    """Convert bit tensor to byte."""
    return sum(int(b > 0.5) << i for i, b in enumerate(bits_t.tolist()))


def generate_batch(batch_size, device):
    """Generate training batch for UART operations."""
    tx_bytes = []
    rx_bytes = []
    operations = []  # 0=TX write, 1=RX read, 2=RX receive

    for _ in range(batch_size):
        tx_byte = random.randint(0, 255)
        rx_byte = random.randint(0, 255)
        op = random.randint(0, 2)

        tx_bytes.append(byte_to_bits(tx_byte))
        rx_bytes.append(byte_to_bits(rx_byte))
        operations.append(op)

    return (
        torch.stack(tx_bytes).to(device),
        torch.stack(rx_bytes).to(device),
        torch.tensor(operations, device=device),
    )


def train():
    print("=" * 60)
    print("TRULY NEURAL UART TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print("ALL UART state stored IN NETWORK WEIGHTS!")

    model = TrulyNeuralUART(
        buffer_size=16,
        key_dim=64
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    os.makedirs("models/final", exist_ok=True)

    best_acc = 0
    batch_size = 128

    for epoch in range(100):
        model.train()
        total_loss = 0
        t0 = time.time()

        for _ in range(100):
            tx_bytes, rx_bytes, ops = generate_batch(batch_size, device)

            optimizer.zero_grad()

            # Test TX write
            success_tx, status = model.write_tx(tx_bytes)

            # Test RX receive (simulate input)
            success_rx = model.receive_byte(rx_bytes)

            # Test RX read
            read_byte, valid = model.read_rx()

            # Losses
            # TX should succeed (buffer not full at start)
            loss_tx = F.binary_cross_entropy(success_tx, torch.ones_like(success_tx))

            # RX receive should succeed
            loss_rx_recv = F.binary_cross_entropy(success_rx, torch.ones_like(success_rx))

            # Read byte should match what was received
            loss_rx_read = F.mse_loss(read_byte.unsqueeze(0).expand(batch_size, -1), rx_bytes)

            loss = loss_tx + loss_rx_recv + loss_rx_read
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

            # Reset buffer state periodically to avoid saturation
            if random.random() < 0.1:
                model.tx_write_ptr.data = torch.zeros_like(model.tx_write_ptr.data)
                model.tx_read_ptr.data = torch.zeros_like(model.tx_read_ptr.data)
                model.rx_write_ptr.data = torch.zeros_like(model.rx_write_ptr.data)
                model.rx_read_ptr.data = torch.zeros_like(model.rx_read_ptr.data)

        scheduler.step()

        # Test
        model.eval()
        correct_roundtrip = 0
        total = 0

        with torch.no_grad():
            for _ in range(20):
                # Reset buffers
                model.rx_write_ptr.data = torch.zeros_like(model.rx_write_ptr.data)
                model.rx_read_ptr.data = torch.zeros_like(model.rx_read_ptr.data)

                # Send bytes through RX and read them back
                test_bytes = torch.stack([byte_to_bits(random.randint(0, 255)) for _ in range(10)]).to(device)

                for i in range(10):
                    model.receive_byte(test_bytes[i:i+1])

                for i in range(10):
                    read_byte, valid = model.read_rx()
                    expected = test_bytes[i]
                    if bits_to_byte(read_byte.cpu()) == bits_to_byte(expected.cpu()):
                        correct_roundtrip += 1
                    total += 1

        acc = correct_roundtrip / total if total > 0 else 0
        elapsed = time.time() - t0

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/100:.4f} roundtrip_acc={100*acc:.1f}% [{elapsed:.1f}s]")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "roundtrip_accuracy": acc,
                "op_name": "TRULY_NEURAL_UART",
            }, "models/final/truly_neural_uart_best.pt")
            print(f"  Saved (acc={100*acc:.1f}%)")

        if acc >= 0.90:
            print("90%+ ACCURACY!")
            break

    print(f"\nBest roundtrip accuracy: {100*best_acc:.1f}%")


if __name__ == "__main__":
    train()
