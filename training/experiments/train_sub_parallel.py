#!/usr/bin/env python3
"""
PARALLEL NEURAL SUBTRACTION
============================
Like ADD with carry, but with BORROW propagation.

SUB: a - b = result
At each bit i:
  diff = a[i] XOR b[i] XOR borrow_in
  borrow_out = (!a[i] AND b[i]) OR ((!a[i] XOR b[i]) AND borrow_in)

This is exactly like carry chain but inverted logic for borrow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print("=" * 60)
print("PARALLEL NEURAL SUBTRACTION (Borrow Propagation)")
print("=" * 60)

class BorrowPredictorTransformer(nn.Module):
    """Predict borrow chain in parallel using transformer."""
    def __init__(self, max_bits=64, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.max_bits = max_bits

        # Input: (a, b) per bit -> d_model
        self.input_proj = nn.Linear(2, d_model)

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)

        # Transformer for borrow prediction (causal - each bit depends on previous)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Borrow predictor head
        self.borrow_head = nn.Linear(d_model, 1)

    def forward(self, a_bits, b_bits):
        batch, bits = a_bits.shape

        # Borrow generate: G = (!a) AND b
        # Borrow propagate: P = a XOR b (same as add)
        not_a = 1 - a_bits
        G = not_a * b_bits
        P = a_bits + b_bits - 2 * a_bits * b_bits  # XOR

        # Stack G and P as input
        gp = torch.stack([G, P], dim=-1)  # [batch, bits, 2]

        # Project and add position
        x = self.input_proj(gp)  # [batch, bits, d_model]
        x = x + self.pos_embedding[:, :bits, :]

        # Causal mask (each position can only see previous positions)
        mask = torch.triu(torch.ones(bits, bits, device=a_bits.device), diagonal=1).bool()

        # Transformer
        x = self.transformer(x, mask=mask)

        # Predict borrows
        borrow_logits = self.borrow_head(x).squeeze(-1)  # [batch, bits]
        borrows = torch.sigmoid(borrow_logits)

        # Compute result: diff[i] = a[i] XOR b[i] XOR borrow[i-1]
        borrow_in = torch.cat([torch.zeros(batch, 1, device=a_bits.device), borrows[:, :-1]], dim=1)

        # diff = P XOR borrow_in
        diffs = P + borrow_in - 2 * P * borrow_in

        return diffs, borrows

def int_to_bits(val, bits=64):
    return torch.tensor([(val >> i) & 1 for i in range(bits)], dtype=torch.float32)

def bits_to_int(bits_t):
    return sum(int(b > 0.5) << i for i, b in enumerate(bits_t.cpu().tolist()))

def compute_true_borrows(a, b, bits=64):
    """Compute true borrow chain for training."""
    result = []
    borrow = 0
    for i in range(bits):
        a_bit = (a >> i) & 1
        b_bit = (b >> i) & 1

        # Current bit result
        diff = (a_bit - b_bit - borrow) & 1

        # Borrow out: set if we needed to borrow
        borrow_out = 1 if (a_bit - b_bit - borrow) < 0 else 0

        result.append(borrow_out)
        borrow = borrow_out

    return torch.tensor(result, dtype=torch.float32)

def generate_batch(batch_size, device):
    a_list, b_list, result_list, borrow_list = [], [], [], []

    for _ in range(batch_size):
        # Mix of ranges
        if random.random() < 0.3:
            # Small numbers
            a = random.randint(0, (1 << 16) - 1)
        elif random.random() < 0.6:
            # Medium numbers
            a = random.randint(0, (1 << 32) - 1)
        else:
            # Full 64-bit
            a = random.randint(0, (1 << 64) - 1)

        b = random.randint(0, a)  # b <= a for unsigned subtraction
        result = (a - b) & ((1 << 64) - 1)
        borrows = compute_true_borrows(a, b)

        a_list.append(int_to_bits(a))
        b_list.append(int_to_bits(b))
        result_list.append(int_to_bits(result))
        borrow_list.append(borrows)

    return (torch.stack(a_list).to(device),
            torch.stack(b_list).to(device),
            torch.stack(result_list).to(device),
            torch.stack(borrow_list).to(device))

def train():
    print("Training Parallel SUB model...")
    model = BorrowPredictorTransformer(64, d_model=64, nhead=4, num_layers=3).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    batch_size = 512
    best_acc = 0

    os.makedirs("models/final", exist_ok=True)

    for epoch in range(300):
        model.train()
        total_loss = 0
        t0 = time.time()

        for _ in range(100):
            a, b, target_result, target_borrows = generate_batch(batch_size, device)

            optimizer.zero_grad()
            pred_result, pred_borrows = model(a, b)

            # Loss on both result and borrows
            loss_result = F.binary_cross_entropy(pred_result, target_result)
            loss_borrows = F.binary_cross_entropy(pred_borrows, target_borrows)
            loss = loss_result + loss_borrows

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Test
        model.eval()
        test_cases = [(100, 42, 58), (10, 5, 5), (255, 127, 128), (1000, 1, 999), (50, 50, 0),
                      (123456789, 12345, 123444444), ((1<<63), (1<<62), (1<<62))]
        correct = 0
        with torch.no_grad():
            for a, b, expected in test_cases:
                a_bits = int_to_bits(a).unsqueeze(0).to(device)
                b_bits = int_to_bits(b).unsqueeze(0).to(device)
                result, _ = model(a_bits, b_bits)
                result_int = bits_to_int(result[0])
                if result_int == expected:
                    correct += 1

        acc = correct / len(test_cases)
        elapsed = time.time() - t0

        if epoch % 20 == 0 or acc > best_acc:
            print(f"Epoch {epoch+1}: loss={total_loss/100:.4f} test_acc={100*acc:.0f}% [{elapsed:.1f}s]")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "accuracy": acc,
                "op_name": "SUB",
                "bits": 64,
                "architecture": "BorrowPredictorTransformer"
            }, "models/final/SUB_64bit_parallel.pt")
            print(f"  Saved (acc={100*acc:.0f}%)")

        if acc >= 1.0:
            print("100% TEST ACCURACY!")
            break

    print(f"Best accuracy: {100*best_acc:.0f}%")

    # Final verification
    print("\nFinal verification:")
    model.eval()
    with torch.no_grad():
        for a, b, expected in test_cases:
            a_bits = int_to_bits(a).unsqueeze(0).to(device)
            b_bits = int_to_bits(b).unsqueeze(0).to(device)
            result, _ = model(a_bits, b_bits)
            result_int = bits_to_int(result[0])
            status = "OK" if result_int == expected else f"GOT {result_int}"
            print(f"  {a} - {b} = {expected}: {status}")


if __name__ == "__main__":
    train()
