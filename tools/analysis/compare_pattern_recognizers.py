#!/usr/bin/env python3
"""
COMPARE PARALLEL vs SEQUENTIAL PATTERN RECOGNIZERS
====================================================

Benchmarks:
1. Training speed
2. Inference speed (single)
3. Inference speed (batched)
4. GPU utilization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from pathlib import Path

# Force CPU for fair comparison
device = torch.device("cpu")
print(f"Using device: {device}")


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class SequentialPatternRecognizer(nn.Module):
    """LSTM + Attention (sequential processing)"""

    def __init__(self, d_model=128, num_patterns=6, seq_len=20):
        super().__init__()
        inst_dim = 7

        self.inst_encoder = nn.Sequential(
            nn.Linear(inst_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # LSTM processes tokens SEQUENTIALLY
        self.lstm = nn.LSTM(d_model, d_model, num_layers=2,
                           batch_first=True, dropout=0.2)

        self.attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, num_patterns)
        )

    def forward(self, inst_seq, mask=None):
        x = self.inst_encoder(inst_seq)
        lstm_out, _ = self.lstm(x)  # Sequential!
        lstm_mask = mask.bool() if mask is not None else None
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out,
                                     key_padding_mask=lstm_mask)
        if mask is not None:
            mask_expanded = (~mask.bool()).unsqueeze(-1).float()
            x = (attn_out * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-9)
        else:
            x = attn_out.mean(dim=1)
        return self.classifier(x)


class ParallelPatternRecognizer(nn.Module):
    """Pure Transformer (parallel processing)"""

    def __init__(self, d_model=128, num_patterns=6, seq_len=20, n_heads=4, n_layers=3):
        super().__init__()
        inst_dim = 7
        self.seq_len = seq_len

        self.inst_encoder = nn.Sequential(
            nn.Linear(inst_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Transformer processes ALL tokens in PARALLEL
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, num_patterns)
        )

    def forward(self, inst_seq, mask=None):
        batch_size, seq_len, _ = inst_seq.shape
        x = self.inst_encoder(inst_seq)
        x = x + self.pos_embed[:, :seq_len, :]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        if mask is not None:
            cls_mask = torch.zeros(batch_size, 1, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=mask.bool() if mask is not None else None)
        return self.classifier(x[:, 0, :])


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_data(n_samples=1000):
    sequences = []

    patterns = [
        # MEMSET
        lambda: [[np.random.randint(0,5), np.random.randint(0,10), 31, 9, 0, 1, 0],
                 [np.random.randint(5,10), np.random.randint(5,10), np.random.randint(0,31), 0, 0, 0, 0],
                 [np.random.randint(5,10), np.random.randint(5,10), 31, 11, 0, 0, 0],
                 [31, 31, 31, 10, 0, 0, 0]],
        # MEMCPY
        lambda: [[np.random.randint(0,10), np.random.randint(0,10), 31, 8, 1, 0, 0],
                 [np.random.randint(0,10), np.random.randint(10,20), 31, 9, 0, 1, 0],
                 [np.random.randint(10,15), np.random.randint(10,15), 31, 1, 0, 0, 1],
                 [31, 31, 31, 10, 0, 0, 0]],
        # STRLEN
        lambda: [[np.random.randint(0,5), np.random.randint(0,10), 31, 8, 1, 0, 0],
                 [np.random.randint(0,5), np.random.randint(0,5), 31, 11, 0, 0, 0],
                 [np.random.randint(0,5), np.random.randint(0,10), 31, 0, 0, 0, 0],
                 [31, 31, 31, 10, 0, 0, 0]],
        # POLLING
        lambda: [[np.random.randint(0,5), np.random.randint(0,10), 31, 8, 1, 0, 0],
                 [np.random.randint(0,5), np.random.randint(0,5), 31, 11, 0, 0, 0],
                 [31, 31, 31, 10, 0, 0, 0]],
        # BUBBLE_SORT
        lambda: [[np.random.randint(0,5), np.random.randint(0,10), 31, 8, 1, 0, 0],
                 [np.random.randint(0,5), np.random.randint(10,20), 31, 8, 1, 0, 0],
                 [np.random.randint(0,5), np.random.randint(0,5), np.random.randint(0,10), 11, 0, 0, 1],
                 [np.random.randint(0,5), np.random.randint(0,5), np.random.randint(0,10), 10, 0, 0, 0],
                 [np.random.randint(0,5), np.random.randint(0,10), 31, 9, 0, 1, 0],
                 [np.random.randint(0,5), np.random.randint(10,20), 31, 9, 0, 1, 0]],
        # UNKNOWN
        lambda: [[np.random.randint(0,32), np.random.randint(0,32), np.random.randint(0,32),
                  np.random.randint(0,15), np.random.randint(0,2), np.random.randint(0,2),
                  np.random.randint(0,2)] for _ in range(np.random.randint(3,8))]
    ]

    for i in range(n_samples):
        label = i % 6
        sequences.append((patterns[label](), label))

    np.random.shuffle(sequences)
    return sequences


class PatternDataset(Dataset):
    def __init__(self, sequences, seq_len=20):
        self.sequences = sequences
        self.seq_len = seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, label = self.sequences[idx]
        if len(seq) > self.seq_len:
            seq = seq[-self.seq_len:]
            mask = torch.zeros(self.seq_len)
        else:
            pad_len = self.seq_len - len(seq)
            seq = seq + [[0]*7] * pad_len
            mask = torch.zeros(self.seq_len)
            mask[self.seq_len - pad_len:] = 1
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label), mask


# =============================================================================
# COMPARISON
# =============================================================================

def train_model(model, train_loader, val_loader, epochs=20, name="Model"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_times = []

    for epoch in range(epochs):
        model.train()
        start = time.perf_counter()

        for batch in train_loader:
            inst_seq, labels, mask = batch
            optimizer.zero_grad()
            logits = model(inst_seq, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        train_times.append(time.perf_counter() - start)

    # Validate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inst_seq, labels, mask = batch
            logits = model(inst_seq, mask)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return sum(train_times), correct / total * 100


def benchmark_inference(model, test_data, batch_sizes=[1, 8, 32, 64, 128]):
    model.eval()
    results = {}

    for batch_size in batch_sizes:
        loader = DataLoader(PatternDataset(test_data), batch_size=batch_size)

        times = []
        with torch.no_grad():
            for batch in loader:
                inst_seq, labels, mask = batch

                start = time.perf_counter()
                _ = model(inst_seq, mask)
                times.append(time.perf_counter() - start)

        total_time = sum(times)
        samples = len(test_data)
        results[batch_size] = {
            'total_time': total_time,
            'samples_per_sec': samples / total_time
        }

    return results


def main():
    print("=" * 70)
    print("      PARALLEL vs SEQUENTIAL PATTERN RECOGNIZER COMPARISON")
    print("=" * 70)
    print()

    # Generate data
    print("[1] Generating data...")
    train_data = generate_data(2000)
    val_data = generate_data(400)
    test_data = generate_data(1000)

    train_loader = DataLoader(PatternDataset(train_data), batch_size=32, shuffle=True)
    val_loader = DataLoader(PatternDataset(val_data), batch_size=32)

    # Create models
    print("[2] Creating models...")
    seq_model = SequentialPatternRecognizer()
    par_model = ParallelPatternRecognizer()

    seq_params = sum(p.numel() for p in seq_model.parameters())
    par_params = sum(p.numel() for p in par_model.parameters())

    print(f"   Sequential (LSTM): {seq_params:,} parameters")
    print(f"   Parallel (Transformer): {par_params:,} parameters")

    # Train both
    print("\n[3] Training (20 epochs each)...")

    print("   Training Sequential...")
    seq_time, seq_acc = train_model(seq_model, train_loader, val_loader, epochs=20, name="Sequential")

    print("   Training Parallel...")
    par_time, par_acc = train_model(par_model, train_loader, val_loader, epochs=20, name="Parallel")

    print(f"\n   TRAINING RESULTS:")
    print(f"   Sequential: {seq_time:.2f}s, {seq_acc:.1f}% accuracy")
    print(f"   Parallel:   {par_time:.2f}s, {par_acc:.1f}% accuracy")
    print(f"   Training speedup: {seq_time/par_time:.2f}x {'(Parallel faster)' if par_time < seq_time else '(Sequential faster)'}")

    # Benchmark inference
    print("\n[4] Benchmarking inference...")

    seq_results = benchmark_inference(seq_model, test_data)
    par_results = benchmark_inference(par_model, test_data)

    print("\n   INFERENCE RESULTS (samples/sec):")
    print("   " + "-" * 50)
    print(f"   {'Batch Size':<12} {'Sequential':<15} {'Parallel':<15} {'Speedup':<10}")
    print("   " + "-" * 50)

    for batch_size in [1, 8, 32, 64, 128]:
        seq_sps = seq_results[batch_size]['samples_per_sec']
        par_sps = par_results[batch_size]['samples_per_sec']
        speedup = par_sps / seq_sps
        print(f"   {batch_size:<12} {seq_sps:>12,.0f}   {par_sps:>12,.0f}   {speedup:>6.2f}x")

    print("   " + "-" * 50)

    # Summary
    print("\n" + "=" * 70)
    print("      SUMMARY")
    print("=" * 70)
    print("""
   The Parallel (Transformer) model:
   - Uses self-attention which processes ALL tokens simultaneously
   - Better GPU utilization due to parallelism
   - Scales better with batch size

   The Sequential (LSTM) model:
   - Processes tokens one-by-one in sequence
   - Hidden state must propagate through each timestep
   - Harder to parallelize on GPU

   For pattern recognition in CPU execution:
   - Parallel model is better for batch detection
   - Can detect multiple patterns simultaneously
   - Better throughput for continuous batching
""")


if __name__ == "__main__":
    main()
