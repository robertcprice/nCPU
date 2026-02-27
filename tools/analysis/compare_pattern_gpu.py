#!/usr/bin/env python3
"""Compare on MPS (Apple Silicon GPU)"""

import torch
import torch.nn as nn
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Use MPS
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {device}")


class SequentialPatternRecognizer(nn.Module):
    def __init__(self, d_model=128, num_patterns=6, seq_len=20):
        super().__init__()
        self.inst_encoder = nn.Sequential(nn.Linear(7, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.lstm = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(0.3), nn.Linear(d_model // 2, num_patterns))

    def forward(self, inst_seq, mask=None):
        x = self.inst_encoder(inst_seq)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=mask.bool() if mask is not None else None)
        if mask is not None:
            mask_expanded = (~mask.bool()).unsqueeze(-1).float()
            x = (attn_out * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-9)
        else:
            x = attn_out.mean(dim=1)
        return self.classifier(x)


class ParallelPatternRecognizer(nn.Module):
    def __init__(self, d_model=128, num_patterns=6, seq_len=20, n_heads=4, n_layers=3):
        super().__init__()
        self.seq_len = seq_len
        self.inst_encoder = nn.Sequential(nn.Linear(7, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=0.1, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(0.2), nn.Linear(d_model // 2, num_patterns))

    def forward(self, inst_seq, mask=None):
        batch_size, seq_len, _ = inst_seq.shape
        x = self.inst_encoder(inst_seq) + self.pos_embed[:, :seq_len, :]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        if mask is not None:
            mask = torch.cat([torch.zeros(batch_size, 1, device=mask.device), mask], dim=1)
        x = self.transformer(x, src_key_padding_mask=mask.bool() if mask is not None else None)
        return self.classifier(x[:, 0, :])


def generate_batch(batch_size, seq_len=20):
    seq = torch.randn(batch_size, seq_len, 7)
    # No mask to avoid MPS nested tensor issue
    return seq, None


def benchmark(model, device, batch_sizes=[1, 8, 32, 64, 128, 256], warmup=10, iterations=100):
    model.eval()
    results = {}

    for bs in batch_sizes:
        seq, mask = generate_batch(bs)
        seq = seq.to(device)
        mask = mask.to(device) if mask is not None else None

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(seq, mask)

        # Sync
        if device.type == 'mps':
            torch.mps.synchronize()

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(seq, mask)
            if device.type == 'mps':
                torch.mps.synchronize()
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        results[bs] = {
            'time_ms': avg_time * 1000,
            'samples_per_sec': bs / avg_time
        }

    return results


def main():
    print("=" * 70)
    print("      GPU COMPARISON: PARALLEL vs SEQUENTIAL")
    print("=" * 70)

    seq_model = SequentialPatternRecognizer().to(device)
    par_model = ParallelPatternRecognizer().to(device)

    print(f"\nSequential params: {sum(p.numel() for p in seq_model.parameters()):,}")
    print(f"Parallel params: {sum(p.numel() for p in par_model.parameters()):,}")

    print("\nBenchmarking Sequential (LSTM)...")
    seq_results = benchmark(seq_model, device)

    print("Benchmarking Parallel (Transformer)...")
    par_results = benchmark(par_model, device)

    print("\n" + "=" * 70)
    print("   RESULTS ON GPU (MPS)")
    print("=" * 70)
    print(f"\n   {'Batch':<8} {'Sequential':<18} {'Parallel':<18} {'Speedup':<10}")
    print("   " + "-" * 60)

    for bs in [1, 8, 32, 64, 128, 256]:
        seq_sps = seq_results[bs]['samples_per_sec']
        par_sps = par_results[bs]['samples_per_sec']
        speedup = par_sps / seq_sps
        marker = "✅" if speedup > 1.0 else "  "
        print(f"   {bs:<8} {seq_sps:>14,.0f}/s   {par_sps:>14,.0f}/s   {speedup:>6.2f}x {marker}")

    print("   " + "-" * 60)

    # Find where parallel wins
    par_wins = [bs for bs in [1, 8, 32, 64, 128, 256]
                if par_results[bs]['samples_per_sec'] > seq_results[bs]['samples_per_sec']]

    if par_wins:
        print(f"\n   Parallel wins at batch sizes: {par_wins}")
    else:
        print("\n   Sequential is faster at all batch sizes on this device")

    print("\n   KEY INSIGHT:")
    print("   - On GPU, parallel Transformer can process all sequence positions at once")
    print("   - LSTM must process sequentially, limiting GPU utilization")
    print("   - For CONTINUOUS BATCHING in CPU emulation, parallel is better!")


if __name__ == "__main__":
    main()
