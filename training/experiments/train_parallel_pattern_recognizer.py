#!/usr/bin/env python3
"""
TRAIN PARALLEL PATTERN RECOGNIZER
==================================

Trains the ParallelPatternRecognizer (pure Transformer, no LSTM)
for GPU-parallel pattern detection.

Unlike the sequential LSTM model, this can:
- Process all sequence positions in PARALLEL
- Batch detect multiple sequences at once
- Better GPU utilization

Run: python train_parallel_pattern_recognizer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import time

# Import the model architecture
from neural_cpu_continuous_batch import ParallelPatternRecognizer, device


# =============================================================================
# TRAINING DATA GENERATION
# =============================================================================

def generate_training_data():
    """Generate diverse training data with clear pattern distinctions."""

    sequences = []

    # MEMSET: STORE → ADD → COMPARE → BRANCH
    for _ in range(200):
        seq = [
            [np.random.randint(0, 5), np.random.randint(0, 10), 31, 9, 0, 1, 0],  # STORE
            [np.random.randint(5, 10), np.random.randint(5, 10), np.random.randint(0, 31), 0, 0, 0, 0],  # ADD
            [np.random.randint(5, 10), np.random.randint(5, 10), 31, 11, 0, 0, 0],  # COMPARE
            [31, 31, 31, 10, 0, 0, 0],  # BRANCH
        ]
        sequences.append((seq, 0))

    # MEMCPY: LOAD → STORE → SUB → BRANCH
    for _ in range(200):
        seq = [
            [np.random.randint(0, 10), np.random.randint(0, 10), 31, 8, 1, 0, 0],  # LOAD
            [np.random.randint(0, 10), np.random.randint(10, 20), 31, 9, 0, 1, 0],  # STORE
            [np.random.randint(10, 15), np.random.randint(10, 15), 31, 1, 0, 0, 1],  # SUB
            [31, 31, 31, 10, 0, 0, 0],  # BRANCH
        ]
        sequences.append((seq, 1))

    # STRLEN: LOAD → COMPARE → ADD → BRANCH
    for _ in range(200):
        seq = [
            [np.random.randint(0, 5), np.random.randint(0, 10), 31, 8, 1, 0, 0],  # LOAD
            [np.random.randint(0, 5), np.random.randint(0, 5), 31, 11, 0, 0, 0],  # COMPARE
            [np.random.randint(0, 5), np.random.randint(0, 10), 31, 0, 0, 0, 0],  # ADD
            [31, 31, 31, 10, 0, 0, 0],  # BRANCH
        ]
        sequences.append((seq, 2))

    # POLLING: LOAD → COMPARE → BRANCH (no ADD/SUB)
    for _ in range(200):
        seq = [
            [np.random.randint(0, 5), np.random.randint(0, 10), 31, 8, 1, 0, 0],  # LOAD
            [np.random.randint(0, 5), np.random.randint(0, 5), 31, 11, 0, 0, 0],  # COMPARE
            [31, 31, 31, 10, 0, 0, 0],  # BRANCH
        ]
        sequences.append((seq, 3))

    # BUBBLE SORT: LOAD → LOAD → COMPARE → BRANCH → STORE → STORE
    for _ in range(200):
        seq = [
            [np.random.randint(0, 5), np.random.randint(0, 10), 31, 8, 1, 0, 0],  # LOAD
            [np.random.randint(0, 5), np.random.randint(10, 20), 31, 8, 1, 0, 0],  # LOAD
            [np.random.randint(0, 5), np.random.randint(0, 5), np.random.randint(0, 10), 11, 0, 0, 1],  # COMPARE
            [np.random.randint(0, 5), np.random.randint(0, 5), np.random.randint(0, 10), 10, 0, 0, 0],  # BRANCH
            [np.random.randint(0, 5), np.random.randint(0, 10), 31, 9, 0, 1, 0],  # STORE
            [np.random.randint(0, 5), np.random.randint(10, 20), 31, 9, 0, 1, 0],  # STORE
        ]
        sequences.append((seq, 4))

    # UNKNOWN: Random sequences
    for _ in range(300):
        seq_len = np.random.randint(3, 8)
        seq = []
        for _ in range(seq_len):
            seq.append([
                np.random.randint(0, 32),
                np.random.randint(0, 32),
                np.random.randint(0, 32),
                np.random.randint(0, 15),
                np.random.randint(0, 2),
                np.random.randint(0, 2),
                np.random.randint(0, 2)
            ])
        sequences.append((seq, 5))

    np.random.shuffle(sequences)
    return sequences


# =============================================================================
# DATASET
# =============================================================================

class PatternDataset(Dataset):
    def __init__(self, sequences, seq_len=20):
        self.sequences = sequences
        self.seq_len = seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, label = self.sequences[idx]

        # Pad/truncate
        if len(seq) > self.seq_len:
            seq = seq[-self.seq_len:]
            mask = torch.zeros(self.seq_len)
        else:
            pad_len = self.seq_len - len(seq)
            seq = seq + [[0]*7] * pad_len
            mask = torch.zeros(self.seq_len)
            mask[self.seq_len - pad_len:] = 1

        inst_seq = torch.tensor(seq, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return inst_seq, label, mask


# =============================================================================
# TRAINING
# =============================================================================

def train():
    print("=" * 70)
    print("      TRAINING PARALLEL PATTERN RECOGNIZER")
    print("=" * 70)
    print(f"   Device: {device}")
    print()

    # Generate data
    print("[1/4] Generating training data...")
    sequences = generate_training_data()
    print(f"   Generated {len(sequences)} sequences")

    # Split train/val
    split = int(len(sequences) * 0.8)
    train_seqs = sequences[:split]
    val_seqs = sequences[split:]

    train_dataset = PatternDataset(train_seqs)
    val_dataset = PatternDataset(val_seqs)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Create model
    print("\n[2/4] Creating model...")
    model = ParallelPatternRecognizer(d_model=128, num_patterns=6, seq_len=20, n_heads=4, n_layers=3).to(device)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("\n[3/4] Training...")
    best_val_acc = 0
    best_state = None

    for epoch in range(50):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            inst_seq, labels, mask = [x.to(device) for x in batch]

            optimizer.zero_grad()
            logits = model(inst_seq, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                inst_seq, labels, mask = [x.to(device) for x in batch]
                logits = model(inst_seq, mask)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1:3d}: Train Loss={train_loss/len(train_loader):.4f} "
                  f"Train Acc={train_acc:.1f}% Val Acc={val_acc:.1f}%")

        scheduler.step()

    # Save best model
    print(f"\n[4/4] Saving model (best val acc: {best_val_acc:.1f}%)...")
    save_path = Path("models/parallel_pattern_recognizer.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, save_path)
    print(f"   Saved to: {save_path}")

    # Also save to models/final/
    final_path = Path("models/final/parallel_pattern_recognizer.pt")
    torch.save(best_state, final_path)
    print(f"   Saved to: {final_path}")

    print("\n" + "=" * 70)
    print("      TRAINING COMPLETE")
    print("=" * 70)
    print(f"   Final validation accuracy: {best_val_acc:.1f}%")
    print(f"   Model saved to: {save_path}")

    # Test batch detection
    print("\n[Testing batch detection...]")
    model.load_state_dict(best_state)
    model.eval()

    test_seqs = [seq for seq, _ in val_seqs[:8]]
    test_labels = [label for _, label in val_seqs[:8]]

    results = model.detect_batch(test_seqs)
    print("   Batch detection results:")
    patterns = ['MEMSET', 'MEMCPY', 'STRLEN', 'POLLING', 'BUBBLE_SORT', 'UNKNOWN']
    for i, ((pred_idx, conf), true_label) in enumerate(zip(results, test_labels)):
        status = "✅" if pred_idx == true_label else "❌"
        print(f"   {status} Pred: {patterns[pred_idx]} ({conf:.2f}), True: {patterns[true_label]}")


if __name__ == "__main__":
    train()
