#!/usr/bin/env python3
"""
Train Memory Oracle LSTM Predictor

Trains the Memory Access Predictor on collected memory traces to:
1. Predict the next N memory address deltas
2. Classify access patterns (sequential, strided, pointer-chase, random)
3. Provide confidence scores for prefetching decisions

The key insight: predicting DELTAS not absolute addresses generalizes much
better across different memory regions and program instances.
"""

import json
import math
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════════

class MemoryTraceDataset(Dataset):
    """PyTorch Dataset for memory access traces."""

    def __init__(self, data_file: str, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        print(f"Loading {data_file}...")
        with open(data_file, 'r') as f:
            data = json.load(f)

        self.sequences = data['sequences']
        self.metadata = data.get('metadata', {})

        # Pattern label mapping
        self.pattern_labels = {
            'sequential': 0,
            'strided': 1,
            'pointer-chase': 2,
            'tree-traversal': 2,  # Similar to pointer-chase
            'hash-lookup': 3,
            'stack': 0,  # Similar to sequential
            'memcpy': 0,  # Similar to sequential
        }

        print(f"  Loaded {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = self.sequences[idx]
        seq = item['sequence']
        deltas = item['deltas']

        # Encode sequence into features
        features = self._encode_sequence(seq)

        # Delta targets (normalized)
        delta_tensor = torch.tensor(deltas, dtype=torch.float32)
        # Normalize deltas using log scale for better learning
        delta_signs = torch.sign(delta_tensor)
        delta_mags = torch.log1p(torch.abs(delta_tensor)) / 20.0  # Normalize to ~[-1, 1]
        delta_normalized = delta_signs * delta_mags

        # Pattern labels (majority vote from sequence)
        pattern_counts = {}
        for trace in seq:
            p = trace.get('pattern', 'unknown')
            # Map to base pattern type
            for key in self.pattern_labels:
                if key in p:
                    p = key
                    break
            pattern_counts[p] = pattern_counts.get(p, 0) + 1

        majority_pattern = max(pattern_counts.items(), key=lambda x: x[1])[0]
        pattern_label = self.pattern_labels.get(majority_pattern, 3)  # 3 = random/unknown

        return features, delta_normalized[:-1], torch.tensor(pattern_label)

    def _encode_sequence(self, seq: List[Dict]) -> torch.Tensor:
        """Encode a sequence of memory accesses into feature tensors."""
        features = []

        prev_addr = seq[0]['addr']

        for trace in seq:
            addr = trace['addr']
            size = trace['size']
            is_load = 1.0 if trace['type'] == 'load' else 0.0

            delta = addr - prev_addr
            abs_delta = abs(delta)

            # Feature vector per access
            feat = torch.tensor([
                delta / 1e6,  # Normalized delta
                math.log1p(abs_delta) / 20.0,  # Log-scale magnitude
                1.0 if delta >= 0 else -1.0,  # Sign
                1.0 if abs_delta <= 8 else 0.0,  # Sequential indicator
                1.0 if abs_delta in [8, 16, 24, 32, 48, 64] else 0.0,  # Common stride
                1.0 if abs_delta % 4 == 0 else 0.0,  # Word aligned
                1.0 if abs_delta % 8 == 0 else 0.0,  # Double-word aligned
                float(min(size, 8)) / 8.0,  # Size encoding
                is_load,  # Load vs store
                (trace['pc'] & 0xFF) / 255.0 if 'pc' in trace else 0.0,  # PC low bits
            ], dtype=torch.float32)

            features.append(feat)
            prev_addr = addr

        return torch.stack(features)


# ════════════════════════════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════════════════════════════

class MemoryOracleLSTM(nn.Module):
    """LSTM model for memory access prediction."""

    def __init__(
        self,
        input_dim: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        lookahead: int = 8,
        num_patterns: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lookahead = lookahead

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Delta predictor (predict next N deltas from each position)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, lookahead),
        )

        # Confidence predictor
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, lookahead),
            nn.Sigmoid()
        )

        # Pattern classifier
        self.pattern_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_patterns)
        )

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        """
        Forward pass.

        Args:
            x: Input features [batch, seq_len, input_dim]
            return_hidden: Whether to return LSTM hidden states

        Returns:
            delta_preds: Predicted deltas [batch, seq_len, lookahead]
            confidence: Confidence scores [batch, seq_len, lookahead]
            pattern_logits: Pattern classification [batch, num_patterns]
            hidden: (optional) LSTM hidden state
        """
        batch_size, seq_len, _ = x.shape

        # Encode features
        encoded = self.feature_encoder(x)  # [batch, seq_len, 64]

        # LSTM
        lstm_out, hidden = self.lstm(encoded)  # [batch, seq_len, hidden]

        # Predict deltas from each position
        delta_preds = self.delta_head(lstm_out)  # [batch, seq_len, lookahead]
        confidence = self.confidence_head(lstm_out)  # [batch, seq_len, lookahead]

        # Pattern classification from final hidden state
        final_hidden = lstm_out[:, -1, :]  # [batch, hidden]
        pattern_logits = self.pattern_head(final_hidden)  # [batch, num_patterns]

        if return_hidden:
            return delta_preds, confidence, pattern_logits, hidden
        return delta_preds, confidence, pattern_logits


# ════════════════════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════════════════════

def train_epoch(model, dataloader, optimizer, device, lookahead: int = 8):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    delta_loss_sum = 0
    pattern_loss_sum = 0
    num_batches = 0

    criterion_delta = nn.MSELoss()
    criterion_pattern = nn.CrossEntropyLoss()

    for features, delta_targets, pattern_targets in dataloader:
        features = features.to(device)
        delta_targets = delta_targets.to(device)
        pattern_targets = pattern_targets.to(device)

        optimizer.zero_grad()

        # Forward
        delta_preds, confidence, pattern_logits = model(features)

        # Delta loss: predict next delta from each position
        # We only predict up to seq_len - lookahead positions
        seq_len = features.shape[1]
        valid_len = seq_len - lookahead

        if valid_len > 0:
            # Align predictions with targets
            # delta_preds[:, i, 0] should match delta_targets[:, i]
            delta_loss = criterion_delta(
                delta_preds[:, :valid_len, 0],
                delta_targets[:, :valid_len]
            )
        else:
            delta_loss = torch.tensor(0.0, device=device)

        # Pattern classification loss
        pattern_loss = criterion_pattern(pattern_logits, pattern_targets)

        # Combined loss
        loss = delta_loss + 0.5 * pattern_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        delta_loss_sum += delta_loss.item()
        pattern_loss_sum += pattern_loss.item()
        num_batches += 1

    return {
        'total': total_loss / num_batches,
        'delta': delta_loss_sum / num_batches,
        'pattern': pattern_loss_sum / num_batches
    }


def validate(model, dataloader, device, lookahead: int = 8):
    """Validate model."""
    model.eval()
    total_loss = 0
    delta_mae_sum = 0
    pattern_correct = 0
    pattern_total = 0
    num_batches = 0

    criterion_delta = nn.MSELoss()
    criterion_pattern = nn.CrossEntropyLoss()

    with torch.no_grad():
        for features, delta_targets, pattern_targets in dataloader:
            features = features.to(device)
            delta_targets = delta_targets.to(device)
            pattern_targets = pattern_targets.to(device)

            # Forward
            delta_preds, confidence, pattern_logits = model(features)

            # Losses
            seq_len = features.shape[1]
            valid_len = seq_len - lookahead

            if valid_len > 0:
                delta_loss = criterion_delta(
                    delta_preds[:, :valid_len, 0],
                    delta_targets[:, :valid_len]
                )

                # MAE in original scale (denormalize)
                pred_denorm = torch.sign(delta_preds[:, :valid_len, 0]) * (
                    torch.expm1(torch.abs(delta_preds[:, :valid_len, 0]) * 20.0)
                )
                target_denorm = torch.sign(delta_targets[:, :valid_len]) * (
                    torch.expm1(torch.abs(delta_targets[:, :valid_len]) * 20.0)
                )
                delta_mae = torch.mean(torch.abs(pred_denorm - target_denorm))
                delta_mae_sum += delta_mae.item()
            else:
                delta_loss = torch.tensor(0.0, device=device)

            pattern_loss = criterion_pattern(pattern_logits, pattern_targets)

            # Pattern accuracy
            _, predicted = torch.max(pattern_logits, 1)
            pattern_correct += (predicted == pattern_targets).sum().item()
            pattern_total += pattern_targets.size(0)

            total_loss += (delta_loss + 0.5 * pattern_loss).item()
            num_batches += 1

    return {
        'total': total_loss / num_batches,
        'delta_mae': delta_mae_sum / num_batches,
        'pattern_acc': pattern_correct / pattern_total if pattern_total > 0 else 0
    }


def main():
    print("=" * 70)
    print("  MEMORY ORACLE LSTM TRAINING")
    print("=" * 70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Hyperparameters
    config = {
        'hidden_size': 128,
        'num_layers': 2,
        'lookahead': 8,
        'num_patterns': 4,
        'dropout': 0.1,
        'batch_size': 64,
        'lr': 0.001,
        'epochs': 50,
        'patience': 10
    }

    print(f"\nConfig:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Data
    data_dir = Path(__file__).parent
    train_file = data_dir / "memory_traces_train.json"
    val_file = data_dir / "memory_traces_val.json"

    if not train_file.exists():
        print(f"\n[ERROR] Training data not found at {train_file}")
        print("Run collect_memory_traces_comprehensive.py first!")
        return

    train_dataset = MemoryTraceDataset(str(train_file), device=device)
    val_dataset = MemoryTraceDataset(str(val_file), device=device)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Model
    model = MemoryOracleLSTM(
        input_dim=10,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        lookahead=config['lookahead'],
        num_patterns=config['num_patterns'],
        dropout=config['dropout']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    print("\n" + "=" * 70)
    print("  TRAINING")
    print("=" * 70)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(config['epochs']):
        start_time = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, config['lookahead'])

        # Validate
        val_metrics = validate(model, val_loader, device, config['lookahead'])

        elapsed = time.time() - start_time
        scheduler.step(val_metrics['total'])
        current_lr = optimizer.param_groups[0]['lr']

        # Print progress
        print(f"Epoch {epoch+1:3d}/{config['epochs']} ({elapsed:.1f}s) | "
              f"Train: {train_metrics['total']:.4f} (δ:{train_metrics['delta']:.4f}, p:{train_metrics['pattern']:.4f}) | "
              f"Val: {val_metrics['total']:.4f} (MAE:{val_metrics['delta_mae']:.0f}, acc:{val_metrics['pattern_acc']:.1%}) | "
              f"LR: {current_lr:.2e}")

        # Early stopping
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION")
    print("=" * 70)

    final_metrics = validate(model, val_loader, device, config['lookahead'])
    print(f"\n  Final Validation Loss: {final_metrics['total']:.4f}")
    print(f"  Delta MAE (bytes): {final_metrics['delta_mae']:.0f}")
    print(f"  Pattern Accuracy: {final_metrics['pattern_acc']:.1%}")

    # Save model
    model_path = data_dir / "memory_oracle_lstm.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': final_metrics
    }, model_path)
    print(f"\n  Saved model to: {model_path}")

    # Also save in format compatible with memory_oracle.py
    predictor_path = data_dir / "memory_oracle_predictor.pt"
    torch.save(model.state_dict(), predictor_path)
    print(f"  Saved predictor to: {predictor_path}")

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Model can be loaded in memory_oracle.py for prefetch predictions")


if __name__ == "__main__":
    main()
