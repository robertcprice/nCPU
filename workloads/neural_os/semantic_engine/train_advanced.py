#!/usr/bin/env python3
"""
ADVANCED TRAINING: Harder Synthesis Tasks for Singularity Core

This script trains on progressively harder tasks:
1. Level 1: Single operations (identity, double, square, etc.)
2. Level 2: Compositions (double then square, add then multiply)
3. Level 3: Conditional operations (if x > 0 then double else negate)
4. Level 4: Recursive patterns (factorial-like, fibonacci-like)

Architecture improvements based on Grok's recommendations:
- Curriculum learning (easy → hard)
- Contrastive loss for better embeddings
- Attention over operation history
- Memory-augmented networks for compositions

Target: 100% on all levels
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import random
import time
import json
import os
from enum import Enum


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AdvancedConfig:
    # Architecture
    embedding_dim: int = 512
    hidden_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1

    # Memory (for compositions)
    memory_size: int = 64
    memory_dim: int = 256

    # Training
    epochs: int = 500
    batch_size: int = 256
    lr: float = 3e-5
    weight_decay: float = 1e-5
    warmup_steps: int = 500

    # Curriculum
    curriculum_levels: int = 4
    level_threshold: float = 0.95  # Advance when accuracy > this

    # Target
    target_accuracy: float = 1.0


# =============================================================================
# OPERATION LEVELS
# =============================================================================

class OpLevel(Enum):
    BASIC = 1       # Single operations
    COMPOSE = 2     # Two-operation compositions
    CONDITIONAL = 3 # Conditional operations
    RECURSIVE = 4   # Recursive/iterative patterns


# Level 1: Basic operations
BASIC_OPS = {
    0: ('identity', lambda x: x),
    1: ('double', lambda x: 2 * x),
    2: ('square', lambda x: x * x),
    3: ('negate', lambda x: -x),
    4: ('add_ten', lambda x: x + 10),
    5: ('times_five', lambda x: 5 * x),
    6: ('increment', lambda x: x + 1),
    7: ('decrement', lambda x: x - 1),
}

# Level 2: Compositions (op1 then op2)
COMPOSE_OPS = {
    8: ('double_then_square', lambda x: (2 * x) ** 2),
    9: ('square_then_double', lambda x: 2 * (x * x)),
    10: ('add_ten_then_double', lambda x: 2 * (x + 10)),
    11: ('negate_then_square', lambda x: (-x) ** 2),
    12: ('increment_twice', lambda x: x + 2),
    13: ('double_then_add_ten', lambda x: 2 * x + 10),
}

# Level 3: Conditional operations
CONDITIONAL_OPS = {
    14: ('abs_value', lambda x: x if x >= 0 else -x),
    15: ('sign', lambda x: 1 if x > 0 else (-1 if x < 0 else 0)),
    16: ('relu', lambda x: max(0, x)),
    17: ('clamp_ten', lambda x: min(max(x, -10), 10)),
}

# Level 4: Recursive-like patterns
RECURSIVE_OPS = {
    18: ('sum_to_n', lambda x: x * (x + 1) // 2 if x > 0 else 0),
    19: ('power_of_two', lambda x: 2 ** x if 0 <= x <= 10 else 0),
    20: ('triangular', lambda x: x * (x + 1) // 2 if x >= 0 else 0),
}

ALL_OPS = {**BASIC_OPS, **COMPOSE_OPS, **CONDITIONAL_OPS, **RECURSIVE_OPS}


# =============================================================================
# DATASET
# =============================================================================

class CurriculumDataset(Dataset):
    """Dataset with curriculum learning - progressively harder tasks."""

    def __init__(self, level: int = 1, num_samples: int = 50000):
        self.level = level
        self.num_samples = num_samples
        self.ops = self._get_ops_for_level(level)
        self.valid_samples = self._compute_valid_samples()
        self.samples = self._generate_samples()
        print(f"  Level {level}: {len(self.ops)} operations, {len(self.valid_samples)} unique pairs")

    def _get_ops_for_level(self, level: int) -> Dict:
        """Get operations available at this curriculum level."""
        ops = dict(BASIC_OPS)
        if level >= 2:
            ops.update(COMPOSE_OPS)
        if level >= 3:
            ops.update(CONDITIONAL_OPS)
        if level >= 4:
            ops.update(RECURSIVE_OPS)
        return ops

    def _compute_valid_samples(self) -> List[Dict]:
        """Find unambiguous (input, output) pairs."""
        valid = []

        # Use appropriate input range per level
        if self.level <= 2:
            inputs = list(range(2, 15))
        elif self.level == 3:
            inputs = list(range(-10, 15))  # Include negatives for conditionals
        else:
            inputs = list(range(0, 12))  # Small positive for recursive

        for input_val in inputs:
            output_to_ops = {}

            for op_id, (op_name, op_fn) in self.ops.items():
                try:
                    output_val = op_fn(input_val)
                    # Skip if output is too large
                    if abs(output_val) > 10000:
                        continue
                    key = (input_val, output_val)
                    if key not in output_to_ops:
                        output_to_ops[key] = []
                    output_to_ops[key].append((op_id, op_name))
                except:
                    pass

            for (inp, out), ops in output_to_ops.items():
                if len(ops) == 1:  # Unambiguous
                    op_id, op_name = ops[0]
                    valid.append({
                        'input': inp,
                        'output': out,
                        'op_id': op_id,
                        'op_name': op_name
                    })

        return valid

    def _tokenize(self, val: int, max_len: int = 16) -> List[int]:
        s = str(val)
        tokens = [ord(c) % 256 for c in s[:max_len]]
        return tokens + [0] * (max_len - len(tokens))

    def _generate_samples(self) -> List[Dict]:
        samples = []
        for _ in range(self.num_samples):
            s = random.choice(self.valid_samples)
            samples.append({
                'input_tokens': self._tokenize(s['input']),
                'output_tokens': self._tokenize(s['output']),
                'op_id': s['op_id']
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'input_tokens': torch.tensor(s['input_tokens'], dtype=torch.long),
            'output_tokens': torch.tensor(s['output_tokens'], dtype=torch.long),
            'op_id': torch.tensor(s['op_id'], dtype=torch.long)
        }


# =============================================================================
# ADVANCED ARCHITECTURE
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MemoryAugmentedEncoder(nn.Module):
    """Encoder with external memory for tracking compositions."""

    def __init__(self, config: AdvancedConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed = nn.Embedding(256, config.embedding_dim)
        self.pos_enc = PositionalEncoding(config.embedding_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # External memory
        self.memory = nn.Parameter(torch.randn(1, config.memory_size, config.memory_dim))
        self.memory_proj = nn.Linear(config.memory_dim, config.embedding_dim)
        self.memory_attention = nn.MultiheadAttention(
            config.embedding_dim, config.num_heads, dropout=config.dropout, batch_first=True
        )

        # Output
        self.output_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.norm = nn.LayerNorm(config.embedding_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape

        # Embed tokens
        x = self.embed(tokens)
        x = self.pos_enc(x)

        # Transformer encoding
        x = self.transformer(x)

        # Memory attention
        memory = self.memory.expand(B, -1, -1)
        memory = self.memory_proj(memory)
        x_mem, _ = self.memory_attention(x, memory, memory)
        x = x + x_mem

        # Pool and project
        x = x.mean(dim=1)
        x = self.norm(self.output_proj(x))

        return x


class ContrastivePolicy(nn.Module):
    """Policy with contrastive learning for better separation."""

    def __init__(self, config: AdvancedConfig, num_classes: int):
        super().__init__()
        self.config = config

        # State encoder
        self.state_net = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Classification head
        self.classifier = nn.Linear(config.hidden_dim, num_classes)

        # Contrastive projection head
        self.projector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 128)
        )

    def forward(self, input_emb: torch.Tensor, output_emb: torch.Tensor):
        x = torch.cat([input_emb, output_emb], dim=-1)
        features = self.state_net(x)
        logits = self.classifier(features)
        projections = self.projector(features)
        return logits, F.normalize(projections, dim=-1)


# =============================================================================
# TRAINER
# =============================================================================

class AdvancedTrainer:
    """Curriculum trainer with contrastive learning."""

    def __init__(self, config: AdvancedConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Models
        self.encoder = MemoryAugmentedEncoder(config).to(self.device)
        self.policy = ContrastivePolicy(config, num_classes=len(ALL_OPS)).to(self.device)

        # Optimizer with warmup
        self.optimizer = optim.AdamW(
            list(self.encoder.parameters()) + list(self.policy.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        # Metrics per level
        self.level_accuracies = {}
        self.current_level = 1

    def _create_dataloader(self, level: int, is_train: bool = True) -> DataLoader:
        """Create dataloader for a curriculum level."""
        dataset = CurriculumDataset(level=level, num_samples=50000 if is_train else 10000)

        if is_train:
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            train_data, _ = torch.utils.data.random_split(dataset, [train_size, val_size])
            return DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
        else:
            _, val_data = torch.utils.data.random_split(dataset,
                [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])
            return DataLoader(val_data, batch_size=self.config.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    def contrastive_loss(self, projections: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07):
        """NT-Xent contrastive loss."""
        # Similarity matrix
        sim_matrix = torch.matmul(projections, projections.T) / temperature

        # Mask for positive pairs (same label)
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()

        # Remove diagonal
        mask = torch.eye(len(projections), device=self.device)
        positive_mask = positive_mask * (1 - mask)

        # Compute loss
        exp_sim = torch.exp(sim_matrix) * (1 - mask)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Average over positive pairs
        loss = -(positive_mask * log_prob).sum() / (positive_mask.sum() + 1e-8)
        return loss

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.encoder.train()
        self.policy.train()

        total_loss = 0
        total_ce_loss = 0
        total_contrastive_loss = 0
        correct = 0
        total = 0

        for batch in dataloader:
            input_tokens = batch['input_tokens'].to(self.device)
            output_tokens = batch['output_tokens'].to(self.device)
            targets = batch['op_id'].to(self.device)

            # Forward
            input_emb = self.encoder(input_tokens)
            output_emb = self.encoder(output_tokens)
            logits, projections = self.policy(input_emb, output_emb)

            # Classification loss
            ce_loss = F.cross_entropy(logits, targets)

            # Contrastive loss
            con_loss = self.contrastive_loss(projections, targets)

            # Combined loss
            loss = ce_loss + 0.1 * con_loss

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.policy.parameters()),
                1.0
            )
            self.optimizer.step()

            # Metrics
            total_loss += loss.detach().item()
            total_ce_loss += ce_loss.detach().item()
            total_contrastive_loss += con_loss.detach().item()
            preds = logits.detach().argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        n_batches = len(dataloader)
        return total_loss / n_batches, correct / total

    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate accuracy."""
        self.encoder.eval()
        self.policy.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                input_tokens = batch['input_tokens'].to(self.device)
                output_tokens = batch['output_tokens'].to(self.device)
                targets = batch['op_id'].to(self.device)

                input_emb = self.encoder(input_tokens)
                output_emb = self.encoder(output_tokens)
                logits, _ = self.policy(input_emb, output_emb)

                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        return correct / total

    def train_level(self, level: int) -> float:
        """Train on a specific curriculum level."""
        print(f"\n{'='*60}")
        print(f"CURRICULUM LEVEL {level}")
        print(f"{'='*60}")

        train_loader = self._create_dataloader(level, is_train=True)
        val_loader = self._create_dataloader(level, is_train=False)

        best_acc = 0.0
        patience = 0
        max_patience = 30

        for epoch in range(self.config.epochs):
            loss, train_acc = self.train_epoch(train_loader)
            val_acc = self.evaluate(val_loader)

            if val_acc > best_acc:
                best_acc = val_acc
                patience = 0
                self._save_checkpoint(f'level_{level}_best.pt', val_acc)
            else:
                patience += 1

            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")

            # Check if we can advance
            if val_acc >= self.config.level_threshold:
                print(f"\n  ✅ Level {level} PASSED with {val_acc:.4f} accuracy!")
                break

            if patience >= max_patience:
                print(f"\n  ⚠️ Early stopping at epoch {epoch}")
                break

        self.level_accuracies[level] = best_acc
        return best_acc

    def train_curriculum(self) -> Dict[str, Any]:
        """Full curriculum training."""
        print("\n" + "=" * 70)
        print("ADVANCED CURRICULUM TRAINING")
        print("=" * 70)
        print(f"Levels: {self.config.curriculum_levels}")
        print(f"Target: {self.config.target_accuracy * 100:.0f}% per level")
        print("=" * 70)

        results = {
            'level_accuracies': {},
            'final_level': 0,
            'all_passed': False
        }

        for level in range(1, self.config.curriculum_levels + 1):
            acc = self.train_level(level)
            results['level_accuracies'][level] = acc
            results['final_level'] = level

            if acc < self.config.level_threshold:
                print(f"\n❌ Failed to pass level {level}. Stopping curriculum.")
                break
        else:
            results['all_passed'] = True
            print(f"\n🎉 ALL LEVELS PASSED!")

        # Final summary
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        for level, acc in results['level_accuracies'].items():
            status = "✅" if acc >= self.config.level_threshold else "❌"
            print(f"  Level {level}: {acc:.4f} {status}")

        return results

    def _save_checkpoint(self, name: str, accuracy: float):
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint = {
            'encoder_state': self.encoder.state_dict(),
            'policy_state': self.policy.state_dict(),
            'accuracy': accuracy,
            'config': self.config.__dict__
        }
        torch.save(checkpoint, f'checkpoints/{name}')


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--levels', type=int, default=4)
    args = parser.parse_args()

    config = AdvancedConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        curriculum_levels=args.levels
    )

    trainer = AdvancedTrainer(config)
    results = trainer.train_curriculum()

    with open('advanced_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✅ Results saved to advanced_training_results.json")
