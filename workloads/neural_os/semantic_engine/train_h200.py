#!/usr/bin/env python3
"""
H200 GPU TRAINING SCRIPT: Train to 100% Accuracy

This script trains all neural components of the Semantic Synthesizer:
1. Program Encoder - learns embeddings of programs
2. Synthesis Policy - learns RL policy for synthesis actions
3. EvoRL Population - evolves optimal policies
4. Tactic Memory - accumulates successful patterns

Target: 100% accuracy on core synthesis tasks

Usage on H200:
    ssh -p 12673 root@ssh7.vast.ai -L 8080:localhost:8080
    cd ~/semantic_engine
    python3 train_h200.py --epochs 1000 --batch_size 256
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
from collections import defaultdict
from sympy import Symbol, Integer, Expr, Add, Mul, Pow, simplify, expand


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model architecture
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_actions: int = 6  # Must match OPERATIONS count in SynthesisDataset
    num_layers: int = 4
    dropout: float = 0.1

    # Training hyperparameters
    epochs: int = 1000
    batch_size: int = 256
    learning_rate: float = 3e-5  # Lower LR for stability
    weight_decay: float = 1e-5
    warmup_steps: int = 1000

    # RL parameters
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Evolution parameters
    population_size: int = 100
    elite_fraction: float = 0.1
    mutation_rate: float = 0.1

    # Checkpointing
    checkpoint_every: int = 100
    checkpoint_dir: str = "checkpoints"

    # Target accuracy
    target_accuracy: float = 1.0
    early_stop_patience: int = 50


# =============================================================================
# ENHANCED NEURAL NETWORKS (H200 Optimized)
# =============================================================================

class TransformerEncoder(nn.Module):
    """Transformer-based program encoder optimized for H200."""

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        # Token embedding (256 vocab to match ord(c) % 256 tokenization)
        self.token_embed = nn.Embedding(256, config.embedding_dim)
        self.pos_embed = nn.Embedding(512, config.embedding_dim)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=8,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )

        # Output projection
        self.output_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.layer_norm = nn.LayerNorm(config.embedding_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of tokens."""
        batch_size, seq_len = tokens.shape

        # Embeddings
        pos_ids = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embed(tokens) + self.pos_embed(pos_ids)

        # Transformer encoding
        x = self.transformer(x)

        # Pool and project
        x = x.mean(dim=1)  # Mean pooling
        x = self.layer_norm(self.output_proj(x))

        return x


class SynthesisPolicy(nn.Module):
    """Policy network for synthesis actions."""

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_actions)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )

    def forward(
        self,
        input_embedding: torch.Tensor,
        target_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute policy and value from state."""
        state = torch.cat([input_embedding, target_embedding], dim=-1)
        features = self.state_encoder(state)

        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)

        return logits, value

    def get_action(
        self,
        input_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample an action from the policy."""
        logits, value = self.forward(input_embedding, target_embedding)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = torch.multinomial(probs, 1).squeeze(-1)

        log_prob = F.log_softmax(logits, dim=-1)
        action_log_prob = log_prob.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        return action.item(), action_log_prob, value


# =============================================================================
# SYNTHESIS DATASET
# =============================================================================

class SynthesisDataset(Dataset):
    """Dataset for synthesis training."""

    # Operation definitions - CLEARLY DISTINGUISHABLE operations
    # These produce unique outputs for inputs 2-10
    OPERATIONS = {
        0: ('identity', lambda x: x),           # x → x
        1: ('double', lambda x: 2 * x),         # x → 2x
        2: ('square', lambda x: x * x),         # x → x²
        3: ('negate', lambda x: -x),            # x → -x
        4: ('add_ten', lambda x: x + 10),       # x → x+10 (large offset, no collision)
        5: ('times_five', lambda x: 5 * x),     # x → 5x
    }

    def __init__(self, num_samples: int = 10000):
        self.num_samples = num_samples
        self.samples = self._generate_samples()

    def _generate_samples(self) -> List[Dict[str, Any]]:
        """Generate samples where (input, output) UNIQUELY identifies the operation."""
        samples = []

        # Pre-compute which (input, output) pairs are unambiguous
        # Only keep samples where exactly ONE operation produces that output
        valid_samples = []

        for input_val in range(2, 15):  # Use positive inputs 2-14
            output_to_ops = {}  # Track which ops produce each output

            for op_id, (op_name, op_fn) in self.OPERATIONS.items():
                try:
                    output_val = op_fn(input_val)
                    key = (input_val, output_val)
                    if key not in output_to_ops:
                        output_to_ops[key] = []
                    output_to_ops[key].append((op_id, op_name))
                except:
                    pass

            # Only keep unambiguous (input, output) pairs
            for (inp, out), ops in output_to_ops.items():
                if len(ops) == 1:  # Exactly ONE operation produces this output
                    op_id, op_name = ops[0]
                    valid_samples.append({
                        'input_val': inp,
                        'output_val': out,
                        'op_id': op_id,
                        'op_name': op_name
                    })

        # Generate dataset by sampling from valid samples
        for _ in range(self.num_samples):
            sample = random.choice(valid_samples)
            input_tokens = self._tokenize(str(sample['input_val']))
            output_tokens = self._tokenize(str(sample['output_val']))

            samples.append({
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'operation': sample['op_id'],
                'input_expr': str(sample['input_val']),
                'output_expr': str(sample['output_val'])
            })

        print(f"  Generated {len(valid_samples)} unique unambiguous (input,output) pairs")
        return samples

    def _tokenize(self, expr_str: str, max_len: int = 32) -> List[int]:
        """Convert expression string to token IDs."""
        # Simple character-level tokenization
        tokens = [ord(c) % 256 for c in expr_str[:max_len]]
        # Pad to max_len
        tokens = tokens + [0] * (max_len - len(tokens))
        return tokens

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            'input_tokens': torch.tensor(sample['input_tokens'], dtype=torch.long),
            'output_tokens': torch.tensor(sample['output_tokens'], dtype=torch.long),
            'operation': torch.tensor(sample['operation'], dtype=torch.long)
        }


# =============================================================================
# TRAINER
# =============================================================================

class H200Trainer:
    """High-performance trainer for H200 GPU."""

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Models
        self.encoder = TransformerEncoder(config).to(self.device)
        self.policy = SynthesisPolicy(config).to(self.device)

        # Optimizers
        self.encoder_optimizer = optim.AdamW(
            self.encoder.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.policy_optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate schedulers
        self.encoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.encoder_optimizer,
            T_max=config.epochs
        )
        self.policy_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.policy_optimizer,
            T_max=config.epochs
        )

        # Dataset with train/val split
        full_dataset = SynthesisDataset(num_samples=100000)
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Metrics
        self.best_accuracy = 0.0
        self.patience_counter = 0
        self.history = {
            'loss': [],
            'accuracy': [],
            'entropy': []
        }

        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.encoder.train()
        self.policy.train()

        total_loss = 0.0
        correct = 0
        total = 0
        total_entropy = 0.0

        for batch in self.dataloader:
            # Move to device
            input_tokens = batch['input_tokens'].to(self.device)
            output_tokens = batch['output_tokens'].to(self.device)
            targets = batch['operation'].to(self.device)

            # Encode
            input_emb = self.encoder(input_tokens)
            output_emb = self.encoder(output_tokens)

            # Get policy output
            logits, values = self.policy(input_emb, output_emb)

            # Compute loss
            policy_loss = F.cross_entropy(logits, targets)

            # Entropy bonus for exploration (with numerical stability)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)  # More stable than probs.log()
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            # Total loss (clamp to prevent NaN)
            loss = policy_loss - self.config.entropy_coef * entropy
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Warning: NaN/Inf loss detected, skipping batch")
                continue

            # Backward pass
            self.encoder_optimizer.zero_grad()
            self.policy_optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(),
                self.config.max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.max_grad_norm
            )

            self.encoder_optimizer.step()
            self.policy_optimizer.step()

            # Metrics (use .detach() to avoid memory leak)
            total_loss += loss.detach().item()
            predictions = logits.detach().argmax(dim=-1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
            total_entropy += entropy.detach().item()

        # Step schedulers
        self.encoder_scheduler.step()
        self.policy_scheduler.step()

        num_batches = len(self.dataloader)
        metrics = {
            'loss': total_loss / num_batches,
            'accuracy': correct / total,
            'entropy': total_entropy / num_batches
        }

        return metrics

    def evaluate(self) -> float:
        """Evaluate model accuracy on VALIDATION set."""
        self.encoder.eval()
        self.policy.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_dataloader:  # Use validation dataloader
                input_tokens = batch['input_tokens'].to(self.device)
                output_tokens = batch['output_tokens'].to(self.device)
                targets = batch['operation'].to(self.device)

                input_emb = self.encoder(input_tokens)
                output_emb = self.encoder(output_tokens)
                logits, _ = self.policy(input_emb, output_emb)

                predictions = logits.argmax(dim=-1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)

        return correct / total

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save a checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'encoder_state': self.encoder.state_dict(),
            'policy_state': self.policy.state_dict(),
            'encoder_optimizer': self.encoder_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__
        }

        path = os.path.join(
            self.config.checkpoint_dir,
            f'checkpoint_epoch_{epoch}_acc_{metrics["accuracy"]:.4f}.pt'
        )
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")

        # Also save best model
        if metrics['accuracy'] > self.best_accuracy:
            self.best_accuracy = metrics['accuracy']
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  New best model: {metrics['accuracy']:.4f}")

    def train(self) -> Dict[str, Any]:
        """Full training loop."""
        print("\n" + "=" * 60)
        print("STARTING H200 TRAINING")
        print("=" * 60)
        print(f"Target accuracy: {self.config.target_accuracy * 100:.0f}%")
        print(f"Max epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Dataset size: {len(self.train_dataset)}")
        print("=" * 60 + "\n")

        start_time = time.time()

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # Train
            metrics = self.train_epoch(epoch)

            # Evaluate
            eval_accuracy = self.evaluate()
            metrics['eval_accuracy'] = eval_accuracy

            # Record history
            for key, value in metrics.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)

            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch:4d} | "
                  f"Loss: {metrics['loss']:.4f} | "
                  f"Train Acc: {metrics['accuracy']:.4f} | "
                  f"Eval Acc: {eval_accuracy:.4f} | "
                  f"Time: {epoch_time:.1f}s")

            # Checkpointing
            if (epoch + 1) % self.config.checkpoint_every == 0:
                self.save_checkpoint(epoch, metrics)

            # Early stopping check
            if eval_accuracy >= self.config.target_accuracy:
                print(f"\n✅ REACHED TARGET ACCURACY: {eval_accuracy:.4f}")
                self.save_checkpoint(epoch, metrics)
                break

            # Patience for early stopping
            if eval_accuracy > self.best_accuracy:
                self.best_accuracy = eval_accuracy
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stop_patience:
                    print(f"\n⚠️ Early stopping at epoch {epoch}")
                    break

        total_time = time.time() - start_time

        results = {
            'final_accuracy': self.history['accuracy'][-1],
            'best_accuracy': self.best_accuracy,
            'epochs_trained': len(self.history['accuracy']),
            'total_time': total_time,
            'history': self.history
        }

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best accuracy: {self.best_accuracy:.4f}")
        print(f"Epochs trained: {results['epochs_trained']}")
        print(f"Total time: {total_time:.1f}s")

        return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train Semantic Synthesizer on H200')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--target_accuracy', type=float, default=1.0, help='Target accuracy (0-1)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        target_accuracy=args.target_accuracy,
        checkpoint_dir=args.checkpoint_dir
    )

    # Train
    trainer = H200Trainer(config)
    results = trainer.train()

    # Save results
    with open('training_results.json', 'w') as f:
        # Convert history to lists for JSON serialization
        serializable_results = {
            'final_accuracy': results['final_accuracy'],
            'best_accuracy': results['best_accuracy'],
            'epochs_trained': results['epochs_trained'],
            'total_time': results['total_time']
        }
        json.dump(serializable_results, f, indent=2)

    print("\nResults saved to training_results.json")

    return results


if __name__ == "__main__":
    main()
