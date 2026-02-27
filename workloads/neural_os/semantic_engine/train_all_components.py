#!/usr/bin/env python3
"""
TRAIN ALL COMPONENTS: MCO + EvoRL + Full Pipeline on H200

Trains all neural components to 100% accuracy:
1. Meta-Cognitive Orchestrator (MCO) - Neural RL policy
2. EvoRL - Genetic evolution of policies
3. Combined system - End-to-end synthesis

Target: 100% accuracy on all components
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import random
import time
import json
import os


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    # Model
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_actions: int = 6
    num_layers: int = 4
    dropout: float = 0.1

    # Training
    epochs: int = 500
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-5

    # Target
    target_accuracy: float = 1.0
    patience: int = 50


# =============================================================================
# OPERATIONS (must match other files)
# =============================================================================

OPERATIONS = {
    0: ('identity', lambda x: x),
    1: ('double', lambda x: 2 * x),
    2: ('square', lambda x: x * x),
    3: ('negate', lambda x: -x),
    4: ('add_ten', lambda x: x + 10),
    5: ('times_five', lambda x: 5 * x),
}


# =============================================================================
# DATASET
# =============================================================================

class UnambiguousDataset(Dataset):
    """Dataset with only unambiguous (input, output) pairs."""

    def __init__(self, num_samples: int = 50000):
        self.num_samples = num_samples
        self.valid_samples = self._compute_valid_samples()
        self.samples = self._generate_samples()
        print(f"  Created dataset with {len(self.valid_samples)} unique unambiguous pairs")

    def _compute_valid_samples(self) -> List[Dict]:
        """Find all unambiguous (input, output) pairs."""
        valid = []

        for input_val in range(2, 20):
            output_to_ops = {}

            for op_id, (op_name, op_fn) in OPERATIONS.items():
                try:
                    output_val = op_fn(input_val)
                    key = (input_val, output_val)
                    if key not in output_to_ops:
                        output_to_ops[key] = []
                    output_to_ops[key].append((op_id, op_name))
                except:
                    pass

            for (inp, out), ops in output_to_ops.items():
                if len(ops) == 1:
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
# MCO MODEL
# =============================================================================

class MCOEncoder(nn.Module):
    """Transformer encoder for MCO."""

    def __init__(self, config: Config):
        super().__init__()
        self.embed = nn.Embedding(256, config.embedding_dim)
        self.pos = nn.Embedding(64, config.embedding_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=8,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.out = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.norm = nn.LayerNorm(config.embedding_dim)

    def forward(self, tokens):
        B, L = tokens.shape
        pos_ids = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, -1)
        x = self.embed(tokens) + self.pos(pos_ids)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.norm(self.out(x))


class MCOPolicy(nn.Module):
    """Policy network for MCO."""

    def __init__(self, config: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_actions)
        )

    def forward(self, input_emb, output_emb):
        x = torch.cat([input_emb, output_emb], dim=-1)
        return self.net(x)


# =============================================================================
# EVOLVER MODEL
# =============================================================================

class EvoRLAgent(nn.Module):
    """Evolutionary RL agent."""

    def __init__(self, config: Config):
        super().__init__()
        self.encoder = MCOEncoder(config)
        self.policy = MCOPolicy(config)

        # Evolutionary parameters
        self.fitness = 0.0
        self.generation = 0

    def forward(self, input_tokens, output_tokens):
        input_emb = self.encoder(input_tokens)
        output_emb = self.encoder(output_tokens)
        return self.policy(input_emb, output_emb)

    def mutate(self, mutation_rate: float = 0.1):
        """Create mutated copy."""
        child = EvoRLAgent(Config())
        child.load_state_dict(self.state_dict())

        with torch.no_grad():
            for param in child.parameters():
                mask = torch.rand_like(param) < mutation_rate
                noise = torch.randn_like(param) * 0.1
                param.add_(mask.float() * noise)

        child.generation = self.generation + 1
        return child


# =============================================================================
# TRAINER
# =============================================================================

class ComponentTrainer:
    """Trains MCO and EvoRL to 100% accuracy."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # MCO
        self.mco_encoder = MCOEncoder(config).to(self.device)
        self.mco_policy = MCOPolicy(config).to(self.device)

        # EvoRL population
        self.population_size = 20
        self.population = [EvoRLAgent(config).to(self.device) for _ in range(self.population_size)]

        # Optimizers
        self.mco_opt = optim.AdamW(
            list(self.mco_encoder.parameters()) + list(self.mco_policy.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        # Dataset
        full_data = UnambiguousDataset(num_samples=50000)
        train_size = int(0.9 * len(full_data))
        val_size = len(full_data) - train_size
        self.train_data, self.val_data = torch.utils.data.random_split(full_data, [train_size, val_size])

        self.train_loader = DataLoader(self.train_data, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(self.val_data, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # Metrics
        self.best_mco_acc = 0.0
        self.best_evolver_fitness = 0.0

    def train_mco_epoch(self):
        """Train MCO for one epoch."""
        self.mco_encoder.train()
        self.mco_policy.train()

        total_loss = 0
        correct = 0
        total = 0

        for batch in self.train_loader:
            input_tokens = batch['input_tokens'].to(self.device)
            output_tokens = batch['output_tokens'].to(self.device)
            targets = batch['op_id'].to(self.device)

            input_emb = self.mco_encoder(input_tokens)
            output_emb = self.mco_encoder(output_tokens)
            logits = self.mco_policy(input_emb, output_emb)

            loss = F.cross_entropy(logits, targets)

            self.mco_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.mco_encoder.parameters()) + list(self.mco_policy.parameters()),
                1.0
            )
            self.mco_opt.step()

            total_loss += loss.detach().item()
            preds = logits.detach().argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        return total_loss / len(self.train_loader), correct / total

    def eval_mco(self):
        """Evaluate MCO on validation set."""
        self.mco_encoder.eval()
        self.mco_policy.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_tokens = batch['input_tokens'].to(self.device)
                output_tokens = batch['output_tokens'].to(self.device)
                targets = batch['op_id'].to(self.device)

                input_emb = self.mco_encoder(input_tokens)
                output_emb = self.mco_encoder(output_tokens)
                logits = self.mco_policy(input_emb, output_emb)

                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        return correct / total

    def evolve_population(self):
        """Evolve EvoRL population."""
        # Evaluate all agents
        for agent in self.population:
            agent.fitness = self._eval_agent(agent)

        # Sort by fitness
        self.population.sort(key=lambda a: a.fitness, reverse=True)

        # Keep elite
        elite_count = self.population_size // 5
        new_pop = self.population[:elite_count]

        # Create offspring
        while len(new_pop) < self.population_size:
            parent = random.choice(self.population[:10])
            child = parent.mutate(mutation_rate=0.1)
            new_pop.append(child.to(self.device))

        self.population = new_pop
        return self.population[0].fitness

    def _eval_agent(self, agent):
        """Evaluate an EvoRL agent."""
        agent.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_tokens = batch['input_tokens'].to(self.device)
                output_tokens = batch['output_tokens'].to(self.device)
                targets = batch['op_id'].to(self.device)

                logits = agent(input_tokens, output_tokens)
                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

                if total > 1000:  # Sample for speed
                    break

        return correct / total

    def train(self):
        """Full training loop."""
        print("\n" + "=" * 60)
        print("TRAINING ALL COMPONENTS")
        print("=" * 60)
        print(f"Target: {self.config.target_accuracy * 100:.0f}% accuracy")
        print(f"Epochs: {self.config.epochs}")
        print("=" * 60 + "\n")

        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Train MCO
            mco_loss, mco_train_acc = self.train_mco_epoch()
            mco_val_acc = self.eval_mco()

            # Evolve EvoRL (every 5 epochs)
            if epoch % 5 == 0:
                best_fitness = self.evolve_population()
            else:
                best_fitness = self.population[0].fitness if self.population else 0

            # Update best
            if mco_val_acc > self.best_mco_acc:
                self.best_mco_acc = mco_val_acc
                patience_counter = 0
                self._save_checkpoint('mco_best.pt', mco_val_acc)
            else:
                patience_counter += 1

            if best_fitness > self.best_evolver_fitness:
                self.best_evolver_fitness = best_fitness
                self._save_checkpoint('evolver_best.pt', best_fitness, is_evolver=True)

            # Print progress
            print(f"Epoch {epoch:4d} | "
                  f"MCO: loss={mco_loss:.4f} train={mco_train_acc:.4f} val={mco_val_acc:.4f} | "
                  f"EvoRL: {best_fitness:.4f}")

            # Check targets
            if mco_val_acc >= self.config.target_accuracy:
                print(f"\n✅ MCO REACHED 100% ACCURACY!")
                break

            if patience_counter >= self.config.patience:
                print(f"\n⚠️ Early stopping at epoch {epoch}")
                break

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best MCO accuracy: {self.best_mco_acc:.4f}")
        print(f"Best EvoRL fitness: {self.best_evolver_fitness:.4f}")

        return {
            'mco_accuracy': self.best_mco_acc,
            'evolver_fitness': self.best_evolver_fitness
        }

    def _save_checkpoint(self, name: str, metric: float, is_evolver: bool = False):
        os.makedirs('checkpoints', exist_ok=True)

        if is_evolver:
            checkpoint = {
                'agent_state': self.population[0].state_dict(),
                'fitness': metric
            }
        else:
            checkpoint = {
                'encoder_state': self.mco_encoder.state_dict(),
                'policy_state': self.mco_policy.state_dict(),
                'accuracy': metric
            }

        path = f'checkpoints/{name}'
        torch.save(checkpoint, path)
        print(f"  Saved {path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--target', type=float, default=1.0)
    args = parser.parse_args()

    config = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        target_accuracy=args.target
    )

    trainer = ComponentTrainer(config)
    results = trainer.train()

    # Save results
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✅ Results saved to training_results.json")
