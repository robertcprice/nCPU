#!/usr/bin/env python3
"""
TRAIN V2 H200: Advanced Training with Multi-Objective Loss
Based on Hybrid AI Review recommendations:
1. Advanced multi-objective loss (CE + Contrastive + MDL)
2. Mixed precision (AMP) for H200
3. Curriculum learning
4. Test-time training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import random
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    # Model
    vocab_size: int = 256
    embed_dim: int = 512
    hidden_dim: int = 1024
    num_layers: int = 4
    num_heads: int = 8
    num_operations: int = 21

    # Training
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 100
    warmup_epochs: int = 5

    # Loss weights (from hybrid review)
    ce_weight: float = 1.0
    contrastive_weight: float = 0.5
    mdl_weight: float = 0.1
    value_weight: float = 0.2

    # Advanced
    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Curriculum
    use_curriculum: bool = True
    curriculum_stages: int = 4


# =============================================================================
# ADVANCED LOSS (From Hybrid Review)
# =============================================================================

class AdvancedSynthesisLoss(nn.Module):
    """
    Multi-objective loss combining:
    - Cross-entropy for classification
    - Contrastive learning for better representations
    - MDL proxy for compression preference
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.temperature = 0.1

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute multi-objective loss."""
        losses = {}

        # 1. Cross-entropy loss
        ce_loss = self.ce_loss(logits, targets)
        losses['ce'] = ce_loss

        # 2. Contrastive loss (NT-Xent style)
        if embeddings is not None:
            contrastive_loss = self._contrastive_loss(embeddings, targets)
            losses['contrastive'] = contrastive_loss
        else:
            contrastive_loss = 0.0
            losses['contrastive'] = torch.tensor(0.0, device=logits.device)

        # 3. MDL proxy (encourage confident predictions)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
        mdl_loss = entropy  # Lower entropy = shorter description
        losses['mdl'] = mdl_loss

        # Total
        total = (
            self.config.ce_weight * ce_loss +
            self.config.contrastive_weight * contrastive_loss +
            self.config.mdl_weight * mdl_loss
        )
        losses['total'] = total

        return losses

    def _contrastive_loss(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Supervised contrastive loss."""
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)

        # Compute similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.T) / self.temperature

        # Create mask for same-class samples
        labels = targets.unsqueeze(0) == targets.unsqueeze(1)

        # Exclude diagonal
        mask = torch.eye(len(targets), device=embeddings.device).bool()
        labels = labels & ~mask

        # InfoNCE-style loss
        exp_sim = torch.exp(sim_matrix)
        exp_sim = exp_sim * (~mask).float()  # Exclude self

        # For each sample, contrast with same-class positives
        pos_sim = (exp_sim * labels.float()).sum(dim=1)
        neg_sim = exp_sim.sum(dim=1)

        # Avoid division by zero
        loss = -torch.log((pos_sim + 1e-8) / (neg_sim + 1e-8)).mean()

        return loss


# =============================================================================
# CURRICULUM DATASET
# =============================================================================

class CurriculumSynthesisDataset(Dataset):
    """
    Curriculum learning dataset with progressive difficulty.
    """

    # Operations by difficulty
    CURRICULUM = {
        1: {  # Easy: single operations
            0: ('identity', lambda x: x),
            1: ('double', lambda x: x * 2),
            2: ('negate', lambda x: -x),
            3: ('add_ten', lambda x: x + 10),
        },
        2: {  # Medium: more operations
            4: ('square', lambda x: x * x),
            5: ('abs', lambda x: abs(x)),
            6: ('increment', lambda x: x + 1),
            7: ('decrement', lambda x: x - 1),
        },
        3: {  # Hard: compositions
            8: ('double_square', lambda x: (x * 2) ** 2),
            9: ('square_add_ten', lambda x: x * x + 10),
            10: ('negate_double', lambda x: -x * 2),
        },
        4: {  # Expert: complex patterns
            11: ('cube', lambda x: x ** 3),
            12: ('x_times_x_minus_1', lambda x: x * (x - 1)),
            13: ('triple', lambda x: x * 3),
            14: ('quadruple', lambda x: x * 4),
        }
    }

    def __init__(self, num_samples: int, stage: int = 1, config: TrainingConfig = None):
        self.num_samples = num_samples
        self.stage = min(stage, 4)
        self.config = config or TrainingConfig()

        # Get operations for current stage (cumulative)
        self.operations = {}
        for s in range(1, self.stage + 1):
            self.operations.update(self.CURRICULUM[s])

        self.samples = self._generate_samples()
        print(f"  Stage {stage}: {len(self.operations)} operations, {len(self.samples)} samples")

    def _generate_samples(self) -> List[Dict]:
        """Generate unambiguous training samples."""
        samples = []
        valid_pairs = []

        # Find unambiguous (input, output) pairs
        for input_val in range(2, 20):
            output_to_ops = {}
            for op_id, (op_name, op_fn) in self.operations.items():
                try:
                    output_val = op_fn(input_val)
                    key = (input_val, output_val)
                    if key not in output_to_ops:
                        output_to_ops[key] = []
                    output_to_ops[key].append((op_id, op_name))
                except:
                    pass

            # Only keep unambiguous pairs
            for (inp, out), ops in output_to_ops.items():
                if len(ops) == 1:
                    valid_pairs.append({
                        'input': inp,
                        'output': out,
                        'op_id': ops[0][0],
                        'op_name': ops[0][1]
                    })

        # Generate samples from valid pairs
        for _ in range(self.num_samples):
            pair = random.choice(valid_pairs)
            samples.append({
                'input_tokens': self._tokenize(str(pair['input'])),
                'output_tokens': self._tokenize(str(pair['output'])),
                'operation': pair['op_id'],
                'input_val': pair['input'],
                'output_val': pair['output']
            })

        return samples

    def _tokenize(self, s: str, max_len: int = 32) -> List[int]:
        """Simple character-level tokenization."""
        tokens = [ord(c) % self.config.vocab_size for c in s]
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        return tokens[:max_len]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'input': torch.tensor(sample['input_tokens'], dtype=torch.long),
            'output': torch.tensor(sample['output_tokens'], dtype=torch.long),
            'operation': torch.tensor(sample['operation'], dtype=torch.long)
        }


# =============================================================================
# MODEL
# =============================================================================

class SynthesisTransformer(nn.Module):
    """
    Transformer model for operation classification.
    Returns both logits and embeddings for contrastive loss.
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.input_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.output_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Embedding(64, config.embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Projection head for embeddings (for contrastive loss)
        self.projection = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embed_dim)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.num_operations)
        )

    def forward(self, input_tokens: torch.Tensor, output_tokens: torch.Tensor):
        batch_size = input_tokens.size(0)
        seq_len = input_tokens.size(1)

        # Embeddings
        pos_ids = torch.arange(seq_len, device=input_tokens.device).unsqueeze(0).expand(batch_size, -1)

        input_emb = self.input_embed(input_tokens) + self.pos_embed(pos_ids)
        output_emb = self.output_embed(output_tokens) + self.pos_embed(pos_ids)

        # Encode
        input_enc = self.encoder(input_emb)
        output_enc = self.encoder(output_emb)

        # Pool (mean pooling)
        input_pooled = input_enc.mean(dim=1)
        output_pooled = output_enc.mean(dim=1)

        # Combine
        combined = torch.cat([input_pooled, output_pooled], dim=-1)

        # Get embeddings for contrastive loss
        embeddings = self.projection(input_pooled + output_pooled)

        # Classify
        logits = self.classifier(combined)

        return logits, embeddings


# =============================================================================
# TRAINER
# =============================================================================

class SynthesisTrainer:
    """
    Advanced trainer with:
    - Mixed precision (AMP)
    - Curriculum learning
    - Multi-objective loss
    - Gradient accumulation
    """

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model
        self.model = SynthesisTransformer(self.config).to(self.device)

        # Loss
        self.criterion = AdvancedSynthesisLoss(self.config)

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Learning rate scheduler (cosine annealing)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.learning_rate / 100
        )

        # AMP scaler
        self.scaler = GradScaler() if self.config.use_amp else None

        # Metrics
        self.best_accuracy = 0.0
        self.best_epoch = 0

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()

        total_loss = 0.0
        total_ce = 0.0
        total_contrastive = 0.0
        total_mdl = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(dataloader):
            input_tokens = batch['input'].to(self.device)
            output_tokens = batch['output'].to(self.device)
            targets = batch['operation'].to(self.device)

            # Forward pass with AMP
            with autocast(enabled=self.config.use_amp):
                logits, embeddings = self.model(input_tokens, output_tokens)
                losses = self.criterion(logits, targets, embeddings)

            loss = losses['total'] / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Metrics
            total_loss += losses['total'].item()
            total_ce += losses['ce'].item()
            total_contrastive += losses['contrastive'].item()
            total_mdl += losses['mdl'].item()

            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        n_batches = len(dataloader)
        return {
            'loss': total_loss / n_batches,
            'ce_loss': total_ce / n_batches,
            'contrastive_loss': total_contrastive / n_batches,
            'mdl_loss': total_mdl / n_batches,
            'accuracy': correct / total
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()

        correct = 0
        total = 0
        total_loss = 0.0

        for batch in dataloader:
            input_tokens = batch['input'].to(self.device)
            output_tokens = batch['output'].to(self.device)
            targets = batch['operation'].to(self.device)

            logits, embeddings = self.model(input_tokens, output_tokens)
            losses = self.criterion(logits, targets, embeddings)

            total_loss += losses['total'].item()
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total
        }

    def train(self, num_samples: int = 50000, target_accuracy: float = 1.0):
        """Full training with curriculum."""
        print("="*60)
        print("TRAIN V2: Advanced Multi-Objective Training")
        print("="*60)
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        start_time = time.time()

        # Curriculum training
        for stage in range(1, self.config.curriculum_stages + 1):
            print(f"\n{'='*40}")
            print(f"CURRICULUM STAGE {stage}/{self.config.curriculum_stages}")
            print('='*40)

            # Create dataset for this stage
            train_dataset = CurriculumSynthesisDataset(
                num_samples=num_samples,
                stage=stage,
                config=self.config
            )

            val_dataset = CurriculumSynthesisDataset(
                num_samples=num_samples // 10,
                stage=stage,
                config=self.config
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0
            )

            # Train this stage
            epochs_per_stage = self.config.epochs // self.config.curriculum_stages

            for epoch in range(epochs_per_stage):
                # Train
                train_metrics = self.train_epoch(train_loader, epoch)

                # Validate
                val_metrics = self.validate(val_loader)

                # Update scheduler
                self.scheduler.step()

                # Report
                if (epoch + 1) % 5 == 0 or val_metrics['accuracy'] >= target_accuracy:
                    print(f"  Epoch {epoch+1}/{epochs_per_stage}: "
                          f"train_acc={train_metrics['accuracy']:.2%} "
                          f"val_acc={val_metrics['accuracy']:.2%} "
                          f"loss={train_metrics['loss']:.4f}")

                # Save best
                if val_metrics['accuracy'] > self.best_accuracy:
                    self.best_accuracy = val_metrics['accuracy']
                    self.best_epoch = epoch
                    self._save_checkpoint('checkpoints/v2_best_model.pt')

                # Early stopping
                if val_metrics['accuracy'] >= target_accuracy:
                    print(f"\n  Target accuracy {target_accuracy:.0%} reached!")
                    break

            # Don't break early - complete all curriculum stages
            # Reset best accuracy per stage to track stage-specific performance
            print(f"  Stage {stage} best: {self.best_accuracy:.2%}")

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best accuracy: {self.best_accuracy:.2%}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Checkpoint: checkpoints/v2_best_model.pt")

        return self.best_accuracy

    def _save_checkpoint(self, path: str):
        """Save model checkpoint."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_accuracy': self.best_accuracy,
            'best_epoch': self.best_epoch
        }, path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--samples', type=int, default=50000)
    parser.add_argument('--target', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    trainer = SynthesisTrainer(config)
    accuracy = trainer.train(
        num_samples=args.samples,
        target_accuracy=args.target
    )

    print(f"\nFinal accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
