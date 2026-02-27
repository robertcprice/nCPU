#!/usr/bin/env python3
"""
NeuralPrefetcher - Learned Memory Access Prediction
====================================================

Predicts which memory addresses will be accessed next based on
observed access patterns. Uses sequence modeling (LSTM/Transformer)
to learn complex access patterns.

Traditional prefetchers use fixed patterns:
- Stride: Detect constant offset patterns
- Stream: Detect sequential access
- Markov: Simple state machine

NeuralPrefetcher learns YOUR program's actual access patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Optional, Dict
from collections import deque
import random


# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class AccessPatternModel(nn.Module):
    """
    Sequence model that predicts next memory accesses.

    Uses a transformer to model complex, long-range dependencies
    in memory access patterns.
    """

    def __init__(
        self,
        address_bits: int = 32,  # Address space size
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        context_length: int = 32,
        num_predictions: int = 4
    ):
        super().__init__()

        self.address_bits = address_bits
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.num_predictions = num_predictions

        # Address embedding using learned hash buckets
        self.num_buckets = 4096
        self.addr_embed = nn.Embedding(self.num_buckets, embed_dim)

        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, context_length, embed_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Delta prediction head (predict offsets from last address)
        self.delta_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_predictions * 2)  # Mean and logvar for each prediction
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_predictions),
            nn.Sigmoid()
        )

    def _hash_address(self, addr: int) -> int:
        """Hash address to bucket index."""
        # Simple multiplicative hash
        return (addr * 2654435761) % self.num_buckets

    def forward(
        self,
        address_history: torch.Tensor,  # [batch, context_length]
        return_confidence: bool = False
    ) -> torch.Tensor:
        """
        Predict next addresses given history.

        Args:
            address_history: Recent address sequence (hashed to buckets)
            return_confidence: Whether to return confidence scores

        Returns:
            delta_predictions: Predicted deltas from last address [batch, num_predictions]
        """
        batch_size = address_history.shape[0]

        # Embed addresses
        x = self.addr_embed(address_history)  # [batch, context, embed]

        # Add positional encoding
        x = x + self.pos_embed[:, :x.shape[1], :]

        # Transform
        x = self.transformer(x)  # [batch, context, embed]

        # Use last position for prediction
        last_hidden = x[:, -1, :]  # [batch, embed]

        # Predict deltas
        delta_params = self.delta_head(last_hidden)  # [batch, num_pred * 2]
        delta_params = delta_params.view(batch_size, self.num_predictions, 2)

        # Mean predictions
        deltas = delta_params[:, :, 0]  # [batch, num_pred]

        if return_confidence:
            confidence = self.confidence_head(last_hidden)
            return deltas, confidence

        return deltas


class NeuralPrefetcher:
    """
    Memory prefetcher with learned access prediction.

    Features:
    - Learns complex access patterns (loops, strides, irregular)
    - Predicts multiple future accesses
    - Online learning from actual access patterns
    - Confidence-based prefetch decisions
    """

    def __init__(
        self,
        context_length: int = 32,
        num_predictions: int = 4,
        learning_rate: float = 1e-4,
        enable_learning: bool = True,
        prefetch_threshold: float = 0.5
    ):
        """
        Initialize NeuralPrefetcher.

        Args:
            context_length: How many past accesses to consider
            num_predictions: How many future accesses to predict
            learning_rate: Learning rate for online updates
            enable_learning: Whether to learn from access patterns
            prefetch_threshold: Confidence threshold for prefetching
        """
        self.context_length = context_length
        self.num_predictions = num_predictions
        self.learning_enabled = enable_learning
        self.prefetch_threshold = prefetch_threshold

        # Model
        self.model = AccessPatternModel(
            context_length=context_length,
            num_predictions=num_predictions
        ).to(device)
        self.model.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Access history
        self.history: deque = deque(maxlen=context_length)
        self.full_history: deque = deque(maxlen=10000)

        # Learning buffer
        self.training_buffer: List[Tuple] = []
        self.batch_size = 32

        # Statistics
        self.predictions_made = 0
        self.correct_predictions = 0
        self.prefetches_issued = 0
        self.prefetch_hits = 0

        # Pending prefetches (to track accuracy)
        self.pending_prefetches: Dict[int, float] = {}

    def record_access(self, address: int):
        """
        Record a memory access.

        This updates the history and triggers learning.
        """
        # Check if this was a prefetched address
        if address in self.pending_prefetches:
            self.prefetch_hits += 1
            del self.pending_prefetches[address]

        # Add to history
        hashed = self._hash_address(address)
        self.history.append(hashed)
        self.full_history.append(address)

        # Generate training example from history
        if len(self.full_history) >= self.context_length + self.num_predictions:
            self._generate_training_example()

        # Learn periodically
        if self.learning_enabled and len(self.training_buffer) >= self.batch_size:
            self._learn()

    def _hash_address(self, addr: int) -> int:
        """Hash address to bucket."""
        return (addr * 2654435761) % 4096

    def predict_next(self, top_k: int = 4) -> List[Tuple[int, float]]:
        """
        Predict next memory accesses.

        Returns:
            List of (predicted_address, confidence) tuples
        """
        if len(self.history) < self.context_length:
            return []

        self.predictions_made += 1

        # Prepare input
        history_tensor = torch.tensor(
            list(self.history),
            dtype=torch.long,
            device=device
        ).unsqueeze(0)

        # Get predictions
        with torch.no_grad():
            deltas, confidence = self.model(history_tensor, return_confidence=True)

        # Convert deltas to addresses
        last_addr = self.full_history[-1] if self.full_history else 0

        predictions = []
        for i in range(min(top_k, self.num_predictions)):
            delta = int(deltas[0, i].item())
            conf = confidence[0, i].item()

            predicted_addr = last_addr + delta
            predictions.append((predicted_addr, conf))

        return predictions

    def get_prefetch_candidates(self) -> List[int]:
        """
        Get addresses that should be prefetched.

        Only returns addresses with confidence above threshold.
        """
        predictions = self.predict_next()

        candidates = []
        for addr, conf in predictions:
            if conf >= self.prefetch_threshold:
                candidates.append(addr)
                self.prefetches_issued += 1
                self.pending_prefetches[addr] = conf

        return candidates

    def _generate_training_example(self):
        """Generate a training example from history."""
        history = list(self.full_history)

        if len(history) < self.context_length + self.num_predictions:
            return

        # Context: last context_length addresses
        context = history[-(self.context_length + self.num_predictions):-self.num_predictions]
        context_hashed = [self._hash_address(a) for a in context]

        # Targets: next num_predictions addresses (as deltas)
        targets = history[-self.num_predictions:]
        last_context = context[-1]
        target_deltas = [t - last_context for t in targets]

        self.training_buffer.append((context_hashed, target_deltas))

    def _learn(self):
        """Learn from collected examples."""
        if len(self.training_buffer) < self.batch_size:
            return

        # Sample batch
        batch = random.sample(self.training_buffer, self.batch_size)
        self.training_buffer = self.training_buffer[-1000:]  # Keep recent

        # Prepare tensors
        contexts = torch.tensor(
            [b[0] for b in batch],
            dtype=torch.long,
            device=device
        )
        targets = torch.tensor(
            [b[1] for b in batch],
            dtype=torch.float32,
            device=device
        )

        # Forward pass
        self.model.train()

        predictions = self.model(contexts)
        loss = F.mse_loss(predictions, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.model.eval()

    def get_stats(self) -> Dict:
        """Get prefetcher statistics."""
        accuracy = (
            self.correct_predictions / max(1, self.predictions_made)
        )
        hit_rate = (
            self.prefetch_hits / max(1, self.prefetches_issued)
        )

        return {
            'predictions_made': self.predictions_made,
            'correct_predictions': self.correct_predictions,
            'prediction_accuracy': accuracy,
            'prefetches_issued': self.prefetches_issued,
            'prefetch_hits': self.prefetch_hits,
            'prefetch_hit_rate': hit_rate,
            'learning_enabled': self.learning_enabled
        }

    def save(self, path: str):
        """Save prefetcher model."""
        torch.save({
            'model_state': self.model.state_dict(),
            'stats': self.get_stats()
        }, path)

    def load(self, path: str):
        """Load prefetcher model."""
        data = torch.load(path, map_location=device, weights_only=False)
        self.model.load_state_dict(data['model_state'])


def benchmark_prefetcher():
    """Benchmark NeuralPrefetcher on synthetic patterns."""
    print("=" * 70)
    print("🧠 NEURAL PREFETCHER BENCHMARK")
    print("=" * 70)

    prefetcher = NeuralPrefetcher(enable_learning=True)

    # Generate access pattern with structure
    # Simulating a loop that accesses array elements
    print("\n📊 Generating structured access pattern...")

    addresses = []

    # Pattern 1: Sequential access (stride = 8)
    base = 0x1000
    for i in range(100):
        addresses.append(base + i * 8)

    # Pattern 2: Strided access (stride = 64)
    base = 0x2000
    for i in range(100):
        addresses.append(base + i * 64)

    # Pattern 3: Loop with inner stride
    base = 0x3000
    for outer in range(10):
        for inner in range(10):
            addresses.append(base + outer * 1024 + inner * 8)

    # Feed to prefetcher
    print(f"   Training on {len(addresses)} accesses...")

    for addr in addresses:
        prefetcher.record_access(addr)

    # Test predictions
    print("\n📈 Testing predictions...")

    # Continue the patterns and check predictions
    test_addrs = []
    base = 0x1000 + 100 * 8  # Continue sequential
    for i in range(20):
        test_addrs.append(base + i * 8)

    hits = 0
    for addr in test_addrs:
        predictions = prefetcher.predict_next()
        predicted_addrs = [p[0] for p in predictions]

        if addr in predicted_addrs:
            hits += 1

        prefetcher.record_access(addr)

    print(f"   Prediction accuracy: {hits}/{len(test_addrs)} = {hits/len(test_addrs)*100:.1f}%")

    stats = prefetcher.get_stats()
    print(f"\n   Total predictions: {stats['predictions_made']}")
    print(f"   Prefetch hit rate: {stats['prefetch_hit_rate']*100:.1f}%")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    benchmark_prefetcher()
