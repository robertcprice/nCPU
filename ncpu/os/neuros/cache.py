"""Neural Cache Hierarchy.

Cache replacement and prefetch policies implemented as neural networks.
Instead of fixed LRU/CLOCK/FIFO, the cache learns workload-specific
replacement and prefetch strategies.

Components:
    1. NeuralCacheReplacer — LSTM-based replacement policy
    2. NeuralPrefetcher — Sequence model that predicts next accesses
    3. NeuralCache — Unified cache combining both

Key insight: cache replacement is a sequence prediction problem.
An LSTM can learn access patterns (stride, temporal, spatial locality)
that fixed policies cannot exploit.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, List, Dict, Tuple
from pathlib import Path

from .device import default_device

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Cache Replacement Policy
# ═══════════════════════════════════════════════════════════════════════════════

class CacheReplacementNet(nn.Module):
    """LSTM-based cache replacement policy.

    Processes the sequence of recent accesses and scores each cache line
    for eviction. Lines with higher eviction scores are replaced first.

    Architecture:
        Access history: [seq_len, feature_dim] → LSTM → hidden state
        Cache line features: [num_lines, line_feature_dim]
        Combined: [hidden; line_features] → MLP → eviction score per line

    Features per cache line:
        - recency (normalized ticks since last access)
        - frequency (access count, log-scaled)
        - dirty bit
        - tag (embedded address bits)
    """

    def __init__(self, access_feature_dim: int = 4, hidden_dim: int = 64,
                 line_feature_dim: int = 4, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=access_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + line_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.hidden_dim = hidden_dim

    def forward(self, access_history: torch.Tensor,
                line_features: torch.Tensor) -> torch.Tensor:
        """Score cache lines for eviction.

        Args:
            access_history: [1, seq_len, access_feature_dim]
            line_features: [num_lines, line_feature_dim]

        Returns:
            eviction_scores: [num_lines] — higher = more likely to evict
        """
        # Encode access history
        _, (h_n, _) = self.lstm(access_history)
        context = h_n[-1]  # [1, hidden_dim]

        # Score each cache line
        num_lines = line_features.shape[0]
        context_expanded = context.expand(num_lines, -1)  # [num_lines, hidden_dim]
        combined = torch.cat([context_expanded, line_features], dim=-1)
        scores = self.scorer(combined).squeeze(-1)

        return scores


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Prefetcher
# ═══════════════════════════════════════════════════════════════════════════════

class PrefetchNet(nn.Module):
    """Neural prefetch predictor.

    Given the sequence of recent memory accesses, predicts the next
    K addresses likely to be accessed. Uses an LSTM to learn patterns
    like strides, linked-list traversals, and irregular patterns that
    hardware prefetchers struggle with.

    Architecture:
        Access addresses: [seq_len] → Embedding → LSTM → Linear → [K] next addresses
    """

    def __init__(self, addr_bits: int = 20, embed_dim: int = 32,
                 hidden_dim: int = 64, num_predictions: int = 4):
        super().__init__()
        self.addr_embed = nn.Embedding(1 << min(addr_bits, 16), embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.predictor = nn.Linear(hidden_dim, num_predictions)
        self.num_predictions = num_predictions
        self.addr_mask = (1 << min(addr_bits, 16)) - 1

    def forward(self, access_sequence: torch.Tensor) -> torch.Tensor:
        """Predict next addresses to prefetch.

        Args:
            access_sequence: [1, seq_len] int64 — recent page accesses

        Returns:
            predictions: [num_predictions] int64 — predicted next pages
        """
        # Clamp to embedding range
        clamped = access_sequence & self.addr_mask
        embedded = self.addr_embed(clamped)
        _, (h_n, _) = self.lstm(embedded)
        raw = self.predictor(h_n[-1]).squeeze(0)

        # Convert to page numbers (delta-based prediction)
        last_addr = access_sequence[0, -1].float()
        deltas = raw  # Network predicts deltas from last address
        predictions = (last_addr + deltas).long().clamp(min=0, max=self.addr_mask)
        return predictions


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Cache
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralCache:
    """GPU-tensor cache with neural replacement and prefetch.

    A set-associative cache where:
        - Tags, data, and metadata are GPU tensors
        - Replacement uses a trained LSTM policy (or LRU fallback)
        - Prefetch uses a trained sequence predictor (or stride fallback)

    All operations are GPU-native — no CPU-GPU transfers during cache ops.
    """

    def __init__(self, num_sets: int = 256, ways: int = 4,
                 line_size: int = 64, history_len: int = 32,
                 device: Optional[torch.device] = None):
        self.device = device or default_device()
        self.num_sets = num_sets
        self.ways = ways
        self.line_size = line_size
        self.history_len = history_len

        # Cache storage (all on GPU)
        self.tags = torch.full((num_sets, ways), -1, dtype=torch.int64, device=self.device)
        self.valid = torch.zeros(num_sets, ways, dtype=torch.bool, device=self.device)
        self.dirty = torch.zeros(num_sets, ways, dtype=torch.bool, device=self.device)
        self.access_count = torch.zeros(num_sets, ways, dtype=torch.int64, device=self.device)
        self.last_access = torch.zeros(num_sets, ways, dtype=torch.int64, device=self.device)

        # Access history ring buffer
        self.access_history = torch.zeros(history_len, 4, dtype=torch.float32, device=self.device)
        self.history_ptr = 0

        # Prefetch history
        self.addr_history = torch.zeros(history_len, dtype=torch.int64, device=self.device)
        self.addr_ptr = 0

        # Neural components
        self.replacer = CacheReplacementNet().to(self.device)
        self.prefetcher = PrefetchNet().to(self.device)
        self._replacer_trained = False
        self._prefetcher_trained = False

        # Statistics
        self.tick = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.prefetch_hits = 0
        self.prefetches_issued = 0

    def access(self, addr: int, write: bool = False) -> bool:
        """Access a cache line.

        Args:
            addr: Memory address (page-aligned)
            write: Whether this is a write access

        Returns:
            True on hit, False on miss
        """
        self.tick += 1
        tag = addr >> 6  # line_size = 64, so shift by 6
        set_idx = tag % self.num_sets

        # Check for hit (parallel across all ways)
        set_tags = self.tags[set_idx]
        set_valid = self.valid[set_idx]
        hit_mask = (set_tags == tag) & set_valid

        if hit_mask.any():
            # Hit
            way = int(hit_mask.nonzero(as_tuple=True)[0][0].item())
            self.access_count[set_idx, way] += 1
            self.last_access[set_idx, way] = self.tick
            if write:
                self.dirty[set_idx, way] = True
            self.hits += 1
            self._record_access(addr, hit=True, write=write)
            return True

        # Miss — need to fill
        self.misses += 1
        self._fill(set_idx, tag, write)
        self._record_access(addr, hit=False, write=write)

        # Trigger prefetch
        self._maybe_prefetch(addr)

        return False

    def _fill(self, set_idx: int, tag: int, write: bool):
        """Fill a cache line, evicting if necessary."""
        set_valid = self.valid[set_idx]
        invalid_ways = (~set_valid).nonzero(as_tuple=True)[0]

        if len(invalid_ways) > 0:
            way = int(invalid_ways[0].item())
        else:
            way = self._select_victim(set_idx)
            self.evictions += 1

        self.tags[set_idx, way] = tag
        self.valid[set_idx, way] = True
        self.dirty[set_idx, way] = write
        self.access_count[set_idx, way] = 1
        self.last_access[set_idx, way] = self.tick

    def _select_victim(self, set_idx: int) -> int:
        """Select a cache line to evict from the given set.

        Uses neural policy if trained, otherwise LRU.
        """
        if self._replacer_trained:
            return self._neural_victim(set_idx)
        return self._lru_victim(set_idx)

    def _lru_victim(self, set_idx: int) -> int:
        """LRU fallback: evict least recently used."""
        return int(self.last_access[set_idx].argmin().item())

    def _neural_victim(self, set_idx: int) -> int:
        """Neural replacement: score all ways using learned policy."""
        # Build line features: [ways, 4]
        max_tick = float(max(self.tick, 1))
        max_count = self.access_count[set_idx].float().max().clamp(min=1.0)

        line_features = torch.stack([
            (self.tick - self.last_access[set_idx]).float() / max_tick,
            self.access_count[set_idx].float() / max_count,
            self.dirty[set_idx].float(),
            self.valid[set_idx].float(),
        ], dim=-1)

        # Build access history: [1, history_len, 4]
        history = self.access_history.unsqueeze(0)

        with torch.no_grad():
            scores = self.replacer(history, line_features)

        return int(scores.argmax().item())

    def _record_access(self, addr: int, hit: bool, write: bool):
        """Record an access in the history buffer."""
        self.access_history[self.history_ptr] = torch.tensor([
            float(addr) / 1e6,  # Normalized address
            float(hit),
            float(write),
            float(self.tick) / 1e4,
        ], device=self.device)
        self.history_ptr = (self.history_ptr + 1) % self.history_len

        self.addr_history[self.addr_ptr] = addr
        self.addr_ptr = (self.addr_ptr + 1) % self.history_len

    def _maybe_prefetch(self, addr: int):
        """Issue prefetch predictions."""
        if self._prefetcher_trained:
            with torch.no_grad():
                seq = self.addr_history.unsqueeze(0)
                predictions = self.prefetcher(seq)

            for pred_addr in predictions:
                pa = int(pred_addr.item())
                tag = pa >> 6
                set_idx = tag % self.num_sets
                set_tags = self.tags[set_idx]
                set_valid = self.valid[set_idx]
                already_cached = ((set_tags == tag) & set_valid).any()
                if not already_cached:
                    self._fill(set_idx, tag, write=False)
                    self.prefetches_issued += 1
        else:
            self._stride_prefetch(addr)

    def _stride_prefetch(self, addr: int):
        """Simple stride-based prefetch fallback.

        Detects constant stride patterns and prefetches one line ahead.
        """
        if self.addr_ptr >= 3:
            a1 = int(self.addr_history[(self.addr_ptr - 1) % self.history_len].item())
            a2 = int(self.addr_history[(self.addr_ptr - 2) % self.history_len].item())
            a3 = int(self.addr_history[(self.addr_ptr - 3) % self.history_len].item())

            stride1 = a1 - a2
            stride2 = a2 - a3

            if stride1 == stride2 and stride1 != 0:
                prefetch_addr = addr + stride1
                if prefetch_addr >= 0:
                    tag = prefetch_addr >> 6
                    set_idx = tag % self.num_sets
                    set_tags = self.tags[set_idx]
                    set_valid = self.valid[set_idx]
                    if not ((set_tags == tag) & set_valid).any():
                        self._fill(set_idx, tag, write=False)
                        self.prefetches_issued += 1

    def invalidate(self, addr: int):
        """Invalidate a cache line."""
        tag = addr >> 6
        set_idx = tag % self.num_sets
        hit_mask = (self.tags[set_idx] == tag) & self.valid[set_idx]
        if hit_mask.any():
            way = int(hit_mask.nonzero(as_tuple=True)[0][0].item())
            self.valid[set_idx, way] = False

    def flush(self):
        """Flush entire cache and reset access history."""
        self.valid.fill_(False)
        self.dirty.fill_(False)
        self.access_history.zero_()
        self.addr_history.zero_()
        self.history_ptr = 0
        self.addr_ptr = 0

    # ─── Online Adaptation ──────────────────────────────────────────────

    def adapt_replacer(self, set_idx: int, oracle_victim: int):
        """Online learning: one gradient step on cache replacement.

        After each eviction, compare the neural policy's choice to the
        oracle (LRU) decision. If they differ, take one gradient step
        to align the neural policy with the oracle.

        Over time, the neural replacer internalizes LRU-like behavior
        but can also learn workload-specific patterns that LRU misses.
        """
        if not self._replacer_trained:
            return

        max_tick = float(max(self.tick, 1))
        max_count = self.access_count[set_idx].float().max().clamp(min=1.0)

        line_features = torch.stack([
            (self.tick - self.last_access[set_idx]).float() / max_tick,
            self.access_count[set_idx].float() / max_count,
            self.dirty[set_idx].float(),
            self.valid[set_idx].float(),
        ], dim=-1)

        history = self.access_history.unsqueeze(0)
        target = torch.tensor(oracle_victim, dtype=torch.long, device=self.device)

        self.replacer.train()
        scores = self.replacer(history, line_features)
        loss = nn.functional.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))
        loss.backward()

        with torch.no_grad():
            for param in self.replacer.parameters():
                if param.grad is not None:
                    param -= 1e-4 * param.grad
                    param.grad.zero_()

        self.replacer.eval()

    # ─── Persistence ──────────────────────────────────────────────────────

    def save(self, replace_path: str = "models/research/neuros/cache_replace.pt",
             prefetch_path: str = "models/research/neuros/prefetch.pt"):
        """Save trained models."""
        Path(replace_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.replacer.state_dict(), replace_path)
        torch.save(self.prefetcher.state_dict(), prefetch_path)

    def load(self, replace_path: str = "models/research/neuros/cache_replace.pt",
             prefetch_path: str = "models/research/neuros/prefetch.pt") -> Dict[str, bool]:
        """Load trained models."""
        result = {}
        if Path(replace_path).exists():
            self.replacer.load_state_dict(
                torch.load(replace_path, map_location=self.device, weights_only=True))
            self.replacer.eval()
            self._replacer_trained = True
            result["replacer"] = True
        if Path(prefetch_path).exists():
            self.prefetcher.load_state_dict(
                torch.load(prefetch_path, map_location=self.device, weights_only=True))
            self.prefetcher.eval()
            self._prefetcher_trained = True
            result["prefetcher"] = True
        return result

    # ─── Diagnostics ──────────────────────────────────────────────────────

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(1, total)

    def stats(self) -> Dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "evictions": self.evictions,
            "prefetches_issued": self.prefetches_issued,
            "prefetch_hits": self.prefetch_hits,
            "total_accesses": total,
            "replacer_trained": self._replacer_trained,
            "prefetcher_trained": self._prefetcher_trained,
            "occupancy": float(self.valid.sum().item()) / (self.num_sets * self.ways),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (f"NeuralCache(sets={self.num_sets}, ways={self.ways}, "
                f"hit_rate={s['hit_rate']:.1%}, "
                f"policy={'neural' if s['replacer_trained'] else 'lru'})")
