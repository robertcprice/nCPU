"""Neural Translation Lookaside Buffer (TLB).

A learned caching policy for the Neural MMU. Instead of hardware-fixed
LRU/FIFO replacement, the TLB uses a small neural network to predict
which entries to evict based on access patterns.

Architecture:
    - Fully-associative GPU-tensor TLB (all entries checked in parallel)
    - Neural eviction policy: LSTM over access history → eviction scores
    - All state on GPU — lookups are batched tensor operations

Key insight: traditional TLBs use fixed replacement (LRU, random).
A neural TLB can learn workload-specific caching patterns — e.g.,
keeping hot code pages while evicting rarely-used data pages.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Tuple, Dict
from pathlib import Path

from .device import default_device
from .mmu import PAGE_SIZE, PAGE_OFFSET_BITS, NUM_PERM_BITS

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Eviction Policy
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralEvictionPolicy(nn.Module):
    """Learns which TLB entry to evict based on access history.

    Input features per entry:
        - access_count: how many times this entry has been accessed
        - recency: cycles since last access (normalized)
        - is_dirty: whether the page has been written
        - is_code: whether the page is executable
        - entry_age: how long the entry has been in the TLB

    Output: eviction score (higher = more likely to evict)
    """

    def __init__(self, feature_dim: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [num_entries, feature_dim] → [num_entries, 1] eviction scores"""
        return self.net(features)


# ═══════════════════════════════════════════════════════════════════════════════
# Neural TLB
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralTLB:
    """GPU-tensor TLB with neural eviction policy.

    Fully-associative: all entries are checked in parallel via tensor ops.
    On a miss, the neural eviction policy selects which entry to replace.
    Falls back to LRU-like eviction if the policy isn't trained.

    All state tensors live on GPU:
        - vpn_tags:    [size] int64   — virtual page numbers (-1 = empty)
        - asid_tags:   [size] int64   — address space IDs
        - pfn_data:    [size] int64   — physical frame numbers
        - perm_data:   [size, 6] float — permission bits
        - access_cnt:  [size] int64   — access counters
        - last_access: [size] int64   — tick of last access
        - entry_age:   [size] int64   — tick when entry was loaded
    """

    def __init__(self, size: int = 64, device: Optional[torch.device] = None):
        self.size = size
        self.device = device or default_device()

        # Tag + data arrays (fully-associative)
        self.vpn_tags = torch.full((size,), -1, dtype=torch.int64, device=self.device)
        self.asid_tags = torch.zeros(size, dtype=torch.int64, device=self.device)
        self.pfn_data = torch.zeros(size, dtype=torch.int64, device=self.device)
        self.perm_data = torch.zeros(size, NUM_PERM_BITS, dtype=torch.float32, device=self.device)

        # Access tracking (for eviction policy features)
        self.access_cnt = torch.zeros(size, dtype=torch.int64, device=self.device)
        self.last_access = torch.zeros(size, dtype=torch.int64, device=self.device)
        self.entry_age = torch.zeros(size, dtype=torch.int64, device=self.device)

        # Neural eviction policy
        self.eviction_policy = NeuralEvictionPolicy().to(self.device)
        self._policy_trained = False

        # Global tick counter
        self.tick = 0

        # Statistics
        self.hits = 0
        self.misses = 0

    def lookup(self, vpn: int, asid: int = 0) -> Tuple[int, Optional[torch.Tensor]]:
        """Look up a VPN in the TLB.

        Args:
            vpn: Virtual page number
            asid: Address space ID

        Returns:
            (pfn, perms) on hit, (-1, None) on miss
        """
        self.tick += 1

        # Parallel match: find entry where vpn AND asid match
        match = (self.vpn_tags == vpn) & (self.asid_tags == asid)
        indices = match.nonzero(as_tuple=True)[0]

        if len(indices) == 0:
            self.misses += 1
            return -1, None

        idx = int(indices[0].item())
        self.access_cnt[idx] += 1
        self.last_access[idx] = self.tick
        self.hits += 1

        return int(self.pfn_data[idx].item()), self.perm_data[idx]

    def lookup_batch(self, vpns: torch.Tensor, asid: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch TLB lookup.

        Args:
            vpns: [batch] int64 — virtual page numbers
            asid: Address space ID

        Returns:
            pfns:  [batch] int64 — physical frame numbers (-1 on miss)
            valid: [batch] bool — whether each lookup hit
        """
        self.tick += 1
        batch = vpns.shape[0]

        # [batch, 1] == [1, size] → [batch, size] match matrix
        vpn_match = vpns.unsqueeze(1) == self.vpn_tags.unsqueeze(0)
        asid_match = self.asid_tags.unsqueeze(0) == asid
        match = vpn_match & asid_match

        # Any match per row
        valid = match.any(dim=1)
        # Get the first matching index per row
        match_idx = match.float().argmax(dim=1)

        pfns = torch.where(valid, self.pfn_data[match_idx], torch.tensor(-1, device=self.device))

        hit_count = int(valid.sum().item())
        self.hits += hit_count
        self.misses += batch - hit_count

        return pfns, valid

    def insert(self, vpn: int, asid: int, pfn: int, perms: torch.Tensor):
        """Insert an entry into the TLB, evicting if necessary.

        Args:
            vpn: Virtual page number
            asid: Address space ID
            pfn: Physical frame number
            perms: [6] float — permission bits
        """
        self.tick += 1

        # Check if already present (update in place)
        match = (self.vpn_tags == vpn) & (self.asid_tags == asid)
        existing = match.nonzero(as_tuple=True)[0]
        if len(existing) > 0:
            idx = int(existing[0].item())
            self.pfn_data[idx] = pfn
            self.perm_data[idx] = perms
            self.last_access[idx] = self.tick
            return

        # Find empty slot
        empty = (self.vpn_tags == -1).nonzero(as_tuple=True)[0]
        if len(empty) > 0:
            idx = int(empty[0].item())
        else:
            # Evict using neural policy or LRU fallback
            idx = self._select_eviction()

        self.vpn_tags[idx] = vpn
        self.asid_tags[idx] = asid
        self.pfn_data[idx] = pfn
        self.perm_data[idx] = perms
        self.access_cnt[idx] = 0
        self.last_access[idx] = self.tick
        self.entry_age[idx] = self.tick

    def invalidate(self, vpn: int, asid: int = 0):
        """Invalidate a specific TLB entry."""
        match = (self.vpn_tags == vpn) & (self.asid_tags == asid)
        self.vpn_tags[match] = -1

    def flush(self, asid: Optional[int] = None):
        """Flush TLB entries. If asid given, flush only that ASID."""
        if asid is None:
            self.vpn_tags.fill_(-1)
            self.access_cnt.zero_()
        else:
            match = self.asid_tags == asid
            self.vpn_tags[match] = -1

    def _select_eviction(self) -> int:
        """Select a TLB entry to evict.

        Uses the neural eviction policy if trained, otherwise falls back
        to evicting the least-recently-used entry.
        """
        if self._policy_trained:
            return self._neural_evict()
        return self._lru_evict()

    def _lru_evict(self) -> int:
        """LRU fallback: evict the entry with the oldest last_access."""
        return int(self.last_access.argmin().item())

    def _neural_evict(self) -> int:
        """Neural eviction: score all entries using learned policy."""
        features = self._build_features()
        with torch.no_grad():
            scores = self.eviction_policy(features)
        return int(scores.argmax().item())

    def _build_features(self) -> torch.Tensor:
        """Build feature vectors for all TLB entries.

        Returns: [size, 5] float tensor
        """
        max_access = self.access_cnt.float().max().clamp(min=1.0)
        max_recency = float(max(self.tick, 1))

        features = torch.stack([
            self.access_cnt.float() / max_access,                        # normalized access count
            (self.tick - self.last_access).float() / max_recency,        # normalized recency (higher = older)
            self.perm_data[:, 4],                                        # dirty bit
            self.perm_data[:, 3],                                        # executable bit
            (self.tick - self.entry_age).float() / max_recency,          # normalized entry age
        ], dim=-1)
        return features

    # ─── Training ─────────────────────────────────────────────────────────

    def train_policy(self, access_trace: torch.Tensor, optimal_evictions: torch.Tensor,
                     epochs: int = 50, lr: float = 1e-3) -> Dict:
        """Train the eviction policy from an optimal trace.

        Args:
            access_trace: [T, 5] float — feature snapshots at each eviction point
            optimal_evictions: [T] int64 — index of optimal entry to evict

        Returns:
            Training statistics
        """
        optimizer = torch.optim.Adam(self.eviction_policy.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        best_acc = 0.0
        for epoch in range(epochs):
            self.eviction_policy.train()
            optimizer.zero_grad()

            # Reshape: treat each eviction decision as a classification
            scores = self.eviction_policy(access_trace).squeeze(-1)
            loss = loss_fn(scores.unsqueeze(0), optimal_evictions.unsqueeze(0))

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = scores.argmax()
                acc = (pred == optimal_evictions).float().mean().item()
                best_acc = max(best_acc, acc)

        self.eviction_policy.eval()
        self._policy_trained = True

        return {"epochs": epochs, "best_accuracy": best_acc}

    # ─── Online Adaptation ──────────────────────────────────────────────

    def adapt(self, oracle_victim: int):
        """Online learning: one gradient step on TLB eviction policy.

        After each eviction, compare neural policy to the LRU oracle.
        Take one gradient step to align. Over time, the neural policy
        learns workload-specific eviction patterns.
        """
        if not self._policy_trained:
            return

        features = self._build_features()
        target = torch.tensor(oracle_victim, dtype=torch.long, device=self.device)

        self.eviction_policy.train()
        scores = self.eviction_policy(features).squeeze(-1)
        loss = nn.functional.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))
        loss.backward()

        with torch.no_grad():
            for param in self.eviction_policy.parameters():
                if param.grad is not None:
                    param -= 1e-4 * param.grad
                    param.grad.zero_()

        self.eviction_policy.eval()

    # ─── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str = "models/research/neuros/tlb.pt"):
        """Save the trained eviction policy."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.eviction_policy.state_dict(), path)

    def load(self, path: str = "models/research/neuros/tlb.pt") -> bool:
        """Load a trained eviction policy."""
        p = Path(path)
        if not p.exists():
            return False
        self.eviction_policy.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.eviction_policy.eval()
        self._policy_trained = True
        return True

    # ─── Diagnostics ──────────────────────────────────────────────────────

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(1, total)

    @property
    def occupancy(self) -> float:
        return float((self.vpn_tags >= 0).sum().item()) / self.size

    def stats(self) -> Dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "occupancy": self.occupancy,
            "policy_trained": self._policy_trained,
            "size": self.size,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (f"NeuralTLB(size={s['size']}, "
                f"hit_rate={s['hit_rate']:.1%}, "
                f"occupancy={s['occupancy']:.1%}, "
                f"policy={'neural' if s['policy_trained'] else 'lru'})")
