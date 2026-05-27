"""Differentiable OS: scheduling, caching, and memory decisions via gradient descent.

Connects the differentiable execution engine to OS-level neural models,
enabling end-to-end gradient flow from OS decisions back to program execution.

The key insight: traditional OS policies are heuristics (LRU, round-robin, first-fit).
A differentiable OS can LEARN optimal policies via gradient descent on actual
workload traces.

Components:
    DifferentiableScheduler — soft attention over process queue (replaces round-robin)
    DifferentiableCache — soft eviction via learned scoring (replaces LRU)
    DifferentiableAllocator — soft placement via learned spatial preference (replaces first-fit)
    DifferentiableOS — unified OS that optimizes all three jointly
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Workload Event Representation ─────────────────────────────────────────

@dataclass
class WorkloadEvent:
    """A single OS event in a workload trace."""
    process_states: torch.Tensor     # (n_processes, feature_dim) process features
    memory_access: torch.Tensor      # (feature_dim,) memory access descriptor
    alloc_request: torch.Tensor      # (feature_dim,) allocation request
    # Ground truth for loss computation
    optimal_process: Optional[int] = None    # which process should run
    optimal_eviction: Optional[int] = None   # which cache line to evict
    optimal_placement: Optional[int] = None  # where to allocate
    is_cache_hit: Optional[bool] = None      # whether the access was a cache hit


# ── Differentiable Scheduler ──────────────────────────────────────────────

class DifferentiableScheduler(nn.Module):
    """Learns process scheduling via soft attention over the process queue.

    Traditional: round-robin assigns equal time slices to all processes.
    Differentiable: learns to prioritize processes based on their state
    (CPU usage, wait time, priority, memory footprint) to minimize total
    completion time.

    Architecture:
        process_features -> Linear -> Attention scores -> softmax -> weighted selection
    """

    def __init__(self, feature_dim: int = 6, hidden_dim: int = 32):
        super().__init__()
        self.feature_dim = feature_dim

        # Self-attention over process features
        self.query_proj = nn.Linear(feature_dim, hidden_dim)
        self.key_proj = nn.Linear(feature_dim, hidden_dim)
        self.value_proj = nn.Linear(feature_dim, hidden_dim)

        # Output scoring
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, process_states: torch.Tensor,
                temperature: float = 1.0) -> torch.Tensor:
        """Score processes and return soft scheduling weights.

        Args:
            process_states: (n_processes, feature_dim) per-process features
                Features: [priority, cpu_time, wait_time, remaining_work,
                           memory_pages, is_interactive]
            temperature: softmax temperature (lower = more decisive)

        Returns:
            (n_processes,) soft scheduling weights summing to 1
        """
        Q = self.query_proj(process_states)
        K = self.key_proj(process_states)
        V = self.value_proj(process_states)

        # Self-attention
        d_k = Q.shape[-1]
        attn = (Q @ K.T) / (d_k ** 0.5)
        attn_weights = F.softmax(attn, dim=-1)
        context = attn_weights @ V

        # Score each process
        scores = self.scorer(context).squeeze(-1)  # (n_processes,)
        return F.softmax(scores / temperature, dim=0)


# ── Differentiable Cache ──────────────────────────────────────────────────

class DifferentiableCache(nn.Module):
    """Learns cache eviction policy via soft scoring.

    Traditional: LRU evicts the least recently used line.
    Differentiable: learns to score cache lines for eviction based on
    access history and line metadata, minimizing miss rate.

    Architecture:
        Access history LSTM -> hidden state
        Line features -> MLP -> eviction scores
        Soft eviction via Gumbel-softmax over scores
    """

    def __init__(self, cache_size: int = 8, access_feature_dim: int = 4,
                 line_feature_dim: int = 4, hidden_dim: int = 32,
                 history_len: int = 32):
        super().__init__()
        self.cache_size = cache_size
        self.history_len = history_len

        # Access history encoder
        self.history_lstm = nn.LSTM(
            input_size=access_feature_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Per-line scoring
        self.line_scorer = nn.Sequential(
            nn.Linear(hidden_dim + line_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # State tracking
        self.register_buffer(
            "access_history",
            torch.zeros(history_len, access_feature_dim),
        )
        self.register_buffer("history_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("cache_tags", torch.full((cache_size,), -1, dtype=torch.long))
        self.register_buffer("cache_recency", torch.zeros(cache_size))
        self.register_buffer("cache_frequency", torch.zeros(cache_size))
        self.register_buffer("cache_valid", torch.zeros(cache_size))
        self.register_buffer("tick", torch.tensor(0, dtype=torch.long))
        self.register_buffer("hits", torch.tensor(0, dtype=torch.long))
        self.register_buffer("misses", torch.tensor(0, dtype=torch.long))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reset(self):
        """Reset cache state for a new workload."""
        self.access_history.zero_()
        self.history_ptr.zero_()
        self.cache_tags.fill_(-1)
        self.cache_recency.zero_()
        self.cache_frequency.zero_()
        self.cache_valid.zero_()
        self.tick.zero_()
        self.hits.zero_()
        self.misses.zero_()

    def _record_access(self, features: torch.Tensor):
        """Record an access in the history buffer."""
        ptr = int(self.history_ptr.item())
        self.access_history[ptr] = features.detach()
        self.history_ptr = (self.history_ptr + 1) % self.history_len
        self.tick += 1

    def _get_line_features(self) -> torch.Tensor:
        """Build per-line features: [recency, frequency, valid, age]."""
        tick_val = max(float(self.tick.item()), 1.0)
        recency = (tick_val - self.cache_recency) / tick_val
        freq_log = torch.log1p(self.cache_frequency)
        max_freq = freq_log.max().clamp(min=1.0)
        frequency = freq_log / max_freq
        valid = self.cache_valid
        age = recency  # Normalized age
        return torch.stack([recency, frequency, valid, age], dim=-1)

    def forward(self, access_features: torch.Tensor,
                address: int = 0,
                temperature: float = 1.0) -> Tuple[torch.Tensor, bool]:
        """Process a memory access through the differentiable cache.

        Returns (eviction_weights, is_hit).
        eviction_weights: (cache_size,) soft eviction scores (for training)
        is_hit: whether the access was a cache hit
        """
        self._record_access(access_features)

        # Check for hit
        is_hit = False
        for i in range(self.cache_size):
            if self.cache_valid[i] > 0.5 and self.cache_tags[i] == address:
                is_hit = True
                self.cache_recency[i] = float(self.tick.item())
                self.cache_frequency[i] += 1
                self.hits += 1
                break

        if is_hit:
            # Return uniform weights (no eviction needed)
            return torch.zeros(self.cache_size, device=access_features.device), True

        self.misses += 1

        # Check for empty slot
        for i in range(self.cache_size):
            if self.cache_valid[i] < 0.5:
                self.cache_tags[i] = address
                self.cache_valid[i] = 1.0
                self.cache_recency[i] = float(self.tick.item())
                self.cache_frequency[i] = 1.0
                return torch.zeros(self.cache_size, device=access_features.device), False

        # Need eviction: score all lines
        history = self.access_history.unsqueeze(0)  # (1, history_len, feat_dim)
        _, (h, _) = self.history_lstm(history)
        context = h.squeeze(0).squeeze(0)  # (hidden_dim,)

        line_features = self._get_line_features()  # (cache_size, line_feat_dim)

        # Concatenate context with each line's features
        context_expanded = context.unsqueeze(0).expand(self.cache_size, -1)
        combined = torch.cat([context_expanded, line_features], dim=-1)

        scores = self.line_scorer(combined).squeeze(-1)  # (cache_size,)
        eviction_weights = F.softmax(scores / temperature, dim=0)

        # Hard eviction for state update
        victim = int(scores.argmax().item())
        self.cache_tags[victim] = address
        self.cache_recency[victim] = float(self.tick.item())
        self.cache_frequency[victim] = 1.0

        return eviction_weights, False

    @property
    def hit_rate(self) -> float:
        total = float((self.hits + self.misses).item())
        return float(self.hits.item()) / max(total, 1.0)


# ── Differentiable Memory Allocator ───────────────────────────────────────

class DifferentiableAllocator(nn.Module):
    """Learns memory placement via soft spatial preference scoring.

    Traditional: first-fit scans linearly for the first available block.
    Differentiable: learns to score memory regions and choose placements
    that minimize fragmentation based on workload patterns.

    Architecture:
        Request features + Memory state -> MLP -> placement scores
        Soft placement via softmax over valid regions
    """

    def __init__(self, memory_size: int = 256, n_regions: int = 16,
                 request_feature_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.memory_size = memory_size
        self.n_regions = n_regions
        self.region_size = memory_size // n_regions

        # Region state features: [utilization, largest_free, fragmentation, age]
        region_feature_dim = 4

        self.scorer = nn.Sequential(
            nn.Linear(request_feature_dim + region_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Track region state
        self.register_buffer("region_used", torch.zeros(n_regions))
        self.register_buffer("region_alloc_count", torch.zeros(n_regions))
        self.register_buffer("region_free_count", torch.zeros(n_regions))
        self.register_buffer("total_allocs", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_frees", torch.tensor(0, dtype=torch.long))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reset(self):
        """Reset allocator state."""
        self.region_used.zero_()
        self.region_alloc_count.zero_()
        self.region_free_count.zero_()
        self.total_allocs.zero_()
        self.total_frees.zero_()

    def _get_region_features(self) -> torch.Tensor:
        """Build per-region features."""
        utilization = self.region_used / self.region_size
        total_ops = (self.region_alloc_count + self.region_free_count).clamp(min=1)
        fragmentation = self.region_free_count / total_ops
        largest_free = 1.0 - utilization  # Simplified
        age = self.region_alloc_count / max(float(self.total_allocs.item()), 1.0)
        return torch.stack([utilization, largest_free, fragmentation, age], dim=-1)

    def forward(self, request_features: torch.Tensor,
                size: int = 1,
                temperature: float = 1.0) -> torch.Tensor:
        """Score memory regions for allocation placement.

        Args:
            request_features: (request_feature_dim,) allocation request
                Features: [size_normalized, alignment, urgency, lifetime_hint]
            size: number of units to allocate
            temperature: softmax temperature

        Returns:
            (n_regions,) soft placement weights
        """
        region_features = self._get_region_features()  # (n_regions, 4)
        request_expanded = request_features.unsqueeze(0).expand(self.n_regions, -1)
        combined = torch.cat([request_expanded, region_features], dim=-1)

        scores = self.scorer(combined).squeeze(-1)  # (n_regions,)

        # Mask out full regions
        capacity_mask = (self.region_used + size <= self.region_size).float()
        scores = scores + (1.0 - capacity_mask) * (-1e9)

        placement_weights = F.softmax(scores / temperature, dim=0)

        # Hard allocation for state update
        region = int(scores.argmax().item())
        self.region_used[region] += size
        self.region_alloc_count[region] += 1
        self.total_allocs += 1

        return placement_weights

    def free(self, region: int, size: int = 1):
        """Free memory in a region."""
        self.region_used[region] = max(0, self.region_used[region] - size)
        self.region_free_count[region] += 1
        self.total_frees += 1

    @property
    def fragmentation(self) -> float:
        """Compute external fragmentation metric."""
        total_free = float((self.region_size - self.region_used).sum().item())
        if total_free <= 0:
            return 1.0
        max_free_region = float((self.region_size - self.region_used).max().item())
        return 1.0 - (max_free_region / max(total_free, 1.0))


# ── Differentiable OS (unified) ──────────────────────────────────────────

class DifferentiableOS(nn.Module):
    """An OS where scheduling, caching, and memory decisions are differentiable.

    Enables gradient-based optimization of OS policies:
    - Cache replacement: learn eviction policy that minimizes miss rate
    - Process scheduling: learn scheduling that minimizes total completion time
    - Memory allocation: learn placement that minimizes fragmentation

    The key insight: traditional OS policies are heuristics (LRU, round-robin).
    A differentiable OS can LEARN optimal policies via gradient descent.
    """

    def __init__(self, n_processes: int = 4, cache_size: int = 8,
                 memory_size: int = 256, process_feature_dim: int = 6):
        super().__init__()
        self.scheduler = DifferentiableScheduler(feature_dim=process_feature_dim)
        self.cache = DifferentiableCache(cache_size=cache_size)
        self.allocator = DifferentiableAllocator(memory_size=memory_size)
        self.n_processes = n_processes

    def reset(self):
        """Reset all OS component states."""
        self.cache.reset()
        self.allocator.reset()

    def step(self, event: WorkloadEvent,
             temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute one OS step with full gradient flow.

        Returns:
            (schedule_weights, eviction_weights, placement_weights)
        """
        # Schedule: soft attention over processes
        schedule_weights = self.scheduler(event.process_states, temperature)

        # Cache: soft eviction decision
        cache_features = event.memory_access
        address = int(cache_features[0].item() * 1000) % 1000
        eviction_weights, _ = self.cache(cache_features, address, temperature)

        # Allocate: soft placement decision
        placement_weights = self.allocator(event.alloc_request, temperature=temperature)

        return schedule_weights, eviction_weights, placement_weights

    def compute_loss(self, schedule_weights: torch.Tensor,
                     eviction_weights: torch.Tensor,
                     event: WorkloadEvent) -> torch.Tensor:
        """Compute combined loss for all OS decisions."""
        loss = torch.tensor(0.0, device=schedule_weights.device)

        # Scheduling loss: cross-entropy against optimal choice
        if event.optimal_process is not None:
            target = torch.tensor([event.optimal_process],
                                  device=schedule_weights.device)
            loss = loss + F.cross_entropy(
                schedule_weights.unsqueeze(0), target
            )

        # Cache loss: penalize misses (eviction of soon-to-be-accessed lines)
        if event.is_cache_hit is not None and not event.is_cache_hit:
            # We want the model to learn good eviction - encourage high entropy
            # on eviction weights (explore), with specific supervision when available
            if event.optimal_eviction is not None and eviction_weights.sum() > 0:
                target = torch.tensor([event.optimal_eviction],
                                      device=eviction_weights.device)
                loss = loss + F.cross_entropy(
                    eviction_weights.unsqueeze(0), target
                )

        return loss

    def optimize(self, workload_trace: List[WorkloadEvent],
                 n_epochs: int = 200, lr: float = 1e-3,
                 temperature_start: float = 2.0,
                 temperature_end: float = 0.1) -> Dict[str, List[float]]:
        """Learn optimal OS policies from a workload trace.

        Returns training metrics.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        metrics = {"loss": [], "cache_hit_rate": [], "fragmentation": []}

        for epoch in range(n_epochs):
            self.reset()
            total_loss = torch.tensor(0.0)
            temperature = temperature_start + (temperature_end - temperature_start) * (epoch / max(n_epochs - 1, 1))

            for event in workload_trace:
                schedule_w, eviction_w, placement_w = self.step(event, temperature)
                loss = self.compute_loss(schedule_w, eviction_w, event)
                total_loss = total_loss + loss

            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            metrics["loss"].append(total_loss.item())
            metrics["cache_hit_rate"].append(self.cache.hit_rate)
            metrics["fragmentation"].append(self.allocator.fragmentation)

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:4d}/{n_epochs}: "
                      f"loss={total_loss.item():.4f}, "
                      f"cache_hr={self.cache.hit_rate:.1%}, "
                      f"frag={self.allocator.fragmentation:.3f}, "
                      f"temp={temperature:.2f}")

        return metrics


# ── Baseline OS Policies (for comparison) ─────────────────────────────────

class RoundRobinScheduler:
    """Round-robin baseline scheduler."""
    def __init__(self, n_processes: int):
        self.n = n_processes
        self.current = 0

    def schedule(self, process_states: torch.Tensor) -> int:
        n_ready = process_states.shape[0]
        if n_ready == 0:
            return 0
        choice = self.current % n_ready
        self.current += 1
        return choice


class LRUCache:
    """LRU baseline cache."""
    def __init__(self, size: int):
        self.size = size
        self.entries: Dict[int, int] = {}  # tag -> tick
        self.tick = 0
        self.hits = 0
        self.misses = 0

    def reset(self):
        self.entries.clear()
        self.tick = 0
        self.hits = 0
        self.misses = 0

    def access(self, address: int) -> bool:
        self.tick += 1
        if address in self.entries:
            self.entries[address] = self.tick
            self.hits += 1
            return True
        self.misses += 1
        if len(self.entries) >= self.size:
            # Evict LRU
            lru_addr = min(self.entries, key=self.entries.get)
            del self.entries[lru_addr]
        self.entries[address] = self.tick
        return False

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(total, 1)


class FirstFitAllocator:
    """First-fit baseline allocator."""
    def __init__(self, memory_size: int, n_regions: int = 16):
        self.region_size = memory_size // n_regions
        self.n_regions = n_regions
        self.region_used = [0] * n_regions

    def reset(self):
        self.region_used = [0] * self.n_regions

    def allocate(self, size: int = 1) -> int:
        for i in range(self.n_regions):
            if self.region_used[i] + size <= self.region_size:
                self.region_used[i] += size
                return i
        return 0  # Fallback

    def free(self, region: int, size: int = 1):
        self.region_used[region] = max(0, self.region_used[region] - size)

    @property
    def fragmentation(self) -> float:
        free_per_region = [self.region_size - u for u in self.region_used]
        total_free = sum(free_per_region)
        if total_free <= 0:
            return 1.0
        max_free = max(free_per_region)
        return 1.0 - max_free / max(total_free, 1)


# ── Synthetic Workload Generator ──────────────────────────────────────────

def generate_workload(n_events: int = 200, n_processes: int = 4,
                      locality: float = 0.7) -> List[WorkloadEvent]:
    """Generate a synthetic workload trace for training.

    The workload has temporal locality (probability of re-accessing recent
    addresses) and bursty allocation patterns.
    """
    import random
    events = []
    recent_addresses = []

    for t in range(n_events):
        # Process states: vary over time
        process_states = torch.rand(n_processes, 6)
        # Make one process "best" (highest remaining work, low CPU time)
        best = t % n_processes
        process_states[best, 3] = 1.0  # remaining_work
        process_states[best, 1] = 0.1  # low cpu_time

        # Memory access with temporal locality
        if recent_addresses and random.random() < locality:
            address = random.choice(recent_addresses[-8:])
        else:
            address = random.randint(0, 999)
        recent_addresses.append(address)
        if len(recent_addresses) > 50:
            recent_addresses = recent_addresses[-50:]

        memory_access = torch.tensor([
            address / 1000.0,  # normalized address
            1.0 if random.random() < 0.7 else 0.0,  # is_read
            len(recent_addresses) / 50.0,  # history depth
            float(t) / n_events,  # progress
        ])

        # Allocation request
        alloc_request = torch.tensor([
            random.uniform(0.05, 0.3),  # size_normalized
            1.0 if random.random() < 0.5 else 0.0,  # alignment
            random.uniform(0.0, 1.0),  # urgency
            random.uniform(0.1, 1.0),  # lifetime_hint
        ])

        event = WorkloadEvent(
            process_states=process_states,
            memory_access=memory_access,
            alloc_request=alloc_request,
            optimal_process=best,
            is_cache_hit=None,  # Will be determined by the cache
        )
        events.append(event)

    return events


def evaluate_baseline(workload: List[WorkloadEvent],
                      n_processes: int = 4,
                      cache_size: int = 8) -> Dict[str, float]:
    """Run baseline policies on the workload for comparison."""
    rr = RoundRobinScheduler(n_processes)
    lru = LRUCache(cache_size)
    ff = FirstFitAllocator(256)

    lru.reset()
    ff.reset()

    schedule_correct = 0
    total = 0

    for event in workload:
        # Scheduling
        chosen = rr.schedule(event.process_states)
        if event.optimal_process is not None and chosen == event.optimal_process:
            schedule_correct += 1
        total += 1

        # Cache
        address = int(event.memory_access[0].item() * 1000) % 1000
        lru.access(address)

        # Allocation
        ff.allocate(1)
        if total % 5 == 0:
            ff.free(total % ff.n_regions, 1)

    return {
        "schedule_accuracy": schedule_correct / max(total, 1),
        "cache_hit_rate": lru.hit_rate,
        "fragmentation": ff.fragmentation,
    }
