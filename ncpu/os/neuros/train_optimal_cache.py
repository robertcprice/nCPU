#!/usr/bin/env python3
"""Train neural cache replacement on Belady's optimal policy.

Belady's algorithm: evict the cache line that will be used furthest in the future.
This is provably optimal but requires oracle knowledge of future accesses.
We train the LSTM to approximate this by learning patterns from recorded traces.

Training data: generate realistic file access traces, compute Belady-optimal
decisions, train LSTM (same architecture as existing cache_replace.pt) on these traces.

The key insight: Belady's algorithm knows the future. The LSTM learns to predict
future reuse distance from access history patterns. A well-trained LSTM should
beat LRU by 5-15% on realistic workloads because it learns:
  - Temporal locality patterns (recently used items likely reused soon)
  - Frequency patterns (frequently accessed items should stay)
  - Sequential scan detection (large scans should NOT evict hot items)
  - Working set transitions (detect when the active set changes)

Architecture: CacheReplacementNet from cache.py
  - LSTM(input_size=4, hidden_size=64, num_layers=1)
  - Scorer: Linear(68, 64) -> ReLU -> Linear(64, 1)
  - Same as existing cache_replace.pt, but trained on optimal decisions

Usage:
    python -m ncpu.os.neuros.train_optimal_cache
    python -m ncpu.os.neuros.train_optimal_cache --epochs 200 --traces 2000
"""

import argparse
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.os.neuros.cache import CacheReplacementNet


# =============================================================================
# TRACE GENERATION
# =============================================================================

def generate_zipf_trace(length: int, num_items: int, alpha: float = 1.1,
                        seed: Optional[int] = None) -> List[int]:
    """Generate a Zipf-distributed access trace.

    Zipf distribution models real-world access patterns: a few items are
    accessed very frequently (hot), while most are accessed rarely (cold).
    Web caches, file systems, and database buffer pools all follow Zipf.

    Args:
        length: Number of accesses in the trace.
        num_items: Number of distinct items (address space size).
        alpha: Zipf exponent. Higher = more skewed toward hot items.
        seed: Random seed for reproducibility.

    Returns:
        List of item IDs (0-indexed).
    """
    rng = random.Random(seed)
    # Precompute CDF for Zipf
    weights = [1.0 / (i + 1) ** alpha for i in range(num_items)]
    total = sum(weights)
    probs = [w / total for w in weights]
    # Use cumulative distribution for sampling
    cum = []
    s = 0.0
    for p in probs:
        s += p
        cum.append(s)

    trace = []
    for _ in range(length):
        r = rng.random()
        # Binary search for the item
        lo, hi = 0, num_items - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        trace.append(lo)
    return trace


def generate_sequential_scan_trace(length: int, num_items: int,
                                   scan_fraction: float = 0.3,
                                   seed: Optional[int] = None) -> List[int]:
    """Generate a trace with mixed sequential scans and random accesses.

    Models real workloads where a process periodically scans through a large
    dataset (e.g., log rotation, backup, table scan) while also doing random
    lookups on a hot working set.

    The key challenge for cache replacement: sequential scans can pollute the
    cache by evicting hot items. Belady's knows the scan items won't be reused
    soon and keeps the hot items. LRU gets tricked by the recency of scan items.
    """
    rng = random.Random(seed)
    hot_set_size = max(4, num_items // 5)
    hot_set = list(range(hot_set_size))
    cold_set = list(range(hot_set_size, num_items))

    trace = []
    i = 0
    while i < length:
        if rng.random() < scan_fraction and i + 20 < length:
            # Sequential scan of 10-30 cold items
            scan_len = rng.randint(10, min(30, len(cold_set)))
            start = rng.randint(0, max(0, len(cold_set) - scan_len))
            for j in range(scan_len):
                trace.append(cold_set[(start + j) % len(cold_set)])
            i += scan_len
        else:
            # Random access to hot working set
            trace.append(rng.choice(hot_set))
            i += 1

    return trace[:length]


def generate_temporal_locality_trace(length: int, num_items: int,
                                     locality_window: int = 50,
                                     seed: Optional[int] = None) -> List[int]:
    """Generate a trace with strong temporal locality.

    Recently accessed items have a higher probability of being accessed again.
    This models iterative algorithms, loop-heavy code, and interactive sessions
    where the user works on a small set of files before moving to a new set.
    """
    rng = random.Random(seed)
    trace = []
    recent = []
    max_recent = locality_window

    for _ in range(length):
        if recent and rng.random() < 0.7:
            # 70% chance: re-access something from recent window
            # Bias toward more recent items
            idx = min(int(rng.expovariate(0.1)), len(recent) - 1)
            trace.append(recent[-(idx + 1)])
        else:
            # 30% chance: access a new or random item
            item = rng.randint(0, num_items - 1)
            trace.append(item)

        recent.append(trace[-1])
        if len(recent) > max_recent:
            recent = recent[-max_recent:]

    return trace


def generate_working_set_shift_trace(length: int, num_items: int,
                                     shift_interval: int = 200,
                                     seed: Optional[int] = None) -> List[int]:
    """Generate a trace where the working set changes periodically.

    Models application phases: a program works on one set of data, then
    transitions to a different set. Good cache policies detect the shift
    and quickly evict the old working set.
    """
    rng = random.Random(seed)
    ws_size = max(4, num_items // 8)
    trace = []

    phase = 0
    for i in range(length):
        if i > 0 and i % shift_interval == 0:
            phase += 1

        # Working set for this phase
        ws_start = (phase * ws_size) % num_items
        ws = [(ws_start + j) % num_items for j in range(ws_size)]

        # Mostly access working set, occasionally access random
        if rng.random() < 0.85:
            trace.append(rng.choice(ws))
        else:
            trace.append(rng.randint(0, num_items - 1))

    return trace


def generate_mixed_trace(length: int, num_items: int,
                         seed: Optional[int] = None) -> List[int]:
    """Generate a trace mixing all patterns — most realistic workload."""
    rng = random.Random(seed)
    generators = [
        lambda s: generate_zipf_trace(length, num_items, alpha=1.1, seed=s),
        lambda s: generate_sequential_scan_trace(length, num_items, seed=s),
        lambda s: generate_temporal_locality_trace(length, num_items, seed=s),
        lambda s: generate_working_set_shift_trace(length, num_items, seed=s),
    ]

    trace = []
    segment_len = length // 4
    for i, gen in enumerate(generators):
        seg = gen(seed + i * 1000 if seed is not None else rng.randint(0, 100000))
        trace.extend(seg[:segment_len])

    # Pad to exact length
    while len(trace) < length:
        trace.append(rng.randint(0, num_items - 1))
    return trace[:length]


# =============================================================================
# BELADY'S OPTIMAL ALGORITHM
# =============================================================================

def compute_belady_decisions(trace: List[int], cache_size: int
                             ) -> List[Tuple[List[int], int, List[float]]]:
    """Compute Belady's optimal eviction decisions for a trace.

    For each cache miss that requires eviction, determines which line to evict
    by looking ahead in the trace to find which cached item will be used
    furthest in the future (or never used again).

    Args:
        trace: Sequence of item IDs.
        cache_size: Number of items the cache can hold.

    Returns:
        List of (cache_state, victim_index, features_per_line) at each eviction.
        Each entry has:
          - cache_state: list of item IDs currently cached
          - victim_index: which index Belady's says to evict (0-indexed into cache_state)
          - access_history_features: recent access pattern features
    """
    # Precompute next-use distance for every position in the trace
    # next_use[i] = next index > i where trace[i] appears, or infinity
    next_use_map: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(trace) - 1, -1, -1):
        next_use_map[trace[i]].append(i)

    # For each item, sorted positions where it appears
    item_positions: Dict[int, List[int]] = {}
    for item, positions in next_use_map.items():
        item_positions[item] = sorted(positions)

    def get_next_use(item: int, after_pos: int) -> int:
        """Get the next position where item is accessed after after_pos."""
        positions = item_positions.get(item, [])
        # Binary search for first position > after_pos
        lo, hi = 0, len(positions)
        while lo < hi:
            mid = (lo + hi) // 2
            if positions[mid] <= after_pos:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(positions):
            return positions[lo]
        return float('inf')

    cache = []  # Items currently in cache
    last_access_tick = {}  # item -> tick of last access
    access_count = {}  # item -> total access count
    decisions = []
    tick = 0

    # Track history window for feature extraction
    history_window = []
    history_maxlen = 32

    for pos, item in enumerate(trace):
        tick += 1
        access_count[item] = access_count.get(item, 0) + 1
        last_access_tick[item] = tick

        history_window.append((item, tick))
        if len(history_window) > history_maxlen:
            history_window = history_window[-history_maxlen:]

        if item in cache:
            # Hit — no eviction needed
            continue

        if len(cache) < cache_size:
            # Cold miss — still filling cache
            cache.append(item)
            continue

        # Miss + cache full: need eviction. Use Belady's to pick victim.
        # Find which cached item has the farthest next use.
        max_next_use = -1
        victim_idx = 0
        for idx, cached_item in enumerate(cache):
            nu = get_next_use(cached_item, pos)
            if nu == float('inf'):
                # Item never used again — optimal to evict
                victim_idx = idx
                break
            if nu > max_next_use:
                max_next_use = nu
                victim_idx = idx

        # Build feature vectors for each cache line (matching CacheReplacementNet input)
        # Absolute per-tick normalization: recency = (tick - last_access) / tick
        # This gives values in [0, 1] where higher = longer since last access.
        # Combined with log-scaled frequency and LSTM history context, the model
        # learns to approximate Belady's by predicting future reuse distance.
        max_tick = float(max(tick, 1))
        max_count = max(access_count.get(c, 1) for c in cache)
        max_count = max(max_count, 1)

        line_features = []
        for cached_item in cache:
            last_acc = last_access_tick.get(cached_item, 0)
            recency = (tick - last_acc) / max_tick
            count = access_count.get(cached_item, 1)
            frequency = math.log1p(count) / max(math.log1p(max_count), 1.0)
            line_features.append([recency, frequency, 0.0, 1.0])

        # Build access history features
        hist_features = []
        for h_item, h_tick in history_window:
            addr_norm = float(h_item) / 1000.0
            hit = 1.0 if h_item in cache else 0.0
            write = 0.0
            tick_norm = float(h_tick) / 10000.0
            hist_features.append([addr_norm, hit, write, tick_norm])

        # Pad history to fixed length
        while len(hist_features) < history_maxlen:
            hist_features.insert(0, [0.0, 0.0, 0.0, 0.0])

        decisions.append({
            "cache_state": list(cache),
            "victim_idx": victim_idx,
            "line_features": line_features,
            "history_features": hist_features[-history_maxlen:],
        })

        # Perform the eviction
        cache[victim_idx] = item

    return decisions


def compute_lru_hit_rate(trace: List[int], cache_size: int) -> float:
    """Compute LRU hit rate on a trace for comparison."""
    cache = {}  # item -> tick
    hits = 0
    tick = 0
    for item in trace:
        tick += 1
        if item in cache:
            hits += 1
            cache[item] = tick
        else:
            if len(cache) >= cache_size:
                # Evict LRU
                lru_item = min(cache, key=cache.get)
                del cache[lru_item]
            cache[item] = tick
    return hits / len(trace) if trace else 0.0


def compute_belady_hit_rate(trace: List[int], cache_size: int) -> float:
    """Compute Belady's optimal hit rate on a trace."""
    # Build next-use index
    item_positions: Dict[int, List[int]] = defaultdict(list)
    for i, item in enumerate(trace):
        item_positions[item].append(i)

    def get_next_use(item: int, after_pos: int) -> int:
        positions = item_positions[item]
        lo, hi = 0, len(positions)
        while lo < hi:
            mid = (lo + hi) // 2
            if positions[mid] <= after_pos:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(positions):
            return positions[lo]
        return float('inf')

    cache = set()
    cache_list = []  # Maintain order for eviction
    hits = 0

    for pos, item in enumerate(trace):
        if item in cache:
            hits += 1
            continue
        if len(cache) < cache_size:
            cache.add(item)
            cache_list.append(item)
            continue
        # Evict: find item with farthest next use
        max_nu = -1
        victim = cache_list[0]
        for c in cache_list:
            nu = get_next_use(c, pos)
            if nu == float('inf'):
                victim = c
                break
            if nu > max_nu:
                max_nu = nu
                victim = c
        cache.remove(victim)
        cache_list.remove(victim)
        cache.add(item)
        cache_list.append(item)

    return hits / len(trace) if trace else 0.0


# =============================================================================
# TRAINING DATA PREPARATION
# =============================================================================

def build_training_dataset(
    num_traces: int = 500,
    trace_length: int = 2000,
    num_items: int = 64,
    cache_size: int = 8,
    seed: int = 42,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Build training dataset from Belady-optimal decisions on generated traces.

    Returns:
        histories: List of [1, seq_len, 4] tensors (access history features)
        features: List of [cache_size, 4] tensors (per-line features)
        targets: List of scalar tensors (victim index)
    """
    histories = []
    features = []
    targets = []

    generators = [
        ("zipf_1.0", lambda s: generate_zipf_trace(trace_length, num_items, 1.0, s)),
        ("zipf_1.2", lambda s: generate_zipf_trace(trace_length, num_items, 1.2, s)),
        ("zipf_1.5", lambda s: generate_zipf_trace(trace_length, num_items, 1.5, s)),
        ("sequential", lambda s: generate_sequential_scan_trace(trace_length, num_items, 0.3, s)),
        ("seq_heavy", lambda s: generate_sequential_scan_trace(trace_length, num_items, 0.5, s)),
        ("temporal", lambda s: generate_temporal_locality_trace(trace_length, num_items, seed=s)),
        ("ws_shift", lambda s: generate_working_set_shift_trace(trace_length, num_items, seed=s)),
        ("mixed", lambda s: generate_mixed_trace(trace_length, num_items, seed=s)),
    ]

    traces_per_gen = max(1, num_traces // len(generators))

    for gen_name, gen_fn in generators:
        for i in range(traces_per_gen):
            s = seed + hash(gen_name) + i
            trace = gen_fn(s)
            decisions = compute_belady_decisions(trace, cache_size)

            for d in decisions:
                hist_t = torch.tensor(d["history_features"], dtype=torch.float32).unsqueeze(0)
                feat_t = torch.tensor(d["line_features"], dtype=torch.float32)
                target_t = torch.tensor(d["victim_idx"], dtype=torch.long)
                histories.append(hist_t)
                features.append(feat_t)
                targets.append(target_t)

    return histories, features, targets


# =============================================================================
# TRAINING LOOP
# =============================================================================

def _batched_forward(model: CacheReplacementNet, hist_batch: torch.Tensor,
                     feat_batch: torch.Tensor) -> torch.Tensor:
    """Batch forward pass through the replacement model.

    The LSTM processes each history independently, but since all samples
    have the same history_len and cache_size, we can batch the LSTM pass
    and score computation efficiently.

    Args:
        hist_batch: [B, seq_len, 4] — access history features
        feat_batch: [B, cache_size, 4] — per-line features

    Returns:
        scores: [B, cache_size] — eviction scores per line per sample
    """
    B = hist_batch.shape[0]
    cache_size = feat_batch.shape[1]

    # LSTM over all histories: [B, seq_len, 4] -> hidden [1, B, hidden_dim]
    _, (h_n, _) = model.lstm(hist_batch)
    context = h_n[-1]  # [B, hidden_dim]

    # Expand context for each cache line: [B, cache_size, hidden_dim]
    context_exp = context.unsqueeze(1).expand(B, cache_size, -1)

    # Combine: [B, cache_size, hidden_dim + line_feature_dim]
    combined = torch.cat([context_exp, feat_batch], dim=-1)

    # Score: [B, cache_size, 1] -> [B, cache_size]
    scores = model.scorer(combined).squeeze(-1)
    return scores


def train_optimal_cache(
    epochs: int = 150,
    num_traces: int = 1000,
    trace_length: int = 2000,
    num_items: int = 64,
    cache_size: int = 8,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 42,
    device_str: str = "auto",
    save_path: Optional[str] = None,
) -> Dict:
    """Train the neural cache replacement model on Belady's optimal decisions.

    Uses proper tensor batching for fast training: since all samples have
    the same cache_size and history_len, we batch the LSTM and scorer
    computation across the entire minibatch.

    Returns:
        Dictionary with training stats and validation results.
    """
    # Device selection
    if device_str == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    print(f"[train] Device: {device}")
    print(f"[train] Generating training data: {num_traces} traces x {trace_length} accesses...")
    t0 = time.perf_counter()

    histories, features, targets = build_training_dataset(
        num_traces=num_traces,
        trace_length=trace_length,
        num_items=num_items,
        cache_size=cache_size,
        seed=seed,
    )

    total_samples = len(histories)
    gen_time = time.perf_counter() - t0
    print(f"[train] Generated {total_samples:,} eviction decisions in {gen_time:.1f}s")

    if total_samples == 0:
        print("[train] ERROR: No training samples generated. Try longer traces or more items.")
        return {"error": "no samples"}

    # Stack all data into contiguous tensors for fast batching.
    # histories[i] is [1, 32, 4], features[i] is [cache_size, 4], targets[i] is scalar
    all_hist = torch.cat(histories, dim=0)  # [N, 32, 4]
    all_feat = torch.stack(features)        # [N, cache_size, 4]
    all_tgt = torch.stack(targets)          # [N]

    # Split train/val (90/10)
    split = int(0.9 * total_samples)
    rng = random.Random(seed)
    indices = list(range(total_samples))
    rng.shuffle(indices)
    train_idx = torch.tensor(indices[:split], dtype=torch.long)
    val_idx = torch.tensor(indices[split:], dtype=torch.long)

    print(f"[train] Train: {len(train_idx):,}, Val: {len(val_idx):,}")

    # Move data to device once
    all_hist = all_hist.to(device)
    all_feat = all_feat.to(device)
    all_tgt = all_tgt.to(device)

    # Build model (same architecture as existing cache_replace.pt)
    model = CacheReplacementNet(
        access_feature_dim=4,
        hidden_dim=64,
        line_feature_dim=4,
        num_layers=1,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[train] Model: CacheReplacementNet ({param_count:,} params)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state = None

    print(f"[train] Training for {epochs} epochs (batch_size={batch_size}, lr={lr})...")
    print()

    for epoch in range(epochs):
        model.train()
        # Shuffle train indices
        perm = train_idx[torch.randperm(len(train_idx))]

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_start in range(0, len(perm), batch_size):
            bi = perm[batch_start:batch_start + batch_size]
            bh = all_hist[bi]   # [B, 32, 4]
            bf = all_feat[bi]   # [B, cache_size, 4]
            bt = all_tgt[bi]    # [B]

            scores = _batched_forward(model, bh, bf)  # [B, cache_size]
            loss = F.cross_entropy(scores, bt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = scores.argmax(dim=-1)
            train_correct += (preds == bt).sum().item()
            train_total += len(bi)
            train_loss += loss.item() * len(bi)

        scheduler.step()

        # Validation (batched)
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for vb_start in range(0, len(val_idx), batch_size):
                vi = val_idx[vb_start:vb_start + batch_size]
                vh = all_hist[vi]
                vf = all_feat[vi]
                vt = all_tgt[vi]

                scores = _batched_forward(model, vh, vf)
                loss = F.cross_entropy(scores, vt)
                preds = scores.argmax(dim=-1)
                val_correct += (preds == vt).sum().item()
                val_total += len(vi)
                val_loss += loss.item() * len(vi)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_train_loss = train_loss / max(train_total, 1)
        avg_val_loss = val_loss / max(val_total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  "
                  f"train_loss={avg_train_loss:.4f}  train_acc={train_acc:.1%}  "
                  f"val_loss={avg_val_loss:.4f}  val_acc={val_acc:.1%}  "
                  f"best_val={best_val_acc:.1%}")

    print()
    print(f"[train] Best validation accuracy: {best_val_acc:.1%}")

    # Save best model
    if save_path is None:
        save_path = str(PROJECT_ROOT / "models" / "os" / "cache_replace_optimal.pt")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, save_path)
    print(f"[train] Saved: {save_path}")

    # =========================================================================
    # VALIDATION: Compare neural-optimal vs LRU vs Belady on held-out traces
    # =========================================================================
    print()
    print("[eval] Evaluating on held-out traces...")

    model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()

    eval_results = evaluate_model(
        model=model,
        device=device,
        num_traces=50,
        trace_length=3000,
        num_items=num_items,
        cache_size=cache_size,
        seed=seed + 99999,
    )

    return {
        "best_val_acc": best_val_acc,
        "total_samples": total_samples,
        "save_path": save_path,
        "eval": eval_results,
    }


def evaluate_model(
    model: CacheReplacementNet,
    device: torch.device,
    num_traces: int = 50,
    trace_length: int = 3000,
    num_items: int = 64,
    cache_size: int = 8,
    seed: int = 99999,
) -> Dict:
    """Evaluate the trained model against LRU and Belady's optimal.

    Simulates the cache with the neural policy making eviction decisions
    and compares hit rates against LRU baseline and Belady's upper bound.
    """
    model.eval()

    generators = [
        ("zipf", lambda s: generate_zipf_trace(trace_length, num_items, 1.1, s)),
        ("sequential_scan", lambda s: generate_sequential_scan_trace(trace_length, num_items, 0.3, s)),
        ("temporal", lambda s: generate_temporal_locality_trace(trace_length, num_items, seed=s)),
        ("ws_shift", lambda s: generate_working_set_shift_trace(trace_length, num_items, seed=s)),
        ("mixed", lambda s: generate_mixed_trace(trace_length, num_items, seed=s)),
    ]

    results = {}
    all_lru = []
    all_neural = []
    all_belady = []

    for gen_name, gen_fn in generators:
        lru_rates = []
        neural_rates = []
        belady_rates = []

        traces_per = max(1, num_traces // len(generators))
        for i in range(traces_per):
            s = seed + hash(gen_name) + i * 7
            trace = gen_fn(s)

            lru_hr = compute_lru_hit_rate(trace, cache_size)
            belady_hr = compute_belady_hit_rate(trace, cache_size)
            neural_hr = simulate_neural_cache(model, device, trace, cache_size)

            lru_rates.append(lru_hr)
            neural_rates.append(neural_hr)
            belady_rates.append(belady_hr)

        avg_lru = sum(lru_rates) / len(lru_rates)
        avg_neural = sum(neural_rates) / len(neural_rates)
        avg_belady = sum(belady_rates) / len(belady_rates)
        delta = avg_neural - avg_lru

        all_lru.extend(lru_rates)
        all_neural.extend(neural_rates)
        all_belady.extend(belady_rates)

        results[gen_name] = {
            "lru": avg_lru,
            "neural": avg_neural,
            "belady": avg_belady,
            "delta_vs_lru": delta,
        }

        delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
        print(f"  {gen_name:20s}  LRU={avg_lru:.1%}  Neural={avg_neural:.1%}  "
              f"Belady={avg_belady:.1%}  Delta={delta_str}")

    overall_lru = sum(all_lru) / len(all_lru)
    overall_neural = sum(all_neural) / len(all_neural)
    overall_belady = sum(all_belady) / len(all_belady)
    overall_delta = overall_neural - overall_lru

    print()
    delta_str = f"+{overall_delta:.1%}" if overall_delta >= 0 else f"{overall_delta:.1%}"
    print(f"  {'OVERALL':20s}  LRU={overall_lru:.1%}  Neural={overall_neural:.1%}  "
          f"Belady={overall_belady:.1%}  Delta={delta_str}")

    # Gap closed: what fraction of the LRU→Belady gap did the neural model close?
    belady_gap = overall_belady - overall_lru
    if belady_gap > 0:
        gap_closed = (overall_neural - overall_lru) / belady_gap
        print(f"  Gap closed (LRU→Belady): {gap_closed:.1%}")
    else:
        gap_closed = 0.0

    results["overall"] = {
        "lru": overall_lru,
        "neural": overall_neural,
        "belady": overall_belady,
        "delta_vs_lru": overall_delta,
        "gap_closed": gap_closed,
    }

    return results


def simulate_neural_cache(
    model: CacheReplacementNet,
    device: torch.device,
    trace: List[int],
    cache_size: int,
) -> float:
    """Simulate a cache using the neural model for eviction decisions.

    Returns the hit rate.
    """
    cache = []  # List of items in cache
    last_access_tick = {}
    access_count = {}
    history_window = []
    history_maxlen = 32
    hits = 0
    tick = 0

    for item in trace:
        tick += 1
        access_count[item] = access_count.get(item, 0) + 1
        last_access_tick[item] = tick

        history_window.append((item, tick))
        if len(history_window) > history_maxlen:
            history_window = history_window[-history_maxlen:]

        if item in cache:
            hits += 1
            continue

        if len(cache) < cache_size:
            cache.append(item)
            continue

        # Need eviction — ask the neural model
        # Use same absolute per-tick normalization as training data
        max_tick_f = float(max(tick, 1))
        max_count = max(access_count.get(c, 1) for c in cache)
        max_count = max(max_count, 1)

        line_features = []
        for c in cache:
            la = last_access_tick.get(c, 0)
            recency = (tick - la) / max_tick_f
            count = access_count.get(c, 1)
            frequency = math.log1p(count) / max(math.log1p(max_count), 1.0)
            line_features.append([recency, frequency, 0.0, 1.0])

        feat_t = torch.tensor(line_features, dtype=torch.float32, device=device)

        # Build history features
        hist_features = []
        for h_item, h_tick in history_window:
            addr_norm = float(h_item) / 1000.0
            hit = 1.0 if h_item in cache else 0.0
            hist_features.append([addr_norm, hit, 0.0, float(h_tick) / 10000.0])

        while len(hist_features) < history_maxlen:
            hist_features.insert(0, [0.0, 0.0, 0.0, 0.0])

        hist_t = torch.tensor(
            hist_features[-history_maxlen:],
            dtype=torch.float32, device=device,
        ).unsqueeze(0)

        with torch.no_grad():
            scores = model(hist_t, feat_t)
            victim_idx = int(scores.argmax().item())

        cache[victim_idx] = item

    return hits / len(trace) if trace else 0.0


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train neural cache replacement on Belady's optimal policy")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Training epochs (default: 150)")
    parser.add_argument("--traces", type=int, default=1000,
                        help="Number of training traces (default: 1000)")
    parser.add_argument("--trace-length", type=int, default=2000,
                        help="Length of each trace (default: 2000)")
    parser.add_argument("--num-items", type=int, default=64,
                        help="Address space size (default: 64)")
    parser.add_argument("--cache-size", type=int, default=8,
                        help="Cache capacity (default: 8)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Training batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, mps, cuda")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Output model path (default: models/research/neuros/cache_replace_optimal.pt)")
    args = parser.parse_args()

    print("=" * 66)
    print("  Belady-Optimal Neural Cache Replacement Training")
    print("=" * 66)
    print()

    results = train_optimal_cache(
        epochs=args.epochs,
        num_traces=args.traces,
        trace_length=args.trace_length,
        num_items=args.num_items,
        cache_size=args.cache_size,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device_str=args.device,
        save_path=args.save_path,
    )

    print()
    print("=" * 66)
    print("  Training Complete")
    print("=" * 66)
    if "eval" in results:
        overall = results["eval"].get("overall", {})
        delta = overall.get("delta_vs_lru", 0)
        print(f"  Neural vs LRU: {'+' if delta >= 0 else ''}{delta:.1%}")
        print(f"  Gap closed:    {overall.get('gap_closed', 0):.1%}")
        print(f"  Saved to:      {results.get('save_path', 'N/A')}")
    print()


if __name__ == "__main__":
    main()
