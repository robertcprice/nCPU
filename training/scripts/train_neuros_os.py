#!/usr/bin/env python3
"""Train all neurOS component models on synthetic data.

Trains 7 neural networks to replace classical OS fallbacks:

1. NeuralEvictionPolicy (TLB)     — LRU-equivalent eviction scorer (~1K params)
2. NeuralPriorityEncoder (GIC)    — Fixed-priority-equivalent dispatch (~12K params)
3. CacheReplacementNet            — LRU-equivalent cache replacement (~22K params)
4. PrefetchNet                    — Stride-equivalent prefetcher (~2M params)
5. SchedulerNet                   — Priority-with-aging scheduler (~50K params)
6. BlockAllocatorNet (FS)         — First-fit-equivalent allocator (~10K params)
7. NeuralPageTable (MMU)          — Page table lookup network (~200K params)

Each model is trained on synthetic data generated from its classical oracle,
so the neural version matches the fallback behavior. This makes the entire
neurOS stack neural — zero classical fallbacks at runtime.

Usage:
    python training/train_neuros_os.py
    python training/train_neuros_os.py --quick
    python training/train_neuros_os.py --epochs 500
"""

import sys
import time
import random
import argparse
import json
import functools
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

# Force unbuffered output for nohup/redirect compatibility
print = functools.partial(print, flush=True)


# =============================================================================
# Device Selection
# =============================================================================

def select_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[Device] CUDA: {name} ({mem:.1f} GB)")
        return dev
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[Device] Apple Silicon MPS")
        return torch.device("mps")
    else:
        print("[Device] CPU")
        return torch.device("cpu")


# =============================================================================
# 1. TLB Neural Eviction Policy (batched)
# =============================================================================

def train_tlb_eviction(device: torch.device, epochs: int = 300,
                       lr: float = 1e-3) -> Tuple[nn.Module, Dict]:
    """Train TLB eviction policy to match enhanced-LRU behavior.

    Oracle: score = recency * 1.0 + (1-dirty) * 0.1 + (1-code) * 0.05
                    + age * 0.2 + (1-access_count) * 0.15
    Highest score = evict first.

    Batched: [batch, tlb_size, 5] → [batch] target indices
    """
    print("\n" + "=" * 70)
    print("TRAINING: TLB NeuralEvictionPolicy (LRU-equivalent)")
    print("=" * 70)

    from ncpu.os.tlb import NeuralEvictionPolicy

    model = NeuralEvictionPolicy(feature_dim=5, hidden_dim=32).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    n_samples = 10000
    tlb_size = 64

    # Generate all training data as a single batched tensor
    features = torch.rand(n_samples, tlb_size, 5, device=device)

    # Oracle: compute eviction scores and get argmax
    oracle_scores = (
        features[:, :, 1] * 1.0 +          # recency
        (1 - features[:, :, 2]) * 0.1 +     # prefer clean
        (1 - features[:, :, 3]) * 0.05 +    # prefer data
        features[:, :, 4] * 0.2 +           # older entries
        (1 - features[:, :, 0]) * 0.15      # less accessed
    )
    targets = oracle_scores.argmax(dim=1)  # [n_samples]

    print(f"  Training samples: {n_samples}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_opt = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_state = None
    t0 = time.time()
    batch_size = 128

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = perm[start:end]
            b = idx.shape[0]

            # [b, tlb_size, 5] → [b * tlb_size, 5]
            batch_f = features[idx].reshape(-1, 5)
            batch_scores = model(batch_f).squeeze(-1)  # [b * tlb_size]
            batch_scores = batch_scores.reshape(b, tlb_size)  # [b, tlb_size]

            loss = loss_fn(batch_scores, targets[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler_opt.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                all_scores = model(features.reshape(-1, 5)).squeeze(-1)
                all_scores = all_scores.reshape(n_samples, tlb_size)
                pred = all_scores.argmax(dim=1)
                acc = (pred == targets).float().mean().item()

            elapsed = time.time() - t0
            print(f"  Epoch {epoch:4d} | Loss: {total_loss/n_batches:.6f} | "
                  f"Acc: {acc:.4f} | Time: {elapsed:.0f}s")

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if best_acc >= 0.99:
                print(f"  [EARLY STOP] 99%+ accuracy at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    elapsed = time.time() - t0
    print(f"  RESULT: {best_acc:.4f} accuracy in {elapsed:.1f}s")
    return model, {"accuracy": best_acc, "time": elapsed, "params": params}


# =============================================================================
# 2. GIC Neural Priority Encoder (batched)
# =============================================================================

def train_gic_priority(device: torch.device, epochs: int = 500,
                       lr: float = 1e-3) -> Tuple[nn.Module, Dict]:
    """Train GIC priority encoder to match fixed-priority dispatch.

    Oracle: lowest pending IRQ number (not masked, not in-service) wins.
    """
    print("\n" + "=" * 70)
    print("TRAINING: GIC NeuralPriorityEncoder (fixed-priority equivalent)")
    print("=" * 70)

    from ncpu.os.interrupts import NeuralPriorityEncoder, NUM_IRQS

    model = NeuralPriorityEncoder(num_irqs=NUM_IRQS, hidden_dim=64).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    n_samples = 30000
    features = torch.zeros(n_samples, NUM_IRQS * 3, device=device)
    targets = torch.zeros(n_samples, NUM_IRQS, device=device)

    valid_count = 0
    while valid_count < n_samples:
        batch = min(1000, n_samples - valid_count)
        irr = (torch.rand(batch, NUM_IRQS, device=device) > 0.7).float()
        isr = (torch.rand(batch, NUM_IRQS, device=device) > 0.9).float()
        imr = (torch.rand(batch, NUM_IRQS, device=device) > 0.85).float()

        pending = irr * (1 - imr) * (1 - isr)
        has_pending = pending.sum(dim=1) > 0

        for i in range(batch):
            if not has_pending[i]:
                continue
            if valid_count >= n_samples:
                break

            # Target: highest score for lowest pending IRQ
            target_scores = torch.zeros(NUM_IRQS, device=device)
            pending_idx = pending[i].nonzero(as_tuple=True)[0]
            for rank, idx in enumerate(pending_idx):
                target_scores[idx] = float(NUM_IRQS - rank)

            features[valid_count] = torch.cat([irr[i], isr[i], imr[i]])
            targets[valid_count] = target_scores
            valid_count += 1

    print(f"  Training samples: {n_samples}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_opt = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    loss_fn = nn.MSELoss()
    best_acc = 0.0
    best_state = None
    t0 = time.time()
    batch_size = 512

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = perm[start:end]

            optimizer.zero_grad()
            pred = model(features[idx])
            loss = loss_fn(pred, targets[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler_opt.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                pred_scores = model(features)
                irr_all = features[:, :NUM_IRQS]
                isr_all = features[:, NUM_IRQS:2*NUM_IRQS]
                imr_all = features[:, 2*NUM_IRQS:]
                pending_all = irr_all * (1 - imr_all) * (1 - isr_all)

                masked_pred = pred_scores.clone()
                masked_pred[pending_all < 0.5] = float('-inf')
                masked_target = targets.clone()
                masked_target[pending_all < 0.5] = float('-inf')

                pred_irq = masked_pred.argmax(dim=1)
                target_irq = masked_target.argmax(dim=1)
                acc = (pred_irq == target_irq).float().mean().item()

            elapsed = time.time() - t0
            print(f"  Epoch {epoch:4d} | Loss: {total_loss/n_batches:.6f} | "
                  f"Dispatch Acc: {acc:.4f} | Time: {elapsed:.0f}s")

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if best_acc >= 0.99:
                print(f"  [EARLY STOP] 99%+ accuracy at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    elapsed = time.time() - t0
    print(f"  RESULT: {best_acc:.4f} dispatch accuracy in {elapsed:.1f}s")
    return model, {"accuracy": best_acc, "time": elapsed, "params": params}


# =============================================================================
# 3. Cache Replacement Network (batched)
# =============================================================================

def train_cache_replacement(device: torch.device, epochs: int = 300,
                            lr: float = 1e-3) -> Tuple[nn.Module, Dict]:
    """Train LSTM cache replacement to match LRU behavior.

    Oracle: LRU evicts line with highest recency (feature 0).
    Fully batched: LSTM processes [batch, seq_len, features] in one shot,
    then scorer processes [batch, num_ways, hidden+features] in parallel.
    """
    print("\n" + "=" * 70)
    print("TRAINING: CacheReplacementNet (LRU-equivalent)")
    print("=" * 70)

    from ncpu.os.cache import CacheReplacementNet

    model = CacheReplacementNet(
        access_feature_dim=4, hidden_dim=64, line_feature_dim=4
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    n_samples = 10000
    num_ways = 4
    history_len = 32

    # Pre-generate all data as tensors — histories are [n, seq_len, 4] for batched LSTM
    histories = torch.rand(n_samples, history_len, 4, device=device)
    line_features = torch.rand(n_samples, num_ways, 4, device=device)
    line_features[:, :, 3] = 1.0  # All valid

    # Oracle: LRU = evict line with highest recency (feature 0)
    targets = line_features[:, :, 0].argmax(dim=1)  # [n_samples]

    print(f"  Training samples: {n_samples}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_opt = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_state = None
    t0 = time.time()
    batch_size = 256

    def batched_score(hist_batch, lines_batch):
        """Batch LSTM + scorer: [b, seq, 4] x [b, ways, 4] → [b, ways]"""
        b = hist_batch.shape[0]
        # LSTM processes all sequences in parallel: [b, seq_len, 4] → h_n [1, b, hidden]
        _, (h_n, _) = model.lstm(hist_batch)
        context = h_n[-1]  # [b, hidden_dim]
        # Expand context for each cache line: [b, 1, hidden] → [b, ways, hidden]
        context_exp = context.unsqueeze(1).expand(-1, num_ways, -1)
        # Concatenate with line features: [b, ways, hidden+4]
        combined = torch.cat([context_exp, lines_batch], dim=-1)
        # Score all lines in parallel: [b, ways, hidden+4] → [b, ways]
        scores = model.scorer(combined).squeeze(-1)
        return scores

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = perm[start:end]

            scores = batched_score(histories[idx], line_features[idx])
            loss = loss_fn(scores, targets[idx])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler_opt.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                all_scores = batched_score(histories, line_features)
                preds = all_scores.argmax(dim=1)
                acc = (preds == targets).float().mean().item()

            elapsed = time.time() - t0
            print(f"  Epoch {epoch:4d} | Loss: {total_loss/n_batches:.6f} | "
                  f"Acc: {acc:.4f} | Time: {elapsed:.0f}s")

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if best_acc >= 0.99:
                print(f"  [EARLY STOP] 99%+ accuracy at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    elapsed = time.time() - t0
    print(f"  RESULT: {best_acc:.4f} accuracy in {elapsed:.1f}s")
    return model, {"accuracy": best_acc, "time": elapsed, "params": params}


# =============================================================================
# 4. Prefetch Network (batched)
# =============================================================================

def train_prefetcher(device: torch.device, epochs: int = 300,
                     lr: float = 1e-3) -> Tuple[nn.Module, Dict]:
    """Train LSTM prefetcher to match stride-based prefetch.

    Oracle: detect constant stride, predict next K=4 addresses.
    Fully batched: LSTM processes [batch, seq_len, embed] in one shot.
    Trains on raw deltas to preserve gradient flow.
    """
    print("\n" + "=" * 70)
    print("TRAINING: PrefetchNet (stride-equivalent)")
    print("=" * 70)

    from ncpu.os.cache import PrefetchNet

    model = PrefetchNet(
        addr_bits=16, embed_dim=32, hidden_dim=64, num_predictions=4
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    addr_mask = (1 << 16) - 1
    n_samples = 10000
    seq_len = 32

    # Pre-generate all stride sequences: [n_samples, seq_len] (no extra dim)
    sequences = torch.zeros(n_samples, seq_len, dtype=torch.int64, device=device)
    target_deltas = torch.zeros(n_samples, 4, dtype=torch.float32, device=device)

    for i in range(n_samples):
        start = random.randint(0, addr_mask // 2)
        stride = random.choice([1, 2, 4, 8, 16, 32, 64, -1, -2, -4])

        addr = start
        for t in range(seq_len):
            sequences[i, t] = addr & addr_mask
            addr += stride

        last_addr = sequences[i, -1].float()
        for k in range(4):
            target_deltas[i, k] = float(((addr + k * stride) & addr_mask)) - last_addr.item()

    print(f"  Training samples: {n_samples}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_opt = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    best_acc = 0.0
    best_state = None
    t0 = time.time()
    batch_size = 256

    def batched_predict_deltas(seq_batch):
        """Batch LSTM + predictor: [b, seq_len] → [b, 4] raw deltas"""
        clamped = seq_batch & model.addr_mask
        embedded = model.addr_embed(clamped)  # [b, seq_len, embed_dim]
        _, (h_n, _) = model.lstm(embedded)
        return model.predictor(h_n[-1])  # [b, 4]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        n_batches = 0

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            idx = perm[start_idx:end_idx]

            pred_deltas = batched_predict_deltas(sequences[idx])
            loss = F.l1_loss(pred_deltas, target_deltas[idx])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler_opt.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                # Batched eval
                all_deltas = batched_predict_deltas(sequences)
                last_addrs = sequences[:, -1].float()
                pred_first = last_addrs + all_deltas[:, 0]
                expected_first = last_addrs + target_deltas[:, 0]
                correct = (pred_first - expected_first).abs() <= 2
                acc = correct.float().mean().item()

            elapsed = time.time() - t0
            print(f"  Epoch {epoch:4d} | Loss: {total_loss/n_batches:.4f} | "
                  f"Pred Acc (±2): {acc:.4f} | Time: {elapsed:.0f}s")

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if best_acc >= 0.95:
                print(f"  [EARLY STOP] 95%+ accuracy at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    elapsed = time.time() - t0
    print(f"  RESULT: {best_acc:.4f} prediction accuracy in {elapsed:.1f}s")
    return model, {"accuracy": best_acc, "time": elapsed, "params": params}


# =============================================================================
# 5. Scheduler Network (batched, fixed queue size)
# =============================================================================

def train_scheduler(device: torch.device, epochs: int = 500,
                    lr: float = 1e-3) -> Tuple[nn.Module, Dict]:
    """Train Transformer scheduler to match priority-with-aging policy.

    Oracle: effective_priority = priority - aging_bonus - interactive_boost
    Trains with fixed queue size for efficient batching.
    """
    print("\n" + "=" * 70)
    print("TRAINING: SchedulerNet (priority-with-aging equivalent)")
    print("=" * 70)

    from ncpu.os.scheduler import SchedulerNet, PROCESS_FEATURE_DIM

    model = SchedulerNet(
        feature_dim=PROCESS_FEATURE_DIM, d_model=64,
        nhead=4, num_layers=2
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    n_samples = 20000
    max_procs = 8  # Fixed queue size for batching

    # Generate batched training data: [n_samples, max_procs, 8]
    features = torch.rand(n_samples, max_procs, PROCESS_FEATURE_DIM, device=device)

    # Oracle: priority-based with aging
    priority = features[:, :, 0]           # normalized, lower = higher priority
    wait_log = features[:, :, 2]           # log-scaled wait time
    interactive = features[:, :, 5]        # interactive flag
    aging = torch.clamp(wait_log * 0.5, max=0.25)
    effective = priority - aging - interactive * 0.05

    # Target: index of process to schedule (lowest effective priority)
    targets = effective.argmin(dim=1)  # [n_samples]

    print(f"  Training samples: {n_samples}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_opt = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_state = None
    t0 = time.time()
    batch_size = 256

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = perm[start:end]

            optimizer.zero_grad()
            scores = model(features[idx])  # [b, max_procs]
            loss = loss_fn(scores, targets[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler_opt.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                all_scores = model(features)
                pred = all_scores.argmax(dim=1)
                acc = (pred == targets).float().mean().item()

            elapsed = time.time() - t0
            print(f"  Epoch {epoch:4d} | Loss: {total_loss/n_batches:.6f} | "
                  f"Top-1 Acc: {acc:.4f} | Time: {elapsed:.0f}s")

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if best_acc >= 0.95:
                print(f"  [EARLY STOP] 95%+ accuracy at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    elapsed = time.time() - t0
    print(f"  RESULT: {best_acc:.4f} top-1 accuracy in {elapsed:.1f}s")
    return model, {"accuracy": best_acc, "time": elapsed, "params": params}


# =============================================================================
# 6. Block Allocator Network (batched)
# =============================================================================

def train_block_allocator(device: torch.device, epochs: int = 300,
                          lr: float = 1e-3) -> Tuple[nn.Module, Dict]:
    """Train block allocator to match first-fit behavior.

    Oracle: pick region with lowest occupancy (prefer lower index on tie).
    """
    print("\n" + "=" * 70)
    print("TRAINING: BlockAllocatorNet (first-fit equivalent)")
    print("=" * 70)

    from ncpu.os.filesystem import BlockAllocatorNet

    num_regions = 64
    feature_dim = 16
    model = BlockAllocatorNet(
        num_regions=num_regions, feature_dim=feature_dim, hidden_dim=64
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    n_samples = 30000
    features = torch.rand(n_samples, feature_dim, device=device)

    # Oracle: first-fit = lowest occupancy, with index tiebreaker
    scores = features + torch.arange(feature_dim, device=device).float() * 0.001
    best_group = scores.argmin(dim=1)  # [n_samples], range [0, 15]
    # Map to 64-region space: each group of 16 features covers 4 regions
    targets = best_group * (num_regions // feature_dim)  # [n_samples], range [0, 60]

    print(f"  Training samples: {n_samples}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_opt = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_state = None
    t0 = time.time()
    batch_size = 512

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = perm[start:end]

            optimizer.zero_grad()
            pred = model(features[idx])  # [b, 64]
            loss = loss_fn(pred, targets[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler_opt.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                pred_all = model(features).argmax(dim=1)
                # Check group accuracy (predicted region / 4 == target group)
                pred_group = pred_all // (num_regions // feature_dim)
                acc = (pred_group == best_group).float().mean().item()

            elapsed = time.time() - t0
            print(f"  Epoch {epoch:4d} | Loss: {total_loss/n_batches:.6f} | "
                  f"Group Acc: {acc:.4f} | Time: {elapsed:.0f}s")

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if best_acc >= 0.99:
                print(f"  [EARLY STOP] 99%+ accuracy at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    elapsed = time.time() - t0
    print(f"  RESULT: {best_acc:.4f} group accuracy in {elapsed:.1f}s")
    return model, {"accuracy": best_acc, "time": elapsed, "params": params}


# =============================================================================
# 7. MMU Neural Page Table
# =============================================================================

def train_mmu(device: torch.device, epochs: int = 500,
              lr: float = 1e-3) -> Tuple[nn.Module, Dict]:
    """Train MMU neural page table on synthetic page mappings.

    Creates a synthetic OS memory layout and trains the network to
    reproduce VPN→PFN mappings with correct permission bits.
    """
    print("\n" + "=" * 70)
    print("TRAINING: NeuralPageTable (page table equivalent)")
    print("=" * 70)

    from ncpu.os.mmu import NeuralPageTable, NUM_PERM_BITS

    max_vpages = 4096
    max_pframes = 4096
    model = NeuralPageTable(
        max_virtual_pages=max_vpages,
        max_physical_frames=max_pframes,
        embed_dim=64, hidden_dim=256,
        asid_dim=16, max_asid=256,
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    # Synthetic page table: typical OS layout
    mappings = []

    def add_range(vpn_start, pfn_start, count, r, w, x):
        for i in range(count):
            perms = torch.zeros(NUM_PERM_BITS)
            perms[0] = 1.0  # valid
            perms[1] = float(r)
            perms[2] = float(w)
            perms[3] = float(x)
            mappings.append((vpn_start + i, 0, pfn_start + i, perms))

    add_range(0, 0, 128, True, True, True)       # Kernel
    add_range(256, 128, 128, True, False, True)   # User code
    add_range(512, 256, 128, True, True, False)   # User heap
    add_range(896, 384, 128, True, True, False)   # User stack
    add_range(1024, 512, 64, True, False, True)   # Shared libs

    n_mapped = len(mappings)
    print(f"  Mapped pages: {n_mapped}")

    vpns = torch.tensor([m[0] for m in mappings], dtype=torch.int64, device=device)
    asids = torch.tensor([m[1] for m in mappings], dtype=torch.int64, device=device)
    target_pfns = torch.tensor([m[2] for m in mappings], dtype=torch.int64, device=device)
    target_perms = torch.stack([m[3] for m in mappings]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_opt = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    pfn_loss_fn = nn.CrossEntropyLoss()
    perm_loss_fn = nn.BCEWithLogitsLoss()
    best_acc = 0.0
    best_state = None
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        vpn_e = model.vpn_embed(vpns)
        asid_e = model.asid_embed(asids)
        x = torch.cat([vpn_e, asid_e], dim=-1)
        out = model.mlp(x)

        pfn_logits = out[:, :max_pframes]
        perm_logits = out[:, max_pframes:]

        loss_pfn = pfn_loss_fn(pfn_logits, target_pfns)
        loss_perm = perm_loss_fn(perm_logits, target_perms)
        loss = loss_pfn + loss_perm

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler_opt.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                vpn_e = model.vpn_embed(vpns)
                asid_e = model.asid_embed(asids)
                x = torch.cat([vpn_e, asid_e], dim=-1)
                out = model.mlp(x)
                pred_pfns = out[:, :max_pframes].argmax(dim=-1)
                acc = (pred_pfns == target_pfns).float().mean().item()

                pred_perms = (torch.sigmoid(out[:, max_pframes:]) > 0.5).float()
                perm_acc = (pred_perms == target_perms).float().mean().item()

            elapsed = time.time() - t0
            print(f"  Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
                  f"PFN Acc: {acc:.4f} | Perm Acc: {perm_acc:.4f} | Time: {elapsed:.0f}s")

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if best_acc >= 1.0:
                print(f"  [EARLY STOP] 100% PFN accuracy at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    elapsed = time.time() - t0
    print(f"  RESULT: {best_acc:.4f} PFN accuracy on {n_mapped} mappings in {elapsed:.1f}s")
    return model, {"accuracy": best_acc, "time": elapsed, "params": params,
                    "mapped_pages": n_mapped}


# =============================================================================
# Save Models
# =============================================================================

def save_os_models(models: Dict[str, nn.Module], output_dir: Path, stats: Dict):
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "tlb": output_dir / "tlb.pt",
        "gic": output_dir / "gic.pt",
        "cache_replace": output_dir / "cache_replace.pt",
        "prefetch": output_dir / "prefetch.pt",
        "scheduler": output_dir / "scheduler.pt",
        "block_alloc": output_dir / "block_alloc.pt",
        "mmu": output_dir / "mmu.pt",
    }

    print(f"\n  Saving models to {output_dir}/")
    for name, model in models.items():
        path = paths[name]
        torch.save(model.state_dict(), path)
        size_kb = path.stat().st_size / 1024
        print(f"    {path.name:25s} ({size_kb:.1f} KB)")

    stats_path = output_dir / "os_training_stats.json"
    clean_stats = {}
    for k, v in stats.items():
        if isinstance(v, dict):
            clean_stats[k] = {kk: vv for kk, vv in v.items()
                              if isinstance(vv, (int, float, str, bool))}
        elif isinstance(v, (int, float, str, bool)):
            clean_stats[k] = v
    with open(stats_path, "w") as f:
        json.dump(clean_stats, f, indent=2)
    print(f"    {stats_path.name}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train neurOS component models")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip", type=str, default="",
                        help="Comma-separated: tlb,gic,cache,prefetch,scheduler,alloc,mmu")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 50
        print("[MODE] Quick smoke test")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = select_device()
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "models" / "os"
    skip = set(args.skip.split(",")) if args.skip else set()

    print(f"\n{'='*70}")
    print(f"neurOS Component Training Pipeline")
    print(f"{'='*70}")
    print(f"  Device:     {device}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Output:     {output_dir}")
    print(f"  Skip:       {skip or 'none'}")

    t_start = time.time()
    models = {}
    all_stats = {}

    if "tlb" not in skip:
        m, s = train_tlb_eviction(device, epochs=args.epochs, lr=args.lr)
        models["tlb"] = m; all_stats["tlb"] = s

    if "gic" not in skip:
        m, s = train_gic_priority(device, epochs=args.epochs, lr=args.lr)
        models["gic"] = m; all_stats["gic"] = s

    if "cache" not in skip:
        m, s = train_cache_replacement(device, epochs=args.epochs, lr=args.lr)
        models["cache_replace"] = m; all_stats["cache_replace"] = s

    if "prefetch" not in skip:
        m, s = train_prefetcher(device, epochs=args.epochs, lr=args.lr)
        models["prefetch"] = m; all_stats["prefetch"] = s

    if "scheduler" not in skip:
        m, s = train_scheduler(device, epochs=args.epochs, lr=args.lr)
        models["scheduler"] = m; all_stats["scheduler"] = s

    if "alloc" not in skip:
        m, s = train_block_allocator(device, epochs=args.epochs, lr=args.lr)
        models["block_alloc"] = m; all_stats["alloc"] = s

    if "mmu" not in skip:
        m, s = train_mmu(device, epochs=args.epochs, lr=args.lr)
        models["mmu"] = m; all_stats["mmu"] = s

    if models:
        save_os_models(models, output_dir, all_stats)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE — {len(models)} models in {elapsed:.1f}s")
    print(f"{'='*70}")
    for name, stats in all_stats.items():
        acc = stats.get("accuracy", 0)
        t = stats.get("time", 0)
        p = stats.get("params", 0)
        print(f"  {name:20s}: {acc:.4f} accuracy | {p:,} params | {t:.1f}s")


if __name__ == "__main__":
    main()
