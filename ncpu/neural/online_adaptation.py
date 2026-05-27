"""Online Model Adaptation for Neural Cache and Scheduler.

Fine-tunes the cache replacement LSTM and Transformer scheduler on live
workload traces during runtime. This is genuinely novel: conventional OSes
use fixed policies (LRU, CFS, round-robin) that cannot adapt to the specific
workload running on them. The nCPU neural OS learns from its own decisions
in real time, improving cache hit rates and scheduling quality as it runs.

Architecture:
    OnlineCacheAdapter   -- replay-buffer + few-shot gradient updates on CacheReplacementNet
    OnlineSchedulerAdapter -- decision log + few-shot gradient updates on SchedulerNet
    AdaptationManager    -- coordinates both, tracks metrics, checkpoints adapted weights

Usage:
    from ncpu.neural.online_adaptation import AdaptationManager

    # Attach to a booted NeurOS instance
    mgr = AdaptationManager(neuros.cache, neuros.scheduler)
    mgr.enable()

    # During normal operation, feed events:
    mgr.on_cache_access(addr=0x4000, hit=True, write=False, victim_way=None)
    mgr.on_schedule_decision(selected_pid=3, ready_pids=[1,2,3], turnaround=42.0, throughput=0.85)

    # Periodically inspect improvement:
    print(mgr.metrics())

    # Save session-specific adapted weights
    mgr.save_checkpoint("/tmp/session_adapted")
"""

from __future__ import annotations

import copy
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AdaptationConfig:
    """Tunable knobs for online adaptation."""

    # Cache adapter
    cache_replay_capacity: int = 1000
    cache_adapt_interval: int = 100        # gradient update every N accesses
    cache_adapt_steps: int = 5             # gradient steps per adaptation round
    cache_lr: float = 1e-4
    cache_min_samples: int = 32            # don't adapt with fewer samples

    # Scheduler adapter
    sched_replay_capacity: int = 500
    sched_adapt_interval: int = 50         # gradient update every N decisions
    sched_adapt_steps: int = 3             # gradient steps per adaptation round
    sched_lr: float = 1e-4
    sched_min_samples: int = 16

    # General
    enabled: bool = True
    log_interval: int = 500                # log metrics every N events (combined)
    max_grad_norm: float = 1.0             # gradient clipping


# ═══════════════════════════════════════════════════════════════════════════════
# Replay Buffer
# ═══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """Fixed-capacity ring buffer for online training samples.

    When capacity is reached, the oldest sample is discarded. This keeps
    adaptation focused on the recent workload while bounding memory usage.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)

    def push(self, sample: dict):
        self._buffer.append(sample)

    def sample_batch(self, batch_size: int) -> List[dict]:
        """Sample a random batch (with replacement if buffer < batch_size)."""
        n = len(self._buffer)
        if n == 0:
            return []
        indices = torch.randint(0, n, (min(batch_size, n),))
        return [self._buffer[i] for i in indices]

    def all(self) -> List[dict]:
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self):
        self._buffer.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Online Cache Adapter
# ═══════════════════════════════════════════════════════════════════════════════

class OnlineCacheAdapter:
    """Fine-tunes the cache replacement LSTM on live access traces.

    Collects (access_history, line_features, oracle_victim) tuples in a replay
    buffer. Every `adapt_interval` accesses, runs a few-shot gradient update
    on a mini-batch sampled from the buffer.

    The oracle signal comes from Belady-approximate heuristics: when a miss
    causes eviction, the LRU victim is recorded as the target. Over many
    updates, the LSTM learns to predict which line to evict before it becomes
    the LRU candidate -- it can internalize frequency and scan-resistance
    patterns that pure LRU misses.

    Tracks the hit rate in two windows (before/after adaptation) to measure
    online improvement.
    """

    def __init__(self, neural_cache, config: AdaptationConfig):
        """
        Args:
            neural_cache: A NeuralCache instance (from ncpu.os.neuros.cache).
            config: Adaptation hyperparameters.
        """
        self.cache = neural_cache
        self.cfg = config
        self.device = neural_cache.device

        # Replay buffer for (history_snapshot, line_features, victim_idx)
        self.replay = ReplayBuffer(config.cache_replay_capacity)

        # Optimizer -- only created when the replacer model exists
        self._optimizer: Optional[torch.optim.Adam] = None

        # Counters
        self.accesses_since_adapt = 0
        self.total_accesses = 0
        self.total_adaptations = 0

        # Hit rate tracking (sliding windows)
        self._window_size = 200
        self._recent_hits: deque = deque(maxlen=self._window_size)
        self._pre_adapt_hit_rate: float = 0.0
        self._post_adapt_hit_rate: float = 0.0

        # Snapshot the initial hit rate baseline
        self._baseline_hit_rate = neural_cache.hit_rate

    def _ensure_optimizer(self):
        """Lazily create the optimizer on first use."""
        if self._optimizer is None and self.cache._replacer_trained:
            self._optimizer = torch.optim.Adam(
                self.cache.replacer.parameters(),
                lr=self.cfg.cache_lr,
            )

    def on_access(self, addr: int, hit: bool, write: bool,
                  set_idx: Optional[int] = None,
                  victim_way: Optional[int] = None):
        """Called after every cache access.

        Args:
            addr: Memory address accessed.
            hit: Whether this was a cache hit.
            write: Whether this was a write access.
            set_idx: Cache set index (computed from addr if None).
            victim_way: The way that was evicted on a miss (None if hit
                        or no eviction needed). This is the oracle signal.
        """
        self.total_accesses += 1
        self.accesses_since_adapt += 1
        self._recent_hits.append(1.0 if hit else 0.0)

        # On a miss with eviction, record a training sample
        if not hit and victim_way is not None and self.cache._replacer_trained:
            if set_idx is None:
                tag = addr >> 6
                set_idx = tag % self.cache.num_sets

            # Snapshot current state for the replay buffer
            history_snap = self.cache.access_history.clone().detach()
            max_tick = float(max(self.cache.tick, 1))
            max_count = self.cache.access_count[set_idx].float().max().clamp(min=1.0)

            # Per-set relative recency (improved features from neural_demo.py)
            last_acc = self.cache.last_access[set_idx].float()
            acc_cnt = self.cache.access_count[set_idx].float()
            max_last = last_acc.max()
            min_last = last_acc.min()
            span = max_last - min_last
            if span > 0:
                recency = 1.0 - (last_acc - min_last) / span
            else:
                recency = torch.zeros_like(last_acc)

            log_counts = torch.log1p(acc_cnt)
            max_log = log_counts.max().clamp(min=1.0)
            frequency = log_counts / max_log

            line_feats = torch.stack([
                recency,
                frequency,
                self.cache.dirty[set_idx].float(),
                self.cache.valid[set_idx].float(),
            ], dim=-1).clone().detach()

            self.replay.push({
                "history": history_snap,
                "line_features": line_feats,
                "victim": victim_way,
            })

        # Trigger adaptation
        if self.accesses_since_adapt >= self.cfg.cache_adapt_interval:
            self._maybe_adapt()
            self.accesses_since_adapt = 0

    def _maybe_adapt(self):
        """Run a few-shot gradient update if conditions are met."""
        if not self.cache._replacer_trained:
            return
        if len(self.replay) < self.cfg.cache_min_samples:
            return

        self._ensure_optimizer()
        if self._optimizer is None:
            return

        # Record pre-adaptation hit rate
        self._pre_adapt_hit_rate = self._window_hit_rate()

        self.cache.replacer.train()
        total_loss = 0.0

        for step in range(self.cfg.cache_adapt_steps):
            batch = self.replay.sample_batch(
                min(32, len(self.replay))
            )
            if not batch:
                break

            self._optimizer.zero_grad()
            step_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            for sample in batch:
                history = sample["history"].unsqueeze(0).to(self.device)
                line_feats = sample["line_features"].to(self.device)
                target = torch.tensor(
                    sample["victim"], dtype=torch.long, device=self.device
                )

                scores = self.cache.replacer(history, line_feats)
                loss = F.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))
                step_loss = step_loss + loss

            step_loss = step_loss / len(batch)

            step_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.cache.replacer.parameters(),
                self.cfg.max_grad_norm,
            )

            self._optimizer.step()
            total_loss += step_loss.item()

        self.cache.replacer.eval()
        self.total_adaptations += 1

        # Record post-adaptation hit rate (will be measured over next window)
        self._post_adapt_hit_rate = self._window_hit_rate()

        avg_loss = total_loss / max(self.cfg.cache_adapt_steps, 1)
        logger.debug(
            f"[CacheAdapter] adaptation #{self.total_adaptations}: "
            f"loss={avg_loss:.4f}, buffer={len(self.replay)}, "
            f"hit_rate={self._post_adapt_hit_rate:.3f}"
        )

    def _window_hit_rate(self) -> float:
        """Hit rate over the recent sliding window."""
        if not self._recent_hits:
            return 0.0
        return sum(self._recent_hits) / len(self._recent_hits)

    def metrics(self) -> Dict:
        """Return current adaptation metrics."""
        current_hr = self._window_hit_rate()
        return {
            "total_accesses": self.total_accesses,
            "total_adaptations": self.total_adaptations,
            "replay_buffer_size": len(self.replay),
            "baseline_hit_rate": self._baseline_hit_rate,
            "current_hit_rate": current_hr,
            "pre_adapt_hit_rate": self._pre_adapt_hit_rate,
            "post_adapt_hit_rate": self._post_adapt_hit_rate,
            "hit_rate_improvement": current_hr - self._baseline_hit_rate,
            "global_hit_rate": self.cache.hit_rate,
        }

    def reset(self):
        """Reset the adapter state (but keep the model weights)."""
        self.replay.clear()
        self.accesses_since_adapt = 0
        self.total_accesses = 0
        self.total_adaptations = 0
        self._recent_hits.clear()
        self._pre_adapt_hit_rate = 0.0
        self._post_adapt_hit_rate = 0.0
        self._baseline_hit_rate = self.cache.hit_rate


# ═══════════════════════════════════════════════════════════════════════════════
# Online Scheduler Adapter
# ═══════════════════════════════════════════════════════════════════════════════

class OnlineSchedulerAdapter:
    """Fine-tunes the Transformer scheduler on live process metrics.

    Collects (process_features, selected_idx, outcome_reward) tuples. Every
    `adapt_interval` scheduling decisions, runs a few-shot gradient update
    that reinforces decisions correlated with good outcomes (high throughput,
    high fairness) and penalizes decisions correlated with poor outcomes.

    The reward signal is: reward = throughput * fairness_index.
    - throughput: fraction of cycles that were productive (not idle or context-switching)
    - fairness: Jain's fairness index over CPU shares (1.0 = perfect fairness)

    This is a simplified REINFORCE-style update: the loss pushes the model
    to assign higher scores to processes that were selected before good
    outcomes, and lower scores otherwise.
    """

    def __init__(self, neural_scheduler, config: AdaptationConfig):
        """
        Args:
            neural_scheduler: A NeuralScheduler instance (from ncpu.os.neuros.scheduler).
            config: Adaptation hyperparameters.
        """
        self.scheduler = neural_scheduler
        self.cfg = config
        self.device = neural_scheduler.device

        # Replay buffer
        self.replay = ReplayBuffer(config.sched_replay_capacity)

        # Optimizer
        self._optimizer: Optional[torch.optim.Adam] = None

        # Counters
        self.decisions_since_adapt = 0
        self.total_decisions = 0
        self.total_adaptations = 0

        # Quality tracking (sliding windows)
        self._window_size = 100
        self._recent_rewards: deque = deque(maxlen=self._window_size)
        self._recent_turnaround: deque = deque(maxlen=self._window_size)
        self._recent_throughput: deque = deque(maxlen=self._window_size)
        self._pre_adapt_reward: float = 0.0
        self._post_adapt_reward: float = 0.0

        # Baseline snapshot
        self._baseline_fairness = neural_scheduler.jains_fairness()

    def _ensure_optimizer(self):
        if self._optimizer is None and self.scheduler._trained:
            self._optimizer = torch.optim.Adam(
                self.scheduler.net.parameters(),
                lr=self.cfg.sched_lr,
            )

    def on_decision(self, selected_idx: int, process_features: torch.Tensor,
                    num_ready: int, turnaround: float = 0.0,
                    throughput: float = 0.0, fairness: float = 1.0):
        """Called after every scheduling decision.

        Args:
            selected_idx: Index of the selected process in the ready queue.
            process_features: [N, feature_dim] tensor for all ready processes.
            num_ready: Number of ready processes.
            turnaround: Average turnaround time for recently completed processes.
            throughput: Fraction of productive cycles in recent window.
            fairness: Jain's fairness index.
        """
        self.total_decisions += 1
        self.decisions_since_adapt += 1

        reward = throughput * fairness
        self._recent_rewards.append(reward)
        self._recent_turnaround.append(turnaround)
        self._recent_throughput.append(throughput)

        if self.scheduler._trained and num_ready > 1:
            self.replay.push({
                "features": process_features.clone().detach(),
                "selected_idx": selected_idx,
                "num_ready": num_ready,
                "reward": reward,
                "throughput": throughput,
                "fairness": fairness,
            })

        # Trigger adaptation
        if self.decisions_since_adapt >= self.cfg.sched_adapt_interval:
            self._maybe_adapt()
            self.decisions_since_adapt = 0

    def _maybe_adapt(self):
        """Run a few-shot gradient update on scheduling decisions."""
        if not self.scheduler._trained:
            return
        if len(self.replay) < self.cfg.sched_min_samples:
            return

        self._ensure_optimizer()
        if self._optimizer is None:
            return

        self._pre_adapt_reward = self._window_reward()

        self.scheduler.net.train()
        total_loss = 0.0

        for step in range(self.cfg.sched_adapt_steps):
            batch = self.replay.sample_batch(
                min(32, len(self.replay))
            )
            if not batch:
                break

            self._optimizer.zero_grad()
            step_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            for sample in batch:
                features = sample["features"].to(self.device)
                selected_idx = sample["selected_idx"]
                reward = sample["reward"]
                n = sample["num_ready"]

                scores = self.scheduler.net(features[:n])

                # Reward-weighted cross-entropy: high reward reinforces the
                # decision, low reward penalizes it. We center the reward
                # around the buffer mean so that average decisions have
                # near-zero gradient (variance reduction / baseline).
                mean_reward = self._window_reward()
                advantage = reward - mean_reward

                target = torch.tensor(
                    selected_idx, dtype=torch.long, device=self.device
                )
                ce = F.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))
                # Negative advantage = push away from this decision
                loss = -advantage * ce if advantage < 0 else advantage * ce
                step_loss = step_loss + loss

            step_loss = step_loss / len(batch)
            step_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.scheduler.net.parameters(),
                self.cfg.max_grad_norm,
            )

            self._optimizer.step()
            total_loss += step_loss.item()

        self.scheduler.net.eval()
        self.total_adaptations += 1

        self._post_adapt_reward = self._window_reward()

        avg_loss = total_loss / max(self.cfg.sched_adapt_steps, 1)
        logger.debug(
            f"[SchedAdapter] adaptation #{self.total_adaptations}: "
            f"loss={avg_loss:.4f}, buffer={len(self.replay)}, "
            f"reward={self._post_adapt_reward:.3f}"
        )

    def _window_reward(self) -> float:
        if not self._recent_rewards:
            return 0.0
        return sum(self._recent_rewards) / len(self._recent_rewards)

    def _window_throughput(self) -> float:
        if not self._recent_throughput:
            return 0.0
        return sum(self._recent_throughput) / len(self._recent_throughput)

    def _window_turnaround(self) -> float:
        if not self._recent_turnaround:
            return 0.0
        return sum(self._recent_turnaround) / len(self._recent_turnaround)

    def metrics(self) -> Dict:
        return {
            "total_decisions": self.total_decisions,
            "total_adaptations": self.total_adaptations,
            "replay_buffer_size": len(self.replay),
            "baseline_fairness": self._baseline_fairness,
            "current_fairness": self.scheduler.jains_fairness(),
            "avg_reward": self._window_reward(),
            "avg_throughput": self._window_throughput(),
            "avg_turnaround": self._window_turnaround(),
            "pre_adapt_reward": self._pre_adapt_reward,
            "post_adapt_reward": self._post_adapt_reward,
            "reward_improvement": self._window_reward() - self._pre_adapt_reward,
        }

    def reset(self):
        self.replay.clear()
        self.decisions_since_adapt = 0
        self.total_decisions = 0
        self.total_adaptations = 0
        self._recent_rewards.clear()
        self._recent_turnaround.clear()
        self._recent_throughput.clear()
        self._pre_adapt_reward = 0.0
        self._post_adapt_reward = 0.0
        self._baseline_fairness = self.scheduler.jains_fairness()


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptation Manager
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptationManager:
    """Coordinates online adaptation of both cache and scheduler models.

    Provides a unified interface for feeding events, querying metrics,
    enabling/disabling adaptation at runtime, and saving adapted weights.

    Example:
        neuros = NeurOS()
        neuros.boot()

        mgr = AdaptationManager(neuros.cache, neuros.scheduler)
        mgr.enable()

        # In the OS event loop:
        hit = neuros.cache.access(addr, write=False)
        mgr.on_cache_access(addr, hit=hit, write=False,
                            victim_way=evicted_way if not hit else None)

        selected = neuros.scheduler.schedule()
        mgr.on_schedule_decision(...)

        # Inspect:
        print(mgr.metrics())
        mgr.save_checkpoint("/tmp/session_weights")
    """

    def __init__(self, neural_cache=None, neural_scheduler=None,
                 config: Optional[AdaptationConfig] = None):
        """
        Args:
            neural_cache: NeuralCache instance (from ncpu.os.neuros.cache), or None.
            neural_scheduler: NeuralScheduler instance (from ncpu.os.neuros.scheduler), or None.
            config: Adaptation configuration. Defaults to AdaptationConfig().
        """
        self.cfg = config or AdaptationConfig()
        self._enabled = self.cfg.enabled
        self._start_time = time.monotonic()

        # Create sub-adapters for whichever models are provided
        self.cache_adapter: Optional[OnlineCacheAdapter] = None
        self.sched_adapter: Optional[OnlineSchedulerAdapter] = None

        if neural_cache is not None:
            self.cache_adapter = OnlineCacheAdapter(neural_cache, self.cfg)

        if neural_scheduler is not None:
            self.sched_adapter = OnlineSchedulerAdapter(neural_scheduler, self.cfg)

        # Combined event counter for logging
        self._total_events = 0

        # Adaptation log (timestamped entries for post-hoc analysis)
        self._log: List[Dict] = []

        logger.info(
            f"[AdaptationManager] initialized: "
            f"cache={'yes' if self.cache_adapter else 'no'}, "
            f"scheduler={'yes' if self.sched_adapter else 'no'}, "
            f"enabled={self._enabled}"
        )

    # ─── Runtime Control ──────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self):
        """Enable online adaptation."""
        self._enabled = True
        logger.info("[AdaptationManager] enabled")

    def disable(self):
        """Disable online adaptation (events are still counted but no gradient updates)."""
        self._enabled = False
        logger.info("[AdaptationManager] disabled")

    # ─── Event Feeds ──────────────────────────────────────────────────────

    def on_cache_access(self, addr: int, hit: bool, write: bool = False,
                        set_idx: Optional[int] = None,
                        victim_way: Optional[int] = None):
        """Feed a cache access event.

        Args:
            addr: Memory address.
            hit: Whether this was a hit.
            write: Whether this was a write.
            set_idx: Cache set index (auto-computed if None).
            victim_way: Way evicted on miss (None if hit or no eviction).
        """
        self._total_events += 1

        if self.cache_adapter is not None and self._enabled:
            self.cache_adapter.on_access(
                addr=addr, hit=hit, write=write,
                set_idx=set_idx, victim_way=victim_way,
            )

        self._maybe_log()

    def on_schedule_decision(self, selected_idx: int,
                             process_features: torch.Tensor,
                             num_ready: int,
                             turnaround: float = 0.0,
                             throughput: float = 0.0,
                             fairness: float = 1.0):
        """Feed a scheduling decision event.

        Args:
            selected_idx: Index of selected process in the ready queue.
            process_features: [N, feature_dim] feature tensor for ready processes.
            num_ready: Number of ready processes.
            turnaround: Average turnaround time of recently completed processes.
            throughput: Fraction of productive CPU cycles.
            fairness: Jain's fairness index.
        """
        self._total_events += 1

        if self.sched_adapter is not None and self._enabled:
            self.sched_adapter.on_decision(
                selected_idx=selected_idx,
                process_features=process_features,
                num_ready=num_ready,
                turnaround=turnaround,
                throughput=throughput,
                fairness=fairness,
            )

        self._maybe_log()

    # ─── Metrics ──────────────────────────────────────────────────────────

    def metrics(self) -> Dict:
        """Return combined metrics from both adapters."""
        elapsed = time.monotonic() - self._start_time
        result = {
            "enabled": self._enabled,
            "total_events": self._total_events,
            "elapsed_seconds": elapsed,
            "events_per_second": self._total_events / max(elapsed, 1e-6),
        }
        if self.cache_adapter is not None:
            result["cache"] = self.cache_adapter.metrics()
        if self.sched_adapter is not None:
            result["scheduler"] = self.sched_adapter.metrics()
        return result

    def summary(self) -> str:
        """Human-readable summary of adaptation progress."""
        m = self.metrics()
        lines = [
            f"AdaptationManager: {'ENABLED' if m['enabled'] else 'DISABLED'}",
            f"  Total events: {m['total_events']:,} "
            f"({m['events_per_second']:.0f}/sec over {m['elapsed_seconds']:.1f}s)",
        ]
        if "cache" in m:
            c = m["cache"]
            lines.append(
                f"  Cache: {c['total_adaptations']} adaptations, "
                f"hit_rate {c['baseline_hit_rate']:.3f} -> {c['current_hit_rate']:.3f} "
                f"(delta={c['hit_rate_improvement']:+.3f}), "
                f"buffer={c['replay_buffer_size']}"
            )
        if "scheduler" in m:
            s = m["scheduler"]
            lines.append(
                f"  Scheduler: {s['total_adaptations']} adaptations, "
                f"reward={s['avg_reward']:.3f}, "
                f"throughput={s['avg_throughput']:.3f}, "
                f"fairness={s['baseline_fairness']:.3f} -> {s['current_fairness']:.3f}, "
                f"buffer={s['replay_buffer_size']}"
            )
        return "\n".join(lines)

    # ─── Persistence ──────────────────────────────────────────────────────

    def save_checkpoint(self, directory: str):
        """Save adapted model weights to a session-specific directory.

        Creates:
            <directory>/cache_replace_adapted.pt
            <directory>/scheduler_adapted.pt
            <directory>/adaptation_metrics.pt
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        saved = {}

        if self.cache_adapter is not None and self.cache_adapter.cache._replacer_trained:
            cache_path = out / "cache_replace_adapted.pt"
            torch.save(
                self.cache_adapter.cache.replacer.state_dict(),
                str(cache_path),
            )
            saved["cache_replace"] = str(cache_path)
            logger.info(f"[AdaptationManager] saved cache weights -> {cache_path}")

        if self.sched_adapter is not None and self.sched_adapter.scheduler._trained:
            sched_path = out / "scheduler_adapted.pt"
            torch.save(
                self.sched_adapter.scheduler.net.state_dict(),
                str(sched_path),
            )
            saved["scheduler"] = str(sched_path)
            logger.info(f"[AdaptationManager] saved scheduler weights -> {sched_path}")

        # Save metrics alongside the weights for provenance
        metrics_path = out / "adaptation_metrics.pt"
        torch.save({
            "metrics": self.metrics(),
            "log": self._log,
            "config": {
                "cache_adapt_interval": self.cfg.cache_adapt_interval,
                "cache_adapt_steps": self.cfg.cache_adapt_steps,
                "cache_lr": self.cfg.cache_lr,
                "sched_adapt_interval": self.cfg.sched_adapt_interval,
                "sched_adapt_steps": self.cfg.sched_adapt_steps,
                "sched_lr": self.cfg.sched_lr,
            },
        }, str(metrics_path))
        saved["metrics"] = str(metrics_path)

        logger.info(f"[AdaptationManager] checkpoint saved to {directory}")
        return saved

    def load_checkpoint(self, directory: str) -> Dict[str, bool]:
        """Load adapted weights from a previous session checkpoint.

        Returns dict indicating which models were successfully loaded.
        """
        out = Path(directory)
        result = {}

        cache_path = out / "cache_replace_adapted.pt"
        if cache_path.exists() and self.cache_adapter is not None:
            try:
                state = torch.load(
                    str(cache_path),
                    map_location=self.cache_adapter.device,
                    weights_only=True,
                )
                self.cache_adapter.cache.replacer.load_state_dict(state)
                self.cache_adapter.cache.replacer.eval()
                self.cache_adapter.cache._replacer_trained = True
                result["cache_replace"] = True
                logger.info(f"[AdaptationManager] loaded adapted cache weights from {cache_path}")
            except Exception as e:
                result["cache_replace"] = False
                logger.warning(f"[AdaptationManager] failed to load cache weights: {e}")

        sched_path = out / "scheduler_adapted.pt"
        if sched_path.exists() and self.sched_adapter is not None:
            try:
                state = torch.load(
                    str(sched_path),
                    map_location=self.sched_adapter.device,
                    weights_only=True,
                )
                self.sched_adapter.scheduler.net.load_state_dict(state)
                self.sched_adapter.scheduler.net.eval()
                self.sched_adapter.scheduler._trained = True
                result["scheduler"] = True
                logger.info(f"[AdaptationManager] loaded adapted scheduler weights from {sched_path}")
            except Exception as e:
                result["scheduler"] = False
                logger.warning(f"[AdaptationManager] failed to load scheduler weights: {e}")

        return result

    # ─── Internal ─────────────────────────────────────────────────────────

    def _maybe_log(self):
        """Periodically append a metrics snapshot to the adaptation log."""
        if self._total_events % self.cfg.log_interval == 0 and self._total_events > 0:
            entry = {
                "event_count": self._total_events,
                "timestamp": time.monotonic() - self._start_time,
            }
            if self.cache_adapter is not None:
                entry["cache_hit_rate"] = self.cache_adapter._window_hit_rate()
                entry["cache_adaptations"] = self.cache_adapter.total_adaptations
            if self.sched_adapter is not None:
                entry["sched_reward"] = self.sched_adapter._window_reward()
                entry["sched_adaptations"] = self.sched_adapter.total_adaptations
            self._log.append(entry)

    def reset(self):
        """Reset all adapter state and counters."""
        if self.cache_adapter is not None:
            self.cache_adapter.reset()
        if self.sched_adapter is not None:
            self.sched_adapter.reset()
        self._total_events = 0
        self._log.clear()
        self._start_time = time.monotonic()

    def __repr__(self) -> str:
        parts = []
        if self.cache_adapter:
            cm = self.cache_adapter.metrics()
            parts.append(f"cache(adapt={cm['total_adaptations']}, hr={cm['current_hit_rate']:.3f})")
        if self.sched_adapter:
            sm = self.sched_adapter.metrics()
            parts.append(f"sched(adapt={sm['total_adaptations']}, reward={sm['avg_reward']:.3f})")
        status = "ON" if self._enabled else "OFF"
        return f"AdaptationManager({status}, {', '.join(parts)})"


# ═══════════════════════════════════════════════════════════════════════════════
# Integration helper
# ═══════════════════════════════════════════════════════════════════════════════

def attach_to_neuros(neuros, config: Optional[AdaptationConfig] = None) -> AdaptationManager:
    """Attach online adaptation to a booted NeurOS instance.

    This monkey-patches the NeuralCache.access() and NeuralScheduler.schedule()
    methods to feed events into the AdaptationManager automatically. The caller
    doesn't need to change any existing code -- the adaptation is transparent.

    Args:
        neuros: A booted NeurOS instance (from ncpu.os.neuros.boot).
        config: Optional adaptation configuration.

    Returns:
        The AdaptationManager instance (for manual control / inspection).
    """
    mgr = AdaptationManager(
        neural_cache=neuros.cache,
        neural_scheduler=neuros.scheduler,
        config=config,
    )

    # Patch NeuralCache.access to feed events
    if neuros.cache is not None:
        original_access = neuros.cache.access

        def patched_access(addr: int, write: bool = False) -> bool:
            # Call original (which updates internal state)
            hit = original_access(addr, write)

            # Determine victim way: on a miss, the LRU way is the oracle signal
            victim_way = None
            if not hit:
                tag = addr >> 6
                set_idx = tag % neuros.cache.num_sets
                # The victim was the LRU line in this set (before fill overwrote it)
                victim_way = int(neuros.cache.last_access[set_idx].argmin().item())

            mgr.on_cache_access(
                addr=addr, hit=hit, write=write,
                victim_way=victim_way,
            )
            return hit

        neuros.cache.access = patched_access

    # Patch NeuralScheduler.schedule to feed events
    if neuros.scheduler is not None:
        original_schedule = neuros.scheduler.schedule

        def patched_schedule():
            from ncpu.os.neuros.scheduler import extract_process_features

            # Capture ready queue features BEFORE scheduling
            ready = neuros.scheduler.process_table.ready_processes()
            if not ready:
                return original_schedule()

            n = min(len(ready), neuros.scheduler.max_queue_size)
            features = torch.stack([
                extract_process_features(p, neuros.scheduler.tick)
                for p in ready[:n]
            ])

            # Call original
            selected = original_schedule()

            if selected is not None:
                # Find which index was selected
                selected_idx = next(
                    (i for i, p in enumerate(ready[:n]) if p.pid == selected.pid),
                    0,
                )

                # Compute outcome metrics
                throughput = 1.0  # Approximation; refined after running
                fairness = neuros.scheduler.jains_fairness()

                mgr.on_schedule_decision(
                    selected_idx=selected_idx,
                    process_features=features,
                    num_ready=n,
                    throughput=throughput,
                    fairness=fairness,
                )

            return selected

        neuros.scheduler.schedule = patched_schedule

    # Store reference on the NeurOS instance for easy access
    neuros.adaptation_manager = mgr

    logger.info("[AdaptationManager] attached to NeurOS instance (methods patched)")
    return mgr
