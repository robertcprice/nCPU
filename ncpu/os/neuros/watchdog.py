"""Neural Watchdog.

LSTM-based anomaly detector for neurOS system health monitoring.
Monitors eight system metrics through a GPU-resident ring buffer and
runs a trained neural network to detect anomalous operating conditions.

When untrained, falls back to simple threshold-based heuristics.
After training on normal-operation baselines, the LSTM learns temporal
patterns that distinguish healthy fluctuations from genuine anomalies.

Architecture:
    Input:  [1, window_size, 8] — sliding window of system metrics
    LSTM:   hidden_size=32, 1 layer
    Scorer: Linear(32, 16) → ReLU → Linear(16, 1) → Sigmoid
    Output: anomaly score 0-1 (higher = more anomalous)

Monitored metrics:
    cpu_util, mem_pressure, interrupt_rate, cache_hit_rate,
    scheduler_fairness, ipc_queue_depth, fs_ops_rate, tlb_miss_rate
"""

import torch
import torch.nn as nn
import time
import logging
from typing import Optional, List, Dict
from pathlib import Path

from .device import default_device

logger = logging.getLogger(__name__)

NUM_METRICS = 8


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Anomaly Detector
# ═══════════════════════════════════════════════════════════════════════════════

class WatchdogNet(nn.Module):
    """LSTM-based anomaly scorer for system health metrics.

    Processes a sliding window of system metrics and produces a scalar
    anomaly score. Trained on normal-operation baselines so that
    deviations from learned patterns produce high scores.

    Architecture:
        [1, seq_len, 8] → LSTM(8, 32) → Linear(32, 16) → ReLU → Linear(16, 1) → Sigmoid
    """

    def __init__(self, input_size: int = NUM_METRICS, hidden_size: int = 32,
                 num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, metrics_window: torch.Tensor) -> torch.Tensor:
        """Score a metrics window for anomalies.

        Args:
            metrics_window: [batch, seq_len, 8] or [1, seq_len, 8]

        Returns:
            anomaly_scores: [batch] float in [0, 1]
        """
        _, (h_n, _) = self.lstm(metrics_window)
        hidden = h_n[-1]  # [batch, hidden_size]
        scores = self.scorer(hidden).squeeze(-1)  # [batch]
        return scores


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Watchdog
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralWatchdog:
    """GPU-native system health monitor with neural anomaly detection.

    Maintains a GPU-resident ring buffer of system metrics and uses a
    trained LSTM to detect anomalous operating conditions. When the
    anomaly score exceeds the configured threshold, an alert is raised.

    Falls back to simple threshold heuristics when untrained.

    Usage:
        wd = NeuralWatchdog()
        wd.record_metrics(cpu_util=0.8, mem_pressure=0.5, ...)
        alert = wd.check()
        if alert:
            print(f"Anomaly detected: score={alert['score']:.3f}")
    """

    def __init__(self, window_size: int = 64, anomaly_threshold: float = 0.8,
                 device: Optional[torch.device] = None):
        self.device = device or default_device()
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold

        # Neural anomaly detector
        self.net = WatchdogNet().to(self.device)
        self._trained = False

        # GPU-resident ring buffer: [window_size, 8]
        self.metrics_buffer = torch.zeros(
            window_size, NUM_METRICS, dtype=torch.float32, device=self.device
        )
        self.buffer_ptr = 0

        # Alert history
        self.alerts: List[Dict] = []
        self.total_checks = 0
        self.total_alerts = 0

    def record_metrics(self, cpu_util: float, mem_pressure: float,
                       interrupt_rate: float, cache_hit_rate: float,
                       scheduler_fairness: float, ipc_queue_depth: float,
                       fs_ops_rate: float, tlb_miss_rate: float):
        """Write a metrics sample into the ring buffer.

        All values should be normalized to [0, 1] or comparable scales.

        Args:
            cpu_util: CPU utilization (0=idle, 1=fully loaded)
            mem_pressure: Memory pressure (0=free, 1=exhausted)
            interrupt_rate: Interrupts per tick (normalized)
            cache_hit_rate: Cache hit ratio (0=all misses, 1=all hits)
            scheduler_fairness: Jain's fairness index (0-1)
            ipc_queue_depth: Normalized IPC queue occupancy
            fs_ops_rate: Filesystem operations per tick (normalized)
            tlb_miss_rate: TLB miss ratio (0=all hits, 1=all misses)
        """
        self.metrics_buffer[self.buffer_ptr] = torch.tensor([
            cpu_util, mem_pressure, interrupt_rate, cache_hit_rate,
            scheduler_fairness, ipc_queue_depth, fs_ops_rate, tlb_miss_rate,
        ], dtype=torch.float32, device=self.device)
        self.buffer_ptr = (self.buffer_ptr + 1) % self.window_size

    def check(self) -> Optional[Dict]:
        """Run anomaly detection on the current metrics window.

        Returns an alert dict if the anomaly score exceeds the threshold,
        or None if the system appears healthy.

        Returns:
            Alert dict with keys {tick, score, metrics_snapshot, timestamp}
            or None if no anomaly detected.
        """
        self.total_checks += 1

        if self._trained:
            score = self._neural_check()
        else:
            score = self._heuristic_check()

        if score > self.anomaly_threshold:
            alert = {
                "tick": self.total_checks,
                "score": float(score),
                "metrics_snapshot": self.metrics_buffer.clone(),
                "timestamp": time.time(),
            }
            self.alerts.append(alert)
            self.total_alerts += 1
            logger.warning(f"[Watchdog] Anomaly detected: score={score:.3f}")
            return alert

        return None

    def _neural_check(self) -> float:
        """Run the trained LSTM on the metrics window."""
        window = self.metrics_buffer.unsqueeze(0)  # [1, window_size, 8]
        with torch.no_grad():
            score = self.net(window)
        return float(score.item())

    def _heuristic_check(self) -> float:
        """Simple threshold-based fallback when untrained.

        Returns a score in [0, 1] based on hard-coded danger thresholds.
        """
        latest_idx = (self.buffer_ptr - 1) % self.window_size
        latest = self.metrics_buffer[latest_idx]

        cpu_util = float(latest[0].item())
        mem_pressure = float(latest[1].item())
        cache_hit_rate = float(latest[3].item())

        if cpu_util > 0.95 or mem_pressure > 0.9 or cache_hit_rate < 0.3:
            return 1.0
        return 0.0

    def collect_from_os(self, nos):
        """Extract all 8 metrics from a live NeurOS instance.

        Reads component state from the NeurOS object and records
        a metrics sample into the ring buffer.

        Args:
            nos: A booted NeurOS instance.
        """
        # CPU utilization: 1.0 if a process is running, 0.0 if idle
        cpu_util = 1.0 if nos.scheduler.current_pid is not None else 0.0

        # Memory pressure: fraction of physical frames allocated
        allocated = int(nos.mmu.frame_bitmap.sum().item())
        mem_pressure = allocated / max(nos.mmu.max_physical_frames, 1)

        # Interrupt rate: dispatched per tick
        gic_tick = max(nos.gic.interrupts_raised, 1)
        interrupt_rate = nos.gic.interrupts_handled / gic_tick

        # Cache hit rate
        cache_total = nos.cache.hits + nos.cache.misses
        cache_hit_rate = nos.cache.hits / max(cache_total, 1)

        # Scheduler fairness (Jain's index)
        scheduler_fairness = nos.scheduler.jains_fairness()

        # IPC queue depth: average queue occupancy across registered processes
        queues = nos.ipc._queues
        if queues:
            total_queued = sum(q.count for q in queues.values())
            max_capacity = sum(q.capacity for q in queues.values())
            ipc_queue_depth = total_queued / max(max_capacity, 1)
        else:
            ipc_queue_depth = 0.0

        # Filesystem operations rate
        fs_total = nos.fs.reads + nos.fs.writes
        fs_tick = max(nos.fs.tick, 1)
        fs_ops_rate = fs_total / fs_tick

        # TLB miss rate
        tlb_total = nos.tlb.hits + nos.tlb.misses
        tlb_miss_rate = nos.tlb.misses / max(tlb_total, 1)

        self.record_metrics(
            cpu_util=cpu_util,
            mem_pressure=mem_pressure,
            interrupt_rate=interrupt_rate,
            cache_hit_rate=cache_hit_rate,
            scheduler_fairness=scheduler_fairness,
            ipc_queue_depth=ipc_queue_depth,
            fs_ops_rate=fs_ops_rate,
            tlb_miss_rate=tlb_miss_rate,
        )

    # ─── Training ──────────────────────────────────────────────────────────

    def train_baseline(self, normal_data: torch.Tensor,
                       epochs: int = 50, lr: float = 1e-3) -> Dict:
        """Train on normal-operation metrics to establish a healthy baseline.

        The network learns to output low anomaly scores for windows that
        resemble normal operation. At inference time, deviations from the
        learned distribution produce high scores.

        Args:
            normal_data: [N, window_size, 8] — windows of normal metrics
            epochs: Training epochs
            lr: Learning rate

        Returns:
            Training statistics dict.
        """
        normal_data = normal_data.to(self.device)
        n_samples = normal_data.shape[0]
        targets = torch.zeros(n_samples, dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        self.net.train()
        final_loss = 0.0

        for epoch in range(epochs):
            optimizer.zero_grad()
            scores = self.net(normal_data)
            loss = loss_fn(scores, targets)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        self.net.eval()
        self._trained = True

        logger.info(f"[Watchdog] Trained on {n_samples} normal windows, "
                    f"final_loss={final_loss:.6f}")

        return {
            "epochs": epochs,
            "samples": n_samples,
            "final_loss": final_loss,
        }

    # ─── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str = "models/research/neuros/watchdog.pt"):
        """Save the trained watchdog model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def load(self, path: str = "models/research/neuros/watchdog.pt") -> bool:
        """Load a trained watchdog model.

        Returns:
            True if the model was loaded successfully.
        """
        if not Path(path).exists():
            return False
        self.net.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.net.eval()
        self._trained = True
        return True

    # ─── Diagnostics ───────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Return watchdog statistics."""
        recent = self.alerts[-5:] if self.alerts else []
        recent_summary = [
            {"tick": a["tick"], "score": a["score"], "timestamp": a["timestamp"]}
            for a in recent
        ]
        return {
            "total_checks": self.total_checks,
            "total_alerts": self.total_alerts,
            "alert_rate": self.total_alerts / max(self.total_checks, 1),
            "anomaly_threshold": self.anomaly_threshold,
            "trained": self._trained,
            "window_size": self.window_size,
            "buffer_ptr": self.buffer_ptr,
            "recent_alerts": recent_summary,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (f"NeuralWatchdog(checks={s['total_checks']}, "
                f"alerts={s['total_alerts']}, "
                f"threshold={s['anomaly_threshold']}, "
                f"policy={'neural' if s['trained'] else 'heuristic'})")
