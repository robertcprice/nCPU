"""Neural Security Monitor — LSTM autoencoder for syscall anomaly detection.

Learns the "normal" distribution of syscall sequences via reconstruction.
Normal sequences reconstruct with low error; anomalous sequences (shell injection,
privilege escalation, fork bombs, etc.) produce high reconstruction error that
triggers alerts.

This complements the bigram-based monitor in ncpu/os/gpu/neural_security.py
and the next-syscall predictor in train_security_model.py with a fundamentally
different detection mechanism: sequence RECONSTRUCTION rather than prediction.
The autoencoder sees the ENTIRE window at once, capturing global structure that
sequential predictors miss.

Architecture (~85K params, ~340 KB):
    Encoder: Embedding(512, 32) -> LSTM(32, 64, num_layers=2) -> Linear(64, 32)
    Decoder: Linear(32, 64) -> LSTM(64, 64, num_layers=2) -> Linear(64, 512)
    Loss: CrossEntropy between input syscall sequence and reconstructed logits
    Anomaly score: mean per-step cross-entropy (high = anomalous)

Integration with NeuralWatchdog:
    The security monitor plugs into the existing watchdog infrastructure in
    metal_neural_os.py. When the watchdog detects a metrics anomaly AND the
    security monitor detects a syscall anomaly, confidence is elevated.

Usage:
    from ncpu.neural.neural_security_monitor import NeuralSecurityMonitor

    monitor = NeuralSecurityMonitor()
    monitor.load()  # load pretrained autoencoder

    # Online monitoring — feed syscalls one at a time
    for syscall_num in syscall_stream:
        result = monitor.observe(syscall_num)
        if result and result["is_anomalous"]:
            print(f"ALERT: {result['anomaly_score']:.3f} — {result['suspicious_subsequence']}")

    # Batch scoring
    scores = monitor.score_trace([63, 64, 220, 220, 220, 220, 93])
"""

from __future__ import annotations

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

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "os" / "security_monitor.pt"

# ── Syscall constants ─────────────────────────────────────────────────────

N_SYSCALLS = 512  # Embedding vocab size — covers Linux ARM64 syscall range

SYSCALL_NAMES = {
    17: "GETCWD", 24: "DUP3", 25: "FCNTL", 29: "IOCTL",
    49: "CHDIR", 56: "OPENAT", 57: "CLOSE", 59: "PIPE2",
    61: "GETDENTS", 62: "LSEEK", 63: "READ", 64: "WRITE",
    80: "FSTAT", 93: "EXIT", 94: "EXIT_GROUP",
    129: "KILL", 172: "GETPID", 214: "BRK", 215: "MUNMAP",
    220: "CLONE", 222: "MMAP", 260: "WAIT4",
    300: "COMPILE", 301: "EXEC",
}

# Syscalls that are inherently sensitive — boost anomaly score when present
SENSITIVE_SYSCALLS = {220, 129, 301}  # CLONE, KILL, EXEC


# ═══════════════════════════════════════════════════════════════════════════════
# LSTM Autoencoder
# ═══════════════════════════════════════════════════════════════════════════════

class SyscallEncoder(nn.Module):
    """Encode a syscall sequence into a fixed-size latent vector.

    Embedding(512, 32) -> LSTM(32, 64, 2 layers) -> Linear(64, 32)
    Takes the final hidden state of the LSTM as the sequence summary.
    """

    def __init__(self, n_syscalls: int = N_SYSCALLS, embed_dim: int = 32,
                 hidden_dim: int = 64, latent_dim: int = 32, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(n_syscalls, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1,
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, syscall_seq: torch.Tensor) -> torch.Tensor:
        """Encode a batch of syscall sequences to latent vectors.

        Args:
            syscall_seq: [batch, seq_len] int64 syscall numbers

        Returns:
            latent: [batch, latent_dim] float32
        """
        embedded = self.embed(syscall_seq)           # [B, T, 32]
        _, (h_n, _) = self.lstm(embedded)            # h_n: [n_layers, B, 64]
        hidden = h_n[-1]                              # [B, 64] — last layer
        return self.to_latent(hidden)                 # [B, 32]


class SyscallDecoder(nn.Module):
    """Decode a latent vector back into syscall sequence logits.

    Linear(32, 64) -> LSTM(64, 64, 2 layers) -> Linear(64, 512)
    The latent vector is expanded and repeated across the target sequence length,
    then the LSTM reconstructs temporal structure.
    """

    def __init__(self, n_syscalls: int = N_SYSCALLS, hidden_dim: int = 64,
                 latent_dim: int = 32, n_layers: int = 2):
        super().__init__()
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1,
        )
        self.to_logits = nn.Linear(hidden_dim, n_syscalls)

    def forward(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode latent vector to syscall logits.

        Args:
            latent: [batch, latent_dim] float32
            seq_len: target sequence length

        Returns:
            logits: [batch, seq_len, n_syscalls] float32
        """
        expanded = self.from_latent(latent)           # [B, 64]
        # Repeat across timesteps as decoder input
        expanded = expanded.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, 64]
        decoded, _ = self.lstm(expanded)              # [B, T, 64]
        return self.to_logits(decoded)                # [B, T, 512]


class SyscallAutoencoder(nn.Module):
    """LSTM autoencoder for syscall sequence anomaly detection.

    Compresses a window of syscall numbers into a latent vector via the encoder,
    then reconstructs the original sequence via the decoder. Normal sequences
    (seen during training) reconstruct with low cross-entropy; anomalous
    sequences produce high reconstruction error.

    Architecture (~85K params):
        Encoder: Embedding(512, 32) -> LSTM(32, 64, 2) -> Linear(64, 32)
        Decoder: Linear(32, 64) -> LSTM(64, 64, 2) -> Linear(64, 512)
    """

    def __init__(self, n_syscalls: int = N_SYSCALLS, embed_dim: int = 32,
                 hidden_dim: int = 64, latent_dim: int = 32, n_layers: int = 2):
        super().__init__()
        self.n_syscalls = n_syscalls
        self.encoder = SyscallEncoder(n_syscalls, embed_dim, hidden_dim,
                                      latent_dim, n_layers)
        self.decoder = SyscallDecoder(n_syscalls, hidden_dim, latent_dim,
                                      n_layers)

    def forward(self, syscall_seq: torch.Tensor) -> torch.Tensor:
        """Reconstruct a syscall sequence through the bottleneck.

        Args:
            syscall_seq: [batch, seq_len] int64 clamped to [0, n_syscalls-1]

        Returns:
            logits: [batch, seq_len, n_syscalls] reconstruction logits
        """
        latent = self.encoder(syscall_seq)
        logits = self.decoder(latent, syscall_seq.shape[1])
        return logits

    def reconstruction_loss(self, syscall_seq: torch.Tensor) -> torch.Tensor:
        """Compute mean per-step cross-entropy reconstruction loss.

        Args:
            syscall_seq: [batch, seq_len] int64

        Returns:
            loss: scalar — mean cross-entropy across batch and timesteps
        """
        logits = self.forward(syscall_seq)  # [B, T, V]
        B, T, V = logits.shape
        return F.cross_entropy(logits.reshape(B * T, V), syscall_seq.reshape(B * T))

    def anomaly_scores(self, syscall_seq: torch.Tensor) -> torch.Tensor:
        """Compute per-position reconstruction error as anomaly scores.

        Higher score = more anomalous (harder to reconstruct from the
        learned normal distribution).

        Args:
            syscall_seq: [batch, seq_len] int64

        Returns:
            scores: [batch, seq_len] float32 per-position anomaly scores
        """
        with torch.no_grad():
            logits = self.forward(syscall_seq)  # [B, T, V]
            log_probs = F.log_softmax(logits, dim=-1)
            # Gather the log-prob assigned to the actual syscall at each position
            actual_log_probs = log_probs.gather(
                2, syscall_seq.unsqueeze(-1)
            ).squeeze(-1)  # [B, T]
            return -actual_log_probs  # Negate: high = anomalous

    def mean_anomaly_score(self, syscall_seq: torch.Tensor) -> torch.Tensor:
        """Scalar anomaly score per sequence (mean over positions).

        Args:
            syscall_seq: [batch, seq_len] int64

        Returns:
            scores: [batch] float32
        """
        return self.anomaly_scores(syscall_seq).mean(dim=-1)

    @staticmethod
    def clamp_syscalls(trace: List[int], n_syscalls: int = N_SYSCALLS) -> List[int]:
        """Clamp syscall numbers to valid embedding range."""
        return [min(max(s, 0), n_syscalls - 1) for s in trace]


# ═══════════════════════════════════════════════════════════════════════════════
# Runtime Security Monitor
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnomalyAlert:
    """A single anomaly detection alert."""
    timestamp: float
    anomaly_score: float
    is_anomalous: bool
    suspicious_subsequence: List[int]
    suspicious_names: List[str]
    window_snapshot: List[int]


class NeuralSecurityMonitor:
    """Online syscall anomaly detection via LSTM autoencoder.

    Maintains a sliding window of recent syscall numbers and periodically
    scores the window through the autoencoder. High reconstruction error
    indicates an anomalous syscall pattern.

    Supports two modes:
        - Training: collects normal traces, no alerts (call train_on_traces)
        - Monitoring: scores every window, alerts on anomalies

    Integration with NeuralWatchdog:
        The monitor can be attached to a MetalNeuralOS or NeuralWatchdog
        instance. When the watchdog's metrics-based anomaly score is also
        elevated, the combined confidence is higher.

    Usage:
        monitor = NeuralSecurityMonitor()
        monitor.load()

        result = monitor.observe(syscall_num)
        if result and result["is_anomalous"]:
            handle_alert(result)
    """

    def __init__(self, window_size: int = 32, anomaly_threshold: float = 4.0,
                 check_interval: int = 1, device: Optional[str] = None):
        """Initialize the security monitor.

        Args:
            window_size: Number of recent syscalls in the sliding window.
            anomaly_threshold: Mean reconstruction error above which a
                sequence is flagged as anomalous.
            check_interval: Score the window every N observations (1 = every
                syscall, higher = less frequent but cheaper).
            device: PyTorch device ("cpu", "mps", "cuda"). Auto-detected
                if None.
        """
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.check_interval = check_interval

        # Autoencoder
        self.model = SyscallAutoencoder(n_syscalls=N_SYSCALLS).to(device)
        self._trained = False

        # Sliding window (ring buffer as deque for O(1) append/popleft)
        self.window: deque[int] = deque(maxlen=window_size)

        # State
        self.mode: str = "monitoring"  # "training" or "monitoring"
        self.total_observations: int = 0
        self.alerts: List[AnomalyAlert] = []
        self.total_alerts: int = 0

        # Training mode: accumulate traces
        self._training_traces: List[List[int]] = []

    # ─── Observation ─────────────────────────────────────────────────────

    def observe(self, syscall_num: int) -> Optional[Dict]:
        """Process a single syscall observation.

        In training mode, simply records the syscall. In monitoring mode,
        scores the current window and returns an alert dict if anomalous.

        Args:
            syscall_num: The observed syscall number (e.g., 63 for READ).

        Returns:
            None if no anomaly (or in training mode).
            Dict with {anomaly_score, is_anomalous, suspicious_subsequence,
                       suspicious_names, window_snapshot} if anomalous.
        """
        clamped = min(max(syscall_num, 0), N_SYSCALLS - 1)
        self.window.append(clamped)
        self.total_observations += 1

        if self.mode == "training":
            return None

        # Only score when window is full and at the right interval
        if len(self.window) < self.window_size:
            return None
        if self.total_observations % self.check_interval != 0:
            return None

        return self._score_current_window()

    def _score_current_window(self) -> Optional[Dict]:
        """Score the current window through the autoencoder."""
        if not self._trained:
            return None

        window_list = list(self.window)
        seq = torch.tensor([window_list], dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            per_pos_scores = self.model.anomaly_scores(seq)[0]  # [T]
            mean_score = per_pos_scores.mean().item()

        is_anomalous = mean_score > self.anomaly_threshold

        # Find the most suspicious subsequence (contiguous high-scoring region)
        suspicious_subseq, suspicious_names = self._find_suspicious_subseq(
            window_list, per_pos_scores
        )

        # Boost score for sensitive syscalls in the suspicious region
        sensitive_in_window = sum(1 for s in suspicious_subseq if s in SENSITIVE_SYSCALLS)
        boosted_score = mean_score + (sensitive_in_window * 0.5)
        is_anomalous = boosted_score > self.anomaly_threshold

        result = {
            "anomaly_score": boosted_score,
            "raw_score": mean_score,
            "is_anomalous": is_anomalous,
            "suspicious_subsequence": suspicious_subseq,
            "suspicious_names": suspicious_names,
            "window_snapshot": window_list,
        }

        if is_anomalous:
            alert = AnomalyAlert(
                timestamp=time.time(),
                anomaly_score=boosted_score,
                is_anomalous=True,
                suspicious_subsequence=suspicious_subseq,
                suspicious_names=suspicious_names,
                window_snapshot=window_list,
            )
            self.alerts.append(alert)
            self.total_alerts += 1
            logger.warning(
                f"[SecurityMonitor] Anomaly detected: score={boosted_score:.3f}, "
                f"suspicious={suspicious_names}"
            )

        return result

    def _find_suspicious_subseq(self, window: List[int],
                                scores: torch.Tensor) -> Tuple[List[int], List[str]]:
        """Find the most suspicious contiguous subsequence in the window.

        Selects a run of positions where per-position anomaly score exceeds
        the per-position threshold (mean + 1 std of the scores).
        Falls back to the top-k highest-scoring positions if no contiguous run.
        """
        score_vals = scores.cpu().tolist()
        mean_s = sum(score_vals) / len(score_vals)
        std_s = (sum((s - mean_s) ** 2 for s in score_vals) / len(score_vals)) ** 0.5
        per_pos_threshold = mean_s + std_s

        # Find contiguous runs above threshold
        suspicious_indices = []
        best_run_start = 0
        best_run_len = 0
        current_start = 0
        current_len = 0

        for i, s in enumerate(score_vals):
            if s > per_pos_threshold:
                if current_len == 0:
                    current_start = i
                current_len += 1
                if current_len > best_run_len:
                    best_run_start = current_start
                    best_run_len = current_len
            else:
                current_len = 0

        if best_run_len > 0:
            suspicious_indices = list(range(best_run_start,
                                            best_run_start + best_run_len))
        else:
            # Fallback: top 5 highest-scoring positions
            ranked = sorted(range(len(score_vals)), key=lambda i: score_vals[i],
                            reverse=True)
            suspicious_indices = sorted(ranked[:5])

        subseq = [window[i] for i in suspicious_indices]
        names = [SYSCALL_NAMES.get(s, f"SYS_{s}") for s in subseq]
        return subseq, names

    # ─── Batch scoring ───────────────────────────────────────────────────

    def score_trace(self, trace: List[int]) -> Dict:
        """Score an entire syscall trace for anomalies.

        Splits the trace into overlapping windows and scores each.
        Returns aggregate statistics.

        Args:
            trace: List of syscall numbers.

        Returns:
            Dict with {mean_score, max_score, is_anomalous, n_windows,
                       anomalous_windows, per_window_scores}.
        """
        if not self._trained:
            return {"error": "Model not trained — call train_on_traces() or load() first"}

        clamped = SyscallAutoencoder.clamp_syscalls(trace, N_SYSCALLS)
        if len(clamped) < self.window_size:
            # Pad short traces
            clamped = clamped + [0] * (self.window_size - len(clamped))

        # Sliding windows with stride = window_size // 2
        stride = max(1, self.window_size // 2)
        windows = []
        for start in range(0, max(1, len(clamped) - self.window_size + 1), stride):
            w = clamped[start:start + self.window_size]
            if len(w) < self.window_size:
                w = w + [0] * (self.window_size - len(w))
            windows.append(w)

        if not windows:
            windows = [clamped[:self.window_size]]

        seq = torch.tensor(windows, dtype=torch.long, device=self.device)
        self.model.eval()
        scores = self.model.mean_anomaly_score(seq)  # [N]
        score_list = scores.cpu().tolist()

        mean_score = sum(score_list) / len(score_list)
        max_score = max(score_list)
        anomalous_count = sum(1 for s in score_list if s > self.anomaly_threshold)

        return {
            "mean_score": mean_score,
            "max_score": max_score,
            "is_anomalous": max_score > self.anomaly_threshold,
            "n_windows": len(windows),
            "anomalous_windows": anomalous_count,
            "per_window_scores": score_list,
        }

    # ─── Training mode ───────────────────────────────────────────────────

    def start_training(self):
        """Switch to training mode — observations are collected, not scored."""
        self.mode = "training"
        self._training_traces = []
        logger.info("[SecurityMonitor] Switched to training mode")

    def finish_training(self, epochs: int = 60, batch_size: int = 64,
                        lr: float = 1e-3) -> Dict:
        """Stop collecting traces and train the autoencoder on them.

        Returns training statistics.
        """
        self.mode = "monitoring"
        if not self._training_traces:
            logger.warning("[SecurityMonitor] No traces collected — skipping training")
            return {"error": "No traces collected"}

        stats = self.train_on_traces(self._training_traces, epochs=epochs,
                                     batch_size=batch_size, lr=lr)
        self._training_traces = []
        return stats

    def flush_training_trace(self):
        """Save the current window as a complete training trace and reset."""
        if len(self.window) > 0:
            self._training_traces.append(list(self.window))

    def train_on_traces(self, traces: List[List[int]], epochs: int = 60,
                        batch_size: int = 64, lr: float = 1e-3) -> Dict:
        """Train the autoencoder on a corpus of normal syscall traces.

        Args:
            traces: List of syscall sequences (variable length).
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            lr: Learning rate.

        Returns:
            Dict with training statistics.
        """
        # Convert traces to fixed-length windows
        windows = []
        for trace in traces:
            clamped = SyscallAutoencoder.clamp_syscalls(trace, N_SYSCALLS)
            if len(clamped) < 3:
                continue
            stride = max(1, self.window_size // 2)
            for start in range(0, max(1, len(clamped) - self.window_size + 1), stride):
                w = clamped[start:start + self.window_size]
                if len(w) < self.window_size:
                    w = w + [0] * (self.window_size - len(w))
                windows.append(w)

        if not windows:
            return {"error": "No valid training windows"}

        data = torch.tensor(windows, dtype=torch.long, device=self.device)
        n = len(data)
        split = int(0.85 * n)
        perm = torch.randperm(n, device="cpu")
        data = data[perm]
        train_data = data[:split]
        val_data = data[split:] if split < n else data[-1:]

        logger.info(f"[SecurityMonitor] Training: {len(train_data)} train, "
                     f"{len(val_data)} val windows")

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        best_val_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            # Train
            self.model.train()
            epoch_perm = torch.randperm(len(train_data), device="cpu")
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(train_data), batch_size):
                idx = epoch_perm[i:i + batch_size]
                batch = train_data[idx]

                logits = self.model(batch)
                B, T, V = logits.shape
                loss = F.cross_entropy(logits.reshape(B * T, V), batch.reshape(B * T))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validate
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(val_data)
                B, T, V = val_logits.shape
                val_loss = F.cross_entropy(
                    val_logits.reshape(B * T, V), val_data.reshape(B * T)
                ).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = total_loss / max(n_batches, 1)
                logger.info(
                    f"  Epoch {epoch + 1:3d}/{epochs}: "
                    f"train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}"
                )

        # Restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to(self.device)
        self.model.eval()
        self._trained = True

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"[SecurityMonitor] Trained: {param_count:,} params, "
                     f"best_val_loss={best_val_loss:.4f}")

        return {
            "epochs": epochs,
            "train_windows": len(train_data),
            "val_windows": len(val_data),
            "best_val_loss": best_val_loss,
            "param_count": param_count,
        }

    # ─── Persistence ─────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None):
        """Save the trained autoencoder weights.

        Args:
            path: Save path. Defaults to models/research/neuros/security_monitor.pt.
        """
        save_path = Path(path) if path else DEFAULT_MODEL_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(save_path))
        logger.info(f"[SecurityMonitor] Saved to {save_path}")

    def load(self, path: Optional[str] = None) -> bool:
        """Load pretrained autoencoder weights.

        Args:
            path: Load path. Defaults to models/research/neuros/security_monitor.pt.

        Returns:
            True if loaded successfully.
        """
        load_path = Path(path) if path else DEFAULT_MODEL_PATH
        if not load_path.exists():
            logger.warning(f"[SecurityMonitor] Model not found at {load_path}")
            return False
        try:
            state = torch.load(str(load_path), map_location=self.device,
                               weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()
            self._trained = True
            logger.info(f"[SecurityMonitor] Loaded from {load_path}")
            return True
        except Exception as e:
            logger.error(f"[SecurityMonitor] Failed to load: {e}")
            return False

    # ─── Watchdog integration ────────────────────────────────────────────

    def integrate_watchdog_score(self, watchdog_score: float,
                                 syscall_score: float) -> Dict:
        """Combine watchdog metrics anomaly score with syscall anomaly score.

        Both scores are in [0, +inf) where higher = more anomalous.
        The combined score uses a weighted geometric mean to require
        evidence from BOTH signals for highest confidence.

        Args:
            watchdog_score: Anomaly score from NeuralWatchdog (0-1, sigmoid output).
            syscall_score: Anomaly score from this monitor (mean cross-entropy).

        Returns:
            Dict with {combined_score, confidence, alert_level}.
        """
        # Normalize watchdog_score to comparable range
        # Watchdog is sigmoid [0, 1], syscall is cross-entropy [0, ~10+]
        wd_normalized = watchdog_score * 5.0  # Scale to roughly match CE range

        # Geometric mean: requires both signals to be elevated
        combined = (wd_normalized * syscall_score) ** 0.5

        # Confidence: high when both signals agree
        both_high = watchdog_score > 0.5 and syscall_score > self.anomaly_threshold
        both_low = watchdog_score < 0.3 and syscall_score < self.anomaly_threshold * 0.5
        if both_high:
            confidence = 0.95
            alert_level = "CRITICAL"
        elif watchdog_score > 0.5 or syscall_score > self.anomaly_threshold:
            confidence = 0.7
            alert_level = "WARNING"
        elif both_low:
            confidence = 0.1
            alert_level = "NORMAL"
        else:
            confidence = 0.4
            alert_level = "INFO"

        return {
            "combined_score": combined,
            "confidence": confidence,
            "alert_level": alert_level,
            "watchdog_score": watchdog_score,
            "syscall_score": syscall_score,
        }

    # ─── Diagnostics ─────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Return security monitor statistics."""
        recent = self.alerts[-5:]
        recent_summary = [
            {
                "timestamp": a.timestamp,
                "anomaly_score": a.anomaly_score,
                "suspicious": a.suspicious_names,
            }
            for a in recent
        ]
        return {
            "mode": self.mode,
            "trained": self._trained,
            "total_observations": self.total_observations,
            "window_size": self.window_size,
            "current_window_fill": len(self.window),
            "anomaly_threshold": self.anomaly_threshold,
            "total_alerts": self.total_alerts,
            "alert_rate": self.total_alerts / max(self.total_observations, 1),
            "recent_alerts": recent_summary,
            "param_count": sum(p.numel() for p in self.model.parameters()),
        }

    def print_report(self):
        """Print a human-readable security monitoring report."""
        s = self.stats()
        print(f"\n  \033[1;37mNeural Security Monitor (LSTM Autoencoder):\033[0m")
        print(f"    Mode:               {s['mode']}")
        print(f"    Model:              {'trained' if s['trained'] else 'untrained'} "
              f"({s['param_count']:,} params)")
        print(f"    Observations:       {s['total_observations']}")
        print(f"    Window:             {s['current_window_fill']}/{s['window_size']}")
        print(f"    Threshold:          {s['anomaly_threshold']:.2f}")
        print(f"    Alerts:             {s['total_alerts']} "
              f"({s['alert_rate']:.2%} rate)")

        if self.alerts:
            print(f"    Recent alerts:")
            for alert in self.alerts[-5:]:
                color = "\033[1;31m" if alert.anomaly_score > self.anomaly_threshold * 1.5 \
                    else "\033[1;33m"
                names = ", ".join(alert.suspicious_names[:6])
                print(f"      {color}[{alert.anomaly_score:.3f}]\033[0m {names}")
        else:
            print(f"    Status:             \033[1;32mNo anomalies detected\033[0m")

    def __repr__(self) -> str:
        s = self.stats()
        return (f"NeuralSecurityMonitor(mode={s['mode']}, "
                f"trained={s['trained']}, "
                f"observations={s['total_observations']}, "
                f"alerts={s['total_alerts']})")
