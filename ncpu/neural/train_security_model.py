#!/usr/bin/env python3
"""Train the Neural Security models.

Trains two complementary anomaly detectors:

1. **NeuralSecurityLSTM** (next-syscall predictor):
   embed(n_syscalls, 32) -> LSTM(32, 64) -> Linear(64, n_syscalls)
   Detects anomalies via low P(observed_syscall | history).
   Target: >70% next-syscall prediction accuracy on normal sequences.

2. **SyscallAutoencoder** (LSTM autoencoder — sequence reconstruction):
   Encoder: Embedding(512, 32) -> LSTM(32, 64, 2) -> Linear(64, 32)
   Decoder: Linear(32, 64) -> LSTM(64, 64, 2) -> Linear(64, 512)
   Detects anomalies via high reconstruction error (cross-entropy).
   Target: >80% detection rate with <15% false alarm rate.

Both models are trained on synthetic normal workload traces (shell sessions,
compilations, file I/O, multi-process patterns) and evaluated against
anomalous traces (fork bombs, random chaos, exec sprays, etc.).
"""

import os
import sys
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent

# ── Syscall definitions ───────────────────────────────────────────────────

SYSCALL_MAP = {
    "READ": 63, "WRITE": 64, "EXIT": 93, "EXIT_GROUP": 94,
    "BRK": 214, "CLOSE": 57, "OPENAT": 56, "LSEEK": 62,
    "GETPID": 172, "CLONE": 220, "WAIT4": 260,
    "PIPE2": 59, "DUP3": 24, "KILL": 129,
    "COMPILE": 300, "EXEC": 301,
    # Additional syscalls for richer patterns
    "MMAP": 222, "MUNMAP": 215, "FSTAT": 80,
    "IOCTL": 29, "GETCWD": 17, "CHDIR": 49,
    "GETDENTS": 61, "FCNTL": 25,
}

# Reverse map for display
SYSCALL_NAMES = {v: k for k, v in SYSCALL_MAP.items()}

# All unique syscall numbers we model
ALL_SYSCALLS = sorted(set(SYSCALL_MAP.values()))
N_SYSCALLS = 320  # Model vocab covers up to syscall 319

# ── Normal workload trace generators ─────────────────────────────────────

def gen_shell_session(n_commands: int = 20) -> List[int]:
    """Generate a typical interactive shell session."""
    trace = []

    # Boot sequence
    trace.extend([214, 222, 222, 56, 63, 57])  # BRK, MMAP, MMAP, OPENAT, READ, CLOSE

    command_patterns = {
        "ls": [56, 61, 64, 57],             # openat, getdents, write, close
        "cat": [56, 63, 64, 57],             # openat, read, write, close
        "echo": [64],                         # write
        "echo_redirect": [56, 64, 57],       # openat, write, close
        "cd": [49],                           # chdir
        "pwd": [17, 64],                      # getcwd, write
        "wc": [56, 63, 63, 64, 57],          # openat, read, read, write, close
        "grep": [56, 63, 63, 64, 57],        # openat, read, read, write, close
        "mkdir": [56, 57],                    # openat (create), close
        "rm": [56, 57],                       # openat, close (unlink)
    }

    for _ in range(n_commands):
        cmd = random.choice(list(command_patterns.keys()))
        pattern = command_patterns[cmd]
        # Read stdin (shell reads command)
        trace.append(63)  # READ from stdin
        trace.extend(pattern)
        # Occasional write to stdout (prompt)
        trace.append(64)  # WRITE prompt

    # Exit
    trace.extend([93, 94])  # EXIT, EXIT_GROUP
    return trace


def gen_compile_session() -> List[int]:
    """Generate a compilation workload."""
    trace = []
    # Compiler init
    trace.extend([214, 222, 222, 56, 63])  # BRK, MMAP, MMAP, OPENAT source, READ

    # Read source file (multiple reads)
    for _ in range(random.randint(5, 15)):
        trace.append(63)  # READ

    trace.append(57)  # CLOSE source

    # Compilation (internal processing, periodic BRK for memory)
    for _ in range(random.randint(3, 8)):
        trace.extend([214, 222])  # BRK, MMAP

    # Write output
    trace.extend([56, 64, 64, 64, 57])  # OPENAT output, WRITE x3, CLOSE

    # Cleanup
    trace.extend([215, 215, 93])  # MUNMAP, MUNMAP, EXIT
    return trace


def gen_file_io_workload() -> List[int]:
    """Generate a file I/O heavy workload."""
    trace = []
    trace.extend([214, 222])  # Init

    for _ in range(random.randint(10, 30)):
        op = random.choice(["read", "write", "readwrite"])
        if op == "read":
            trace.extend([56, 63, 63, 57])  # OPENAT, READ, READ, CLOSE
        elif op == "write":
            trace.extend([56, 64, 64, 57])  # OPENAT, WRITE, WRITE, CLOSE
        else:
            trace.extend([56, 63, 64, 57])  # OPENAT, READ, WRITE, CLOSE

    trace.append(93)  # EXIT
    return trace


def gen_multiprocess_workload() -> List[int]:
    """Generate a fork/exec workload."""
    trace = []
    trace.extend([214, 222])  # Init

    # Shell reads command
    trace.append(63)

    for _ in range(random.randint(2, 5)):
        # Fork
        trace.append(220)  # CLONE

        # Parent waits
        trace.append(260)  # WAIT4

        # Between forks, some I/O
        if random.random() < 0.5:
            trace.extend([56, 63, 57])  # OPENAT, READ, CLOSE

    # Pipe operations
    if random.random() < 0.7:
        trace.extend([59, 220, 24, 64, 63, 260])  # PIPE2, CLONE, DUP3, WRITE, READ, WAIT4

    trace.extend([93, 94])
    return trace


def gen_normal_traces(n_traces: int = 500) -> List[List[int]]:
    """Generate a corpus of normal workload traces."""
    traces = []
    generators = [
        (gen_shell_session, 0.4),
        (gen_compile_session, 0.2),
        (gen_file_io_workload, 0.25),
        (gen_multiprocess_workload, 0.15),
    ]

    for _ in range(n_traces):
        r = random.random()
        cumulative = 0.0
        for gen_fn, prob in generators:
            cumulative += prob
            if r < cumulative:
                traces.append(gen_fn())
                break
        else:
            traces.append(gen_shell_session())

    return traces


def gen_anomalous_traces(n_traces: int = 100) -> List[List[int]]:
    """Generate anomalous traces for testing."""
    traces = []
    for _ in range(n_traces):
        anomaly_type = random.choice(["fork_bomb", "random_chaos", "exec_spray",
                                       "kill_spam", "reversed"])

        if anomaly_type == "fork_bomb":
            # Rapid clone without wait
            trace = [220] * random.randint(20, 50) + [93]
        elif anomaly_type == "random_chaos":
            # Random syscall sequence
            trace = [random.choice(ALL_SYSCALLS) for _ in range(30)]
        elif anomaly_type == "exec_spray":
            # Repeated exec attempts
            trace = [301, 56, 301, 56, 301, 301, 301, 93]
        elif anomaly_type == "kill_spam":
            # Repeated kill
            trace = [129] * 15 + [93]
        elif anomaly_type == "reversed":
            # Normal trace but reversed (unnatural sequence)
            trace = list(reversed(gen_shell_session(5)))
        else:
            trace = [random.choice(ALL_SYSCALLS) for _ in range(20)]

        traces.append(trace)
    return traces


# ── Model ─────────────────────────────────────────────────────────────────

class NeuralSecurityLSTM(nn.Module):
    """LSTM that learns normal syscall patterns and flags anomalies.

    Trained on sequences of normal syscall numbers. At inference time,
    computes the probability of each observed syscall given the history.
    Low-probability syscalls are flagged as anomalous.
    """

    def __init__(self, n_syscalls=320, embed_dim=32, hidden=64, n_layers=2):
        super().__init__()
        self.n_syscalls = n_syscalls
        self.embed_dim = embed_dim
        self.hidden_dim = hidden

        self.embed = nn.Embedding(n_syscalls, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True,
                            num_layers=n_layers, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, n_syscalls),
        )

    def forward(self, syscall_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            syscall_seq: (batch, seq_len) of syscall numbers (clamped to 0..n_syscalls-1)
        Returns:
            (batch, seq_len, n_syscalls) logits for next-syscall prediction
        """
        embedded = self.embed(syscall_seq)
        out, _ = self.lstm(embedded)
        return self.head(out)

    def anomaly_scores(self, syscall_seq: torch.Tensor) -> torch.Tensor:
        """Compute per-position anomaly scores (negative log-probability).

        Higher score = more anomalous.
        """
        with torch.no_grad():
            logits = self.forward(syscall_seq)
            log_probs = F.log_softmax(logits, dim=-1)
            # For positions 1..T, the anomaly score is -log P(x_t | x_{<t})
            # Shift: logits at position t predict position t+1
            shifted_seq = syscall_seq[:, 1:]  # (batch, seq_len-1)
            shifted_logprobs = log_probs[:, :-1]  # (batch, seq_len-1, n_syscalls)
            # Gather the log-prob of the actual next syscall
            actual_logprobs = shifted_logprobs.gather(
                2, shifted_seq.unsqueeze(-1)
            ).squeeze(-1)
            return -actual_logprobs  # Higher = more anomalous

    @staticmethod
    def clamp_syscalls(trace: List[int], n_syscalls: int = 320) -> List[int]:
        """Clamp syscall numbers to valid range."""
        return [min(max(s, 0), n_syscalls - 1) for s in trace]


# ── Training ──────────────────────────────────────────────────────────────

def prepare_sequences(traces: List[List[int]], seq_len: int = 64,
                      n_syscalls: int = 320) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert traces to fixed-length input/target pairs for training."""
    inputs = []
    targets = []

    for trace in traces:
        trace = NeuralSecurityLSTM.clamp_syscalls(trace, n_syscalls)
        if len(trace) < 3:
            continue

        # Sliding window over the trace
        for start in range(0, max(1, len(trace) - seq_len), seq_len // 2):
            end = min(start + seq_len + 1, len(trace))
            window = trace[start:end]
            if len(window) < 3:
                continue

            # Pad if necessary
            while len(window) < seq_len + 1:
                window.append(0)

            inp = window[:seq_len]
            tgt = window[1:seq_len + 1]
            inputs.append(inp)
            targets.append(tgt)

    return (torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long))


def train(epochs: int = 80, batch_size: int = 64, lr: float = 1e-3,
          n_traces: int = 800, seq_len: int = 64):
    """Train the security LSTM and save to models/os/security_lstm.pt."""
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    print(f"Training NeuralSecurityLSTM on {device}")
    print(f"  {n_traces} normal traces, seq_len={seq_len}")
    print(f"  {epochs} epochs, batch_size={batch_size}, lr={lr}")

    # Generate data
    normal_traces = gen_normal_traces(n_traces)
    anomalous_traces = gen_anomalous_traces(100)

    # Prepare sequences
    inputs, targets = prepare_sequences(normal_traces, seq_len)
    n = len(inputs)
    split = int(0.85 * n)

    perm = torch.randperm(n)
    inputs, targets = inputs[perm], targets[perm]

    train_inp, train_tgt = inputs[:split].to(device), targets[:split].to(device)
    val_inp, val_tgt = inputs[split:].to(device), targets[split:].to(device)
    print(f"  Train sequences: {len(train_inp)}, Val sequences: {len(val_inp)}")

    model = NeuralSecurityLSTM().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_inp))
        total_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, len(train_inp), batch_size):
            idx = perm[i:i + batch_size]
            inp_batch = train_inp[idx]
            tgt_batch = train_tgt[idx]

            logits = model(inp_batch)  # (batch, seq_len, n_syscalls)
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.reshape(-1, model.n_syscalls),
                tgt_batch.reshape(-1),
                ignore_index=0,  # Ignore padding
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Accuracy (ignoring padding)
            mask = tgt_batch != 0
            preds = logits.argmax(dim=-1)
            correct += ((preds == tgt_batch) & mask).sum().item()
            total += mask.sum().item()
            total_loss += loss.item() * len(idx)

        scheduler.step()
        train_acc = correct / max(total, 1)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for i in range(0, len(val_inp), batch_size):
                inp_batch = val_inp[i:i + batch_size]
                tgt_batch = val_tgt[i:i + batch_size]
                logits = model(inp_batch)
                mask = tgt_batch != 0
                preds = logits.argmax(dim=-1)
                val_correct += ((preds == tgt_batch) & mask).sum().item()
                val_total += mask.sum().item()

        val_acc = val_correct / max(val_total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"loss={total_loss / max(len(train_inp), 1):.4f}, "
                  f"train_acc={train_acc:.1%}, val_acc={val_acc:.1%}")

    print(f"\n  Best validation accuracy: {best_val_acc:.1%}")

    # Save
    save_path = PROJECT_ROOT / "models" / "os" / "security_lstm.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, str(save_path))
    print(f"  Saved to {save_path}")

    # Anomaly detection evaluation
    model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()

    print("\n  Anomaly Detection Evaluation:")

    # Score normal traces
    normal_scores = []
    for trace in normal_traces[:100]:
        trace = NeuralSecurityLSTM.clamp_syscalls(trace)
        if len(trace) < 3:
            continue
        seq = torch.tensor([trace[:seq_len]], dtype=torch.long, device=device)
        if seq.shape[1] < seq_len:
            seq = F.pad(seq, (0, seq_len - seq.shape[1]))
        scores = model.anomaly_scores(seq)
        normal_scores.append(scores.mean().item())

    # Score anomalous traces
    anomalous_scores = []
    for trace in anomalous_traces:
        trace = NeuralSecurityLSTM.clamp_syscalls(trace)
        if len(trace) < 3:
            continue
        seq = torch.tensor([trace[:seq_len]], dtype=torch.long, device=device)
        if seq.shape[1] < seq_len:
            seq = F.pad(seq, (0, seq_len - seq.shape[1]))
        scores = model.anomaly_scores(seq)
        anomalous_scores.append(scores.mean().item())

    if normal_scores and anomalous_scores:
        avg_normal = sum(normal_scores) / len(normal_scores)
        avg_anomalous = sum(anomalous_scores) / len(anomalous_scores)
        # Find threshold that separates
        threshold = (avg_normal + avg_anomalous) / 2

        # Detection rate
        true_pos = sum(1 for s in anomalous_scores if s > threshold)
        true_neg = sum(1 for s in normal_scores if s <= threshold)
        detection_rate = true_pos / len(anomalous_scores)
        false_alarm_rate = 1.0 - true_neg / len(normal_scores)

        print(f"    Normal mean score:    {avg_normal:.3f}")
        print(f"    Anomalous mean score: {avg_anomalous:.3f}")
        print(f"    Separation ratio:     {avg_anomalous / max(avg_normal, 0.01):.1f}x")
        print(f"    Threshold:            {threshold:.3f}")
        print(f"    Detection rate:       {detection_rate:.1%}")
        print(f"    False alarm rate:     {false_alarm_rate:.1%}")

    return best_val_acc


# ── Autoencoder training ─────────────────────────────────────────────────

def train_autoencoder(epochs: int = 80, batch_size: int = 64, lr: float = 1e-3,
                      n_traces: int = 800, window_size: int = 32):
    """Train the SyscallAutoencoder and save to models/os/security_monitor.pt.

    The autoencoder learns to reconstruct normal syscall windows through a
    latent bottleneck. Anomalous sequences that deviate from normal patterns
    produce high reconstruction error.

    Returns:
        Tuple of (best_val_loss, detection_rate, false_alarm_rate).
    """
    from ncpu.neural.neural_security_monitor import (
        NeuralSecurityMonitor, SyscallAutoencoder, N_SYSCALLS,
    )

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    print(f"\nTraining SyscallAutoencoder on {device}")
    print(f"  {n_traces} normal traces, window_size={window_size}")
    print(f"  {epochs} epochs, batch_size={batch_size}, lr={lr}")

    # Generate data
    normal_traces = gen_normal_traces(n_traces)
    anomalous_traces = gen_anomalous_traces(200)

    # Create monitor and train
    monitor = NeuralSecurityMonitor(
        window_size=window_size, device=device,
    )
    stats = monitor.train_on_traces(normal_traces, epochs=epochs,
                                    batch_size=batch_size, lr=lr)

    if "error" in stats:
        print(f"  Training failed: {stats['error']}")
        return 0.0, 0.0, 1.0

    param_count = stats["param_count"]
    print(f"  Model params: {param_count:,}")
    print(f"  Best validation loss: {stats['best_val_loss']:.4f}")

    # Save
    save_path = PROJECT_ROOT / "models" / "os" / "security_monitor.pt"
    monitor.save(str(save_path))
    print(f"  Saved to {save_path}")

    # Evaluate anomaly detection
    print("\n  Autoencoder Anomaly Detection Evaluation:")

    # Score normal traces
    normal_scores = []
    for trace in normal_traces[:200]:
        result = monitor.score_trace(trace)
        if "error" not in result:
            normal_scores.append(result["mean_score"])

    # Score anomalous traces
    anomalous_scores = []
    for trace in anomalous_traces:
        result = monitor.score_trace(trace)
        if "error" not in result:
            anomalous_scores.append(result["mean_score"])

    detection_rate = 0.0
    false_alarm_rate = 1.0

    if normal_scores and anomalous_scores:
        avg_normal = sum(normal_scores) / len(normal_scores)
        avg_anomalous = sum(anomalous_scores) / len(anomalous_scores)
        separation = avg_anomalous / max(avg_normal, 0.01)

        # Determine threshold: use the monitor's default or compute from data
        threshold = monitor.anomaly_threshold

        # Also compute optimal threshold for reporting
        all_scores = [(s, False) for s in normal_scores] + [(s, True) for s in anomalous_scores]
        all_scores.sort(key=lambda x: x[0])

        best_f1 = 0.0
        best_thresh = threshold
        # Sweep thresholds
        for s, _ in all_scores:
            tp = sum(1 for sc in anomalous_scores if sc > s)
            tn = sum(1 for sc in normal_scores if sc <= s)
            fp = len(normal_scores) - tn
            fn = len(anomalous_scores) - tp
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = s

        # Final metrics at the best threshold
        true_pos = sum(1 for s in anomalous_scores if s > best_thresh)
        true_neg = sum(1 for s in normal_scores if s <= best_thresh)
        detection_rate = true_pos / len(anomalous_scores)
        false_alarm_rate = 1.0 - true_neg / len(normal_scores)

        print(f"    Normal mean score:    {avg_normal:.3f}")
        print(f"    Anomalous mean score: {avg_anomalous:.3f}")
        print(f"    Separation ratio:     {separation:.1f}x")
        print(f"    Optimal threshold:    {best_thresh:.3f}")
        print(f"    Detection rate:       {detection_rate:.1%}")
        print(f"    False alarm rate:     {false_alarm_rate:.1%}")
        print(f"    Best F1:              {best_f1:.3f}")

        # Update monitor threshold to the optimal value
        monitor.anomaly_threshold = best_thresh
        # Re-save with updated threshold info
        monitor.save(str(save_path))

    return stats["best_val_loss"], detection_rate, false_alarm_rate


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train neural security models")
    parser.add_argument("--model", choices=["lstm", "autoencoder", "both"],
                        default="both", help="Which model(s) to train")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--traces", type=int, default=800, help="Number of normal traces")
    args = parser.parse_args()

    if args.model in ("lstm", "both"):
        print("=" * 60)
        print("Training NeuralSecurityLSTM (next-syscall predictor)")
        print("=" * 60)
        acc = train(epochs=args.epochs, n_traces=args.traces)
        target = 0.70
        if acc >= target:
            print(f"\n  LSTM target met: {acc:.1%} >= {target:.0%}")
        else:
            print(f"\n  Below target ({acc:.1%} < {target:.0%}), retraining...")
            acc = train(epochs=120, n_traces=1500, seq_len=96)
            print(f"  Final LSTM accuracy: {acc:.1%}")

    if args.model in ("autoencoder", "both"):
        print("\n" + "=" * 60)
        print("Training SyscallAutoencoder (reconstruction-based)")
        print("=" * 60)
        val_loss, det_rate, fa_rate = train_autoencoder(
            epochs=args.epochs, n_traces=args.traces,
        )
        print(f"\n  Autoencoder: val_loss={val_loss:.4f}, "
              f"detection={det_rate:.1%}, false_alarm={fa_rate:.1%}")
        if det_rate >= 0.80:
            print(f"  Detection target met: {det_rate:.1%} >= 80%")
        else:
            print(f"  Below target ({det_rate:.1%} < 80%), retraining with more data...")
            val_loss, det_rate, fa_rate = train_autoencoder(
                epochs=120, n_traces=1500, window_size=32,
            )
            print(f"  Final autoencoder: detection={det_rate:.1%}, "
                  f"false_alarm={fa_rate:.1%}")
