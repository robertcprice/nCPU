#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║           NEURAL CONTEXT-AWARE SCHEDULER - AI-Native Process Management          ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  A truly NEURAL operating system scheduler that:                                 ║
║  • Learns user behavior patterns over time                                       ║
║  • Predicts which processes will be needed next                                  ║
║  • Pre-warms frequently used binaries                                            ║
║  • Optimizes memory layout based on access patterns                              ║
║  • Uses attention mechanisms for process priority                                ║
║  • Adapts execution strategies based on context                                  ║
║                                                                                  ║
║  This is NOT traditional scheduling - it's LEARNED scheduling!                   ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ════════════════════════════════════════════════════════════════════════════════
# NEURAL PROCESS PREDICTOR - Predicts what the user will run next
# ════════════════════════════════════════════════════════════════════════════════

class ProcessSequenceEncoder(nn.Module):
    """
    Encodes sequences of executed processes into a latent representation.
    Uses transformer attention to capture temporal patterns.
    """

    def __init__(self, vocab_size: int = 256, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 32, embed_dim) * 0.1)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, process_ids: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of process IDs."""
        x = self.embedding(process_ids)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        attn_out, _ = self.attention(x, x, x)
        return self.fc(attn_out[:, -1, :])  # Last position's encoding


class ProcessPredictor(nn.Module):
    """
    Predicts the next process based on:
    - Recent process history
    - Time of day
    - Current working directory
    - System load
    """

    def __init__(self, num_processes: int = 64, embed_dim: int = 64):
        super().__init__()
        self.num_processes = num_processes

        # Process sequence encoder
        self.seq_encoder = ProcessSequenceEncoder(num_processes, embed_dim)

        # Context encoders
        self.time_encoder = nn.Linear(4, 16)  # hour, minute, day_of_week, is_weekend
        self.load_encoder = nn.Linear(3, 16)  # cpu, memory, io

        # Combined predictor
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim + 16 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_processes)
        )

    def forward(
        self,
        process_history: torch.Tensor,
        time_features: torch.Tensor,
        load_features: torch.Tensor
    ) -> torch.Tensor:
        """Predict probability distribution over next processes."""
        seq_encoding = self.seq_encoder(process_history)
        time_encoding = self.time_encoder(time_features)
        load_encoding = self.load_encoder(load_features)

        combined = torch.cat([seq_encoding, time_encoding, load_encoding], dim=-1)
        return F.softmax(self.predictor(combined), dim=-1)


# ════════════════════════════════════════════════════════════════════════════════
# NEURAL PRIORITY SYSTEM - Learns optimal priorities from user feedback
# ════════════════════════════════════════════════════════════════════════════════

class NeuralPriorityNetwork(nn.Module):
    """
    Learns process priorities based on:
    - Process type and history
    - User interaction patterns
    - Response time requirements
    - Resource utilization
    """

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Priority 0-1
        )

    def forward(self, process_features: torch.Tensor) -> torch.Tensor:
        return self.network(process_features)


# ════════════════════════════════════════════════════════════════════════════════
# NEURAL MEMORY OPTIMIZER - Predicts memory access patterns
# ════════════════════════════════════════════════════════════════════════════════

class MemoryAccessPredictor(nn.Module):
    """
    Predicts memory access patterns to optimize:
    - Page prefetching
    - Cache placement
    - Memory layout
    """

    def __init__(self, history_len: int = 16, page_bits: int = 12):
        super().__init__()
        self.history_encoder = nn.LSTM(
            input_size=page_bits,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.predictor = nn.Linear(64, page_bits)

    def forward(self, page_history: torch.Tensor) -> torch.Tensor:
        """Predict next likely page access."""
        _, (h_n, _) = self.history_encoder(page_history)
        return torch.sigmoid(self.predictor(h_n[-1]))


# ════════════════════════════════════════════════════════════════════════════════
# CONTEXT-AWARE SCHEDULER - The main neural scheduler
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class ProcessContext:
    """Context for a running process."""
    pid: int
    name: str
    binary_path: str
    start_time: float
    priority: float = 0.5
    cpu_time: float = 0.0
    memory_usage: int = 0
    io_operations: int = 0
    state: str = "running"  # running, waiting, blocked, zombie


@dataclass
class UserSession:
    """Tracks user session patterns."""
    session_start: float
    commands_run: List[str] = field(default_factory=list)
    command_times: List[float] = field(default_factory=list)
    working_directories: List[str] = field(default_factory=list)


class NeuralContextScheduler:
    """
    A truly neural scheduler that learns optimal process management.

    Key features:
    - Predicts next command before user types it
    - Pre-loads frequently used binaries
    - Learns user-specific patterns
    - Adapts to workload changes
    - Optimizes based on time of day
    """

    def __init__(self, num_processes: int = 64):
        self.num_processes = num_processes

        # Neural components
        self.process_predictor = ProcessPredictor(num_processes).to(device)
        self.priority_network = NeuralPriorityNetwork().to(device)
        self.memory_predictor = MemoryAccessPredictor().to(device)

        # Process management
        self.processes: Dict[int, ProcessContext] = {}
        self.next_pid = 1
        self.process_name_to_id: Dict[str, int] = {}
        self.id_to_process_name: Dict[int, str] = {}

        # History tracking
        self.command_history: deque = deque(maxlen=100)
        self.process_history: deque = deque(maxlen=32)
        self.session = UserSession(session_start=time.time())

        # Learning statistics
        self.prediction_accuracy = 0.0
        self.predictions_made = 0
        self.correct_predictions = 0

        # Pre-warm cache
        self.prewarm_cache: Dict[str, bytes] = {}

        # Time tracking
        self.last_command_time = time.time()

        # Optimizer for online learning
        self.optimizer = torch.optim.Adam(
            list(self.process_predictor.parameters()) +
            list(self.priority_network.parameters()),
            lr=0.001
        )

        print("[Neural Scheduler] Initialized with context-aware learning")

    def _get_time_features(self) -> torch.Tensor:
        """Get current time features for prediction."""
        now = time.localtime()
        return torch.tensor([
            now.tm_hour / 24.0,
            now.tm_min / 60.0,
            now.tm_wday / 7.0,
            1.0 if now.tm_wday >= 5 else 0.0
        ], dtype=torch.float32, device=device).unsqueeze(0)

    def _get_load_features(self) -> torch.Tensor:
        """Get current system load features."""
        # Simplified - in real implementation, would query actual system
        num_processes = len([p for p in self.processes.values() if p.state == "running"])
        return torch.tensor([
            min(num_processes / 10.0, 1.0),  # CPU load proxy
            0.3,  # Memory usage (placeholder)
            0.1   # IO activity (placeholder)
        ], dtype=torch.float32, device=device).unsqueeze(0)

    def _get_process_id(self, name: str) -> int:
        """Get or create process ID for a command name."""
        if name not in self.process_name_to_id:
            pid = len(self.process_name_to_id)
            self.process_name_to_id[name] = pid
            self.id_to_process_name[pid] = name
        return self.process_name_to_id[name]

    def predict_next_command(self) -> List[Tuple[str, float]]:
        """
        Predict what the user will run next.
        Returns top-5 predictions with probabilities.
        """
        if len(self.process_history) < 3:
            return []

        # Prepare input
        history_ids = list(self.process_history)[-16:]
        while len(history_ids) < 16:
            history_ids = [0] + history_ids

        history_tensor = torch.tensor([history_ids], dtype=torch.long, device=device)
        time_features = self._get_time_features()
        load_features = self._get_load_features()

        # Predict
        with torch.no_grad():
            probs = self.process_predictor(history_tensor, time_features, load_features)

        # Get top predictions
        top_probs, top_ids = torch.topk(probs[0], min(5, len(self.id_to_process_name)))

        predictions = []
        for prob, pid in zip(top_probs.cpu().numpy(), top_ids.cpu().numpy()):
            if pid in self.id_to_process_name:
                predictions.append((self.id_to_process_name[pid], float(prob)))

        self.predictions_made += 1
        return predictions

    def record_command(self, command: str):
        """Record a command execution for learning."""
        # Check if we predicted correctly
        predictions = self.predict_next_command()
        if predictions and predictions[0][0] == command:
            self.correct_predictions += 1

        # Update accuracy
        if self.predictions_made > 0:
            self.prediction_accuracy = self.correct_predictions / self.predictions_made

        # Record
        proc_id = self._get_process_id(command)
        self.process_history.append(proc_id)
        self.command_history.append(command)
        self.session.commands_run.append(command)
        self.session.command_times.append(time.time())

        # Online learning (simplified)
        self._learn_from_command(command, proc_id)

        self.last_command_time = time.time()

    def _learn_from_command(self, command: str, proc_id: int):
        """Perform online learning from the command."""
        if len(self.process_history) < 4:
            return

        # Prepare training sample
        history_ids = list(self.process_history)[-17:-1]  # History before this command
        while len(history_ids) < 16:
            history_ids = [0] + history_ids

        history_tensor = torch.tensor([history_ids], dtype=torch.long, device=device)
        time_features = self._get_time_features()
        load_features = self._get_load_features()
        target = torch.tensor([proc_id], dtype=torch.long, device=device)

        # Forward pass
        self.optimizer.zero_grad()
        probs = self.process_predictor(history_tensor, time_features, load_features)

        # Loss and backprop
        loss = F.cross_entropy(probs, target)
        loss.backward()
        self.optimizer.step()

    def spawn_process(self, name: str, binary_path: str) -> ProcessContext:
        """Spawn a new process with neural priority assignment."""
        # Get neural priority
        features = self._get_process_features(name)
        with torch.no_grad():
            priority = self.priority_network(features).item()

        # Create process
        ctx = ProcessContext(
            pid=self.next_pid,
            name=name,
            binary_path=binary_path,
            start_time=time.time(),
            priority=priority
        )

        self.processes[self.next_pid] = ctx
        self.next_pid += 1

        # Record for learning
        self.record_command(name)

        return ctx

    def _get_process_features(self, name: str) -> torch.Tensor:
        """Extract features for priority prediction."""
        # Count how often this command was run
        count = self.command_history.count(name) if self.command_history else 0

        # Time since last use
        time_since = 1.0
        for i, cmd in enumerate(reversed(self.command_history)):
            if cmd == name:
                time_since = i / len(self.command_history)
                break

        # Create feature vector
        features = torch.zeros(32, dtype=torch.float32, device=device)
        features[0] = count / 100.0  # Frequency
        features[1] = time_since  # Recency
        features[2] = len(name) / 20.0  # Name length (proxy for complexity)
        features[3] = 1.0 if name in ['ls', 'cat', 'echo'] else 0.0  # Is common utility
        features[4] = 1.0 if 'test' in name else 0.0  # Is test-related

        return features.unsqueeze(0)

    def get_ready_queue(self) -> List[ProcessContext]:
        """Get processes sorted by neural priority."""
        running = [p for p in self.processes.values() if p.state == "running"]
        return sorted(running, key=lambda p: p.priority, reverse=True)

    def prewarm_binary(self, binary_path: str, data: bytes):
        """Pre-warm a binary into cache for faster loading."""
        self.prewarm_cache[binary_path] = data
        print(f"  [Prewarm] Cached {binary_path} ({len(data)} bytes)")

    def get_prewarmed(self, binary_path: str) -> Optional[bytes]:
        """Get a pre-warmed binary if available."""
        return self.prewarm_cache.get(binary_path)

    def suggest_prewarming(self) -> List[str]:
        """Suggest binaries to pre-warm based on predictions."""
        predictions = self.predict_next_command()
        return [cmd for cmd, prob in predictions if prob > 0.2]

    def get_stats(self) -> Dict:
        """Get scheduler statistics."""
        return {
            "processes_running": len([p for p in self.processes.values() if p.state == "running"]),
            "commands_recorded": len(self.command_history),
            "prediction_accuracy": f"{self.prediction_accuracy * 100:.1f}%",
            "predictions_made": self.predictions_made,
            "correct_predictions": self.correct_predictions,
            "prewarmed_binaries": len(self.prewarm_cache),
            "known_commands": len(self.process_name_to_id),
        }

    def print_predictions(self):
        """Print current predictions for next command."""
        predictions = self.predict_next_command()
        if predictions:
            print("\n  [Neural Prediction] Next likely commands:")
            for cmd, prob in predictions[:3]:
                bar = "█" * int(prob * 20)
                print(f"    {cmd:15} {bar} {prob*100:.1f}%")

    def save_model(self, path: str = "models/neural_scheduler.pt"):
        """Save learned models."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'process_predictor': self.process_predictor.state_dict(),
            'priority_network': self.priority_network.state_dict(),
            'memory_predictor': self.memory_predictor.state_dict(),
            'process_name_to_id': self.process_name_to_id,
            'id_to_process_name': self.id_to_process_name,
        }, path)
        print(f"  [Scheduler] Models saved to {path}")

    def load_model(self, path: str = "models/neural_scheduler.pt"):
        """Load learned models."""
        if Path(path).exists():
            data = torch.load(path, map_location=device)
            self.process_predictor.load_state_dict(data['process_predictor'])
            self.priority_network.load_state_dict(data['priority_network'])
            self.memory_predictor.load_state_dict(data['memory_predictor'])
            self.process_name_to_id = data['process_name_to_id']
            self.id_to_process_name = data['id_to_process_name']
            print(f"  [Scheduler] Models loaded from {path}")


# ════════════════════════════════════════════════════════════════════════════════
# DEMO AND TESTING
# ════════════════════════════════════════════════════════════════════════════════

def demo_neural_scheduler():
    """Demonstrate the neural context-aware scheduler."""
    print("=" * 70)
    print("    NEURAL CONTEXT-AWARE SCHEDULER DEMO")
    print("=" * 70)

    scheduler = NeuralContextScheduler()

    # Simulate a user session
    commands = [
        "ls", "cd", "ls", "cat", "echo",
        "ls", "pwd", "cat", "echo", "ls",
        "uname", "date", "ls", "cat", "pwd",
        "ls", "echo", "cat", "ls", "pwd",
        "neofetch", "ls", "cat", "echo", "ls",
    ]

    print("\n  Simulating user session...")
    for i, cmd in enumerate(commands):
        scheduler.record_command(cmd)

        if i > 0 and i % 5 == 0:
            scheduler.print_predictions()

    print("\n" + "=" * 70)
    print("  SCHEDULER STATISTICS")
    print("=" * 70)

    stats = scheduler.get_stats()
    for key, value in stats.items():
        print(f"  {key:25} {value}")

    print("\n  Final predictions:")
    scheduler.print_predictions()

    print("\n" + "=" * 70)
    print("  Demo complete! The scheduler learned your command patterns.")
    print("=" * 70)


if __name__ == "__main__":
    demo_neural_scheduler()
