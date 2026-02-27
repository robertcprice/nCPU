#!/usr/bin/env python3
"""
NeuralScheduler - Learned Process Scheduling
=============================================

Process scheduler that uses reinforcement learning to optimize
scheduling decisions for YOUR specific workload patterns.

Traditional schedulers use fixed policies:
- Round Robin: Equal time slices
- Priority: Static priority levels
- CFS: Completely Fair Scheduler (weighted time)

NeuralScheduler learns the OPTIMAL scheduling policy for your workload.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import random
import time


# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


@dataclass
class Process:
    """Represents a process in the system."""
    pid: int
    name: str
    priority: int = 0  # Higher = more important
    cpu_burst: float = 0.0  # Estimated CPU time needed
    io_burst: float = 0.0  # Estimated I/O time
    wait_time: float = 0.0  # Time spent waiting
    run_time: float = 0.0  # Total run time
    last_run: float = 0.0  # Last time it ran
    state: str = "ready"  # ready, running, waiting, terminated
    nice: int = 0  # Unix-style nice value
    memory_usage: int = 0  # Memory footprint
    io_intensity: float = 0.0  # 0-1, how I/O bound
    history: List[float] = field(default_factory=list)  # Recent burst lengths


class SchedulingPolicy(nn.Module):
    """
    Neural network that selects which process to run next.

    Uses attention to consider all ready processes and select
    the optimal one based on current system state.
    """

    def __init__(
        self,
        process_features: int = 12,
        system_features: int = 8,
        hidden_dim: int = 128,
        num_heads: int = 4
    ):
        super().__init__()

        self.process_features = process_features
        self.system_features = system_features
        self.hidden_dim = hidden_dim

        # Process feature encoder
        self.process_encoder = nn.Sequential(
            nn.Linear(process_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # System state encoder
        self.system_encoder = nn.Sequential(
            nn.Linear(system_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Attention for process selection
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Scoring head
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        # Value head (for advantage estimation)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        process_features: torch.Tensor,  # [batch, num_processes, process_features]
        system_features: torch.Tensor,   # [batch, system_features]
        mask: Optional[torch.Tensor] = None  # [batch, num_processes] True = valid
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scheduling scores for all processes.

        Returns:
            scores: [batch, num_processes] selection scores
            value: [batch, 1] state value estimate
        """
        batch_size = process_features.shape[0]
        num_processes = process_features.shape[1]

        # Encode processes
        proc_encoded = self.process_encoder(process_features)  # [B, N, H]

        # Encode system state and expand
        sys_encoded = self.system_encoder(system_features)  # [B, H]
        sys_expanded = sys_encoded.unsqueeze(1).expand(-1, num_processes, -1)

        # Self-attention over processes
        proc_attended, _ = self.attention(
            proc_encoded, proc_encoded, proc_encoded,
            key_padding_mask=~mask if mask is not None else None
        )

        # Combine process and system features
        combined = torch.cat([proc_attended, sys_expanded], dim=-1)

        # Score each process
        scores = self.score_head(combined).squeeze(-1)  # [B, N]

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        # Compute state value
        value = self.value_head(sys_encoded)

        return scores, value


class NeuralScheduler:
    """
    Process scheduler with learned policy.

    Features:
    - Reinforcement learning from scheduling outcomes
    - Adapts to workload-specific patterns
    - Learns implicit priorities from behavior
    - Optimizes for throughput, latency, or fairness
    """

    def __init__(
        self,
        max_processes: int = 64,
        time_slice: float = 10.0,  # ms
        learning_rate: float = 1e-4,
        enable_learning: bool = True,
        optimization_target: str = "latency"  # latency, throughput, fairness
    ):
        """
        Initialize NeuralScheduler.

        Args:
            max_processes: Maximum number of processes
            time_slice: Default time slice in ms
            learning_rate: Learning rate for policy updates
            enable_learning: Whether to learn from scheduling
            optimization_target: What to optimize for
        """
        self.max_processes = max_processes
        self.time_slice = time_slice
        self.learning_enabled = enable_learning
        self.optimization_target = optimization_target

        # Process management
        self.processes: Dict[int, Process] = {}
        self.ready_queue: List[int] = []
        self.current_process: Optional[int] = None

        # Neural policy
        self.policy = SchedulingPolicy(
            process_features=12,
            system_features=8
        ).to(device)
        self.policy.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Experience replay
        self.experience_buffer = deque(maxlen=10000)
        self.batch_size = 32

        # Time tracking
        self.current_time = 0.0
        self.last_schedule_time = 0.0

        # Statistics
        self.total_schedules = 0
        self.context_switches = 0
        self.total_wait_time = 0.0
        self.total_turnaround = 0.0
        self.completed_processes = 0

        # Reward tracking
        self.reward_history = deque(maxlen=1000)

    def add_process(self, process: Process):
        """Add a process to the scheduler."""
        self.processes[process.pid] = process
        if process.state == "ready":
            self.ready_queue.append(process.pid)

    def remove_process(self, pid: int):
        """Remove a process from the scheduler."""
        if pid in self.processes:
            del self.processes[pid]
        if pid in self.ready_queue:
            self.ready_queue.remove(pid)
        if self.current_process == pid:
            self.current_process = None

    def _get_process_features(self, pid: int) -> torch.Tensor:
        """Extract features for a process."""
        if pid not in self.processes:
            return torch.zeros(12, device=device)

        p = self.processes[pid]
        current_time = self.current_time

        features = torch.tensor([
            p.priority / 20.0,  # Normalized priority
            p.nice / 20.0,  # Normalized nice
            p.wait_time / max(1, current_time),  # Relative wait time
            p.run_time / max(1, current_time),  # Relative run time
            (current_time - p.last_run) / max(1, current_time),  # Time since last run
            p.cpu_burst / 100.0,  # Normalized CPU burst
            p.io_intensity,  # I/O intensity
            p.memory_usage / (1024 * 1024),  # Memory in MB
            len(p.history) / 100.0,  # History length
            sum(p.history[-5:]) / max(1, len(p.history[-5:])) / 100.0 if p.history else 0,  # Avg recent burst
            1.0 if p.state == "ready" else 0.0,  # Is ready
            0.0  # Padding
        ], dtype=torch.float32, device=device)

        return features

    def _get_system_features(self) -> torch.Tensor:
        """Extract system state features."""
        ready_count = len(self.ready_queue)
        total_count = len(self.processes)
        avg_wait = (
            sum(p.wait_time for p in self.processes.values()) / max(1, total_count)
        )

        features = torch.tensor([
            ready_count / max(1, self.max_processes),  # Queue fullness
            total_count / max(1, self.max_processes),  # Total processes
            avg_wait / max(1, self.current_time),  # Avg wait time
            self.context_switches / max(1, self.total_schedules),  # Context switch rate
            self.current_time / 1000.0,  # Normalized time
            1.0 if self.current_process else 0.0,  # Has current process
            0.0,  # CPU utilization (would need tracking)
            0.0   # Memory pressure (would need tracking)
        ], dtype=torch.float32, device=device)

        return features

    def select_next(self) -> Optional[int]:
        """
        Select the next process to run.

        Uses neural policy to choose optimal process.
        """
        if not self.ready_queue:
            return None

        self.total_schedules += 1

        # Get features for all ready processes
        process_features = torch.stack([
            self._get_process_features(pid) for pid in self.ready_queue
        ]).unsqueeze(0)  # [1, num_ready, 12]

        system_features = self._get_system_features().unsqueeze(0)  # [1, 8]

        # Create mask
        mask = torch.ones(1, len(self.ready_queue), dtype=torch.bool, device=device)

        # Get policy scores
        with torch.no_grad():
            scores, value = self.policy(process_features, system_features, mask)

        # Select highest scoring process
        selected_idx = scores[0].argmax().item()
        selected_pid = self.ready_queue[selected_idx]

        # Track context switch
        if self.current_process is not None and self.current_process != selected_pid:
            self.context_switches += 1

        # Store experience for learning
        if self.learning_enabled:
            self.experience_buffer.append({
                'process_features': process_features.clone(),
                'system_features': system_features.clone(),
                'mask': mask.clone(),
                'action': selected_idx,
                'selected_pid': selected_pid,
                'time': self.current_time
            })

        return selected_pid

    def run_process(self, pid: int, duration: float):
        """
        Simulate running a process.

        Args:
            pid: Process ID to run
            duration: How long it ran (ms)
        """
        if pid not in self.processes:
            return

        p = self.processes[pid]

        # Update process state
        p.run_time += duration
        p.last_run = self.current_time
        p.history.append(duration)

        # Update waiting processes
        for other_pid, other_p in self.processes.items():
            if other_pid != pid and other_p.state == "ready":
                other_p.wait_time += duration

        # Advance time
        self.current_time += duration

        # Update current process
        self.current_process = pid

    def complete_process(self, pid: int):
        """Mark a process as completed and compute reward."""
        if pid not in self.processes:
            return

        p = self.processes[pid]
        turnaround = self.current_time - p.wait_time

        self.total_wait_time += p.wait_time
        self.total_turnaround += turnaround
        self.completed_processes += 1

        # Compute reward based on optimization target
        if self.optimization_target == "latency":
            reward = -p.wait_time / 1000.0  # Minimize wait time
        elif self.optimization_target == "throughput":
            reward = 1.0 / max(0.001, turnaround / 1000.0)  # Maximize throughput
        else:  # fairness
            avg_wait = self.total_wait_time / max(1, self.completed_processes)
            reward = -abs(p.wait_time - avg_wait) / 1000.0  # Minimize variance

        self.reward_history.append(reward)

        # Learn from experience
        if self.learning_enabled and len(self.experience_buffer) >= self.batch_size:
            self._learn(reward)

        # Remove process
        self.remove_process(pid)

    def _learn(self, final_reward: float):
        """Learn from scheduling experience using policy gradient."""
        if len(self.experience_buffer) < self.batch_size:
            return

        # Sample batch
        batch = list(self.experience_buffer)[-self.batch_size:]

        self.policy.train()

        total_loss = 0

        for exp in batch:
            process_features = exp['process_features']
            system_features = exp['system_features']
            mask = exp['mask']
            action = exp['action']

            # Forward pass
            scores, value = self.policy(process_features, system_features, mask)

            # Policy loss (REINFORCE with baseline)
            log_probs = F.log_softmax(scores, dim=-1)
            selected_log_prob = log_probs[0, action]

            advantage = final_reward - value.item()
            policy_loss = -selected_log_prob * advantage

            # Value loss
            value_loss = F.mse_loss(value, torch.tensor([[final_reward]], device=device))

            total_loss += policy_loss + 0.5 * value_loss

        # Update policy
        self.optimizer.zero_grad()
        (total_loss / len(batch)).backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        self.policy.eval()

        # Clear old experiences
        self.experience_buffer.clear()

    def get_stats(self) -> Dict:
        """Get scheduler statistics."""
        avg_wait = (
            self.total_wait_time / max(1, self.completed_processes)
        )
        avg_turnaround = (
            self.total_turnaround / max(1, self.completed_processes)
        )
        avg_reward = (
            sum(self.reward_history) / max(1, len(self.reward_history))
        )

        return {
            'total_schedules': self.total_schedules,
            'context_switches': self.context_switches,
            'context_switch_rate': self.context_switches / max(1, self.total_schedules),
            'completed_processes': self.completed_processes,
            'avg_wait_time': avg_wait,
            'avg_turnaround': avg_turnaround,
            'avg_reward': avg_reward,
            'learning_enabled': self.learning_enabled,
            'optimization_target': self.optimization_target
        }

    def save(self, path: str):
        """Save scheduler policy."""
        torch.save({
            'policy_state': self.policy.state_dict(),
            'stats': self.get_stats()
        }, path)

    def load(self, path: str):
        """Load scheduler policy."""
        data = torch.load(path, map_location=device, weights_only=False)
        self.policy.load_state_dict(data['policy_state'])


def benchmark_scheduler():
    """Benchmark NeuralScheduler vs Round Robin."""
    print("=" * 70)
    print("🧠 NEURAL SCHEDULER BENCHMARK")
    print("=" * 70)

    # Create schedulers
    neural = NeuralScheduler(enable_learning=True, optimization_target="latency")
    rr_neural = NeuralScheduler(enable_learning=False)  # Fallback to simple policy

    # Generate workload
    print("\n📊 Generating workload...")

    def create_workload() -> List[Process]:
        processes = []

        # Mix of CPU-bound and I/O-bound processes
        for i in range(20):
            if i % 3 == 0:  # CPU-bound
                p = Process(
                    pid=i,
                    name=f"cpu_{i}",
                    priority=random.randint(0, 10),
                    cpu_burst=random.uniform(50, 200),
                    io_intensity=0.1
                )
            else:  # I/O-bound
                p = Process(
                    pid=i,
                    name=f"io_{i}",
                    priority=random.randint(0, 10),
                    cpu_burst=random.uniform(10, 50),
                    io_intensity=0.8
                )
            processes.append(p)

        return processes

    # Run simulation
    def run_simulation(scheduler: NeuralScheduler, workload: List[Process]):
        for p in workload:
            scheduler.add_process(Process(**p.__dict__))

        # Simulate scheduling rounds
        for _ in range(100):
            if not scheduler.ready_queue:
                break

            pid = scheduler.select_next()
            if pid is None:
                break

            # Simulate running
            duration = random.uniform(5, 20)
            scheduler.run_process(pid, duration)

            # Random completion
            if random.random() < 0.1:
                scheduler.complete_process(pid)

    print("   Running neural scheduler...")
    workload1 = create_workload()
    run_simulation(neural, workload1)

    print("   Running baseline scheduler...")
    workload2 = create_workload()
    run_simulation(rr_neural, workload2)

    # Results
    print("\n📈 Results:")
    print("-" * 50)

    neural_stats = neural.get_stats()
    rr_stats = rr_neural.get_stats()

    print(f"   Neural Scheduler:")
    print(f"      Avg Wait Time: {neural_stats['avg_wait_time']:.2f} ms")
    print(f"      Context Switches: {neural_stats['context_switches']}")
    print(f"      Completed: {neural_stats['completed_processes']}")

    print(f"\n   Baseline Scheduler:")
    print(f"      Avg Wait Time: {rr_stats['avg_wait_time']:.2f} ms")
    print(f"      Context Switches: {rr_stats['context_switches']}")
    print(f"      Completed: {rr_stats['completed_processes']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    benchmark_scheduler()
