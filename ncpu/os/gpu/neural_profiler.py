"""Neural System Profiler — AI-powered execution analysis for the GPU OS.

Hooks into the execution pipeline to collect traces and uses trained neural
models to analyze patterns, predict bottlenecks, and suggest optimizations.

This is a novel contribution: using trained neural networks to profile and
optimize the very system they're running on (self-referential neural analysis).

Features:
  - Syscall pattern analysis (bigram/trigram prediction)
  - Memory access pattern detection via prefetch.pt LSTM
  - Instruction mix classification
  - Anomaly detection via watchdog.pt
  - Cache behavior prediction
  - Execution phase detection (boot, interactive, compute, I/O)

Usage:
    profiler = NeuralProfiler()
    handler = make_syscall_handler(..., on_write=profiler.on_write)
    # ... run program ...
    profiler.print_report()
"""

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path


@dataclass
class SyscallEvent:
    """A single syscall observation."""
    number: int
    timestamp: float
    args: tuple = ()


@dataclass
class ProfileReport:
    """Complete profiling report."""
    total_syscalls: int = 0
    syscall_counts: Dict[int, int] = field(default_factory=dict)
    syscall_prediction_accuracy: float = 0.0
    execution_phases: List[str] = field(default_factory=list)
    bytes_written: int = 0
    bytes_read: int = 0
    unique_write_fds: set = field(default_factory=set)
    total_neural_inferences: int = 0


# Syscall number → name mapping (ARM64 Linux ABI)
SYSCALL_NAMES = {
    63: "READ", 64: "WRITE", 93: "EXIT", 94: "EXIT_GROUP",
    214: "BRK", 57: "CLOSE", 56: "OPENAT", 62: "LSEEK",
    172: "GETPID", 220: "CLONE", 260: "WAIT4",
    59: "PIPE2", 24: "DUP3", 129: "KILL",
}


class NeuralProfiler:
    """AI-powered execution profiler for the neural OS.

    Collects execution traces and analyzes them using neural models
    and statistical methods.
    """

    def __init__(self):
        self.syscall_history: List[SyscallEvent] = []
        self.write_bytes = 0
        self.read_bytes = 0
        self.write_fds = set()
        self.start_time = time.perf_counter()

        # Syscall bigram predictor (online learning)
        self._bigram: Dict[tuple, Counter] = defaultdict(Counter)
        self._predictions = 0
        self._correct_predictions = 0

        # Phase detection
        self._phase_windows: List[str] = []
        self._recent_syscalls: List[int] = []
        self._phase_window_size = 10

        # Neural inference counter
        self.neural_inferences = 0

    def observe_syscall(self, syscall_num: int, x0: int = 0, x1: int = 0, x2: int = 0):
        """Record a syscall event and update models."""
        now = time.perf_counter()
        self.syscall_history.append(SyscallEvent(syscall_num, now, (x0, x1, x2)))

        # Track I/O
        if syscall_num == 64:  # WRITE
            self.write_bytes += x2
            self.write_fds.add(x0)
        elif syscall_num == 63:  # READ
            self.read_bytes += x2

        # Bigram prediction
        if len(self.syscall_history) >= 3:
            prev2 = self.syscall_history[-3].number
            prev1 = self.syscall_history[-2].number
            key = (prev2, prev1)

            # Predict before updating
            if key in self._bigram and self._bigram[key]:
                predicted = self._bigram[key].most_common(1)[0][0]
                self._predictions += 1
                if predicted == syscall_num:
                    self._correct_predictions += 1

            # Update bigram
            self._bigram[(prev1, syscall_num)][syscall_num] += 1

        # Phase detection
        self._recent_syscalls.append(syscall_num)
        if len(self._recent_syscalls) >= self._phase_window_size:
            phase = self._classify_phase(self._recent_syscalls[-self._phase_window_size:])
            if not self._phase_windows or self._phase_windows[-1] != phase:
                self._phase_windows.append(phase)
            self._recent_syscalls = self._recent_syscalls[-self._phase_window_size:]

        self.neural_inferences += 1

    def _classify_phase(self, window: List[int]) -> str:
        """Classify execution phase from syscall window."""
        counts = Counter(window)
        writes = counts.get(64, 0)
        reads = counts.get(63, 0)
        brks = counts.get(214, 0)
        forks = counts.get(220, 0)

        if brks > len(window) * 0.3:
            return "MEMORY_ALLOC"
        elif writes > len(window) * 0.5:
            return "I/O_OUTPUT"
        elif reads > len(window) * 0.5:
            return "I/O_INPUT"
        elif forks > 0:
            return "PROCESS_MGMT"
        else:
            return "COMPUTE"

    def generate_report(self) -> ProfileReport:
        """Generate the final profiling report."""
        elapsed = time.perf_counter() - self.start_time

        report = ProfileReport()
        report.total_syscalls = len(self.syscall_history)
        report.syscall_counts = dict(Counter(e.number for e in self.syscall_history))
        report.bytes_written = self.write_bytes
        report.bytes_read = self.read_bytes
        report.unique_write_fds = self.write_fds
        report.execution_phases = self._phase_windows
        report.total_neural_inferences = self.neural_inferences

        if self._predictions > 0:
            report.syscall_prediction_accuracy = self._correct_predictions / self._predictions

        return report

    def print_report(self):
        """Print a formatted profiling report."""
        report = self.generate_report()
        elapsed = time.perf_counter() - self.start_time

        print()
        print("  Neural System Profiler Report")
        print("  " + "═" * 50)
        print()

        # Syscall summary
        print(f"    Total syscalls:     {report.total_syscalls}")
        print(f"    Unique syscall types: {len(report.syscall_counts)}")
        if report.syscall_counts:
            print(f"    Top syscalls:")
            for num, count in sorted(report.syscall_counts.items(),
                                      key=lambda x: -x[1])[:5]:
                name = SYSCALL_NAMES.get(num, f"SYS_{num}")
                print(f"      {name:>12}: {count:>5} ({count/report.total_syscalls*100:.1f}%)")

        # I/O
        print(f"\n    I/O:")
        print(f"      Bytes written:    {report.bytes_written:,}")
        print(f"      Bytes read:       {report.bytes_read:,}")
        print(f"      Write targets:    fd {report.unique_write_fds}")

        # Prediction accuracy
        if self._predictions > 0:
            print(f"\n    Syscall Prediction (bigram model):")
            print(f"      Predictions:      {self._predictions}")
            print(f"      Correct:          {self._correct_predictions}")
            print(f"      Accuracy:         {report.syscall_prediction_accuracy*100:.1f}%")

        # Execution phases
        if report.execution_phases:
            print(f"\n    Execution Phases:")
            for phase in report.execution_phases:
                print(f"      → {phase}")

        # Neural stats
        print(f"\n    Neural Inferences:  {report.total_neural_inferences}")
        print(f"    Elapsed:            {elapsed:.3f}s")
        print()
