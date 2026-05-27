"""Neural Security Monitor — anomaly detection for the GPU OS.

Uses the syscall predictor's learned "normal" patterns to detect anomalous
command sequences that could indicate shell injection, privilege escalation,
or other security threats.

This is a novel application: using an online-learned bigram model as an
intrusion detection system with zero pre-training. The model learns
"normal" behavior from the first N commands, then flags deviations.

Architecture:
    1. Learning phase (first 20 syscalls): build normal transition table
    2. Detection phase: flag syscalls with probability < threshold
    3. Alert levels: INFO (unusual), WARNING (suspicious), CRITICAL (anomalous)

Usage:
    monitor = NeuralSecurityMonitor()
    monitor.observe(syscall_num)  # called on every syscall
    monitor.print_report()
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class SecurityAlert:
    """A single security alert."""
    level: str          # INFO, WARNING, CRITICAL
    syscall: int
    prev_syscall: int
    probability: float
    message: str


SYSCALL_NAMES = {
    63: "READ", 64: "WRITE", 93: "EXIT", 94: "EXIT_GROUP",
    214: "BRK", 57: "CLOSE", 56: "OPENAT", 62: "LSEEK",
    172: "GETPID", 220: "CLONE", 260: "WAIT4",
    59: "PIPE2", 24: "DUP3", 129: "KILL",
    300: "COMPILE", 301: "EXEC",
}

# Syscalls that are inherently sensitive
SENSITIVE_SYSCALLS = {
    220: "CLONE (fork)",
    129: "KILL",
    301: "EXEC",
}


class NeuralSecurityMonitor:
    """Online anomaly detection via learned syscall patterns.

    Learns the normal syscall transition probabilities during the first
    N observations, then flags deviations as potential security events.
    """

    def __init__(self, learning_window: int = 50, alert_threshold: float = 0.02):
        self.learning_window = learning_window
        self.alert_threshold = alert_threshold

        self.history: List[int] = []
        self.bigrams: Dict[int, Counter] = defaultdict(Counter)
        self.alerts: List[SecurityAlert] = []
        self.total_observations = 0
        self.in_learning_phase = True

        # Track sensitive syscall usage
        self.sensitive_counts: Counter = Counter()

    def observe(self, syscall_num: int):
        """Process a syscall observation."""
        self.total_observations += 1
        self.history.append(syscall_num)

        # Track sensitive syscalls
        if syscall_num in SENSITIVE_SYSCALLS:
            self.sensitive_counts[syscall_num] += 1

        # Update bigram model
        if len(self.history) >= 2:
            prev = self.history[-2]
            self.bigrams[prev][syscall_num] += 1

        # Switch from learning to detection after window
        if self.total_observations == self.learning_window:
            self.in_learning_phase = False

        # Detection phase: check if this transition is anomalous
        if not self.in_learning_phase and len(self.history) >= 2:
            prev = self.history[-2]
            if prev in self.bigrams:
                total_from_prev = sum(self.bigrams[prev].values())
                count = self.bigrams[prev].get(syscall_num, 0)
                prob = count / total_from_prev if total_from_prev > 0 else 0

                if prob < self.alert_threshold:
                    # Determine alert level
                    if syscall_num in SENSITIVE_SYSCALLS:
                        level = "CRITICAL"
                        msg = f"Rare sensitive syscall: {SENSITIVE_SYSCALLS[syscall_num]} after {SYSCALL_NAMES.get(prev, f'SYS_{prev}')}"
                    elif prob == 0:
                        level = "WARNING"
                        msg = f"Never-seen transition: {SYSCALL_NAMES.get(prev, f'SYS_{prev}')} → {SYSCALL_NAMES.get(syscall_num, f'SYS_{syscall_num}')}"
                    else:
                        level = "INFO"
                        msg = f"Unusual transition (p={prob:.3f}): {SYSCALL_NAMES.get(prev, f'SYS_{prev}')} → {SYSCALL_NAMES.get(syscall_num, f'SYS_{syscall_num}')}"

                    self.alerts.append(SecurityAlert(
                        level=level,
                        syscall=syscall_num,
                        prev_syscall=prev,
                        probability=prob,
                        message=msg,
                    ))

    def stats(self) -> Dict:
        """Return security monitoring statistics."""
        return {
            "total_observations": self.total_observations,
            "learning_window": self.learning_window,
            "in_learning_phase": self.in_learning_phase,
            "unique_transitions": sum(len(v) for v in self.bigrams.values()),
            "total_alerts": len(self.alerts),
            "critical_alerts": sum(1 for a in self.alerts if a.level == "CRITICAL"),
            "warning_alerts": sum(1 for a in self.alerts if a.level == "WARNING"),
            "info_alerts": sum(1 for a in self.alerts if a.level == "INFO"),
            "sensitive_syscalls": dict(self.sensitive_counts),
        }

    def print_report(self):
        """Print security monitoring report."""
        s = self.stats()
        print(f"\n  \033[1;37mNeural Security Monitor:\033[0m")
        print(f"    Observations:       {s['total_observations']}")
        print(f"    Learned transitions: {s['unique_transitions']}")
        print(f"    Alerts: {s['critical_alerts']} critical, {s['warning_alerts']} warning, {s['info_alerts']} info")

        if self.sensitive_counts:
            print(f"    Sensitive syscalls:")
            for num, count in self.sensitive_counts.most_common():
                name = SENSITIVE_SYSCALLS.get(num, f"SYS_{num}")
                print(f"      {name}: {count}")

        if self.alerts:
            print(f"    Recent alerts:")
            for alert in self.alerts[-5:]:  # last 5
                color = "\033[1;31m" if alert.level == "CRITICAL" else \
                        "\033[1;33m" if alert.level == "WARNING" else "\033[0;37m"
                print(f"      {color}[{alert.level}]\033[0m {alert.message}")

        if not self.alerts:
            print(f"    Status:             \033[1;32mNo anomalies detected\033[0m")
