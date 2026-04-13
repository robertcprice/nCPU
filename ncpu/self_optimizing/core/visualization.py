"""
Visualization Module for Self-Optimizing Engine

Provides plotting and visualization capabilities for:
- Training curves (success rate over iterations)
- Gradient signal heatmaps
- Benchmark comparison charts
- Pattern success bar charts
- Multi-objective Pareto plots
"""

import json
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# Try to import matplotlib - if not available, use ASCII fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class VisualizationConfig:
    """Configuration for visualizations"""
    output_dir: str = "outputs/visualizations"
    figsize: tuple = (10, 6)
    dpi: int = 100
    style: str = "seaborn-v0_8-darkgrid" if HAS_MATPLOTLIB else "ascii"


@dataclass
class TrainingProgress:
    """Training progress data for visualization"""
    iteration: int
    success_rate: float
    avg_attempts: float
    gradient_magnitude: float
    pattern_success: dict = field(default_factory=dict)
    loss: Optional[float] = None


class VisualizationEngine:
    """
    Visualization engine for self-optimizing engine metrics.

    Can work with or without matplotlib - falls back to ASCII/text output.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.history: list[TrainingProgress] = []
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def add_progress(self, progress: TrainingProgress):
        """Add a training progress snapshot"""
        self.history.append(progress)

    def plot_training_curve(self, save_path: Optional[str] = None) -> str:
        """
        Plot success rate and attempts over iterations.

        Returns:
            Path to saved plot or ASCII representation
        """
        if not self.history:
            return "No training history to plot"

        if not HAS_MATPLOTLIB:
            return self._ascii_training_curve()

        iterations = [p.iteration for p in self.history]
        success_rates = [p.success_rate * 100 for p in self.history]
        avg_attempts = [p.avg_attempts for p in self.history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figsize)

        # Success rate
        ax1.plot(iterations, success_rates, 'b-', linewidth=2, marker='o')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_xlabel('Iteration')
        ax1.set_title('Training Progress: Success Rate')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)

        # Average attempts
        ax2.plot(iterations, avg_attempts, 'r-', linewidth=2, marker='s')
        ax2.set_ylabel('Avg Attempts')
        ax2.set_xlabel('Iteration')
        ax2.set_title('Training Progress: Average Attempts')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = save_path or f"{self.config.output_dir}/training_curve.png"
        plt.savefig(save_path, dpi=self.config.dpi)
        plt.close()

        return save_path

    def _ascii_training_curve(self) -> str:
        """Generate ASCII art training curve"""
        if not self.history:
            return "No data"

        lines = ["\n=== Training Progress ===\n"]

        # Success rate
        max_rate = max(p.success_rate for p in self.history) or 1
        lines.append("Success Rate:")
        for p in self.history:
            bar_len = int(p.success_rate * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            lines.append(f"  Iter {p.iteration:3d}: |{bar}| {p.success_rate*100:5.1f}%")

        lines.append("\nAvg Attempts:")
        max_attempts = max(p.avg_attempts for p in self.history) or 1
        for p in self.history:
            bar_len = int((p.avg_attempts / max_attempts) * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            lines.append(f"  Iter {p.iteration:3d}: |{bar}| {p.avg_attempts:5.2f}")

        return "\n".join(lines)

    def plot_gradient_heatmap(self, save_path: Optional[str] = None) -> str:
        """
        Plot gradient signal heatmap across patterns.

        Returns:
            Path to saved plot or ASCII representation
        """
        if not self.history:
            return "No training history"

        # Collect pattern data
        all_patterns = set()
        for p in self.history:
            all_patterns.update(p.pattern_success.keys())

        if not all_patterns:
            return "No pattern data available"

        if not HAS_MATPLOTLIB:
            return self._ascii_gradient_heatmap(all_patterns)

        pattern_list = sorted(all_patterns)
        iterations = [p.iteration for p in self.history]
        data = []

        for p in self.history:
            row = [p.pattern_success.get(pat, 0) for pat in pattern_list]
            data.append(row)

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(data, aspect='auto', cmap='viridis')

        ax.set_yticks(range(len(iterations)))
        ax.set_yticklabels(iterations)
        ax.set_xticks(range(len(pattern_list)))
        ax.set_xticklabels(pattern_list, rotation=45, ha='right')
        ax.set_xlabel('Pattern')
        ax.set_ylabel('Iteration')
        ax.set_title('Gradient Signal Heatmap')

        plt.colorbar(im, ax=ax, label='Success Rate')
        plt.tight_layout()

        save_path = save_path or f"{self.config.output_dir}/gradient_heatmap.png"
        plt.savefig(save_path, dpi=self.config.dpi)
        plt.close()

        return save_path

    def _ascii_gradient_heatmap(self, patterns) -> str:
        """Generate ASCII heatmap"""
        lines = ["\n=== Gradient Signal Heatmap ===\n"]

        # Use characters for intensity
        chars = " .:-=+*#%@"

        for p in self.history:
            row = []
            for pat in sorted(patterns):
                val = p.pattern_success.get(pat, 0)
                idx = min(int(val * len(chars)), len(chars) - 1)
                row.append(chars[idx])
            lines.append(f"Iter {p.iteration:3d}: {''.join(row)}")

        lines.append(f"\nIntensity: {chars[0]} (low) -> {chars[-1]} (high)")

        return "\n".join(lines)

    def plot_benchmark_comparison(self, results: dict, save_path: Optional[str] = None) -> str:
        """
        Plot comparison chart for multiple approaches/providers.

        Args:
            results: Dict of {approach_name: BenchmarkResult}

        Returns:
            Path to saved plot or ASCII representation
        """
        if not results:
            return "No benchmark results"

        if not HAS_MATPLOTLIB:
            return self._ascii_benchmark_comparison(results)

        names = list(results.keys())
        success_rates = [r.success_rate * 100 for r in results.values()]
        avg_times = [r.avg_execution_time for r in results.values()]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figsize)

        # Success rate
        bars1 = ax1.bar(names, success_rates, color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'])
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Benchmark: Success Rate')
        ax1.set_ylim(0, 105)
        for bar, rate in zip(bars1, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')

        # Execution time
        bars2 = ax2.bar(names, avg_times, color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'])
        ax2.set_ylabel('Avg Execution Time (s)')
        ax2.set_title('Benchmark: Execution Time')
        for bar, t in zip(bars2, avg_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{t:.2f}s', ha='center', va='bottom')

        plt.xticks(rotation=15)
        plt.tight_layout()

        save_path = save_path or f"{self.config.output_dir}/benchmark_comparison.png"
        plt.savefig(save_path, dpi=self.config.dpi)
        plt.close()

        return save_path

    def _ascii_benchmark_comparison(self, results: dict) -> str:
        """Generate ASCII benchmark comparison"""
        lines = ["\n=== Benchmark Comparison ===\n"]

        names = list(results.keys())
        max_name_len = max(len(n) for n in names)

        # Header
        lines.append(f"{'Approach':<{max_name_len}} | {'Success':>10} | {'Avg Time':>10}")
        lines.append("-" * (max_name_len + 26))

        for name, result in results.items():
            lines.append(f"{name:<{max_name_len}} | {result.success_rate*100:>9.1f}% | {result.avg_execution_time:>9.2f}s")

        # Bar chart
        lines.append("\nSuccess Rate:")
        max_rate = max(r.success_rate for r in results.values()) or 1
        for name, result in results.items():
            bar_len = int((result.success_rate / max_rate) * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            lines.append(f"  {name:<{max_name_len}}: |{bar}| {result.success_rate*100:.1f}%")

        return "\n".join(lines)

    def plot_pareto_frontier(self, objectives: list[dict], save_path: Optional[str] = None) -> str:
        """
        Plot multi-objective Pareto frontier.

        Args:
            objectives: List of {name, accuracy, speed, memory}

        Returns:
            Path to saved plot or ASCII representation
        """
        if not objectives:
            return "No objective data"

        if not HAS_MATPLOTLIB:
            return self._ascii_pareto_frontier(objectives)

        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Extract data
        names = [o['name'] for o in objectives]
        accuracy = [o['accuracy'] * 100 for o in objectives]
        speed = [o['speed'] for o in objectives]

        # Plot
        for i, (name, acc, spd) in enumerate(zip(names, accuracy, speed)):
            ax.scatter(spd, acc, s=200, label=name, alpha=0.7)
            ax.annotate(name, (spd, acc), xytext=(5, 5), textcoords='offset points')

        ax.set_xlabel('Speed (tokens/sec)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Multi-Objective: Accuracy vs Speed')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = save_path or f"{self.config.output_dir}/pareto_frontier.png"
        plt.savefig(save_path, dpi=self.config.dpi)
        plt.close()

        return save_path

    def _ascii_pareto_frontier(self, objectives: list[dict]) -> str:
        """Generate ASCII pareto plot"""
        lines = ["\n=== Pareto Frontier: Accuracy vs Speed ===\n"]

        # Sort by speed
        sorted_obj = sorted(objectives, key=lambda x: x['speed'])

        max_name_len = max(len(o['name']) for o in objectives)
        max_speed = max(o['speed'] for o in objectives) or 1

        lines.append(f"{'Name':<{max_name_len}} | {'Speed':>8} | {'Accuracy':>10} | Visualization")
        lines.append("-" * (max_name_len + 35))

        for obj in sorted_obj:
            bar_len = int((obj['speed'] / max_speed) * 20)
            bar = "▓" * bar_len + "░" * (20 - bar_len)
            lines.append(
                f"{obj['name']:<{max_name_len}} | {obj['speed']:>8.2f} | "
                f"{obj['accuracy']*100:>9.1f}% | {bar}"
            )

        return "\n".join(lines)

    def save_history(self, path: str):
        """Save training history to JSON"""
        data = {
            "history": [
                {
                    "iteration": p.iteration,
                    "success_rate": p.success_rate,
                    "avg_attempts": p.avg_attempts,
                    "gradient_magnitude": p.gradient_magnitude,
                    "pattern_success": p.pattern_success,
                    "loss": p.loss,
                }
                for p in self.history
            ]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_history(self, path: str):
        """Load training history from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)

        self.history = [
            TrainingProgress(
                iteration=h["iteration"],
                success_rate=h["success_rate"],
                avg_attempts=h["avg_attempts"],
                gradient_magnitude=h["gradient_magnitude"],
                pattern_success=h.get("pattern_success", {}),
                loss=h.get("loss"),
            )
            for h in data.get("history", [])
        ]

    def generate_report(self) -> str:
        """Generate a comprehensive text report"""
        if not self.history:
            return "No data available"

        lines = [
            "=" * 60,
            "SELF-OPTIMIZING ENGINE - VISUALIZATION REPORT",
            "=" * 60,
            "",
        ]

        # Summary stats
        final = self.history[-1]
        initial = self.history[0]

        lines.extend([
            "SUMMARY",
            "-" * 40,
            f"Total iterations: {len(self.history)}",
            f"Initial success rate: {initial.success_rate*100:.1f}%",
            f"Final success rate: {final.success_rate*100:.1f}%",
            f"Improvement: {(final.success_rate - initial.success_rate)*100:+.1f}%",
            f"Final avg attempts: {final.avg_attempts:.2f}",
            "",
        ])

        # Add visualizations
        lines.extend([
            "VISUALIZATIONS",
            "-" * 40,
            self._ascii_training_curve(),
            "",
        ])

        if final.pattern_success:
            lines.extend([
                self._ascii_gradient_heatmap(final.pattern_success.keys()),
                "",
            ])

        return "\n".join(lines)


def demo():
    """Demo of visualization capabilities"""
    print("=== Visualization Demo ===\n")

    # Create sample data
    viz = VisualizationEngine()

    import random
    random.seed(42)

    # Generate training progress
    for i in range(20):
        progress = TrainingProgress(
            iteration=i,
            success_rate=0.3 + (i / 20) * 0.6 + random.uniform(-0.05, 0.05),
            avg_attempts=5 - (i / 20) * 3 + random.uniform(-0.2, 0.2),
            gradient_magnitude=random.uniform(0.1, 0.9),
            pattern_success={
                "fibonacci": 0.5 + i * 0.025,
                "factorial": 0.4 + i * 0.03,
                "palindrome": 0.6 + i * 0.02,
                "binary_search": 0.3 + i * 0.035,
            },
            loss=1.0 - (i / 20) * 0.8,
        )
        viz.add_progress(progress)

    # Print report
    print(viz.generate_report())


if __name__ == "__main__":
    demo()
