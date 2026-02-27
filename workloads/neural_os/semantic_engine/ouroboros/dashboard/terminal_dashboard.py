"""
Terminal ASCII Dashboard for OUROBOROS
=======================================
Real-time ASCII visualization for terminal/SSH environments.

Shows:
- Fitness progression sparklines
- Agent flow diagram
- Emergence indicators
- Memory heatmap
- Override alerts
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict


class TerminalDashboard:
    """
    ASCII dashboard for terminal visualization.

    Designed for:
    - SSH sessions
    - CI/CD logs
    - Low-bandwidth monitoring
    """

    # Unicode characters for visualization
    BLOCKS = " ▁▂▃▄▅▆▇█"
    ARROWS = {"up": "↑", "down": "↓", "flat": "→", "spike": "⚡"}
    INDICATORS = {
        "good": "●",
        "warning": "◐",
        "bad": "○",
        "alert": "◉",
    }

    def __init__(self, width: int = 80):
        self.width = width
        self.history: List[Dict] = []

    def render(self, data: Dict[str, Any]) -> str:
        """
        Render full dashboard.

        Args:
            data: Visualization data from OuroborosOrganism.get_visualization_data()

        Returns:
            Multi-line ASCII string
        """
        lines = []

        # Header
        lines.extend(self._render_header(data))

        # Alerts (override requests)
        alerts = self._render_alerts(data)
        if alerts:
            lines.extend(alerts)

        # Fitness progression
        lines.extend(self._render_fitness(data))

        # Agent status grid
        lines.extend(self._render_agent_grid(data))

        # Memory activity
        lines.extend(self._render_memory_activity(data))

        # Emergence signals
        lines.extend(self._render_emergence(data))

        # Meta-learning insights
        lines.extend(self._render_meta_learning(data))

        # Footer
        lines.extend(self._render_footer(data))

        return "\n".join(lines)

    def _render_header(self, data: Dict) -> List[str]:
        """Render dashboard header."""
        gen = len(data.get("generation_summaries", []))
        timestamp = datetime.now().strftime("%H:%M:%S")

        return [
            "╔" + "═" * (self.width - 2) + "╗",
            "║" + " OUROBOROS DASHBOARD ".center(self.width - 2) + "║",
            "║" + f" Generation: {gen} | {timestamp} ".center(self.width - 2) + "║",
            "╠" + "═" * (self.width - 2) + "╣",
        ]

    def _render_alerts(self, data: Dict) -> List[str]:
        """Render alert section for override requests."""
        overrides = data.get("pending_overrides", [])
        if not overrides:
            return []

        lines = [
            "║" + " ⚠️  OVERRIDE REQUESTS PENDING ".center(self.width - 2) + "║",
            "╟" + "─" * (self.width - 2) + "╢",
        ]

        for ovr in overrides[:3]:
            reason = str(ovr.get("reason", "?"))[:40]
            lines.append("║" + f"  {reason}".ljust(self.width - 2) + "║")

        lines.append("╟" + "─" * (self.width - 2) + "╢")
        return lines

    def _render_fitness(self, data: Dict) -> List[str]:
        """Render fitness progression with sparklines."""
        fitness = data.get("fitness_history", {})
        agent_modes = data.get("agent_modes", {})

        lines = [
            "║" + " FITNESS PROGRESSION ".ljust(self.width - 2) + "║",
            "╟" + "─" * (self.width - 2) + "╢",
        ]

        if not fitness:
            lines.append("║" + "  No fitness data yet".ljust(self.width - 2) + "║")
        else:
            for agent, history in fitness.items():
                if not history:
                    continue

                # Get mode indicator
                mode = agent_modes.get(agent, "?")
                mode_char = "C" if mode == "competitive" else "O" if mode == "cooperative" else "N"

                # Create sparkline
                sparkline = self._sparkline(history[-20:], width=30)

                # Current value and trend
                current = history[-1]
                trend = self._get_trend(history)
                trend_char = self.ARROWS.get(trend, "?")

                # Format line
                agent_short = agent[:12].ljust(12)
                value_str = f"{current:.2f}"
                line = f"  [{mode_char}] {agent_short} {sparkline} {value_str} {trend_char}"
                lines.append("║" + line.ljust(self.width - 2) + "║")

        lines.append("╟" + "─" * (self.width - 2) + "╢")
        return lines

    def _render_agent_grid(self, data: Dict) -> List[str]:
        """Render agent status grid."""
        lines = [
            "║" + " AGENT STATUS ".ljust(self.width - 2) + "║",
            "╟" + "─" * (self.width - 2) + "╢",
        ]

        narrator = data.get("narrator_status", {})
        agent_modes = data.get("agent_modes", {})

        # Simple agent grid
        agents = [a for a in agent_modes.keys() if "narrator" not in a.lower()]
        if not agents:
            lines.append("║" + "  No agents".ljust(self.width - 2) + "║")
        else:
            # Grid layout
            grid_line = "  "
            for agent in agents[:6]:
                mode = agent_modes.get(agent, "?")
                indicator = "🔴" if mode == "competitive" else "🔵"
                short_name = agent.split("_")[-1] if "_" in agent else agent[:3]
                grid_line += f"{indicator}{short_name} "

            lines.append("║" + grid_line.ljust(self.width - 2) + "║")

            # Legend
            lines.append("║" + "  🔴=competitive 🔵=cooperative".ljust(self.width - 2) + "║")

        # Narrator status
        if narrator:
            trust = narrator.get("trust_level", "?")
            patterns = narrator.get("patterns_detected", 0)
            lines.append("║" + f"  Narrator: {trust} | Patterns: {patterns}".ljust(self.width - 2) + "║")

        lines.append("╟" + "─" * (self.width - 2) + "╢")
        return lines

    def _render_memory_activity(self, data: Dict) -> List[str]:
        """Render memory activity heatmap."""
        memory = data.get("memory_snapshot", {})

        lines = [
            "║" + " MEMORY ACTIVITY ".ljust(self.width - 2) + "║",
            "╟" + "─" * (self.width - 2) + "╢",
        ]

        by_source = memory.get("entries_by_source", {})
        total = memory.get("entry_count", 0)

        if not by_source:
            lines.append("║" + "  No memory entries".ljust(self.width - 2) + "║")
        else:
            # Bar chart of entries by source
            max_count = max(by_source.values()) if by_source else 1
            for source, count in sorted(by_source.items(), key=lambda x: -x[1])[:5]:
                bar_len = int(count / max_count * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                line = f"  {source[:10]:<10} {bar} {count}"
                lines.append("║" + line.ljust(self.width - 2) + "║")

            lines.append("║" + f"  Total entries: {total}".ljust(self.width - 2) + "║")

        lines.append("╟" + "─" * (self.width - 2) + "╢")
        return lines

    def _render_emergence(self, data: Dict) -> List[str]:
        """Render emergence signals."""
        signals = data.get("emergence_signals", [])

        lines = [
            "║" + " EMERGENCE SIGNALS ".ljust(self.width - 2) + "║",
            "╟" + "─" * (self.width - 2) + "╢",
        ]

        if not signals:
            lines.append("║" + "  No emergence detected".ljust(self.width - 2) + "║")
        else:
            # Show recent signals
            for sig in signals[-5:]:
                sig_type = sig.get("type", "?")
                strength = sig.get("strength", 0)
                gen = sig.get("generation", "?")

                # Icon by type
                icon = {
                    "convergence": "~",
                    "cooperation": "+",
                    "innovation": "!",
                    "stagnation": "-",
                }.get(sig_type, "?")

                # Strength bar
                bar = "█" * int(strength * 10) + "░" * (10 - int(strength * 10))

                line = f"  [{icon}] Gen {gen:>3} {bar} {sig_type}"
                lines.append("║" + line.ljust(self.width - 2) + "║")

        lines.append("╟" + "─" * (self.width - 2) + "╢")
        return lines

    def _render_meta_learning(self, data: Dict) -> List[str]:
        """Render meta-learning insights."""
        meta = data.get("meta_learning")

        lines = [
            "║" + " META-LEARNING ".ljust(self.width - 2) + "║",
            "╟" + "─" * (self.width - 2) + "╢",
        ]

        if not meta:
            lines.append("║" + "  Meta-learning disabled".ljust(self.width - 2) + "║")
        else:
            signals = meta.get("total_signals", 0)
            strategies = meta.get("strategies_tracked", 0)
            warnings = meta.get("warnings", [])

            lines.append("║" + f"  Signals: {signals} | Strategies: {strategies}".ljust(self.width - 2) + "║")

            best = meta.get("best_strategy")
            if best:
                strat_name = str(best.get("strategy", "?"))[:30]
                lines.append("║" + f"  Best: {strat_name}".ljust(self.width - 2) + "║")

            if warnings:
                lines.append("║" + f"  ⚠ {len(warnings)} warnings".ljust(self.width - 2) + "║")

        lines.append("╟" + "─" * (self.width - 2) + "╢")
        return lines

    def _render_footer(self, data: Dict) -> List[str]:
        """Render dashboard footer."""
        events = data.get("event_timeline", [])
        last_event = events[-1] if events else {}
        last_type = last_event.get("event_type", "none")

        return [
            "║" + f" Last event: {last_type}".ljust(self.width - 2) + "║",
            "╚" + "═" * (self.width - 2) + "╝",
        ]

    def _sparkline(self, values: List[float], width: int = 20) -> str:
        """Create ASCII sparkline from values."""
        if not values:
            return " " * width

        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val > min_val else 1

        result = ""
        for v in values[-width:]:
            idx = int((v - min_val) / range_val * (len(self.BLOCKS) - 1))
            result += self.BLOCKS[min(idx, len(self.BLOCKS) - 1)]

        return result.ljust(width)

    def _get_trend(self, values: List[float], window: int = 3) -> str:
        """Determine trend from recent values."""
        if len(values) < 2:
            return "flat"

        recent = values[-window:]
        if len(recent) < 2:
            return "flat"

        diff = recent[-1] - recent[0]

        if abs(diff) < 0.01:
            return "flat"
        elif diff > 0.1:
            return "spike"
        elif diff > 0:
            return "up"
        else:
            return "down"

    def live_update(self, data: Dict[str, Any], clear: bool = True):
        """
        Print dashboard with optional screen clear.

        Args:
            data: Visualization data
            clear: Whether to clear screen first
        """
        if clear:
            os.system('cls' if os.name == 'nt' else 'clear')

        print(self.render(data))


def demo():
    """Demo the terminal dashboard."""
    # Sample data
    sample_data = {
        "fitness_history": {
            "competitive_0": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.55, 0.6, 0.65],
            "competitive_1": [0.1, 0.15, 0.25, 0.35, 0.4, 0.45, 0.5, 0.52, 0.55],
            "cooperative_0": [0.1, 0.2, 0.35, 0.45, 0.55, 0.6, 0.65, 0.7, 0.75],
        },
        "agent_modes": {
            "competitive_0": "competitive",
            "competitive_1": "competitive",
            "cooperative_0": "cooperative",
        },
        "narrator_status": {
            "trust_level": "GUIDE",
            "patterns_detected": 3,
        },
        "memory_snapshot": {
            "entry_count": 45,
            "entries_by_source": {
                "competitive_0": 15,
                "competitive_1": 12,
                "cooperative_0": 10,
                "meta_narrator": 5,
                "meta_learner": 3,
            },
        },
        "emergence_signals": [
            {"type": "convergence", "strength": 0.3, "generation": 5},
            {"type": "cooperation", "strength": 0.7, "generation": 7},
            {"type": "innovation", "strength": 0.9, "generation": 9},
        ],
        "meta_learning": {
            "total_signals": 27,
            "strategies_tracked": 5,
            "best_strategy": {"strategy": "cooperative:solution_gen_8"},
            "warnings": ["Strategy 'competitive:mutation' has negative impact"],
        },
        "pending_overrides": [],
        "event_timeline": [{"event_type": "generation_complete"}],
        "generation_summaries": [{}, {}, {}, {}, {}, {}, {}, {}, {}],
    }

    dashboard = TerminalDashboard()
    print(dashboard.render(sample_data))


if __name__ == "__main__":
    demo()
