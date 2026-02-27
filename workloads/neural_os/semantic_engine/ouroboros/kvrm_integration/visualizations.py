"""
Data-Driven Visualizations for OUROBOROS-KVRM
==============================================
Creates visualizations from ACTUAL SharedKVMemory data patterns.

NOT canned templates - these show real:
- Fitness progression over generations
- Emergence signal patterns
- Memory key relationships
- Agent interaction flows
- Meta-learning insights
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime


class OuroborosVisualizer:
    """
    Creates data-driven visualizations from OUROBOROS data.

    All visualizations derive directly from experiment data,
    not pre-made templates.
    """

    def __init__(self, data: Dict[str, Any]):
        """
        Initialize with visualization data from OuroborosOrganism.

        Args:
            data: Output from organism.get_visualization_data()
        """
        self.data = data

    def render_terminal_dashboard(self) -> str:
        """
        Render ASCII dashboard for terminal display.

        Returns:
            Multi-line string showing current state.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("           OUROBOROS-KVRM DASHBOARD")
        lines.append("=" * 70)

        # Fitness History Chart (ASCII sparkline)
        lines.append("\n[FITNESS PROGRESSION]")
        fitness = self.data.get("fitness_history", {})
        if fitness:
            for agent, history in fitness.items():
                if history:
                    sparkline = self._ascii_sparkline(history[-20:])  # Last 20 gens
                    latest = history[-1] if history else 0
                    mode = self.data.get("agent_modes", {}).get(agent, "?")
                    lines.append(f"  {agent[:15]:<15} [{mode[0]}] {sparkline} {latest:.2f}")
        else:
            lines.append("  No fitness data yet")

        # Emergence Signals
        lines.append("\n[EMERGENCE SIGNALS]")
        emergence = self.data.get("emergence_signals", [])
        if emergence:
            for sig in emergence[-5:]:  # Last 5
                symbol = {
                    "convergence": "~",
                    "cooperation": "+",
                    "innovation": "!",
                    "stagnation": "-",
                }.get(sig.get("type", ""), "?")
                strength_bar = "#" * int(sig.get("strength", 0) * 10)
                lines.append(f"  [{symbol}] Gen {sig.get('generation', '?'):>3} {strength_bar:<10} {sig.get('type', '')}")
        else:
            lines.append("  No emergence detected")

        # Memory Snapshot
        lines.append("\n[SHARED MEMORY STATE]")
        memory = self.data.get("memory_snapshot", {})
        if memory:
            lines.append(f"  Entries: {memory.get('entry_count', 0)}")
            by_source = memory.get("entries_by_source", {})
            for source, count in sorted(by_source.items(), key=lambda x: -x[1])[:5]:
                bar = "|" * min(count, 20)
                lines.append(f"    {source[:12]:<12} {bar} ({count})")

        # Meta-Learning Summary
        lines.append("\n[META-LEARNING]")
        meta = self.data.get("meta_learning")
        if meta:
            lines.append(f"  Signals tracked: {meta.get('total_signals', 0)}")
            lines.append(f"  Strategies: {meta.get('strategies_tracked', 0)}")
            best = meta.get("best_strategy")
            if best:
                lines.append(f"  Best strategy: {best.get('strategy', '?')[:30]}")
            warnings = meta.get("warnings", [])
            if warnings:
                lines.append(f"  Warnings: {len(warnings)}")
                for w in warnings[:2]:
                    lines.append(f"    ! {w[:50]}")
        else:
            lines.append("  Meta-learning disabled")

        # Pending Overrides (HUMAN ATTENTION NEEDED)
        overrides = self.data.get("pending_overrides", [])
        if overrides:
            lines.append("\n" + "!" * 70)
            lines.append("  PENDING OVERRIDE REQUESTS - HUMAN APPROVAL REQUIRED")
            lines.append("!" * 70)
            for ovr in overrides:
                lines.append(f"  ID: {ovr.get('request_id', '?')}")
                lines.append(f"  Reason: {ovr.get('reason', '?')[:50]}")
                lines.append(f"  Action: {ovr.get('action', '?')[:50]}")
                lines.append("")

        # Narrator Status
        lines.append("\n[NARRATOR STATUS]")
        narrator = self.data.get("narrator_status", {})
        if narrator:
            lines.append(f"  Trust level: {narrator.get('trust_level', '?')}")
            lines.append(f"  Generation: {narrator.get('generation', 0)}")
            lines.append(f"  Patterns detected: {narrator.get('patterns_detected', 0)}")

        lines.append("\n" + "=" * 70)
        lines.append(f"  Last updated: {datetime.now().strftime('%H:%M:%S')}")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _ascii_sparkline(self, values: List[float], width: int = 20) -> str:
        """Create ASCII sparkline from values."""
        if not values:
            return " " * width

        chars = " ▁▂▃▄▅▆▇█"
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val > min_val else 1

        sparkline = ""
        for v in values[-width:]:
            idx = int((v - min_val) / range_val * (len(chars) - 1))
            sparkline += chars[min(idx, len(chars) - 1)]

        return sparkline.ljust(width)

    def render_json_report(self) -> str:
        """
        Render JSON report for programmatic consumption.

        Returns:
            JSON string with complete experiment data.
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "fitness_summary": self._summarize_fitness(),
            "emergence_summary": self._summarize_emergence(),
            "meta_learning_summary": self.data.get("meta_learning"),
            "memory_stats": self.data.get("memory_snapshot", {}).get("stats"),
            "agent_modes": self.data.get("agent_modes"),
            "pending_overrides": self.data.get("pending_overrides"),
            "generation_summaries": self.data.get("generation_summaries"),
        }
        return json.dumps(report, indent=2, default=str)

    def _summarize_fitness(self) -> Dict[str, Any]:
        """Summarize fitness history."""
        fitness = self.data.get("fitness_history", {})
        if not fitness:
            return {}

        summary = {}
        for agent, history in fitness.items():
            if history:
                summary[agent] = {
                    "latest": history[-1],
                    "max": max(history),
                    "min": min(history),
                    "avg": sum(history) / len(history),
                    "generations": len(history),
                    "trend": "up" if len(history) > 1 and history[-1] > history[0] else "down",
                }
        return summary

    def _summarize_emergence(self) -> Dict[str, Any]:
        """Summarize emergence signals."""
        signals = self.data.get("emergence_signals", [])
        if not signals:
            return {"total": 0}

        by_type = {}
        for sig in signals:
            sig_type = sig.get("type", "unknown")
            if sig_type not in by_type:
                by_type[sig_type] = {"count": 0, "avg_strength": 0, "generations": []}
            by_type[sig_type]["count"] += 1
            by_type[sig_type]["avg_strength"] += sig.get("strength", 0)
            by_type[sig_type]["generations"].append(sig.get("generation", 0))

        # Compute averages
        for sig_type in by_type:
            count = by_type[sig_type]["count"]
            by_type[sig_type]["avg_strength"] /= count

        return {
            "total": len(signals),
            "by_type": by_type,
        }

    def render_html_dashboard(self) -> str:
        """
        Render HTML dashboard with charts.

        Returns:
            HTML string with embedded Chart.js visualizations.
        """
        fitness = self.data.get("fitness_history", {})
        emergence = self.data.get("emergence_signals", [])
        meta = self.data.get("meta_learning", {})

        # Prepare chart data
        labels = []
        datasets = []
        max_gen = 0

        for agent, history in fitness.items():
            if history:
                max_gen = max(max_gen, len(history))
                mode = self.data.get("agent_modes", {}).get(agent, "unknown")
                color = "#e74c3c" if mode == "competitive" else "#3498db"
                datasets.append({
                    "label": agent,
                    "data": history,
                    "borderColor": color,
                    "fill": False,
                })

        labels = list(range(1, max_gen + 1))

        # Emergence markers
        emergence_data = [
            {"x": e.get("generation", 0), "type": e.get("type", ""), "strength": e.get("strength", 0)}
            for e in emergence
        ]

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>OUROBOROS-KVRM Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Courier New', monospace; background: #1a1a2e; color: #eee; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: #16213e; border-radius: 8px; padding: 20px; margin: 10px 0; }}
        .card h2 {{ color: #00ff88; margin-top: 0; }}
        .chart-container {{ position: relative; height: 300px; }}
        .alert {{ background: #e74c3c; padding: 15px; border-radius: 8px; margin: 10px 0; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }}
        .stat {{ background: #0f3460; padding: 15px; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 24px; color: #00ff88; }}
        .stat-label {{ font-size: 12px; color: #888; }}
        .emergence {{ display: flex; gap: 10px; flex-wrap: wrap; }}
        .emergence-tag {{ padding: 5px 10px; border-radius: 4px; font-size: 12px; }}
        .convergence {{ background: #9b59b6; }}
        .cooperation {{ background: #2ecc71; }}
        .innovation {{ background: #f39c12; }}
        .stagnation {{ background: #e74c3c; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>OUROBOROS-KVRM Dashboard</h1>
        <p>Data-driven visualization from SharedKVMemory</p>

        {"".join(f'<div class="alert">OVERRIDE REQUEST PENDING: {o.get("reason", "")[:100]}</div>' for o in self.data.get("pending_overrides", []))}

        <div class="stats">
            <div class="stat">
                <div class="stat-value">{len(self.data.get("generation_summaries", []))}</div>
                <div class="stat-label">Generations</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(fitness)}</div>
                <div class="stat-label">Agents</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(emergence)}</div>
                <div class="stat-label">Emergence Signals</div>
            </div>
            <div class="stat">
                <div class="stat-value">{meta.get("total_signals", 0) if meta else 0}</div>
                <div class="stat-label">Learning Signals</div>
            </div>
        </div>

        <div class="card">
            <h2>Fitness Progression</h2>
            <div class="chart-container">
                <canvas id="fitnessChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Emergence Signals</h2>
            <div class="emergence">
                {"".join(f'<span class="emergence-tag {e.get("type", "")}">[Gen {e.get("generation", "?")}] {e.get("type", "")} ({e.get("strength", 0):.2f})</span>' for e in emergence[-10:])}
            </div>
        </div>

        <div class="card">
            <h2>Meta-Learning Insights</h2>
            <pre>{json.dumps(meta, indent=2) if meta else "Disabled"}</pre>
        </div>

        <div class="card">
            <h2>Memory State</h2>
            <pre>{json.dumps(self.data.get("memory_snapshot", {}), indent=2, default=str)}</pre>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('fitnessChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(labels)},
                datasets: {json.dumps(datasets)}
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{ beginAtZero: true, max: 1 }}
                }},
                plugins: {{
                    legend: {{ labels: {{ color: '#eee' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
        return html


def print_generation_update(gen_result: Dict[str, Any]) -> None:
    """
    Print a generation update to terminal.

    This is meant to be used as a callback for on_generation.
    Shows what actually happened in this generation.
    """
    gen = gen_result.get("generation", "?")
    best_agent = gen_result.get("best_agent", "none")
    best_fitness = gen_result.get("best_fitness", 0)

    # Handle narrator observation (may be string or dict)
    narrator_obs_raw = gen_result.get("narrator_observation", "")
    if isinstance(narrator_obs_raw, dict):
        narrator_obs = str(narrator_obs_raw.get("value", narrator_obs_raw))[:80]
    else:
        narrator_obs = str(narrator_obs_raw)[:80]

    patterns = gen_result.get("patterns_detected", [])
    overrides = gen_result.get("override_requests", [])

    print(f"\n{'='*60}")
    print(f"GENERATION {gen}")
    print(f"{'='*60}")

    # Solutions
    solutions = gen_result.get("solutions", [])
    if solutions:
        print("\n[AGENT SOLUTIONS]")
        for sol in sorted(solutions, key=lambda s: s.get("fitness", 0), reverse=True):
            agent = sol.get("agent", "?")
            fitness = sol.get("fitness", 0)
            mode = sol.get("mode", "?")
            bar = "#" * int(fitness * 20)
            print(f"  {agent:<20} [{mode[0]}] {bar:<20} {fitness:.3f}")

    # Narrator
    if narrator_obs:
        print(f"\n[NARRATOR OBSERVES]")
        print(f"  {narrator_obs}")

    guidance_raw = gen_result.get("narrator_guidance", "")
    if guidance_raw:
        if isinstance(guidance_raw, dict):
            guidance = str(guidance_raw.get("value", guidance_raw))[:100]
        else:
            guidance = str(guidance_raw)[:100]
        print(f"\n[NARRATOR GUIDANCE]")
        print(f"  {guidance}")

    # Patterns
    if patterns:
        print(f"\n[PATTERNS DETECTED]")
        # Handle patterns as list or dict
        pattern_list = patterns if isinstance(patterns, list) else [patterns]
        for p in pattern_list[:3]:
            if isinstance(p, dict):
                print(f"  - {p.get('type', '?')}: {str(p.get('description', ''))[:50]}")
            else:
                print(f"  - {str(p)[:50]}")

    # Emergence
    emergence = gen_result.get("emergence_signals", [])
    if emergence:
        print(f"\n[EMERGENCE SIGNALS]")
        emergence_list = emergence if isinstance(emergence, list) else [emergence]
        for e in emergence_list:
            if isinstance(e, dict):
                print(f"  ! {e.get('signal_type', '?')}: {str(e.get('description', ''))[:50]}")
            else:
                print(f"  ! {str(e)[:50]}")

    # Override requests
    if overrides:
        print(f"\n{'!'*60}")
        print("  OVERRIDE REQUEST - HUMAN APPROVAL REQUIRED")
        print(f"{'!'*60}")
        for ovr in overrides:
            print(f"  Request ID: {ovr.get('request_id', '?')}")
            print(f"  Reason: {ovr.get('reason', '?')}")

    # Meta-learning
    meta = gen_result.get("meta_learning")
    if meta and isinstance(meta, dict):
        warnings = meta.get("warnings", [])
        if warnings and isinstance(warnings, list):
            print(f"\n[META-LEARNING WARNINGS]")
            for w in warnings[:2]:
                print(f"  ! {str(w)[:50]}")

    print(f"\n  Best: {best_agent} at {best_fitness:.3f}")
    print(f"  Duration: {gen_result.get('duration_ms', 0):.1f}ms")
