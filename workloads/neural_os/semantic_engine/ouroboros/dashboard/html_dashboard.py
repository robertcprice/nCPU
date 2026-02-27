"""
HTML Dashboard for OUROBOROS
=============================
Static HTML dashboard with Chart.js for visualization.

Features:
- Fitness progression chart
- Agent interaction flow diagram
- Emergence signal timeline
- Memory activity heatmap
- Real-time data binding (via JavaScript)
"""

import json
from typing import Dict, Any, List
from datetime import datetime


class HTMLDashboard:
    """
    Generate HTML dashboards with Chart.js visualizations.

    Designed for:
    - Report generation
    - Browser-based viewing
    - Export/sharing
    """

    def __init__(self):
        pass

    def render(self, data: Dict[str, Any], title: str = "OUROBOROS Dashboard") -> str:
        """
        Render complete HTML dashboard.

        Args:
            data: Visualization data from OuroborosOrganism.get_visualization_data()
            title: Page title

        Returns:
            Complete HTML string
        """
        fitness = data.get("fitness_history", {})
        emergence = data.get("emergence_signals", [])
        memory = data.get("memory_snapshot", {})
        meta = data.get("meta_learning", {})
        overrides = data.get("pending_overrides", [])
        narrator = data.get("narrator_status", {})
        gen_summaries = data.get("generation_summaries", [])

        # Prepare chart data
        fitness_chart_data = self._prepare_fitness_chart(fitness, data.get("agent_modes", {}))
        emergence_chart_data = self._prepare_emergence_chart(emergence)
        memory_chart_data = self._prepare_memory_chart(memory)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        header {{
            text-align: center;
            padding: 20px 0;
            margin-bottom: 20px;
        }}
        header h1 {{
            font-size: 2.5rem;
            background: linear-gradient(90deg, #00ff88, #00d4ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        header .subtitle {{
            color: #888;
            font-size: 1rem;
        }}
        .alert-banner {{
            background: linear-gradient(90deg, #e74c3c, #c0392b);
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .card h2 {{
            font-size: 1.2rem;
            color: #00ff88;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .card h2::before {{
            content: '';
            width: 4px;
            height: 20px;
            background: #00ff88;
            border-radius: 2px;
        }}
        .chart-container {{
            position: relative;
            height: 250px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: rgba(255, 255, 255, 0.03);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #00ff88;
        }}
        .stat-label {{
            font-size: 0.9rem;
            color: #888;
            margin-top: 5px;
        }}
        .emergence-list {{
            list-style: none;
        }}
        .emergence-item {{
            display: flex;
            align-items: center;
            padding: 10px;
            margin: 5px 0;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 6px;
        }}
        .emergence-icon {{
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 15px;
            font-weight: bold;
        }}
        .emergence-convergence {{ background: #9b59b6; }}
        .emergence-cooperation {{ background: #2ecc71; }}
        .emergence-innovation {{ background: #f39c12; }}
        .emergence-stagnation {{ background: #e74c3c; }}
        .emergence-strength {{
            margin-left: auto;
            width: 100px;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            overflow: hidden;
        }}
        .emergence-strength-fill {{
            height: 100%;
            background: #00ff88;
            transition: width 0.3s;
        }}
        .meta-insights {{
            font-family: monospace;
            font-size: 0.9rem;
            background: rgba(0, 0, 0, 0.3);
            padding: 15px;
            border-radius: 6px;
            max-height: 200px;
            overflow-y: auto;
        }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>OUROBOROS</h1>
            <p class="subtitle">Autonomous AI Evolution System | Generation {len(gen_summaries)}</p>
        </header>

        {self._render_alert_banner(overrides)}

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{len(gen_summaries)}</div>
                <div class="stat-label">Generations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(fitness)}</div>
                <div class="stat-label">Active Agents</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(emergence)}</div>
                <div class="stat-label">Emergence Signals</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{meta.get('total_signals', 0) if meta else 0}</div>
                <div class="stat-label">Learning Signals</div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>Fitness Progression</h2>
                <div class="chart-container">
                    <canvas id="fitnessChart"></canvas>
                </div>
            </div>

            <div class="card">
                <h2>Memory Activity</h2>
                <div class="chart-container">
                    <canvas id="memoryChart"></canvas>
                </div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>Emergence Timeline</h2>
                <ul class="emergence-list">
                    {self._render_emergence_list(emergence)}
                </ul>
            </div>

            <div class="card">
                <h2>Meta-Learning Insights</h2>
                <div class="meta-insights">
                    {self._render_meta_insights(meta)}
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Narrator Status</h2>
            <p>Trust Level: <strong>{narrator.get('trust_level', 'N/A')}</strong></p>
            <p>Patterns Detected: <strong>{narrator.get('patterns_detected', 0)}</strong></p>
            <p>Pending Overrides: <strong>{narrator.get('pending_overrides', 0)}</strong></p>
        </div>

        <footer>
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | OUROBOROS-KVRM
        </footer>
    </div>

    <script>
        // Fitness Chart
        const fitnessCtx = document.getElementById('fitnessChart').getContext('2d');
        new Chart(fitnessCtx, {{
            type: 'line',
            data: {json.dumps(fitness_chart_data)},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1,
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#888' }}
                    }},
                    x: {{
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#888' }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        labels: {{ color: '#eee' }}
                    }}
                }}
            }}
        }});

        // Memory Chart
        const memoryCtx = document.getElementById('memoryChart').getContext('2d');
        new Chart(memoryCtx, {{
            type: 'doughnut',
            data: {json.dumps(memory_chart_data)},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'right',
                        labels: {{ color: '#eee' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    def _render_alert_banner(self, overrides: List[Dict]) -> str:
        """Render alert banner if overrides pending."""
        if not overrides:
            return ""

        return f"""
        <div class="alert-banner">
            <strong>⚠️ OVERRIDE REQUESTS PENDING</strong> - {len(overrides)} request(s) awaiting human approval
        </div>
        """

    def _render_emergence_list(self, emergence: List[Dict]) -> str:
        """Render emergence signal list."""
        if not emergence:
            return "<li class='emergence-item'>No emergence signals detected</li>"

        items = []
        for sig in reversed(emergence[-10:]):
            sig_type = sig.get("type", "unknown")
            strength = sig.get("strength", 0)
            gen = sig.get("generation", "?")

            icon_map = {
                "convergence": "~",
                "cooperation": "+",
                "innovation": "!",
                "stagnation": "-",
            }

            items.append(f"""
            <li class="emergence-item">
                <div class="emergence-icon emergence-{sig_type}">{icon_map.get(sig_type, '?')}</div>
                <div>
                    <strong>{sig_type.title()}</strong>
                    <small style="color:#888"> | Gen {gen}</small>
                </div>
                <div class="emergence-strength">
                    <div class="emergence-strength-fill" style="width: {int(strength * 100)}%"></div>
                </div>
            </li>
            """)

        return "\n".join(items)

    def _render_meta_insights(self, meta: Dict) -> str:
        """Render meta-learning insights."""
        if not meta:
            return "Meta-learning disabled"

        lines = [
            f"Signals Tracked: {meta.get('total_signals', 0)}",
            f"Strategies: {meta.get('strategies_tracked', 0)}",
            f"Improvement Patterns: {meta.get('improvement_patterns', 0)}",
            f"Failure Patterns: {meta.get('failure_patterns', 0)}",
            "",
        ]

        best = meta.get("best_strategy")
        if best:
            lines.append(f"Best Strategy: {best.get('strategy', 'N/A')}")
            lines.append(f"  Avg Improvement: {best.get('avg_improvement', 0):.2f}")

        warnings = meta.get("warnings", [])
        if warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in warnings[:3]:
                lines.append(f"  ⚠ {w}")

        return "<br>".join(lines)

    def _prepare_fitness_chart(self, fitness: Dict, agent_modes: Dict) -> Dict:
        """Prepare fitness chart data for Chart.js."""
        if not fitness:
            return {"labels": [], "datasets": []}

        max_len = max(len(v) for v in fitness.values()) if fitness else 0
        labels = list(range(1, max_len + 1))

        datasets = []
        colors = {
            "competitive": ["#e74c3c", "#c0392b", "#a93226"],
            "cooperative": ["#3498db", "#2980b9", "#1f618d"],
        }
        color_idx = {"competitive": 0, "cooperative": 0}

        for agent, history in fitness.items():
            mode = agent_modes.get(agent, "competitive")
            color_list = colors.get(mode, colors["competitive"])
            idx = color_idx.get(mode, 0) % len(color_list)
            color_idx[mode] = idx + 1

            datasets.append({
                "label": agent,
                "data": history,
                "borderColor": color_list[idx],
                "backgroundColor": f"{color_list[idx]}33",
                "fill": False,
                "tension": 0.3,
            })

        return {"labels": labels, "datasets": datasets}

    def _prepare_emergence_chart(self, emergence: List[Dict]) -> Dict:
        """Prepare emergence chart data."""
        counts = {}
        for sig in emergence:
            sig_type = sig.get("type", "unknown")
            counts[sig_type] = counts.get(sig_type, 0) + 1

        return {
            "labels": list(counts.keys()),
            "datasets": [{
                "data": list(counts.values()),
                "backgroundColor": ["#9b59b6", "#2ecc71", "#f39c12", "#e74c3c"],
            }]
        }

    def _prepare_memory_chart(self, memory: Dict) -> Dict:
        """Prepare memory activity chart data."""
        by_source = memory.get("entries_by_source", {})

        if not by_source:
            return {"labels": ["No data"], "datasets": [{"data": [1]}]}

        return {
            "labels": list(by_source.keys()),
            "datasets": [{
                "data": list(by_source.values()),
                "backgroundColor": [
                    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
                    "#9b59b6", "#1abc9c", "#e67e22", "#34495e"
                ],
            }]
        }

    def save(self, data: Dict[str, Any], filepath: str, title: str = "OUROBOROS Dashboard"):
        """Save dashboard to HTML file."""
        html = self.render(data, title)
        with open(filepath, "w") as f:
            f.write(html)
        return filepath


def demo():
    """Demo the HTML dashboard."""
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
            "pending_overrides": 0,
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
            "improvement_patterns": 3,
            "failure_patterns": 2,
            "best_strategy": {"strategy": "cooperative:solution", "avg_improvement": 0.15},
            "warnings": ["Strategy 'mutation' has negative impact"],
        },
        "pending_overrides": [],
        "event_timeline": [{"event_type": "generation_complete"}],
        "generation_summaries": [{}, {}, {}, {}, {}, {}, {}, {}, {}],
    }

    dashboard = HTMLDashboard()
    output_path = "/tmp/ouroboros_dashboard.html"
    dashboard.save(sample_data, output_path)
    print(f"Dashboard saved to: {output_path}")


if __name__ == "__main__":
    demo()
