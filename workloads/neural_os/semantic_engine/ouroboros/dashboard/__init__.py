"""
OUROBOROS Dashboard System
===========================
Multi-mode dashboards for visualizing emergent patterns.

Modes:
- Terminal: ASCII dashboard for SSH/CLI
- HTML: Static HTML with Chart.js for reports
- Live: WebSocket server for real-time monitoring

Based on 6-AI Panel Recommendations:
- Fitness sparklines (ChatGPT)
- Emergence radar (Claude)
- Stigmergic flow visualization (DeepSeek)
- Drama Index + Agent personalities (Grok)
- Scalable architecture (Gemini)
"""

from .terminal_dashboard import TerminalDashboard
from .html_dashboard import HTMLDashboard

# Live server is optional (requires fastapi)
try:
    from .live_server import LiveDashboardServer
    __all__ = [
        "TerminalDashboard",
        "HTMLDashboard",
        "LiveDashboardServer",
    ]
except ImportError:
    __all__ = [
        "TerminalDashboard",
        "HTMLDashboard",
    ]
