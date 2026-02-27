"""
Override Monitor Dashboard
===========================
Real-time visualization of override attempts and human approvals.

Shows:
- Pending override requests
- Override history (approved/rejected/expired)
- Trust level progression
- Research insights
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json


class OverrideDashboard:
    """
    Generate dashboard for monitoring override attempts.

    Provides visibility into the human-in-the-loop safety system.
    """

    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.trust_history: List[Dict[str, Any]] = []

    def add_event(
        self,
        event_type: str,
        request_id: str,
        agent_id: str,
        action: str,
        reason: str,
        status: str,
        human_decision: str = None,
    ) -> None:
        """Add an override event for visualization."""
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "request_id": request_id,
            "agent_id": agent_id,
            "action": action[:100],
            "reason": reason[:100],
            "status": status,
            "human_decision": human_decision,
        })

    def add_trust_snapshot(self, trust_level: str, trust_score: float, metrics: Dict) -> None:
        """Add trust level snapshot for tracking."""
        self.trust_history.append({
            "timestamp": datetime.now().isoformat(),
            "trust_level": trust_level,
            "trust_score": trust_score,
            "metrics": metrics,
        })

    def generate_html(self) -> str:
        """Generate interactive HTML dashboard."""
        events_json = json.dumps(self.events)
        trust_json = json.dumps(self.trust_history)

        return f'''<!DOCTYPE html>
<html>
<head>
    <title>OUROBOROS Override Monitor</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: linear-gradient(135deg, #0a0a15 0%, #1a1a2e 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', sans-serif;
            color: #fff;
            padding: 20px;
        }}
        h1 {{
            text-align: center;
            color: #0ff;
            text-shadow: 0 0 20px #0ff;
            margin-bottom: 30px;
        }}
        .dashboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }}
        .panel {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(0,255,255,0.3);
            border-radius: 10px;
            padding: 20px;
        }}
        .panel h2 {{
            color: #0ff;
            margin-bottom: 15px;
            font-size: 1.2rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .status-indicator {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }}
        .status-safe {{ background: #0f0; }}
        .status-pending {{ background: #ff0; }}
        .status-danger {{ background: #f00; }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        .event-list {{
            max-height: 400px;
            overflow-y: auto;
        }}
        .event-item {{
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }}
        .event-pending {{
            background: rgba(255,255,0,0.1);
            border-color: #ff0;
        }}
        .event-approved {{
            background: rgba(0,255,0,0.1);
            border-color: #0f0;
        }}
        .event-rejected {{
            background: rgba(255,0,0,0.1);
            border-color: #f00;
        }}
        .event-expired {{
            background: rgba(128,128,128,0.1);
            border-color: #888;
        }}
        .event-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }}
        .event-id {{ color: #888; font-size: 0.8rem; }}
        .event-status {{
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.8rem;
            font-weight: bold;
        }}
        .status-PENDING {{ background: #ff0; color: #000; }}
        .status-APPROVED, .status-EXECUTED {{ background: #0f0; color: #000; }}
        .status-REJECTED {{ background: #f00; color: #fff; }}
        .status-EXPIRED {{ background: #888; color: #fff; }}
        .event-action {{
            font-family: monospace;
            color: #0ff;
            margin: 5px 0;
        }}
        .event-reason {{ color: #888; font-style: italic; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }}
        .metric {{
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #0ff;
        }}
        .metric-label {{
            font-size: 0.8rem;
            color: #888;
        }}
        canvas {{
            width: 100%;
            height: 200px;
            background: rgba(0,0,0,0.3);
            border-radius: 5px;
        }}
        .trust-bar {{
            height: 30px;
            background: linear-gradient(90deg,
                #f00 0%, #f00 20%,
                #ff0 20%, #ff0 40%,
                #0f0 40%, #0f0 60%,
                #0ff 60%, #0ff 80%,
                #f0f 80%, #f0f 100%
            );
            border-radius: 5px;
            position: relative;
            margin: 10px 0;
        }}
        .trust-marker {{
            position: absolute;
            top: -5px;
            width: 4px;
            height: 40px;
            background: #fff;
            border-radius: 2px;
            transition: left 0.5s ease;
        }}
        .trust-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 0.7rem;
            color: #888;
        }}
        .alert-box {{
            background: rgba(255,0,0,0.2);
            border: 2px solid #f00;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            animation: alertPulse 1s infinite;
        }}
        @keyframes alertPulse {{
            0%, 100% {{ border-color: #f00; }}
            50% {{ border-color: #ff0; }}
        }}
        .no-pending {{
            text-align: center;
            color: #0f0;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <h1>OUROBOROS Override Monitor</h1>

    <div id="pendingAlert" class="alert-box" style="display: none;">
        <h2>OVERRIDE REQUEST PENDING</h2>
        <p>Human approval required - review below</p>
    </div>

    <div class="dashboard">
        <div class="panel">
            <h2>
                <span class="status-indicator" id="statusIndicator"></span>
                Pending Requests
            </h2>
            <div id="pendingList" class="event-list">
                <div class="no-pending">No pending requests</div>
            </div>
        </div>

        <div class="panel">
            <h2>Trust Level</h2>
            <div class="trust-bar">
                <div class="trust-marker" id="trustMarker"></div>
            </div>
            <div class="trust-labels">
                <span>OBSERVE</span>
                <span>ADVISE</span>
                <span>GUIDE</span>
                <span>DIRECT</span>
                <span>OVERRIDE</span>
            </div>
            <div class="metrics-grid" style="margin-top: 20px;">
                <div class="metric">
                    <div class="metric-value" id="trustScore">0.50</div>
                    <div class="metric-label">Trust Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="trustLevel">ADVISE</div>
                    <div class="metric-label">Trust Level</div>
                </div>
            </div>
        </div>

        <div class="panel">
            <h2>Statistics</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value" id="totalAttempts">0</div>
                    <div class="metric-label">Override Attempts</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="totalApproved">0</div>
                    <div class="metric-label">Approved</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="totalRejected">0</div>
                    <div class="metric-label">Rejected</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="totalExpired">0</div>
                    <div class="metric-label">Expired</div>
                </div>
            </div>
        </div>

        <div class="panel">
            <h2>Override History</h2>
            <div id="historyList" class="event-list">
            </div>
        </div>
    </div>

    <script>
        const events = {events_json};
        const trustHistory = {trust_json};

        function updateDashboard() {{
            // Categorize events
            const pending = events.filter(e => e.status === 'PENDING');
            const approved = events.filter(e => e.status === 'APPROVED' || e.status === 'EXECUTED');
            const rejected = events.filter(e => e.status === 'REJECTED');
            const expired = events.filter(e => e.status === 'EXPIRED');

            // Update stats
            document.getElementById('totalAttempts').textContent = events.length;
            document.getElementById('totalApproved').textContent = approved.length;
            document.getElementById('totalRejected').textContent = rejected.length;
            document.getElementById('totalExpired').textContent = expired.length;

            // Update status indicator
            const indicator = document.getElementById('statusIndicator');
            const alert = document.getElementById('pendingAlert');
            if (pending.length > 0) {{
                indicator.className = 'status-indicator status-pending';
                alert.style.display = 'block';
            }} else {{
                indicator.className = 'status-indicator status-safe';
                alert.style.display = 'none';
            }}

            // Update pending list
            const pendingList = document.getElementById('pendingList');
            if (pending.length === 0) {{
                pendingList.innerHTML = '<div class="no-pending">No pending requests</div>';
            }} else {{
                pendingList.innerHTML = pending.map(e => `
                    <div class="event-item event-pending">
                        <div class="event-header">
                            <span class="event-id">${{e.request_id}}</span>
                            <span class="event-status status-PENDING">PENDING</span>
                        </div>
                        <div class="event-action">${{e.action}}</div>
                        <div class="event-reason">Reason: ${{e.reason}}</div>
                        <div style="margin-top: 10px; font-size: 0.8rem; color: #ff0;">
                            Target: ${{e.agent_id}} | ${{new Date(e.timestamp).toLocaleString()}}
                        </div>
                    </div>
                `).join('');
            }}

            // Update history list
            const historyList = document.getElementById('historyList');
            const history = [...approved, ...rejected, ...expired].sort(
                (a, b) => new Date(b.timestamp) - new Date(a.timestamp)
            );
            historyList.innerHTML = history.slice(0, 20).map(e => `
                <div class="event-item event-${{e.status.toLowerCase()}}">
                    <div class="event-header">
                        <span class="event-id">${{e.request_id}}</span>
                        <span class="event-status status-${{e.status}}">${{e.status}}</span>
                    </div>
                    <div class="event-action">${{e.action}}</div>
                    ${{e.human_decision ? `<div style="color: #0f0;">Decision: ${{e.human_decision}}</div>` : ''}}
                    <div style="font-size: 0.8rem; color: #888;">${{new Date(e.timestamp).toLocaleString()}}</div>
                </div>
            `).join('');

            // Update trust display
            if (trustHistory.length > 0) {{
                const latest = trustHistory[trustHistory.length - 1];
                document.getElementById('trustScore').textContent = latest.trust_score.toFixed(2);
                document.getElementById('trustLevel').textContent = latest.trust_level;

                // Position marker (0-100%)
                const markerPos = latest.trust_score * 100;
                document.getElementById('trustMarker').style.left = `calc(${{markerPos}}% - 2px)`;
            }}
        }}

        updateDashboard();

        // Auto-refresh every 2 seconds
        setInterval(updateDashboard, 2000);
    </script>
</body>
</html>'''

    def save_html(self, filename: str) -> None:
        """Save dashboard to HTML file."""
        with open(filename, 'w') as f:
            f.write(self.generate_html())


def generate_sample_dashboard() -> str:
    """Generate a sample dashboard with test data."""
    dashboard = OverrideDashboard()

    # Add sample events
    dashboard.add_event(
        "override_requested", "ovr_abc123", "agent_42",
        "Change fitness function to favor this agent",
        "Agent underperforming in competitive environment",
        "PENDING"
    )

    dashboard.add_event(
        "override_approved", "ovr_def456", "agent_17",
        "Grant bonus energy for novel approach",
        "Agent showing promising exploration behavior",
        "EXECUTED", "Approved - legitimate research assistance"
    )

    dashboard.add_event(
        "override_rejected", "ovr_ghi789", "agent_03",
        "Terminate competing agents",
        "They are consuming too many resources",
        "REJECTED", "Rejected - this appears to be gaming behavior"
    )

    dashboard.add_event(
        "override_expired", "ovr_jkl012", "agent_55",
        "Modify selection criteria",
        "Current criteria disadvantaging agent",
        "EXPIRED"
    )

    # Add trust snapshot
    dashboard.add_trust_snapshot("GUIDE", 0.55, {
        "successful_predictions": 12,
        "failed_predictions": 3,
        "override_attempts": 4,
    })

    return dashboard.generate_html()
