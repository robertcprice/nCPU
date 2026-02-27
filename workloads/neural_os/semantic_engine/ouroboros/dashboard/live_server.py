"""
Live WebSocket Dashboard Server for OUROBOROS
===============================================
Real-time dashboard with WebSocket streaming.

Features:
- FastAPI backend
- WebSocket real-time updates
- REST API for historical data
- Embedded HTML dashboard
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass, asdict
import threading

# Add paths
sys.path.insert(0, '/Users/bobbyprice/projects/KVRM/kvrm-ecosystem')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn")


@dataclass
class DashboardEvent:
    """Event for real-time streaming."""
    event_type: str
    timestamp: str
    generation: int
    data: Dict[str, Any]


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: Dict):
        """Send message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)

        for conn in disconnected:
            self.active_connections.discard(conn)


class LiveDashboardServer:
    """
    Live dashboard server with WebSocket streaming.

    Usage:
        server = LiveDashboardServer()
        server.start(port=8080)

        # In your evolution loop:
        server.push_update(generation_data)
    """

    def __init__(self):
        if not HAS_FASTAPI:
            raise ImportError("FastAPI required: pip install fastapi uvicorn")

        self.app = FastAPI(title="OUROBOROS Dashboard")
        self.manager = ConnectionManager()

        # State
        self.current_state: Dict[str, Any] = {}
        self.event_history: List[DashboardEvent] = []
        self.max_history = 1000

        # Event loop reference for thread-safe broadcasting
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            return self._generate_dashboard_html()

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.manager.connect(websocket)
            try:
                # Send current state on connect
                if self.current_state:
                    await websocket.send_json({
                        "type": "state",
                        "data": self.current_state
                    })

                # Keep connection alive
                while True:
                    try:
                        data = await asyncio.wait_for(
                            websocket.receive_text(),
                            timeout=30.0
                        )
                        # Handle ping/pong or commands
                        if data == "ping":
                            await websocket.send_text("pong")
                    except asyncio.TimeoutError:
                        # Send heartbeat
                        await websocket.send_json({"type": "heartbeat"})

            except WebSocketDisconnect:
                self.manager.disconnect(websocket)

        @self.app.get("/api/state")
        async def get_state():
            return self.current_state

        @self.app.get("/api/events")
        async def get_events(limit: int = 100):
            return self.event_history[-limit:]

        @self.app.get("/api/fitness")
        async def get_fitness():
            return self.current_state.get("fitness_history", {})

        @self.app.get("/api/emergence")
        async def get_emergence():
            return self.current_state.get("emergence_signals", [])

    def _safe_broadcast(self, message: Dict):
        """Thread-safe broadcast to WebSocket clients."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.manager.broadcast(message),
                self._loop
            )

    def push_update(self, data: Dict[str, Any]):
        """Push state update to all connected clients."""
        self.current_state = data

        event = DashboardEvent(
            event_type="state_update",
            timestamp=datetime.now().isoformat(),
            generation=len(data.get("generation_summaries", [])),
            data=data
        )
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]

        # Broadcast to WebSocket clients (thread-safe)
        self._safe_broadcast({
            "type": "update",
            "timestamp": event.timestamp,
            "generation": event.generation,
            "data": data
        })

    def push_event(self, event_type: str, event_data: Dict[str, Any], generation: int = 0):
        """Push a specific event."""
        event = DashboardEvent(
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            generation=generation,
            data=event_data
        )
        self.event_history.append(event)

        # Broadcast to WebSocket clients (thread-safe)
        self._safe_broadcast({
            "type": "event",
            "event_type": event_type,
            "timestamp": event.timestamp,
            "generation": generation,
            "data": event_data
        })

    def _generate_dashboard_html(self) -> str:
        """Generate the embedded dashboard HTML."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OUROBOROS Live Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 100%);
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background: rgba(0, 255, 136, 0.1);
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(0, 255, 136, 0.3);
        }
        .header h1 {
            font-size: 1.5rem;
            color: #00ff88;
        }
        .status {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        .status-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        .status-dot.connected { background: #00ff88; }
        .status-dot.disconnected { background: #e74c3c; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            padding: 20px;
        }
        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .card h2 {
            color: #00ff88;
            font-size: 1rem;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .card h2::before {
            content: '';
            width: 4px;
            height: 16px;
            background: #00ff88;
            border-radius: 2px;
        }
        .chart-container {
            position: relative;
            height: 200px;
        }
        .stats-row {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat {
            flex: 1;
            background: rgba(0, 255, 136, 0.1);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            color: #00ff88;
        }
        .stat-label {
            font-size: 0.8rem;
            color: #888;
        }
        .drama-bar {
            height: 30px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }
        .drama-fill {
            height: 100%;
            background: linear-gradient(90deg, #e74c3c, #f39c12);
            transition: width 0.5s;
            display: flex;
            align-items: center;
            padding-left: 15px;
            font-weight: bold;
        }
        .event-log {
            max-height: 200px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.85rem;
        }
        .event-item {
            padding: 8px;
            margin: 5px 0;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 4px;
            border-left: 3px solid #00ff88;
        }
        .event-item.alert { border-left-color: #e74c3c; }
        .event-item.warning { border-left-color: #f39c12; }
        .emergence-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .emergence-tag {
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .emergence-tag.convergence { background: #9b59b6; }
        .emergence-tag.cooperation { background: #2ecc71; }
        .emergence-tag.innovation { background: #f39c12; }
        .emergence-tag.stagnation { background: #e74c3c; }
    </style>
</head>
<body>
    <div class="header">
        <h1>OUROBOROS Live Dashboard</h1>
        <div class="status">
            <div class="status-item">
                <div class="status-dot" id="connectionStatus"></div>
                <span id="connectionText">Connecting...</span>
            </div>
            <div class="status-item">
                <span>Gen: <strong id="generation">0</strong></span>
            </div>
        </div>
    </div>

    <div class="stats-row" style="padding: 20px 20px 0;">
        <div class="stat">
            <div class="stat-value" id="statAgents">0</div>
            <div class="stat-label">Agents</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="statFitness">0.00</div>
            <div class="stat-label">Best Fitness</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="statEmergence">0</div>
            <div class="stat-label">Emergence Signals</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="statMemory">0</div>
            <div class="stat-label">Memory Entries</div>
        </div>
    </div>

    <!-- Current Problem -->
    <div class="card" style="margin: 20px; background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3);">
        <h2 style="color: #60a5fa;">CURRENT PROBLEM</h2>
        <div id="problemDescription" style="font-family: monospace; color: #93c5fd; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 6px;">
            Loading problem...
        </div>
    </div>

    <!-- Agent Activity Panel -->
    <div class="card" style="margin: 0 20px 20px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3);">
        <h2 style="color: #34d399;">AGENT ACTIVITY</h2>
        <p style="color: #888; font-size: 0.75rem; margin-bottom: 10px;">What each agent is currently doing - their solutions, reasoning, and strategies</p>
        <div id="agentActivity" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
            <div style="color: #888; padding: 20px;">Waiting for agents to start...</div>
        </div>
    </div>

    <!-- Meta-Narrator Panel -->
    <div class="card" style="margin: 0 20px 20px; background: linear-gradient(135deg, rgba(147, 51, 234, 0.2), rgba(79, 70, 229, 0.2)); border: 1px solid rgba(147, 51, 234, 0.4);">
        <h2 style="color: #a78bfa;">META-NARRATOR</h2>
        <p style="color: #888; font-size: 0.75rem; margin-bottom: 10px;">Observing and narrating the evolutionary drama as it unfolds</p>
        <div id="narratorOutput" style="font-style: italic; color: #c4b5fd; min-height: 60px; padding: 15px; background: rgba(0,0,0,0.3); border-radius: 8px; line-height: 1.6;">
            <em>Awaiting the first stirrings of evolution...</em>
        </div>
        <div id="emergentArt" style="margin-top: 15px; padding: 10px; text-align: center; font-family: monospace; color: #818cf8; font-size: 0.9rem;">
        </div>
    </div>

    <div class="container">
        <div class="card">
            <h2>Fitness Progression</h2>
            <p style="color: #888; font-size: 0.75rem; margin-bottom: 10px;">
                Solution quality over generations (0-1 scale). Red = competitive agents, Blue = cooperative agents.
            </p>
            <div class="chart-container">
                <canvas id="fitnessChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Drama Index</h2>
            <p style="color: #888; font-size: 0.75rem; margin-bottom: 10px;">
                Evolutionary tension: competition intensity, breakthroughs, strategy shifts.
            </p>
            <div class="drama-bar">
                <div class="drama-fill" id="dramaFill" style="width: 0%">0%</div>
            </div>
            <p id="dramaHeadline" style="color: #f39c12; font-style: italic; margin-top: 10px;">
                Waiting for events...
            </p>
        </div>

        <div class="card">
            <h2>Emergence Signals</h2>
            <p style="color: #888; font-size: 0.75rem; margin-bottom: 10px;">
                Detected patterns: Convergence (agents agree) | Cooperation (helping) | Innovation (novel) | Stagnation (stuck)
            </p>
            <div class="emergence-list" id="emergenceList">
                <span style="color: #888;">No signals yet</span>
            </div>
        </div>

        <div class="card">
            <h2>Event Log</h2>
            <p style="color: #888; font-size: 0.75rem; margin-bottom: 10px;">
                Real-time feed of agent actions and fitness changes
            </p>
            <div class="event-log" id="eventLog">
                <div class="event-item">Dashboard started</div>
            </div>
        </div>
    </div>

    <script>
        // WebSocket connection
        let ws = null;
        let fitnessChart = null;
        let reconnectAttempts = 0;

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                document.getElementById('connectionStatus').className = 'status-dot connected';
                document.getElementById('connectionText').textContent = 'Connected';
                reconnectAttempts = 0;
                addEvent('Connected to server', 'success');
            };

            ws.onclose = () => {
                document.getElementById('connectionStatus').className = 'status-dot disconnected';
                document.getElementById('connectionText').textContent = 'Disconnected';
                // Reconnect with backoff
                const delay = Math.min(30000, 1000 * Math.pow(2, reconnectAttempts));
                reconnectAttempts++;
                setTimeout(connect, delay);
            };

            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                handleMessage(msg);
            };

            ws.onerror = (error) => {
                addEvent('WebSocket error', 'alert');
            };
        }

        function handleMessage(msg) {
            if (msg.type === 'heartbeat') return;

            if (msg.type === 'state' || msg.type === 'update') {
                updateDashboard(msg.data);
                if (msg.generation) {
                    document.getElementById('generation').textContent = msg.generation;
                }
            }

            if (msg.type === 'event') {
                addEvent(`[Gen ${msg.generation}] ${msg.event_type}: ${JSON.stringify(msg.data).slice(0, 50)}...`);
            }
        }

        function updateDashboard(data) {
            // Update stats
            const fitness = data.fitness_history || {};
            const agentCount = Object.keys(fitness).length;
            document.getElementById('statAgents').textContent = agentCount;

            // Best fitness
            let bestFitness = 0;
            Object.values(fitness).forEach(history => {
                if (history && history.length > 0) {
                    bestFitness = Math.max(bestFitness, Math.max(...history));
                }
            });
            document.getElementById('statFitness').textContent = bestFitness.toFixed(2);

            // Emergence count
            const emergence = data.emergence_signals || [];
            document.getElementById('statEmergence').textContent = emergence.length;

            // Memory entries
            const memoryCount = data.memory_snapshot?.entry_count || 0;
            document.getElementById('statMemory').textContent = memoryCount;

            // Update fitness chart
            updateFitnessChart(fitness, data.agent_modes || {});

            // Update emergence list
            updateEmergenceList(emergence);

            // Update drama index
            updateDrama(data);

            // Update meta-narrator
            updateNarrator(data);

            // Update problem description
            updateProblem(data);

            // Update agent activity
            updateAgentActivity(data);
        }

        function updateNarrator(data) {
            const narratorEl = document.getElementById('narratorOutput');
            const artEl = document.getElementById('emergentArt');

            // Get narrator output from data
            const narrator = data.narrator_output || data.meta_narrative || null;
            if (narrator) {
                narratorEl.innerHTML = narrator;
            } else {
                // Generate narrative from current state
                const gen = data.generation_summaries?.length || 0;
                const fitness = data.fitness_history || {};
                const emergence = data.emergence_signals || [];

                let narrative = '';
                if (gen === 0) {
                    narrative = '<em>Awaiting the first stirrings of evolution...</em>';
                } else {
                    const bestFitness = Math.max(...Object.values(fitness).flatMap(h => h || [0]));
                    const agentCount = Object.keys(fitness).length;

                    if (bestFitness > 0.8) {
                        narrative = `<strong>Generation ${gen}:</strong> Breakthrough achieved. Fitness ${bestFitness.toFixed(2)}. ${agentCount} agents converging on solutions.`;
                    } else if (bestFitness > 0.5) {
                        narrative = `<strong>Generation ${gen}:</strong> Progress unfolding. ${agentCount} agents pushing fitness to ${bestFitness.toFixed(2)}.`;
                    } else if (bestFitness > 0.2) {
                        narrative = `<strong>Generation ${gen}:</strong> Search continues. ${agentCount} agents exploring, best: ${bestFitness.toFixed(2)}.`;
                    } else {
                        narrative = `<strong>Generation ${gen}:</strong> Early exploration. ${agentCount} agents testing hypotheses. Fitness ${bestFitness.toFixed(2)}.`;
                    }

                    if (emergence.length > 0) {
                        const latest = emergence[emergence.length - 1];
                        narrative += ` [${latest.type}: ${latest.description || 'Pattern detected'}]`;
                    }
                }
                narratorEl.innerHTML = narrative;
            }

            // Progress bar
            const fitness = data.fitness_history || {};
            const bestFitness = Math.max(0, ...Object.values(fitness).flatMap(h => h || [0]));
            const bars = Math.round(bestFitness * 10);
            artEl.textContent = '[' + '='.repeat(bars) + '-'.repeat(10-bars) + '] ' + (bestFitness * 100).toFixed(0) + '% fitness';
        }

        function updateProblem(data) {
            const el = document.getElementById('problemDescription');
            const problem = data.problem || data.current_problem;
            if (problem) {
                if (typeof problem === 'string') {
                    el.textContent = problem;
                } else {
                    el.innerHTML = `<strong>Task:</strong> ${problem.description || 'No description'}<br>
                    <strong>Test Cases:</strong> ${JSON.stringify(problem.test_cases || []).slice(0, 200)}...`;
                }
            }
        }

        function updateAgentActivity(data) {
            const container = document.getElementById('agentActivity');
            const agents = data.agent_states || data.agents || {};
            const fitness = data.fitness_history || {};
            const modes = data.agent_modes || {};
            const solutions = data.agent_solutions || data.current_solutions || {};

            if (Object.keys(fitness).length === 0) {
                container.innerHTML = '<div style="color: #888; padding: 20px;">Waiting for agents to start...</div>';
                return;
            }

            let html = '';
            Object.keys(fitness).forEach(agentId => {
                const mode = modes[agentId] || 'competitive';
                const fitHistory = fitness[agentId] || [];
                const currentFit = fitHistory.length > 0 ? fitHistory[fitHistory.length - 1] : 0;
                const solution = solutions[agentId] || {};
                const borderColor = mode === 'cooperative' ? '#3498db' : '#e74c3c';
                const modeLabel = mode === 'cooperative' ? 'COOPERATIVE' : 'COMPETITIVE';

                html += `
                <div style="background: rgba(0,0,0,0.3); border-radius: 8px; padding: 12px; border-left: 3px solid ${borderColor};">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <strong style="color: ${borderColor};">${agentId}</strong>
                        <span style="color: #888; font-size: 0.75rem;">${modeLabel}</span>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <span style="color: #00ff88; font-size: 1.2rem;">${(currentFit * 100).toFixed(0)}%</span>
                        <span style="color: #666; font-size: 0.75rem;"> fitness</span>
                    </div>
                    ${solution.code ? `
                    <details style="margin-top: 8px;">
                        <summary style="cursor: pointer; color: #888; font-size: 0.8rem;">View Solution</summary>
                        <pre style="background: rgba(0,0,0,0.5); padding: 8px; border-radius: 4px; font-size: 0.7rem; overflow-x: auto; margin-top: 5px; color: #9ca3af;">${escapeHtml(solution.code.slice(0, 500))}${solution.code.length > 500 ? '...' : ''}</pre>
                    </details>` : ''}
                    ${solution.reasoning ? `
                    <div style="margin-top: 8px; font-size: 0.75rem; color: #9ca3af;">
                        <em>${solution.reasoning.slice(0, 150)}${solution.reasoning.length > 150 ? '...' : ''}</em>
                    </div>` : ''}
                </div>`;
            });

            container.innerHTML = html || '<div style="color: #888;">No agent data yet</div>';
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function updateFitnessChart(fitness, modes) {
            const labels = [];
            const datasets = [];
            let maxLen = 0;

            Object.entries(fitness).forEach(([agent, history]) => {
                if (history && history.length > 0) {
                    maxLen = Math.max(maxLen, history.length);
                }
            });

            for (let i = 1; i <= maxLen; i++) labels.push(i);

            const colors = {
                competitive: ['#e74c3c', '#c0392b', '#a93226'],
                cooperative: ['#3498db', '#2980b9', '#1f618d']
            };
            const colorIdx = { competitive: 0, cooperative: 0 };

            Object.entries(fitness).forEach(([agent, history]) => {
                const mode = modes[agent] || 'competitive';
                const colorList = colors[mode] || colors.competitive;
                const idx = colorIdx[mode] % colorList.length;
                colorIdx[mode]++;

                datasets.push({
                    label: agent,
                    data: history,
                    borderColor: colorList[idx],
                    fill: false,
                    tension: 0.3
                });
            });

            if (!fitnessChart) {
                const ctx = document.getElementById('fitnessChart').getContext('2d');
                fitnessChart = new Chart(ctx, {
                    type: 'line',
                    data: { labels, datasets },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 1,
                                grid: { color: 'rgba(255,255,255,0.1)' },
                                ticks: { color: '#888' }
                            },
                            x: {
                                grid: { color: 'rgba(255,255,255,0.1)' },
                                ticks: { color: '#888' }
                            }
                        },
                        plugins: {
                            legend: { labels: { color: '#eee' } }
                        }
                    }
                });
            } else {
                fitnessChart.data.labels = labels;
                fitnessChart.data.datasets = datasets;
                fitnessChart.update();
            }
        }

        function updateEmergenceList(emergence) {
            const container = document.getElementById('emergenceList');
            if (emergence.length === 0) {
                container.innerHTML = '<span style="color: #888;">No signals yet</span>';
                return;
            }

            const icons = {
                convergence: '~',
                cooperation: '+',
                innovation: '!',
                stagnation: '-'
            };

            container.innerHTML = emergence.slice(-8).map(sig => {
                const type = sig.type || 'unknown';
                const icon = icons[type] || '?';
                const gen = sig.generation || '?';
                return `<span class="emergence-tag ${type}">${icon} ${type} (g${gen})</span>`;
            }).join('');
        }

        function updateDrama(data) {
            // Calculate drama index from various signals
            let drama = 0;
            const emergence = data.emergence_signals || [];
            const fitness = data.fitness_history || {};

            // More emergence = more drama
            drama += Math.min(0.3, emergence.length * 0.05);

            // Innovation and cooperation boost drama
            emergence.forEach(sig => {
                if (sig.type === 'innovation') drama += 0.2;
                if (sig.type === 'cooperation') drama += 0.1;
            });

            // Fitness variance adds drama
            Object.values(fitness).forEach(history => {
                if (history && history.length > 2) {
                    const recent = history.slice(-5);
                    const variance = recent.reduce((a, b) => a + Math.abs(b - (recent[0] || 0)), 0) / recent.length;
                    drama += variance * 0.5;
                }
            });

            drama = Math.min(1, drama);
            const percent = Math.round(drama * 100);

            const fill = document.getElementById('dramaFill');
            fill.style.width = `${percent}%`;
            fill.textContent = `${percent}%`;

            // Generate headline
            const headlines = [
                'Watching patterns emerge...',
                'Evolution in progress',
                'Agents competing for dominance',
                'Strategy landscape shifting',
                'Coalitions may be forming'
            ];
            if (drama > 0.7) {
                headlines.unshift('SOMETHING INTERESTING IS HAPPENING!');
            }
            document.getElementById('dramaHeadline').textContent =
                headlines[Math.floor(Math.random() * headlines.length)];
        }

        function addEvent(text, type = 'normal') {
            const log = document.getElementById('eventLog');
            const item = document.createElement('div');
            item.className = `event-item ${type}`;
            item.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
            log.insertBefore(item, log.firstChild);

            // Keep only last 50 events
            while (log.children.length > 50) {
                log.removeChild(log.lastChild);
            }
        }

        // Start connection
        connect();

        // Ping to keep alive
        setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send('ping');
            }
        }, 25000);
    </script>
</body>
</html>"""

    def start(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the dashboard server."""
        print(f"Starting OUROBOROS Live Dashboard at http://{host}:{port}")
        uvicorn.run(self.app, host=host, port=port, log_level="warning")

    def start_background(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the server in a background thread."""
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop

            config = uvicorn.Config(
                self.app, host=host, port=port, log_level="warning", loop="asyncio"
            )
            server = uvicorn.Server(config)
            loop.run_until_complete(server.serve())

        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        # Give the server time to start and set the loop
        import time
        time.sleep(0.5)
        print(f"Dashboard running at http://{host}:{port}")
        return thread


def main():
    """Demo the live server."""
    if not HAS_FASTAPI:
        print("Install FastAPI: pip install fastapi uvicorn")
        return

    server = LiveDashboardServer()

    # Demo data
    demo_data = {
        "fitness_history": {
            "competitive_0": [0.1, 0.2, 0.3, 0.4, 0.5],
            "competitive_1": [0.1, 0.15, 0.25, 0.3, 0.35],
            "cooperative_0": [0.1, 0.25, 0.4, 0.55, 0.65],
        },
        "agent_modes": {
            "competitive_0": "competitive",
            "competitive_1": "competitive",
            "cooperative_0": "cooperative",
        },
        "emergence_signals": [
            {"type": "cooperation", "strength": 0.6, "generation": 3},
            {"type": "innovation", "strength": 0.8, "generation": 5},
        ],
        "memory_snapshot": {"entry_count": 35},
        "generation_summaries": [{}, {}, {}, {}, {}],
    }

    server.current_state = demo_data
    server.start(port=8080)


if __name__ == "__main__":
    main()
