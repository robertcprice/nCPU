#!/usr/bin/env python3
"""
ENHANCED WEB DEMO: Singularity Core with Visualizations
Shows actual holographic superposition and thermodynamic annealing
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
from singularity_core import SingularityCore
from holographic_programs import HolographicSearch
from thermodynamic_annealing import ThermodynamicAnnealer
from sympy import Integer, Symbol
import time
import numpy as np
import math

# Initialize core
print("Loading Singularity Core...")
core = SingularityCore(enable_all=True)
holo_search = HolographicSearch(dimension=128)
print("Ready!")

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Singularity Core - Full Visualization</title>
    <style>
        * { box-sizing: border-box; font-family: -apple-system, system-ui, sans-serif; }
        body {
            margin: 0; padding: 20px;
            background: #0a0a1a; color: #eee;
            min-height: 100vh;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 {
            text-align: center;
            background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 28px;
            margin-bottom: 10px;
        }
        h2 {
            font-size: 16px;
            color: #888;
            text-align: center;
            margin-top: 0;
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .tab {
            flex: 1;
            min-width: 140px;
            padding: 12px;
            background: #16213e;
            border: 2px solid #333;
            border-radius: 8px;
            color: #888;
            cursor: pointer;
            text-align: center;
            font-weight: bold;
            transition: all 0.3s;
        }
        .tab.active {
            border-color: #00d4ff;
            color: #00d4ff;
            background: #1a2a4a;
        }
        .tab:hover { border-color: #555; }
        .panel {
            display: none;
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .panel.active { display: block; }
        .input-group {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .input-box {
            flex: 1;
            min-width: 100px;
        }
        label { display: block; margin-bottom: 5px; color: #888; font-size: 12px; }
        input[type="number"] {
            width: 100%;
            padding: 12px;
            font-size: 20px;
            border: 2px solid #333;
            border-radius: 8px;
            background: #0f0f23;
            color: #fff;
        }
        input:focus { border-color: #00d4ff; outline: none; }
        button.main-btn {
            width: 100%;
            padding: 15px;
            font-size: 18px;
            background: linear-gradient(90deg, #00d4ff, #7b2ff7);
            border: none;
            border-radius: 8px;
            color: white;
            cursor: pointer;
            font-weight: bold;
            margin-top: 10px;
        }
        button.main-btn:disabled { opacity: 0.5; cursor: not-allowed; }

        /* Holographic Visualization */
        .holo-viz {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        .wave-canvas {
            flex: 1;
            min-width: 280px;
            height: 200px;
            background: #0a0a1a;
            border-radius: 8px;
            border: 1px solid #333;
        }
        .superposition-list {
            flex: 1;
            min-width: 200px;
        }
        .super-item {
            padding: 10px;
            margin-bottom: 5px;
            background: #0f0f23;
            border-radius: 5px;
            border-left: 3px solid;
        }
        .super-item .name { font-weight: bold; }
        .super-item .amplitude { color: #00d4ff; font-family: monospace; }
        .super-item .phase { color: #7b2ff7; font-size: 12px; }

        /* Thermodynamic Visualization */
        .thermo-viz {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        .thermo-canvas {
            flex: 2;
            min-width: 300px;
            height: 250px;
            background: #0a0a1a;
            border-radius: 8px;
            border: 1px solid #333;
        }
        .thermo-stats {
            flex: 1;
            min-width: 180px;
        }
        .stat-box {
            padding: 15px;
            margin-bottom: 10px;
            background: #0f0f23;
            border-radius: 8px;
            text-align: center;
        }
        .stat-label { color: #888; font-size: 12px; }
        .stat-value { font-size: 24px; font-weight: bold; margin-top: 5px; }
        .temp { color: #ff6b6b; }
        .energy { color: #4ecdc4; }
        .phase { color: #ffd93d; }

        /* Particle field */
        .particle-canvas {
            width: 100%;
            height: 200px;
            background: #0a0a1a;
            border-radius: 8px;
            border: 1px solid #333;
            margin-top: 15px;
        }

        /* Results */
        .results {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
        }
        .result-item {
            background: #0f0f23;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .method-badge {
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: bold;
            white-space: nowrap;
        }
        .neural { background: #ff6b6b; }
        .coded { background: #4ecdc4; color: #000; }
        .hybrid { background: #7b2ff7; }
        .solution-text {
            flex: 1;
            font-size: 20px;
            font-family: monospace;
            color: #00d4ff;
        }
        .confidence { color: #888; font-size: 12px; }

        .phase-indicator {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .phase-gas { background: #ff6b6b33; color: #ff6b6b; }
        .phase-liquid { background: #4ecdc433; color: #4ecdc4; }
        .phase-solid { background: #7b2ff733; color: #7b2ff7; }
        .phase-crystal { background: #ffd93d33; color: #ffd93d; }

        .loading { text-align: center; padding: 40px; }
        .spinner {
            border: 4px solid #333;
            border-top: 4px solid #00d4ff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .legend {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 10px;
            flex-wrap: wrap;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 12px;
            color: #888;
        }
        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>SINGULARITY CORE</h1>
        <h2>Neural + Coded + Holographic + Thermodynamic</h2>

        <div class="tabs">
            <div class="tab active" onclick="switchTab('synthesize')">Synthesize</div>
            <div class="tab" onclick="switchTab('holographic')">Holographic</div>
            <div class="tab" onclick="switchTab('thermodynamic')">Thermodynamic</div>
            <div class="tab" onclick="switchTab('all')">All Methods</div>
        </div>

        <!-- SYNTHESIZE PANEL -->
        <div id="panel-synthesize" class="panel active">
            <div class="input-group">
                <div class="input-box">
                    <label>INPUT</label>
                    <input type="number" id="input" value="5">
                </div>
                <div class="input-box">
                    <label>OUTPUT</label>
                    <input type="number" id="output" value="25">
                </div>
            </div>
            <button class="main-btn" onclick="synthesize()">SYNTHESIZE</button>
        </div>

        <!-- HOLOGRAPHIC PANEL -->
        <div id="panel-holographic" class="panel">
            <h3 style="margin-top:0; color:#00d4ff;">Holographic Superposition</h3>
            <p style="color:#888; font-size:14px;">
                Programs encoded as wave functions. Interference finds solutions.
            </p>
            <div class="holo-viz">
                <canvas id="waveCanvas" class="wave-canvas"></canvas>
                <div class="superposition-list" id="superList"></div>
            </div>
            <div class="legend">
                <div class="legend-item"><div class="legend-color" style="background:#00d4ff"></div> Real</div>
                <div class="legend-item"><div class="legend-color" style="background:#ff6b6b"></div> Imaginary</div>
                <div class="legend-item"><div class="legend-color" style="background:#7b2ff7"></div> Phase</div>
            </div>
            <button class="main-btn" onclick="runHolographic()" style="margin-top:15px;">
                RUN HOLOGRAPHIC SEARCH
            </button>
        </div>

        <!-- THERMODYNAMIC PANEL -->
        <div id="panel-thermodynamic" class="panel">
            <h3 style="margin-top:0; color:#ff6b6b;">Thermodynamic Annealing</h3>
            <p style="color:#888; font-size:14px;">
                Programs as particles. Energy = MDL. Phase transitions reveal structure.
            </p>
            <div class="thermo-viz">
                <canvas id="thermoCanvas" class="thermo-canvas"></canvas>
                <div class="thermo-stats">
                    <div class="stat-box">
                        <div class="stat-label">TEMPERATURE</div>
                        <div class="stat-value temp" id="tempValue">1000</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">ENERGY</div>
                        <div class="stat-value energy" id="energyValue">0.00</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">PHASE</div>
                        <div id="phaseValue">
                            <span class="phase-indicator phase-gas">GAS</span>
                        </div>
                    </div>
                </div>
            </div>
            <canvas id="particleCanvas" class="particle-canvas"></canvas>
            <button class="main-btn" onclick="runAnnealing()" style="margin-top:15px;">
                RUN THERMODYNAMIC ANNEALING
            </button>
        </div>

        <!-- ALL METHODS PANEL -->
        <div id="panel-all" class="panel">
            <h3 style="margin-top:0; color:#7b2ff7;">All 9 Moonshots</h3>
            <p style="color:#888; font-size:14px;">
                Run all methods: Neural, Holographic, Thermodynamic, Grammar Discovery, and more.
            </p>
            <div class="input-group">
                <div class="input-box">
                    <label>INPUT</label>
                    <input type="number" id="input-all" value="3">
                </div>
                <div class="input-box">
                    <label>OUTPUT</label>
                    <input type="number" id="output-all" value="27">
                </div>
            </div>
            <button class="main-btn" onclick="runAll()">RUN ALL MOONSHOTS</button>
        </div>

        <!-- RESULTS -->
        <div class="results" id="results">
            <p style="color: #888; text-align: center;">
                Select a method and run to see visualizations
            </p>
        </div>
    </div>

    <script>
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            document.querySelector(`[onclick="switchTab('${tab}')"]`).classList.add('active');
            document.getElementById(`panel-${tab}`).classList.add('active');
        }

        // =========================================
        // HOLOGRAPHIC VISUALIZATION
        // =========================================
        function drawWaveFunction(data) {
            const canvas = document.getElementById('waveCanvas');
            const ctx = canvas.getContext('2d');
            canvas.width = canvas.offsetWidth * 2;
            canvas.height = canvas.offsetHeight * 2;
            ctx.scale(2, 2);

            const w = canvas.offsetWidth;
            const h = canvas.offsetHeight;
            const midY = h / 2;

            ctx.fillStyle = '#0a0a1a';
            ctx.fillRect(0, 0, w, h);

            // Draw grid
            ctx.strokeStyle = '#222';
            ctx.lineWidth = 0.5;
            for (let x = 0; x < w; x += 20) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, h);
                ctx.stroke();
            }
            for (let y = 0; y < h; y += 20) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(w, y);
                ctx.stroke();
            }

            // Draw center line
            ctx.strokeStyle = '#444';
            ctx.beginPath();
            ctx.moveTo(0, midY);
            ctx.lineTo(w, midY);
            ctx.stroke();

            if (!data || !data.real) return;

            const step = w / data.real.length;

            // Draw real part (cyan)
            ctx.beginPath();
            ctx.strokeStyle = '#00d4ff';
            ctx.lineWidth = 2;
            for (let i = 0; i < data.real.length; i++) {
                const x = i * step;
                const y = midY - data.real[i] * (h / 3);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Draw imaginary part (red)
            ctx.beginPath();
            ctx.strokeStyle = '#ff6b6b';
            ctx.lineWidth = 2;
            for (let i = 0; i < data.imag.length; i++) {
                const x = i * step;
                const y = midY - data.imag[i] * (h / 3);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Draw phase as filled area (purple)
            ctx.beginPath();
            ctx.fillStyle = 'rgba(123, 47, 247, 0.3)';
            ctx.moveTo(0, midY);
            for (let i = 0; i < data.phase.length; i++) {
                const x = i * step;
                const y = midY - data.phase[i] * (h / 4);
                ctx.lineTo(x, y);
            }
            ctx.lineTo(w, midY);
            ctx.closePath();
            ctx.fill();
        }

        function updateSuperpositionList(data) {
            const list = document.getElementById('superList');
            if (!data || !data.programs) {
                list.innerHTML = '<p style="color:#888">No data</p>';
                return;
            }

            const colors = ['#00d4ff', '#ff6b6b', '#4ecdc4', '#ffd93d', '#7b2ff7'];
            list.innerHTML = data.programs.map((p, i) => `
                <div class="super-item" style="border-color: ${colors[i % colors.length]}">
                    <div class="name">${p.name}</div>
                    <div class="amplitude">|&psi;| = ${p.amplitude.toFixed(3)}</div>
                    <div class="phase">&phi; = ${p.phase.toFixed(2)} rad</div>
                </div>
            `).join('');
        }

        async function runHolographic() {
            const input = document.getElementById('input').value;
            const output = document.getElementById('output').value;
            const results = document.getElementById('results');

            results.innerHTML = '<div class="loading"><div class="spinner"></div><p>Running holographic search...</p></div>';

            try {
                const response = await fetch(`/api/holographic?input=${input}&output=${output}`);
                const data = await response.json();

                // Draw wave function
                drawWaveFunction(data.wave_function);
                updateSuperpositionList(data.superposition);

                // Show results
                let html = '<h3 style="margin-top:0">Holographic Search Results</h3>';
                html += `<p style="color:#888">Time: ${data.time_ms.toFixed(1)}ms | Dimension: ${data.dimension}</p>`;

                if (data.matches && data.matches.length > 0) {
                    data.matches.forEach(m => {
                        html += `
                            <div class="result-item">
                                <span class="method-badge hybrid">HOLOGRAPHIC</span>
                                <span class="solution-text">${m.program}</span>
                                <span class="confidence">similarity: ${(m.similarity * 100).toFixed(0)}%</span>
                            </div>
                        `;
                    });
                } else {
                    html += '<p style="color:#ff6b6b">No matching programs found via interference</p>';
                }

                results.innerHTML = html;
            } catch (e) {
                results.innerHTML = `<p style="color:#ff6b6b">Error: ${e.message}</p>`;
            }
        }

        // =========================================
        // THERMODYNAMIC VISUALIZATION
        // =========================================
        let thermoHistory = { temps: [], energies: [], phases: [] };
        let particles = [];

        function drawThermoGraph() {
            const canvas = document.getElementById('thermoCanvas');
            const ctx = canvas.getContext('2d');
            canvas.width = canvas.offsetWidth * 2;
            canvas.height = canvas.offsetHeight * 2;
            ctx.scale(2, 2);

            const w = canvas.offsetWidth;
            const h = canvas.offsetHeight;

            ctx.fillStyle = '#0a0a1a';
            ctx.fillRect(0, 0, w, h);

            // Draw grid
            ctx.strokeStyle = '#222';
            ctx.lineWidth = 0.5;
            for (let x = 0; x < w; x += 30) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, h);
                ctx.stroke();
            }
            for (let y = 0; y < h; y += 30) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(w, y);
                ctx.stroke();
            }

            if (thermoHistory.temps.length < 2) return;

            const step = w / thermoHistory.temps.length;
            const maxTemp = Math.max(...thermoHistory.temps);
            const maxEnergy = Math.max(...thermoHistory.energies);

            // Draw temperature (red)
            ctx.beginPath();
            ctx.strokeStyle = '#ff6b6b';
            ctx.lineWidth = 2;
            for (let i = 0; i < thermoHistory.temps.length; i++) {
                const x = i * step;
                const y = h - (thermoHistory.temps[i] / maxTemp) * (h - 20);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Draw energy (cyan)
            ctx.beginPath();
            ctx.strokeStyle = '#4ecdc4';
            ctx.lineWidth = 2;
            for (let i = 0; i < thermoHistory.energies.length; i++) {
                const x = i * step;
                const y = h - (thermoHistory.energies[i] / maxEnergy) * (h - 20);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Mark phase transitions
            ctx.fillStyle = '#ffd93d';
            for (let i = 1; i < thermoHistory.phases.length; i++) {
                if (thermoHistory.phases[i] !== thermoHistory.phases[i-1]) {
                    const x = i * step;
                    ctx.beginPath();
                    ctx.arc(x, 20, 5, 0, Math.PI * 2);
                    ctx.fill();
                }
            }

            // Labels
            ctx.font = '11px sans-serif';
            ctx.fillStyle = '#ff6b6b';
            ctx.fillText('Temperature', 10, 15);
            ctx.fillStyle = '#4ecdc4';
            ctx.fillText('Energy', 90, 15);
            ctx.fillStyle = '#ffd93d';
            ctx.fillText('Phase Transition', 150, 15);
        }

        function drawParticles() {
            const canvas = document.getElementById('particleCanvas');
            const ctx = canvas.getContext('2d');
            canvas.width = canvas.offsetWidth * 2;
            canvas.height = canvas.offsetHeight * 2;
            ctx.scale(2, 2);

            const w = canvas.offsetWidth;
            const h = canvas.offsetHeight;

            ctx.fillStyle = '#0a0a1a';
            ctx.fillRect(0, 0, w, h);

            // Draw particles
            const phaseColors = {
                'gas': '#ff6b6b',
                'liquid': '#4ecdc4',
                'solid': '#7b2ff7',
                'crystal': '#ffd93d'
            };

            particles.forEach(p => {
                const x = (p.x + 1) / 2 * w;
                const y = (p.y + 1) / 2 * h;
                const radius = 3 + (1 - p.energy / 10) * 5;

                ctx.beginPath();
                ctx.fillStyle = phaseColors[p.phase] || '#888';
                ctx.globalAlpha = 0.8;
                ctx.arc(x, y, radius, 0, Math.PI * 2);
                ctx.fill();

                // Velocity arrow
                ctx.strokeStyle = '#fff';
                ctx.globalAlpha = 0.3;
                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.lineTo(x + p.vx * 20, y + p.vy * 20);
                ctx.stroke();
            });

            ctx.globalAlpha = 1;
        }

        function updateThermoStats(data) {
            document.getElementById('tempValue').textContent = data.temperature.toFixed(0);
            document.getElementById('energyValue').textContent = data.energy.toFixed(2);

            const phaseEl = document.getElementById('phaseValue');
            const phaseClass = `phase-${data.phase}`;
            phaseEl.innerHTML = `<span class="phase-indicator ${phaseClass}">${data.phase.toUpperCase()}</span>`;
        }

        async function runAnnealing() {
            const input = document.getElementById('input').value;
            const output = document.getElementById('output').value;
            const results = document.getElementById('results');

            // Reset history
            thermoHistory = { temps: [], energies: [], phases: [] };
            particles = [];

            results.innerHTML = '<div class="loading"><div class="spinner"></div><p>Running thermodynamic annealing...</p></div>';

            try {
                const response = await fetch(`/api/annealing?input=${input}&output=${output}`);
                const data = await response.json();

                // Animate the process
                const steps = data.steps;
                let frame = 0;

                function animate() {
                    if (frame < steps.length) {
                        const step = steps[frame];
                        thermoHistory.temps.push(step.temperature);
                        thermoHistory.energies.push(step.energy);
                        thermoHistory.phases.push(step.phase);
                        particles = step.particles;

                        updateThermoStats(step);
                        drawThermoGraph();
                        drawParticles();

                        frame++;
                        requestAnimationFrame(animate);
                    } else {
                        // Show final results
                        showAnnealingResults(data);
                    }
                }

                animate();
            } catch (e) {
                results.innerHTML = `<p style="color:#ff6b6b">Error: ${e.message}</p>`;
            }
        }

        function showAnnealingResults(data) {
            const results = document.getElementById('results');
            let html = '<h3 style="margin-top:0">Thermodynamic Annealing Results</h3>';
            html += `<p style="color:#888">Time: ${data.time_ms.toFixed(1)}ms | Steps: ${data.steps.length}</p>`;

            if (data.discoveries && data.discoveries.length > 0) {
                html += '<p style="color:#ffd93d">Phase transitions detected:</p>';
                data.discoveries.forEach(d => {
                    html += `<div class="result-item">
                        <span class="method-badge" style="background:#ffd93d;color:#000">TRANSITION</span>
                        <span style="color:#fff">${d.type} at T=${d.temperature.toFixed(0)}</span>
                    </div>`;
                });
            }

            if (data.best_programs && data.best_programs.length > 0) {
                html += '<p style="color:#4ecdc4; margin-top:15px">Best programs found:</p>';
                data.best_programs.forEach(p => {
                    html += `
                        <div class="result-item">
                            <span class="method-badge coded">ANNEALING</span>
                            <span class="solution-text">${p.program}</span>
                            <span class="confidence">energy: ${p.energy.toFixed(2)}</span>
                        </div>
                    `;
                });
            }

            results.innerHTML = html;
        }

        // =========================================
        // MAIN SYNTHESIZE
        // =========================================
        async function synthesize() {
            const input = document.getElementById('input').value;
            const output = document.getElementById('output').value;
            const results = document.getElementById('results');

            results.innerHTML = '<div class="loading"><div class="spinner"></div><p>Running synthesis...</p></div>';

            try {
                const response = await fetch(`/api/synthesize?input=${input}&output=${output}`);
                const data = await response.json();

                let html = `<h3 style="margin-top:0">Synthesis: ${input} &rarr; ${output}</h3>`;
                html += `<p style="color:#888">Time: ${data.time_ms.toFixed(1)}ms</p>`;

                if (data.solutions && data.solutions.length > 0) {
                    data.solutions.forEach(sol => {
                        const isNeural = ['trained_model', 'mco'].includes(sol.method);
                        const isHybrid = ['holographic', 'annealing'].includes(sol.method);
                        const cls = isNeural ? 'neural' : (isHybrid ? 'hybrid' : 'coded');
                        const label = isNeural ? 'NEURAL' : (isHybrid ? 'HYBRID' : 'CODED');

                        html += `
                            <div class="result-item">
                                <span class="method-badge ${cls}">${label}</span>
                                <span class="solution-text">${sol.result}</span>
                                <span class="confidence">${sol.method} (${(sol.confidence * 100).toFixed(0)}%)</span>
                            </div>
                        `;
                    });

                    html += `<p style="margin-top:15px"><strong>Best:</strong>
                        <code style="color:#00d4ff">${data.best_solution}</code> via ${data.method}</p>`;
                } else {
                    html += '<p style="color:#ff6b6b">No solution found</p>';
                }

                results.innerHTML = html;
            } catch (e) {
                results.innerHTML = `<p style="color:#ff6b6b">Error: ${e.message}</p>`;
            }
        }

        // =========================================
        // RUN ALL MOONSHOTS
        // =========================================
        async function runAll() {
            const input = document.getElementById('input-all').value;
            const output = document.getElementById('output-all').value;
            const results = document.getElementById('results');

            results.innerHTML = '<div class="loading"><div class="spinner"></div><p>Running all 9 moonshots...</p></div>';

            try {
                const response = await fetch(`/api/all?input=${input}&output=${output}`);
                const data = await response.json();

                let html = `<h3 style="margin-top:0">All Moonshots: ${input} &rarr; ${output}</h3>`;
                html += `<p style="color:#888">Time: ${data.time_ms.toFixed(1)}ms</p>`;

                // Group by type
                const groups = {neural: [], coded: [], hybrid: []};
                data.solutions.forEach(sol => {
                    const isNeural = ['trained_model', 'mco'].includes(sol.method);
                    const isHybrid = ['holographic', 'annealing', 'novel_discovery'].includes(sol.method);
                    if (isNeural) groups.neural.push(sol);
                    else if (isHybrid) groups.hybrid.push(sol);
                    else groups.coded.push(sol);
                });

                for (const [type, sols] of Object.entries(groups)) {
                    if (sols.length > 0) {
                        const cls = type;
                        const label = type.toUpperCase();
                        html += `<p style="color:#888; margin-top:15px">${label} Methods:</p>`;
                        sols.forEach(sol => {
                            html += `
                                <div class="result-item">
                                    <span class="method-badge ${cls}">${sol.method}</span>
                                    <span class="solution-text">${sol.result}</span>
                                    <span class="confidence">${(sol.confidence * 100).toFixed(0)}%</span>
                                </div>
                            `;
                        });
                    }
                }

                html += `<p style="margin-top:15px; font-size:18px"><strong>BEST:</strong>
                    <code style="color:#00d4ff; font-size:20px">${data.best_solution}</code></p>`;

                results.innerHTML = html;
            } catch (e) {
                results.innerHTML = `<p style="color:#ff6b6b">Error: ${e.message}</p>`;
            }
        }

        // Initialize canvases
        window.onload = function() {
            drawWaveFunction(null);
            drawThermoGraph();
            drawParticles();
        };
        window.onresize = function() {
            drawWaveFunction(null);
            drawThermoGraph();
            drawParticles();
        };
    </script>
</body>
</html>
"""

class EnhancedDemoHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())

        elif self.path.startswith('/api/synthesize'):
            self._handle_synthesize()

        elif self.path.startswith('/api/holographic'):
            self._handle_holographic()

        elif self.path.startswith('/api/annealing'):
            self._handle_annealing()

        elif self.path.startswith('/api/all'):
            self._handle_all()

        else:
            self.send_response(404)
            self.end_headers()

    def _parse_params(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        input_val = int(params.get('input', [5])[0])
        output_val = int(params.get('output', [25])[0])
        return input_val, output_val

    def _send_json(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def _handle_synthesize(self):
        input_val, output_val = self._parse_params()
        start = time.time()
        result = core.synthesize(
            Integer(input_val),
            Integer(output_val),
            use_holographic=True,
            use_annealing=True
        )
        elapsed = (time.time() - start) * 1000

        response = {
            'input': input_val,
            'output': output_val,
            'solutions': result.get('solutions', []),
            'best_solution': result.get('best_solution'),
            'method': result.get('method'),
            'time_ms': elapsed
        }
        self._send_json(response)

    def _handle_holographic(self):
        input_val, output_val = self._parse_params()
        start = time.time()

        x = Symbol('x')

        # Run holographic search
        matches = holo_search.search_by_example(Integer(input_val), Integer(output_val))

        # Get wave function data for visualization
        query = f"{input_val} -> {output_val}"
        wave = holo_search.encoder.encode_program(query)

        # Sample wave function for visualization (first 64 points)
        n_points = 64
        real_part = [float(np.real(wave.coefficients[i])) for i in range(n_points)]
        imag_part = [float(np.imag(wave.coefficients[i])) for i in range(n_points)]
        phase_part = [float(wave.phase[i]) for i in range(n_points)]

        # Get superposition of candidate programs
        programs_data = []
        for prog_name in list(holo_search.program_library.keys())[:5]:
            if prog_name in holo_search.encoder.program_registry:
                prog_wave = holo_search.encoder.program_registry[prog_name]
                amplitude = float(np.abs(wave.inner_product(prog_wave)))
                phase = float(np.angle(wave.inner_product(prog_wave)))
                programs_data.append({
                    'name': prog_name,
                    'amplitude': amplitude,
                    'phase': phase
                })

        elapsed = (time.time() - start) * 1000

        response = {
            'dimension': holo_search.dimension,
            'wave_function': {
                'real': real_part,
                'imag': imag_part,
                'phase': phase_part
            },
            'superposition': {
                'programs': sorted(programs_data, key=lambda x: -x['amplitude'])
            },
            'matches': [{'program': m[0], 'similarity': m[1]} for m in matches],
            'time_ms': elapsed
        }
        self._send_json(response)

    def _handle_annealing(self):
        input_val, output_val = self._parse_params()
        start = time.time()

        # Create annealer
        annealer = ThermodynamicAnnealer(
            initial_temperature=500.0,
            cooling_rate=0.9,
            min_temperature=1.0,
            num_particles=20
        )

        # Seed programs that might solve this
        x = Symbol('x')
        seed_programs = ['x', 'x*x', '2*x', 'x+x', 'x*x*x', '3*x', '-x', 'x+10']
        annealer.initialize_state(seed_programs)

        # Run annealing and collect steps for animation
        steps_data = []
        n_steps = 50

        for step in range(n_steps):
            step_result = annealer.anneal_step()

            # Collect particle data
            particles_data = []
            for i, p in enumerate(annealer.state.particles):
                particles_data.append({
                    'x': float(p.position[0]),
                    'y': float(p.position[1]),
                    'vx': float(p.velocity[0] * 0.1),
                    'vy': float(p.velocity[1] * 0.1),
                    'energy': p.energy,
                    'phase': annealer.state.phase,
                    'program': p.program[:10]
                })

            steps_data.append({
                'temperature': step_result['temperature'],
                'energy': step_result['total_energy'],
                'phase': step_result['phase'],
                'particles': particles_data
            })

        # Get best programs
        best_programs = annealer.get_best_programs(top_k=5)

        # Check which ones actually solve the problem
        verified = []
        for prog, energy in best_programs:
            try:
                fn_result = eval(prog.replace('x', str(input_val)))
                if fn_result == output_val:
                    verified.append({'program': prog, 'energy': energy, 'verified': True})
                else:
                    verified.append({'program': prog, 'energy': energy, 'verified': False})
            except:
                verified.append({'program': prog, 'energy': energy, 'verified': False})

        elapsed = (time.time() - start) * 1000

        response = {
            'steps': steps_data,
            'discoveries': annealer.structure_discoveries,
            'best_programs': verified,
            'time_ms': elapsed
        }
        self._send_json(response)

    def _handle_all(self):
        input_val, output_val = self._parse_params()
        start = time.time()

        result = core.synthesize(
            Integer(input_val),
            Integer(output_val),
            use_holographic=True,
            use_annealing=True
        )

        elapsed = (time.time() - start) * 1000

        response = {
            'input': input_val,
            'output': output_val,
            'solutions': result.get('solutions', []),
            'best_solution': result.get('best_solution'),
            'method': result.get('method'),
            'time_ms': elapsed
        }
        self._send_json(response)


def run_server(port=4000):
    server = HTTPServer(('0.0.0.0', port), EnhancedDemoHandler)
    print(f"\n{'='*50}")
    print(f"SINGULARITY CORE - ENHANCED DEMO")
    print(f"{'='*50}")
    print(f"Server: http://localhost:{port}")
    print(f"Mobile: http://<your-ip>:{port}")
    print(f"Press Ctrl+C to stop")
    print(f"{'='*50}\n")
    server.serve_forever()


if __name__ == '__main__':
    run_server(4000)
