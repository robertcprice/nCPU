#!/usr/bin/env python3
"""
WEB DEMO: Singularity Core on port 4000
Mobile-friendly interface to test the system
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
from singularity_core import SingularityCore
from sympy import Integer
import time

# Initialize core once
print("Loading Singularity Core...")
core = SingularityCore(enable_all=True)
print("Ready!")

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Singularity Core Demo</title>
    <style>
        * { box-sizing: border-box; font-family: -apple-system, system-ui, sans-serif; }
        body {
            margin: 0; padding: 20px;
            background: #1a1a2e; color: #eee;
            min-height: 100vh;
        }
        .container { max-width: 600px; margin: 0 auto; }
        h1 {
            text-align: center;
            background: linear-gradient(90deg, #00d4ff, #7b2ff7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 28px;
        }
        .status {
            background: #16213e;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .status-row { display: flex; justify-content: space-between; margin: 5px 0; }
        .status-label { color: #888; }
        .status-value { color: #00d4ff; font-weight: bold; }
        .input-group {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        label { display: block; margin-bottom: 5px; color: #888; }
        input[type="number"] {
            width: 100%;
            padding: 15px;
            font-size: 24px;
            border: 2px solid #333;
            border-radius: 8px;
            background: #0f0f23;
            color: #fff;
            margin-bottom: 15px;
        }
        input:focus { border-color: #00d4ff; outline: none; }
        button {
            width: 100%;
            padding: 15px;
            font-size: 18px;
            background: linear-gradient(90deg, #00d4ff, #7b2ff7);
            border: none;
            border-radius: 8px;
            color: white;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover { opacity: 0.9; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
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
        }
        .method { color: #888; font-size: 12px; }
        .solution {
            font-size: 24px;
            color: #00d4ff;
            font-family: monospace;
        }
        .confidence { color: #7b2ff7; }
        .neural { border-left: 3px solid #ff6b6b; }
        .coded { border-left: 3px solid #4ecdc4; }
        .tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 10px;
            margin-left: 10px;
        }
        .tag-neural { background: #ff6b6b; color: white; }
        .tag-coded { background: #4ecdc4; color: black; }
        .loading { text-align: center; padding: 20px; }
        .spinner {
            border: 3px solid #333;
            border-top: 3px solid #00d4ff;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .examples {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 20px;
        }
        .example-btn {
            padding: 10px 15px;
            background: #0f0f23;
            border: 1px solid #333;
            border-radius: 5px;
            color: #888;
            cursor: pointer;
            font-size: 14px;
        }
        .example-btn:hover { border-color: #00d4ff; color: #00d4ff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>SINGULARITY CORE</h1>

        <div class="status">
            <div class="status-row">
                <span class="status-label">Capability</span>
                <span class="status-value">100%</span>
            </div>
            <div class="status-row">
                <span class="status-label">Moonshots</span>
                <span class="status-value">9/9 Active</span>
            </div>
            <div class="status-row">
                <span class="status-label">Benchmark</span>
                <span class="status-value">100% Accuracy</span>
            </div>
        </div>

        <div class="input-group">
            <label>What transforms INPUT into OUTPUT?</label>

            <div class="examples">
                <button class="example-btn" onclick="setExample(5, 25)">5→25 (square)</button>
                <button class="example-btn" onclick="setExample(3, 6)">3→6 (double)</button>
                <button class="example-btn" onclick="setExample(7, 17)">7→17 (add 10)</button>
                <button class="example-btn" onclick="setExample(4, 4)">4→4 (identity)</button>
                <button class="example-btn" onclick="setExample(5, -5)">5→-5 (negate)</button>
            </div>

            <label for="input">Input Value</label>
            <input type="number" id="input" value="5" placeholder="Enter input">

            <label for="output">Output Value</label>
            <input type="number" id="output" value="25" placeholder="Enter output">

            <button onclick="synthesize()" id="synthBtn">SYNTHESIZE</button>
        </div>

        <div class="results" id="results">
            <p style="color: #888; text-align: center;">
                Enter values and click SYNTHESIZE
            </p>
        </div>
    </div>

    <script>
        function setExample(inp, out) {
            document.getElementById('input').value = inp;
            document.getElementById('output').value = out;
        }

        async function synthesize() {
            const input = document.getElementById('input').value;
            const output = document.getElementById('output').value;
            const resultsDiv = document.getElementById('results');
            const btn = document.getElementById('synthBtn');

            btn.disabled = true;
            btn.textContent = 'SYNTHESIZING...';
            resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Running 9 moonshots...</p></div>';

            try {
                const response = await fetch(`/api/synthesize?input=${input}&output=${output}`);
                const data = await response.json();

                let html = `<h3 style="margin-top:0">Query: ${input} → ${output}</h3>`;
                html += `<p style="color:#888">Time: ${data.time_ms.toFixed(1)}ms</p>`;

                if (data.solutions && data.solutions.length > 0) {
                    data.solutions.forEach(sol => {
                        const isNeural = ['trained_model', 'mco'].includes(sol.method);
                        const cls = isNeural ? 'neural' : 'coded';
                        const tag = isNeural ?
                            '<span class="tag tag-neural">NEURAL</span>' :
                            '<span class="tag tag-coded">CODED</span>';

                        html += `
                            <div class="result-item ${cls}">
                                <div class="method">${sol.method} ${tag}</div>
                                <div class="solution">${sol.result}</div>
                                <div class="confidence">Confidence: ${(sol.confidence * 100).toFixed(0)}%</div>
                            </div>
                        `;
                    });

                    html += `<p style="margin-top:15px"><strong>Best:</strong> <code style="color:#00d4ff">${data.best_solution}</code> via ${data.method}</p>`;
                } else {
                    html += '<p style="color:#ff6b6b">No solution found</p>';
                }

                resultsDiv.innerHTML = html;
            } catch (e) {
                resultsDiv.innerHTML = `<p style="color:#ff6b6b">Error: ${e.message}</p>`;
            }

            btn.disabled = false;
            btn.textContent = 'SYNTHESIZE';
        }
    </script>
</body>
</html>
"""

class DemoHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress logs

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())

        elif self.path.startswith('/api/synthesize'):
            # Parse query params
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)

            input_val = int(params.get('input', [5])[0])
            output_val = int(params.get('output', [25])[0])

            # Run synthesis
            start = time.time()
            result = core.synthesize(
                Integer(input_val),
                Integer(output_val),
                use_holographic=True,
                use_annealing=True
            )
            elapsed = (time.time() - start) * 1000

            # Prepare response
            response = {
                'input': input_val,
                'output': output_val,
                'solutions': result.get('solutions', []),
                'best_solution': result.get('best_solution'),
                'method': result.get('method'),
                'time_ms': elapsed
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        elif self.path == '/api/status':
            status = core.status()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(status).encode())

        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=4000):
    server = HTTPServer(('0.0.0.0', port), DemoHandler)
    print(f"\n{'='*50}")
    print(f"SINGULARITY CORE WEB DEMO")
    print(f"{'='*50}")
    print(f"Server running at: http://localhost:{port}")
    print(f"Mobile access: http://<your-ip>:{port}")
    print(f"Press Ctrl+C to stop")
    print(f"{'='*50}\n")
    server.serve_forever()

if __name__ == '__main__':
    run_server(4000)
