"""NPCoT-as-a-Service HTTP server (NV1b).

A stdlib-only HTTP server that loads an `ArrayProgramLibrary` once at
startup and serves `POST /consult` requests against it. Each request
contains a hidden-state vector and an input array; the server returns the
result of consulting the library.

Why stdlib-only: no `fastapi`, no `flask`, no `uvicorn`. The server should
work on *any* Python 3.8+ install without pulling dependencies. The
process footprint is <100 MB (most of which is the underlying Python
runtime and the loaded library).

The handler is also available as a standalone function
`handle_consult_request(request_json, library)` so you can embed it in
your own server framework without running ours.

Usage:

    python3 -m ncpu.self_optimizing.npcot_server \
        --library path/to/library.json --port 8080

    # In another shell:
    curl -X POST http://localhost:8080/consult \\
         -H 'Content-Type: application/json' \\
         -d '{"hidden": [1.0, 0.0, 0.0], "array": [1.0, 2.0, 3.0], "length": 3}'
    # → {"result": 6.0, "hit": true, "elapsed_us": 34}

Endpoints:
    GET  /health    → 200 + JSON status
    GET  /audit     → 200 + library.audit_report()
    GET  /fingerprint → 200 + library.fingerprint()
    POST /consult   → 200 + {result, hit, elapsed_us} or 400/500 on bad input
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import torch

from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    get_native_backend,
)


log = logging.getLogger("npcot_server")


def handle_consult_request(
    request: dict[str, Any],
    library: ArrayProgramLibrary,
) -> tuple[int, dict[str, Any]]:
    """Translate a consult request into a library lookup + execute.

    Returns `(status_code, response_body)`. `status_code` is 200 on hit
    OR on a clean miss (miss is a valid library answer, not an error).
    Malformed requests return 400; unexpected crashes return 500.
    """
    for key in ("hidden", "array", "length"):
        if key not in request:
            return 400, {"error": f"missing required field: {key}"}

    try:
        hidden = [float(v) for v in request["hidden"]]
        array = [float(v) for v in request["array"]]
        length = int(request["length"])
    except (TypeError, ValueError) as exc:
        return 400, {"error": f"malformed input: {exc}"}

    start = time.perf_counter()
    hidden_tensor = torch.tensor(hidden, dtype=torch.float32)
    entry = library.lookup(hidden_tensor)
    if entry is None:
        elapsed_us = (time.perf_counter() - start) * 1e6
        return 200, {
            "result": None,
            "hit": False,
            "elapsed_us": round(elapsed_us, 3),
        }
    # Execute the discrete program.
    arrays = torch.tensor([array], dtype=torch.float32)
    lengths = torch.tensor([float(length)], dtype=torch.float32)
    result = entry.program.execute(arrays, lengths)[0].item()
    elapsed_us = (time.perf_counter() - start) * 1e6
    return 200, {
        "result": float(result),
        "hit": True,
        "elapsed_us": round(elapsed_us, 3),
        "task_name": entry.task_name,
        "program": entry.program.to_dict(),
    }


class NpcotRequestHandler(BaseHTTPRequestHandler):
    library: ArrayProgramLibrary  # set by start_server

    def log_message(self, format, *args):
        # Route BaseHTTPRequestHandler logs through the module logger.
        log.info("%s - %s", self.address_string(), format % args)

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "library_entries": len(self.library),
                    "fingerprint": self.library.fingerprint(),
                },
            )
        elif self.path == "/audit":
            self._send_json(200, self.library.audit_report())
        elif self.path == "/fingerprint":
            self._send_json(
                200, {"fingerprint": self.library.fingerprint()}
            )
        else:
            self._send_json(404, {"error": f"unknown path: {self.path}"})

    def do_POST(self) -> None:
        if self.path != "/consult":
            self._send_json(404, {"error": f"unknown path: {self.path}"})
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length).decode("utf-8")
        try:
            request = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"bad json: {exc}"})
            return
        status, payload = handle_consult_request(request, self.library)
        self._send_json(status, payload)


def start_server(
    library: ArrayProgramLibrary,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> HTTPServer:
    """Start the HTTP server — blocking. Use `run_in_thread` for async."""
    NpcotRequestHandler.library = library
    server = HTTPServer((host, port), NpcotRequestHandler)
    log.info("NPCoT server listening on %s:%d", host, port)
    log.info("  library: %d entries, fingerprint=%s", len(library), library.fingerprint())
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--library", required=True, type=Path, help="library JSON path"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    library = ArrayProgramLibrary.load(args.library)
    native = get_native_backend()
    if native is not None and hasattr(native, "NpcotLibraryIndex"):
        if library.build_native_index():
            log.info("native Rust sharded index active")

    server = start_server(library, host=args.host, port=args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
        server.shutdown()
    return 0


__all__ = [
    "handle_consult_request",
    "start_server",
    "NpcotRequestHandler",
]


if __name__ == "__main__":
    raise SystemExit(main())
