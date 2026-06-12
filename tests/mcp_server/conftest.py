"""Fixtures: spawn the MCP server as a subprocess speaking stdio JSON-RPC.

Every persistent nsynth memory bank (and the method-router state, which
changes which solver families even run) is pointed at a per-session tmp
directory so tests neither read nor pollute the user's real banks at
``~/.nsynth_*``.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = REPO_ROOT / "nsynth" / "target" / "release" / "mog_synth"

# Every env var that points the Rust backend at persistent state.
_BANK_ENV = {
    "NSYNTH_CACHE_PATH": "solved_programs.json",
    "NSYNTH_BIAS_BANK_PATH": "learned_biases.jsonl",
    "NSYNTH_REJECTED_PATH": "rejected_programs.tsv",
    "NSYNTH_METHOD_ROUTER_PATH": "method_router.json",
    "NSYNTH_SEARCH_FAMILY_ROUTER_PATH": "search_family_router.json",
    "NSYNTH_TEACHER_FAILURES_PATH": "teacher_failures.json",
    "NSYNTH_META_WEIGHTS_PATH": "meta_weights.json",
    "NSYNTH_BOOTSTRAP_STATE_PATH": "bootstrap_state.json",
    "NSYNTH_BOOTSTRAP_MARKER_PATH": "bootstrap_marker.json",
    "NSYNTH_AUTOTUNE_CONFIG": "autotune_config.json",
}


def isolated_env(bank_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    for var, fname in _BANK_ENV.items():
        env[var] = str(bank_dir / fname)
    return env


class McpClient:
    """Minimal stdio JSON-RPC 2.0 client for driving the server in tests."""

    def __init__(self, env: dict[str, str]) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "ncpu.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
        )
        self._next_id = 0

    # -- transport ----------------------------------------------------------

    def _send(self, message: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def _read_line(self, timeout_s: float) -> str:
        assert self.proc.stdout is not None
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout_s)
        if not ready:
            raise TimeoutError(
                f"no response from server within {timeout_s}s "
                f"(stderr tail: {self._stderr_tail()})"
            )
        line = self.proc.stdout.readline()
        if not line:
            raise EOFError(
                f"server closed stdout (stderr tail: {self._stderr_tail()})"
            )
        return line

    def _stderr_tail(self) -> str:
        if self.proc.poll() is None:
            return "<server still running>"
        assert self.proc.stderr is not None
        return self.proc.stderr.read()[-500:]

    # -- protocol -----------------------------------------------------------

    def request(
        self, method: str, params: Optional[dict[str, Any]] = None, timeout_s: float = 150.0
    ) -> dict[str, Any]:
        self._next_id += 1
        msg_id = self._next_id
        self._send(
            {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}}
        )
        while True:
            response = json.loads(self._read_line(timeout_s))
            if response.get("id") == msg_id:
                return response

    def notify(self, method: str, params: Optional[dict[str, Any]] = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def initialize(self) -> dict[str, Any]:
        response = self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "0"},
            },
            timeout_s=30.0,
        )
        self.notify("notifications/initialized")
        return response

    def call_tool(
        self, name: str, arguments: dict[str, Any], timeout_s: float = 150.0
    ) -> dict[str, Any]:
        """tools/call + unwrap the text-content JSON payload."""
        response = self.request(
            "tools/call", {"name": name, "arguments": arguments}, timeout_s
        )
        assert "result" in response, f"expected result, got: {response}"
        result = response["result"]
        payload = json.loads(result["content"][0]["text"])
        payload["_isError"] = result.get("isError", False)
        return payload

    def close(self) -> None:
        if self.proc.poll() is None:
            assert self.proc.stdin is not None
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


@pytest.fixture(scope="session")
def bank_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("nsynth_banks")


@pytest.fixture(scope="session")
def client(bank_dir: Path):
    if not BACKEND.is_file():
        pytest.skip(f"mog_synth release binary not built: {BACKEND}")
    c = McpClient(isolated_env(bank_dir))
    c.initialize()
    yield c
    c.close()
