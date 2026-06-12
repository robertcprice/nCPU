"""Live-server tests for the synthesis API (ncpu/synthesis_api/server.py).

Spawns the stdlib HTTP server in a subprocess on a free port with all
three nsynth memory banks pointed at private tmp paths, then exercises
the cascade contract end to end:

1. /health reports the backend binary.
2. An easy problem (add_two) synthesizes verified code + transpiles.
3. The same request again is near-instant (solved-cache hit).
4. A junk mapping with no pattern is refused honestly (success: false).
5. Malformed bodies get a 400, never a 500.

Requires the release binary at nsynth/target/release/mog_synth
(`cargo build --release` in nsynth/). Tests skip if it is missing.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = REPO_ROOT / "nsynth" / "target" / "release" / "mog_synth"

pytestmark = pytest.mark.skipif(
    not BACKEND.is_file(),
    reason="mog_synth release binary not built (run: cargo build --release in nsynth/)",
)

ADD_TWO_REQUEST = {
    "name": "add_two",
    "examples": [
        {"inputs": [1, 2], "expected": 3},
        {"inputs": [5, 7], "expected": 12},
        {"inputs": [0, 0], "expected": 0},
    ],
}

# Verified manually against the release binary: the solver refuses this
# patternless string→int mapping instantly (search teachers all miss and
# the differentiable solver does not support string inputs) — an honest
# end-to-end refusal in well under a second.
JUNK_REQUEST = {
    "name": "junk_no_pattern",
    "examples": [
        {"inputs": ["ab"], "expected": 941},
        {"inputs": ["xyz"], "expected": -17},
        {"inputs": ["q"], "expected": 30303},
        {"inputs": ["hello"], "expected": -2},
        {"inputs": ["zz"], "expected": 777777},
    ],
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _request(
    base_url: str, path: str, body: dict | str | None = None, timeout: float = 200.0
) -> tuple[int, dict]:
    """GET (body None) or POST (body given) returning (status, json)."""
    data = None
    if body is not None:
        raw = body if isinstance(body, str) else json.dumps(body)
        data = raw.encode("utf-8")
    req = urllib.request.Request(
        base_url + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


@pytest.fixture(scope="module")
def server(tmp_path_factory, nsynth_isolated_env):
    """A live server on a free port with isolated memory banks.

    The explicit ``--*-cache`` flags cover the three banks the server
    manages directly; ``nsynth_isolated_env`` (see conftest) isolates
    every remaining ``NSYNTH_*`` bank — most importantly the method
    router, which would otherwise be read from the user's real
    ``~/.nsynth_method_router.json`` and reroute the easy problems in
    this suite onto slow solver paths.
    """
    banks = tmp_path_factory.mktemp("nsynth_banks")
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ncpu.synthesis_api.server",
            "--port",
            str(port),
            "--backend",
            str(BACKEND),
            "--solved-cache",
            str(banks / "solved.json"),
            "--bias-bank",
            str(banks / "biases.jsonl"),
            "--rejected-cache",
            str(banks / "rejected.tsv"),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=nsynth_isolated_env,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 15.0
        last_err = None
        while time.time() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"server died at startup:\n{out}")
            try:
                status, _ = _request(base_url, "/health", timeout=2.0)
                if status == 200:
                    break
            except OSError as exc:
                last_err = exc
                time.sleep(0.1)
        else:
            raise RuntimeError(f"server never came up: {last_err}")
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_health_ok(server):
    status, body = _request(server, "/health")
    assert status == 200
    assert body["status"] == "ok"
    assert body["backend_present"] is True
    assert body["backend"].endswith("mog_synth")


def test_solve_easy_problem(server):
    status, body = _request(server, "/synthesize", ADD_TWO_REQUEST)
    assert status == 200
    assert body["success"] is True, body
    assert body["error"] is None
    assert isinstance(body["method"], str) and body["method"]
    assert isinstance(body["code"], str) and body["code"].strip()
    assert "add_two" in body["code"]
    transpiled = body["transpiled"]
    assert isinstance(transpiled, dict)
    assert "def" in transpiled["python"]
    assert "fn" in transpiled["rust"]
    assert "function" in transpiled["typescript"]
    assert body["elapsed_ms"] >= 0


def test_repeat_request_hits_solved_cache(server):
    # First call (test above) populated the solved cache; this repeat must
    # come back near-instantly. The cache makes this ~30 ms; 1000 ms gives
    # generous CI headroom while still ruling out a re-synthesis.
    status, body = _request(server, "/synthesize", ADD_TWO_REQUEST)
    assert status == 200
    assert body["success"] is True
    assert body["elapsed_ms"] < 1000, f"expected cache hit, took {body['elapsed_ms']} ms"


def test_stats_counts_memory_banks(server):
    # After at least one successful solve, the solved bank must be non-empty.
    status, body = _request(server, "/stats")
    assert status == 200
    assert set(body) == {
        "solved_entries",
        "bias_entries",
        "rejected_rows",
        "rejected_hashes",
    }
    assert body["solved_entries"] >= 1
    assert all(isinstance(v, int) and v >= 0 for v in body.values())


def test_unsolvable_junk_is_refused_honestly(server):
    status, body = _request(server, "/synthesize", JUNK_REQUEST)
    assert status == 200
    assert body["success"] is False, f"solver fabricated code for junk: {body}"
    assert body["code"] in (None, "")
    assert body["transpiled"] is None
    assert isinstance(body["error"], str) and body["error"]


def test_request_timeout_is_honest_refusal(server):
    # A patternless scalar mapping sends the solver into a long grind; the
    # per-request timeout converts that into an honest structured refusal.
    request = {
        "name": "junk_scalar",
        "timeout_s": 3,
        "examples": [
            {"inputs": [1], "expected": 7},
            {"inputs": [2], "expected": -3},
            {"inputs": [3], "expected": 100},
            {"inputs": [4], "expected": 5},
            {"inputs": [5], "expected": -911},
            {"inputs": [6], "expected": 42424},
        ],
    }
    status, body = _request(server, "/synthesize", request)
    assert status == 200
    assert body["success"] is False
    assert body["error"] == "timeout"
    assert body["code"] is None
    assert body["transpiled"] is None


def test_malformed_json_body_is_400(server):
    status, body = _request(server, "/synthesize", "{not json")
    assert status == 400
    assert "error" in body


@pytest.mark.parametrize(
    "bad_body",
    [
        {},  # missing everything
        {"name": "x"},  # missing examples
        {"name": "x", "examples": []},  # empty examples
        {"name": "", "examples": [{"inputs": [1], "expected": 2}]},  # empty name
        {"name": "x", "examples": [{"inputs": [1]}]},  # missing expected
        {"name": "x", "examples": [{"expected": 2}]},  # missing inputs
        {"name": "x", "examples": [{"inputs": [1.5], "expected": 2}]},  # float input
        {"name": "x", "examples": [{"inputs": [1], "expected": "two"}]},  # str expected
        {"name": "x", "examples": [{"inputs": [True], "expected": 1}]},  # bool input
        {"name": "x", "examples": [{"inputs": [2**70], "expected": 1}]},  # > i64
        {"name": "x", "examples": [{"inputs": [1], "expected": 2}], "timeout_s": -1},
        {"name": "x", "examples": "nope"},  # examples wrong type
        [1, 2, 3],  # body not an object
    ],
)
def test_invalid_requests_are_400_not_500(server, bad_body):
    status, body = _request(server, "/synthesize", bad_body)
    assert status == 400, f"expected 400 for {bad_body!r}, got {status}: {body}"
    assert isinstance(body.get("error"), str) and body["error"]


def test_unknown_paths_are_404(server):
    status, _ = _request(server, "/nope")
    assert status == 404
    status, _ = _request(server, "/nope", {"x": 1})
    assert status == 404
