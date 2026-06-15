"""Close-the-loop test: registry rejection -> misses.jsonl -> WorkItem.

The registry now optionally writes a per-line JSON record of every 422
to the path in `NCPU_REGISTRY_MISSES_PATH`. The autoresearch driver can
then run `python -m ncpu.autoresearch.cli mine-registry --misses ...`
to convert those records into canonical WorkItems, run the cascade, and
re-POST any recovered programs. This test pins both halves of that
contract so a future regression in either half shows up in CI.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tools.registry import executor

REPO_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _request(port: int, method: str, path: str, body: dict | None = None):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


# A subtly-wrong v1 program: claims `sum` but runs `sum of squares`.
WRONG_SUM = {
    "name": "sum_but_actually_sum_of_squares",
    "author": "alice",
    "examples": [
        {"data": [1.0, 2.0, 3.0], "n_points": 3, "target": 6.0},
        {"data": [10.0, -4.0], "n_points": 2, "target": 6.0},
    ],
    "program": {
        "init_idx": 0, "transform_idx": 1, "reduce_idx": 0,
        "post_scale_idx": 0, "offset": 0.0,
    },
}


@pytest.fixture()
def registry_with_misses(tmp_path):
    port = _free_port()
    db = tmp_path / "registry.sqlite"
    misses = tmp_path / "misses.jsonl"
    proc = subprocess.Popen(
        [sys.executable, "-m", "tools.registry.server", "--port", str(port), "--db", str(db)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, "NCPU_REGISTRY_MISSES_PATH": str(misses)},
    )
    try:
        deadline = time.time() + 15.0
        while True:
            try:
                status, _ = _request(port, "GET", "/health")
                if status == 200:
                    break
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            if proc.poll() is not None:
                out = proc.stdout.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"server died: {out!r}")
            if time.time() > deadline:
                raise RuntimeError("server did not become ready in 15s")
            time.sleep(0.05)
        yield port, misses
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_422_writes_to_misses_file(registry_with_misses):
    port, misses = registry_with_misses
    status, body = _request(port, "POST", "/skills", WRONG_SUM)
    assert status == 422
    assert body["accepted"] is False
    assert misses.exists()
    lines = misses.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["name"] == WRONG_SUM["name"]
    assert record["author"] == "alice"
    assert record["examples"] == WRONG_SUM["examples"]
    assert record["first_failure"]["example_index"] == 0
    assert "ts" in record


def test_accepted_submission_does_not_write_miss(registry_with_misses):
    port, misses = registry_with_misses
    good = json.loads(json.dumps(WRONG_SUM))
    good["name"] = "real_sum"
    good["program"]["transform_idx"] = 0
    status, _ = _request(port, "POST", "/skills", good)
    assert status == 200
    # No miss was written; the file may not exist.
    if misses.exists():
        assert misses.read_text() == ""


def test_registry_miss_to_workitem_round_trip(tmp_path, monkeypatch):
    """End-to-end: registry miss JSONL -> mine-registry -> WorkItem.

    Goes directly through the helper rather than spawning a server, so it
    stays fast and self-contained.
    """
    from ncpu.autoresearch.sources.registry import mine_registry_misses

    misses = tmp_path / "misses.jsonl"
    queue = tmp_path / "queue.jsonl"
    misses.write_text(json.dumps({
        "name": "sum_but_actually_sum_of_squares",
        "author": "alice",
        "examples": WRONG_SUM["examples"],
        "first_failure": {"example_index": 0, "expected": 6.0, "got": 14.0},
        "ts": "2026-06-14T20:00:00+00:00",
    }) + "\n")
    counters = mine_registry_misses(misses, queue)
    assert counters == {"read": 1, "emitted": 1, "skipped": 0}
    payload = json.loads(queue.read_text().strip().splitlines()[0])
    assert payload["source_benchmark"] == "registry"
    assert payload["entry_point"] == "sum_but_actually_sum_of_squares"
    assert payload["io_pairs"][0]["args_repr"] == ["[1.0, 2.0, 3.0]"]
    assert payload["io_pairs"][0]["expected_repr"] == "6"
    # The provenance carries the registry's failure context, so the
    # cascade (and any human reviewer) can see *why* the skill was
    # rejected without re-running the registry.
    assert payload["provenance"]["registry_first_failure"]["example_index"] == 0
