"""Tests for POST /prompt — the natural-language front door.

Free-form prompt in → deterministic parser extracts I/O pairs → cascade
(template_match, then nsynth_fast) → verified Python function out.
Covers:

1. Arrow-notation prompt solved by the template tier.
2. A template-resistant mapping solved by the nsynth tier (proving the
   formerly-stubbed ``nsynth_fast`` bridge works end to end).
3. Honest refusals: no examples, no entry point, patternless mapping.
4. Parse-layer unit tests on ``handle_prompt_request`` (no backend hit).
5. ``nsynth_fast`` type-gate unit tests.

Reuses the live-server fixture conventions from test_server.py — the
isolated env matters here too (the method router would otherwise reroute
the easy problems onto slow paths).
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from tests.synthesis_api.test_server import (
    BACKEND,
    REPO_ROOT,
    _free_port,
    _request,
)

pytestmark = pytest.mark.skipif(
    not BACKEND.is_file(),
    reason="mog_synth release binary not built (run: cargo build --release in nsynth/)",
)


ARROW_PROMPT = """Implement integer addition.

def add(a, b):
    \"\"\"Return the sum of a and b.\"\"\"

add(1, 2) -> 3
add(5, 7) -> 12
add(0, 0) -> 0
"""

# a + 2*b is not in the template library's scalar-2 set, so this must
# fall through to the nsynth tier.
NSYNTH_PROMPT = """def weighted(a, b):
    \"\"\"Return a plus twice b.\"\"\"

weighted(1, 2) -> 5
weighted(3, 4) -> 11
weighted(0, 0) -> 0
weighted(2, 1) -> 4
"""

# String→int with no pattern: templates skip non-scalar args, nsynth
# refuses (same mapping as test_server's JUNK_REQUEST).
JUNK_PROMPT = """def junk(s):
    \"\"\"???\"\"\"

junk("ab") -> 941
junk("xyz") -> -17
junk("q") -> 30303
junk("hello") -> -2
junk("zz") -> 777777
"""


@pytest.fixture(scope="module")
def server(tmp_path_factory, nsynth_isolated_env):
    """Live server on a free port with isolated banks (see test_server)."""
    banks = tmp_path_factory.mktemp("nsynth_banks_prompt")
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


def test_arrow_prompt_solved_by_templates(server):
    status, body = _request(
        server, "/prompt", {"prompt": ARROW_PROMPT, "timeout_s": 20}
    )
    assert status == 200
    assert body["success"] is True, body
    assert body["method"] == "template_match"
    assert body["entry_point"] == "add"
    assert body["io_pairs"] == 3
    assert "arrow" in body["pair_sources"]
    assert "def add" in body["code"]
    # the returned function must actually work
    ns: dict = {}
    exec(body["code"], ns)  # noqa: S102 — our own verified output
    assert ns["add"](20, 22) == 42


def test_template_resistant_prompt_falls_through_to_nsynth(server):
    status, body = _request(
        server, "/prompt", {"prompt": NSYNTH_PROMPT, "timeout_s": 30}
    )
    assert status == 200
    assert body["success"] is True, body
    assert body["method"] == "nsynth_fast"
    assert "def weighted" in body["code"]
    ns: dict = {}
    exec(body["code"], ns)  # noqa: S102
    assert ns["weighted"](10, 5) == 20


def test_patternless_prompt_is_refused_honestly(server):
    status, body = _request(
        server, "/prompt", {"prompt": JUNK_PROMPT, "timeout_s": 20}
    )
    assert status == 200
    assert body["success"] is False
    assert body["code"] is None
    assert body["io_pairs"] == 5


def test_prompt_without_examples_refuses_with_guidance(server):
    status, body = _request(
        server,
        "/prompt",
        {"prompt": "def mystery(x):\n    \"\"\"Do something.\"\"\"\n"},
    )
    assert status == 200
    assert body["success"] is False
    assert "no examples" in body["error"]


def test_prompt_without_entry_point_refuses(server):
    status, body = _request(
        server, "/prompt", {"prompt": "please write me something nice"}
    )
    assert status == 200
    assert body["success"] is False
    assert "no entry point" in body["error"]


def test_missing_prompt_field_is_400(server):
    status, body = _request(server, "/prompt", {"nope": 1})
    assert status == 400
    assert "prompt" in body["error"]


# ---------------------------------------------------------------------------
# unit tests — no HTTP, no backend
# ---------------------------------------------------------------------------


def test_handle_prompt_request_rejects_oversized_prompt():
    from ncpu.synthesis_api.server import (
        MAX_PROMPT_CHARS,
        SynthConfig,
        handle_prompt_request,
    )

    status, body = handle_prompt_request(
        {"prompt": "x" * (MAX_PROMPT_CHARS + 1)}, SynthConfig()
    )
    assert status == 400
    assert "too long" in body["error"]


def test_nsynth_fast_type_gates():
    from ncpu.autoresearch.solvers import nsynth_fast
    from ncpu.autoresearch.types import IoPair, WorkItem

    def _item(pairs):
        return WorkItem(
            task_id="t",
            source_benchmark="user",
            prompt="def f(a):\n    pass\n",
            entry_point="f",
            test_source="def check(candidate):\n    pass\n",
            io_pairs=pairs,
        )

    # kwargs are not expressible in nsynth's positional problem format
    assert nsynth_fast(_item([IoPair(args=[1], kwargs={"k": 2}, expected=3)])) is None
    # boolean outputs are not i64
    assert nsynth_fast(_item([IoPair(args=[1], kwargs={}, expected=True)])) is None
    # float args are unsupported
    assert nsynth_fast(_item([IoPair(args=[1.5], kwargs={}, expected=3)])) is None
    # empty pairs: nothing to synthesize from
    assert nsynth_fast(_item([])) is None
