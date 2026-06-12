"""Tests for the out-of-domain tier: verify_candidate + run_program.

The cascade made complete: synthesis covers its verified space; anything
else is refused with a protocol the client can follow — draft the code
itself, push it through ``verify_candidate`` (the same example-
verification gate), then execute it with ``run_program``.

Protocol-level tests drive the real server subprocess via the shared
``client`` fixture; pure sandbox edge cases call the functions directly
(no Rust backend needed for those).
"""

from __future__ import annotations

import shutil
import time

from ncpu.mcp_server.sandbox import run_program, verify_candidate

ADD_TWO_CODE = "def add_two(a, b):\n    return a + b\n"

ADD_TWO_EXAMPLES = [
    {"inputs": [1, 2], "expected": 3},
    {"inputs": [5, 7], "expected": 12},
    {"inputs": [0, 0], "expected": 0},
]

# Patternless string→int junk: refused fast by the solver (no such family).
JUNK_EXAMPLES = [
    {"inputs": ["qzjvw"], "expected": 174},
    {"inputs": ["xb"], "expected": -93},
    {"inputs": ["plmnor"], "expected": 40183},
]


# ---------------------------------------------------------------------------
# Tool 5: verify_candidate
# ---------------------------------------------------------------------------


def test_verify_candidate_accepts_correct_function(client):
    payload = client.call_tool(
        "verify_candidate",
        {"name": "add_two", "code": ADD_TWO_CODE, "examples": ADD_TWO_EXAMPLES},
        timeout_s=60.0,
    )
    assert payload["verified"] is True
    assert payload["examples_checked"] == 3
    assert payload["_isError"] is False


def test_verify_candidate_rejects_subtly_wrong_function(client):
    # Correct everywhere except a == 5 — first_failure must name it.
    wrong = "def add_two(a, b):\n    return a + b + (1 if a == 5 else 0)\n"
    payload = client.call_tool(
        "verify_candidate",
        {"name": "add_two", "code": wrong, "examples": ADD_TWO_EXAMPLES},
        timeout_s=60.0,
    )
    assert payload["verified"] is False
    assert payload["failures"] == 1
    failure = payload["first_failure"]
    assert failure["example_index"] == 1
    assert failure["inputs"] == [5, 7]
    assert failure["expected"] == 12
    assert failure["got"] == 13


def test_verify_candidate_catches_runtime_exception(client):
    code = "def safe_div(a, b):\n    return a // b\n"
    examples = [
        {"inputs": [10, 2], "expected": 5},
        {"inputs": [7, 0], "expected": 0},  # ZeroDivisionError here
    ]
    payload = client.call_tool(
        "verify_candidate",
        {"name": "safe_div", "code": code, "examples": examples},
        timeout_s=60.0,
    )
    assert payload["verified"] is False
    failure = payload["first_failure"]
    assert failure["example_index"] == 1
    assert failure["inputs"] == [7, 0]
    assert "ZeroDivisionError" in failure["error"]


def test_verify_candidate_enforces_timeout(client):
    code = "def spin(x):\n    while True:\n        pass\n"
    start = time.monotonic()
    payload = client.call_tool(
        "verify_candidate",
        {
            "name": "spin",
            "code": code,
            "examples": [{"inputs": [1], "expected": 1}],
            "timeout_s": 2.0,
        },
        timeout_s=60.0,
    )
    elapsed = time.monotonic() - start
    assert payload["verified"] is False
    assert "timeout" in payload["error"].lower()
    # Completes in ~timeout_s, not a hang (generous margin for CI load).
    assert elapsed < 8.0, f"timeout took {elapsed:.1f}s, expected ~2s"


def test_verify_candidate_float_isclose_fallback():
    # 3 * (1/3) != 1.0 exactly; math.isclose(rel_tol=1e-6) must accept it.
    result = verify_candidate(
        "third",
        "def third(x):\n    return x * (1.0 / 3.0) * 3.0\n",
        [{"inputs": [1.0], "expected": 1.0}, {"inputs": [7.0], "expected": 7.0}],
    )
    assert result["verified"] is True
    # ...but a genuinely different float must still fail.
    result = verify_candidate(
        "third",
        "def third(x):\n    return x + 0.001\n",
        [{"inputs": [1.0], "expected": 1.0}],
    )
    assert result["verified"] is False


def test_verify_candidate_missing_function_name():
    result = verify_candidate(
        "expected_name",
        "def other_name(a):\n    return a\n",
        [{"inputs": [1], "expected": 1}],
    )
    assert result["verified"] is False
    assert "expected_name" in result["error"]


def test_verify_candidate_syntax_error_captured():
    result = verify_candidate(
        "f",
        "def f(a:\n    return a\n",
        [{"inputs": [1], "expected": 1}],
    )
    assert result["verified"] is False
    assert "SyntaxError" in result["error"]


def test_verify_candidate_unsupported_language():
    result = verify_candidate(
        "f",
        "IDENTIFICATION DIVISION.",
        [{"inputs": [1], "expected": 1}],
        language="cobol",
    )
    assert result["verified"] is False
    assert "unsupported" in result["error"]


def test_verify_candidate_javascript_or_clean_skip():
    """With node installed the JS path verifies; without it, the tool
    must return a clear unsupported message — never crash."""
    result = verify_candidate(
        "addTwo",
        "function addTwo(a, b) { return a + b; }",
        [{"inputs": [1, 2], "expected": 3}, {"inputs": [5, 7], "expected": 12}],
        language="javascript",
    )
    if shutil.which("node"):
        assert result["verified"] is True
        assert result["examples_checked"] == 2
    else:
        assert result["verified"] is False
        assert "node" in result["error"]


def test_verify_candidate_malformed_examples():
    result = verify_candidate("f", "def f(a):\n    return a\n", [])
    assert result["verified"] is False
    result = verify_candidate(
        "f", "def f(a):\n    return a\n", [{"inputs": 5, "expected": 5}]
    )
    assert result["verified"] is False


# ---------------------------------------------------------------------------
# Tool 6: run_program
# ---------------------------------------------------------------------------


def test_run_program_returns_output(client):
    payload = client.call_tool(
        "run_program",
        {"name": "add_two", "code": ADD_TWO_CODE, "inputs": [20, 22]},
        timeout_s=60.0,
    )
    assert payload["ok"] is True
    assert payload["output"] == 42


def test_run_program_batch_mode(client):
    payload = client.call_tool(
        "run_program",
        {
            "name": "add_two",
            "code": ADD_TWO_CODE,
            "inputs": [[1, 2], [10, 20], [-5, 5]],
            "batch": True,
        },
        timeout_s=60.0,
    )
    assert payload["ok"] is True
    assert [entry["output"] for entry in payload["outputs"]] == [3, 30, 0]
    assert all(entry["ok"] for entry in payload["outputs"])


def test_run_program_error_path(client):
    payload = client.call_tool(
        "run_program",
        {
            "name": "boom",
            "code": "def boom(a):\n    return 1 // a\n",
            "inputs": [0],
        },
        timeout_s=60.0,
    )
    assert payload["ok"] is False
    assert "ZeroDivisionError" in payload["error"]


def test_run_program_batch_isolates_per_call_errors():
    result = run_program(
        "inv",
        "def inv(a):\n    return 10 // a\n",
        [[2], [0], [5]],
        batch=True,
    )
    assert result["ok"] is True
    assert result["outputs"][0] == {"ok": True, "output": 5}
    assert result["outputs"][1]["ok"] is False
    assert "ZeroDivisionError" in result["outputs"][1]["error"]
    assert result["outputs"][2] == {"ok": True, "output": 2}


def test_run_program_batch_requires_list_of_lists():
    result = run_program("f", "def f(a):\n    return a\n", [1, 2], batch=True)
    assert result["ok"] is False
    assert "list of argument lists" in result["error"]


def test_run_program_timeout():
    start = time.monotonic()
    result = run_program(
        "spin",
        "def spin(x):\n    while True:\n        pass\n",
        [1],
        timeout_s=2.0,
    )
    assert result["ok"] is False
    assert "timeout" in result["error"].lower()
    assert time.monotonic() - start < 8.0


# ---------------------------------------------------------------------------
# Refusal protocol: refusals now carry the verify_candidate hand-off
# ---------------------------------------------------------------------------


def test_refusal_includes_protocol_and_echoed_examples(client):
    payload = client.call_tool(
        "synthesize_from_examples",
        {"name": "junk", "examples": JUNK_EXAMPLES},
        timeout_s=60.0,
    )
    assert payload["verified"] is False
    assert "code" not in payload  # refusals still never carry code
    assert "verify_candidate" in payload["guidance"]
    assert "verified code" in payload["guidance"]
    assert payload["examples"] == JUNK_EXAMPLES  # ready for verify_candidate


# ---------------------------------------------------------------------------
# End-to-end cascade: synthesis refuses → client drafts → verify → run
# ---------------------------------------------------------------------------


def test_full_cascade_refuse_draft_verify_run(client):
    """One walk through all three calls: a task outside the synthesizer's
    domain (string output) but trivially valid Python. The server refuses
    honestly, the client (this test) drafts the function, the sandbox
    verifies it against the SAME echoed examples, and run_program turns a
    fresh input into actual output."""
    examples = [
        {"inputs": ["hello"], "expected": "olleh"},
        {"inputs": ["abc"], "expected": "cba"},
        {"inputs": [""], "expected": ""},
    ]

    # 1. Synthesis refuses: string outputs are outside the solver domain.
    refusal = client.call_tool(
        "synthesize_from_examples",
        {"name": "reverse_string", "examples": examples},
        timeout_s=60.0,
    )
    assert refusal["verified"] is False
    assert "code" not in refusal
    assert "verify_candidate" in refusal["guidance"]
    assert refusal["examples"] == examples

    # 2. Client-side draft (this is what the MCP client would generate).
    draft = "def reverse_string(s):\n    return s[::-1]\n"

    # 3. The draft is admitted only through the same verification gate,
    #    using the examples echoed by the refusal — no re-parsing.
    verdict = client.call_tool(
        "verify_candidate",
        {"name": "reverse_string", "code": draft, "examples": refusal["examples"]},
        timeout_s=60.0,
    )
    assert verdict["verified"] is True
    assert verdict["examples_checked"] == 3

    # 4. NL → code → actual program output on an unseen input.
    run = client.call_tool(
        "run_program",
        {"name": "reverse_string", "code": draft, "inputs": ["cascade"]},
        timeout_s=60.0,
    )
    assert run["ok"] is True
    assert run["output"] == "edacsac"
