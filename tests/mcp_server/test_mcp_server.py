"""End-to-end tests for the NL → verified-program MCP server (Rung 7).

The server runs as a real subprocess speaking newline-delimited JSON-RPC
over stdio — exactly how an MCP client (Claude Code, Cursor, ...) drives
it. All persistent banks are isolated in a tmp directory (see conftest).
"""

from __future__ import annotations

import pytest

ADD_TWO_EXAMPLES = [
    {"inputs": [1, 2], "expected": 3},
    {"inputs": [5, 7], "expected": 12},
    {"inputs": [0, 0], "expected": 0},
]

# Patternless string→int junk: the solver has no string→arbitrary-int
# family, so it refuses fast (<1s measured) instead of grinding.
JUNK_EXAMPLES = [
    {"inputs": ["qzjvw"], "expected": 174},
    {"inputs": ["xb"], "expected": -93},
    {"inputs": ["plmnor"], "expected": 40183},
]


# ---------------------------------------------------------------------------
# Protocol surface
# ---------------------------------------------------------------------------


def test_initialize_handshake(client):
    response = client.request("ping", timeout_s=30.0)
    assert response["result"] == {}


def test_tools_list_returns_four_tools(client):
    response = client.request("tools/list", timeout_s=30.0)
    tools = response["result"]["tools"]
    assert len(tools) == 4
    names = {t["name"] for t in tools}
    assert names == {
        "synthesize_from_examples",
        "synthesize_from_prompt",
        "consult_library",
        "library_stats",
    }
    for tool in tools:
        assert tool["description"]
        assert tool["inputSchema"]["type"] == "object"


def test_unknown_tool_is_invalid_params(client):
    response = client.request(
        "tools/call", {"name": "no_such_tool", "arguments": {}}, timeout_s=30.0
    )
    assert response["error"]["code"] == -32602


def test_unknown_method_not_found(client):
    response = client.request("resources/list", timeout_s=30.0)
    assert response["error"]["code"] == -32601


# ---------------------------------------------------------------------------
# Tool 1: synthesize_from_examples
# ---------------------------------------------------------------------------


def test_synthesize_from_examples_add_two(client):
    payload = client.call_tool(
        "synthesize_from_examples",
        {"name": "add_two", "examples": ADD_TWO_EXAMPLES, "language": "python"},
    )
    assert payload["verified"] is True
    assert payload["examples_checked"] == 3
    assert payload["method"]
    assert "fn add_two" in payload["mog"]
    assert "def" in payload["code"]
    assert payload["_isError"] is False


def test_synthesize_from_examples_honest_refusal(client):
    payload = client.call_tool(
        "synthesize_from_examples",
        {"name": "junk", "examples": JUNK_EXAMPLES},
        timeout_s=60.0,
    )
    assert payload["verified"] is False
    assert payload["reason"]
    assert "code" not in payload  # refusals carry no code, ever


def test_synthesize_from_examples_bad_language(client):
    payload = client.call_tool(
        "synthesize_from_examples",
        {"name": "add_two", "examples": ADD_TWO_EXAMPLES, "language": "cobol"},
        timeout_s=30.0,
    )
    assert payload["verified"] is False
    assert "cobol" in payload["reason"]


def test_synthesize_from_examples_malformed_examples(client):
    payload = client.call_tool(
        "synthesize_from_examples",
        {"name": "f", "examples": [{"inputs": [1.5], "expected": 2}]},
        timeout_s=30.0,
    )
    assert payload["verified"] is False


# ---------------------------------------------------------------------------
# Tool 2: synthesize_from_prompt
# ---------------------------------------------------------------------------


def test_synthesize_from_prompt_end_to_end(client):
    prompt = (
        "Write a function double_plus_one such that double_plus_one(3) -> 7, "
        "double_plus_one(0) -> 1, double_plus_one(10) -> 21, "
        "double_plus_one(-2) -> -3"
    )
    payload = client.call_tool(
        "synthesize_from_prompt", {"prompt": prompt, "language": "python"}
    )
    assert payload["function_name"] == "double_plus_one"
    assert len(payload["extracted_examples"]) == 4
    assert {"args": [3], "kwargs": {}, "expected": 7} in payload["extracted_examples"]
    assert payload["verified"] is True
    assert "def" in payload["code"]
    assert "double_plus_one" in payload["mog"]


def test_synthesize_from_prompt_no_examples_guidance(client):
    payload = client.call_tool(
        "synthesize_from_prompt",
        {"prompt": "Write me a function that sorts a list, please."},
        timeout_s=30.0,
    )
    assert payload["verified"] is False
    assert payload["reason"] == "no I/O examples found"
    assert "f(2,3) -> 5" in payload["guidance"]


# ---------------------------------------------------------------------------
# Tool 3: consult_library
# ---------------------------------------------------------------------------


def test_consult_library_hit_after_synthesis(client):
    """Cross-validates the Python fingerprint mirror against the Rust one:
    the synthesizer records add_two under its own fingerprint, and we must
    find it under ours."""
    first = client.call_tool(
        "synthesize_from_examples",
        {"name": "add_two", "examples": ADD_TWO_EXAMPLES},
    )
    assert first["verified"] is True

    payload = client.call_tool(
        "consult_library", {"examples": ADD_TWO_EXAMPLES}, timeout_s=30.0
    )
    assert payload["hit"] is True
    assert payload["fingerprint"] == "i:1|i:2~3;;i:5|i:7~12;;i:0|i:0~0"
    assert "fn add_two" in payload["mog"]  # the verified cached program
    assert "+" in payload["mog"]
    assert payload["method"]


def test_consult_library_miss(client):
    payload = client.call_tool(
        "consult_library",
        {"examples": [{"inputs": [123456, 7], "expected": 999999}]},
        timeout_s=30.0,
    )
    assert payload["hit"] is False
    assert payload["fingerprint"] == "i:123456|i:7~999999"


# ---------------------------------------------------------------------------
# Tool 4: library_stats
# ---------------------------------------------------------------------------


def test_library_stats_shape(client):
    payload = client.call_tool("library_stats", {}, timeout_s=30.0)
    for key in ("solved_entries", "rejected_rows", "rejected_hashes", "bias_entries"):
        assert key in payload, f"missing key: {key}"
        assert isinstance(payload[key], int)
        assert payload[key] >= 0


def test_library_stats_grows_after_solve(client):
    before = client.call_tool("library_stats", {}, timeout_s=30.0)
    payload = client.call_tool(
        "synthesize_from_examples",
        {
            "name": "triple",
            "examples": [
                {"inputs": [1], "expected": 3},
                {"inputs": [4], "expected": 12},
                {"inputs": [-2], "expected": -6},
            ],
        },
    )
    assert payload["verified"] is True
    after = client.call_tool("library_stats", {}, timeout_s=30.0)
    assert after["solved_entries"] >= before["solved_entries"] + 1
