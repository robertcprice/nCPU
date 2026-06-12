"""Stdio MCP server: natural language → verified program (Rung 7).

A minimal, dependency-free implementation of the Model Context Protocol
(2024-11-05 revision) over stdio: newline-delimited JSON-RPC 2.0 on
stdin/stdout, logs on stderr. The official ``mcp`` Python package is not
a dependency of this repo (stdlib-only norm), so the protocol surface —
``initialize`` / ``notifications/initialized`` / ``tools/list`` /
``tools/call`` / ``ping`` — is implemented by hand. Any MCP client
(Claude Code, Claude Desktop, Cursor, ...) can connect:

    claude mcp add ncpu-synth -- python3 -m ncpu.mcp_server

Exposed tools (see ``tools.py`` / ``sandbox.py`` for behavior,
``README.md`` for docs):

1. ``synthesize_from_examples`` — examples in, proof-carrying code out.
2. ``synthesize_from_prompt``   — NL prompt → extracted examples → tool 1.
3. ``consult_library``          — instant answer from the solved bank.
4. ``library_stats``            — memory-bank sizes (observable learning).
5. ``verify_candidate``         — client-drafted code through the same
   example-verification gate (out-of-domain tier of the cascade).
6. ``run_program``              — execute a verified program on new inputs.

Honest-refusal contract: a result is either ``verified: true`` with code
that reproduced every example (inside the synthesizer for tools 1-3,
inside the subprocess sandbox for tool 5), or ``verified: false`` with a
reason. The server never fabricates code — out-of-domain refusals tell
the client to draft code itself and submit it through ``verify_candidate``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional, TextIO

from ncpu.mcp_server import tools as _tools
from ncpu.mcp_server.sandbox import (
    DEFAULT_SANDBOX_TIMEOUT_S,
    MAX_SANDBOX_TIMEOUT_S,
)
from ncpu.synthesis_api.server import (
    DEFAULT_TIMEOUT_S,
    MAX_TIMEOUT_S,
    SynthConfig,
    default_backend_path,
)

log = logging.getLogger("ncpu.mcp_server")

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "ncpu-synth", "version": "0.1.0"}

# JSON-RPC error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


_EXAMPLES_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "description": (
        "Input/output examples. Each input value is an int, a list of "
        "ints, or a string; expected is always an int."
    ),
    "items": {
        "type": "object",
        "properties": {
            "inputs": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "anyOf": [
                        {"type": "integer"},
                        {"type": "array", "items": {"type": "integer"}},
                        {"type": "string"},
                    ]
                },
            },
            "expected": {"type": "integer"},
        },
        "required": ["inputs", "expected"],
    },
}

_LANGUAGE_SCHEMA = {
    "type": "string",
    "enum": ["python", "rust", "typescript"],
    "default": "python",
    "description": "Language to transpile the verified Mog program into.",
}

_TIMEOUT_SCHEMA = {
    "type": "number",
    "exclusiveMinimum": 0,
    "description": (
        f"Optional solver timeout in seconds (default "
        f"{DEFAULT_TIMEOUT_S:.0f}, max {MAX_TIMEOUT_S:.0f}). On timeout "
        "the tool returns an honest refusal, never fabricated code."
    ),
}

_CANDIDATE_EXAMPLES_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "description": (
        "Input/output examples for the candidate. Unlike the synthesis "
        "tools, any JSON values are allowed for inputs and expected."
    ),
    "items": {
        "type": "object",
        "properties": {
            "inputs": {
                "type": "array",
                "description": "Positional arguments for one call.",
            },
            "expected": {
                "description": (
                    "Expected return value (any JSON value). Compared "
                    "with == plus a math.isclose fallback for floats "
                    "(rel_tol 1e-6)."
                )
            },
        },
        "required": ["inputs", "expected"],
    },
}

_SANDBOX_LANGUAGE_SCHEMA = {
    "type": "string",
    "enum": ["python", "javascript"],
    "default": "python",
    "description": (
        "Language the candidate code is written in. 'python' always "
        "works; 'javascript' requires node on the host (a clean "
        "unsupported message is returned otherwise)."
    ),
}

_SANDBOX_TIMEOUT_SCHEMA = {
    "type": "number",
    "exclusiveMinimum": 0,
    "default": DEFAULT_SANDBOX_TIMEOUT_S,
    "description": (
        f"Wall-clock sandbox timeout in seconds (default "
        f"{DEFAULT_SANDBOX_TIMEOUT_S:.0f}, max {MAX_SANDBOX_TIMEOUT_S:.0f}). "
        "On expiry the whole process tree is killed."
    ),
}

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "synthesize_from_examples",
        "description": (
            "Synthesize a program verified against the given input/output "
            "examples, then transpile it to the requested language. The "
            "result is proof-carrying: `verified: true` means the program "
            "reproduced every example inside the synthesizer. When no "
            "program is found the tool returns `verified: false` with a "
            "reason — it never fabricates code."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Function name for the generated code.",
                },
                "examples": _EXAMPLES_SCHEMA,
                "language": _LANGUAGE_SCHEMA,
                "timeout_s": _TIMEOUT_SCHEMA,
            },
            "required": ["name", "examples"],
        },
    },
    {
        "name": "synthesize_from_prompt",
        "description": (
            "Extract input/output examples from a natural-language prompt "
            "(arrow notation `f(2,3) -> 5`, doctests, asserts, 'returns' "
            "prose), echo the extracted examples back, and synthesize a "
            "verified program from them. If the prompt contains no "
            "examples, the tool replies with guidance on what to provide — "
            "supply examples, not code."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Free-form natural-language request.",
                },
                "language": _LANGUAGE_SCHEMA,
                "timeout_s": _TIMEOUT_SCHEMA,
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "consult_library",
        "description": (
            "Check the persistent solved bank for an exact "
            "examples-fingerprint match and return the cached verified "
            "program instantly on a hit. Misses return `hit: false` with "
            "the computed fingerprint."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"examples": _EXAMPLES_SCHEMA},
            "required": ["examples"],
        },
    },
    {
        "name": "library_stats",
        "description": (
            "Sizes of the synthesizer's three persistent memory banks "
            "(solved programs, learned biases, rejected programs) — the "
            "observable cross-session learning state."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "verify_candidate",
        "description": (
            "Execute a client-drafted candidate function against EVERY "
            "example in a subprocess sandbox and report whether all of "
            "them reproduce. This is the out-of-domain tier of the "
            "cascade: when synthesis refuses, draft the function yourself "
            "and submit it here with the same examples — only verified "
            "code should be shown to the user. The result is "
            "`verified: true` with `examples_checked`, or `verified: "
            "false` with the first counterexample (`first_failure`) or "
            "the captured error. Sandbox: fresh `python3 -I` process in a "
            "temp dir, scrubbed env, hard timeout with process-tree kill "
            "— the same trust as any local coding agent executing code "
            "it just wrote."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Function name the code must define; called once "
                        "per example as name(*inputs)."
                    ),
                },
                "code": {
                    "type": "string",
                    "description": "Candidate source code defining `name`.",
                },
                "examples": _CANDIDATE_EXAMPLES_SCHEMA,
                "language": _SANDBOX_LANGUAGE_SCHEMA,
                "timeout_s": _SANDBOX_TIMEOUT_SCHEMA,
            },
            "required": ["name", "code", "examples"],
        },
    },
    {
        "name": "run_program",
        "description": (
            "Execute a (previously verified) program on new inputs in the "
            "same subprocess sandbox as verify_candidate: calls "
            "name(*inputs) once and returns `{ok: true, output}` or "
            "`{ok: false, error}`. With batch=true, inputs must be a "
            "list of argument lists and the result carries one entry per "
            "call. Closes the loop: NL -> code -> actual program output."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Function name the code defines.",
                },
                "code": {
                    "type": "string",
                    "description": "Source code defining `name`.",
                },
                "inputs": {
                    "type": "array",
                    "description": (
                        "Positional arguments for one call, or (with "
                        "batch=true) a list of argument lists."
                    ),
                },
                "batch": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Treat inputs as a list of calls and return "
                        "outputs per call."
                    ),
                },
                "language": _SANDBOX_LANGUAGE_SCHEMA,
                "timeout_s": _SANDBOX_TIMEOUT_SCHEMA,
            },
            "required": ["name", "code", "inputs"],
        },
    },
]


class McpServer:
    """Newline-delimited JSON-RPC 2.0 loop over a pair of text streams."""

    def __init__(self, config: SynthConfig) -> None:
        self.config = config

    # -- tool dispatch ------------------------------------------------------

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "synthesize_from_examples":
            return _tools.synthesize_from_examples(
                self.config,
                name=arguments.get("name", ""),
                examples=arguments.get("examples", []),
                language=arguments.get("language", "python"),
                timeout_s=arguments.get("timeout_s"),
            )
        if name == "synthesize_from_prompt":
            return _tools.synthesize_from_prompt(
                self.config,
                prompt=arguments.get("prompt", ""),
                language=arguments.get("language", "python"),
                timeout_s=arguments.get("timeout_s"),
            )
        if name == "consult_library":
            return _tools.consult_library(
                self.config, examples=arguments.get("examples", [])
            )
        if name == "library_stats":
            return _tools.library_stats(self.config)
        if name == "verify_candidate":
            return _tools.verify_candidate(
                name=arguments.get("name", ""),
                code=arguments.get("code", ""),
                examples=arguments.get("examples", []),
                language=arguments.get("language", "python"),
                timeout_s=arguments.get("timeout_s", DEFAULT_SANDBOX_TIMEOUT_S),
            )
        if name == "run_program":
            return _tools.run_program(
                name=arguments.get("name", ""),
                code=arguments.get("code", ""),
                inputs=arguments.get("inputs", []),
                language=arguments.get("language", "python"),
                timeout_s=arguments.get("timeout_s", DEFAULT_SANDBOX_TIMEOUT_S),
                batch=bool(arguments.get("batch", False)),
            )
        raise KeyError(name)

    # -- JSON-RPC plumbing --------------------------------------------------

    def handle_message(self, message: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Process one JSON-RPC message; return the response (or None for
        notifications, which never get a reply)."""
        msg_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}
        is_notification = msg_id is None

        if not isinstance(method, str):
            if is_notification:
                return None
            return _error(msg_id, INVALID_REQUEST, "missing method")

        if method == "initialize":
            return _result(
                msg_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER_INFO,
                },
            )
        if method in ("notifications/initialized", "initialized"):
            return None
        if method.startswith("notifications/"):
            return None
        if method == "ping":
            return _result(msg_id, {})
        if method == "tools/list":
            return _result(msg_id, {"tools": TOOL_DEFINITIONS})
        if method == "tools/call":
            return self._handle_tools_call(msg_id, params)

        if is_notification:
            return None
        return _error(msg_id, METHOD_NOT_FOUND, f"method not found: {method}")

    def _handle_tools_call(
        self, msg_id: Any, params: dict[str, Any]
    ) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        known = {t["name"] for t in TOOL_DEFINITIONS}
        if name not in known:
            return _error(msg_id, INVALID_PARAMS, f"unknown tool: {name!r}")
        if not isinstance(arguments, dict):
            return _error(msg_id, INVALID_PARAMS, "arguments must be an object")
        try:
            payload = self.call_tool(name, arguments)
            is_error = False
        except Exception as exc:  # tool-level failure → in-band error result
            log.exception("tool %s raised", name)
            payload = {"error": f"{type(exc).__name__}: {exc}"}
            is_error = True
        return _result(
            msg_id,
            {
                "content": [
                    {"type": "text", "text": json.dumps(payload, indent=2)}
                ],
                "isError": is_error,
            },
        )

    # -- main loop ----------------------------------------------------------

    def serve(self, stdin: TextIO, stdout: TextIO) -> int:
        log.info(
            "ncpu-synth MCP server on stdio (backend=%s, present=%s)",
            self.config.backend,
            self.config.backend.is_file(),
        )
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                _write(stdout, _error(None, PARSE_ERROR, f"parse error: {exc}"))
                continue
            if not isinstance(message, dict):
                _write(stdout, _error(None, INVALID_REQUEST, "expected an object"))
                continue
            try:
                response = self.handle_message(message)
            except Exception as exc:  # pragma: no cover — defensive
                log.exception("unhandled error")
                response = _error(
                    message.get("id"), INTERNAL_ERROR, f"internal error: {exc}"
                )
            if response is not None:
                _write(stdout, response)
        log.info("stdin closed; shutting down")
        return 0


def _result(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _write(stdout: TextIO, response: dict[str, Any]) -> None:
    stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
    stdout.flush()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m ncpu.mcp_server",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument(
        "--backend",
        type=Path,
        default=default_backend_path(),
        help="path to the mog_synth release binary",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"default solver timeout in seconds (max {MAX_TIMEOUT_S:.0f})",
    )
    parser.add_argument(
        "--solved-cache",
        default=None,
        help="override NSYNTH_CACHE_PATH for the backend ('' disables)",
    )
    parser.add_argument(
        "--bias-bank",
        default=None,
        help="override NSYNTH_BIAS_BANK_PATH for the backend ('' disables)",
    )
    parser.add_argument(
        "--rejected-cache",
        default=None,
        help="override NSYNTH_REJECTED_PATH for the backend ('' disables)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    # Logs must go to stderr — stdout is the protocol channel.
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    config = SynthConfig(
        backend=args.backend,
        timeout_s=min(args.timeout, MAX_TIMEOUT_S),
        solved_cache=args.solved_cache,
        bias_bank=args.bias_bank,
        rejected_cache=args.rejected_cache,
    )
    return McpServer(config).serve(sys.stdin, sys.stdout)


__all__ = ["McpServer", "TOOL_DEFINITIONS", "PROTOCOL_VERSION", "SERVER_INFO", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
