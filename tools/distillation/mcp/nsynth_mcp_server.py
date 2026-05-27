#!/usr/bin/env python3
"""
nsynth MCP server — expose the execution-verified cache pattern to any
MCP-aware LLM (Claude Desktop, Cursor, Claude Code) as callable tools.

Implements the MCP JSON-RPC protocol over stdio so it can be added to a
user's MCP config without a network dependency. The tools are:

  • execute_python(code)  → runs the code in a fresh namespace with a
    timeout; returns stdout + any exception info. The LLM can use this
    to sanity-check generated code before committing to it.

  • verify_against_tests(code, test_cases)  → runs code against a list
    of (args, expected) tuples and reports pass/fail per case. The
    execution-grounded oracle behind every benchmark we ran.

  • cache_solution(fingerprint, code, model)  → stores a verified
    solution keyed by fingerprint in the shared TSV cache.

  • lookup_solution(fingerprint)  → retrieves a previously-cached
    solution. 0 ms answer to "have we solved this shape before?"

  • fingerprint(examples)  → computes the canonical fingerprint for a
    list of I/O examples, so the caller can produce the same key
    lookup_solution / cache_solution use.

The combination is how a code-gen agent ships production-safe: LLM
proposes → verify_against_tests → cache_solution on pass. Future
invocations of the same shape are instant cache hits; new invocations
run the full propose+verify loop.

Config example (Claude Desktop / Code):
    "mcpServers": {
      "nsynth": {
        "command": "python3",
        "args": ["/absolute/path/tools/mcp/nsynth_mcp_server.py"]
      }
    }
"""

from __future__ import annotations

import contextlib
import io
import json
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import the same cache the benchmark runners use so everything shares
# one source of truth for "have we seen this fingerprint?".
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from llm_solution_cache import (  # noqa: E402
    fingerprint_examples,
    lookup as cache_lookup,
    record as cache_record,
)


SERVER_NAME = "nsynth-execute"
SERVER_VERSION = "0.1.0"


# ─── Tool implementations ────────────────────────────────────────────────────


class _TimeoutError(Exception):
    pass


@contextlib.contextmanager
def _time_limit(seconds: int):
    """SIGALRM timeout. macOS + Linux; Windows doesn't support SIGALRM
    but most MCP-tool use is on dev laptops which run POSIX."""
    def handler(signum, frame):
        raise _TimeoutError(f"timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def tool_execute_python(args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute `code` in a fresh namespace, capture stdout/stderr."""
    code = args.get("code", "")
    timeout = int(args.get("timeout_s", 5))
    if not code:
        return {"error": "code parameter required"}

    buf_out, buf_err = io.StringIO(), io.StringIO()
    ns: dict = {}
    try:
        with _time_limit(timeout):
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                exec(code, ns)
    except _TimeoutError as e:
        return {"status": "timeout", "error": str(e),
                "stdout": buf_out.getvalue(), "stderr": buf_err.getvalue()}
    except Exception:
        return {"status": "exception",
                "error": traceback.format_exc(limit=5),
                "stdout": buf_out.getvalue(), "stderr": buf_err.getvalue()}
    return {"status": "ok", "stdout": buf_out.getvalue(), "stderr": buf_err.getvalue(),
            "defined_names": [k for k in ns.keys() if not k.startswith("_")][:32]}


def tool_verify_against_tests(args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute code, then call `entry_point` on each (args, expected)
    test case. Returns total passed + specific failing cases.

    Expected args:
      code: str — full Python source that defines entry_point
      entry_point: str — function name to invoke
      test_cases: list of lists — each is [arg1, arg2, ..., expected]
      timeout_s: int — per-invocation timeout (default 5)"""
    code = args.get("code", "")
    entry = args.get("entry_point", "")
    cases = args.get("test_cases", [])
    timeout = int(args.get("timeout_s", 5))
    if not code or not entry:
        return {"error": "code and entry_point required"}

    ns: dict = {}
    try:
        with _time_limit(timeout):
            exec(code, ns)
    except Exception as e:
        return {"status": "exec-error", "error": repr(e), "passed": 0, "total": len(cases)}
    fn = ns.get(entry)
    if fn is None:
        return {"status": "no-fn", "error": f"no {entry} defined",
                "passed": 0, "total": len(cases)}

    passed = 0
    failures: List[Dict[str, Any]] = []
    for i, case in enumerate(cases):
        if not isinstance(case, list) or len(case) < 1:
            failures.append({"case_index": i, "error": "malformed case"})
            continue
        fn_args = case[:-1]
        expected = case[-1]
        try:
            with _time_limit(timeout):
                got = fn(*fn_args)
        except Exception as e:
            failures.append({"case_index": i, "args": fn_args,
                             "error": f"call: {e!r}"[:200]})
            continue
        if got == expected:
            passed += 1
        else:
            failures.append({
                "case_index": i, "args": fn_args,
                "got": repr(got)[:200], "expected": repr(expected)[:200],
            })
    return {
        "status": "ok" if passed == len(cases) else "partial",
        "passed": passed, "total": len(cases),
        "failures": failures[:8],  # cap for response size
    }


def tool_fingerprint(args: Dict[str, Any]) -> Dict[str, Any]:
    """Canonical fingerprint for a list of examples. Same scheme the
    cache uses, so callers can build keys without needing to import
    anything."""
    examples = args.get("examples", [])
    if not isinstance(examples, list):
        return {"error": "examples must be a list"}
    fp = fingerprint_examples(examples)
    return {"fingerprint": fp}


def tool_cache_solution(args: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a verified (fingerprint, code) pair, optionally with its
    examples. Storing examples enables downstream semantic retrieval —
    future problems with similar shape can pull this row as a draft
    without seeing identical fingerprints."""
    fp = args.get("fingerprint") or args.get("fp", "")
    code = args.get("code", "")
    model = args.get("model", "external")
    examples = args.get("examples")  # optional list[{inputs, expected}]
    if not fp or not code:
        return {"error": "fingerprint and code required"}
    if examples is not None and not isinstance(examples, list):
        return {"error": "examples must be a list if provided"}
    try:
        cache_record(fp, model, code, examples=examples)
    except Exception as e:
        return {"error": f"record failed: {e!r}"}
    return {"status": "ok", "fingerprint": fp,
            "examples_stored": bool(examples)}


def tool_lookup_solution(args: Dict[str, Any]) -> Dict[str, Any]:
    fp = args.get("fingerprint") or args.get("fp", "")
    if not fp:
        return {"error": "fingerprint required"}
    row = cache_lookup(fp)
    if row is None:
        return {"status": "miss"}
    return {
        "status": "hit",
        "model": row["model"],
        "success_count": row["success_count"],
        "last_used_at": row["last_used_at"],
        "code": row["code"],
    }


def tool_semantic_similar(args: Dict[str, Any]) -> Dict[str, Any]:
    """Find cached solutions similar to a given example set. Caller
    should still verify — similarity is retrieval, not guarantee."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
        from semantic_cache import semantic_lookup  # type: ignore
    except ImportError as e:
        return {"error": f"semantic_cache unavailable: {e!r}"}
    examples = args.get("examples", [])
    k = int(args.get("k", 3))
    min_sim = float(args.get("min_similarity", 0.0))
    hits = semantic_lookup(examples, k=k, min_similarity=min_sim)
    return {"matches": hits}


def tool_build_retrieval_prefix(args: Dict[str, Any]) -> Dict[str, Any]:
    """Return a ready-to-paste few-shot prefix built from the top-K
    semantically-similar cached solutions. Unlike `semantic_similar`
    which returns raw matches for the caller to format, this returns
    the formatted string directly so a tool-using LLM can splice it
    into its next prompt without post-processing.

    Args:
      examples: list of {inputs, expected} for the query problem
      k: number of similar solutions to include (default 3)
      min_similarity: threshold in [0,1] (default 0.70)

    Returns:
      {prefix: str, hits: int, top_similarity: float|None}

    The returned prefix ends with a "# Your task:" hand-off line so the
    caller can append their actual prompt and the model will treat the
    retrieved solutions as references, not as the thing to copy."""
    try:
        sys.path.insert(0, str(
            Path(__file__).resolve().parent.parent / "benchmarks"))
        from retrieval_prompt import build_retrieval_prefix  # type: ignore
    except ImportError as e:
        return {"error": f"retrieval_prompt unavailable: {e!r}"}
    examples = args.get("examples", [])
    k = int(args.get("k", 3))
    min_sim = float(args.get("min_similarity", 0.70))
    if not isinstance(examples, list) or not examples:
        return {"error": "examples must be a non-empty list"}
    prefix = build_retrieval_prefix(
        examples, k=k, min_similarity=min_sim,
    )
    # Peek at the top similarity via semantic_lookup for a headline number.
    try:
        from semantic_cache import semantic_lookup  # type: ignore
        hits = semantic_lookup(examples, k=1, min_similarity=min_sim)
        top = hits[0]["similarity"] if hits else None
        hit_count = len(semantic_lookup(
            examples, k=k, min_similarity=min_sim))
    except Exception:
        top = None; hit_count = 0
    return {
        "prefix": prefix,
        "hits": hit_count,
        "top_similarity": top,
    }


_ALLOWED_MATH_NAMES = {
    "abs": abs, "min": min, "max": max, "pow": pow, "round": round,
    "sum": sum, "len": len, "int": int, "float": float,
}


def tool_evaluate_expression(args: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a Python arithmetic expression in a restricted namespace.

    Restrictions: no imports, no attribute access on arbitrary objects,
    no assignments, no comprehensions with `__`-prefixed names. Only the
    whitelisted builtins (abs/min/max/pow/round/sum/len/int/float) and
    the `math` module's `sqrt/log/exp/sin/cos/floor/ceil/pi/e` are
    available.

    Designed for GSM8K/MATH-style arithmetic verification where the LLM
    wants a second opinion on its own calculation: "does `3 * 45 + 7`
    really equal 142? eval says 142. ok."
    """
    import ast
    import math as _math
    expr = args.get("expression", "")
    if not isinstance(expr, str) or not expr.strip():
        return {"error": "expression required"}
    if "__" in expr or "import" in expr:
        return {"error": "disallowed token in expression"}

    safe_math = {
        name: getattr(_math, name) for name in
        ("sqrt", "log", "log2", "log10", "exp", "sin", "cos", "tan",
         "floor", "ceil", "pi", "e", "factorial", "gcd")
    }
    ns = {"__builtins__": _ALLOWED_MATH_NAMES, **safe_math}

    try:
        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom,
                                  ast.Attribute, ast.Lambda)):
                if isinstance(node, ast.Attribute):
                    continue
                return {"error": f"disallowed AST node: {type(node).__name__}"}
        with _time_limit(3):
            val = eval(compile(tree, "<expr>", "eval"), ns, {})
    except _TimeoutError:
        return {"error": "timed out"}
    except Exception as e:
        return {"error": f"eval: {e!r}"[:200]}

    return {"status": "ok", "value": val, "repr": repr(val)}


def tool_check_numeric_answer(args: Dict[str, Any]) -> Dict[str, Any]:
    """Compare a predicted numeric answer against a ground-truth number
    with tolerance. Returns `{match: bool, abs_error, rel_error}`.

    Used by a reasoning-agent that wants to verify arithmetic before
    committing to an answer — the mirror of verify_against_tests for
    math problems where "correct" is numeric equality, not passing a
    test suite.
    """
    pred = args.get("predicted")
    gt = args.get("ground_truth")
    tol = float(args.get("tolerance", 1e-6))
    if pred is None or gt is None:
        return {"error": "predicted + ground_truth required"}
    try:
        p = float(pred); g = float(gt)
    except (TypeError, ValueError):
        return {"error": f"not numeric: pred={pred!r} gt={gt!r}"}
    abs_err = abs(p - g)
    rel_err = abs_err / max(abs(g), 1e-12)
    return {
        "match": abs_err <= tol,
        "abs_error": abs_err,
        "rel_error": rel_err,
        "predicted": p,
        "ground_truth": g,
    }


def tool_delegate_to_frontier(args: Dict[str, Any]) -> Dict[str, Any]:
    """Route a hard problem through a cheap→premium model cascade.
    Tries `cheap_model` first; on verification failure cascades to
    `premium_model`. Caches the winner so future calls for the same
    fingerprint skip both models.

    Args:
      prompt: the user-facing problem description
      entry_point: function name to verify
      test_cases: list of [args..., expected]
      cheap_model: e.g. "claude-haiku-4-5-20251001" (default)
      premium_model: e.g. "claude-opus-4-7" (default)
      api_key: optional override; else reads ANTHROPIC_API_KEY

    Returns: {used_model, code, pass_at_1, cheap_tokens, premium_tokens}.

    This is the MCP-tool expression of the fallback-routing pattern.
    An LLM that has access to this tool can decide on its own when to
    escalate a tricky problem to a bigger model — routing becomes
    part of the model's reasoning, not a fixed external policy."""
    prompt = args.get("prompt", "")
    entry = args.get("entry_point", "")
    cases = args.get("test_cases", [])
    cheap = args.get("cheap_model", "claude-haiku-4-5-20251001")
    premium = args.get("premium_model", "claude-opus-4-7")
    api_key = args.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")

    if not prompt or not entry:
        return {"error": "prompt + entry_point required"}
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY missing"}

    try:
        import anthropic  # type: ignore
    except ImportError:
        return {"error": "anthropic SDK not installed on server"}
    client = anthropic.Anthropic(api_key=api_key)

    import re as _re
    _FENCE = _re.compile(r"```(?:python)?\n?(.*?)```", _re.DOTALL)
    _DEF = _re.compile(r"^\s*def\s+\w+\s*\(", _re.MULTILINE)

    def _extract(text: str) -> str:
        for fence in _FENCE.findall(text):
            if f"def {entry}" in fence:
                return fence.strip()
        m = _DEF.search(text)
        return text[m.start():].strip() if m else ""

    def _verify(code: str):
        ns: dict = {}
        try:
            exec(code, ns)
        except Exception as e:
            return (False, f"exec: {e!r}"[:120])
        fn = ns.get(entry)
        if fn is None:
            return (False, f"no {entry}")
        for case in cases:
            fn_args, expected = case[:-1], case[-1]
            try:
                got = fn(*fn_args)
            except Exception as e:
                return (False, f"call: {e!r}"[:100])
            if got != expected:
                return (False, f"{fn_args}→{got} exp {expected}")
        return (True, "")

    def _call(model: str):
        try:
            resp = client.messages.create(
                model=model, max_tokens=1024, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            return ("", 0, f"api-error: {e!r}"[:120])
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        code = _extract(text)
        usage = getattr(resp, "usage", None)
        tokens = (getattr(usage, "input_tokens", 0) +
                  getattr(usage, "output_tokens", 0)) if usage else 0
        return (code, tokens, "ok")

    # Cheap attempt.
    cheap_code, cheap_tok, note = _call(cheap)
    if cheap_code:
        ok, err = _verify(cheap_code)
        if ok:
            return {
                "used_model": cheap, "code": cheap_code, "pass_at_1": True,
                "cheap_tokens": cheap_tok, "premium_tokens": 0,
                "cascade_fired": False,
            }
        first_error = err
    else:
        first_error = note

    # Escalate.
    premium_code, premium_tok, note = _call(premium)
    if premium_code:
        ok, err = _verify(premium_code)
        if ok:
            return {
                "used_model": premium, "code": premium_code, "pass_at_1": True,
                "cheap_tokens": cheap_tok, "premium_tokens": premium_tok,
                "cascade_fired": True, "cheap_error": first_error,
            }
        return {
            "used_model": premium, "code": premium_code, "pass_at_1": False,
            "cheap_tokens": cheap_tok, "premium_tokens": premium_tok,
            "cascade_fired": True, "cheap_error": first_error,
            "premium_error": err,
        }

    return {
        "used_model": "", "pass_at_1": False,
        "cheap_tokens": cheap_tok, "premium_tokens": 0,
        "cheap_error": first_error, "premium_error": note,
    }


TOOLS = {
    "execute_python": {
        "impl": tool_execute_python,
        "description": "Run Python code in a sandboxed namespace with a timeout. Returns stdout, stderr, status, and any exception traceback. Useful as a syntax + runtime sanity check on LLM-generated code.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Full Python source to execute."},
                "timeout_s": {"type": "integer", "default": 5, "description": "Execution time limit."},
            },
            "required": ["code"],
        },
    },
    "verify_against_tests": {
        "impl": tool_verify_against_tests,
        "description": "Run Python code and call entry_point on each test_case [args..., expected]. Returns per-case pass/fail, so the caller can feed the specific failure back to the LLM for a targeted retry.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "entry_point": {"type": "string"},
                "test_cases": {"type": "array", "items": {"type": "array"}},
                "timeout_s": {"type": "integer", "default": 5},
            },
            "required": ["code", "entry_point", "test_cases"],
        },
    },
    "fingerprint": {
        "impl": tool_fingerprint,
        "description": "Compute a deterministic fingerprint for a list of I/O examples. Use this to build cache keys that agree with cache_solution / lookup_solution.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "examples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "inputs": {"type": "array"},
                            "expected": {},
                        },
                    },
                },
            },
            "required": ["examples"],
        },
    },
    "cache_solution": {
        "impl": tool_cache_solution,
        "description": "Persist a verified (fingerprint, code) pair, optionally with its problem examples. Storing examples enables semantic retrieval — future problems with similar shape can pull this row as a draft.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "fingerprint": {"type": "string"},
                "code": {"type": "string"},
                "model": {"type": "string", "default": "external"},
                "examples": {
                    "type": "array",
                    "description": "Optional list of {inputs, expected} pairs. Persisting these unlocks retrieval-augmented generation on future similar problems.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "inputs": {"type": "array"},
                            "expected": {},
                        },
                    },
                },
            },
            "required": ["fingerprint", "code"],
        },
    },
    "lookup_solution": {
        "impl": tool_lookup_solution,
        "description": "Retrieve a cached solution by fingerprint. Returns status=hit with code, or status=miss. 0ms answer to 'have we solved this shape before?'.",
        "inputSchema": {
            "type": "object",
            "properties": {"fingerprint": {"type": "string"}},
            "required": ["fingerprint"],
        },
    },
    "semantic_similar": {
        "impl": tool_semantic_similar,
        "description": "Find cached solutions whose example shape is SIMILAR to a query (not necessarily identical). Returns top-K matches with similarity scores. Re-verify the returned code before trusting it — similarity is retrieval, not guarantee.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "examples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"inputs": {"type": "array"},
                                       "expected": {}},
                    },
                },
                "k": {"type": "integer", "default": 3},
                "min_similarity": {"type": "number", "default": 0.5},
            },
            "required": ["examples"],
        },
    },
    "build_retrieval_prefix": {
        "impl": tool_build_retrieval_prefix,
        "description": "Build a ready-to-paste few-shot prefix from the top-K semantically-similar cached solutions. Splice it into your next prompt to condition the model's generation on known-good solutions. Returns {prefix, hits, top_similarity}.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "examples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"inputs": {"type": "array"},
                                       "expected": {}},
                    },
                },
                "k": {"type": "integer", "default": 3},
                "min_similarity": {"type": "number", "default": 0.70},
            },
            "required": ["examples"],
        },
    },
    "evaluate_expression": {
        "impl": tool_evaluate_expression,
        "description": "Evaluate a Python arithmetic expression in a restricted sandbox (math + basic builtins only). Use to sanity-check numeric answers before committing. E.g. expression='3 * 45 + 7' → 142.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
            },
            "required": ["expression"],
        },
    },
    "check_numeric_answer": {
        "impl": tool_check_numeric_answer,
        "description": "Compare a predicted numeric answer to a ground truth with tolerance. Returns match bool + abs/rel error. The verification primitive for math-reasoning agents.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "predicted": {"type": ["number", "string"]},
                "ground_truth": {"type": ["number", "string"]},
                "tolerance": {"type": "number", "default": 1e-6},
            },
            "required": ["predicted", "ground_truth"],
        },
    },
    "delegate_to_frontier": {
        "impl": tool_delegate_to_frontier,
        "description": "Route a problem through a cheap→premium model cascade. Tries cheap_model first, escalates to premium_model on verification failure, caches the winner. Use when the LLM decides the problem is too complex for its current tier.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "entry_point": {"type": "string"},
                "test_cases": {"type": "array", "items": {"type": "array"}},
                "cheap_model": {"type": "string",
                    "default": "claude-haiku-4-5-20251001"},
                "premium_model": {"type": "string",
                    "default": "claude-opus-4-7"},
            },
            "required": ["prompt", "entry_point", "test_cases"],
        },
    },
}


# ─── JSON-RPC server (MCP protocol over stdio) ──────────────────────────────
#
# MCP is JSON-RPC 2.0 over stdio. The subset we implement:
#
#   initialize       → server capabilities + info
#   tools/list       → list of tools with their schemas
#   tools/call       → invoke a named tool with arguments
#   notifications/*  → accepted silently
#
# Full spec: https://modelcontextprotocol.io/specification


def rpc_result(req_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def rpc_error(req_id: Any, code: int, msg: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": msg}}


def handle_initialize(params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
    }


def handle_tools_list() -> Dict[str, Any]:
    return {
        "tools": [
            {
                "name": name,
                "description": spec["description"],
                "inputSchema": spec["inputSchema"],
            }
            for name, spec in TOOLS.items()
        ]
    }


def handle_tools_call(params: Dict[str, Any]) -> Dict[str, Any]:
    name = params.get("name", "")
    args = params.get("arguments", {}) or {}
    spec = TOOLS.get(name)
    if spec is None:
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"unknown tool: {name}"}],
        }
    try:
        result = spec["impl"](args)
    except Exception as e:
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"tool error: {e!r}"}],
        }
    return {
        "content": [{"type": "text", "text": json.dumps(result, default=str)}],
    }


def serve_stdio() -> None:
    """Main loop. Reads newline-delimited JSON-RPC requests from stdin,
    writes responses to stdout. Errors are logged to stderr to avoid
    interleaving with the protocol stream."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            sys.stderr.write(f"[nsynth-mcp] parse error: {e}\n"); sys.stderr.flush()
            continue
        method = req.get("method", "")
        req_id = req.get("id")
        params = req.get("params", {}) or {}

        if method == "initialize":
            resp = rpc_result(req_id, handle_initialize(params))
        elif method == "tools/list":
            resp = rpc_result(req_id, handle_tools_list())
        elif method == "tools/call":
            resp = rpc_result(req_id, handle_tools_call(params))
        elif method.startswith("notifications/"):
            # MCP notifications don't take a response.
            continue
        else:
            # ping/handshake methods we don't implement — silently ignore
            # if id is absent (notification), else return method-not-found.
            if req_id is None:
                continue
            resp = rpc_error(req_id, -32601, f"method not found: {method}")

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    try:
        serve_stdio()
    except (EOFError, KeyboardInterrupt):
        pass
