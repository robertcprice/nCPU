"""Synthesis-as-a-Service HTTP server (Rung 3).

A stdlib-only HTTP server that puts the nsynth Rust synthesizer behind
HTTP. Each `POST /synthesize` request carries a named problem plus
input/output examples; the server shells out to the release binary
(`nsynth/target/release/mog_synth --problem-json -`), returns the
synthesized + verified Mog program, and transpiles it to Python, Rust,
and TypeScript via the binary's `--transpile` flag.

Why stdlib-only: no `fastapi`, no `flask`, no `uvicorn`. The server should
work on *any* Python 3.8+ install without pulling dependencies. The
process footprint is tiny — all heavy lifting happens in the Rust
subprocess, which carries its own persistent memory banks (solved
programs, learned biases, rejected programs) so repeat requests for the
same examples are near-instant.

Refusals are passed through honestly: when the synthesizer cannot find a
program that reproduces the examples, the response is `success: false`
with the backend's error — the server never fabricates code. This is the
middle tier of the cascade: browser WASM tier refuses → this endpoint →
LLM tier.

The handler is also available as a standalone function
`handle_synthesize_request(request_json, config)` so you can embed it in
your own server framework without running ours.

Usage:

    python3 -m ncpu.synthesis_api.server --port 8093

    # In another shell:
    curl -X POST http://localhost:8093/synthesize \\
         -H 'Content-Type: application/json' \\
         -d '{"name": "add_two", "examples": [
               {"inputs": [1, 2], "expected": 3},
               {"inputs": [5, 7], "expected": 12},
               {"inputs": [0, 0], "expected": 0}]}'
    # → {"success": true, "method": "search_scalar_expr", "code": "fn ...",
    #    "transpiled": {"python": "def ...", ...}, "elapsed_ms": 24.1}

Endpoints:
    GET  /health     → 200 + {"status", "backend", "backend_present"}
    GET  /stats      → 200 + memory-bank sizes (solved/bias/rejected)
    POST /synthesize → 200 + result (refusal included) or 400 on bad input

Cache isolation (e.g. for tests): pass `--solved-cache`, `--bias-bank`,
and `--rejected-cache` to point the backend's persistent banks at private
paths (they map to the NSYNTH_CACHE_PATH / NSYNTH_BIAS_BANK_PATH /
NSYNTH_REJECTED_PATH environment variables of the subprocess).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional


log = logging.getLogger("synthesis_api")

# Hard ceiling on per-request solver time. Requests may lower the limit
# via `timeout_s` but can never raise it past this (a deployment may set
# a tighter ceiling via SynthConfig.max_timeout_s / --max-timeout).
MAX_TIMEOUT_S = 300.0
DEFAULT_TIMEOUT_S = 120.0

# Largest accepted /synthesize body. Example sets are tiny; anything
# bigger is abuse or a bug.
MAX_BODY_BYTES = 64 * 1024

# i64 bounds — the Rust side silently drops examples whose numbers don't
# fit in an i64, which would corrupt the problem. Reject them up front.
_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1

_TRANSPILE_TARGETS = ("python", "rust", "typescript", "go")


def default_backend_path() -> Path:
    """Repo-relative release binary: <repo>/nsynth/target/release/mog_synth."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "nsynth" / "target" / "release" / "mog_synth"


@dataclass
class SynthConfig:
    """Server configuration shared by the HTTP handler and the embeddable
    `handle_synthesize_request` function."""

    backend: Path = field(default_factory=default_backend_path)
    timeout_s: float = DEFAULT_TIMEOUT_S
    # Deployment-level ceiling on request-supplied `timeout_s`. Public
    # deployments set this well below MAX_TIMEOUT_S so a client cannot
    # pin a CPU for five minutes per request.
    max_timeout_s: float = MAX_TIMEOUT_S
    # `Access-Control-Allow-Origin` value; empty string disables CORS
    # headers entirely (the default for local/embedded use).
    cors_origin: str = ""
    # Maximum concurrent backend solves; further requests get 429 rather
    # than queueing unboundedly.
    max_concurrency: int = 2
    # Optional overrides for the backend's three persistent memory banks.
    # `None` → inherit the calling environment (which itself defaults to
    # `$HOME/.nsynth_*` inside the Rust binary).
    solved_cache: Optional[str] = None
    bias_bank: Optional[str] = None
    rejected_cache: Optional[str] = None

    def subprocess_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.solved_cache is not None:
            env["NSYNTH_CACHE_PATH"] = self.solved_cache
        if self.bias_bank is not None:
            env["NSYNTH_BIAS_BANK_PATH"] = self.bias_bank
        if self.rejected_cache is not None:
            env["NSYNTH_REJECTED_PATH"] = self.rejected_cache
        return env

    def bank_path(self, kind: str) -> Optional[Path]:
        """Resolve a memory-bank path the same way the Rust binary does:
        explicit server flag > environment variable > $HOME default.
        Empty string means "disabled" (matches the Rust semantics)."""
        flag, env_var, home_name = {
            "solved": (self.solved_cache, "NSYNTH_CACHE_PATH", ".nsynth_solved_programs.json"),
            "bias": (self.bias_bank, "NSYNTH_BIAS_BANK_PATH", ".nsynth_learned_biases.jsonl"),
            "rejected": (self.rejected_cache, "NSYNTH_REJECTED_PATH", ".nsynth_rejected_programs.tsv"),
        }[kind]
        value = flag if flag is not None else os.environ.get(env_var)
        if value is not None:
            return Path(value) if value else None
        return Path(os.environ.get("HOME", ".")) / home_name


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def _sanitize_fn_name(name: str) -> str:
    """Reduce a request name to a valid Mog identifier for the signature."""
    cleaned = "".join(c if (c.isalnum() or c == "_") else "_" for c in name.strip())
    if not cleaned or not (cleaned[0].isalpha() or cleaned[0] == "_"):
        cleaned = f"f_{cleaned}" if cleaned else "synthesized"
    return cleaned


def _build_signature(name: str, first_inputs: list) -> str:
    """Build a Mog signature from the request name and the first example's
    input types, so generated code carries the caller's chosen name instead
    of the backend default `fn unknown(...)`. Type spellings match
    nsynth/src/benchmark.rs: `i64`, `[i64]`, `string`."""
    params: list[str] = []
    n_scalar = n_array = n_str = 0
    for value in first_inputs:
        if isinstance(value, list):
            n_array += 1
            pname = "arr" if n_array == 1 else f"arr{n_array}"
            params.append(f"{pname}: [i64]")
        elif isinstance(value, str):
            n_str += 1
            pname = "s" if n_str == 1 else f"s{n_str}"
            params.append(f"{pname}: string")
        else:
            pname = chr(ord("a") + min(n_scalar, 25))
            n_scalar += 1
            params.append(f"{pname}: i64")
    return f"fn {_sanitize_fn_name(name)}({', '.join(params)}) -> i64"


def _validate_value(value: Any, where: str) -> None:
    """An example input is an i64, an array of i64, or a string."""
    if isinstance(value, bool):
        raise ValueError(f"{where}: booleans are not valid inputs")
    if isinstance(value, int):
        if not (_I64_MIN <= value <= _I64_MAX):
            raise ValueError(f"{where}: integer out of i64 range")
        return
    if isinstance(value, str):
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"{where}[{i}]: array elements must be integers")
            if not (_I64_MIN <= item <= _I64_MAX):
                raise ValueError(f"{where}[{i}]: integer out of i64 range")
        return
    raise ValueError(
        f"{where}: unsupported input type {type(value).__name__} "
        "(expected int, [int], or str)"
    )


def _validate_examples(examples: Any, where: str, allow_empty: bool) -> None:
    if not isinstance(examples, list):
        raise ValueError(f"{where} must be a list")
    if not examples and not allow_empty:
        raise ValueError(f"{where} must be a non-empty list")
    for i, ex in enumerate(examples):
        if not isinstance(ex, dict):
            raise ValueError(f"{where}[{i}] must be an object")
        if "inputs" not in ex:
            raise ValueError(f"{where}[{i}]: missing required field: inputs")
        if "expected" not in ex:
            raise ValueError(f"{where}[{i}]: missing required field: expected")
        inputs = ex["inputs"]
        if not isinstance(inputs, list) or not inputs:
            raise ValueError(f"{where}[{i}].inputs must be a non-empty list")
        for j, value in enumerate(inputs):
            _validate_value(value, f"{where}[{i}].inputs[{j}]")
        expected = ex["expected"]
        if isinstance(expected, bool):
            # Booleans are first-class on the wire (real code returns
            # true/false), but the Mog solver currently treats the
            # 0/1 int lane as the canonical predicate form. Coerce on
            # the way through so a `{expected: true}` request lands
            # the same problem-json the existing i64 search teachers
            # already solve against.
            ex["expected"] = int(expected)
            expected = ex["expected"]
        if not isinstance(expected, int) or isinstance(expected, bool):
            raise ValueError(f"{where}[{i}].expected must be an integer or bool")
        if not (_I64_MIN <= expected <= _I64_MAX):
            raise ValueError(f"{where}[{i}].expected out of i64 range")


def validate_synthesize_request(request: Any) -> tuple[dict[str, Any], Optional[float]]:
    """Validate a /synthesize body. Returns `(problem_json, timeout_s)`.

    Raises ValueError with a human-readable message on any malformed
    input — the HTTP layer maps that to a 400, never a 500.
    """
    if not isinstance(request, dict):
        raise ValueError("request body must be a JSON object")
    name = request.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("missing required field: name (non-empty string)")
    if "examples" not in request:
        raise ValueError("missing required field: examples")
    _validate_examples(request["examples"], "examples", allow_empty=False)

    problem: dict[str, Any] = {
        "name": name.strip(),
        "examples": request["examples"],
    }
    signature = request.get("signature")
    if signature is not None:
        if not isinstance(signature, str):
            raise ValueError("signature must be a string when provided")
        problem["signature"] = signature
    else:
        # Default backend signature is `fn unknown(...)` — build one from
        # the request name + first example so generated code is named.
        problem["signature"] = _build_signature(
            problem["name"], request["examples"][0]["inputs"]
        )
    holdouts = request.get("holdouts")
    if holdouts is not None:
        _validate_examples(holdouts, "holdouts", allow_empty=True)
        problem["holdouts"] = holdouts

    timeout_s: Optional[float] = None
    if "timeout_s" in request and request["timeout_s"] is not None:
        raw = request["timeout_s"]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError("timeout_s must be a number")
        if raw <= 0:
            raise ValueError("timeout_s must be positive")
        timeout_s = min(float(raw), MAX_TIMEOUT_S)

    return problem, timeout_s


# ---------------------------------------------------------------------------
# Backend invocation
# ---------------------------------------------------------------------------


def _run_backend(
    config: SynthConfig, args: list[str], stdin_text: str, timeout_s: float
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(config.backend), *args],
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=config.subprocess_env(),
    )


def _transpile(config: SynthConfig, code: str) -> dict[str, Optional[str]]:
    """Transpile Mog source to all targets. A failed target maps to None —
    the verified Mog `code` is the source of truth either way."""
    out: dict[str, Optional[str]] = {}
    for target in _TRANSPILE_TARGETS:
        try:
            proc = _run_backend(config, ["--transpile", target], code, timeout_s=15.0)
            out[target] = proc.stdout.strip() if proc.returncode == 0 else None
        except (subprocess.TimeoutExpired, OSError) as exc:
            log.warning("transpile to %s failed: %s", target, exc)
            out[target] = None
    return out


def handle_synthesize_request(
    request: Any, config: SynthConfig
) -> tuple[int, dict[str, Any]]:
    """Translate a /synthesize request into a backend solve + transpile.

    Returns `(status_code, response_body)`. A refusal (the synthesizer
    honestly reporting it found no program) is a *valid answer*, returned
    as 200 with `success: false` — never fabricated code. Malformed
    requests return 400.
    """
    try:
        problem, timeout_override = validate_synthesize_request(request)
    except ValueError as exc:
        return 400, {"error": str(exc)}

    if not config.backend.is_file():
        return 503, {"error": f"backend binary not found: {config.backend}"}

    timeout_s = timeout_override if timeout_override is not None else config.timeout_s
    timeout_s = min(timeout_s, MAX_TIMEOUT_S, config.max_timeout_s)

    start = time.perf_counter()

    def _elapsed_ms() -> float:
        return round((time.perf_counter() - start) * 1000.0, 3)

    try:
        proc = _run_backend(
            config, ["--problem-json", "-"], json.dumps(problem), timeout_s
        )
    except subprocess.TimeoutExpired:
        return 200, {
            "success": False,
            "method": "timeout",
            "code": None,
            "error": "timeout",
            "transpiled": None,
            "elapsed_ms": _elapsed_ms(),
        }
    except OSError as exc:
        return 503, {"error": f"failed to launch backend: {exc}"}

    # The binary prints exactly one JSON object on stdout (logs go to
    # stderr). Parse the last non-empty stdout line, defensively.
    result: Optional[dict[str, Any]] = None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            result = parsed
            break
    if result is None:
        log.error(
            "backend produced no parseable result (rc=%d, stderr tail: %s)",
            proc.returncode,
            proc.stderr[-500:],
        )
        return 200, {
            "success": False,
            "method": "backend_error",
            "code": None,
            "error": f"backend produced no parseable result (exit code {proc.returncode})",
            "transpiled": None,
            "elapsed_ms": _elapsed_ms(),
        }

    success = bool(result.get("success"))
    code = result.get("code") if isinstance(result.get("code"), str) else None
    transpiled = _transpile(config, code) if (success and code) else None

    return 200, {
        "success": success,
        "method": str(result.get("method", "unknown")),
        "code": code,
        "error": result.get("error"),
        "transpiled": transpiled,
        "holdouts": result.get("holdouts"),
        "elapsed_ms": _elapsed_ms(),
    }


# ---------------------------------------------------------------------------
# Memory-bank stats
# ---------------------------------------------------------------------------


def read_bank_stats(config: SynthConfig) -> dict[str, int]:
    """Sizes of the backend's three persistent memory banks.

    Parses the on-disk formats directly (missing or disabled → 0):
    - solved cache: one record per line (`fp \\t method \\t ... \\t code`)
    - bias bank: JSONL, one bias per line
    - rejected cache: TSV `last_used \\t hash,hash,... \\t fingerprint`
    """

    def _lines(path: Optional[Path]) -> list[str]:
        if path is None or not path.is_file():
            return []
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return [line for line in text.splitlines() if line.strip()]

    solved_entries = len(_lines(config.bank_path("solved")))
    bias_entries = len(_lines(config.bank_path("bias")))

    rejected_rows = 0
    rejected_hashes = 0
    for line in _lines(config.bank_path("rejected")):
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        rejected_rows += 1
        rejected_hashes += sum(1 for h in parts[1].split(",") if h.strip())

    return {
        "solved_entries": solved_entries,
        "bias_entries": bias_entries,
        "rejected_rows": rejected_rows,
        "rejected_hashes": rejected_hashes,
    }


# ---------------------------------------------------------------------------
# /prompt — natural-language front door
# ---------------------------------------------------------------------------

# Bounds for the prompt path: prompts are short docstrings/specs, and the
# verification harness asserts one line per pair — cap both.
MAX_PROMPT_CHARS = 16_384
MAX_PROMPT_PAIRS = 64


def handle_prompt_request(
    request: Any, config: SynthConfig
) -> tuple[int, dict[str, Any]]:
    """Handle one /prompt body: free-form text → parsed examples → cascade.

    The deterministic prompt parser (no LLM) extracts I/O pairs from
    asserts, doctests, arrow notation, and "returns" prose. The cascade
    then tries ``template_match`` (fixed Python template library) and
    ``nsynth_fast`` (the Rust synthesizer via this module's own handler).
    Every candidate is verified against an assert harness built from the
    parsed pairs before it is returned. No examples → honest refusal,
    never fabricated code.
    """
    # Lazy import: the autoresearch package is stdlib-only on this path,
    # but keeping it lazy means /synthesize works even if it's absent.
    try:
        from ncpu.autoresearch.cascade import CascadeConfig, run_cascade
        from ncpu.autoresearch.prompt_parser import (
            build_work_item,
            extract_from_prompt,
        )
    except ImportError as exc:  # pragma: no cover — deploy misconfiguration
        return 503, {"error": f"prompt pipeline unavailable: {exc}"}

    if not isinstance(request, dict):
        return 400, {"error": "request body must be a JSON object"}
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return 400, {"error": "missing required field: prompt (non-empty string)"}
    if len(prompt) > MAX_PROMPT_CHARS:
        return 400, {"error": f"prompt too long (max {MAX_PROMPT_CHARS} chars)"}
    entry_point = request.get("entry_point")
    if entry_point is not None and not isinstance(entry_point, str):
        return 400, {"error": "entry_point must be a string when provided"}

    timeout_s = config.timeout_s
    raw_timeout = request.get("timeout_s")
    if raw_timeout is not None:
        if isinstance(raw_timeout, bool) or not isinstance(raw_timeout, (int, float)):
            return 400, {"error": "timeout_s must be a number"}
        if raw_timeout <= 0:
            return 400, {"error": "timeout_s must be positive"}
        timeout_s = float(raw_timeout)
    timeout_s = min(timeout_s, MAX_TIMEOUT_S, config.max_timeout_s)

    start = time.perf_counter()

    report = extract_from_prompt(prompt, entry_point=entry_point)
    base = {
        "entry_point": report.entry_point,
        "io_pairs": len(report.io_pairs),
        "pair_sources": report.sources,
    }
    if report.entry_point is None:
        return 200, {
            **base,
            "success": False,
            "method": None,
            "code": None,
            "error": (
                "no entry point found — include a `def name(...):` stub "
                "or name the function in your examples"
            ),
            "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 3),
        }
    if not report.io_pairs:
        return 200, {
            **base,
            "success": False,
            "method": None,
            "code": None,
            "error": (
                "no examples found — include asserts, doctests, arrow "
                "notation (f(x) -> y), or 'f(x) returns y' prose"
            ),
            "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 3),
        }
    if len(report.io_pairs) > MAX_PROMPT_PAIRS:
        return 400, {"error": f"too many examples (max {MAX_PROMPT_PAIRS})"}

    item = build_work_item(prompt, entry_point=entry_point)
    if item is None:  # pragma: no cover — report.entry_point was non-None
        return 200, {
            **base,
            "success": False,
            "method": None,
            "code": None,
            "error": "could not build a work item from this prompt",
            "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 3),
        }

    cascade_cfg = CascadeConfig(
        solver_names=["template_match", "nsynth_fast"],
        per_solver_seconds=timeout_s,
    )
    result = run_cascade(item, config=cascade_cfg)
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)

    if result.solved and result.solved_item is not None:
        solved = result.solved_item
        code = solved.program_python
        # template_match returns a pre-indented bare body — prepend the
        # runtime prompt's def stub so the client always gets a runnable
        # function (this mirrors how the verifier composed the module).
        if "def " not in code:
            code = item.prompt + code
        return 200, {
            **base,
            "success": True,
            "method": result.solver,
            "code": code,
            "error": None,
            "elapsed_ms": elapsed_ms,
        }
    return 200, {
        **base,
        "success": False,
        "method": None,
        "code": None,
        "error": result.error or "no solver produced a verified program",
        "elapsed_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


class SynthesisRequestHandler(BaseHTTPRequestHandler):
    config: SynthConfig  # set by start_server
    solve_slots: threading.Semaphore  # set by start_server

    def log_message(self, format, *args):
        # Route BaseHTTPRequestHandler logs through the module logger.
        log.info("%s - %s", self.address_string(), format % args)

    def _cors_headers(self) -> None:
        if self.config.cors_origin:
            self.send_header("Access-Control-Allow-Origin", self.config.cors_origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "86400")

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "backend": str(self.config.backend),
                    "backend_present": self.config.backend.is_file(),
                },
            )
        elif self.path == "/stats":
            self._send_json(200, read_bank_stats(self.config))
        else:
            self._send_json(404, {"error": f"unknown path: {self.path}"})

    def do_POST(self) -> None:
        if self.path not in ("/synthesize", "/prompt"):
            self._send_json(404, {"error": f"unknown path: {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            self._send_json(400, {"error": "bad Content-Length header"})
            return
        if length > MAX_BODY_BYTES:
            self._send_json(
                413, {"error": f"body too large (max {MAX_BODY_BYTES} bytes)"}
            )
            return
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            request = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"bad json: {exc}"})
            return
        if not self.solve_slots.acquire(blocking=False):
            self._send_json(
                429, {"error": "server at synthesis capacity, retry shortly"}
            )
            return
        handler = (
            handle_prompt_request
            if self.path == "/prompt"
            else handle_synthesize_request
        )
        try:
            status, payload = handler(request, self.config)
        except Exception as exc:  # pragma: no cover — defensive last resort
            log.exception("unexpected error handling %s", self.path)
            status, payload = 500, {"error": f"internal error: {exc}"}
        finally:
            self.solve_slots.release()
        self._send_json(status, payload)


def start_server(
    config: SynthConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 8093,
) -> ThreadingHTTPServer:
    """Create the HTTP server — call `.serve_forever()` to block.

    Threaded so /health stays responsive while solves run; actual solver
    concurrency is bounded by `config.max_concurrency` (excess → 429)."""
    SynthesisRequestHandler.config = config
    SynthesisRequestHandler.solve_slots = threading.Semaphore(
        max(1, config.max_concurrency)
    )
    server = ThreadingHTTPServer((host, port), SynthesisRequestHandler)
    log.info("synthesis API listening on %s:%d", host, port)
    log.info(
        "  backend: %s (present=%s)", config.backend, config.backend.is_file()
    )
    return server


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--backend",
        type=Path,
        default=default_backend_path(),
        help="path to the mog_synth release binary",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"default per-request solver timeout in seconds (max {MAX_TIMEOUT_S:.0f})",
    )
    parser.add_argument(
        "--max-timeout",
        type=float,
        default=MAX_TIMEOUT_S,
        help="ceiling for request-supplied timeout_s (public deployments set this low)",
    )
    parser.add_argument(
        "--cors-origin",
        default="",
        help="Access-Control-Allow-Origin value; empty disables CORS headers",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=2,
        help="max concurrent backend solves; excess requests get 429",
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

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = SynthConfig(
        backend=args.backend,
        timeout_s=min(args.timeout, MAX_TIMEOUT_S),
        max_timeout_s=min(args.max_timeout, MAX_TIMEOUT_S),
        cors_origin=args.cors_origin,
        max_concurrency=args.max_concurrency,
        solved_cache=args.solved_cache,
        bias_bank=args.bias_bank,
        rejected_cache=args.rejected_cache,
    )
    server = start_server(config, host=args.host, port=args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
        server.shutdown()
    return 0


__all__ = [
    "SynthConfig",
    "handle_synthesize_request",
    "validate_synthesize_request",
    "read_bank_stats",
    "start_server",
    "SynthesisRequestHandler",
]


if __name__ == "__main__":
    raise SystemExit(main())
