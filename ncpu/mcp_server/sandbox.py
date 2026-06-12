"""Subprocess sandbox: verify and run client-drafted candidate programs.

This is the out-of-domain half of the cascade (Rung 7.5). The synthesizer
only covers its verified search space; everything else is refused. The MCP
*client* (Claude, Cursor, ...) is itself a capable code generator, so
out-of-domain handling is: let the client draft the code, then admit it
through the **same example-verification gate** the synthesizer uses —
``verify_candidate`` executes the draft against every example and only
reports ``verified: true`` when all of them reproduce. ``run_program``
then executes a verified program on new inputs, closing the loop from
natural language to actual program output.

Trust model — stated plainly
----------------------------

The candidate code is **executed**. There is deliberately no pattern
matching / blocklisting of "dangerous" code: naive blocklists create
false confidence and are trivially bypassed. The control is the sandbox:

- fresh process per call: ``python3 -I`` (isolated mode: no user site,
  no ``PYTHON*`` env vars, script dir not on ``sys.path``), or ``node``
  for JavaScript;
- scrubbed environment: minimal ``PATH``, no ``HOME``, nothing inherited;
- cwd is a fresh temporary directory, deleted afterwards;
- hard wall-clock timeout; on expiry the whole **process tree** is
  killed (``start_new_session`` + ``killpg``).

This runs with the same trust as any local coding agent executing code
the client just wrote — no more, no less. Do not point it at code you
would not let your coding agent run.

Harness design
--------------

The candidate source is written to a temp file and a harness is appended
*in the same file*. The Python harness imports nothing: the examples are
embedded as Python literals, the function is looked up via
``globals().get(name)`` (no identifier injection), float comparison
re-implements ``math.isclose(rel_tol=1e-6)`` inline, and the result is
written as a ``repr()`` literal to a result file in the sandbox cwd (so
candidate prints to stdout cannot corrupt it). The parent parses it back
with ``ast.literal_eval``. The JavaScript harness mirrors this with
embedded JSON and ``fs.writeFileSync``.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

DEFAULT_SANDBOX_TIMEOUT_S = 10.0
MAX_SANDBOX_TIMEOUT_S = 60.0

_RESULT_FILE = "_ncpu_result.out"
_STDERR_TAIL = 1000

_JS_IDENTIFIER_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_examples(examples: Any) -> Optional[str]:
    """Return an error string for malformed examples, or None if OK."""
    if not isinstance(examples, list) or not examples:
        return "examples must be a non-empty list"
    for i, ex in enumerate(examples):
        if not isinstance(ex, dict) or "inputs" not in ex or "expected" not in ex:
            return f"examples[{i}] must be an object with 'inputs' and 'expected'"
        if not isinstance(ex["inputs"], list):
            return f"examples[{i}].inputs must be a list (positional arguments)"
    try:
        json.dumps(examples, allow_nan=False)
    except (TypeError, ValueError) as exc:
        return f"examples must be JSON-serializable (no NaN/Infinity): {exc}"
    return None


def _check_basics(name: Any, code: Any) -> Optional[str]:
    if not isinstance(name, str) or not name.strip():
        return "name must be a non-empty string"
    if not isinstance(code, str) or not code.strip():
        return "code must be a non-empty string"
    return None


def _clamp_timeout(timeout_s: Any) -> float:
    try:
        value = float(timeout_s)
    except (TypeError, ValueError):
        return DEFAULT_SANDBOX_TIMEOUT_S
    if value <= 0:
        return DEFAULT_SANDBOX_TIMEOUT_S
    return min(value, MAX_SANDBOX_TIMEOUT_S)


def _unsupported_language(language: Any) -> str:
    return (
        f"unsupported language {language!r}: the sandbox executes 'python' "
        "(and 'javascript' when node is installed)"
    )


# ---------------------------------------------------------------------------
# Sandboxed process execution
# ---------------------------------------------------------------------------


def _scrubbed_env() -> dict[str, str]:
    """Minimal environment: PATH only, no HOME, nothing inherited."""
    return {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}


def _run_sandboxed(
    argv: list[str], cwd: Path, timeout_s: float
) -> tuple[Optional[int], str, bool]:
    """Run argv in its own session; return (returncode, stderr, timed_out).

    On timeout the entire process group is SIGKILLed so grandchildren
    spawned by the candidate die with it.
    """
    proc = subprocess.Popen(
        argv,
        cwd=str(cwd),
        env=_scrubbed_env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        _, stderr = proc.communicate(timeout=timeout_s)
        return proc.returncode, (stderr or "")[-_STDERR_TAIL:], False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.communicate()
        return None, "", True


def _read_result(path: Path, language: str) -> Optional[dict[str, Any]]:
    """Parse the harness result file (repr literal for Python, JSON for JS)."""
    try:
        text = path.read_text()
    except OSError:
        return None
    try:
        parsed = json.loads(text) if language == "javascript" else ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    return parsed if isinstance(parsed, dict) else None


# ---------------------------------------------------------------------------
# Harness builders — Python (imports nothing; literals in, repr out)
# ---------------------------------------------------------------------------

_PY_COMMON = """

# --- ncpu sandbox harness (appended; imports nothing) ---------------------
def _ncpu_norm(v):
    if isinstance(v, tuple):
        v = list(v)
    if isinstance(v, list):
        return [_ncpu_norm(x) for x in v]
    if isinstance(v, dict):
        return dict((str(k), _ncpu_norm(x)) for k, x in v.items())
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if v == v and v not in (float("inf"), float("-inf")) else repr(v)
    return repr(v)

def _ncpu_eq(got, want):
    if isinstance(got, tuple):
        got = list(got)
    if isinstance(got, bool) or isinstance(want, bool):
        return isinstance(got, bool) and isinstance(want, bool) and got == want
    if isinstance(got, (int, float)) and isinstance(want, (int, float)):
        if got == want:
            return True
        a, b = float(got), float(want)
        # math.isclose(a, b, rel_tol=1e-6, abs_tol=0.0), without the import
        return abs(a - b) <= 1e-6 * max(abs(a), abs(b))
    if isinstance(got, list) and isinstance(want, list):
        if len(got) != len(want):
            return False
        for g, w in zip(got, want):
            if not _ncpu_eq(g, w):
                return False
        return True
    if isinstance(got, dict) and isinstance(want, dict):
        if set(got) != set(want):
            return False
        for k in got:
            if not _ncpu_eq(got[k], want[k]):
                return False
        return True
    return got == want

def _ncpu_write(payload):
    f = open({result_file!r}, "w")
    f.write(repr(payload))
    f.close()

def _ncpu_resolve():
    fn = globals().get({name!r})
    if not callable(fn):
        _ncpu_write({{"fatal": "no callable named " + {name!r} + " defined by the code"}})
        return None
    return fn
"""

_PY_VERIFY = _PY_COMMON + """
def _ncpu_main():
    fn = _ncpu_resolve()
    if fn is None:
        return
    failures = []
    for i, (inputs, expected) in enumerate({examples!r}):
        try:
            got = fn(*inputs)
        except BaseException as exc:
            failures.append({{"example_index": i,
                              "error": type(exc).__name__ + ": " + str(exc)}})
            continue
        if not _ncpu_eq(got, expected):
            failures.append({{"example_index": i, "got": _ncpu_norm(got)}})
    _ncpu_write({{"failures": failures}})

_ncpu_main()
"""

_PY_RUN = _PY_COMMON + """
def _ncpu_main():
    fn = _ncpu_resolve()
    if fn is None:
        return
    results = []
    for inputs in {calls!r}:
        try:
            results.append({{"ok": True, "output": _ncpu_norm(fn(*inputs))}})
        except BaseException as exc:
            results.append({{"ok": False,
                             "error": type(exc).__name__ + ": " + str(exc)}})
    _ncpu_write({{"results": results}})

_ncpu_main()
"""


def _py_verify_source(code: str, name: str, pairs: list[tuple[Any, Any]]) -> str:
    harness = _PY_VERIFY.format(result_file=_RESULT_FILE, name=name, examples=pairs)
    return code + "\n" + harness


def _py_run_source(code: str, name: str, calls: list[list[Any]]) -> str:
    harness = _PY_RUN.format(result_file=_RESULT_FILE, name=name, calls=calls)
    return code + "\n" + harness


# ---------------------------------------------------------------------------
# Harness builders — JavaScript (node; embedded JSON in, JSON file out)
# ---------------------------------------------------------------------------

_JS_COMMON = """

// --- ncpu sandbox harness (appended) ---------------------------------------
;(function () {{
  const _fs = require("fs");
  function _write(payload) {{
    _fs.writeFileSync({result_file_json}, JSON.stringify(payload));
  }}
  function _norm(v) {{
    if (v === undefined) return null;
    try {{ return JSON.parse(JSON.stringify(v)); }}
    catch (e) {{ return String(v); }}
  }}
  function _eq(got, want) {{
    if (typeof got === "number" && typeof want === "number") {{
      if (got === want) return true;
      // math.isclose(rel_tol=1e-6, abs_tol=0)
      return Math.abs(got - want) <= 1e-6 * Math.max(Math.abs(got), Math.abs(want));
    }}
    if (Array.isArray(got) && Array.isArray(want)) {{
      if (got.length !== want.length) return false;
      return got.every((g, i) => _eq(g, want[i]));
    }}
    if (got && want && typeof got === "object" && typeof want === "object") {{
      const kg = Object.keys(got), kw = Object.keys(want);
      if (kg.length !== kw.length) return false;
      return kg.every((k) => k in want && _eq(got[k], want[k]));
    }}
    return got === want;
  }}
  if (typeof {name} !== "function") {{
    _write({{ fatal: "no callable named " + {name_json} + " defined by the code" }});
    return;
  }}
{body}
}})();
"""

_JS_VERIFY_BODY = """  const examples = {examples_json};
  const failures = [];
  for (let i = 0; i < examples.length; i++) {{
    const [inputs, expected] = examples[i];
    let got;
    try {{ got = {name}(...inputs); }}
    catch (e) {{ failures.push({{ example_index: i, error: String(e) }}); continue; }}
    if (!_eq(got, expected)) failures.push({{ example_index: i, got: _norm(got) }});
  }}
  _write({{ failures: failures }});"""

_JS_RUN_BODY = """  const calls = {calls_json};
  const results = [];
  for (const inputs of calls) {{
    try {{ results.push({{ ok: true, output: _norm({name}(...inputs)) }}); }}
    catch (e) {{ results.push({{ ok: false, error: String(e) }}); }}
  }}
  _write({{ results: results }});"""


def _js_source(code: str, name: str, body: str) -> str:
    harness = _JS_COMMON.format(
        result_file_json=json.dumps(_RESULT_FILE),
        name=name,
        name_json=json.dumps(name),
        body=body,
    )
    return code + "\n" + harness


# ---------------------------------------------------------------------------
# Language dispatch
# ---------------------------------------------------------------------------


def _build_argv_and_source(
    language: str, mode: str, code: str, name: str, payload: Any
) -> tuple[Optional[list[str]], Optional[str], Optional[str], Optional[str]]:
    """Return (argv_prefix, source, filename, error). Exactly one of
    (argv_prefix, error) is None."""
    if language == "python":
        if mode == "verify":
            source = _py_verify_source(code, name, payload)
        else:
            source = _py_run_source(code, name, payload)
        # sys.executable is the interpreter already running this server;
        # -I = isolated mode (no user site, PYTHON* env ignored).
        return [sys.executable, "-I"], source, "_candidate.py", None

    if language == "javascript":
        node = shutil.which("node")
        if node is None:
            return None, None, None, (
                "language 'javascript' unsupported on this host: "
                "node not found on PATH"
            )
        if not _JS_IDENTIFIER_RE.match(name):
            return None, None, None, (
                f"name {name!r} is not a valid JavaScript identifier"
            )
        if mode == "verify":
            body = _JS_VERIFY_BODY.format(
                examples_json=json.dumps(payload), name=name
            )
        else:
            body = _JS_RUN_BODY.format(calls_json=json.dumps(payload), name=name)
        return [node], _js_source(code, name, body), "_candidate.js", None

    return None, None, None, _unsupported_language(language)


def _execute(
    language: str, mode: str, code: str, name: str, payload: Any, timeout_s: float
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Run a harnessed candidate; return (result_dict, error). One is None."""
    argv_prefix, source, filename, error = _build_argv_and_source(
        language, mode, code, name, payload
    )
    if error is not None:
        return None, error
    assert argv_prefix is not None and source is not None and filename is not None

    with tempfile.TemporaryDirectory(prefix="ncpu_sandbox_") as tmp:
        cwd = Path(tmp)
        script = cwd / filename
        script.write_text(source)
        returncode, stderr, timed_out = _run_sandboxed(
            argv_prefix + [str(script)], cwd, timeout_s
        )
        if timed_out:
            return None, (
                f"timeout: candidate did not finish within {timeout_s:g}s "
                "(process tree killed)"
            )
        result = _read_result(cwd / _RESULT_FILE, language)

    if result is None:
        detail = stderr.strip() or "no output"
        return None, f"candidate process exited with code {returncode}: {detail}"
    if "fatal" in result:
        return None, str(result["fatal"])
    return result, None


# ---------------------------------------------------------------------------
# Tool 5: verify_candidate
# ---------------------------------------------------------------------------


def verify_candidate(
    name: str,
    code: str,
    examples: list[dict[str, Any]],
    language: str = "python",
    timeout_s: float = DEFAULT_SANDBOX_TIMEOUT_S,
) -> dict[str, Any]:
    """Execute a client-drafted candidate against EVERY example.

    Returns ``{"verified": True, "examples_checked": n}`` only when all
    examples reproduce; otherwise ``{"verified": False, ...}`` with the
    first counterexample (or the error) attached.
    """
    error = _check_basics(name, code) or _check_examples(examples)
    if error is not None:
        return {"verified": False, "error": error}
    timeout = _clamp_timeout(timeout_s)

    pairs = [(list(ex["inputs"]), ex["expected"]) for ex in examples]
    result, error = _execute(language, "verify", code, name, pairs, timeout)
    if error is not None:
        return {"verified": False, "error": error, "language": language}

    failures = result.get("failures") if result else None
    if not isinstance(failures, list):
        return {
            "verified": False,
            "error": "sandbox harness produced an unreadable result",
            "language": language,
        }
    if not failures:
        return {
            "verified": True,
            "examples_checked": len(examples),
            "language": language,
        }

    first = failures[0]
    idx = first.get("example_index", 0)
    idx = idx if isinstance(idx, int) and 0 <= idx < len(examples) else 0
    first_failure: dict[str, Any] = {
        "example_index": idx,
        "inputs": examples[idx]["inputs"],
        "expected": examples[idx]["expected"],
    }
    if "error" in first:
        first_failure["error"] = first["error"]
    else:
        first_failure["got"] = first.get("got")
    return {
        "verified": False,
        "first_failure": first_failure,
        "failures": len(failures),
        "examples_checked": len(examples),
        "language": language,
    }


# ---------------------------------------------------------------------------
# Tool 6: run_program
# ---------------------------------------------------------------------------


def run_program(
    name: str,
    code: str,
    inputs: list[Any],
    language: str = "python",
    timeout_s: float = DEFAULT_SANDBOX_TIMEOUT_S,
    batch: bool = False,
) -> dict[str, Any]:
    """Call ``name(*inputs)`` once in the sandbox and return the output.

    With ``batch=True``, ``inputs`` must be a list of call-argument lists;
    the result carries one ``{"ok", "output"|"error"}`` entry per call.
    """
    error = _check_basics(name, code)
    if error is not None:
        return {"ok": False, "error": error}
    if not isinstance(inputs, list):
        return {"ok": False, "error": "inputs must be a list"}
    if batch:
        if not inputs or not all(isinstance(c, list) for c in inputs):
            return {
                "ok": False,
                "error": "batch=true requires inputs to be a non-empty "
                "list of argument lists",
            }
        calls = inputs
    else:
        calls = [inputs]
    try:
        json.dumps(calls, allow_nan=False)
    except (TypeError, ValueError) as exc:
        return {"ok": False, "error": f"inputs must be JSON-serializable: {exc}"}
    timeout = _clamp_timeout(timeout_s)

    result, error = _execute(language, "run", code, name, calls, timeout)
    if error is not None:
        return {"ok": False, "error": error, "language": language}

    results = result.get("results") if result else None
    if not isinstance(results, list) or len(results) != len(calls):
        return {
            "ok": False,
            "error": "sandbox harness produced an unreadable result",
            "language": language,
        }

    if batch:
        return {"ok": True, "outputs": results, "language": language}
    single = results[0]
    if single.get("ok"):
        return {"ok": True, "output": single.get("output"), "language": language}
    return {"ok": False, "error": single.get("error", "unknown error"), "language": language}


__all__ = [
    "verify_candidate",
    "run_program",
    "DEFAULT_SANDBOX_TIMEOUT_S",
    "MAX_SANDBOX_TIMEOUT_S",
]
