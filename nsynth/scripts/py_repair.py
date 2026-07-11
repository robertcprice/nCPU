#!/usr/bin/env python3
"""Never-wrong Python repair: VibeThinker proposes a corrected file, pytest is the oracle.

The engine's synthesis/mutation lanes are Rust-only, so Python repo repair (the SWE-bench
direction) is a MODEL lane gated by a real test runner. Same guarantee as the Rust path: apply the
model's fix to a work copy, run the tests, keep it ONLY if they pass, else feed the failure back and
retry; on exhaustion revert (never ship unverified). Model output is untrusted; the pytest oracle is
the trust boundary.

Usage:
    python3 py_repair.py <repo_dir> <target_file> "<test_command>" [max_iters]

Env: NSYNTH_LOCAL_LLM_URL (chat/completions), NSYNTH_LOCAL_LLM_MODEL. Inert (no-op fail) if unset.
"""
import glob
import json
import os
import re
import subprocess
import sys
import urllib.request

# Builtins / assert helpers that a failing-assert line names but which are never the buggy target.
_NON_TARGETS = {
    "assert", "print", "range", "len", "int", "str", "float", "bool", "list", "dict", "set",
    "tuple", "sorted", "sum", "min", "max", "abs", "self", "super", "isinstance", "enumerate",
    "zip", "map", "filter", "round", "any", "all", "repr", "format", "open", "type",
}

URL = os.environ.get("NSYNTH_LOCAL_LLM_URL", "")
MODEL = os.environ.get("NSYNTH_LOCAL_LLM_MODEL", "mlx-community/VibeThinker-3B-4bit")


def chat(prompt, max_tokens=3500):
    """One completion. VibeThinker is a reasoning model — it burns ~1-2k tokens reasoning before it
    emits the final `content`, so max_tokens must leave headroom or `content` comes back empty."""
    if not URL:
        return ""
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.load(r)
        msg = d["choices"][0]["message"]
        # A reasoning model that runs out of tokens leaves `content` empty with the code stranded in
        # `reasoning` — fall back to it so a fenced block there is still recoverable.
        return msg.get("content") or msg.get("reasoning") or ""
    except Exception as e:  # noqa: BLE001 - a dead model must not crash the harness
        print(f"  [model error: {e}]", file=sys.stderr)
        return ""


def extract_code(text):
    """The fenced ```python block, else the raw text."""
    m = re.search(r"```(?:python|py)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def run_tests(repo, testcmd):
    try:
        p = subprocess.run(
            testcmd, cwd=repo, shell=True, capture_output=True, text=True, timeout=180
        )
        return p.returncode == 0, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"


def localize_target(repo, out):
    """The buggy file to repair, inferred from the failure traceback: the DEEPEST (last) non-test
    `.py` frame that lives inside the repo. This is what makes it repo-level (SWE-bench shape) — no
    need to name the file; the failing test points at the code it exercised."""
    # Match BOTH the Python native traceback (`File "path.py", line N, in func`) and pytest's compact
    # frame (`path.py:N: in func`), capturing the enclosing function so the repair can be scoped to it.
    frames = re.findall(r'File "([^"]+\.py)", line \d+, in ([\w<>]+)', out)
    frames += re.findall(r'^\s*([^\s:]+\.py):\d+: in ([\w<>]+)', out, re.M)
    cand = None
    for f, fn in frames:
        base = os.path.basename(f)
        if base.startswith("test_") or base.endswith("_test.py") or base == "conftest.py":
            continue
        rel = os.path.relpath(f, repo) if os.path.isabs(f) else f
        if os.path.exists(os.path.join(repo, rel)) and not rel.startswith(".."):
            hint = fn if (fn and fn != "<module>" and not fn.startswith("test")) else None
            cand = (rel, hint)  # last frame wins (deepest in the call stack)
    return cand


def localize_by_assert(repo, out):
    """Fallback for WRONG-VALUE bugs: an assert failure names only the test file in the traceback,
    but the failing assert's SOURCE (which pytest prints) calls the function under test. Extract the
    free-function calls, then find the NON-test repo module that DEFINES one of them. The Python
    analog of the Rust named-fn localization — handles the common non-crashing bug the traceback
    alone can't place."""
    names = []
    # `NAME(` not preceded by a word char or `.` -> a free call, not a method/attribute access.
    for m in re.finditer(r"(?<![\w.])([A-Za-z_]\w*)\s*\(", out):
        n = m.group(1)
        if n in _NON_TARGETS or n.startswith("assert"):
            continue
        if n not in names:
            names.append(n)
    py_files = [
        p
        for p in glob.glob(os.path.join(repo, "**", "*.py"), recursive=True)
        if not (
            os.path.basename(p).startswith("test_")
            or os.path.basename(p).endswith("_test.py")
            or os.path.basename(p) == "conftest.py"
        )
    ]
    for name in names:
        pat = re.compile(rf"^\s*def {re.escape(name)}\s*\(", re.M)
        for path in py_files:
            try:
                if pat.search(open(path, encoding="utf-8", errors="ignore").read()):
                    return os.path.relpath(path, repo), name
            except OSError:
                continue
    return None


def extract_function(src, name):
    """`(start_line, end_line, text)` of the top-level (or method) `def name(...)` block by Python
    indentation, or None. Repairing only this span keeps the model's output small (a 3B reasoning
    model blows its token budget rewriting a whole file) and leaves sibling functions untouched."""
    lines = src.split("\n")
    for i, ln in enumerate(lines):
        m = re.match(r"^(\s*)def\s+" + re.escape(name) + r"\s*\(", ln)
        if not m:
            continue
        indent = len(m.group(1))
        j = i + 1
        while j < len(lines):
            s = lines[j].strip()
            if s == "" or s.startswith("#"):
                j += 1
                continue
            if len(lines[j]) - len(lines[j].lstrip()) <= indent:
                break
            j += 1
        while j > i + 1 and lines[j - 1].strip() == "":
            j -= 1
        return i, j, "\n".join(lines[i:j])
    return None


def repair(repo, target, testcmd, iters=3):
    """Returns (success, iterations_used, note). Reverts the file on failure. `target` may be None or
    "auto" to localize the buggy file from the baseline failure traceback."""
    ok, out = run_tests(repo, testcmd)
    if ok:
        return True, 0, "baseline already green"
    fn_hint = None
    if not target or target == "auto":
        # crash bugs: the traceback names the code file + its frame's function; wrong-value bugs:
        # fall back to the function named in the failing assert and the module that defines it.
        loc = localize_target(repo, out) or localize_by_assert(repo, out)
        if not loc:
            return False, 0, "could not localize a non-test file from the failure"
        target, fn_hint = loc
        print(f"  [localized target: {target} fn={fn_hint}]", file=sys.stderr)
    path = os.path.join(repo, target)
    orig = open(path).read()
    # FUNCTION-SCOPED repair when we know the buggy function and can extract it: fix only that span,
    # splice it back (siblings preserved, small prompt/output). Else whole-file.
    span = extract_function(orig, fn_hint) if fn_hint else None
    prior = span[2] if span else orig
    for i in range(1, iters + 1):
        if span:
            prompt = (
                f"A Python function `{fn_hint}` has a bug: a test that calls it fails. Return ONLY the "
                "corrected function (same name and signature) in a single ```python fenced block, "
                f"nothing else.\n\n=== FUNCTION ===\n{prior}\n\n=== FAILING TEST OUTPUT ===\n{out[-1800:]}\n"
            )
        else:
            prompt = (
                "A Python file has a bug: its tests fail. Fix the file so ALL tests pass. Do not change "
                "the tests. Return the COMPLETE corrected file in a single ```python fenced block, "
                f"nothing else.\n\n=== FILE ({target}) ===\n{prior}\n\n=== FAILING TEST OUTPUT ===\n{out[-2000:]}\n"
            )
        # Generous token budget: VibeThinker reasons at length before emitting the fix, and a starved
        # call returns empty content (the code never gets written).
        code = extract_code(chat(prompt, max_tokens=6000))
        if "def " not in code:
            continue
        if span:
            lines = orig.split("\n")
            new_file = "\n".join(lines[: span[0]] + code.split("\n") + lines[span[1] :])
        else:
            new_file = code
        open(path, "w").write(new_file)
        ok, out = run_tests(repo, testcmd)
        if ok:
            return True, i, "verified green"
        prior = code
    open(path, "w").write(orig)  # never ship unverified
    return False, iters, "exhausted"


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    repo, target, testcmd = sys.argv[1], sys.argv[2], sys.argv[3]
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    success, used, note = repair(repo, target, testcmd, iters)
    print(f"success={success} iterations={used} note={note}")
    sys.exit(0 if success else 1)
