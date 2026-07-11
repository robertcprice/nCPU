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
import json
import os
import re
import subprocess
import sys
import urllib.request

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
        with urllib.request.urlopen(req, timeout=400) as r:
            d = json.load(r)
        return d["choices"][0]["message"].get("content") or ""
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
    cand = None
    for f in re.findall(r'File "([^"]+\.py)"', out):
        base = os.path.basename(f)
        if base.startswith("test_") or base.endswith("_test.py") or base == "conftest.py":
            continue
        rel = os.path.relpath(f, repo) if os.path.isabs(f) else f
        if os.path.exists(os.path.join(repo, rel)) and not rel.startswith(".."):
            cand = rel  # last frame wins (deepest in the call stack)
    return cand


def repair(repo, target, testcmd, iters=3):
    """Returns (success, iterations_used, note). Reverts the file on failure. `target` may be None or
    "auto" to localize the buggy file from the baseline failure traceback."""
    ok, out = run_tests(repo, testcmd)
    if ok:
        return True, 0, "baseline already green"
    if not target or target == "auto":
        target = localize_target(repo, out)
        if not target:
            return False, 0, "could not localize a non-test file from the failure"
        print(f"  [localized target: {target}]", file=sys.stderr)
    path = os.path.join(repo, target)
    orig = open(path).read()
    prior = orig
    for i in range(1, iters + 1):
        prompt = (
            "A Python file has a bug: its tests fail. Fix the file so ALL tests pass. Do not change "
            "the tests. Return the COMPLETE corrected file in a single ```python fenced block, nothing "
            f"else.\n\n=== FILE ({target}) ===\n{prior}\n\n=== FAILING TEST OUTPUT ===\n{out[-2000:]}\n"
        )
        code = extract_code(chat(prompt))
        if "def " not in code:
            continue
        open(path, "w").write(code)
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
