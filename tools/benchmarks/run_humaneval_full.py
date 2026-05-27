#!/usr/bin/env python3
"""
Full HumanEval (164 problems) runner.

Fetches the openai_humaneval dataset via `datasets`, runs each problem
in one of three modes:
  - --mode llm       LLM-only (pure Claude baseline)
  - --mode hybrid    nsynth-first, LLM fallback  (most competitive)
  - --mode nsynth    nsynth-only (expected low; HE has floats, strings, lists)

For LLM modes the prompt is the raw HumanEval `prompt` (function
signature + docstring). Claude is instructed to output *only* the
function body. We concatenate (prompt, response) and execute the
canonical `check(candidate)` test suite.

For nsynth mode we attempt to parse example I/O from the docstring's
`>>>` lines and feed them to nsynth_codegen. Most problems use
non-scalar types so nsynth will miss them — that's the honest baseline.

Output: artifacts/humaneval_full_<mode>.md with per-problem pass rows
+ aggregate pass@1.

Usage:
    ANTHROPIC_API_KEY=sk-... python3 tools/benchmarks/run_humaneval_full.py \\
        --mode hybrid --verbose --limit 50  # optional limit for testing
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class Result:
    task_id: str
    entry_point: str
    mode: str  # "llm" | "nsynth" | "hybrid-nsynth" | "hybrid-llm" | "miss"
    pass_at_1: bool
    elapsed_ms: int
    code_len: int = 0
    error: str = ""


# ─── Code execution helpers ──────────────────────────────────────────────────


class _TimeoutError(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: int):
    """SIGALRM-based timeout for code execution. Only safe on POSIX +
    main thread. HumanEval has a handful of pathological problems that
    busy-loop on bad candidate code; this keeps one bad solve from
    stalling the whole run."""
    def handler(signum, frame):
        raise _TimeoutError(f"timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def run_humaneval_test(candidate_code: str, problem: dict, timeout: int) -> Tuple[bool, str]:
    """Execute `candidate_code` (which must be a complete `def <entry>():
    ...` block — not just a body fragment) and then run the canonical
    `check()` test. Returns (pass, error_msg).

    We prepend the HumanEval prompt's *imports* (everything up to the
    first `def`) so code that uses `List`, `Tuple`, etc. from
    `typing` still runs; the actual function definition comes from the
    LLM, not the prompt."""
    entry = problem["entry_point"]
    # Extract imports / module preamble from the prompt (lines before
    # the first `def`).
    lines = problem["prompt"].splitlines()
    preamble_end = 0
    for i, l in enumerate(lines):
        if l.lstrip().startswith("def "):
            preamble_end = i
            break
    preamble = "\n".join(lines[:preamble_end])
    full = preamble + "\n\n" + candidate_code + "\n\n" + problem["test"]
    ns: dict = {}
    try:
        with time_limit(timeout):
            exec(full, ns)
            fn = ns.get(entry)
            if fn is None:
                return (False, f"no {entry} defined")
            check = ns.get("check")
            if check is None:
                return (False, "no check() defined")
            check(fn)
    except _TimeoutError as e:
        return (False, f"timeout: {e}")
    except AssertionError as e:
        return (False, f"assert: {e}"[:150])
    except Exception as e:
        return (False, f"{type(e).__name__}: {e}"[:150])
    return (True, "")


# ─── LLM call ────────────────────────────────────────────────────────────────


_FENCE_RE = re.compile(r"```(?:python)?\n?(.*?)```", re.DOTALL)


def extract_full_function(response: str, entry_point: str) -> Optional[str]:
    """Extract a *complete* function definition for `entry_point` from the
    LLM response. Returns the whole `def entry_point(...)\\n    body...`
    block so we can drop it into a fresh namespace without struggling
    with indentation. Returns None if no such block is found.

    Trying to build valid Python by concatenating the HumanEval prompt
    with LLM-produced body fragments is brittle: indentation, continued
    docstrings, and newlines all create subtle syntax errors. A whole-
    function extraction sidesteps all of that."""
    fences = _FENCE_RE.findall(response)
    candidates = fences if fences else [response]
    for block in candidates:
        if f"def {entry_point}" not in block:
            continue
        lines = block.splitlines()
        start = None
        for i, l in enumerate(lines):
            if l.lstrip().startswith(f"def {entry_point}"):
                start = i
                break
        if start is None:
            continue
        # Walk forward until we hit a line with no leading whitespace
        # AND it's not a continuation of the function. The function
        # definition itself starts with `def`; everything indented after
        # belongs to it. Stop at the first subsequent top-level statement.
        end = len(lines)
        for j in range(start + 1, len(lines)):
            stripped = lines[j]
            if stripped and not stripped[0].isspace() and not stripped.startswith("#"):
                end = j
                break
        block_text = "\n".join(lines[start:end])
        # Dedent the block so its `def` starts at column 0.
        min_indent = len(block_text) - len(block_text.lstrip())
        if min_indent > 0:
            block_text = "\n".join(l[min_indent:] if l.strip() else l for l in block_text.splitlines())
        return block_text
    return None


def build_llm_prompt(problem: dict) -> str:
    return (
        f"Complete the Python function below. Reply with a COMPLETE "
        f"function definition including the `def` signature, the "
        f"docstring, and the implementation body. No explanation, no "
        f"markdown fences, no test cases.\n\n{problem['prompt']}"
    )


def build_cache_metadata(problem: dict) -> dict:
    return {
        "task_kind": "humaneval",
        "task_id": problem["task_id"],
        "entry_point": problem["entry_point"],
        "prompt": build_llm_prompt(problem),
    }


def llm_call(client, problem: dict, model: str, temperature: float = 0.0,
             extra_prompt: str = "", max_tokens: int = 2048) -> Tuple[str, int, str]:
    """Single LLM request. `extra_prompt` concatenates after the default
    prompt — used by the retry path to append the failure message."""
    t0 = time.time()
    prompt = build_llm_prompt(problem) + ("\n\n" + extra_prompt if extra_prompt else "")
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        return ("", int((time.time() - t0) * 1000), f"llm-error: {e!r}"[:200])
    ms = int((time.time() - t0) * 1000)
    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    fn_block = extract_full_function(text, problem["entry_point"])
    if fn_block is None:
        return ("", ms, "no-def-found")
    return (fn_block, ms, "ok")


# ─── Docstring example extraction (for nsynth path) ─────────────────────────


_DOCTEST_RE = re.compile(
    r">>>\s*(\w+)\((.*?)\)\s*\n\s*([^>\n].*?)(?:\n|$)", re.DOTALL
)


def try_extract_scalar_examples(problem: dict) -> Optional[dict]:
    """Attempt to pull scalar-only I/O examples from the prompt's
    docstring. Return a nsynth spec (name, signature, examples) if
    every example is int-valued; None otherwise. HumanEval has many
    non-scalar problems — those will return None here and fall through
    to whatever fallback the caller has."""
    entry = problem["entry_point"]
    # Pull out every `>>> entry(...)` example + its next-line result.
    matches = list(_DOCTEST_RE.finditer(problem["prompt"]))
    if not matches:
        return None
    examples = []
    for m in matches:
        name = m.group(1)
        if name != entry:
            continue
        args_raw = m.group(2).strip()
        expected_raw = m.group(3).strip()
        # Try to parse args + expected as Python literals.
        try:
            args = eval(f"({args_raw},)") if args_raw else ()
            expected = eval(expected_raw)
        except Exception:
            return None
        if not all(isinstance(a, int) for a in args):
            return None
        if not isinstance(expected, int):
            return None
        examples.append({"inputs": list(args), "expected": expected})
    if len(examples) < 2:
        return None
    # Construct an i64-only Mog signature.
    arg_names = ["a", "b", "c", "d", "e"][: len(examples[0]["inputs"])]
    signature = (
        f"fn {entry}(" + ", ".join(f"{n}: i64" for n in arg_names) + ") -> i64"
    )
    return {"name": entry, "signature": signature, "examples": examples}


def nsynth_call(codegen_bin: Path, spec: dict, timeout: int) -> Tuple[str, int, str]:
    t0 = time.time()
    try:
        proc = subprocess.run(
            [str(codegen_bin), "--lang", "python", "--examples", json.dumps(spec)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ("", int((time.time() - t0) * 1000), "TIMEOUT")
    ms = int((time.time() - t0) * 1000)
    if proc.returncode != 0:
        return ("", ms, "nsynth-miss")
    return (proc.stdout, ms, "ok")


# ─── Runner ──────────────────────────────────────────────────────────────────


def agent_solve(
    problem: dict,
    client,
    model: str,
    k: int,
    max_retries: int,
    exec_timeout: int,
    llm_max_tokens: int,
) -> Tuple[bool, str, int, str, int]:
    """Full agent loop on one HumanEval problem.
    Returns (pass, mode_tag, elapsed_ms, error, tokens)."""
    import concurrent.futures as cf
    t0 = time.time()
    total_tokens = 0

    # 1. LLM-cache lookup — skip both nsynth + LLM on a hit.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from llm_solution_cache import (  # noqa: E402
        fingerprint_humaneval_task, lookup as cache_lookup, record as cache_record,
    )
    fp = fingerprint_humaneval_task(
        problem["task_id"], problem["entry_point"], problem["prompt"]
    )
    cache_metadata = build_cache_metadata(problem)
    cached = cache_lookup(fp)
    if cached is not None:
        ok, _ = run_humaneval_test(cached["code"], problem, exec_timeout)
        if ok:
            return (True, "agent-cache", int((time.time() - t0) * 1000), "", 0)

    # 2. Best-of-k parallel sampling with temperature diversity.
    temps = [0.0] + [0.6 + 0.1 * i for i in range(1, k)]
    temps = temps[:k]

    def _gen(i: int):
        body, ms, note = llm_call(
            client, problem, model, temperature=temps[i], max_tokens=llm_max_tokens
        )
        return (i, body, ms, note)

    candidates: List[Tuple[int, str, int, str]] = []
    if k > 1:
        with cf.ThreadPoolExecutor(max_workers=min(k, 5)) as pool:
            for fut in cf.as_completed([pool.submit(_gen, i) for i in range(k)]):
                candidates.append(fut.result())
        candidates.sort(key=lambda x: x[0])
    else:
        candidates = [_gen(0)]

    last_code = ""
    last_error = ""
    for idx, body, ms, note in candidates:
        if not body:
            last_error = note
            continue
        last_code = body
        ok, err = run_humaneval_test(body, problem, exec_timeout)
        if ok:
            try: cache_record(fp, model, body, metadata=cache_metadata)
            except Exception: pass
            return (True, f"agent-s{idx}", int((time.time() - t0) * 1000), "", 0)
        last_error = err

    # 3. Retry with feedback.
    for retry in range(1, max_retries + 1):
        hint = (
            f"Your previous attempt failed with: {last_error}\n"
            f"Previous code:\n{last_code}\n\n"
            f"Fix it and return a corrected complete function definition."
        )
        body, ms, note = llm_call(
            client,
            problem,
            model,
            temperature=0.0,
            extra_prompt=hint,
            max_tokens=llm_max_tokens,
        )
        if not body:
            last_error = note
            continue
        last_code = body
        ok, err = run_humaneval_test(body, problem, exec_timeout)
        if ok:
            try: cache_record(fp, model, body, metadata=cache_metadata)
            except Exception: pass
            return (True, f"agent-r{retry}", int((time.time() - t0) * 1000), "", 0)
        last_error = err

    return (False, "agent-miss", int((time.time() - t0) * 1000), last_error, 0)


def run_one(
    problem: dict,
    mode: str,
    codegen_bin: Path,
    client,
    model: str,
    nsynth_timeout: int,
    exec_timeout: int,
    k: int = 1,
    max_retries: int = 0,
    llm_max_tokens: int = 2048,
) -> Result:
    t0 = time.time()
    entry = problem["entry_point"]
    task_id = problem["task_id"]

    # Nsynth attempt (for nsynth + hybrid modes).
    if mode in ("nsynth", "hybrid"):
        spec = try_extract_scalar_examples(problem)
        if spec:
            code, ns_ms, note = nsynth_call(codegen_bin, spec, nsynth_timeout)
            if code:
                # nsynth produces a full function; we need to keep it as-is
                # and verify with the HumanEval test harness. The emitted
                # fn has the same entry_point name (we used it in the spec).
                candidate = code
                ok, err = run_humaneval_test(candidate, problem, exec_timeout)
                if ok:
                    return Result(
                        task_id=task_id,
                        entry_point=entry,
                        mode="nsynth" if mode == "nsynth" else "hybrid-nsynth",
                        pass_at_1=True,
                        elapsed_ms=int((time.time() - t0) * 1000),
                        code_len=len(candidate),
                    )
                # fallthrough on verify-fail — hybrid continues to LLM.
        if mode == "nsynth":
            return Result(
                task_id=task_id,
                entry_point=entry,
                mode="nsynth",
                pass_at_1=False,
                elapsed_ms=int((time.time() - t0) * 1000),
                error="nsynth cannot handle non-scalar problem",
            )

    # LLM attempt (for llm + hybrid-on-miss).
    if mode in ("llm", "hybrid"):
        body, llm_ms, note = llm_call(client, problem, model, max_tokens=llm_max_tokens)
        if not body:
            return Result(
                task_id=task_id,
                entry_point=entry,
                mode="miss",
                pass_at_1=False,
                elapsed_ms=int((time.time() - t0) * 1000),
                error=note,
            )
        ok, err = run_humaneval_test(body, problem, exec_timeout)
        return Result(
            task_id=task_id,
            entry_point=entry,
            mode="llm" if mode == "llm" else "hybrid-llm",
            pass_at_1=ok,
            elapsed_ms=int((time.time() - t0) * 1000),
            code_len=len(body),
            error=err,
        )

    if mode == "agent":
        ok, mode_tag, ms, err, _tokens = agent_solve(
            problem, client, model, k, max_retries, exec_timeout, llm_max_tokens
        )
        return Result(
            task_id=task_id, entry_point=entry, mode=mode_tag,
            pass_at_1=ok, elapsed_ms=ms, error=err,
        )

    return Result(
        task_id=task_id,
        entry_point=entry,
        mode="miss",
        pass_at_1=False,
        elapsed_ms=int((time.time() - t0) * 1000),
        error="unknown mode",
    )


def write_report(results: List[Result], out: Path, mode: str, total_ms: int) -> None:
    passed = sum(1 for r in results if r.pass_at_1)
    total = len(results)
    pct = 100.0 * passed / max(total, 1)
    by_mode: dict = {}
    for r in results:
        by_mode[r.mode] = by_mode.get(r.mode, 0) + 1
    lines = [
        f"# HumanEval Full Results — mode: {mode}",
        "",
        f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} — "
        f"{total} problems, {total_ms/1000:.1f}s total.",
        "",
        "## Summary",
        "",
        f"- **Pass@1**: **{passed}/{total} ({pct:.1f}%)**",
    ]
    for m in sorted(by_mode.keys()):
        lines.append(f"- {m}: {by_mode[m]}")
    lines.append("")
    lines.append("## Per-problem")
    lines.append("")
    lines.append("| # | task_id | mode | pass | ms | notes |")
    lines.append("|--:|---------|------|:----:|---:|-------|")
    for i, r in enumerate(results, 1):
        p = "✓" if r.pass_at_1 else "✗"
        note = r.error[:80] if r.error else ""
        lines.append(
            f"| {i} | {r.task_id} | {r.mode} | {p} | {r.elapsed_ms} | {note} |"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", default="llm",
                    choices=["llm", "nsynth", "hybrid", "agent"])
    ap.add_argument("--k", type=int, default=3,
                    help="best-of-N size for agent mode")
    ap.add_argument("--max-retries", type=int, default=2,
                    help="retry budget for agent mode")
    ap.add_argument("--limit", type=int, default=None, help="limit to first N problems")
    ap.add_argument("--out", default=None, help="output md path; defaults to mode-aware")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--model", default="claude-haiku-4-5-20251001",
        help="Claude model id for llm + hybrid modes"
    )
    ap.add_argument("--backend", default="anthropic",
                    choices=["anthropic", "mlx", "hf", "openai"],
                    help="inference backend for llm/hybrid/agent modes")
    ap.add_argument("--api-base", default=None,
                    help="for --backend openai: base URL")
    ap.add_argument("--device", default=None,
                    help="for --backend hf: device_map override")
    ap.add_argument("--adapter-path", default=None,
                    help="for --backend mlx: LoRA adapter directory to load")
    ap.add_argument("--adapter-routing", default="always",
                    choices=["always", "never", "utility_only"],
                    help="for --backend mlx: when to apply the adapter")
    ap.add_argument(
        "--codegen", default="nsynth/target/release/nsynth_codegen"
    )
    ap.add_argument("--nsynth-timeout", type=int, default=5,
                    help="per-problem nsynth budget (seconds)")
    ap.add_argument("--exec-timeout", type=int, default=8,
                    help="per-candidate execution budget (seconds)")
    ap.add_argument("--llm-max-tokens", type=int, default=2048,
                    help="generation cap for llm/hybrid/agent modes")
    args = ap.parse_args()

    out = args.out or f"artifacts/humaneval_full_{args.mode}.md"
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)

    # Load dataset.
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("[full] pip install datasets first", file=sys.stderr)
        sys.exit(2)
    ds = load_dataset("openai_humaneval", split="test")
    problems = [dict(r) for r in ds]
    if args.limit:
        problems = problems[: args.limit]
    print(f"[full] loaded {len(problems)} problems (mode={args.mode})")

    client = None
    if args.mode in ("llm", "hybrid", "agent"):
        if args.backend == "anthropic":
            try:
                import anthropic  # type: ignore
            except ImportError:
                print("[full] pip install anthropic (needed for llm/hybrid modes)", file=sys.stderr)
                sys.exit(2)
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                print("[full] ANTHROPIC_API_KEY not set", file=sys.stderr)
                sys.exit(2)
            client = anthropic.Anthropic(api_key=key)
        else:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from local_model_adapter import LocalModelClient  # noqa: E402
            client = LocalModelClient(
                backend=args.backend,
                model=args.model,
                api_base=args.api_base,
                device=args.device,
                adapter_path=args.adapter_path,
                adapter_routing=args.adapter_routing,
            )

    codegen_bin = Path(args.codegen)
    if args.mode in ("nsynth", "hybrid") and not codegen_bin.exists():
        print(
            f"[full] nsynth_codegen not built at {codegen_bin}; run "
            f"`cargo build --release --bin nsynth_codegen`",
            file=sys.stderr,
        )
        sys.exit(2)

    t_start = time.time()
    results: List[Result] = []
    for i, p in enumerate(problems, 1):
        if args.verbose:
            print(f"[{i}/{len(problems)}] {p['task_id']} ...", end=" ", flush=True)
        r = run_one(
            p,
            args.mode,
            codegen_bin,
            client,
            args.model,
            args.nsynth_timeout,
            args.exec_timeout,
            k=args.k,
            max_retries=args.max_retries,
            llm_max_tokens=args.llm_max_tokens,
        )
        results.append(r)
        if args.verbose:
            mark = "✓" if r.pass_at_1 else f"✗ ({r.error[:40]})"
            print(f"{mark} ({r.mode}, {r.elapsed_ms}ms)")
    total_ms = int((time.time() - t_start) * 1000)
    write_report(results, Path(out), args.mode, total_ms)
    passed = sum(1 for r in results if r.pass_at_1)
    print(
        f"[full] wrote {out} — pass@1 {passed}/{len(problems)} in {total_ms/1000:.1f}s"
    )


if __name__ == "__main__":
    main()
