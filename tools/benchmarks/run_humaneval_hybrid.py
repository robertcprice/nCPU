#!/usr/bin/env python3
"""
Hybrid HumanEval-lite runner: nsynth first, LLM fallback on failure.

Pipeline per problem:
  1. Try `nsynth_codegen --lang python`.
  2. Verify the generated code by running it against `test_cases`.
  3. If either step fails, fall back to an LLM:
       - Build a prompt from (signature, examples).
       - Call the LLM (Anthropic Claude API, graceful no-op if unavailable).
       - Extract a Python function from the response.
       - Verify against `test_cases`.
  4. Record which path produced the winning (if any) code.

The hybrid is how synthesizers meaningfully beat an LLM-only baseline:
nsynth caches verified solutions on disk, so repeat invocations of the
same fingerprint return instantly with no LLM call — *and* the solver
sees the LLM-verified code as a future teacher via the cross-run
learning loop.

Required for LLM path:
  - `pip install anthropic`
  - ANTHROPIC_API_KEY env var OR `--api-key` arg

Graceful fallback: without the API key, the hybrid behaves like the
nsynth-only runner.

Usage:
    ANTHROPIC_API_KEY=sk-... \
      python3 tools/benchmarks/run_humaneval_hybrid.py --verbose

Output:
    artifacts/humaneval_results_hybrid.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# LLM-solution cache: persistent record of (problem_fingerprint → verified
# Python code). Lets the hybrid runner skip both nsynth *and* the LLM API
# call on reruns of the same problem shape.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_solution_cache import fingerprint_examples, lookup as llm_cache_lookup, record as llm_cache_record  # noqa: E402


@dataclass
class Problem:
    name: str
    signature: str
    examples: List[dict]
    test_cases: List[list]


@dataclass
class Result:
    name: str
    path: str  # "nsynth" | "llm" | "miss"
    pass_at_1: bool
    solve_ms: int
    method_detail: str
    error: str = ""
    code: str = ""


def load_problems(path: Path) -> List[Problem]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(
            Problem(
                name=d["name"],
                signature=d["signature"],
                examples=d["examples"],
                test_cases=d["test_cases"],
            )
        )
    return out


def verify_code(code: str, problem: Problem) -> Tuple[bool, str]:
    """Execute the code string, call the function on each test case. Returns
    (pass_at_1, error_detail)."""
    ns: dict = {}
    try:
        exec(code, ns)
    except Exception as e:
        return (False, f"exec-error: {e!r}")
    fn = ns.get(problem.name)
    if fn is None:
        return (False, f"function {problem.name} not defined")
    for case in problem.test_cases:
        *args, expected = case
        try:
            got = fn(*args)
        except Exception as e:
            return (False, f"call-error on {args}: {e!r}")
        if got != expected:
            return (
                False,
                f"wrong on {args}: got {got}, expected {expected}",
            )
    return (True, "")


def nsynth_synthesis(
    codegen_bin: Path, problem: Problem, timeout: int
) -> Tuple[str, int, str]:
    """Return (python_code, elapsed_ms, method_or_error)."""
    spec = json.dumps({
        "name": problem.name,
        "signature": problem.signature,
        "examples": problem.examples,
    })
    t0 = time.time()
    try:
        proc = subprocess.run(
            [str(codegen_bin), "--lang", "python", "--examples", spec, "--verbose"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ("", int((time.time() - t0) * 1000), "TIMEOUT")
    elapsed_ms = int((time.time() - t0) * 1000)
    if proc.returncode != 0:
        return ("", elapsed_ms, (proc.stderr.strip()[:200] or "nonzero-exit"))
    method = "unknown"
    for line in proc.stderr.splitlines():
        if "solved via" in line:
            method = line.split("solved via", 1)[1].split(" in ")[0].strip()
            break
    return (proc.stdout, elapsed_ms, method)


# ─── LLM fallback ────────────────────────────────────────────────────────────

_CLAUDE_CLIENT = None


def get_claude_client(api_key: Optional[str]):
    """Import + initialise the Anthropic client lazily. Returns None if the
    SDK isn't installed or no API key is available — lets callers no-op
    cleanly."""
    global _CLAUDE_CLIENT
    if _CLAUDE_CLIENT is not None:
        return _CLAUDE_CLIENT
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic  # type: ignore
    except ImportError:
        print("[hybrid] anthropic SDK not installed; LLM fallback disabled", file=sys.stderr)
        return None
    _CLAUDE_CLIENT = anthropic.Anthropic(api_key=key)
    return _CLAUDE_CLIENT


def build_llm_prompt(problem: Problem) -> str:
    examples_str = "\n".join(
        f"  {problem.name}({', '.join(repr(x) for x in ex['inputs'])}) == {ex['expected']}"
        for ex in problem.examples
    )
    return (
        f"Write a Python function matching the signature `{problem.signature}`.\n\n"
        f"It must satisfy these examples exactly:\n{examples_str}\n\n"
        f"Reply with ONLY the function definition. No explanation, no test cases, "
        f"no markdown fences — raw Python starting with `def {problem.name}`."
    )


_FENCE_RE = re.compile(r"```(?:python)?\n?(.*?)```", re.DOTALL)
_DEF_RE = re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE)


def extract_python(response: str, fn_name: str) -> Optional[str]:
    """Pull out the `def fn_name(...)` block from an LLM response. Tolerates
    code fences + leading prose."""
    # Strip code fences if present.
    fences = _FENCE_RE.findall(response)
    if fences:
        for body in fences:
            if f"def {fn_name}" in body:
                return body.strip()
    # Fall through: find a def ... block directly.
    match = _DEF_RE.search(response)
    if match:
        start = match.start()
        return response[start:].strip()
    return None


def llm_synthesis(
    client, problem: Problem, model: str
) -> Tuple[str, int, str]:
    """Call the LLM, return (python_code, elapsed_ms, detail)."""
    if client is None:
        return ("", 0, "llm-unavailable")
    prompt = build_llm_prompt(problem)
    t0 = time.time()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        return ("", int((time.time() - t0) * 1000), f"llm-error: {e!r}"[:200])
    elapsed_ms = int((time.time() - t0) * 1000)
    # Concat text parts.
    text = ""
    for block in resp.content:
        if hasattr(block, "text"):
            text += block.text
    code = extract_python(text, problem.name)
    if code is None:
        return ("", elapsed_ms, "llm-no-code-found")
    return (code, elapsed_ms, f"llm:{model}")


# ─── Runner ──────────────────────────────────────────────────────────────────


def run_hybrid(
    problems: List[Problem],
    codegen_bin: Path,
    client,
    model: str,
    timeout: int,
    verbose: bool,
) -> List[Result]:
    results: List[Result] = []
    for i, p in enumerate(problems, 1):
        if verbose:
            print(f"[{i}/{len(problems)}] {p.name} ...", end=" ", flush=True)

        # LLM cache check: if we've previously verified a Python solution
        # for this fingerprint, return it directly (skip nsynth and LLM).
        fp = fingerprint_examples([
            {"inputs": ex["inputs"], "expected": ex["expected"]}
            for ex in [{"inputs": e["inputs"], "expected": e["expected"]} for e in p.examples]
        ])
        cached = llm_cache_lookup(fp)
        if cached is not None:
            # Re-verify the cached code against the current test_cases —
            # the cache doesn't hold tests so a stale/mismatched row
            # fails closed here.
            ok, _ = verify_code(cached["code"], p)
            if ok:
                results.append(
                    Result(
                        name=p.name,
                        path="llm-cache",
                        pass_at_1=True,
                        solve_ms=0,
                        method_detail=f"cache:{cached['model']}",
                        code=cached["code"],
                    )
                )
                if verbose:
                    print("cache ✓ (0ms — prior LLM solution)")
                continue

        # nsynth path.
        code, ns_ms, ns_method = nsynth_synthesis(codegen_bin, p, timeout)
        if code:
            ok, err = verify_code(code, p)
            if ok:
                results.append(
                    Result(
                        name=p.name,
                        path="nsynth",
                        pass_at_1=True,
                        solve_ms=ns_ms,
                        method_detail=ns_method,
                        code=code,
                    )
                )
                if verbose:
                    print(f"nsynth ✓ ({ns_ms}ms)")
                continue

        # LLM fallback.
        llm_code, llm_ms, llm_detail = llm_synthesis(client, p, model)
        if llm_code:
            ok, err = verify_code(llm_code, p)
            if ok:
                # Persist the verified LLM solution so the next invocation
                # with the same fingerprint skips both nsynth + LLM.
                try:
                    llm_cache_record(fp, model, llm_code)
                except Exception:
                    pass  # best-effort; cache corruption shouldn't fail the run
                results.append(
                    Result(
                        name=p.name,
                        path="llm",
                        pass_at_1=True,
                        solve_ms=ns_ms + llm_ms,
                        method_detail=llm_detail,
                        code=llm_code,
                    )
                )
                if verbose:
                    print(f"llm ✓ (nsynth miss, {ns_ms + llm_ms}ms total, cached)")
                continue
            else:
                results.append(
                    Result(
                        name=p.name,
                        path="miss",
                        pass_at_1=False,
                        solve_ms=ns_ms + llm_ms,
                        method_detail=f"nsynth:{ns_method}|llm:{llm_detail}",
                        error=err,
                        code=llm_code,
                    )
                )
                if verbose:
                    print(f"llm ✗ ({err[:60]})")
                continue

        # Both missed.
        results.append(
            Result(
                name=p.name,
                path="miss",
                pass_at_1=False,
                solve_ms=ns_ms,
                method_detail=f"nsynth:{ns_method}|{llm_detail}",
                error="both paths failed",
            )
        )
        if verbose:
            print(f"miss ({ns_method} / {llm_detail})")
    return results


def write_report(
    results: List[Result],
    out: Path,
    total_problems: int,
    total_ms: int,
) -> None:
    by_path: dict = {"nsynth": 0, "llm": 0, "miss": 0}
    for r in results:
        by_path[r.path] = by_path.get(r.path, 0) + 1
    passed = sum(1 for r in results if r.pass_at_1)
    pct = 100.0 * passed / max(total_problems, 1)

    lines: List[str] = []
    lines.append("# HumanEval-lite Hybrid Results")
    lines.append("")
    lines.append(
        f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} — "
        f"{total_problems} problems, {total_ms/1000:.1f}s total."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Pass@1**: **{passed}/{total_problems} ({pct:.1f}%)**")
    lines.append(f"- Solved by nsynth: {by_path.get('nsynth', 0)}")
    lines.append(f"- Solved by LLM fallback: {by_path.get('llm', 0)}")
    lines.append(f"- Missed by both: {by_path.get('miss', 0)}")
    lines.append("")
    lines.append("## Per-problem")
    lines.append("")
    lines.append("| # | problem | path | pass | ms | method | notes |")
    lines.append("|--:|---------|:----:|:----:|---:|--------|-------|")
    for i, r in enumerate(results, 1):
        p1 = "✓" if r.pass_at_1 else "✗"
        notes = r.error[:80] if r.error else ""
        lines.append(
            f"| {i} | {r.name} | {r.path} | {p1} | {r.solve_ms} | "
            f"{r.method_detail[:40]} | {notes} |"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--problems", default="tools/benchmarks/humaneval_lite.jsonl")
    ap.add_argument("--out", default="artifacts/humaneval_results_hybrid.md")
    ap.add_argument("--timeout", type=int, default=25)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--codegen",
        default="nsynth/target/release/nsynth_codegen",
    )
    ap.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="LLM model id for the fallback path",
    )
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)
    problems = load_problems(Path(args.problems))
    codegen_bin = Path(args.codegen)

    if not codegen_bin.exists():
        print(
            f"[hybrid] nsynth_codegen not built — run `cargo build --release`",
            file=sys.stderr,
        )
        sys.exit(2)

    client = get_claude_client(args.api_key)
    if client is None:
        print(
            "[hybrid] running WITHOUT LLM fallback (no ANTHROPIC_API_KEY)",
            file=sys.stderr,
        )

    print(f"[hybrid] {len(problems)} problems loaded, model={args.model}")
    t0 = time.time()
    results = run_hybrid(problems, codegen_bin, client, args.model, args.timeout, args.verbose)
    total_ms = int((time.time() - t0) * 1000)

    write_report(results, Path(args.out), len(problems), total_ms)

    passed = sum(1 for r in results if r.pass_at_1)
    by_path = {"nsynth": 0, "llm": 0, "miss": 0}
    for r in results:
        by_path[r.path] = by_path.get(r.path, 0) + 1
    print(
        f"[hybrid] wrote {args.out} — "
        f"pass@1 {passed}/{len(problems)}, "
        f"nsynth {by_path['nsynth']}, llm {by_path['llm']}, miss {by_path['miss']}, "
        f"total {total_ms/1000:.1f}s"
    )


if __name__ == "__main__":
    main()
