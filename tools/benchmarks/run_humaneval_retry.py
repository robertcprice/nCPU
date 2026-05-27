#!/usr/bin/env python3
"""
Retry-with-error-feedback runner.

For each problem: generate a candidate, verify, on failure prompt the
LLM again with the specific error message ("your last attempt failed
on input X with output Y, expected Z — fix it"). Up to N retry rounds.

This is what a code-gen agent actually does — the LLM iterates against
the verification oracle. Each retry is informed by the previous
attempt's failure, not a blind re-roll.

Expected: tight compounding with best-of-N. Pass@1 ceiling approaches
the problem set's inherent decidability.

Usage:
    ANTHROPIC_API_KEY=sk-... python3 \\
        tools/benchmarks/run_humaneval_retry.py --verbose --max-retries 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_solution_cache import fingerprint_examples, lookup as cache_lookup, record as cache_record  # noqa: E402


@dataclass
class Problem:
    name: str
    signature: str
    examples: List[dict]
    test_cases: List[list]


@dataclass
class Result:
    name: str
    pass_at_1: bool
    retries_used: int
    solve_ms: int
    total_tokens: int = 0
    final_error: str = ""


def load_problems(path: Path) -> List[Problem]:
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(Problem(
                name=d["name"], signature=d["signature"],
                examples=d["examples"], test_cases=d["test_cases"]))
    return out


def verify_code(code: str, problem: Problem) -> Tuple[bool, str, Optional[list]]:
    """Return (pass, error_msg, failing_case).
    failing_case is the specific test that broke, so we can feed it back
    to the LLM for a targeted fix."""
    ns: dict = {}
    try:
        exec(code, ns)
    except Exception as e:
        return (False, f"exec-error: {e!r}"[:150], None)
    fn = ns.get(problem.name)
    if fn is None:
        return (False, f"function {problem.name} not defined", None)
    for case in problem.test_cases:
        *args, expected = case
        try:
            got = fn(*args)
        except Exception as e:
            return (False, f"call({args}): {e!r}"[:150], case)
        if got != expected:
            return (
                False,
                f"{problem.name}({', '.join(repr(a) for a in args)}) "
                f"returned {got!r}, expected {expected!r}",
                case,
            )
    return (True, "", None)


_FENCE_RE = re.compile(r"```(?:python)?\n?(.*?)```", re.DOTALL)
_DEF_RE = re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE)


def extract_python(response: str, fn_name: str) -> Optional[str]:
    fences = _FENCE_RE.findall(response)
    if fences:
        for body in fences:
            if f"def {fn_name}" in body:
                return body.strip()
    match = _DEF_RE.search(response)
    if match:
        return response[match.start() :].strip()
    return None


def build_initial_prompt(problem: Problem) -> str:
    examples_str = "\n".join(
        f"  {problem.name}({', '.join(repr(x) for x in ex['inputs'])}) == {ex['expected']}"
        for ex in problem.examples
    )
    return (
        f"Write a Python function matching the signature `{problem.signature}`.\n\n"
        f"It must satisfy these examples exactly:\n{examples_str}\n\n"
        f"Reply with ONLY the function definition. No explanation, no test cases."
    )


def build_retry_prompt(problem: Problem, previous_code: str, error: str) -> str:
    examples_str = "\n".join(
        f"  {problem.name}({', '.join(repr(x) for x in ex['inputs'])}) == {ex['expected']}"
        for ex in problem.examples
    )
    return (
        f"Your previous attempt at `{problem.signature}` had a bug.\n\n"
        f"Previous code:\n```python\n{previous_code}\n```\n\n"
        f"The failure: {error}\n\n"
        f"It must satisfy these examples exactly:\n{examples_str}\n\n"
        f"Reply with ONLY a corrected function definition. No explanation."
    )


def llm_call(client, prompt: str, model: str, temperature: float = 0.0) -> Tuple[str, int, int]:
    """Return (text, elapsed_ms, tokens)."""
    t0 = time.time()
    try:
        resp = client.messages.create(
            model=model, max_tokens=768, temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        return ("", int((time.time() - t0) * 1000), 0)
    ms = int((time.time() - t0) * 1000)
    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    usage = getattr(resp, "usage", None)
    tokens = 0
    if usage is not None:
        tokens = getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)
    return (text, ms, tokens)


def solve_with_retry(
    client, problem: Problem, model: str, max_retries: int
) -> Result:
    t0 = time.time()

    # Cache check.
    fp = fingerprint_examples([
        {"inputs": ex["inputs"], "expected": ex["expected"]} for ex in problem.examples
    ])
    cached = cache_lookup(fp)
    if cached is not None:
        ok, _, _ = verify_code(cached["code"], problem)
        if ok:
            return Result(
                name=problem.name, pass_at_1=True, retries_used=-1,
                solve_ms=0, total_tokens=0,
            )

    total_tokens = 0
    last_code = ""
    last_error = ""

    # Attempt 0: initial prompt. Then up to `max_retries` corrections.
    for attempt in range(max_retries + 1):
        if attempt == 0:
            prompt = build_initial_prompt(problem)
        else:
            prompt = build_retry_prompt(problem, last_code, last_error)

        text, ms, tokens = llm_call(client, prompt, model)
        total_tokens += tokens
        code = extract_python(text, problem.name)
        if code is None:
            last_error = "no-code-found"
            continue
        last_code = code
        ok, err, failing_case = verify_code(code, problem)
        if ok:
            try:
                cache_record(fp, model, code)
            except Exception:
                pass
            return Result(
                name=problem.name, pass_at_1=True, retries_used=attempt,
                solve_ms=int((time.time() - t0) * 1000),
                total_tokens=total_tokens,
            )
        last_error = err

    return Result(
        name=problem.name, pass_at_1=False,
        retries_used=max_retries,
        solve_ms=int((time.time() - t0) * 1000),
        total_tokens=total_tokens,
        final_error=last_error,
    )


def write_report(results: List[Result], out: Path, max_retries: int, total_ms: int, model: str) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.pass_at_1)
    pct = 100.0 * passed / max(total, 1)
    cache_hits = sum(1 for r in results if r.retries_used == -1)
    retries_dist: dict = {}
    for r in results:
        if r.pass_at_1 and r.retries_used >= 0:
            retries_dist[r.retries_used] = retries_dist.get(r.retries_used, 0) + 1
    total_tok = sum(r.total_tokens for r in results)

    lines = [
        "# HumanEval-lite Retry-with-Feedback Results", "",
        f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} — "
        f"{total} problems, max_retries={max_retries}, model={model}, {total_ms/1000:.1f}s total.",
        "",
        "## Summary", "",
        f"- **Pass@1 (up to {max_retries} retries)**: **{passed}/{total} ({pct:.1f}%)**",
        f"- Cache hits: {cache_hits}",
        f"- Total tokens: ~{total_tok}",
        "",
        "### Retries-to-pass distribution",
        "",
        "How many retry rounds did each winning solution need?",
        "",
        "| retries | count |",
        "|--------:|------:|",
    ]
    for k in sorted(retries_dist.keys()):
        lines.append(f"| {k} | {retries_dist[k]} |")
    lines += ["", "## Per-problem", "", "| # | problem | pass | retries | ms | notes |",
              "|--:|---------|:----:|:-------:|---:|-------|"]
    for i, r in enumerate(results, 1):
        p = "✓" if r.pass_at_1 else "✗"
        rstr = "cache" if r.retries_used == -1 else str(r.retries_used)
        note = r.final_error[:90] if r.final_error else ""
        lines.append(f"| {i} | {r.name} | {p} | {rstr} | {r.solve_ms} | {note} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--problems", default="tools/benchmarks/humaneval_lite.jsonl")
    ap.add_argument("--out", default="artifacts/humaneval_results_retry.md")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)

    try:
        import anthropic
    except ImportError:
        print("[retry] anthropic SDK required", file=sys.stderr); sys.exit(2)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("[retry] ANTHROPIC_API_KEY not set", file=sys.stderr); sys.exit(2)
    client = anthropic.Anthropic(api_key=key)

    problems = load_problems(Path(args.problems))
    print(f"[retry] {len(problems)} problems, max_retries={args.max_retries}, model={args.model}")

    t0 = time.time()
    results: List[Result] = []
    for i, p in enumerate(problems, 1):
        if args.verbose:
            print(f"[{i}/{len(problems)}] {p.name} ...", end=" ", flush=True)
        r = solve_with_retry(client, p, args.model, args.max_retries)
        results.append(r)
        if args.verbose:
            if r.retries_used == -1:
                mark = "cache ✓"
            elif r.pass_at_1:
                mark = f"✓ (retries={r.retries_used})"
            else:
                mark = f"✗ ({r.final_error[:50]})"
            print(f"{mark} ({r.solve_ms}ms)")
    total_ms = int((time.time() - t0) * 1000)

    write_report(results, Path(args.out), args.max_retries, total_ms, args.model)
    passed = sum(1 for r in results if r.pass_at_1)
    print(f"[retry] wrote {args.out} — pass@1 {passed}/{len(problems)} in {total_ms/1000:.1f}s")


if __name__ == "__main__":
    main()
