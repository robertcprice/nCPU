#!/usr/bin/env python3
"""
LLM-only runner for humaneval_lite — the pure-Claude baseline.

Skips nsynth entirely, always queries the configured Claude model and
verifies by executing against `test_cases`. The honest "what does the
LLM alone score?" number, for three-way comparison against
  - nsynth-only (artifacts/humaneval_results.md)
  - hybrid      (artifacts/humaneval_results_hybrid.md)
  - this file   (artifacts/humaneval_results_llm_only.md)

Usage:
    ANTHROPIC_API_KEY=sk-... \\
        python3 tools/benchmarks/run_humaneval_llm_only.py --verbose

Arguments mirror run_humaneval_hybrid.py so the runs are directly
comparable.
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
    solve_ms: int
    model: str
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
    ns: dict = {}
    try:
        exec(code, ns)
    except Exception as e:
        return (False, f"exec: {e!r}"[:80])
    fn = ns.get(problem.name)
    if fn is None:
        return (False, f"no fn {problem.name}")
    for case in problem.test_cases:
        *args, expected = case
        try:
            got = fn(*args)
        except Exception as e:
            return (False, f"call: {e!r}"[:60])
        if got != expected:
            return (False, f"wrong {args}→{got} exp {expected}"[:60])
    return (True, "")


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


def build_prompt(problem: Problem) -> str:
    examples_str = "\n".join(
        f"  {problem.name}({', '.join(repr(x) for x in ex['inputs'])}) == {ex['expected']}"
        for ex in problem.examples
    )
    return (
        f"Write a Python function matching the signature `{problem.signature}`.\n\n"
        f"It must satisfy these examples exactly:\n{examples_str}\n\n"
        f"Reply with ONLY the function definition starting with `def {problem.name}`. "
        f"No explanation, no test cases, no markdown fences."
    )


def run_one(client, problem: Problem, model: str) -> Result:
    t0 = time.time()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": build_prompt(problem)}],
        )
    except Exception as e:
        ms = int((time.time() - t0) * 1000)
        return Result(
            name=problem.name,
            pass_at_1=False,
            solve_ms=ms,
            model=model,
            error=f"llm-error: {e!r}"[:200],
        )
    ms = int((time.time() - t0) * 1000)
    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    code = extract_python(text, problem.name)
    if code is None:
        return Result(
            name=problem.name,
            pass_at_1=False,
            solve_ms=ms,
            model=model,
            error="no-code-found",
        )
    ok, err = verify_code(code, problem)
    return Result(
        name=problem.name,
        pass_at_1=ok,
        solve_ms=ms,
        model=model,
        error=err,
        code=code,
    )


def write_report(
    results: List[Result], out: Path, total: int, total_ms: int, model: str
) -> None:
    passed = sum(1 for r in results if r.pass_at_1)
    pct = 100.0 * passed / max(total, 1)
    lines = [
        "# HumanEval-lite LLM-only Results",
        "",
        f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} — "
        f"{total} problems, {total_ms/1000:.1f}s total, model={model}.",
        "",
        "## Summary",
        "",
        f"- **Pass@1**: **{passed}/{total} ({pct:.1f}%)**",
        "",
        "## Per-problem",
        "",
        "| # | problem | pass | ms | notes |",
        "|--:|---------|:----:|---:|-------|",
    ]
    for i, r in enumerate(results, 1):
        p1 = "✓" if r.pass_at_1 else "✗"
        note = r.error[:80] if r.error else ""
        lines.append(f"| {i} | {r.name} | {p1} | {r.solve_ms} | {note} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--problems", default="tools/benchmarks/humaneval_lite.jsonl")
    ap.add_argument("--out", default="artifacts/humaneval_results_llm_only.md")
    ap.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Claude model id",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    problems = load_problems(Path(args.problems))

    try:
        import anthropic
    except ImportError:
        print("[llm_only] anthropic SDK not installed", file=sys.stderr)
        sys.exit(2)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("[llm_only] ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(2)
    client = anthropic.Anthropic(api_key=key)

    print(f"[llm_only] {len(problems)} problems, model={args.model}")
    t0 = time.time()
    results: List[Result] = []
    for i, p in enumerate(problems, 1):
        if args.verbose:
            print(f"[{i}/{len(problems)}] {p.name} ...", end=" ", flush=True)
        r = run_one(client, p, args.model)
        results.append(r)
        if args.verbose:
            mark = "✓" if r.pass_at_1 else f"✗ {r.error[:60]}"
            print(f"{mark} ({r.solve_ms}ms)")
    total_ms = int((time.time() - t0) * 1000)

    write_report(results, Path(args.out), len(problems), total_ms, args.model)
    passed = sum(1 for r in results if r.pass_at_1)
    print(f"[llm_only] wrote {args.out} — pass@1 {passed}/{len(problems)} in {total_ms/1000:.1f}s")


if __name__ == "__main__":
    main()
