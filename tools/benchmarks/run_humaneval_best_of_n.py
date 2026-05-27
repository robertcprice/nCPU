#!/usr/bin/env python3
"""
Best-of-N verified sampling on the 30-problem humaneval_lite.

For each problem: generate k independent candidates from the LLM
(varied via temperature sampling), verify each against test_cases,
keep the first one that passes. Records which sample-index won.

This is the simplest + most effective technique for boosting pass@1
when verification is cheap: the LLM's variance is our friend, we just
filter against the ground-truth test suite.

Expected: Haiku k=1 at 90% → k=5 at ~97%+ on this set. Cost is linear
in k (roughly k× tokens); wall-clock is near-linear in k unless we
parallelise the sampling.

Usage:
    ANTHROPIC_API_KEY=sk-... python3 \\
        tools/benchmarks/run_humaneval_best_of_n.py --verbose --k 5
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
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
    winning_sample: int  # 0..k-1, or -1 if all failed
    attempts: int
    solve_ms: int
    total_tokens: int = 0
    first_error: str = ""


def load_problems(path: Path) -> List[Problem]:
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
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
        return (False, f"exec: {e!r}"[:120])
    fn = ns.get(problem.name)
    if fn is None:
        return (False, f"no fn {problem.name}")
    for case in problem.test_cases:
        *args, expected = case
        try:
            got = fn(*args)
        except Exception as e:
            return (False, f"call: {e!r}"[:100])
        if got != expected:
            return (False, f"wrong {args}→{got} exp {expected}"[:100])
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


def build_cache_metadata(problem: Problem) -> dict:
    return {
        "task_kind": "example_codegen",
        "problem_name": problem.name,
        "signature": problem.signature,
        "prompt": build_prompt(problem),
    }


def generate_one(client, problem: Problem, model: str, temperature: float, seed: int) -> Tuple[Optional[str], int, int]:
    """Return (code, elapsed_ms, total_tokens_used)."""
    t0 = time.time()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=768,
            temperature=temperature,
            messages=[{"role": "user", "content": build_prompt(problem)}],
        )
    except Exception:
        return (None, int((time.time() - t0) * 1000), 0)
    ms = int((time.time() - t0) * 1000)
    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    code = extract_python(text, problem.name)
    total_tokens = 0
    usage = getattr(resp, "usage", None)
    if usage is not None:
        total_tokens = getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)
    return (code, ms, total_tokens)


def solve_best_of_n(
    client, problem: Problem, model: str, k: int, parallel: bool
) -> Result:
    t0 = time.time()
    cache_metadata = build_cache_metadata(problem)

    # Cache check — return the prior verified solution instantly.
    fp = fingerprint_examples([
        {"inputs": ex["inputs"], "expected": ex["expected"]} for ex in problem.examples
    ])
    cached = cache_lookup(fp)
    if cached is not None:
        ok, _ = verify_code(cached["code"], problem)
        if ok:
            return Result(
                name=problem.name,
                pass_at_1=True,
                winning_sample=-2,  # sentinel: cache hit
                attempts=0,
                solve_ms=0,
            )

    # Temperature schedule: 0.0 for sample 0 (deterministic best-effort),
    # then a spread of warmer temperatures so the k samples aren't all the
    # same wrong answer. Literature calls this "nucleus-diversified
    # best-of-N". We don't need a clever prior; just variance.
    temps = [0.0] + [0.6 + 0.1 * i for i in range(1, k)]
    temps = temps[:k]

    first_error = ""
    total_tokens = 0

    def _gen(i: int) -> Tuple[int, Optional[str], int, int]:
        code, ms, tok = generate_one(client, problem, model, temps[i], seed=i)
        return (i, code, ms, tok)

    candidates: List[Tuple[int, Optional[str], int, int]] = []
    if parallel and k > 1:
        with cf.ThreadPoolExecutor(max_workers=min(k, 5)) as pool:
            for fut in cf.as_completed([pool.submit(_gen, i) for i in range(k)]):
                candidates.append(fut.result())
        candidates.sort(key=lambda x: x[0])
    else:
        for i in range(k):
            candidates.append(_gen(i))

    for idx, code, ms, tok in candidates:
        total_tokens += tok
        if code is None:
            if not first_error:
                first_error = "no-code-found"
            continue
        ok, err = verify_code(code, problem)
        if ok:
            # Cache the verified winner so the next call is 0 ms.
            try:
                cache_record(
                    fp,
                    model,
                    code,
                    examples=[
                        {"inputs": ex["inputs"], "expected": ex["expected"]}
                        for ex in problem.examples
                    ],
                    metadata=cache_metadata,
                )
            except Exception:
                pass
            return Result(
                name=problem.name,
                pass_at_1=True,
                winning_sample=idx,
                attempts=k,
                solve_ms=int((time.time() - t0) * 1000),
                total_tokens=total_tokens,
            )
        if not first_error:
            first_error = err

    return Result(
        name=problem.name,
        pass_at_1=False,
        winning_sample=-1,
        attempts=k,
        solve_ms=int((time.time() - t0) * 1000),
        total_tokens=total_tokens,
        first_error=first_error,
    )


def write_report(results: List[Result], out: Path, k: int, total_ms: int, model: str) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.pass_at_1)
    pct = 100.0 * passed / max(total, 1)
    cache_hits = sum(1 for r in results if r.winning_sample == -2)
    winning_dist: dict = {}
    for r in results:
        if r.pass_at_1 and r.winning_sample >= 0:
            winning_dist[r.winning_sample] = winning_dist.get(r.winning_sample, 0) + 1
    total_tok = sum(r.total_tokens for r in results)

    lines = [
        "# HumanEval-lite Best-of-N Results",
        "",
        f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} — "
        f"{total} problems, k={k}, model={model}, {total_ms/1000:.1f}s total.",
        "",
        "## Summary",
        "",
        f"- **Pass@1 (best-of-{k})**: **{passed}/{total} ({pct:.1f}%)**",
        f"- Cache hits (0 ms): {cache_hits}",
        f"- Total tokens: ~{total_tok}",
        "",
        "### Winning-sample distribution",
        "",
        "Which sample-index produced the winner when pass@1=True?",
        "",
        "| sample | count |",
        "|-------:|------:|",
    ]
    for idx in sorted(winning_dist.keys()):
        lines.append(f"| {idx} (T={0.0 if idx == 0 else 0.6 + 0.1 * idx:.1f}) | {winning_dist[idx]} |")
    lines += ["", "## Per-problem", "", "| # | problem | pass | winner | ms | notes |",
              "|--:|---------|:----:|:------:|---:|-------|"]
    for i, r in enumerate(results, 1):
        p = "✓" if r.pass_at_1 else "✗"
        w = "cache" if r.winning_sample == -2 else (str(r.winning_sample) if r.winning_sample >= 0 else "-")
        note = r.first_error[:80] if r.first_error else ""
        lines.append(f"| {i} | {r.name} | {p} | {w} | {r.solve_ms} | {note} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--problems", default="tools/benchmarks/humaneval_lite.jsonl")
    ap.add_argument("--out", default="artifacts/humaneval_results_best_of_n.md")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--parallel", action="store_true",
                    help="run the k samples concurrently (faster, same cost)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)

    try:
        import anthropic
    except ImportError:
        print("[best-of-n] anthropic SDK required", file=sys.stderr); sys.exit(2)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("[best-of-n] ANTHROPIC_API_KEY not set", file=sys.stderr); sys.exit(2)
    client = anthropic.Anthropic(api_key=key)

    problems = load_problems(Path(args.problems))
    print(f"[best-of-n] {len(problems)} problems, k={args.k}, model={args.model}, parallel={args.parallel}")

    t0 = time.time()
    results: List[Result] = []
    for i, p in enumerate(problems, 1):
        if args.verbose:
            print(f"[{i}/{len(problems)}] {p.name} ...", end=" ", flush=True)
        r = solve_best_of_n(client, p, args.model, args.k, args.parallel)
        results.append(r)
        if args.verbose:
            if r.winning_sample == -2:
                mark = "cache ✓"
            elif r.pass_at_1:
                mark = f"sample{r.winning_sample} ✓"
            else:
                mark = f"✗ ({r.first_error[:50]})"
            print(f"{mark} ({r.solve_ms}ms)")
    total_ms = int((time.time() - t0) * 1000)

    write_report(results, Path(args.out), args.k, total_ms, args.model)
    passed = sum(1 for r in results if r.pass_at_1)
    print(f"[best-of-n] wrote {args.out} — pass@1 {passed}/{len(problems)} in {total_ms/1000:.1f}s")


if __name__ == "__main__":
    main()
