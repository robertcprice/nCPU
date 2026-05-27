#!/usr/bin/env python3
"""
Full agent-loop runner: cache → best-of-N → retry-with-feedback → cache.

The production pattern for verified LLM code generation:

  1. Lookup the problem fingerprint in the LLM cache. Hit → 0 ms, done.
  2. Generate k=N candidates with temperature-spread sampling.
  3. Verify each candidate against test_cases. First pass wins.
  4. If all k miss, run a retry loop: prompt re-prompts with the specific
     failure message. Up to M retries.
  5. On any success, cache the winning code.

This is what Claude Code / Cursor Composer / competitive-programming
agents actually do. We measure its pass@1 ceiling on the 30-problem
set.

Usage:
    ANTHROPIC_API_KEY=sk-... python3 \\
        tools/benchmarks/run_humaneval_agent.py --verbose --k 3 --max-retries 2
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
from llm_solution_cache import (  # noqa: E402
    fingerprint_examples, lookup as cache_lookup, record as cache_record,
)


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
    path: str  # "cache" | "sample" | "retry" | "miss"
    extra: str  # sample-idx, retry-count, etc.
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


def verify(code: str, problem: Problem) -> Tuple[bool, str]:
    ns: dict = {}
    try:
        exec(code, ns)
    except Exception as e:
        return (False, f"exec: {e!r}"[:150])
    fn = ns.get(problem.name)
    if fn is None:
        return (False, f"no fn {problem.name}")
    for case in problem.test_cases:
        *args, expected = case
        try:
            got = fn(*args)
        except Exception as e:
            return (False, f"call({args}): {e!r}"[:120])
        if got != expected:
            return (
                False,
                f"{problem.name}({', '.join(repr(a) for a in args)}) returned "
                f"{got!r}, expected {expected!r}",
            )
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


def build_initial_prompt(problem: Problem, retrieval_k: int = 0) -> str:
    examples_str = "\n".join(
        f"  {problem.name}({', '.join(repr(x) for x in ex['inputs'])}) == {ex['expected']}"
        for ex in problem.examples
    )
    base = (
        f"Write a Python function matching the signature `{problem.signature}`.\n\n"
        f"Satisfy these examples exactly:\n{examples_str}\n\n"
        f"Reply with ONLY the function definition."
    )
    if retrieval_k > 0:
        try:
            from retrieval_prompt import build_retrieval_prefix  # noqa: E402
            prefix = build_retrieval_prefix(
                [{"inputs": ex["inputs"], "expected": ex["expected"]}
                 for ex in problem.examples],
                k=retrieval_k, min_similarity=0.70,
            )
            if prefix:
                return prefix + base
        except Exception:
            pass
    return base


def build_retry_prompt(problem: Problem, previous_code: str, error: str) -> str:
    examples_str = "\n".join(
        f"  {problem.name}({', '.join(repr(x) for x in ex['inputs'])}) == {ex['expected']}"
        for ex in problem.examples
    )
    return (
        f"Your previous `{problem.signature}` had a bug.\n\n"
        f"Previous code:\n```python\n{previous_code}\n```\n\n"
        f"Failure: {error}\n\n"
        f"Examples:\n{examples_str}\n\nReply with a corrected function."
    )


def build_cache_metadata(problem: Problem) -> dict:
    return {
        "task_kind": "example_codegen",
        "problem_name": problem.name,
        "signature": problem.signature,
        "prompt": build_initial_prompt(problem, retrieval_k=0),
    }


def llm_call(client, prompt: str, model: str, temperature: float) -> Tuple[str, int]:
    try:
        resp = client.messages.create(
            model=model, max_tokens=768, temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:
        return ("", 0)
    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    usage = getattr(resp, "usage", None)
    tokens = (getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)
              if usage else 0)
    return (text, tokens)


def solve_agent(
    client, problem: Problem, model: str, k: int, max_retries: int,
    retrieval_k: int = 0,
) -> Result:
    t0 = time.time()
    total_tokens = 0
    cache_metadata = build_cache_metadata(problem)

    # 1. Cache lookup.
    fp = fingerprint_examples([
        {"inputs": ex["inputs"], "expected": ex["expected"]} for ex in problem.examples
    ])
    cached = cache_lookup(fp)
    if cached is not None:
        ok, _ = verify(cached["code"], problem)
        if ok:
            return Result(
                name=problem.name, pass_at_1=True, path="cache",
                extra=cached.get("model", "?"),
                solve_ms=0, total_tokens=0,
            )

    # 2. Best-of-k sampling.
    temps = [0.0] + [0.6 + 0.1 * i for i in range(1, k)]
    temps = temps[:k]

    def _gen(i: int) -> Tuple[int, Optional[str], int, str]:
        prompt = build_initial_prompt(problem, retrieval_k=retrieval_k)
        text, tokens = llm_call(client, prompt, model, temps[i])
        code = extract_python(text, problem.name)
        return (i, code, tokens, text)

    candidates: List[Tuple[int, Optional[str], int, str]] = []
    if k > 1:
        with cf.ThreadPoolExecutor(max_workers=min(k, 5)) as pool:
            for fut in cf.as_completed([pool.submit(_gen, i) for i in range(k)]):
                candidates.append(fut.result())
        candidates.sort(key=lambda x: x[0])
    else:
        candidates = [_gen(0)]

    last_code = ""
    last_error = ""
    for idx, code, toks, _text in candidates:
        total_tokens += toks
        if code is None:
            if not last_error:
                last_error = "no-code-found"
            continue
        last_code = code
        ok, err = verify(code, problem)
        if ok:
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
            except Exception: pass
            return Result(
                name=problem.name, pass_at_1=True, path="sample",
                extra=f"s{idx} (T={temps[idx]:.1f})",
                solve_ms=int((time.time() - t0) * 1000),
                total_tokens=total_tokens,
            )
        last_error = err

    # 3. Retry loop — re-prompt with the specific failure.
    for retry in range(1, max_retries + 1):
        prompt = build_retry_prompt(problem, last_code, last_error)
        text, tokens = llm_call(client, prompt, model, temperature=0.0)
        total_tokens += tokens
        code = extract_python(text, problem.name)
        if code is None:
            last_error = "no-code-found"
            continue
        last_code = code
        ok, err = verify(code, problem)
        if ok:
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
            except Exception: pass
            return Result(
                name=problem.name, pass_at_1=True, path="retry",
                extra=f"r{retry}",
                solve_ms=int((time.time() - t0) * 1000),
                total_tokens=total_tokens,
            )
        last_error = err

    return Result(
        name=problem.name, pass_at_1=False, path="miss",
        extra=f"k={k},r={max_retries}",
        solve_ms=int((time.time() - t0) * 1000),
        total_tokens=total_tokens, final_error=last_error,
    )


def write_report(results: List[Result], out: Path, k: int, max_retries: int,
                 total_ms: int, model: str) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.pass_at_1)
    pct = 100.0 * passed / max(total, 1)
    path_counts: dict = {}
    for r in results:
        path_counts[r.path] = path_counts.get(r.path, 0) + 1
    total_tok = sum(r.total_tokens for r in results)

    lines = [
        "# HumanEval-lite Agent Loop Results", "",
        f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} — "
        f"{total} problems, k={k}, max_retries={max_retries}, model={model}, "
        f"{total_ms/1000:.1f}s total.", "",
        "## Summary", "",
        f"- **Pass@1 (cache → best-of-{k} → {max_retries} retries)**: "
        f"**{passed}/{total} ({pct:.1f}%)**",
        f"- Path distribution:",
    ]
    for p, n in sorted(path_counts.items()):
        lines.append(f"  - {p}: {n}")
    lines += [f"- Total tokens: ~{total_tok}", "",
              "## Per-problem", "",
              "| # | problem | pass | path | detail | ms | notes |",
              "|--:|---------|:----:|------|--------|---:|-------|"]
    for i, r in enumerate(results, 1):
        p = "✓" if r.pass_at_1 else "✗"
        note = r.final_error[:80] if r.final_error else ""
        lines.append(f"| {i} | {r.name} | {p} | {r.path} | {r.extra} | {r.solve_ms} | {note} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--problems", default="tools/benchmarks/humaneval_lite.jsonl")
    ap.add_argument("--out", default="artifacts/humaneval_results_agent.md")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--backend", default="anthropic",
                    choices=["anthropic", "mlx", "hf", "openai"],
                    help="inference backend. `anthropic` uses the Claude API; "
                         "`mlx`/`hf`/`openai` use LocalModelClient from "
                         "local_model_adapter.py (Qwen3.5, Gemma 4, etc.)")
    ap.add_argument("--api-base", default=None,
                    help="for --backend openai: base URL (vLLM / llama.cpp / Ollama)")
    ap.add_argument("--device", default=None,
                    help="for --backend hf: device_map override")
    ap.add_argument("--adapter-path", default=None,
                    help="for --backend mlx: LoRA adapter directory to load")
    ap.add_argument("--adapter-routing", default="always",
                    choices=["always", "never", "utility_only"],
                    help="for --backend mlx: when to apply the adapter")
    ap.add_argument("--retrieval", type=int, default=0,
                    help="Top-K semantic-similar cached solutions to "
                         "inline as few-shot context before generation. "
                         "0 = disabled (default).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)

    if args.backend == "anthropic":
        try:
            import anthropic
        except ImportError:
            print("[agent] anthropic SDK required", file=sys.stderr); sys.exit(2)
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            print("[agent] ANTHROPIC_API_KEY not set", file=sys.stderr); sys.exit(2)
        client = anthropic.Anthropic(api_key=key)
    else:
        # Local backend — swap in the adapter. Keeps the rest of the
        # runner unchanged; that's the point of the adapter.
        from local_model_adapter import LocalModelClient
        client = LocalModelClient(
            backend=args.backend,
            model=args.model,
            api_base=args.api_base,
            device=args.device,
            adapter_path=args.adapter_path,
            adapter_routing=args.adapter_routing,
        )

    problems = load_problems(Path(args.problems))
    print(f"[agent] {len(problems)} problems, k={args.k}, "
          f"max_retries={args.max_retries}, model={args.model}")

    t0 = time.time()
    results: List[Result] = []
    for i, p in enumerate(problems, 1):
        if args.verbose:
            print(f"[{i}/{len(problems)}] {p.name} ...", end=" ", flush=True)
        r = solve_agent(client, p, args.model, args.k, args.max_retries,
                         retrieval_k=args.retrieval)
        results.append(r)
        if args.verbose:
            mark = "✓" if r.pass_at_1 else f"✗ ({r.final_error[:50]})"
            print(f"{r.path} {mark} [{r.extra}] ({r.solve_ms}ms)")
    total_ms = int((time.time() - t0) * 1000)

    write_report(results, Path(args.out), args.k, args.max_retries, total_ms, args.model)
    passed = sum(1 for r in results if r.pass_at_1)
    print(f"[agent] wrote {args.out} — pass@1 {passed}/{len(problems)} in {total_ms/1000:.1f}s")


if __name__ == "__main__":
    main()
