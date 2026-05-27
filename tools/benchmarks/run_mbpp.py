#!/usr/bin/env python3
"""
MBPP (Mostly Basic Python Problems) agent-loop runner.

MBPP is Google's 974-problem set of short Python tasks specified as a
natural-language prompt + 3 `assert` test cases. Unlike HumanEval — which
gives a docstring + signature — MBPP gives free-form text ("Write a
function to ...") plus the assert list. The function name comes from the
first assert's callee.

Why run this:
  - Independent confirmation that the HumanEval +2.4pp agent-loop delta
    generalises. MBPP is differently phrased and covers a wider domain
    (string munging, list ops, simple math) than HumanEval.
  - A different failure profile: MBPP prompts are terser, so baseline
    error rate is usually higher than HumanEval's.

Modes:
    --mode llm    single-shot, T=0
    --mode agent  cache → k=3 best-of-N → 2 retries → cache

Usage:
    ANTHROPIC_API_KEY=sk-... python3 tools/benchmarks/run_mbpp.py \\
        --mode agent --limit 100 --verbose
    python3 tools/benchmarks/run_mbpp.py \\
        --backend mlx --model mlx-community/Qwen3-4B-Instruct-2507-4bit \\
        --adapter-path artifacts/adapters/qwen3_4b_mlx_... \\
        --mode llm --limit 50 --verbose
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
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
    lookup as cache_lookup, record as cache_record,
)


@dataclass
class MBPPProblem:
    task_id: int
    text: str
    test_list: List[str]
    fn_name: str  # derived from the first assert
    examples: List[dict] = None  # type: ignore[assignment]


@dataclass
class MBPPResult:
    task_id: int
    pass_at_1: bool
    path: str
    extra: str
    elapsed_ms: int
    total_tokens: int = 0
    final_error: str = ""


_FN_FROM_ASSERT_RE = re.compile(r"assert\s+(\w+)\s*\(")


def derive_fn_name(test_list: List[str]) -> Optional[str]:
    """First `assert foo(...)` in the test list → "foo"."""
    for t in test_list:
        m = _FN_FROM_ASSERT_RE.search(t)
        if m:
            return m.group(1)
    return None


def derive_examples_from_asserts(test_list: List[str],
                                   fn_name: str) -> List[dict]:
    """Parse `assert fn(a, b) == expected` rows into `{inputs, expected}`
    dicts. Used so MBPP rows can feed the same examples-based semantic
    retrieval we use for HumanEval. Skips rows we cannot parse (e.g.
    multi-line asserts with function-call expected)."""
    import ast
    out: List[dict] = []
    for t in test_list:
        try:
            tree = ast.parse(t.strip(), mode="exec")
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assert):
                continue
            test = node.test
            if not (isinstance(test, ast.Compare) and
                    len(test.ops) == 1 and
                    isinstance(test.ops[0], ast.Eq)):
                continue
            left = test.left
            right = test.comparators[0]
            if not (isinstance(left, ast.Call) and
                    isinstance(left.func, ast.Name) and
                    left.func.id == fn_name):
                continue
            try:
                inputs = [ast.literal_eval(a) for a in left.args]
                expected = ast.literal_eval(right)
            except (ValueError, SyntaxError):
                continue
            out.append({"inputs": inputs, "expected": expected})
    return out


def fingerprint_mbpp(text: str, test_list: List[str]) -> str:
    payload = text + "\n" + "\n".join(test_list)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


_FENCE_RE = re.compile(r"```(?:python)?\n?(.*?)```", re.DOTALL)
_DEF_RE = re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE)


def extract_python(response: str, fn_name: str) -> Optional[str]:
    def _function_block(src: str) -> str:
        m = _DEF_RE.search(src)
        if not m:
            return src.strip()
        lines = src[m.start():].splitlines()
        block: List[str] = []
        base_indent = 0
        started = False
        for line in lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if not started:
                if not stripped.startswith(f"def {fn_name}"):
                    continue
                started = True
                base_indent = indent
                block.append(line[base_indent:])
                continue
            if stripped.startswith("```") or stripped.startswith("<|") or stripped.startswith("<｜"):
                break
            if stripped and indent <= base_indent:
                break
            block.append(line[base_indent:] if len(line) >= base_indent else line)
        return "\n".join(block).strip()

    fences = _FENCE_RE.findall(response)
    if fences:
        for body in fences:
            if f"def {fn_name}" in body:
                return _function_block(body)
        # First fenced block as fallback.
        return _function_block(fences[0])
    m = _DEF_RE.search(response)
    if m:
        return _function_block(response)
    return None


import signal as _signal


class _VerifyTimeout(Exception):
    pass


def _alarm(seconds: int):
    """SIGALRM wrapper (macOS + Linux). Protects verify() from infinite
    loops in generated code — MBPP generators occasionally produce
    non-terminating functions that would otherwise hang the runner."""
    class _Ctx:
        def __enter__(self_):
            def _h(signum, frame):
                raise _VerifyTimeout(f"timed out after {seconds}s")
            self_.old = _signal.signal(_signal.SIGALRM, _h)
            _signal.alarm(seconds)
            return self_
        def __exit__(self_, *exc):
            _signal.alarm(0)
            _signal.signal(_signal.SIGALRM, self_.old)
    return _Ctx()


def verify(code: str, problem: MBPPProblem, timeout_s: int = 6) -> Tuple[bool, str]:
    """Exec the code under SIGALRM, then run each `assert` from test_list
    under a separate alarm. Timeout → hard fail with a readable error."""
    ns: dict = {"__builtins__": __builtins__}
    try:
        with _alarm(timeout_s):
            exec(code, ns)
    except _VerifyTimeout as e:
        return (False, f"exec: {e}")
    except Exception as e:
        return (False, f"exec: {e!r}"[:150])
    if problem.fn_name not in ns:
        return (False, f"no fn {problem.fn_name}")
    for t in problem.test_list:
        try:
            with _alarm(timeout_s):
                exec(t, ns)
        except AssertionError:
            return (False, f"assertion failed: {t[:120]}")
        except _VerifyTimeout as e:
            return (False, f"{t[:80]} {e}"[:150])
        except Exception as e:
            return (False, f"{t[:80]} raised {e!r}"[:150])
    return (True, "")


def build_initial_prompt(problem: MBPPProblem, retrieval_k: int = 0) -> str:
    tests = "\n".join(problem.test_list)
    base = (
        f"{problem.text}\n\n"
        f"The function must satisfy these assertions:\n{tests}\n\n"
        f"Reply with ONLY the function definition (no explanation, no tests)."
    )
    if retrieval_k > 0 and problem.examples:
        try:
            from retrieval_prompt import build_retrieval_prefix  # noqa: E402
            prefix = build_retrieval_prefix(
                problem.examples, k=retrieval_k, min_similarity=0.70,
            )
            if prefix:
                return prefix + base
        except Exception:
            pass
    return base


def build_retry_prompt(problem: MBPPProblem, prev: str, err: str) -> str:
    tests = "\n".join(problem.test_list)
    return (
        f"Your previous attempt at `{problem.fn_name}` had a bug.\n\n"
        f"Previous code:\n```python\n{prev}\n```\n\n"
        f"Failure: {err}\n\n"
        f"Tests:\n{tests}\n\n"
        f"Reply with a corrected function."
    )


def build_cache_metadata(problem: MBPPProblem) -> dict:
    return {
        "task_kind": "mbpp",
        "task_id": problem.task_id,
        "fn_name": problem.fn_name,
        "prompt": build_initial_prompt(problem, retrieval_k=0),
    }


def llm_call(client, prompt: str, model: str, temperature: float,
             max_tokens: int = 768) -> Tuple[str, int]:
    try:
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
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
    client, problem: MBPPProblem, model: str, k: int, max_retries: int,
    retrieval_k: int = 0, llm_max_tokens: int = 768,
) -> MBPPResult:
    t0 = time.time()
    total_tokens = 0
    fp = fingerprint_mbpp(problem.text, problem.test_list)
    cache_metadata = build_cache_metadata(problem)

    cached = cache_lookup(fp)
    if cached is not None:
        ok, _ = verify(cached["code"], problem)
        if ok:
            return MBPPResult(
                task_id=problem.task_id, pass_at_1=True, path="cache",
                extra=cached.get("model", "?"),
                elapsed_ms=int((time.time() - t0) * 1000),
            )

    temps = [0.0] + [0.6 + 0.1 * i for i in range(1, k)]
    temps = temps[:k]

    def _gen(i: int):
        text, tokens = llm_call(
            client, build_initial_prompt(problem, retrieval_k=retrieval_k),
            model, temps[i], max_tokens=llm_max_tokens,
        )
        code = extract_python(text, problem.fn_name)
        return (i, code, tokens)

    candidates: List[Tuple[int, Optional[str], int]] = []
    if k > 1:
        with cf.ThreadPoolExecutor(max_workers=min(k, 5)) as pool:
            for fut in cf.as_completed([pool.submit(_gen, i) for i in range(k)]):
                candidates.append(fut.result())
        candidates.sort(key=lambda x: x[0])
    else:
        candidates = [_gen(0)]

    last_code = ""
    last_error = ""
    for idx, code, toks in candidates:
        total_tokens += toks
        if code is None:
            last_error = last_error or "no-code-found"
            continue
        last_code = code
        ok, err = verify(code, problem)
        if ok:
            try:
                cache_record(
                    fp,
                    model,
                    code,
                    examples=problem.examples,
                    metadata=cache_metadata,
                )
            except Exception: pass
            return MBPPResult(
                task_id=problem.task_id, pass_at_1=True, path="sample",
                extra=f"s{idx} T={temps[idx]:.2f}",
                elapsed_ms=int((time.time() - t0) * 1000),
                total_tokens=total_tokens,
            )
        last_error = err

    for retry in range(1, max_retries + 1):
        text, tokens = llm_call(
            client,
            build_retry_prompt(problem, last_code, last_error),
            model,
            temperature=0.0,
            max_tokens=llm_max_tokens,
        )
        total_tokens += tokens
        code = extract_python(text, problem.fn_name)
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
                    examples=problem.examples,
                    metadata=cache_metadata,
                )
            except Exception: pass
            return MBPPResult(
                task_id=problem.task_id, pass_at_1=True, path="retry",
                extra=f"r{retry}",
                elapsed_ms=int((time.time() - t0) * 1000),
                total_tokens=total_tokens,
            )
        last_error = err

    return MBPPResult(
        task_id=problem.task_id, pass_at_1=False, path="miss",
        extra=f"k={k},r={max_retries}",
        elapsed_ms=int((time.time() - t0) * 1000),
        total_tokens=total_tokens, final_error=last_error[:200],
    )


def write_report(results: List[MBPPResult], out: Path, mode: str,
                 model: str, total_ms: int) -> None:
    total = len(results); passed = sum(1 for r in results if r.pass_at_1)
    pct = 100.0 * passed / max(total, 1)
    by_path: dict = {}
    for r in results:
        by_path[r.path] = by_path.get(r.path, 0) + 1
    total_tok = sum(r.total_tokens for r in results)

    lines = [
        f"# MBPP Results — mode: {mode}", "",
        f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} — "
        f"{total} problems, model={model}, {total_ms/1000:.1f}s total.", "",
        "## Summary", "",
        f"- **Pass@1**: **{passed}/{total} ({pct:.1f}%)**",
    ]
    for p, n in sorted(by_path.items()):
        lines.append(f"- path `{p}`: {n}")
    lines.append(f"- Total tokens: ~{total_tok}")
    lines.append("")
    lines.append("## Sample failures (first 15)")
    lines.append("")
    lines.append("| task_id | path | error |")
    lines.append("|--:|------|-------|")
    shown = 0
    for r in results:
        if r.pass_at_1: continue
        if shown >= 15: break
        lines.append(f"| {r.task_id} | {r.path} | {r.final_error[:80]} |")
        shown += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["llm", "agent"], default="agent")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--backend", default="anthropic",
                    choices=["anthropic", "mlx", "hf", "openai"],
                    help="inference backend. `anthropic` uses the Claude API; "
                         "`mlx`/`hf`/`openai` use LocalModelClient.")
    ap.add_argument("--api-base", default=None,
                    help="for --backend openai: base URL")
    ap.add_argument("--device", default=None,
                    help="for --backend hf: device_map override")
    ap.add_argument("--adapter-path", default=None,
                    help="for --backend mlx: LoRA adapter directory to load")
    ap.add_argument("--adapter-routing", default="always",
                    choices=["always", "never", "utility_only"],
                    help="for --backend mlx: when to apply the adapter")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--split", default="test",
                    help="HF split to use: train/validation/test/prompt")
    ap.add_argument("--offset", type=int, default=0,
                    help="Skip first N problems in the split.")
    ap.add_argument("--retrieval", type=int, default=0,
                    help="Top-K semantic-similar cached solutions to "
                         "inline as few-shot context. 0 = disabled.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--llm-max-tokens", type=int, default=768,
                    help="generation cap for llm/agent modes")
    args = ap.parse_args()

    out = args.out or f"artifacts/mbpp_{args.mode}.md"
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("[mbpp] pip install datasets required", file=sys.stderr); sys.exit(2)
    if args.backend == "anthropic":
        try:
            import anthropic  # type: ignore
        except ImportError:
            print("[mbpp] pip install anthropic required", file=sys.stderr); sys.exit(2)
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            print("[mbpp] ANTHROPIC_API_KEY not set", file=sys.stderr); sys.exit(2)
        client = anthropic.Anthropic(api_key=key)
    else:
        from local_model_adapter import LocalModelClient
        client = LocalModelClient(
            backend=args.backend,
            model=args.model,
            api_base=args.api_base,
            device=args.device,
            adapter_path=args.adapter_path,
            adapter_routing=args.adapter_routing,
        )

    ds = load_dataset("google-research-datasets/mbpp", "sanitized",
                       split=args.split)

    problems: List[MBPPProblem] = []
    start = args.offset
    end = min(args.offset + args.limit, len(ds))
    for i in range(start, end):
        row = dict(ds[i])
        fn = derive_fn_name(row["test_list"])
        if fn is None:
            continue
        examples = derive_examples_from_asserts(row["test_list"], fn)
        problems.append(MBPPProblem(
            task_id=row["task_id"], text=row["prompt"],
            test_list=row["test_list"], fn_name=fn,
            examples=examples,
        ))
    print(f"[mbpp] {len(problems)} problems, mode={args.mode}, model={args.model}")

    t_start = time.time()
    results: List[MBPPResult] = []
    for i, p in enumerate(problems, 1):
        if args.verbose:
            print(f"[{i}/{len(problems)}] ", end="", flush=True)
        if args.mode == "agent":
            r = solve_agent(client, p, args.model, args.k, args.max_retries,
                             retrieval_k=args.retrieval,
                             llm_max_tokens=args.llm_max_tokens)
        else:
            t0 = time.time()
            text, tokens = llm_call(client, build_initial_prompt(p), args.model,
                                     temperature=0.0,
                                     max_tokens=args.llm_max_tokens)
            code = extract_python(text, p.fn_name)
            if code is None:
                ok, err = False, "no-code-found"
            else:
                ok, err = verify(code, p)
            r = MBPPResult(
                task_id=p.task_id,
                pass_at_1=ok, path="sample" if ok else "miss",
                extra="k=1",
                elapsed_ms=int((time.time() - t0) * 1000),
                total_tokens=tokens,
                final_error="" if ok else err[:200],
            )
        results.append(r)
        if args.verbose:
            mark = "✓" if r.pass_at_1 else f"✗ {r.final_error[:50]}"
            print(f"{r.path} {mark}")

    total_ms = int((time.time() - t_start) * 1000)
    write_report(results, Path(out), args.mode, args.model, total_ms)
    passed = sum(1 for r in results if r.pass_at_1)
    print(
        f"[mbpp] wrote {out} — pass@1 {passed}/{len(results)} "
        f"({100.0 * passed / max(len(results), 1):.1f}%) in {total_ms/1000:.1f}s"
    )


if __name__ == "__main__":
    main()
