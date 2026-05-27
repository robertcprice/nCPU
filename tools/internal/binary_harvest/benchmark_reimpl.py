#!/usr/bin/env python3
"""
Benchmark: does harvest data actually help LLMs reimplement utilities?

Signal test before we spend on LoRA fine-tuning. We generate a
held-out set of probes (unseen by the harvester), ask Haiku to
emit `def solve(stdin: str) -> str:` given the utility name +
one I/O example, execute the response, and measure pass rate
byte-equal against the real binary.

Two conditions:
  - `--mode baseline` — prompt Haiku cold (no retrieval).
  - `--mode few-shot` — prepend k similar past solve()s from the
    harvest cache as reference examples.

If the few-shot version beats baseline, the harvest data has
genuine learning signal. That's the go/no-go for the full
distillation run.

Usage:
    python3 tools/binary_harvest/benchmark_reimpl.py \\
        --n 5 --cache /path/to/harvest.tsv --mode baseline
    python3 tools/binary_harvest/benchmark_reimpl.py \\
        --n 5 --cache /path/to/harvest.tsv --mode few-shot --k 3
    python3 tools/binary_harvest/benchmark_reimpl.py \\
        --backend mlx --model mlx-community/Qwen3-4B-Instruct-2507-4bit \\
        --adapter-path artifacts/adapters/qwen3_4b_mlx_... \\
        --n 3 --mode baseline
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harvest import TOOLS as HARVEST_TOOLS, reference_python  # noqa: E402

# The fuzzers in fuzz.py produce aggressive random inputs — perfect
# for held-out probes since the harvester never saw them.
from fuzz import FUZZERS  # noqa: E402


BENCH_TOOLS = ["sort", "uniq", "wc", "head", "tail", "cut", "grep",
                "jq", "base64", "sha256sum", "md5sum", "tr", "rev", "fold"]


def _build_prompt(tool: str, stdin: str, expected: str,
                   args: Tuple[str, ...],
                   few_shot_k: int = 0) -> str:
    """Construct the reimplementation prompt. In few-shot mode we
    include k similar solved cases from the cache as examples."""
    base = (
        f"Reimplement the Unix utility `{tool}` with flags {list(args)} "
        f"in Python. Given the stdin below, your `solve(stdin: str) -> str` "
        f"function must return exactly the stdout that `{tool}` produces.\n\n"
        f"Input:\n```\n{stdin[:600]}\n```\n\n"
        f"Expected output:\n```\n{expected[:600]}\n```\n\n"
        f"Reply with ONLY the function definition. No explanation, "
        f"no test cases. Start with `def solve(stdin: str) -> str:`."
    )
    if few_shot_k <= 0:
        return base

    # Few-shot: inline k other verified solutions from the harvest
    # cache (for a DIFFERENT tool — so the model generalises pattern,
    # not copies exactly).
    few_shots = []
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
    from llm_solution_cache import _load_all
    rows = _load_all()
    # Prefer rows from different tools than the one we're testing.
    other_rows = [(fp, r) for fp, r in rows.items()
                   if r.get("model", "").startswith("binary:")
                   and not r["model"].endswith(f":{tool}")]
    # Seeded sample for determinism.
    rng = random.Random(42 + hash(tool) % 1000)
    if len(other_rows) > few_shot_k:
        other_rows = rng.sample(other_rows, few_shot_k)
    for fp, r in other_rows:
        ex = r.get("examples", [])
        if not ex:
            continue
        ex_stdin = ex[0]["inputs"][0]
        ex_out = ex[0]["expected"]
        ex_tool = r["model"].split(":", 1)[1]
        few_shots.append(
            f"--- Example: reimplementing `{ex_tool}` ---\n"
            f"Input:\n```\n{ex_stdin[:200]}\n```\n"
            f"Expected:\n```\n{ex_out[:200]}\n```\n"
            f"Solution:\n```python\n{r['code']}\n```"
        )
    prefix = (
        "# Reference solutions from similar tasks:\n\n"
        + "\n\n".join(few_shots)
        + "\n\n# Your task (DIFFERENT utility, write a fresh solution):\n\n"
    )
    return prefix + base


def _extract_solve(text: str) -> str:
    """Extract a `def solve(...)` block from LLM output. Strips fences."""
    import re as _re

    def _function_block(src: str) -> str:
        m = _re.search(r"^\s*def solve\s*\(", src, _re.MULTILINE)
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
                if not stripped.startswith("def solve"):
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

    fences = _re.findall(r"```(?:python)?\n?(.*?)```", text, _re.DOTALL)
    for body in fences:
        if "def solve" in body:
            return _function_block(body)
    m = _re.search(r"^\s*def solve", text, _re.MULTILINE)
    if m:
        return _function_block(text)
    return text.strip()


def _exec_solve(code: str, stdin: str, timeout_s: int = 5) -> Tuple[bool, str]:
    ns: dict = {"__builtins__": __builtins__}
    def _h(signum, frame): raise TimeoutError("solve() timed out")
    old = signal.signal(signal.SIGALRM, _h)
    signal.alarm(timeout_s)
    try:
        exec(code, ns)
        if "solve" not in ns:
            return (False, "no solve()")
        got = ns["solve"](stdin)
        if not isinstance(got, str):
            return (False, f"non-str: {type(got).__name__}")
        return (True, got)
    except TimeoutError as e:
        return (False, str(e))
    except Exception as e:
        return (False, f"exec: {e!r}"[:200])
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _run_binary(tool: str, args: Tuple[str, ...], stdin: str,
                 timeout_s: int = 5) -> Tuple[int, str]:
    binary = HARVEST_TOOLS[tool]["bin"]
    env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    try:
        r = subprocess.run(
            [binary, *args], input=stdin,
            capture_output=True, text=True, timeout=timeout_s, env=env,
        )
    except Exception:
        return (-1, "")
    return (r.returncode, r.stdout)


def _llm_call(client, prompt: str, model: str) -> Tuple[str, int]:
    try:
        resp = client.messages.create(
            model=model, max_tokens=768, temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        return (f"[api-error: {e!r}]", 0)
    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    u = getattr(resp, "usage", None)
    tokens = (getattr(u, "input_tokens", 0) + getattr(u, "output_tokens", 0)
              if u else 0)
    return (text, tokens)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=3,
                    help="Probes per tool (default 3).")
    ap.add_argument("--mode", choices=["baseline", "few-shot"],
                    default="baseline")
    ap.add_argument("--k", type=int, default=3,
                    help="Few-shot examples when --mode few-shot.")
    ap.add_argument("--backend", choices=["anthropic", "mlx", "hf", "openai"],
                    default="anthropic")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--cache", default=os.environ.get("NSYNTH_LLM_CACHE_PATH"),
                    help="Harvest cache TSV to read few-shot examples from. "
                         "Required for --mode few-shot.")
    ap.add_argument("--adapter-path", default=None,
                    help="MLX adapter directory to load on top of --model.")
    ap.add_argument("--adapter-routing", default="always",
                    choices=["always", "never", "utility_only"],
                    help="for --backend mlx: when to apply the adapter")
    ap.add_argument("--api-base", default=None,
                    help="OpenAI-compatible API base URL for --backend openai.")
    ap.add_argument("--device", default=None,
                    help="Device override for --backend hf.")
    ap.add_argument("--tools", default=",".join(BENCH_TOOLS),
                    help="Comma-separated subset of tools to benchmark.")
    ap.add_argument("--seed", type=int, default=123,
                    help="Use a different seed than the harvester (42) "
                         "to ensure held-out probes.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.mode == "few-shot":
        if not args.cache:
            print("--cache is required for --mode few-shot", file=sys.stderr)
            sys.exit(2)
        os.environ["NSYNTH_LLM_CACHE_PATH"] = args.cache

    if args.backend == "anthropic":
        try:
            import anthropic
        except ImportError:
            print("pip install anthropic required", file=sys.stderr)
            sys.exit(2)
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            print("ANTHROPIC_API_KEY not set", file=sys.stderr)
            sys.exit(2)
        client = anthropic.Anthropic(api_key=key)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
        from local_model_adapter import LocalModelClient  # noqa: E402
        client = LocalModelClient(
            backend=args.backend,
            model=args.model,
            device=args.device,
            api_base=args.api_base,
            adapter_path=args.adapter_path,
            adapter_routing=args.adapter_routing,
        )

    rng = random.Random(args.seed)
    per_tool: Dict[str, Dict] = {}
    total_tokens = 0
    selected_tools = [t.strip() for t in args.tools.split(",") if t.strip()]

    for tool in selected_tools:
        if tool not in FUZZERS or tool not in HARVEST_TOOLS:
            continue
        if not Path(HARVEST_TOOLS[tool]["bin"]).exists():
            continue
        per_tool[tool] = {"pass": 0, "total": 0, "fails": []}
        for _ in range(args.n):
            fz_rng = random.Random(rng.randint(0, 2**31))
            probe_args, stdin = FUZZERS[tool](fz_rng)
            rc, expected = _run_binary(tool, probe_args, stdin)
            ok_codes = {0, 1} if tool == "grep" else {0}
            if rc not in ok_codes:
                continue
            per_tool[tool]["total"] += 1

            prompt = _build_prompt(
                tool, stdin, expected, probe_args,
                few_shot_k=args.k if args.mode == "few-shot" else 0,
            )
            text, tok = _llm_call(client, prompt, args.model)
            total_tokens += tok
            code = _extract_solve(text)
            ok, got = _exec_solve(code, stdin)
            correct = ok and got == expected
            if correct:
                per_tool[tool]["pass"] += 1
            elif len(per_tool[tool]["fails"]) < 2:
                per_tool[tool]["fails"].append({
                    "args": list(probe_args),
                    "expected": expected[:80],
                    "got": (got[:80] if ok else got[:80]),
                })
            if args.verbose:
                mark = "✓" if correct else "✗"
                print(f"  [{tool:<10}] {mark}")

    total_pass = sum(d["pass"] for d in per_tool.values())
    total = sum(d["total"] for d in per_tool.values())
    print(f"\n{args.mode.upper()} @ {args.backend}:{args.model}")
    if args.adapter_path:
        print(f"adapter: {args.adapter_path}")
    print(f"{'tool':<10} {'pass':>5} {'total':>5} {'pct':>5}")
    for tool, d in sorted(per_tool.items()):
        p, t = d["pass"], d["total"]
        pct = 100.0 * p / max(t, 1)
        print(f"{tool:<10} {p:>5} {t:>5} {pct:>4.0f}%")
    print(f"{'TOTAL':<10} {total_pass:>5} {total:>5} "
          f"{100.0*total_pass/max(total,1):>4.0f}%  "
          f"({total_tokens} tokens)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "mode": args.mode,
            "backend": args.backend,
            "model": args.model,
            "adapter_path": args.adapter_path,
            "per_tool": per_tool,
            "total_pass": total_pass, "total": total,
            "total_tokens": total_tokens,
        }, indent=2))


if __name__ == "__main__":
    main()
