#!/usr/bin/env python3
"""
MCP-mediated inference loop: the model verifies its own code via tool
calls *during* generation, not after.

This differs from our existing `run_humaneval_agent.py` in a crucial
way: the agent runner is an *outer* loop (runner calls model, runner
verifies, runner retries). This loop is *inner* — Claude is given the
nsynth MCP tools directly and chooses when to call `verify_against_tests`,
`lookup_solution`, and `cache_solution` as part of its own reasoning.

The model decides the control flow. We just provide the tools. This is
the "shared memory for coding agents" pattern at its most concrete:
every session's solve populates the cache, every ask-before-write
hits the cache first, every verified code gets written back.

## Why this is novel

Public literature has examples of `execute_code` tools for LLM agents
(e.g. OpenAI Code Interpreter, Claude's tool use). Our addition is the
*persistent verified-code cache* as a peer tool. The LLM can choose:

  - "I've seen this shape — call lookup_solution first."
  - "I drafted code — call verify_against_tests before I commit."
  - "It passed — call cache_solution so the next agent gets it free."

All three calls happen inside a single conversation turn. The
verify-and-cache loop becomes part of the model's reasoning, not an
external wrapper.

## How to run

Needs ANTHROPIC_API_KEY (Claude is one of the few frontier models with
mature tool-use support that speaks MCP natively).

```bash
ANTHROPIC_API_KEY=sk-... python3 tools/mcp/inference_loop.py \\
    --problems tools/benchmarks/humaneval_lite.jsonl \\
    --limit 10 \\
    --verbose
```

Per problem, writes a transcript of the tool calls to
`artifacts/mcp_sessions/{problem}.md`. Each transcript shows the full
reasoning chain — which tools the LLM chose to call, in what order,
what the results were.

This is also the artifact that demonstrates "MCP server usage" to a
reviewer or paper reader. A Claude session with verify + cache + lookup
looks qualitatively different from a Claude session without them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─── Tool schemas (match nsynth_mcp_server.py) ──────────────────────────────


MCP_TOOLS = [
    {
        "name": "lookup_solution",
        "description": "Look up a previously-verified solution for this problem shape before writing new code. Returns {status: 'hit', code: '...'} on hit, {status: 'miss'} otherwise. Call this FIRST to avoid re-solving a problem the team has already solved.",
        "input_schema": {
            "type": "object",
            "properties": {"fingerprint": {"type": "string",
                "description": "Deterministic hash of the problem's I/O examples. Get it from fingerprint()."}},
            "required": ["fingerprint"],
        },
    },
    {
        "name": "fingerprint",
        "description": "Compute the canonical fingerprint for a set of I/O examples. Pass the output to lookup_solution or cache_solution.",
        "input_schema": {
            "type": "object",
            "properties": {"examples": {
                "type": "array",
                "items": {"type": "object",
                    "properties": {"inputs": {"type": "array"}, "expected": {}}},
            }},
            "required": ["examples"],
        },
    },
    {
        "name": "verify_against_tests",
        "description": "Execute candidate Python and run its entry_point against a list of (args, expected) test cases. Returns per-case pass/fail. Use BEFORE committing to code — failures tell you exactly which input to fix.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "entry_point": {"type": "string"},
                "test_cases": {"type": "array", "items": {"type": "array"}},
            },
            "required": ["code", "entry_point", "test_cases"],
        },
    },
    {
        "name": "cache_solution",
        "description": "Persist a verified (fingerprint, code) pair. Call this AFTER verify_against_tests passes — populates the team's shared cache so the next agent gets this solution for free.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fingerprint": {"type": "string"},
                "code": {"type": "string"},
                "model": {"type": "string", "default": "claude-via-mcp-loop"},
            },
            "required": ["fingerprint", "code"],
        },
    },
]


# ─── Direct in-process tool implementations (reused from mcp server) ────────


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from llm_solution_cache import (  # noqa: E402
    fingerprint_examples, lookup as cache_lookup, record as cache_record,
)
import signal, contextlib  # noqa: E402


class _TimeoutError(Exception):
    pass


@contextlib.contextmanager
def _time_limit(seconds: int):
    def handler(signum, frame):
        raise _TimeoutError(f"timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _run_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == "fingerprint":
        return {"fingerprint": fingerprint_examples(args.get("examples", []))}
    if name == "lookup_solution":
        row = cache_lookup(args.get("fingerprint", ""))
        if row is None:
            return {"status": "miss"}
        return {"status": "hit", **row}
    if name == "cache_solution":
        cache_record(args["fingerprint"], args.get("model", "mcp-loop"), args["code"])
        return {"status": "ok"}
    if name == "verify_against_tests":
        code = args["code"]; entry = args["entry_point"]; cases = args["test_cases"]
        ns: dict = {}
        try:
            with _time_limit(5):
                exec(code, ns)
        except Exception as e:
            return {"status": "exec-error", "error": repr(e), "passed": 0, "total": len(cases)}
        fn = ns.get(entry)
        if fn is None:
            return {"status": "no-fn", "passed": 0, "total": len(cases)}
        passed = 0
        failures = []
        for i, case in enumerate(cases):
            fn_args, expected = case[:-1], case[-1]
            try:
                with _time_limit(5):
                    got = fn(*fn_args)
            except Exception as e:
                failures.append({"i": i, "args": fn_args, "error": repr(e)[:120]})
                continue
            if got == expected:
                passed += 1
            else:
                failures.append({"i": i, "args": fn_args,
                                 "got": repr(got)[:80], "expected": repr(expected)[:80]})
        return {"status": "ok" if passed == len(cases) else "partial",
                "passed": passed, "total": len(cases), "failures": failures[:5]}
    return {"error": f"unknown tool {name}"}


# ─── Claude tool-use loop ────────────────────────────────────────────────────


@dataclass
class SessionResult:
    problem: str
    pass_at_1: bool
    tool_calls: int
    cache_hit: bool = False
    final_code: str = ""
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    total_ms: int = 0
    final_error: str = ""


def run_session(
    client, model: str, problem: dict, max_turns: int = 8
) -> SessionResult:
    """One problem, one Claude conversation with MCP-style tools.
    Claude drives; we just dispatch tool calls back."""
    t0 = time.time()

    system = (
        "You are a careful Python function writer. For each problem:\n"
        "1. First call `fingerprint` with the examples. Then call `lookup_solution` "
        "with the fingerprint — if you get a hit, the job is already done.\n"
        "2. If miss, write candidate code, call `verify_against_tests` to check it.\n"
        "3. If verify fails, fix the specific failing case and re-verify.\n"
        "4. Once all tests pass, call `cache_solution` with the fingerprint + code.\n"
        "5. Reply with the final verified code only.\n\n"
        "Prefer to use the tools over explaining. The tools are how we share memory "
        "across sessions — every cache_solution call helps the next agent."
    )

    user_msg = (
        f"Problem `{problem['name']}` with signature `{problem['signature']}`.\n\n"
        f"Examples (list of {{inputs, expected}} objects):\n"
        f"```json\n{json.dumps(problem['examples'], indent=2)}\n```\n\n"
        f"Test cases (to pass to verify_against_tests — list of "
        f"[arg1, arg2, ..., expected]):\n"
        f"```json\n{json.dumps(problem['test_cases'], indent=2)}\n```\n\n"
        f"Entry point: `{problem['name']}`.\n\n"
        f"Solve it using the tools."
    )

    messages = [{"role": "user", "content": user_msg}]
    tool_calls = 0
    transcript: List[Dict[str, Any]] = []
    final_code = ""
    last_verify_pass = False
    cache_hit = False

    for turn in range(max_turns):
        try:
            resp = client.messages.create(
                model=model, max_tokens=2048,
                system=system,
                tools=MCP_TOOLS,
                messages=messages,
            )
        except Exception as e:
            return SessionResult(
                problem=problem["name"], pass_at_1=False,
                tool_calls=tool_calls, transcript=transcript,
                total_ms=int((time.time() - t0) * 1000),
                final_error=f"api-error: {e!r}"[:200],
            )

        content_blocks = resp.content
        stop_reason = getattr(resp, "stop_reason", "")

        # Append assistant turn.
        messages.append({"role": "assistant", "content": content_blocks})

        tool_results = []
        for block in content_blocks:
            btype = getattr(block, "type", "")
            if btype == "tool_use":
                tool_calls += 1
                name = block.name
                args = block.input
                result = _run_tool(name, args)
                transcript.append({
                    "turn": turn, "tool": name, "input": args,
                    "output": result,
                })
                if name == "lookup_solution" and result.get("status") == "hit":
                    cache_hit = True
                    final_code = result.get("code", "")
                if name == "verify_against_tests":
                    last_verify_pass = (result.get("status") == "ok")
                    if last_verify_pass:
                        final_code = args.get("code", "")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })
            elif btype == "text":
                transcript.append({"turn": turn, "assistant_text": block.text[:400]})
                # Catch a final code in assistant text if it's fenced.
                txt = block.text
                import re as _re
                fence = _re.findall(r"```(?:python)?\n?(.*?)```", txt, _re.DOTALL)
                if fence and last_verify_pass:
                    final_code = final_code or fence[0].strip()

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
            # Loop again; Claude may do more tool calls.
            continue

        if stop_reason in ("end_turn", "stop_sequence", "max_tokens") or not tool_results:
            # No more tool calls — we're done.
            break

    # Verify the final code one more time against test cases (belt +
    # suspenders — Claude's verify call used the same engine we're
    # using here, but replay defends against a stale final-text extract).
    pass_at_1 = False
    if final_code:
        result = _run_tool("verify_against_tests", {
            "code": final_code,
            "entry_point": problem["name"],
            "test_cases": problem["test_cases"],
        })
        pass_at_1 = (result.get("status") == "ok")

    return SessionResult(
        problem=problem["name"], pass_at_1=pass_at_1,
        tool_calls=tool_calls, cache_hit=cache_hit,
        final_code=final_code, transcript=transcript,
        total_ms=int((time.time() - t0) * 1000),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--problems", default="tools/benchmarks/humaneval_lite.jsonl")
    ap.add_argument("--out-dir", default="artifacts/mcp_sessions")
    ap.add_argument("--summary", default="artifacts/mcp_inference_loop.md")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)

    try:
        import anthropic
    except ImportError:
        print("[mcp-loop] anthropic SDK required", file=sys.stderr); sys.exit(2)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("[mcp-loop] ANTHROPIC_API_KEY not set", file=sys.stderr); sys.exit(2)
    client = anthropic.Anthropic(api_key=key)

    problems = []
    for line in Path(args.problems).read_text().splitlines():
        if line.strip():
            problems.append(json.loads(line))
    problems = problems[: args.limit]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[SessionResult] = []
    for i, p in enumerate(problems, 1):
        if args.verbose:
            print(f"[{i}/{len(problems)}] {p['name']} ...", end=" ", flush=True)
        r = run_session(client, args.model, p, max_turns=args.max_turns)
        results.append(r)
        if args.verbose:
            mark = f"✓ (cache)" if r.cache_hit else ("✓" if r.pass_at_1 else "✗")
            print(f"{mark} tools={r.tool_calls} ms={r.total_ms}")

        # Write per-session transcript.
        transcript_lines = [
            f"# MCP session: {r.problem}", "",
            f"Pass@1: {'✓' if r.pass_at_1 else '✗'}  •  "
            f"Tool calls: {r.tool_calls}  •  "
            f"Cache hit: {'yes' if r.cache_hit else 'no'}  •  "
            f"Total ms: {r.total_ms}", "",
            "## Tool trace",
            "",
        ]
        for step in r.transcript:
            if "tool" in step:
                transcript_lines.append(f"### turn {step['turn']} — `{step['tool']}`")
                transcript_lines.append("**input:**")
                transcript_lines.append(f"```json\n{json.dumps(step['input'], indent=2, default=str)[:500]}\n```")
                transcript_lines.append("**output:**")
                transcript_lines.append(f"```json\n{json.dumps(step['output'], indent=2, default=str)[:500]}\n```")
                transcript_lines.append("")
            elif "assistant_text" in step:
                transcript_lines.append(f"### turn {step['turn']} — assistant reply")
                transcript_lines.append(f"> {step['assistant_text']}")
                transcript_lines.append("")
        if r.final_code:
            transcript_lines += ["## Final code", "", "```python", r.final_code, "```"]
        (out_dir / f"{r.problem}.md").write_text("\n".join(transcript_lines) + "\n")

    # Summary table.
    total = len(results); passed = sum(1 for r in results if r.pass_at_1)
    cache_hits = sum(1 for r in results if r.cache_hit)
    avg_tools = sum(r.tool_calls for r in results) / max(total, 1)
    lines = [
        "# MCP Inference Loop — summary", "",
        f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "", "## Summary", "",
        f"- Problems: {total}",
        f"- Pass@1: **{passed}/{total}**",
        f"- Cache hits: {cache_hits}",
        f"- Avg tool calls per problem: {avg_tools:.1f}",
        "", "## Per-problem",
        "",
        "| # | problem | pass | tools | cache | ms |",
        "|--:|---------|:----:|:-----:|:-----:|---:|",
    ]
    for i, r in enumerate(results, 1):
        lines.append(
            f"| {i} | [{r.problem}]({out_dir.name}/{r.problem}.md) | "
            f"{'✓' if r.pass_at_1 else '✗'} | {r.tool_calls} | "
            f"{'yes' if r.cache_hit else 'no'} | {r.total_ms} |"
        )
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"[mcp-loop] wrote {summary_path} (passed {passed}/{total})")


if __name__ == "__main__":
    main()
