#!/usr/bin/env python3
"""
GSM8K agent-loop runner — can our cache + verify + retry pattern
improve *reasoning* pass rates, not just code?

GSM8K = 1319 math word problems with single numeric answers. The same
three-stage pattern applies: generate → verify (numeric equality) →
retry with error feedback on fail → cache verified solutions for
future lookups of the same question fingerprint.

Differences from the coding runners:
  - "verification" is numeric-answer equality, not running tests
  - "code" stored in the cache is the LLM's chain-of-thought reply,
    keyed on question fingerprint
  - retry prompt shows the ground-truth expected number only if we're
    in "teacher-forced" mode; default is blind retry

Modes:
  --mode llm        Single-shot baseline (no retry, no cache)
  --mode agent      cache → k=3 parallel sampling → 2 retries → cache

Usage:
    ANTHROPIC_API_KEY=sk-... python3 tools/benchmarks/run_gsm8k.py \\
        --mode agent --limit 100 --verbose

The hybrid-agent numbers on GSM8K are the direct measurement of
"does this pattern help reasoning, or just code?"
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_solution_cache import (  # noqa: E402
    lookup as cache_lookup, record as cache_record,
)


# ─── GSM8K helpers ──────────────────────────────────────────────────────────


GSM8K_ANSWER_RE = re.compile(r"####\s*([\-0-9,.]+)")


def extract_ground_truth(answer_field: str) -> Optional[float]:
    """Pull the numeric final answer from GSM8K's `#### N` convention."""
    m = GSM8K_ANSWER_RE.search(answer_field)
    if not m:
        return None
    num = m.group(1).replace(",", "").strip().rstrip(".")
    try:
        return float(num)
    except ValueError:
        return None


_FINAL_ANSWER_RES = [
    # "The answer is X"
    re.compile(r"(?:final\s+answer|answer)\s*(?:is|=|:)\s*\$?\s*([\-0-9,.]+)", re.I),
    # "= 18"
    re.compile(r"=\s*\$?\s*([\-0-9,.]+)\s*(?:dollars?|\.|$)", re.I),
    # "#### 18" (GSM8K-style from LLM)
    re.compile(r"####\s*([\-0-9,.]+)"),
    # Standalone number on last content line.
    re.compile(r"^\s*\$?\s*([\-0-9,.]+)\s*$", re.M),
]


def extract_predicted_answer(llm_response: str) -> Optional[float]:
    """Extract a numeric answer from free-form LLM text. Try several
    common shapes; return the last number matched (late-stage answers
    override earlier incidental numbers)."""
    last: Optional[float] = None
    for rx in _FINAL_ANSWER_RES:
        for m in rx.finditer(llm_response):
            raw = m.group(1).replace(",", "").strip().rstrip(".")
            try:
                last = float(raw)
            except ValueError:
                continue
    return last


def fingerprint_gsm8k(question: str) -> str:
    """Deterministic hash of a GSM8K question. Exact-match cache key —
    not semantic similarity. Questions word-for-word identical get the
    same fingerprint."""
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:32]


# ─── LLM calls ──────────────────────────────────────────────────────────────


def build_prompt(question: str) -> str:
    return (
        "Solve this math problem. Reason step by step, then write your "
        "final numeric answer after the string `#### `.\n\n"
        f"Problem: {question}\n\nSolution:"
    )


# ─── Program-of-Thought: write a Python function, we execute it ────────────


def build_pot_prompt(question: str) -> str:
    """Program-of-Thought prompt: ask the model to emit a single Python
    function `solve()` that returns the numeric answer. We then exec it
    and use the return value. Based on Chen et al. 2022 "Program of
    Thoughts Prompting"."""
    return (
        "Solve this math problem by writing a Python function named "
        "`solve` that returns the numeric answer. Do NOT include test "
        "cases, prints, or explanation — just the function definition. "
        "Use only basic Python arithmetic (+, -, *, /, //, %, **) and "
        "the `math` module. The function takes no arguments.\n\n"
        f"Problem: {question}\n\n"
        "```python\ndef solve():\n    # your code here\n    return ...\n```\n\n"
        "Now write the function that solves the problem:"
    )


_POT_FENCE_RE = re.compile(r"```(?:python)?\n?(.*?)```", re.DOTALL)


def extract_and_run_pot(code_text: str,
                          timeout_s: int = 5) -> Tuple[Optional[float], str]:
    """Extract Python code from the LLM response, exec it in a
    restricted namespace with a timeout, and call solve(). Returns
    (numeric_answer_or_None, error_message)."""
    # Prefer fenced block if present.
    fences = _POT_FENCE_RE.findall(code_text)
    body = fences[0] if fences else code_text
    # Only keep from first `def solve` onward.
    idx = body.find("def solve")
    if idx >= 0:
        body = body[idx:]
    if "def solve" not in body:
        return (None, "no solve() function in response")

    import math as _math
    import signal as _signal
    ns = {"math": _math, "__builtins__": {
        "abs": abs, "min": min, "max": max, "pow": pow, "round": round,
        "sum": sum, "len": len, "int": int, "float": float, "range": range,
        "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "sorted": sorted, "reversed": reversed, "all": all, "any": any,
    }}

    def _handler(signum, frame):
        raise TimeoutError(f"pot exec timed out after {timeout_s}s")
    old = _signal.signal(_signal.SIGALRM, _handler)
    _signal.alarm(timeout_s)
    try:
        exec(body, ns)
        if "solve" not in ns:
            return (None, "solve() not defined after exec")
        result = ns["solve"]()
    except TimeoutError as e:
        return (None, str(e))
    except Exception as e:
        return (None, f"exec: {e!r}"[:150])
    finally:
        _signal.alarm(0)
        _signal.signal(_signal.SIGALRM, old)

    if isinstance(result, (int, float)):
        return (float(result), "")
    try:
        return (float(result), "")
    except (TypeError, ValueError):
        return (None, f"non-numeric result: {result!r}"[:100])


def build_retry_prompt(question: str, previous: str, prev_answer) -> str:
    return (
        "Your previous attempt gave the wrong final answer. Re-examine your "
        "arithmetic carefully; the correct answer is a single integer or "
        "decimal.\n\n"
        f"Problem: {question}\n\n"
        f"Previous solution:\n{previous[:800]}\n\n"
        f"Previous numeric answer: {prev_answer}\n\n"
        "Check each step. Then write the corrected final numeric answer "
        "after `#### `."
    )


def llm_call(client, prompt: str, model: str,
             temperature: float = 0.0, max_tokens: int = 1024) -> Tuple[str, int]:
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


# ─── Tool use: give the model a calculator during reasoning ─────────────────


CALCULATOR_TOOL = {
    "name": "calculator",
    "description": (
        "Evaluate a Python arithmetic expression in a restricted sandbox. "
        "Use this for every arithmetic computation in your reasoning — "
        "do NOT compute arithmetic in your head. Returns the numeric result. "
        "Example inputs: '3 * 45 + 7', 'sqrt(144) + floor(pi * 10)', "
        "'100 - (18 + 24)'. Disallowed: imports, attribute access, "
        "assignments. Whitelisted names: abs, min, max, pow, round, sum, "
        "len, int, float + math module's sqrt/log/exp/sin/cos/floor/ceil/"
        "pi/e/factorial/gcd."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {"type": "string",
                           "description": "The Python expression to evaluate."},
        },
        "required": ["expression"],
    },
}


def _safe_eval(expression: str) -> Tuple[bool, object, str]:
    """Evaluate the expression using the MCP tool's sandbox. Returns
    (ok, value_or_None, err_message)."""
    # Reuse the MCP server's implementation so behaviour matches the
    # MCP exposed tool verbatim.
    mcp_dir = Path(__file__).resolve().parent.parent / "mcp"
    sys.path.insert(0, str(mcp_dir))
    try:
        from nsynth_mcp_server import tool_evaluate_expression  # type: ignore
    except ImportError as e:
        return (False, None, f"tool unavailable: {e!r}")
    r = tool_evaluate_expression({"expression": expression})
    if "error" in r:
        return (False, None, r["error"])
    return (True, r.get("value"), "")


def build_tool_prompt(question: str) -> str:
    return (
        "Solve this math problem. Reason step by step, and USE the "
        "`calculator` tool for EVERY arithmetic operation — do not "
        "compute in your head. After your reasoning, write the final "
        "numeric answer after `#### `.\n\n"
        f"Problem: {question}\n\nSolution:"
    )


def llm_call_with_tools(client, question: str, model: str,
                         max_turns: int = 8,
                         max_tokens: int = 1024) -> Tuple[str, int, int]:
    """Multi-turn conversation where the model can call the calculator
    tool mid-generation. Returns (final_text, total_tokens, tool_calls).

    The loop continues until either:
      - the model's response has no tool_use blocks (done)
      - max_turns reached (truncate)
    """
    prompt = build_tool_prompt(question)
    messages = [{"role": "user", "content": prompt}]
    total_tokens = 0
    tool_calls = 0
    for turn in range(max_turns):
        try:
            resp = client.messages.create(
                model=model, max_tokens=max_tokens, temperature=0.0,
                tools=[CALCULATOR_TOOL], messages=messages,
            )
        except Exception:
            return ("", total_tokens, tool_calls)
        usage = getattr(resp, "usage", None)
        if usage:
            total_tokens += (getattr(usage, "input_tokens", 0)
                             + getattr(usage, "output_tokens", 0))

        # Append assistant response.
        assistant_blocks = resp.content
        messages.append({"role": "assistant", "content": assistant_blocks})

        tool_uses = [b for b in assistant_blocks
                     if getattr(b, "type", None) == "tool_use"]
        if not tool_uses or resp.stop_reason != "tool_use":
            # Final response — extract text.
            text = "".join(b.text for b in assistant_blocks
                           if getattr(b, "type", None) == "text")
            return (text, total_tokens, tool_calls)

        tool_calls += len(tool_uses)
        tool_results = []
        for tu in tool_uses:
            expr = (tu.input or {}).get("expression", "")
            ok, val, err = _safe_eval(expr)
            result_str = str(val) if ok else f"error: {err}"
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result_str,
            })
        messages.append({"role": "user", "content": tool_results})

    # Exhausted turns — return whatever text the last assistant turn had.
    last_text = ""
    for m in reversed(messages):
        if m["role"] == "assistant":
            for b in m["content"]:
                if getattr(b, "type", None) == "text":
                    last_text += b.text
            break
    return (last_text, total_tokens, tool_calls)


# ─── Agent loop (single problem) ────────────────────────────────────────────


@dataclass
class GSMResult:
    question_hash: str
    path: str            # "cache" | "sample" | "retry" | "miss"
    correct: bool
    predicted: Optional[float]
    ground_truth: float
    elapsed_ms: int
    total_tokens: int = 0
    extra: str = ""


def _majority_vote(preds: List[Optional[float]]) -> Tuple[Optional[float], int, int]:
    """Return (majority_answer, vote_count, total_numeric). When no
    extractable number wins, majority_answer is None.

    Rounding: values within 1e-6 of each other count as the same vote.
    This handles cases where one sample outputs `18` and another `18.0`."""
    nums = [p for p in preds if p is not None]
    if not nums:
        return (None, 0, 0)
    buckets: List[Tuple[float, int]] = []
    for v in nums:
        for i, (k, c) in enumerate(buckets):
            if abs(k - v) < 1e-6:
                buckets[i] = (k, c + 1)
                break
        else:
            buckets.append((v, 1))
    buckets.sort(key=lambda kc: -kc[1])
    top_val, top_count = buckets[0]
    return (top_val, top_count, len(nums))


def solve_agent(
    client, question: str, ground_truth: float, model: str,
    k: int, max_retries: int, timeout_s: int,
    self_consistency: bool = False,
    escalate_model: Optional[str] = None,
    use_calculator: bool = False,
    program_of_thought: bool = False,
    ensemble: bool = False,
    vpot: bool = False,
    vpot_max_retries: int = 2,
    vpot_retrieval: int = 0,
) -> GSMResult:
    """Agent loop for GSM8K with optional self-consistency and escalation.

    Modes:
      - default (self_consistency=False): best-of-N first-correct, k-diverse
        temperatures, falls through to retry loop.
      - self_consistency=True: sample k at varied T, majority-vote the
        extracted numeric answer. If majority has ≥2 votes, use it.
        Otherwise fall back to the T=0 sample and proceed to retry.
      - escalate_model set: after all retries fail, one final call to
        escalate_model before giving up.
    """
    t0 = time.time()
    fp = fingerprint_gsm8k(question)

    cached = cache_lookup(fp)
    if cached is not None:
        pred = extract_predicted_answer(cached["code"])
        if pred is not None and abs(pred - ground_truth) < 1e-6:
            return GSMResult(
                question_hash=fp, path="cache", correct=True,
                predicted=pred, ground_truth=ground_truth,
                elapsed_ms=int((time.time() - t0) * 1000),
                extra=f"prior_model={cached['model']}",
            )

    total_tokens = 0

    # VPoT — Verified Program-of-Thought:
    #   PoT (one call → exec), if exec errors OR result fails invariants,
    #   retry with the specific error; cache verified solve() on success.
    #   Optional final escalation to escalate_model on persistent failure.
    #
    # This is our coding agent loop (cache → generate → verify → retry →
    # cache) generalised to math: the "code" is solve() and verification
    # is exec + invariant checks instead of running test cases.
    if vpot:
        # Optionally prepend few-shot context from similar past problems.
        retrieval_prefix = ""
        if vpot_retrieval > 0:
            try:
                from text_retrieval import build_text_retrieval_prefix
                retrieval_prefix = build_text_retrieval_prefix(
                    question, k=vpot_retrieval, min_similarity=0.15,
                )
            except Exception:
                retrieval_prefix = ""
        pot_prompt = retrieval_prefix + build_pot_prompt(question)
        last_text = ""
        last_err = ""
        last_answer: Optional[float] = None

        def _check_invariants(val: Optional[float]) -> str:
            """Return '' if ok, else a short reason for rejection."""
            if val is None:
                return "result is None / non-numeric"
            if not isinstance(val, (int, float)):
                return f"result not numeric: {type(val).__name__}"
            import math as _math
            if _math.isnan(val):
                return "result is NaN"
            if _math.isinf(val):
                return "result is infinite"
            if abs(val) > 1e15:
                return f"|result| = {abs(val):.2e} exceeds plausible range 1e15"
            return ""

        for attempt in range(vpot_max_retries + 1):
            if attempt == 0:
                prompt = pot_prompt
                temperature = 0.0
            else:
                # Retry with explicit exec/invariant feedback.
                prompt = (
                    "Your previous `solve()` failed. "
                    f"Error: {last_err[:400]}\n\n"
                    f"Previous code:\n```python\n{last_text[:800]}\n```\n\n"
                    f"Problem: {question}\n\n"
                    "Write a corrected `def solve():` that returns the "
                    "correct numeric answer. Do NOT include test cases, "
                    "prints, or explanation — just the function definition."
                )
                temperature = 0.3  # small diversity to escape same error
            text, tokens = llm_call(client, prompt, model,
                                     temperature=temperature, max_tokens=768)
            total_tokens += tokens
            if not text:
                last_err = "empty response"
                continue
            last_text = text
            val, exec_err = extract_and_run_pot(text)
            if exec_err:
                last_err = exec_err
                last_answer = val
                continue
            inv_err = _check_invariants(val)
            if inv_err:
                last_err = inv_err
                last_answer = val
                continue
            last_answer = val
            correct = abs(val - ground_truth) < 1e-6
            if correct and text:
                try:
                    if vpot_retrieval > 0:
                        from text_retrieval import record_with_question
                        record_with_question(fp, model, text, question)
                    else:
                        cache_record(fp, model, text)
                except Exception:
                    pass
            return GSMResult(
                question_hash=fp,
                path="vpot" if attempt == 0 else "vpot-retry",
                correct=correct, predicted=val, ground_truth=ground_truth,
                elapsed_ms=int((time.time() - t0) * 1000),
                total_tokens=total_tokens,
                extra=f"attempt={attempt}",
            )

        # All VPoT attempts left us without a valid numeric result OR we
        # exhausted retries. Optional Sonnet escalation.
        if escalate_model:
            text, tokens = llm_call(client, pot_prompt, escalate_model,
                                     temperature=0.0, max_tokens=768)
            total_tokens += tokens
            if text:
                val, _ = extract_and_run_pot(text)
                if val is not None and not _check_invariants(val):
                    correct = abs(val - ground_truth) < 1e-6
                    if correct:
                        try: cache_record(fp, escalate_model, text)
                        except Exception: pass
                    return GSMResult(
                        question_hash=fp,
                        path="vpot-escalate" if correct else "vpot-miss",
                        correct=correct, predicted=val,
                        ground_truth=ground_truth,
                        elapsed_ms=int((time.time() - t0) * 1000),
                        total_tokens=total_tokens,
                        extra=f"to={escalate_model}",
                    )

        return GSMResult(
            question_hash=fp, path="vpot-miss", correct=False,
            predicted=last_answer, ground_truth=ground_truth,
            elapsed_ms=int((time.time() - t0) * 1000),
            total_tokens=total_tokens,
            extra=f"last_err={last_err[:60]}",
        )

    # Ensemble: run PoT + tool-use + plain CoT in parallel, majority-vote.
    # Different strategies fail on different problem subtypes; agreement
    # among 2+ methods is strong evidence of correctness. When all 3
    # disagree the problem is genuinely ambiguous → optional escalation.
    if ensemble:
        pot_prompt = build_pot_prompt(question)
        cot_prompt = build_prompt(question)

        def _call_pot():
            t, tk = llm_call(client, pot_prompt, model,
                             temperature=0.0, max_tokens=768)
            return ("pot", t, tk)

        def _call_cot():
            t, tk = llm_call(client, cot_prompt, model,
                             temperature=0.0, max_tokens=1024)
            return ("cot", t, tk)

        def _call_tool():
            t, tk, _ = llm_call_with_tools(client, question, model,
                                             max_turns=6, max_tokens=1024)
            return ("tool", t, tk)

        with cf.ThreadPoolExecutor(max_workers=3) as pool:
            futs = [pool.submit(_call_pot),
                    pool.submit(_call_cot),
                    pool.submit(_call_tool)]
            raw = [f.result() for f in futs]

        preds_by_method: List[Tuple[str, Optional[float], str]] = []
        for method, text, tokens in raw:
            total_tokens += tokens
            if not text:
                preds_by_method.append((method, None, ""))
                continue
            if method == "pot":
                val, _ = extract_and_run_pot(text)
            else:
                val = extract_predicted_answer(text)
            preds_by_method.append((method, val, text))

        preds = [p for _, p, _ in preds_by_method]
        answer, votes, n_num = _majority_vote(preds)

        # Optional escalation when the 3 methods disagree entirely.
        if (answer is None or votes < 2) and escalate_model:
            esc_prompt = build_pot_prompt(question)
            text, tokens = llm_call(client, esc_prompt, escalate_model,
                                     temperature=0.0, max_tokens=768)
            total_tokens += tokens
            if text:
                val, _ = extract_and_run_pot(text)
                if val is not None:
                    answer = val
                    votes = 1
                    preds_by_method.append(("escalate", val, text))

        correct = (answer is not None
                   and abs(answer - ground_truth) < 1e-6)
        if correct:
            # Cache the first method's text that produced the winning answer.
            for method, val, text in preds_by_method:
                if val is not None and abs(val - answer) < 1e-6 and text:
                    try: cache_record(fp, f"{model}:{method}", text)
                    except Exception: pass
                    break
        winning_methods = [m for m, v, _ in preds_by_method
                            if v is not None and abs(v - (answer or 0)) < 1e-6]
        return GSMResult(
            question_hash=fp,
            path="ensemble" if correct else "ensemble-miss",
            correct=correct, predicted=answer, ground_truth=ground_truth,
            elapsed_ms=int((time.time() - t0) * 1000),
            total_tokens=total_tokens,
            extra=f"votes={votes}/{n_num} [{','.join(winning_methods)}]",
        )

    # Program-of-Thought: one call → Python function → exec → answer.
    # Skips best-of-N since the computation is symbolic not textual.
    # Optional self-consistency: k calls at varied T, majority-vote.
    if program_of_thought:
        pot_prompt = build_pot_prompt(question)
        if self_consistency and k > 1:
            pot_temps = [0.0] + [0.3 + 0.15 * i for i in range(1, k)]
            pot_temps = pot_temps[:k]

            # Threads do network I/O only (the LLM call); exec of the
            # returned code runs on the main thread because our SIGALRM
            # timeout only works there.
            def _pot_call(i: int):
                t, tk = llm_call(client, pot_prompt, model,
                                  temperature=pot_temps[i], max_tokens=768)
                return (i, t, tk)

            with cf.ThreadPoolExecutor(max_workers=min(k, 5)) as pool:
                raw = sorted(
                    [f.result() for f in cf.as_completed(
                        [pool.submit(_pot_call, i) for i in range(k)])],
                    key=lambda x: x[0])
            samples = []
            for i, text, tokens in raw:
                val = None
                if text:
                    val, _ = extract_and_run_pot(text)
                samples.append((i, text, tokens, val))
            total_tokens += sum(s[2] for s in samples)
            preds = [s[3] for s in samples]
            answer, votes, _ = _majority_vote(preds)
            correct = (answer is not None
                       and abs(answer - ground_truth) < 1e-6)
            # Cache the first sample whose answer matches the majority.
            if correct:
                for _, text, _, v in samples:
                    if v is not None and abs(v - answer) < 1e-6 and text:
                        try: cache_record(fp, model, text)
                        except Exception: pass
                        break
            return GSMResult(
                question_hash=fp,
                path="pot-sc" if correct else "pot-sc-miss",
                correct=correct, predicted=answer,
                ground_truth=ground_truth,
                elapsed_ms=int((time.time() - t0) * 1000),
                total_tokens=total_tokens,
                extra=f"votes={votes}/{k}",
            )

        # Single-shot PoT.
        text, tokens = llm_call(client, pot_prompt, model,
                                 temperature=0.0, max_tokens=768)
        total_tokens += tokens
        answer, err = extract_and_run_pot(text) if text else (None, "no response")
        correct = (answer is not None and abs(answer - ground_truth) < 1e-6)
        if correct and text:
            try: cache_record(fp, model, text)
            except Exception: pass
        return GSMResult(
            question_hash=fp,
            path="pot" if correct else "pot-miss",
            correct=correct, predicted=answer, ground_truth=ground_truth,
            elapsed_ms=int((time.time() - t0) * 1000),
            total_tokens=total_tokens,
            extra=(err[:60] if err and not correct else "ok"),
        )

    # Tool-use mode: single greedy call with mid-generation calculator.
    # Skips best-of-N and self-consistency since every arithmetic step
    # is already verified by the Python sandbox.
    if use_calculator:
        text, toks, ncalls = llm_call_with_tools(client, question, model)
        total_tokens += toks
        pred = extract_predicted_answer(text)
        correct = (pred is not None and abs(pred - ground_truth) < 1e-6)
        if correct and text:
            try: cache_record(fp, model, text)
            except Exception: pass
        return GSMResult(
            question_hash=fp,
            path="tool" if correct else "tool-miss",
            correct=correct, predicted=pred, ground_truth=ground_truth,
            elapsed_ms=int((time.time() - t0) * 1000),
            total_tokens=total_tokens,
            extra=f"tool_calls={ncalls}",
        )

    prompt = build_prompt(question)

    temps = [0.0] + [0.4 + 0.15 * i for i in range(1, k)]
    temps = temps[:k]

    def _gen(i: int):
        text, tokens = llm_call(client, prompt, model, temperature=temps[i])
        return (i, text, tokens)

    candidates: List[Tuple[int, str, int]] = []
    if k > 1:
        with cf.ThreadPoolExecutor(max_workers=min(k, 5)) as pool:
            for fut in cf.as_completed([pool.submit(_gen, i) for i in range(k)]):
                candidates.append(fut.result())
        candidates.sort(key=lambda x: x[0])
    else:
        candidates = [_gen(0)]

    last_text = ""
    last_pred: Optional[float] = None

    # Self-consistency path: majority-vote the extracted numeric answers.
    if self_consistency:
        preds: List[Optional[float]] = []
        sample_texts: List[str] = []
        for idx, text, tokens in candidates:
            total_tokens += tokens
            if not text:
                preds.append(None); sample_texts.append(""); continue
            p = extract_predicted_answer(text)
            preds.append(p); sample_texts.append(text)
        # Last non-empty sample for potential retry.
        for i, t in enumerate(sample_texts):
            if t:
                last_text = t; last_pred = preds[i]
        top, votes, n_num = _majority_vote(preds)
        # Require at least 2 votes for the majority to count as confident.
        if top is not None and votes >= 2:
            correct = abs(top - ground_truth) < 1e-6
            # Cache the text of the first sample that matched the majority.
            if correct:
                for i, p in enumerate(preds):
                    if p is not None and abs(p - top) < 1e-6 and sample_texts[i]:
                        try: cache_record(fp, model, sample_texts[i])
                        except Exception: pass
                        break
            if correct:
                return GSMResult(
                    question_hash=fp, path="sc-vote", correct=True,
                    predicted=top, ground_truth=ground_truth,
                    elapsed_ms=int((time.time() - t0) * 1000),
                    total_tokens=total_tokens,
                    extra=f"votes={votes}/{n_num}",
                )
            # Majority is wrong — fall through to retry to try to recover.
    else:
        # First-correct best-of-N (original behaviour).
        for idx, text, tokens in candidates:
            total_tokens += tokens
            if not text:
                continue
            pred = extract_predicted_answer(text)
            last_text = text
            last_pred = pred
            if pred is not None and abs(pred - ground_truth) < 1e-6:
                try: cache_record(fp, model, text)
                except Exception: pass
                return GSMResult(
                    question_hash=fp, path="sample", correct=True,
                    predicted=pred, ground_truth=ground_truth,
                    elapsed_ms=int((time.time() - t0) * 1000),
                    total_tokens=total_tokens,
                    extra=f"s{idx} T={temps[idx]:.2f}",
                )

    # Retry loop (blind — we don't give the model the ground truth).
    for retry in range(1, max_retries + 1):
        retry_prompt = build_retry_prompt(question, last_text, last_pred)
        text, tokens = llm_call(client, retry_prompt, model, temperature=0.0)
        total_tokens += tokens
        if not text:
            continue
        pred = extract_predicted_answer(text)
        last_text = text
        last_pred = pred
        if pred is not None and abs(pred - ground_truth) < 1e-6:
            try: cache_record(fp, model, text)
            except Exception: pass
            return GSMResult(
                question_hash=fp, path="retry", correct=True,
                predicted=pred, ground_truth=ground_truth,
                elapsed_ms=int((time.time() - t0) * 1000),
                total_tokens=total_tokens,
                extra=f"r{retry}",
            )

    # Final cascade: one call to a stronger model if configured.
    if escalate_model:
        esc_prompt = build_retry_prompt(question, last_text, last_pred)
        text, tokens = llm_call(client, esc_prompt, escalate_model,
                                 temperature=0.0)
        total_tokens += tokens
        if text:
            pred = extract_predicted_answer(text)
            if pred is not None and abs(pred - ground_truth) < 1e-6:
                try: cache_record(fp, escalate_model, text)
                except Exception: pass
                return GSMResult(
                    question_hash=fp, path="escalate", correct=True,
                    predicted=pred, ground_truth=ground_truth,
                    elapsed_ms=int((time.time() - t0) * 1000),
                    total_tokens=total_tokens,
                    extra=f"to={escalate_model}",
                )
            last_pred = pred  # use escalated answer even if wrong

    return GSMResult(
        question_hash=fp, path="miss", correct=False,
        predicted=last_pred, ground_truth=ground_truth,
        elapsed_ms=int((time.time() - t0) * 1000),
        total_tokens=total_tokens,
        extra=f"k={k},r={max_retries}",
    )


# ─── Runner ─────────────────────────────────────────────────────────────────


def write_report(results: List[GSMResult], out: Path, mode: str,
                 model: str, total_ms: int):
    total = len(results); correct = sum(1 for r in results if r.correct)
    pct = 100.0 * correct / max(total, 1)
    by_path: dict = {}
    for r in results:
        by_path[r.path] = by_path.get(r.path, 0) + 1
    total_tok = sum(r.total_tokens for r in results)

    lines = [
        f"# GSM8K Results — mode: {mode}",
        "",
        f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} — "
        f"{total} problems, model={model}, {total_ms/1000:.1f}s total.",
        "",
        "## Summary",
        "",
        f"- **Accuracy**: **{correct}/{total} ({pct:.1f}%)**",
    ]
    for p, n in sorted(by_path.items()):
        lines.append(f"- path `{p}`: {n}")
    lines.append(f"- Total tokens: ~{total_tok}")
    lines.append("")
    lines.append("## Sample failures (first 15)")
    lines.append("")
    lines.append("| # | path | predicted | ground truth | extra |")
    lines.append("|--:|------|----------:|-------------:|-------|")
    shown = 0
    for i, r in enumerate(results):
        if r.correct: continue
        if shown >= 15: break
        lines.append(
            f"| {i} | {r.path} | {r.predicted!r} | {r.ground_truth!r} | {r.extra[:40]} |"
        )
        shown += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["llm", "agent"], default="agent")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0,
                    help="Skip first N problems in the test split.")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--self-consistency", action="store_true",
                    help="Majority-vote over k samples instead of first-correct.")
    ap.add_argument("--escalate-model", default=None,
                    help="Fall back to this model after all retries fail "
                         "(e.g. claude-sonnet-4-6).")
    ap.add_argument("--tool-use", action="store_true",
                    help="Give the model a calculator tool during reasoning. "
                         "Replaces best-of-N: every arithmetic step is "
                         "verified by the Python sandbox.")
    ap.add_argument("--program-of-thought", action="store_true",
                    help="Program-of-Thought: model writes a Python function "
                         "`solve()` that we exec to get the answer. "
                         "Alternative to tool-use; computation moves entirely "
                         "into Python instead of being split across tool calls.")
    ap.add_argument("--vpot", action="store_true",
                    help="Verified Program-of-Thought: PoT + retry on "
                         "exec/invariant failure + optional Sonnet escalate. "
                         "Our coding agent loop applied to math via solve().")
    ap.add_argument("--vpot-max-retries", type=int, default=2,
                    help="VPoT retry budget (default 2).")
    ap.add_argument("--vpot-retrieval", type=int, default=0,
                    help="Retrieval-augmented VPoT: prepend top-K similar "
                         "past solve() functions from cache as few-shot. "
                         "Requires the cache to have populated questions "
                         "(we add them automatically on each solve). "
                         "0 = disabled.")
    ap.add_argument("--ensemble", action="store_true",
                    help="Run PoT + tool-use + plain CoT in parallel, "
                         "majority-vote across methods. Catches problems "
                         "where one strategy happens to fail but another "
                         "succeeds. Combine with --escalate-model to route "
                         "3-way disagreements to a stronger model.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--timeout-s", type=int, default=30)
    args = ap.parse_args()

    out = args.out or f"artifacts/gsm8k_{args.mode}.md"
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("[gsm8k] pip install datasets required", file=sys.stderr); sys.exit(2)
    try:
        import anthropic  # type: ignore
    except ImportError:
        print("[gsm8k] pip install anthropic required", file=sys.stderr); sys.exit(2)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("[gsm8k] ANTHROPIC_API_KEY not set", file=sys.stderr); sys.exit(2)
    client = anthropic.Anthropic(api_key=key)

    ds = load_dataset("openai/gsm8k", "main", split="test")
    start = args.offset
    end = min(args.offset + args.limit, len(ds))
    problems = [dict(ds[i]) for i in range(start, end)]
    print(f"[gsm8k] {len(problems)} problems, mode={args.mode}, model={args.model}")

    t_start = time.time()
    results: List[GSMResult] = []

    for i, p in enumerate(problems, 1):
        q = p["question"]; gt = extract_ground_truth(p["answer"])
        if gt is None:
            continue
        if args.verbose:
            print(f"[{i}/{len(problems)}] ", end="", flush=True)
        if args.mode == "agent":
            r = solve_agent(
                client, q, gt, args.model,
                args.k, args.max_retries, args.timeout_s,
                self_consistency=args.self_consistency,
                escalate_model=args.escalate_model,
                use_calculator=args.tool_use,
                program_of_thought=args.program_of_thought,
                ensemble=args.ensemble,
                vpot=args.vpot,
                vpot_max_retries=args.vpot_max_retries,
                vpot_retrieval=args.vpot_retrieval,
            )
        else:
            # Single-shot baseline.
            t0 = time.time()
            text, tokens = llm_call(client, build_prompt(q), args.model,
                                     temperature=0.0)
            pred = extract_predicted_answer(text)
            ok = pred is not None and abs(pred - gt) < 1e-6
            r = GSMResult(
                question_hash=fingerprint_gsm8k(q),
                path="sample" if ok else "miss",
                correct=ok, predicted=pred, ground_truth=gt,
                elapsed_ms=int((time.time() - t0) * 1000),
                total_tokens=tokens,
                extra="k=1" if args.mode == "llm" else "",
            )
        results.append(r)
        if args.verbose:
            mark = "✓" if r.correct else f"✗ pred={r.predicted} gt={r.ground_truth}"
            print(f"{r.path} {mark}")

    total_ms = int((time.time() - t_start) * 1000)
    write_report(results, Path(out), args.mode, args.model, total_ms)
    correct = sum(1 for r in results if r.correct)
    print(
        f"[gsm8k] wrote {out} — accuracy {correct}/{len(results)} "
        f"({100.0 * correct / max(len(results), 1):.1f}%) in {total_ms/1000:.1f}s"
    )


if __name__ == "__main__":
    main()
