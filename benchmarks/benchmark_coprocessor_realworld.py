#!/usr/bin/env python3
"""
Real-world benchmark: baseline LLM vs nCPU coprocessor-injected LLM.

Tests whether the differentiable nCPU coprocessor (trained on arithmetic)
transfers to real-world tasks:
  - HumanEval (164 coding problems, pass@1)
  - GSM8K (math word problems, exact match)
  - Custom coding tasks (10 problems with test suites)
  - Custom reasoning tasks (10 deterministic problems)

Usage:
    # Full benchmark
    python benchmark_coprocessor_realworld.py \
        --model Qwen/Qwen3.5-2B \
        --coprocessor-weights coprocessor_weights.pt \
        --models-dir models \
        --humaneval-path HumanEval.jsonl

    # Quick smoke test
    python benchmark_coprocessor_realworld.py \
        --model Qwen/Qwen3.5-2B \
        --benchmarks coding,reasoning \
        --coprocessor-weights coprocessor_weights.pt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import torch

# Add project root to sys.path so ncpu.coprocessor can be imported
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Text Generation
# ---------------------------------------------------------------------------

def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    stop_sequences: list[str] | None = None,
    use_chat: bool = True,
) -> str:
    """Generate text from the model.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: The user prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        stop_sequences: Stop generation at these strings
        use_chat: Whether to wrap in chat template
    """
    if use_chat:
        messages = [
            {"role": "system", "content": "You are a helpful coding and math assistant. Follow instructions exactly."},
            {"role": "user", "content": prompt},
        ]
        # Qwen3/3.5 chat template supports enable_thinking=False to suppress
        # reasoning mode. Without it the model burns the token budget on planning
        # text before emitting code. Pass the flag when available; fall back for
        # older tokenizer versions.
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["do_sample"] = True
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Strip <think>...</think> blocks (Qwen3.5 thinking mode)
    completion = re.sub(r'<think>.*?</think>\s*', '', completion, flags=re.DOTALL)

    # Apply stop sequences
    if stop_sequences:
        for stop in stop_sequences:
            idx = completion.find(stop)
            if idx >= 0:
                completion = completion[:idx]

    return completion.strip()


def extract_code(text: str) -> str:
    """Extract Python code from a model response."""
    # Try fenced code blocks first
    match = re.search(r'```(?:python)?\s*\n(.*?)```', text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find code starting with def/class/import/from
    lines = text.split('\n')
    code_start = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(('def ', 'from ', 'import ', 'class ')):
            code_start = i
            break

    if code_start is not None:
        return '\n'.join(lines[code_start:]).strip()

    return text.strip()


# ---------------------------------------------------------------------------
# HumanEval Benchmark
# ---------------------------------------------------------------------------

def load_humaneval(path: str) -> list[dict]:
    """Load HumanEval problems from JSONL."""
    problems = []
    with open(path) as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    return problems


def run_code_in_subprocess(code: str, timeout: int = 10) -> dict:
    """Run Python code in a subprocess with timeout."""
    script = textwrap.dedent(f"""\
        import json, sys
        try:
            ns = {{}}
            exec({code!r}, ns)
            json.dump({{"ok": True}}, sys.stdout)
        except AssertionError as e:
            json.dump({{"ok": False, "error": f"Assertion: {{e}}"}}, sys.stdout)
        except Exception as e:
            json.dump({{"ok": False, "error": f"{{type(e).__name__}}: {{e}}"}}, sys.stdout)
    """)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            err = proc.stderr.strip()[-300:] if proc.stderr else "non-zero exit"
            return {"ok": False, "error": f"Process error: {err}"}
        return json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Timeout ({timeout}s)"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def run_humaneval_test(code: str, entry_point: str, test_code: str, timeout: int = 10) -> dict:
    """Run HumanEval check() in subprocess."""
    full_script = textwrap.dedent(f"""\
        import json, sys
        try:
            ns = {{}}
            exec({code!r}, ns)
            entry = ns.get({entry_point!r})
            if entry is None:
                json.dump({{"ok": False, "error": "Function '{entry_point}' not found"}}, sys.stdout)
                sys.exit(0)
            test_ns = dict(ns)
            exec({test_code!r}, test_ns)
            test_ns["check"](entry)
            json.dump({{"ok": True}}, sys.stdout)
        except AssertionError as e:
            json.dump({{"ok": False, "error": f"Assertion: {{e}}"}}, sys.stdout)
        except Exception as e:
            json.dump({{"ok": False, "error": f"{{type(e).__name__}}: {{e}}"}}, sys.stdout)
    """)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", full_script],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            err = proc.stderr.strip()[-300:] if proc.stderr else "non-zero exit"
            return {"ok": False, "error": f"Process error: {err}"}
        return json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Timeout ({timeout}s)"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def benchmark_humaneval(
    model, tokenizer, humaneval_path: str, max_problems: int | None = None,
) -> dict:
    """Run HumanEval pass@1 benchmark."""
    problems = load_humaneval(humaneval_path)
    if max_problems:
        problems = problems[:max_problems]

    correct = 0
    total = len(problems)
    results = []

    print(f"  Running HumanEval ({total} problems)...")

    for i, problem in enumerate(problems):
        task_id = problem["task_id"]
        prompt_code = problem["prompt"]
        entry_point = problem["entry_point"]
        test_code = problem["test"]

        # Use chat format: ask model to complete the function
        chat_prompt = (
            "Complete the following Python function. "
            "Write ONLY the complete function (including the signature). "
            "No markdown, no backticks, no explanation.\n\n"
            f"{prompt_code}"
        )

        t0 = time.time()
        completion = generate_text(
            model, tokenizer, chat_prompt,
            max_new_tokens=512,
            use_chat=True,
        )
        gen_time = time.time() - t0

        # Extract code
        code = extract_code(completion)

        # If model only output the body, prepend the function signature
        if f"def {entry_point}" not in code:
            code = prompt_code + code

        # Run test
        result = run_humaneval_test(code, entry_point, test_code)
        passed = result.get("ok", False)
        if passed:
            correct += 1

        results.append({
            "task_id": task_id,
            "passed": passed,
            "error": result.get("error"),
            "gen_time": gen_time,
        })

        if (i + 1) % 20 == 0 or passed is False and i < 5:
            status = "PASS" if passed else "FAIL"
            print(f"    [{i+1}/{total}] {task_id}: {status} ({gen_time:.1f}s) "
                  f"running={correct}/{i+1}={correct/(i+1):.1%}")

    accuracy = correct / total if total > 0 else 0
    print(f"  HumanEval: {correct}/{total} = {accuracy:.1%}")

    return {
        "benchmark": "humaneval",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


# ---------------------------------------------------------------------------
# GSM8K Benchmark
# ---------------------------------------------------------------------------

def extract_gsm8k_answer(text: str) -> float | None:
    """Extract the numeric answer from GSM8K format (after ####)."""
    match = re.search(r'####\s*(.+)', text)
    if match:
        answer = match.group(1).strip().replace(",", "").replace("$", "")
        try:
            return float(answer)
        except ValueError:
            return None
    return None


def extract_number_from_response(text: str) -> float | None:
    """Extract the final numerical answer from a model response."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.strip()

    # Try #### format first (if model follows the prompt format)
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Look for explicit answer patterns
    patterns = [
        r'(?:answer|result|total|final answer)\s*(?:is|=|:)\s*\$?(-?[\d,]+\.?\d*)',
        r'\*\*(-?[\d,]+\.?\d*)\*\*\s*$',
        r'\\boxed\{(-?[\d,]+\.?\d*)\}',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                continue

    # Last number in the response
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass

    return None


def benchmark_gsm8k(
    model, tokenizer, num_problems: int = 200,
) -> dict | None:
    """Run GSM8K benchmark on a subset of the test set."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception as e:
        print(f"  Could not load GSM8K: {e}")
        return None

    problems = list(ds)[:num_problems]
    correct = 0
    total = len(problems)
    results = []

    print(f"  Running GSM8K ({total} problems)...")

    for i, problem in enumerate(problems):
        question = problem["question"]
        expected = extract_gsm8k_answer(problem["answer"])

        prompt = (
            "Solve this math problem step by step. "
            "End your response with the final numerical answer after '#### '.\n\n"
            f"Question: {question}\n\nSolution:"
        )

        t0 = time.time()
        response = generate_text(model, tokenizer, prompt, max_new_tokens=256, use_chat=True)
        gen_time = time.time() - t0

        predicted = extract_number_from_response(response)

        passed = False
        if predicted is not None and expected is not None:
            try:
                passed = abs(float(predicted) - float(expected)) < 0.01
            except (ValueError, TypeError):
                passed = False

        if passed:
            correct += 1

        results.append({
            "question": question[:80],
            "expected": expected,
            "predicted": predicted,
            "passed": passed,
            "gen_time": gen_time,
        })

        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{total}] GSM8K running: {correct}/{i+1}={correct/(i+1):.1%}")

    accuracy = correct / total if total > 0 else 0
    print(f"  GSM8K: {correct}/{total} = {accuracy:.1%}")

    return {
        "benchmark": "gsm8k",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Custom Coding Tasks
# ---------------------------------------------------------------------------

CODING_TASKS = [
    # --- Easy ---
    {
        "name": "fibonacci",
        "prompt": (
            "Write a Python function `fibonacci(n)` that returns the nth Fibonacci number "
            "(0-indexed). fibonacci(0)=0, fibonacci(1)=1.\n"
            "Write raw Python only. No markdown, no explanation."
        ),
        "test_code": (
            "assert fibonacci(0) == 0\n"
            "assert fibonacci(1) == 1\n"
            "assert fibonacci(10) == 55\n"
            "assert fibonacci(20) == 6765\n"
        ),
        "entry_point": "fibonacci",
        "difficulty": "easy",
    },
    {
        "name": "factorial",
        "prompt": (
            "Write a Python function `factorial(n)` that returns n! (n factorial). "
            "factorial(0)=1.\n"
            "Write raw Python only. No markdown, no explanation."
        ),
        "test_code": (
            "assert factorial(0) == 1\n"
            "assert factorial(1) == 1\n"
            "assert factorial(5) == 120\n"
            "assert factorial(10) == 3628800\n"
        ),
        "entry_point": "factorial",
        "difficulty": "easy",
    },
    {
        "name": "is_prime",
        "prompt": (
            "Write a Python function `is_prime(n)` that returns True if n is prime, False otherwise.\n"
            "Write raw Python only. No markdown, no explanation."
        ),
        "test_code": (
            "assert is_prime(2) == True\n"
            "assert is_prime(17) == True\n"
            "assert is_prime(1) == False\n"
            "assert is_prime(4) == False\n"
            "assert is_prime(97) == True\n"
        ),
        "entry_point": "is_prime",
        "difficulty": "easy",
    },
    {
        "name": "binary_search",
        "prompt": (
            "Write a Python function `binary_search(arr, target)` that returns the index "
            "of target in sorted array arr, or -1 if not found.\n"
            "Write raw Python only. No markdown, no explanation."
        ),
        "test_code": (
            "assert binary_search([1,3,5,7,9], 5) == 2\n"
            "assert binary_search([1,3,5,7,9], 1) == 0\n"
            "assert binary_search([1,3,5,7,9], 9) == 4\n"
            "assert binary_search([1,3,5,7,9], 4) == -1\n"
            "assert binary_search([], 1) == -1\n"
        ),
        "entry_point": "binary_search",
        "difficulty": "easy",
    },
    {
        "name": "quick_sort",
        "prompt": (
            "Write a Python function `quick_sort(arr)` that returns a new sorted list.\n"
            "Write raw Python only. No markdown, no explanation."
        ),
        "test_code": (
            "assert quick_sort([3,1,4,1,5,9,2,6]) == [1,1,2,3,4,5,6,9]\n"
            "assert quick_sort([]) == []\n"
            "assert quick_sort([1]) == [1]\n"
            "assert quick_sort([5,4,3,2,1]) == [1,2,3,4,5]\n"
        ),
        "entry_point": "quick_sort",
        "difficulty": "easy",
    },
    # --- Hard ---
    {
        "name": "dijkstra",
        "prompt": (
            "Write a Python function `dijkstra(graph, start, end)` where graph is a dict "
            "mapping node to list of (neighbor, weight) tuples. Return the shortest distance "
            "from start to end, or float('inf') if unreachable.\n"
            "Write raw Python only. No markdown, no explanation."
        ),
        "test_code": (
            "g = {'A': [('B',4),('C',2)], 'B': [('D',5)], 'C': [('B',1),('D',8)], 'D': []}\n"
            "assert dijkstra(g, 'A', 'D') == 8\n"
            "assert dijkstra(g, 'A', 'A') == 0\n"
            "assert dijkstra(g, 'A', 'B') == 3\n"
            "g2 = {'A': [('B',1)], 'B': [], 'C': []}\n"
            "assert dijkstra(g2, 'A', 'C') == float('inf')\n"
        ),
        "entry_point": "dijkstra",
        "difficulty": "hard",
    },
    {
        "name": "merge_intervals",
        "prompt": (
            "Write a Python function `merge_intervals(intervals)` that takes a list of "
            "[start, end] intervals and returns merged overlapping intervals, sorted by start.\n"
            "Write raw Python only. No markdown, no explanation."
        ),
        "test_code": (
            "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
            "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]\n"
            "assert merge_intervals([]) == []\n"
            "assert merge_intervals([[1,2]]) == [[1,2]]\n"
        ),
        "entry_point": "merge_intervals",
        "difficulty": "hard",
    },
    {
        "name": "longest_substring",
        "prompt": (
            "Write a Python function `longest_substring(s)` that returns the length of the "
            "longest substring without repeating characters.\n"
            "Write raw Python only. No markdown, no explanation."
        ),
        "test_code": (
            "assert longest_substring('abcabcbb') == 3\n"
            "assert longest_substring('bbbbb') == 1\n"
            "assert longest_substring('pwwkew') == 3\n"
            "assert longest_substring('') == 0\n"
        ),
        "entry_point": "longest_substring",
        "difficulty": "hard",
    },
    {
        "name": "lru_cache",
        "prompt": (
            "Write a Python class `LRUCache` with:\n"
            "  - __init__(self, capacity: int)\n"
            "  - get(self, key: int) -> int: return value or -1\n"
            "  - put(self, key: int, value: int) -> None: insert/update, evict LRU if over capacity\n"
            "Write raw Python only. No markdown, no explanation."
        ),
        "test_code": (
            "cache = LRUCache(2)\n"
            "cache.put(1, 1)\n"
            "cache.put(2, 2)\n"
            "assert cache.get(1) == 1\n"
            "cache.put(3, 3)  # evicts key 2\n"
            "assert cache.get(2) == -1\n"
            "cache.put(4, 4)  # evicts key 1\n"
            "assert cache.get(1) == -1\n"
            "assert cache.get(3) == 3\n"
            "assert cache.get(4) == 4\n"
        ),
        "entry_point": "LRUCache",
        "difficulty": "hard",
    },
    {
        "name": "topological_sort",
        "prompt": (
            "Write a Python function `topological_sort(graph)` where graph is a dict mapping "
            "node to list of dependencies. Return a valid topological ordering (list) or "
            "empty list if there's a cycle.\n"
            "Write raw Python only. No markdown, no explanation."
        ),
        "test_code": (
            "result = topological_sort({'a': ['b','c'], 'b': ['d'], 'c': ['d'], 'd': []})\n"
            "assert isinstance(result, list)\n"
            "assert len(result) == 4\n"
            "assert result.index('d') < result.index('b')\n"
            "assert result.index('d') < result.index('c')\n"
            "assert result.index('b') < result.index('a')\n"
            "assert result.index('c') < result.index('a')\n"
            "assert topological_sort({'a': ['b'], 'b': ['a']}) == []\n"
        ),
        "entry_point": "topological_sort",
        "difficulty": "hard",
    },
]


def benchmark_coding(model, tokenizer) -> dict:
    """Run custom coding benchmark."""
    correct = 0
    total = len(CODING_TASKS)
    results = []

    print(f"  Running custom coding ({total} problems)...")

    for i, task in enumerate(CODING_TASKS):
        t0 = time.time()
        response = generate_text(model, tokenizer, task["prompt"], max_new_tokens=256, use_chat=True)
        gen_time = time.time() - t0

        code = extract_code(response)

        # Build test script: define function + run assertions
        test_script = code + "\n\n" + task["test_code"]

        result = run_code_in_subprocess(test_script, timeout=10)
        passed = result.get("ok", False)
        if passed:
            correct += 1

        results.append({
            "name": task["name"],
            "difficulty": task["difficulty"],
            "passed": passed,
            "error": result.get("error"),
            "gen_time": gen_time,
        })

        status = "PASS" if passed else "FAIL"
        print(f"    [{i+1}/{total}] {task['name']}: {status} ({gen_time:.1f}s)")

    accuracy = correct / total if total > 0 else 0
    print(f"  Coding: {correct}/{total} = {accuracy:.1%}")

    return {
        "benchmark": "coding",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Custom Reasoning Tasks
# ---------------------------------------------------------------------------

REASONING_TASKS = [
    # --- Easy ---
    {
        "name": "weighted_path",
        "prompt": (
            "A weighted graph has edges: A-B=4, A-C=2, C-B=1, B-D=5, C-D=8. "
            "What is the shortest distance from A to D? "
            'Return JSON only: {"answer": <number>}'
        ),
        "expected": 8,
        "normalizer": int,
        "difficulty": "easy",
    },
    {
        "name": "meeting_slot",
        "prompt": (
            "A person is busy from 09:00-09:45, 10:15-11:00, and 11:30-12:00. "
            "What is the earliest 30-minute free slot that starts at or after 09:00 "
            "and ends by 12:00? Return the start time in HH:MM. "
            'Return JSON only: {"answer": "<HH:MM>"}'
        ),
        "expected": "09:45",
        "normalizer": lambda v: str(v).strip(),
        "difficulty": "easy",
    },
    {
        "name": "price_reasoning",
        "prompt": (
            "An item costs 80 dollars. Apply a 25% discount, then apply 10% sales tax. "
            "What is the final price? "
            'Return JSON only: {"answer": <number>}'
        ),
        "expected": 66.0,
        "normalizer": lambda v: round(float(v), 2),
        "difficulty": "easy",
    },
    {
        "name": "logic_constraint",
        "prompt": (
            "Exactly one of A, B, and C is true. If A is true then B is true. "
            "If C is true then B is false. Which variable must be true? "
            'Return JSON only: {"answer": "<letter>"}'
        ),
        "expected": "C",
        "normalizer": lambda v: str(v).strip().upper(),
        "difficulty": "easy",
    },
    {
        "name": "robot_grid",
        "prompt": (
            "A robot starts at [0, 0] facing north. It executes: forward 2, right, "
            "forward 3, right, forward 1, left, forward 2. What is the final coordinate? "
            'Return JSON only: {"answer": [x, y]}'
        ),
        "expected": [5, 1],
        "normalizer": lambda v: [int(v[0]), int(v[1])] if isinstance(v, (list, tuple)) else None,
        "difficulty": "easy",
    },
    # --- Hard ---
    {
        "name": "seating_constraint",
        "prompt": (
            "Five people A, B, C, D, E sit in seats numbered 1 through 5. "
            "B is in seat 2. A is in seat 5. D sits immediately to the right of C "
            "(i.e., D's seat number is exactly one more than C's). "
            "What seat number is E in? "
            'Return JSON only: {"answer": <number>}'
        ),
        "expected": 1,
        "normalizer": int,
        "difficulty": "hard",
    },
    {
        "name": "probability_calc",
        "prompt": (
            "A bag contains 3 red, 4 blue, and 2 green marbles (9 total). "
            "Two marbles are drawn without replacement. "
            "What is the probability that both marbles are the same color? "
            "Express your answer as a simplified fraction like '5/18'. "
            'Return JSON only: {"answer": "<fraction>"}'
        ),
        "expected": "5/18",
        "normalizer": lambda v: str(v).strip(),
        "difficulty": "hard",
    },
    {
        "name": "modular_arithmetic",
        "prompt": (
            "What is the remainder when 7^123 is divided by 13? "
            'Return JSON only: {"answer": <number>}'
        ),
        "expected": 5,
        "normalizer": int,
        "difficulty": "hard",
    },
    {
        "name": "train_meeting",
        "prompt": (
            "Train A leaves a station at 8:15 AM traveling east at 60 mph. "
            "Train B leaves the same station at 9:00 AM traveling east at 90 mph. "
            "At what time does Train B catch up to Train A? "
            'Return JSON only: {"answer": "<HH:MM>"}'
        ),
        "expected": "10:30",
        "normalizer": lambda v: str(v).strip(),
        "difficulty": "hard",
    },
    {
        "name": "counting_strings",
        "prompt": (
            "How many 4-letter strings can be formed from the set {A, B, C} "
            "such that no two adjacent letters are the same? "
            'Return JSON only: {"answer": <number>}'
        ),
        "expected": 24,
        "normalizer": int,
        "difficulty": "hard",
    },
]


def extract_json_answer(text: str) -> Any:
    """Extract the 'answer' field from a JSON response."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.strip()

    # Try to find JSON in fenced blocks
    match = re.search(r'```(?:json)?\s*\n(.*?)```', text, flags=re.DOTALL)
    if match:
        text = match.group(1).strip()

    # Find first JSON object
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != '{':
            continue
        try:
            payload, _ = decoder.raw_decode(text[idx:])
            if isinstance(payload, dict) and "answer" in payload:
                return payload["answer"]
        except json.JSONDecodeError:
            continue

    return None


def benchmark_reasoning(model, tokenizer) -> dict:
    """Run custom reasoning benchmark."""
    correct = 0
    total = len(REASONING_TASKS)
    results = []

    print(f"  Running custom reasoning ({total} problems)...")

    for i, task in enumerate(REASONING_TASKS):
        t0 = time.time()
        response = generate_text(model, tokenizer, task["prompt"], max_new_tokens=256, use_chat=True)
        gen_time = time.time() - t0

        raw_answer = extract_json_answer(response)

        passed = False
        if raw_answer is not None:
            try:
                normalized = task["normalizer"](raw_answer)
                expected = task["normalizer"](task["expected"]) if not isinstance(task["expected"], str) else task["expected"]
                if isinstance(task["expected"], str):
                    expected = task["normalizer"](task["expected"])
                passed = normalized == expected
            except Exception:
                pass

        if passed:
            correct += 1

        results.append({
            "name": task["name"],
            "difficulty": task["difficulty"],
            "passed": passed,
            "raw_answer": str(raw_answer),
            "expected": str(task["expected"]),
            "gen_time": gen_time,
        })

        status = "PASS" if passed else "FAIL"
        print(f"    [{i+1}/{total}] {task['name']}: {status} "
              f"(got={raw_answer}, expected={task['expected']}, {gen_time:.1f}s)")

    accuracy = correct / total if total > 0 else 0
    print(f"  Reasoning: {correct}/{total} = {accuracy:.1%}")

    return {
        "benchmark": "reasoning",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Coprocessor Injection
# ---------------------------------------------------------------------------

def inject_coprocessor(
    model,
    weights_path: str,
    models_dir: str,
    confidence_aware: bool | None = None,
    max_gate: float | None = None,
) -> None:
    """Inject nCPU coprocessor into model and load trained weights.

    Args:
        model: HuggingFace model
        weights_path: Path to coprocessor_weights.pt
        models_dir: Path to nCPU ALU model directory
        confidence_aware: Override confidence-aware mode (auto-detected from weights if None)
        max_gate: Override max gate value (auto-detected from weights if None)
    """
    from ncpu.coprocessor.config import NCPUCoprocessorConfig
    from ncpu.coprocessor.inject import inject_ncpu_coprocessor

    # Load weights first to detect config metadata
    print(f"  Loading coprocessor weights from {weights_path}...")
    state = torch.load(weights_path, map_location=model.device, weights_only=True)

    # Auto-detect config from saved weights (if available)
    saved_cfg = state.get("_config", {})
    if confidence_aware is None:
        confidence_aware = saved_cfg.get("confidence_aware", False)
    if max_gate is None:
        max_gate = saved_cfg.get("max_gate", 0.1)

    config = NCPUCoprocessorConfig(
        layer_indices=[-1],
        n_bits=8,
        num_ops=7,
        models_dir=models_dir,
        freeze_alu=True,
        confidence_aware=confidence_aware,
        max_gate=max_gate,
        target_load=saved_cfg.get("target_load", 0.01),
    )

    print(f"  Injecting coprocessor (layer -1, confidence_aware={confidence_aware}, max_gate={max_gate})...")
    injected = inject_ncpu_coprocessor(model, config)

    for i, module in enumerate(injected):
        router_key = f"layer_{i}_router"
        expert_key = f"layer_{i}_expert"

        if router_key in state:
            module.router.load_state_dict(state[router_key])
            print(f"    Loaded router weights for layer {i}")
        if expert_key in state:
            module.expert.load_state_dict(state[expert_key], strict=False)
            print(f"    Loaded expert weights for layer {i}")

    # Verify gate statistics
    for i, module in enumerate(injected):
        with torch.no_grad():
            test_hidden = torch.randn(
                1, 1, model.config.hidden_size,
                device=model.device,
                dtype=next(module.parameters()).dtype,
            )
            gate, _ = module.router(test_hidden)
            print(f"    Layer {i} gate sample: {gate.mean().item():.4f}")

    print(f"  Coprocessor injection complete ({len(injected)} layers)")


# ---------------------------------------------------------------------------
# Main Benchmark Runner
# ---------------------------------------------------------------------------

def run_all_benchmarks(
    model,
    tokenizer,
    benchmarks: list[str],
    humaneval_path: str | None = None,
    gsm8k_count: int = 200,
    humaneval_count: int | None = None,
) -> dict[str, dict]:
    """Run all selected benchmarks and return results."""
    results = {}

    if "humaneval" in benchmarks and humaneval_path:
        results["humaneval"] = benchmark_humaneval(
            model, tokenizer, humaneval_path,
            max_problems=humaneval_count,
        )

    if "gsm8k" in benchmarks:
        gsm8k_result = benchmark_gsm8k(model, tokenizer, num_problems=gsm8k_count)
        if gsm8k_result:
            results["gsm8k"] = gsm8k_result

    if "coding" in benchmarks:
        results["coding"] = benchmark_coding(model, tokenizer)

    if "reasoning" in benchmarks:
        results["reasoning"] = benchmark_reasoning(model, tokenizer)

    return results


def print_comparison(baseline: dict, coprocessor: dict) -> dict:
    """Print and return A/B comparison."""
    print(f"\n{'='*72}")
    print("COMPARISON: Baseline vs Coprocessor")
    print(f"{'='*72}")
    print(f"{'Benchmark':<15} {'Baseline':>10} {'Coprocessor':>12} {'Delta':>10}")
    print("-" * 50)

    deltas = {}
    for bench_name in sorted(set(list(baseline.keys()) + list(coprocessor.keys()))):
        b = baseline.get(bench_name, {}).get("accuracy", 0)
        c = coprocessor.get(bench_name, {}).get("accuracy", 0)
        d = c - b
        deltas[bench_name] = d
        print(f"{bench_name:<15} {b:>9.1%} {c:>11.1%} {d:>+9.1%}")

    # Overall
    b_scores = [v.get("accuracy", 0) for v in baseline.values()]
    c_scores = [v.get("accuracy", 0) for v in coprocessor.values()]
    if b_scores and c_scores:
        b_avg = sum(b_scores) / len(b_scores)
        c_avg = sum(c_scores) / len(c_scores)
        d_avg = c_avg - b_avg
        print("-" * 50)
        print(f"{'AVERAGE':<15} {b_avg:>9.1%} {c_avg:>11.1%} {d_avg:>+9.1%}")
        deltas["average"] = d_avg

    return deltas


def main():
    parser = argparse.ArgumentParser(
        description="Real-world benchmark: baseline LLM vs nCPU coprocessor",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B",
                        help="HuggingFace model ID")
    parser.add_argument("--coprocessor-weights", required=True,
                        help="Path to coprocessor_weights.pt")
    parser.add_argument("--models-dir", default="models",
                        help="Path to nCPU ALU models directory")
    parser.add_argument("--humaneval-path", default=None,
                        help="Path to HumanEval.jsonl")
    parser.add_argument("--benchmarks", default="humaneval,gsm8k,coding,reasoning",
                        help="Comma-separated benchmarks to run")
    parser.add_argument("--gsm8k-count", type=int, default=200,
                        help="Number of GSM8K problems")
    parser.add_argument("--humaneval-count", type=int, default=None,
                        help="Max HumanEval problems (default: all 164)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run only baseline (skip coprocessor)")
    parser.add_argument("--confidence-aware", action="store_true", default=None,
                        help="Force confidence-aware gating ON (auto-detected from weights if omitted)")
    parser.add_argument("--no-confidence-aware", action="store_true",
                        help="Force confidence-aware gating OFF")
    parser.add_argument("--max-gate", type=float, default=None,
                        help="Override max gate value (auto-detected from weights if omitted)")
    args = parser.parse_args()

    # Resolve confidence_aware: explicit flag > auto-detect
    if args.no_confidence_aware:
        args.confidence_aware = False
    elif args.confidence_aware is None:
        args.confidence_aware = None  # auto-detect from weights

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    model_dtype = dtype_map[args.dtype]

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}, dtype: {args.dtype}")

    # Load model and tokenizer
    print(f"\nLoading model: {args.model}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Qwen3.5 models are multimodal (VL) — need to load text-only CausalLM
    # with the nested text_config, not the top-level composite config
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if hasattr(config, "text_config") and not hasattr(config, "vocab_size"):
        print("  Detected composite (VL) config, loading text-only CausalLM...")
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        text_cfg_dict = config.text_config if isinstance(config.text_config, dict) else config.text_config.to_dict()
        text_config = Qwen3_5TextConfig(**text_cfg_dict)
        model = Qwen3_5ForCausalLM.from_pretrained(
            args.model,
            config=text_config,
            dtype=model_dtype,
            device_map=device,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=model_dtype,
            device_map=device,
            trust_remote_code=True,
        )
    model.eval()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Phase 1: Baseline benchmarks
    print(f"\n{'='*72}")
    print("PHASE 1: BASELINE (stock model)")
    print(f"{'='*72}")

    t0 = time.time()
    baseline_results = run_all_benchmarks(
        model, tokenizer, benchmarks,
        humaneval_path=args.humaneval_path,
        gsm8k_count=args.gsm8k_count,
        humaneval_count=args.humaneval_count,
    )
    baseline_time = time.time() - t0
    print(f"\nBaseline completed in {baseline_time:.1f}s")

    coprocessor_results = {}
    if not args.baseline_only:
        # Phase 2: Inject coprocessor and re-run
        print(f"\n{'='*72}")
        print("PHASE 2: COPROCESSOR (nCPU-injected model)")
        print(f"{'='*72}")

        inject_coprocessor(
            model, args.coprocessor_weights, args.models_dir,
            confidence_aware=args.confidence_aware,
            max_gate=args.max_gate,
        )

        t0 = time.time()
        coprocessor_results = run_all_benchmarks(
            model, tokenizer, benchmarks,
            humaneval_path=args.humaneval_path,
            gsm8k_count=args.gsm8k_count,
            humaneval_count=args.humaneval_count,
        )
        copro_time = time.time() - t0
        print(f"\nCoprocessor completed in {copro_time:.1f}s")

        # Comparison
        deltas = print_comparison(baseline_results, coprocessor_results)
    else:
        deltas = {}

    # Save results
    output_path = args.output or f"coprocessor_realworld_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report = {
        "model": args.model,
        "device": device,
        "dtype": args.dtype,
        "benchmarks": benchmarks,
        "coprocessor_weights": args.coprocessor_weights,
        "baseline": {k: {kk: vv for kk, vv in v.items() if kk != "results"} for k, v in baseline_results.items()},
        "coprocessor": {k: {kk: vv for kk, vv in v.items() if kk != "results"} for k, v in coprocessor_results.items()},
        "deltas": deltas,
        "baseline_detailed": baseline_results,
        "coprocessor_detailed": coprocessor_results,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
