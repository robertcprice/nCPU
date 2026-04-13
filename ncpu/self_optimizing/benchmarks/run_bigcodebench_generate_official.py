#!/usr/bin/env python3
"""
Generate BigCodeBench samples with official prompts plus per-task timeouts.

This keeps the benchmark real:
- official BigCodeBench task set
- official prompt formatting
- official sanitizer
- official evaluator-compatible JSONL output

It adds operational safeguards the stock generator lacks:
- one-task-at-a-time flushing to disk
- request timeouts
- explicit timeout/error records so runs finish instead of hanging forever
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import requests

try:
    from bigcodebench.data import get_bigcodebench
    from bigcodebench.provider.utility import make_raw_chat_prompt
    from bigcodebench.sanitize import sanitize
except ImportError as exc:  # pragma: no cover - runtime dependency on benchmark boxes
    raise SystemExit(
        "BigCodeBench is required on the benchmark host. "
        "Install it there before running this script."
    ) from exc


DEFAULT_INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
DEFAULT_RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the "
    "problem and passes corresponding tests:"
)


@dataclass
class GenerationResult:
    raw_solution: str
    elapsed_seconds: float
    timed_out: bool
    error: str | None
    metadata: dict[str, Any]


def build_output_path(root: str, model: str, subset: str, split: str) -> str:
    os.makedirs(root, exist_ok=True)
    basename = (
        f"{model.replace('/', '--')}--main--bigcodebench-{subset}-{split}"
        "--openai-0-1-sanitized_calibrated.jsonl"
    )
    return os.path.join(root, basename)


def load_completed_task_ids(path: str) -> set[str]:
    completed: set[str] = set()
    if not os.path.exists(path):
        return completed
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = item.get("task_id")
            if isinstance(task_id, str):
                completed.add(task_id)
    return completed


def generate_once(
    *,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    max_completion_tokens: int,
    timeout_seconds: int,
) -> GenerationResult:
    started = time.perf_counter()
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    try:
        response = requests.post(
            endpoint,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "n": 1,
                "temperature": temperature,
                "top_p": 0.95,
                "max_completion_tokens": max_completion_tokens,
            },
            timeout=timeout_seconds,
        )
        elapsed = time.perf_counter() - started
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        content = ""
        if choices:
            message = choices[0].get("message") or {}
            content = str(message.get("content") or "")
        metadata = {
            "id": payload.get("id"),
            "model": payload.get("model"),
            "created": payload.get("created"),
            "usage": payload.get("usage"),
        }
        return GenerationResult(
            raw_solution=content,
            elapsed_seconds=elapsed,
            timed_out=False,
            error=None,
            metadata=metadata,
        )
    except requests.Timeout:
        elapsed = time.perf_counter() - started
        return GenerationResult(
            raw_solution="",
            elapsed_seconds=elapsed,
            timed_out=True,
            error=f"timeout after {timeout_seconds}s",
            metadata={},
        )
    except Exception as exc:  # pragma: no cover - depends on remote runtime/provider
        elapsed = time.perf_counter() - started
        return GenerationResult(
            raw_solution="",
            elapsed_seconds=elapsed,
            timed_out=False,
            error=f"{type(exc).__name__}: {exc}",
            metadata={},
        )


def append_record(path: str, record: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_generation(args: argparse.Namespace) -> str:
    dataset = get_bigcodebench(subset=args.subset)
    output_path = args.output or build_output_path(
        root=args.root,
        model=args.model,
        subset=args.subset,
        split=args.split,
    )
    completed = load_completed_task_ids(output_path) if args.resume else set()
    total = len(dataset)

    if not args.resume and os.path.exists(output_path):
        os.remove(output_path)

    print(f"Output: {output_path}", flush=True)
    print(
        f"BigCodeBench generation: model={args.model} subset={args.subset} "
        f"split={args.split} timeout={args.timeout_seconds}s "
        f"max_completion_tokens={args.max_completion_tokens}",
        flush=True,
    )
    print(f"Resuming with {len(completed)}/{total} tasks already saved", flush=True)

    saved = len(completed)
    for index, (task_id, task) in enumerate(dataset.items(), start=1):
        if task_id in completed:
            continue

        task_prompt = str(task[f"{args.split}_prompt"])
        prompt = make_raw_chat_prompt(
            task_prompt=task_prompt,
            subset=args.subset,
            split=args.split,
            instruction_prefix=args.instruction_prefix,
            response_prefix=args.response_prefix,
            tokenizer=None,
            prefill=not args.skip_prefill,
            direct_completion=False,
        )

        print(f"[{saved + 1}/{total}] {task_id}: generating", flush=True)
        result = generate_once(
            base_url=args.base_url,
            model=args.model,
            prompt=prompt,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            timeout_seconds=args.timeout_seconds,
        )
        sanitized = sanitize(result.raw_solution, entrypoint=task["entry_point"])

        record = {
            "task_id": task_id,
            "solution": sanitized,
            "raw_solution": result.raw_solution,
            "prompt": prompt,
            "entry_point": task["entry_point"],
            "elapsed_seconds": result.elapsed_seconds,
            "timed_out": result.timed_out,
            "generation_error": result.error,
            "provider_metadata": result.metadata,
        }
        append_record(output_path, record)

        saved += 1
        status = "timeout" if result.timed_out else ("error" if result.error else "ok")
        print(
            f"[{saved}/{total}] {task_id}: {status} "
            f"elapsed={result.elapsed_seconds:.2f}s "
            f"chars={len(result.raw_solution)}",
            flush=True,
        )

    print(f"Completed generation for {saved}/{total} tasks", flush=True)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate official BigCodeBench samples with request timeouts",
    )
    parser.add_argument("--model", required=True, help="Model name, e.g. qwen3.5:9b")
    parser.add_argument(
        "--subset",
        default="hard",
        choices=["full", "hard"],
        help="BigCodeBench subset",
    )
    parser.add_argument(
        "--split",
        default="instruct",
        choices=["instruct", "complete"],
        help="Prompt split",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434/v1",
        help="OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--root",
        default="bcb_results_timeout",
        help="Directory for output JSONL files if --output is omitted",
    )
    parser.add_argument("--output", help="Explicit JSONL output path")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=1280,
        help="Completion token cap per task",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="HTTP timeout per task",
    )
    parser.add_argument(
        "--instruction-prefix",
        default=DEFAULT_INSTRUCTION_PREFIX,
        help="Official instruction prefix",
    )
    parser.add_argument(
        "--response-prefix",
        default=DEFAULT_RESPONSE_PREFIX,
        help="Official response prefix",
    )
    parser.add_argument(
        "--skip-prefill",
        action="store_true",
        help="Disable the official assistant prefill text",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from an existing JSONL if present (default: enabled)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start from scratch even if the output file exists",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_generation(args)


if __name__ == "__main__":
    main()
