#!/usr/bin/env python3
"""
Export verified code-generation solutions as a fine-tuning dataset.

The `llm_solution_cache` grows every time the hybrid / best-of-N /
retry / agent runner successfully generates Python that passes a
problem's test_cases. Over weeks of real code-gen usage, the cache
accumulates a high-quality supervised dataset: (prompt → verified
Python) pairs with *execution-grounded* correctness labels.

By default this exporter drops non-code rows from the shared cache
(for example GSM8K reasoning traces) and tries to recover the original
benchmark prompt from stored metadata or known benchmark datasets.

This script emits that dataset as JSONL in the standard
`{"prompt": ..., "completion": ...}` schema suitable for:

  - HuggingFace fine-tuning APIs
  - OpenAI fine-tuning (reformat as messages if needed)
  - local LoRA training on small open models (Qwen, Llama, etc.)

The cache-as-training-set pattern is the concrete answer to "the LLM
costs $X per call; can we get equivalent performance cheaper?" — the
answer is "yes, distill a smaller model on the problems you actually
hit in production."

Output: artifacts/distillation_dataset.jsonl

Usage:
    python3 tools/export_distillation_dataset.py \\
        [--cache ~/.nsynth_llm_solutions.tsv] \\
        [--out artifacts/distillation_dataset.jsonl] \\
        [--min-success 1]    skip rows that never transferred
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent / "benchmarks"))
from llm_solution_cache import _load_all, _cache_path  # noqa: E402


def _looks_like_python_function(code: str) -> bool:
    if not code or "def " not in code:
        return False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body)


def _row_to_prompt_fallback(
    fp: str,
    code: str,
    model_src: str = "",
    examples: Optional[List[dict]] = None,
) -> str:
    """Reconstruct a training prompt from the stored row.

    Binary-harvest rows (model starts with 'binary:') get a utility-
    reimplementation prompt grounded in the captured I/O pair — this
    matches what the harvester emits in its own --emit-jsonl path and
    unifies the two dataset sources under a single schema.

    Other rows fall back to signature-derivation since we don't store
    the original prompt. For those, the function's first line carries
    enough structure for a training signal."""
    # Binary-harvest: richer prompt with the captured stdin + expected stdout.
    if model_src.startswith("binary:") and examples:
        tool = model_src.split(":", 1)[1]
        stdin = examples[0].get("inputs", [""])[0] if examples else ""
        expected = examples[0].get("expected", "")
        return (
            f"Reimplement the Unix utility `{tool}` in Python. "
            f"Given the following stdin input, your `solve(stdin)` "
            f"function should return the stdout that `{tool}` would "
            f"produce.\n\n"
            f"Input:\n```\n{stdin[:600]}\n```\n\n"
            f"Expected output:\n```\n{expected[:600]}\n```\n\n"
            f"Write `def solve(stdin: str) -> str:` now."
        )

    first_line = code.splitlines()[0] if code else ""
    if first_line.startswith("def "):
        return (
            f"Write a Python function:\n\n{first_line}:\n"
            f"(fingerprint={fp})\n\n"
            f"Reply with only the function definition."
        )
    return f"Write Python: {first_line}"


def _build_benchmark_prompt_catalog() -> Dict[str, dict]:
    """Recover richer prompts for legacy cache rows by matching the cache
    fingerprint back to benchmark datasets.

    Best-effort only: if datasets or cached corpora are unavailable, the
    exporter falls back to prompt reconstruction from the cache row."""
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return {}

    catalog: Dict[str, dict] = {}

    try:
        from llm_solution_cache import fingerprint_humaneval_task
        from run_humaneval_full import build_llm_prompt

        for row in load_dataset("openai_humaneval", split="test"):
            problem = dict(row)
            fp = fingerprint_humaneval_task(
                problem["task_id"], problem["entry_point"], problem["prompt"]
            )
            catalog[fp] = {
                "prompt": build_llm_prompt(problem),
                "metadata": {
                    "task_kind": "humaneval",
                    "task_id": problem["task_id"],
                    "entry_point": problem["entry_point"],
                    "prompt_source": "benchmark_recovery",
                },
            }
    except Exception:
        pass

    try:
        from run_mbpp import (
            MBPPProblem,
            build_initial_prompt,
            derive_examples_from_asserts,
            derive_fn_name,
            fingerprint_mbpp,
        )

        ds = load_dataset(
            "google-research-datasets/mbpp",
            "sanitized",
            trust_remote_code=True,
        )
        for split_name, split in ds.items():
            for row in split:
                problem = dict(row)
                fn_name = derive_fn_name(problem["test_list"])
                if fn_name is None:
                    continue
                examples = derive_examples_from_asserts(problem["test_list"], fn_name)
                mbpp_problem = MBPPProblem(
                    task_id=problem["task_id"],
                    text=problem["prompt"],
                    test_list=problem["test_list"],
                    fn_name=fn_name,
                    examples=examples,
                )
                fp = fingerprint_mbpp(problem["prompt"], problem["test_list"])
                catalog[fp] = {
                    "prompt": build_initial_prompt(mbpp_problem, retrieval_k=0),
                    "metadata": {
                        "task_kind": "mbpp",
                        "task_id": problem["task_id"],
                        "split": split_name,
                        "fn_name": fn_name,
                        "prompt_source": "benchmark_recovery",
                    },
                }
    except Exception:
        pass

    return catalog


def _resolve_prompt(
    fp: str,
    row: dict,
    prompt_catalog: Optional[Dict[str, dict]] = None,
) -> Tuple[str, Dict[str, Any]]:
    cache_metadata = dict(row.get("metadata") or {})
    if cache_metadata.get("prompt"):
        prompt = cache_metadata.pop("prompt")
        cache_metadata.setdefault("prompt_source", "cache_metadata")
        return prompt, cache_metadata

    if prompt_catalog and fp in prompt_catalog:
        recovered = dict(prompt_catalog[fp].get("metadata") or {})
        return prompt_catalog[fp]["prompt"], recovered

    return _row_to_prompt_fallback(
        fp,
        row["code"],
        model_src=row.get("model", ""),
        examples=row.get("examples", []),
    ), {"prompt_source": "fallback"}


def _build_export_record(
    fp: str,
    row: dict,
    fmt: str,
    prompt_catalog: Optional[Dict[str, dict]],
    include_noncode: bool,
) -> Optional[Dict[str, Any]]:
    completion = row["code"]
    if not include_noncode and not _looks_like_python_function(completion):
        return None

    prompt, prompt_meta = _resolve_prompt(fp, row, prompt_catalog)
    source_meta = {
        "fingerprint": fp,
        "source_model": row["model"],
        "success_count": row["success_count"],
        "last_used_at": row["last_used_at"],
        **prompt_meta,
    }

    if fmt == "openai":
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "metadata": source_meta,
        }
    if fmt == "hf":
        return {
            "prompt": prompt,
            "completion": completion,
            "metadata": source_meta,
        }
    return {
        "fingerprint": fp,
        "model": row["model"],
        "success_count": row["success_count"],
        "last_used_at": row["last_used_at"],
        "prompt": prompt,
        "code": completion,
        "metadata": prompt_meta,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", default=None,
                    help="override llm solution cache path (defaults to ~/.nsynth_llm_solutions.tsv)")
    ap.add_argument("--out", default="artifacts/distillation_dataset.jsonl")
    ap.add_argument("--min-success", type=int, default=0,
                    help="skip entries with success_count below this (default 0 = include all)")
    ap.add_argument("--format", choices=["openai", "hf", "raw"], default="raw",
                    help="dataset format")
    ap.add_argument("--include-noncode", action="store_true",
                    help="include non-function rows such as reasoning traces")
    ap.add_argument("--no-recover-benchmark-prompts", action="store_true",
                    help="skip dataset-backed prompt recovery for older cache rows")
    args = ap.parse_args()

    if args.cache:
        os.environ["NSYNTH_LLM_CACHE_PATH"] = args.cache

    entries = _load_all()
    kept = 0
    dropped_low_success = 0
    dropped_noncode = 0
    prompt_catalog = None if args.no_recover_benchmark_prompts else _build_benchmark_prompt_catalog()

    repo = Path(__file__).resolve().parents[1]
    out_path = (repo / args.out) if not args.out.startswith("/") else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for fp, row in entries.items():
            if row["success_count"] < args.min_success:
                dropped_low_success += 1
                continue
            record = _build_export_record(
                fp,
                row,
                fmt=args.format,
                prompt_catalog=prompt_catalog,
                include_noncode=args.include_noncode,
            )
            if record is None:
                dropped_noncode += 1
                continue
            f.write(json.dumps(record) + "\n")
            kept += 1

    print(
        f"[distill-export] cache={_cache_path()} "
        f"→ {out_path} ({kept} records, "
        f"{dropped_low_success} dropped below min_success={args.min_success}, "
        f"{dropped_noncode} dropped as non-code, "
        f"format={args.format})"
    )


if __name__ == "__main__":
    main()
