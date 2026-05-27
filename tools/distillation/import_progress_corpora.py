#!/usr/bin/env python3
"""
Import successful code-generation rows from benchmark progress logs.

Supported sources today:
  - EvalPlus HumanEval progress JSONL
  - BigCodeBench progress JSONL

Each imported row is normalized to:
  {"prompt", "completion", "metadata"}

We only keep syntactically valid Python completions and collapse
multiple successful variants of the same task to one representative
completion (default: the fastest successful attempt).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, Optional


_FENCE_RE = re.compile(r"```(?:python)?\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_python(text: str) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None
    match = _FENCE_RE.search(text)
    candidate = match.group(1) if match else text
    candidate = candidate.strip()
    if candidate.startswith("python\n"):
        candidate = candidate.split("\n", 1)[1].strip()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(candidate)
    except SyntaxError:
        return None
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom, ast.ClassDef)) for node in tree.body):
        return None
    return candidate


def _iter_success_rows_from_progress(progress_path: Path) -> Iterable[dict]:
    for line in progress_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") != "task_complete" or not row.get("success"):
            continue
        task_result = row.get("task_result") or {}
        attempt_details = task_result.get("attempt_details") or []
        if not attempt_details:
            continue
        detail = attempt_details[-1]
        prompt = detail.get("prompt")
        completion = _extract_python(detail.get("response_text", ""))
        if not prompt or not completion:
            continue
        yield {
            "task_name": row.get("task_name") or task_result.get("name"),
            "prompt": prompt,
            "completion": completion,
            "elapsed_seconds": detail.get("elapsed_seconds", row.get("elapsed_seconds")),
            "model": row.get("model"),
            "source_file": str(progress_path),
            "dataset": row.get("dataset") or task_result.get("category"),
            "approach": row.get("approach"),
            "verification": detail.get("verification"),
            "detail_metadata": detail.get("metadata") or {},
        }


def _iter_success_rows_from_summary(summary_path: Path) -> Iterable[dict]:
    payload = json.loads(summary_path.read_text())
    for section_name in ("baseline", "some"):
        section = payload.get(section_name)
        if not isinstance(section, dict):
            continue
        task_results = section.get("task_results") or []
        for task_result in task_results:
            if not task_result.get("success"):
                continue
            attempt_details = task_result.get("attempt_details") or []
            if not attempt_details:
                continue
            detail = attempt_details[-1]
            prompt = detail.get("prompt")
            completion = _extract_python(detail.get("response_text", ""))
            if not prompt or not completion:
                continue
            yield {
                "task_name": task_result.get("name"),
                "prompt": prompt,
                "completion": completion,
                "elapsed_seconds": detail.get("elapsed_seconds", task_result.get("elapsed_seconds")),
                "model": payload.get("model"),
                "source_file": str(summary_path),
                "dataset": task_result.get("category") or payload.get("dataset") or payload.get("subset"),
                "approach": section_name,
                "verification": detail.get("verification"),
                "detail_metadata": detail.get("metadata") or {},
            }


def import_progress_files(paths: list[Path], summary_paths: Optional[list[Path]] = None) -> tuple[list[dict], dict]:
    summary_paths = summary_paths or []
    best_by_task: Dict[str, dict] = {}
    stats = {
        "progress_files": [str(p) for p in paths],
        "summary_files": [str(p) for p in summary_paths],
        "success_rows_seen": 0,
        "rows_kept": 0,
        "tasks_with_multiple_successes": 0,
    }
    success_counts: Dict[str, int] = {}

    for path in paths:
        for row in _iter_success_rows_from_progress(path):
            stats["success_rows_seen"] += 1
            task_name = row["task_name"] or row["prompt"]
            success_counts[task_name] = success_counts.get(task_name, 0) + 1
            current = best_by_task.get(task_name)
            if current is None or (row.get("elapsed_seconds") or 10**9) < (current.get("elapsed_seconds") or 10**9):
                best_by_task[task_name] = row

    for path in summary_paths:
        for row in _iter_success_rows_from_summary(path):
            stats["success_rows_seen"] += 1
            task_name = row["task_name"] or row["prompt"]
            success_counts[task_name] = success_counts.get(task_name, 0) + 1
            current = best_by_task.get(task_name)
            if current is None or (row.get("elapsed_seconds") or 10**9) < (current.get("elapsed_seconds") or 10**9):
                best_by_task[task_name] = row

    stats["tasks_with_multiple_successes"] = sum(1 for c in success_counts.values() if c > 1)
    rows = []
    for task_name, row in sorted(best_by_task.items()):
        rows.append({
            "prompt": row["prompt"],
            "completion": row["completion"],
            "metadata": {
                "task_kind": "benchmark_codegen",
                "task_name": task_name,
                "source_dataset": row["dataset"],
                "source_model": row["model"],
                "approach": row["approach"],
                "elapsed_seconds": row["elapsed_seconds"],
                "verification": row["verification"],
                "success_variants": success_counts.get(task_name, 1),
                "source_file": row["source_file"],
                **row.get("detail_metadata", {}),
            },
        })
    stats["rows_kept"] = len(rows)
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--progress", action="append", default=[], type=Path,
                    help="progress JSONL file; may be repeated")
    ap.add_argument("--summary-json", action="append", default=[], type=Path,
                    help="benchmark summary JSON with baseline/some task_results; may be repeated")
    ap.add_argument("--out", required=True, type=Path,
                    help="output JSONL path")
    ap.add_argument("--manifest-out", default=None, type=Path,
                    help="optional manifest path")
    args = ap.parse_args()
    if not args.progress and not args.summary_json:
        ap.error("provide at least one --progress or --summary-json input")

    rows, stats = import_progress_files(args.progress, args.summary_json)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    if args.manifest_out is not None:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
