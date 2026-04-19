"""Mine eval JSONs for hard-fails and emit a WorkItem queue.

An eval JSON (produced by `humaneval_runner` / `npcot_agent_runner` /
`mbpp_runner`) records per-problem pass/fail outcomes. The miner:

1. Loads the eval JSON.
2. Selects the hard-fails (``passed == False``).
3. Re-joins with the benchmark dataset (HumanEval / MBPP) to get the
   prompt, entry point, and test source.
4. Best-effort-parses the test source via :mod:`ast` to extract I/O pairs
   of the form ``assert candidate(*args, **kwargs) == expected`` (plus
   variants such as ``assert expected == candidate(...)``).
5. Emits a :class:`WorkItem` JSONL: one JSON object per line.

The I/O extraction is deliberately conservative: when an assertion cannot
be reduced to a concrete ``(args, kwargs, expected)`` triple at parse
time (e.g. it references a helper variable, uses randomness, or mutates
state), it is skipped. This keeps the downstream synthesizer focused on
the test cases it can actually use.

Usage::

    python3 -m ncpu.autoresearch.miner \\
        --eval training_results/realworld_vastai/humaneval_agent_4B.json \\
        --benchmark humaneval \\
        --out .nCPU_autoresearch/queue.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Optional

from ncpu.autoresearch.types import (
    DEFAULT_ARTIFACT_DIR,
    IoPair,
    WorkItem,
)


def _literal_value(node: ast.AST) -> Any:
    """Return the Python value of an AST node if it's a compile-time literal.

    Raises ``ValueError`` when the node is not a literal.
    """
    return ast.literal_eval(node)  # lets ast.literal_eval raise on failure


def extract_io_pairs(test_source: str, entry_point: str) -> list[IoPair]:
    """Extract ``assert candidate(args) == expected`` I/O pairs from a test script.

    ``entry_point`` is the function name the test is supposed to check. The
    convention in HumanEval/MBPP is that tests dispatch through a
    ``candidate`` parameter bound inside a ``check(candidate)`` function, so
    we accept both ``candidate(...)`` and ``entry_point(...)`` as the call
    site.
    """
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return []

    pairs: list[IoPair] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if not isinstance(test, ast.Compare):
            continue
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue
        left, right = test.left, test.comparators[0]

        def _is_target_call(n: ast.AST) -> bool:
            return (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id in ("candidate", entry_point)
            )

        if _is_target_call(left) and not _is_target_call(right):
            call, expected_node = left, right
        elif _is_target_call(right) and not _is_target_call(left):
            call, expected_node = right, left
        else:
            continue

        try:
            args = [_literal_value(a) for a in call.args]
        except (ValueError, SyntaxError):
            continue
        try:
            kwargs = {kw.arg: _literal_value(kw.value) for kw in call.keywords if kw.arg}
        except (ValueError, SyntaxError):
            continue
        try:
            expected = _literal_value(expected_node)
        except (ValueError, SyntaxError):
            continue

        pairs.append(IoPair(args=args, kwargs=kwargs, expected=expected))
    return pairs


def _load_humaneval() -> dict[str, dict[str, Any]]:
    """Load the HumanEval test split keyed by task_id."""
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    return {row["task_id"]: dict(row) for row in ds}


def _load_mbpp() -> dict[str, dict[str, Any]]:
    """Load MBPP test split keyed by `mbpp/<task_id>` (matching runner convention)."""
    from datasets import load_dataset
    ds = load_dataset("mbpp", split="test")
    out: dict[str, dict[str, Any]] = {}
    for row in ds:
        tid = f"mbpp/{row['task_id']}"
        out[tid] = {
            "task_id": tid,
            "prompt": row["text"] + "\n\n" + row["test_list"][0] if row.get("test_list") else row["text"],
            "test": "\n".join(row.get("test_list", [])) + "\n",
            "canonical_solution": row.get("code", ""),
            "entry_point": _guess_entry_point_from_mbpp(row),
        }
    return out


def _guess_entry_point_from_mbpp(row: dict[str, Any]) -> str:
    """MBPP rows don't carry entry_point; infer from canonical code."""
    code = row.get("code") or ""
    for line in code.splitlines():
        if line.startswith("def "):
            return line.split("def ", 1)[1].split("(")[0].strip()
    # Fallback: try test asserts.
    for t in row.get("test_list", []):
        if "assert " in t:
            rest = t.split("assert ", 1)[1]
            if "(" in rest:
                return rest.split("(")[0].strip()
    return "candidate"


def mine(
    *,
    eval_json_path: Path,
    benchmark: str,
    out_path: Path,
    min_io_pairs: int = 2,
    task_filter: Optional[set[str]] = None,
) -> dict[str, int]:
    """Read an eval JSON, emit WorkItems for its hard-fails.

    Returns a counters dict describing what happened.
    """
    with open(eval_json_path) as fh:
        eval_data = json.load(fh)

    per_problem = eval_data.get("per_problem", [])
    hard_fails = [p for p in per_problem if not p.get("passed")]

    if benchmark == "humaneval":
        by_id = _load_humaneval()
    elif benchmark == "mbpp":
        by_id = _load_mbpp()
    else:
        raise ValueError(f"unknown benchmark: {benchmark}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = {
        "hard_fails_total": len(hard_fails),
        "written": 0,
        "skipped_no_task": 0,
        "skipped_no_io_pairs": 0,
    }

    seen_ids: set[str] = set()
    if out_path.exists():
        with open(out_path) as fh:
            for line in fh:
                try:
                    seen_ids.add(json.loads(line)["task_id"])
                except Exception:
                    pass
        count["preexisting"] = len(seen_ids)

    with open(out_path, "a") as fh:
        for rec in hard_fails:
            tid = rec["task_id"]
            if task_filter is not None and tid not in task_filter:
                continue
            if tid in seen_ids:
                continue
            task = by_id.get(tid)
            if task is None:
                count["skipped_no_task"] += 1
                continue
            pairs = extract_io_pairs(task["test"], task.get("entry_point", "candidate"))
            if len(pairs) < min_io_pairs:
                count["skipped_no_io_pairs"] += 1
                continue
            item = WorkItem(
                task_id=tid,
                source_benchmark=benchmark,
                prompt=task["prompt"],
                entry_point=task.get("entry_point", "candidate"),
                test_source=task["test"],
                io_pairs=pairs,
                canonical_solution=task.get("canonical_solution"),
                priority=1.0 + 0.1 * len(pairs),
                provenance={
                    "eval_json": str(eval_json_path),
                    "attempts_in_eval": rec.get("total_attempts", 1),
                },
            )
            fh.write(json.dumps(item.to_dict()) + "\n")
            count["written"] += 1
    return count


def load_queue(path: Path) -> list[WorkItem]:
    """Load a queue from JSONL."""
    items: list[WorkItem] = []
    if not path.exists():
        return items
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(WorkItem.from_dict(json.loads(line)))
    items.sort(key=lambda it: it.priority, reverse=True)
    return items


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--eval", dest="eval_json", type=Path, required=True)
    p.add_argument("--benchmark", choices=["humaneval", "mbpp"], required=True)
    p.add_argument("--out", dest="out_path", type=Path,
                   default=DEFAULT_ARTIFACT_DIR / "queue.jsonl")
    p.add_argument("--min-io-pairs", type=int, default=2)
    p.add_argument("--task", action="append", default=None,
                   help="Filter to specific task IDs (repeatable).")
    args = p.parse_args(argv)
    counters = mine(
        eval_json_path=args.eval_json,
        benchmark=args.benchmark,
        out_path=args.out_path,
        min_io_pairs=args.min_io_pairs,
        task_filter=set(args.task) if args.task else None,
    )
    print(json.dumps(counters, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
