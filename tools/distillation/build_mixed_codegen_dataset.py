#!/usr/bin/env python3
"""
Build a mixed code-distillation dataset with source balancing.

This script exists to avoid repeating the failure mode we already saw:
training a strong utility-specialist adapter and then expecting it to
improve general coding when utility rows dominate the corpus.

Inputs may come from:
  - coding JSONL exports (`prompt` + `completion` or `code`)
  - utility JSONL exports
  - split directories with train/valid/test JSONL files

The builder:
  1. normalizes records into {"prompt", "completion", "metadata"}
  2. drops non-code completions
  3. de-duplicates prompt/completion pairs
  4. caps the utility share against coding rows
  5. emits train/valid/test JSONL plus a manifest
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _looks_like_python_function(text: str) -> bool:
    if not text or "def " not in text:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
    except SyntaxError:
        return False
    return any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body)


def _normalize_row(row: dict) -> Optional[dict]:
    prompt = row.get("prompt")
    completion = row.get("completion")
    if completion is None:
        completion = row.get("code")
    if prompt is None and isinstance(row.get("messages"), list):
        messages = row["messages"]
        if len(messages) >= 2:
            user = next((m.get("content") for m in messages if m.get("role") == "user"), None)
            assistant = next((m.get("content") for m in messages if m.get("role") == "assistant"), None)
            prompt = user
            completion = assistant if completion is None else completion
    if not isinstance(prompt, str) or not isinstance(completion, str):
        return None
    if not _looks_like_python_function(completion):
        return None
    metadata = dict(row.get("metadata") or {})
    return {
        "prompt": prompt,
        "completion": completion,
        "metadata": metadata,
    }


def _iter_jsonl(path: Path) -> Iterable[dict]:
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def _read_input_paths(paths: List[Path]) -> List[dict]:
    rows: List[dict] = []
    for path in paths:
        rows.extend(_iter_jsonl(path))
    return rows


def _read_split_dirs(dirs: List[Path]) -> List[dict]:
    rows: List[dict] = []
    for directory in dirs:
        for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
            path = directory / name
            if path.exists():
                rows.extend(_iter_jsonl(path))
    return rows


def _dedupe_rows(rows: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for row in rows:
        key = hashlib.sha256(
            (row["prompt"] + "\0" + row["completion"]).encode("utf-8")
        ).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _cap_utility_rows(
    coding_rows: List[dict],
    utility_rows: List[dict],
    max_utility_share: float,
    rng: random.Random,
) -> List[dict]:
    if not utility_rows:
        return []
    if not coding_rows:
        return list(utility_rows)
    max_allowed = int((len(coding_rows) * max_utility_share) / max(1e-9, 1.0 - max_utility_share))
    max_allowed = max(1, max_allowed)
    if len(utility_rows) <= max_allowed:
        return list(utility_rows)
    sampled = list(utility_rows)
    rng.shuffle(sampled)
    return sampled[:max_allowed]


def _split_rows(rows: List[dict], valid_fraction: float, test_fraction: float, rng: random.Random) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for row in rows:
        source_group = row["metadata"].get("source_group", "unknown")
        groups.setdefault(source_group, []).append(row)

    splits = {"train": [], "valid": [], "test": []}
    for source_group, source_rows in groups.items():
        rng.shuffle(source_rows)
        n = len(source_rows)
        n_valid = int(round(n * valid_fraction))
        n_test = int(round(n * test_fraction))
        if n >= 10:
            n_valid = max(1, n_valid)
            n_test = max(1, n_test)
        while n_valid + n_test >= n and n_valid > 0:
            n_valid -= 1
        while n_valid + n_test >= n and n_test > 0:
            n_test -= 1
        train_end = n - n_valid - n_test
        splits["train"].extend(source_rows[:train_end])
        splits["valid"].extend(source_rows[train_end:train_end + n_valid])
        splits["test"].extend(source_rows[train_end + n_valid:])
    for name in splits:
        rng.shuffle(splits[name])
    return splits


def _source_counts(rows: List[dict]) -> Dict[str, int]:
    return dict(sorted(Counter(r["metadata"].get("source_group", "unknown") for r in rows).items()))


def build_dataset(
    coding_jsonl: List[Path],
    utility_jsonl: List[Path],
    coding_split_dirs: List[Path],
    utility_split_dirs: List[Path],
    max_utility_share: float,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, List[dict]]:
    rng = random.Random(seed)

    raw_coding = _read_input_paths(coding_jsonl) + _read_split_dirs(coding_split_dirs)
    raw_utility = _read_input_paths(utility_jsonl) + _read_split_dirs(utility_split_dirs)

    coding_rows = []
    for row in raw_coding:
        norm = _normalize_row(row)
        if norm is None:
            continue
        norm["metadata"] = {**norm["metadata"], "source_group": "coding"}
        coding_rows.append(norm)

    utility_rows = []
    for row in raw_utility:
        norm = _normalize_row(row)
        if norm is None:
            continue
        norm["metadata"] = {**norm["metadata"], "source_group": "utility"}
        utility_rows.append(norm)

    coding_rows = _dedupe_rows(coding_rows)
    utility_rows = _dedupe_rows(utility_rows)
    utility_rows = _cap_utility_rows(coding_rows, utility_rows, max_utility_share, rng)

    combined = _dedupe_rows(coding_rows + utility_rows)
    return _split_rows(combined, valid_fraction, test_fraction, rng)


def write_splits(out_dir: Path, splits: Dict[str, List[dict]], args: argparse.Namespace) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        with (out_dir / f"{split_name}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    manifest = {
        "total_rows": sum(len(v) for v in splits.values()),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "source_counts": {k: _source_counts(v) for k, v in splits.items()},
        "max_utility_share": args.max_utility_share,
        "seed": args.seed,
        "coding_jsonl": [str(p) for p in args.coding_jsonl],
        "utility_jsonl": [str(p) for p in args.utility_jsonl],
        "coding_split_dirs": [str(p) for p in args.coding_split_dir],
        "utility_split_dirs": [str(p) for p in args.utility_split_dir],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--coding-jsonl", action="append", default=[],
                    type=Path, help="coding JSONL input; may be repeated")
    ap.add_argument("--utility-jsonl", action="append", default=[],
                    type=Path, help="utility JSONL input; may be repeated")
    ap.add_argument("--coding-split-dir", action="append", default=[],
                    type=Path, help="directory containing train/valid/test JSONL coding splits")
    ap.add_argument("--utility-split-dir", action="append", default=[],
                    type=Path, help="directory containing train/valid/test JSONL utility splits")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="output directory for train/valid/test JSONL")
    ap.add_argument("--max-utility-share", type=float, default=0.40,
                    help="maximum allowed utility share in the mixed corpus")
    ap.add_argument("--valid-fraction", type=float, default=0.10)
    ap.add_argument("--test-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    splits = build_dataset(
        coding_jsonl=args.coding_jsonl,
        utility_jsonl=args.utility_jsonl,
        coding_split_dirs=args.coding_split_dir,
        utility_split_dirs=args.utility_split_dir,
        max_utility_share=args.max_utility_share,
        valid_fraction=args.valid_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    manifest = write_splits(args.out_dir, splits, args)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
