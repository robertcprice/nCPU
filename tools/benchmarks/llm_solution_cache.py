#!/usr/bin/env python3
"""
Persistent LLM-solution cache shared by the hybrid and llm-only runners.

When the LLM (any model) produces a Python function that passes the
problem's verification, we store (fingerprint, python_code) in a TSV at
`~/.nsynth_llm_solutions.tsv`. Subsequent hybrid runs that hit the same
fingerprint skip both nsynth's gradient stack AND the LLM API call —
the cached Python code is returned directly, ~0ms.

The cache is complementary to nsynth's own `solved_cache`:
  - nsynth's cache   → Mog programs (source DSL, transpile to any lang)
  - this LLM cache   → Python code verified against the problem

Schema (one row per line, `\\t`-separated):
    fingerprint \\t model \\t success_count \\t last_used_at \\t code_b64_like
        [\\t examples_json] [\\t question_text] [\\t metadata_json]

Where `code_b64_like` escapes newlines/tabs via `\\n`, `\\t`, `\\\\`
to keep the row single-line (same encoding the Rust solved_cache uses).
Trailing columns are optional for backward compatibility: legacy files
have 5 columns, retrieval-augmented files have 6, GSM8K text-retrieval
rows have 7, and richer coding-distillation rows can carry an 8th
metadata column.

Usage as a library:
    from llm_solution_cache import lookup, record, fingerprint_examples
    fp = fingerprint_examples(problem_examples)
    cached = lookup(fp)
    if cached is None:
        code = call_llm(...)
        record(fp, 'claude-haiku-4-5-20251001', code)

Usage as a CLI (for inspection):
    python3 tools/benchmarks/llm_solution_cache.py --list
    python3 tools/benchmarks/llm_solution_cache.py --purge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional


def _cache_path() -> Path:
    override = os.environ.get("NSYNTH_LLM_CACHE_PATH")
    if override == "":
        return Path(os.devnull)
    if override:
        return Path(override)
    home = os.environ.get("HOME", ".")
    return Path(home) / ".nsynth_llm_solutions.tsv"


def fingerprint_examples(examples: list) -> str:
    """Deterministic hash of the I/O shape. Matches nsynth's fingerprint
    semantics closely enough to correlate entries across the two caches.
    Format: sha256 of JSON(examples) truncated to 16 hex chars."""
    serialised = json.dumps(examples, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()[:32]


def fingerprint_humaneval_task(task_id: str, entry_point: str, prompt: str) -> str:
    """Deterministic hash for a HumanEval problem. Uses (task_id, entry,
    prompt) so minor prompt edits invalidate the cache — the expected
    behaviour of a strict verification-based cache."""
    payload = f"{task_id}|{entry_point}|{prompt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def _encode(code: str) -> str:
    return (
        code.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
    )


def _decode(enc: str) -> str:
    out = []
    i = 0
    while i < len(enc):
        c = enc[i]
        if c == "\\" and i + 1 < len(enc):
            nxt = enc[i + 1]
            if nxt == "n":
                out.append("\n"); i += 2; continue
            if nxt == "t":
                out.append("\t"); i += 2; continue
            if nxt == "r":
                out.append("\r"); i += 2; continue
            if nxt == "\\":
                out.append("\\"); i += 2; continue
        out.append(c)
        i += 1
    return "".join(out)


def _load_all() -> dict:
    """Read the cache.

    Supported row shapes:
      - 5 cols: legacy cache
      - 6 cols: examples JSON
      - 7 cols: examples JSON + question text
      - 8 cols: examples JSON + question text + metadata JSON

    The examples column lets semantic retrieval embed problems in the
    same feature space as a new query. The question column powers
    GSM8K-style text retrieval. The metadata column carries richer task
    provenance such as the original coding prompt for distillation."""
    path = _cache_path()
    if str(path) == os.devnull or not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 7)
        if len(parts) < 5:
            continue
        fp, model, sc, lu, code_enc = parts[:5]
        examples: list = []
        question = ""
        metadata: dict = {}
        if len(parts) >= 6 and parts[5]:
            try:
                examples = json.loads(_decode(parts[5]))
                if not isinstance(examples, list):
                    examples = []
            except Exception:
                examples = []
        if len(parts) >= 7 and parts[6]:
            question = _decode(parts[6])
        if len(parts) >= 8 and parts[7]:
            try:
                metadata = json.loads(_decode(parts[7]))
                if not isinstance(metadata, dict):
                    metadata = {}
            except Exception:
                metadata = {}
        out[fp] = {
            "model": model,
            "success_count": int(sc) if sc.isdigit() else 0,
            "last_used_at": int(lu) if lu.isdigit() else 0,
            "code": _decode(code_enc),
            "examples": examples,
            "question": question,
            "metadata": metadata,
        }
    return out


def _save_all(entries: dict) -> None:
    path = _cache_path()
    if str(path) == os.devnull:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        for fp, row in entries.items():
            ex_enc = ""
            if row.get("examples"):
                try:
                    ex_enc = _encode(json.dumps(row["examples"], sort_keys=True))
                except Exception:
                    ex_enc = ""
            q_enc = _encode(row.get("question", "")) if row.get("question") else ""
            meta_enc = ""
            if row.get("metadata"):
                try:
                    meta_enc = _encode(json.dumps(row["metadata"], sort_keys=True))
                except Exception:
                    meta_enc = ""
            fields = [
                fp,
                row["model"],
                str(row["success_count"]),
                str(row["last_used_at"]),
                _encode(row["code"]),
            ]
            if ex_enc or q_enc or meta_enc:
                fields.append(ex_enc)
            if q_enc or meta_enc:
                if len(fields) == 5:
                    fields.append("")
                fields.append(q_enc)
            if meta_enc:
                if len(fields) == 5:
                    fields.extend(["", ""])
                elif len(fields) == 6:
                    fields.append("")
                fields.append(meta_enc)
            f.write("\t".join(fields) + "\n")
    tmp.replace(path)


def lookup(fp: str) -> Optional[dict]:
    """Return the cached entry for `fp`, or None. Callers should still
    *verify* the code against the current problem's test cases — the
    cache doesn't hold the tests, so a stale / mismatched code row is
    caught by the caller's verifier."""
    return _load_all().get(fp)


def record(
    fp: str,
    model: str,
    code: str,
    examples: Optional[list] = None,
    question: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    """Record a verified LLM solution. Increments success_count if the
    same (fp, model, code) already exists. Optional `examples` param
    persists the original I/O examples for downstream retrieval. When
    None, preserves any examples already on the row. Optional
    `question` and `metadata` fields behave the same way."""
    entries = _load_all()
    now = int(time.time())
    existing = entries.get(fp)
    if existing and existing["model"] == model and existing["code"] == code:
        existing["success_count"] += 1
        existing["last_used_at"] = now
        if examples is not None:
            existing["examples"] = examples
        if question is not None:
            existing["question"] = question
        if metadata is not None:
            existing["metadata"] = metadata
    else:
        entries[fp] = {
            "model": model,
            "success_count": 1,
            "last_used_at": now,
            "code": code,
            "examples": examples or [],
            "question": question or (existing.get("question", "") if existing else ""),
            "metadata": metadata or (existing.get("metadata", {}) if existing else {}),
        }
    _save_all(entries)


def size() -> int:
    return len(_load_all())


def purge() -> int:
    entries = _load_all()
    _save_all({})
    return len(entries)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true")
    group.add_argument("--purge", action="store_true")
    group.add_argument("--size", action="store_true")
    args = ap.parse_args()

    if args.list:
        entries = _load_all()
        print(f"# {len(entries)} cached LLM solutions at {_cache_path()}")
        for fp, row in sorted(entries.items(), key=lambda x: -x[1]["success_count"]):
            code_preview = row["code"].splitlines()[0] if row["code"].strip() else ""
            print(f"{fp}  wins={row['success_count']:<3}  model={row['model']:<30}  {code_preview[:60]}")
        return
    if args.purge:
        n = purge()
        print(f"purged {n} entries from {_cache_path()}")
        return
    if args.size:
        print(size())
        return
    ap.print_help()


if __name__ == "__main__":
    main()
