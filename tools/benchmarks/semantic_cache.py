#!/usr/bin/env python3
"""
Semantic cache layer over llm_solution_cache.

Problem: our current cache is keyed on exact-fingerprint of the
examples. Two problems with identical I/O shapes hit; two problems
with *similar* shapes (same function, different argument values) miss.

This module adds approximate nearest-neighbor lookup over the cache.
For a new problem, it finds the K most similar cached problems by
example-space L2 distance, returns them as candidate solutions. The
caller still runs verify_against_tests to confirm the similar
solution actually works — semantic lookup is advisory.

**No network calls, no extra model downloads.** Similarity uses a
hash-based cheap-embedding on the examples themselves:
  - n_args, n_examples
  - mean/range of input values
  - mean/range of expected outputs
  - output parity (is_even, is_positive fractions)

This is the same idea as the Rust-side meta_learner but built in
Python so the benchmark tooling can use it without crossing the
process boundary. Accuracy is lower than a real embedding but latency
is near-zero and it works entirely offline.

If a team wants richer similarity (token-level cosine over the code
string, for example), the interface is `embed(examples) -> List[float]`
and the rest is drop-in. Swap in sentence-transformers locally, an
OpenAI embeddings API, or whatever fits your deployment.

Usage:
    from semantic_cache import semantic_lookup

    candidates = semantic_lookup(
        examples=problem_examples,
        k=3,
        min_similarity=0.85,
    )
    for cand in candidates:
        ok = verify(cand["code"], problem)
        if ok:
            return cand["code"]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_solution_cache import (  # noqa: E402
    _load_all, _cache_path, fingerprint_examples,
)


def _examples_embed(examples: List[dict]) -> List[float]:
    """8-dim hand-designed feature vector. Good enough for "same
    function shape" similarity; not a learned embedding. Slots:

      0  n_args
      1  n_examples
      2  mean |input_0|
      3  range(input_0)      — max(input_0) − min(input_0)
      4  mean output
      5  range(output)
      6  fraction outputs ≥ 0
      7  output / input_0 ratio (best-effort; 0 when arg0 is 0)
    """
    f = [0.0] * 8
    if not examples:
        return f
    first_inputs = examples[0].get("inputs", [])
    f[0] = float(len(first_inputs))
    f[1] = float(len(examples))

    input0s = []
    for ex in examples:
        ins = ex.get("inputs", [])
        if ins and isinstance(ins[0], (int, float)):
            input0s.append(float(ins[0]))
    if input0s:
        f[2] = sum(abs(v) for v in input0s) / len(input0s)
        f[3] = max(input0s) - min(input0s)

    outs = [float(ex.get("expected", 0)) for ex in examples
            if isinstance(ex.get("expected"), (int, float))]
    if outs:
        f[4] = sum(outs) / len(outs)
        f[5] = max(outs) - min(outs)
        f[6] = sum(1 for v in outs if v >= 0) / len(outs)

    if input0s and sum(input0s) != 0:
        f[7] = sum(outs) / sum(input0s) if outs else 0.0

    return f


def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity. -1..1; 1 = identical direction."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _l2_normalized_distance(a: List[float], b: List[float]) -> float:
    """L2 distance on unit-normalized vectors. 0 = identical,
    √2 = opposite. Return 1 - dist/√2 for a similarity in [0, 1]."""
    if not a or not b or len(a) != len(b):
        return 0.0
    # Normalize.
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    norm_a = [x / na for x in a]
    norm_b = [y / nb for y in b]
    d = math.sqrt(sum((x - y) ** 2 for x, y in zip(norm_a, norm_b)))
    return max(0.0, 1.0 - d / math.sqrt(2))


def semantic_lookup(
    examples: List[dict],
    k: int = 3,
    min_similarity: float = 0.85,
    include_self: bool = False,
) -> List[Dict[str, Any]]:
    """Find the K most-similar cached entries to the query examples.

    Returns a list of `{fingerprint, model, code, success_count,
    similarity}` dicts, sorted by similarity desc. Empty if the cache
    is empty or no entry exceeds `min_similarity`.

    The caller should still verify the returned code — semantic match
    on I/O shape doesn't prove the code is correct for a DIFFERENT
    problem with the same shape. This is retrieval, not solution."""
    query_fp = fingerprint_examples(examples)
    query_vec = _examples_embed(examples)

    entries = _load_all()
    candidates: List[Dict[str, Any]] = []

    for fp, row in entries.items():
        if fp == query_fp and not include_self:
            continue
        # Prefer stored examples when present — query and candidate are
        # then embedded in the same feature space, so L2 distance is
        # meaningful. Fall back to code-shape features for legacy rows.
        row_examples = row.get("examples") or []
        if row_examples:
            cand_vec = _examples_embed(row_examples)
        else:
            cand_vec = _vec_from_code(row["code"])
        sim = _l2_normalized_distance(query_vec, cand_vec)
        if sim < min_similarity:
            continue
        candidates.append({
            "fingerprint": fp,
            "model": row["model"],
            "code": row["code"],
            "success_count": row["success_count"],
            "similarity": round(sim, 4),
        })

    candidates.sort(key=lambda c: -c["similarity"])
    return candidates[:k]


def _vec_from_code(code: str) -> List[float]:
    """Very coarse 8-dim feature from a code string. Mirror of the
    slots in `_examples_embed` so the L2 distance makes rough sense:

      0  n_params
      1  lines_of_code
      2  (unused — uses signature param count as proxy)
      3  0
      4  contains 'return' (1/0)
      5  contains 'if'     (1/0)
      6  contains 'for'    (1/0)
      7  contains '%' or '//' (integer math indicator)
    """
    f = [0.0] * 8
    lines = [l for l in code.splitlines() if l.strip()]
    f[1] = float(len(lines))
    # Count params in the first def line.
    first = lines[0] if lines else ""
    if first.lstrip().startswith("def ") and "(" in first and ")" in first:
        params_str = first[first.index("(") + 1 : first.index(")")]
        if params_str.strip():
            f[0] = float(len([p for p in params_str.split(",") if p.strip()]))
    f[4] = 1.0 if "return" in code else 0.0
    f[5] = 1.0 if " if " in code or code.lstrip().startswith("if ") else 0.0
    f[6] = 1.0 if " for " in code or code.lstrip().startswith("for ") else 0.0
    f[7] = 1.0 if "%" in code or "//" in code else 0.0
    return f


def _main() -> None:
    """Quick CLI to inspect the semantic cache's similarity ranking."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--examples", required=True,
                    help="JSON list of {inputs, expected}")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-sim", type=float, default=0.0)
    args = ap.parse_args()

    examples = json.loads(args.examples)
    hits = semantic_lookup(examples, k=args.k, min_similarity=args.min_sim)
    print(f"# semantic_lookup (cache={_cache_path()})\n")
    print(f"Query: {len(examples)} examples → "
          f"fingerprint={fingerprint_examples(examples)[:12]}…")
    print(f"Matches: {len(hits)}\n")
    for h in hits:
        preview = h["code"].splitlines()[0] if h["code"] else ""
        print(f"sim={h['similarity']:.3f}  "
              f"wins={h['success_count']:<3}  "
              f"model={h['model'][:30]:<30}  "
              f"{preview[:60]}")


if __name__ == "__main__":
    _main()
