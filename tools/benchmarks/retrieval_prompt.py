#!/usr/bin/env python3
"""
Retrieval-augmented prompt builder.

Before the model generates, pull the top-K semantically-similar
verified solutions from our own cache and inline them as few-shot
context. The model then *conditions* on real previously-solved
problems — its generation itself becomes cache-aware, not just the
loop around it. This is the closest thing to "fused into inference"
achievable through a hosted API.

Usage:
    from retrieval_prompt import build_retrieval_prefix

    prefix = build_retrieval_prefix(examples, k=3, min_similarity=0.75)
    # prefix is either "" (no hits) or a few-shot block ending in "\n\n"
    final_prompt = prefix + original_prompt

The retrieval corpus is our shared TSV of verified solutions. As the
system solves more problems, the corpus grows and retrieval hit rate
improves — a closed learning loop where yesterday's solves condition
tomorrow's generation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_cache import semantic_lookup  # noqa: E402


def build_retrieval_prefix(
    examples: List[dict],
    k: int = 3,
    min_similarity: float = 0.75,
    max_code_chars: int = 400,
) -> str:
    """Return a few-shot block from similar cached solutions, or "" if
    the cache has no sufficiently-similar hits.

    The block is formatted as worked examples:

        # Similar verified solutions from cache:
        # Example 1 (sim=0.91):
        <code>
        # Example 2 (sim=0.87):
        <code>
        ...

    `max_code_chars` caps per-example length so a retrieved monster
    function doesn't blow out the prompt.
    """
    hits = semantic_lookup(
        examples, k=k, min_similarity=min_similarity,
        include_self=False,
    )
    if not hits:
        return ""

    lines = ["# Similar verified solutions retrieved from cache:", ""]
    for i, h in enumerate(hits, 1):
        code = h["code"]
        if len(code) > max_code_chars:
            code = code[:max_code_chars] + "\n# ...(truncated)"
        lines.append(f"# --- Example {i} (sim={h['similarity']:.2f}, "
                     f"wins={h['success_count']}) ---")
        lines.append(code)
        lines.append("")
    lines.append("# Your task — use the above as reference where "
                 "relevant, but write a fresh solution:")
    lines.append("")
    return "\n".join(lines)


def _main() -> None:
    """Quick CLI for inspecting what retrieval would surface for a query."""
    import argparse, json
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--examples", required=True,
                    help="JSON list of {inputs, expected}")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min-sim", type=float, default=0.75)
    args = ap.parse_args()

    examples = json.loads(args.examples)
    prefix = build_retrieval_prefix(examples, k=args.k,
                                     min_similarity=args.min_sim)
    if not prefix:
        print("# No retrieval hits above threshold.")
    else:
        print(prefix)


if __name__ == "__main__":
    _main()
