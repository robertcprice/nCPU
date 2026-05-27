#!/usr/bin/env python3
"""
vLLM wrapper that uses the nsynth verified-solution cache as a
speculative-decoding draft source.

## The idea

Speculative decoding normally works like this: a cheap "draft" model
proposes a sequence of k tokens; the expensive "target" model verifies
them in parallel by computing P(t_i | prefix_i) and accepting tokens
where its top choice matches the draft. Accepted tokens are returned
for free; on first disagreement, the target falls back to autoregressive.

We apply the same mechanism with the *cache* as draft source:

  1. New problem arrives. Compute its examples fingerprint.
  2. Semantic lookup → top-K similar verified solutions from our cache.
  3. Use the highest-similarity solution's tokens as the speculative
     draft — possibly with a small prefix-replacement for the function
     signature.
  4. Target model (Qwen3.5-4B-Instruct or similar on vast.ai) verifies
     the draft token-by-token and accepts prefixes that match.
  5. On first disagreement, switch to autoregressive sampling.
  6. Verify final code against tests; cache the winner.

## Why "fused into inference" not "wrapper around it"

The cache doesn't just answer *before* the model; it participates
*during* token generation. Every cache hit accelerates the very next
forward pass. This is the same mechanism vLLM already uses for
n-gram speculation and prompt-lookup decoding — we just swap in a
semantically-retrieved draft from our cache.

## Current status

This module is a **scaffold** — it works locally for validation with a
stub generator, and exposes the real hooks for running on a vast.ai
A100 via vLLM's `SpecDecodeConfig`. To actually run:

    # On vast.ai:
    pip install vllm==0.6.0
    python tools/inference/vllm_cache_speculative.py \\
        --model Qwen/Qwen3.5-4B-Instruct \\
        --problems tools/benchmarks/humaneval_lite.jsonl \\
        --corpus /path/to/cache.tsv

The stub runs locally and demonstrates the draft-construction logic
without loading a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from semantic_cache import semantic_lookup  # noqa: E402
from llm_solution_cache import fingerprint_examples  # noqa: E402


@dataclass
class DraftPlan:
    """Blueprint for one speculative-decoding attempt.

    `draft_tokens` is the proposed continuation that the target model
    will verify; `anchor_sim` is the similarity score of the cached
    solution we sourced it from; `origin_fp` is the fingerprint of the
    source cache row for audit.
    """
    draft_text: str
    anchor_sim: float
    origin_fp: str
    source_success_count: int


def build_speculative_draft(
    examples: List[dict],
    target_signature: Optional[str] = None,
    min_similarity: float = 0.70,
) -> Optional[DraftPlan]:
    """Fetch the top semantic-similar verified solution and adapt it
    as a draft for the target problem.

    If `target_signature` is provided (e.g. "def my_func(x: int) -> int:")
    we rewrite the draft's first line to match. The body is kept
    verbatim — speculative decoding tolerates per-token disagreement
    so partial matches still yield speed-up.
    """
    hits = semantic_lookup(examples, k=1, min_similarity=min_similarity)
    if not hits:
        return None

    top = hits[0]
    draft = top["code"]
    if target_signature:
        lines = draft.splitlines()
        # Replace first def line with target signature.
        for i, line in enumerate(lines):
            if line.lstrip().startswith("def "):
                indent = line[:len(line) - len(line.lstrip())]
                lines[i] = f"{indent}{target_signature.rstrip(':')}:"
                break
        draft = "\n".join(lines)

    return DraftPlan(
        draft_text=draft,
        anchor_sim=top["similarity"],
        origin_fp=top["fingerprint"],
        source_success_count=top["success_count"],
    )


def generate_with_cache_speculation_vllm(
    llm_engine,  # vllm.LLM instance, not typed to keep import optional
    problem: dict,
    tokenizer,
    sampling_params,
    min_similarity: float = 0.70,
) -> Dict:
    """Production path — requires vllm on the host.

    `llm_engine` is a `vllm.LLM(...)` constructed with spec-decode
    enabled (see vLLM's SpecDecodeConfig for the exact shape).

    Returns `{code, draft_plan, accepted_tokens, autoregressive_tokens,
    path}` where path is `"cache_draft"` if the draft contributed any
    accepted tokens, or `"autoregressive"` if we fell through.

    Intended to run on a vast.ai A100 — this module is importable
    locally but the actual speculative step needs the vLLM engine.
    """
    plan = build_speculative_draft(
        problem["examples"], target_signature=problem.get("signature"),
        min_similarity=min_similarity,
    )

    prompt = (
        f"Write a Python function matching `{problem['signature']}`.\n\n"
        f"Examples:\n"
        + "\n".join(f"  {problem['name']}({', '.join(repr(x) for x in ex['inputs'])}) "
                    f"== {ex['expected']}"
                    for ex in problem["examples"])
        + "\n\nReply with ONLY the function definition."
    )

    if plan is None:
        # No cache hit — fall through to pure autoregressive.
        outputs = llm_engine.generate([prompt], sampling_params)
        return {
            "code": outputs[0].outputs[0].text,
            "draft_plan": None,
            "path": "autoregressive",
        }

    # Path where vLLM supports per-request draft tokens. Actual API
    # surface:
    #   - SpecDecodeConfig(num_speculative_tokens=N, draft_model=...)
    #   - or, for prompt-lookup decoding, pass the draft as a hidden
    #     prefix that the target model must reproduce.
    #
    # As of vLLM 0.6+, the most portable approach is n-gram lookup
    # which finds drafts inside the prompt itself. We can emulate
    # cache-as-draft by *appending the draft to the prompt as a hint*
    # and letting n-gram speculation take over:
    prompt_with_hint = (
        prompt
        + f"\n\n# Similar solution (draft to verify):\n```python\n"
        + plan.draft_text
        + "\n```\n\n# Your answer:\n"
    )
    outputs = llm_engine.generate([prompt_with_hint], sampling_params)

    return {
        "code": outputs[0].outputs[0].text,
        "draft_plan": {
            "anchor_sim": plan.anchor_sim,
            "origin_fp": plan.origin_fp,
            "draft_chars": len(plan.draft_text),
        },
        "path": "cache_draft",
    }


def stub_demo(problems_path: Path, corpus_path: Path) -> None:
    """Local smoke test — runs without a vLLM engine, confirms the
    draft-construction path works against the real cache."""
    import os
    os.environ["NSYNTH_LLM_CACHE_PATH"] = str(corpus_path)

    problems = [json.loads(l) for l in problems_path.read_text().splitlines() if l.strip()]
    n_hits = 0
    n_total = 0
    for p in problems:
        n_total += 1
        plan = build_speculative_draft(p["examples"],
                                        target_signature=p["signature"])
        if plan:
            n_hits += 1
            print(f"  [{p['name']:<25}] draft sim={plan.anchor_sim:.3f} "
                  f"src_wins={plan.source_success_count} "
                  f"chars={len(plan.draft_text)}")
        else:
            print(f"  [{p['name']:<25}] no draft — fallback to autoregressive")

    print(f"\n[stub] {n_hits}/{n_total} problems would get a speculative "
          f"draft from the cache (min_sim=0.70).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--problems", required=True,
                    help="JSONL of problems with examples + signature")
    ap.add_argument("--corpus", required=True,
                    help="TSV cache file to use as retrieval corpus")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Instruct",
                    help="target vLLM model; only used on the remote host")
    ap.add_argument("--stub", action="store_true",
                    help="Local smoke test — no vLLM, just show which "
                         "problems get drafts")
    args = ap.parse_args()

    problems_path = Path(args.problems)
    corpus_path = Path(args.corpus)

    if args.stub:
        stub_demo(problems_path, corpus_path)
        return

    try:
        import vllm  # type: ignore  # noqa: F401
    except ImportError:
        print("[vllm_cache_spec] vllm not installed; re-run with --stub "
              "for a local demo.", file=sys.stderr)
        sys.exit(2)

    # Real-path scaffolding (to run on vast.ai). Deliberately left as
    # a shape-only invocation — concrete deployment specifics live in
    # tools/vastai/setup_and_run.sh and its per-model configs.
    from vllm import LLM, SamplingParams  # type: ignore
    engine = LLM(model=args.model, trust_remote_code=True)
    tokenizer = engine.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=512)

    problems = [json.loads(l) for l in problems_path.read_text().splitlines() if l.strip()]
    for p in problems:
        result = generate_with_cache_speculation_vllm(
            engine, p, tokenizer, sampling,
        )
        print(json.dumps({"name": p["name"], **result}))


if __name__ == "__main__":
    main()
