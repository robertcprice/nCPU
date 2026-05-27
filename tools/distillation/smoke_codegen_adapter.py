#!/usr/bin/env python3
"""
Cheap post-train smoke gate for coding adapters.

This is not a benchmark. It is a fast format/behavior sanity check meant
to catch the two failure modes we saw in practice:

1. the adapter stops emitting extractable Python functions
2. the adapter degenerates into special-token spam such as <tool_call>
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "tools" / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

from local_model_adapter import LocalModelClient  # noqa: E402
from run_humaneval_full import build_llm_prompt, extract_full_function  # noqa: E402
from run_mbpp import MBPPProblem, build_initial_prompt, derive_fn_name, extract_python  # noqa: E402


def _call(client: LocalModelClient, model: str, prompt: str, max_tokens: int) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if hasattr(block, "text"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--model", default="mlx-community/Qwen3-4B-Instruct-2507-4bit")
    ap.add_argument("--backend", default="mlx", choices=["mlx", "hf", "openai"])
    ap.add_argument("--adapter-routing", default="always",
                    choices=["always", "never", "utility_only"])
    ap.add_argument("--humaneval-limit", type=int, default=3)
    ap.add_argument("--mbpp-limit", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    client = LocalModelClient(
        backend=args.backend,
        model=args.model,
        adapter_path=args.adapter_path,
        adapter_routing=args.adapter_routing,
    )

    he_ds = load_dataset("openai_humaneval", split="test")
    mbpp_ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

    he_extractable = 0
    mbpp_extractable = 0
    tool_call_spam = 0

    print("== HumanEval ==")
    for i in range(args.humaneval_limit):
        problem = dict(he_ds[i])
        text = _call(client, args.model, build_llm_prompt(problem), args.max_tokens)
        extractable = extract_full_function(text, problem["entry_point"]) is not None
        if "<tool_call>" in text or "</tool_call>" in text:
            tool_call_spam += 1
        if extractable:
            he_extractable += 1
        head = text[:120].replace("\n", "\\n")
        print(f"{problem['task_id']}: extractable={extractable} head={head}")

    print("\n== MBPP ==")
    used = 0
    i = 0
    while used < args.mbpp_limit and i < len(mbpp_ds):
        row = dict(mbpp_ds[i])
        i += 1
        fn_name = derive_fn_name(row["test_list"])
        if fn_name is None:
            continue
        prompt = build_initial_prompt(
            MBPPProblem(
                task_id=row["task_id"],
                text=row["prompt"],
                test_list=row["test_list"],
                fn_name=fn_name,
                examples=[],
            )
        )
        text = _call(client, args.model, prompt, args.max_tokens)
        extractable = extract_python(text, fn_name) is not None
        if "<tool_call>" in text or "</tool_call>" in text:
            tool_call_spam += 1
        if extractable:
            mbpp_extractable += 1
        head = text[:120].replace("\n", "\\n")
        print(f"{row['task_id']}: extractable={extractable} head={head}")
        used += 1

    min_he = max(1, math.ceil(args.humaneval_limit * 0.67))
    min_mbpp = max(1, math.ceil(args.mbpp_limit * 0.67))

    print("\n== Summary ==")
    print(f"HumanEval extractable: {he_extractable}/{args.humaneval_limit}")
    print(f"MBPP extractable: {mbpp_extractable}/{args.mbpp_limit}")
    print(f"Tool-call spam outputs: {tool_call_spam}")

    if tool_call_spam or he_extractable < min_he or mbpp_extractable < min_mbpp:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
