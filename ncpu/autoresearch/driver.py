"""Production autoresearch driver — loads the model, wires LLM-resample, runs.

Purpose: on a GPU, run the autoresearch cascade with the LLM-resample
solver installed so hard-fails get a real chance at being solved. This
is the binary the vast.ai VM actually runs. It's factored out of
``cli.py`` because it has heavy imports (torch, transformers) we don't
want the lightweight CLI path to carry.

Usage::

    python -m ncpu.autoresearch.driver \\
        --model Qwen/Qwen3.5-4B \\
        --queue .nCPU_autoresearch/humaneval_queue.jsonl \\
        --solved .nCPU_autoresearch/solved_programs.jsonl \\
        --wall-seconds 1800 \\
        --max-problems 30 \\
        --library /workspace/checkpoints/npcot_qwen3.5-4B_library.json \\
        --coprocessor-checkpoint /workspace/checkpoints/npcot_qwen3.5-4B.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from ncpu.autoresearch.cascade import CascadeConfig
from ncpu.autoresearch.llm_resample import make_llm_resampler
from ncpu.autoresearch.runner import run_session
from ncpu.autoresearch.types import Budget


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--solved", type=Path, required=True)
    p.add_argument("--status", type=Path, default=None)
    p.add_argument("--library", type=Path, default=None)
    p.add_argument("--coprocessor-checkpoint", type=Path, default=None)
    p.add_argument("--target-layers", default="-2,-1")
    p.add_argument("--array-max-len", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--wall-seconds", type=float, default=1800.0)
    p.add_argument("--max-problems", type=int, default=30)
    p.add_argument("--max-cost-usd", type=float, default=1.0)
    p.add_argument("--per-problem-seconds", type=float, default=120.0)
    p.add_argument("--temperatures", default="0.3,0.5,0.7,0.9")
    p.add_argument("--samples-per-temp", type=int, default=4)
    p.add_argument("--include-templates-first", action="store_true",
                   help="Try local template_match before LLM-resample (cheap first).")
    args = p.parse_args(argv)

    from ncpu.self_optimizing.humaneval_runner import (
        HumanEvalConfig,
        _extract_code,
        generate_solution,
        load_model_with_optional_npcot,
    )

    use_npcot = args.library is not None and args.coprocessor_checkpoint is not None
    he_cfg = HumanEvalConfig(
        model=args.model,
        library_path=args.library,
        coprocessor_checkpoint=args.coprocessor_checkpoint,
        target_layers=[int(x) for x in args.target_layers.split(",") if x.strip()],
        array_max_len=args.array_max_len,
        array_thought_max_gate=0.05,
        max_problems=0,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        use_npcot=use_npcot,
    )

    print(f"[driver] loading model {args.model} (npcot={use_npcot})", flush=True)
    model, tokenizer, device, meta = load_model_with_optional_npcot(he_cfg)
    print(f"[driver] loaded on {device}: {meta}", flush=True)

    def _gen(prompt: str, temperature: float, max_new_tokens: int) -> str:
        return generate_solution(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens, temperature=temperature, device=device,
        )

    temps = tuple(float(t) for t in args.temperatures.split(","))
    resampler = make_llm_resampler(
        generate_fn=_gen,
        extract_code_fn=_extract_code,
        temperatures=temps,
        samples_per_temp=args.samples_per_temp,
    )

    solver_names = []
    if args.include_templates_first:
        solver_names.append("template_match")
    solver_names.append("llm_resample")

    cfg = CascadeConfig(
        solver_names=solver_names,
        per_solver_seconds=args.per_problem_seconds,
        extra_solvers={"llm_resample": resampler},
    )
    budget = Budget(
        wall_seconds=args.wall_seconds,
        max_cost_usd=args.max_cost_usd,
        max_problems=args.max_problems,
        per_problem_seconds=args.per_problem_seconds,
    )

    def _progress(result, report):
        tag = f"SOLVED by {result.solver}" if result.solved else "unsolved"
        print(
            f"[driver] {report.problems_attempted}/{report.problems_attempted}: "
            f"{result.task_id}: {tag} "
            f"(cumulative solved={report.problems_solved}, "
            f"wall={report.wall_seconds:.0f}s)",
            flush=True,
        )

    report = run_session(
        queue_path=args.queue,
        solved_path=args.solved,
        budget=budget,
        cascade_config=cfg,
        status_path=args.status,
        on_result=_progress,
    )
    print("\n[driver] done.")
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
