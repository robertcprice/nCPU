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
from ncpu.autoresearch.distiller import io_pair_int_array, translate_to_5tuple
from ncpu.autoresearch.llm_resample import make_llm_resampler
from ncpu.autoresearch.miner import load_queue
from ncpu.autoresearch.runner import run_session
from ncpu.autoresearch.types import Budget, SolvedItem, WorkItem


# ---------------------------------------------------------------------------
# FIX-4: library growth from cascade solves (the compounding-loop closure)
# ---------------------------------------------------------------------------


def _find_array_thought_coprocessor(model):
    """Locate the deepest wrapped ArrayThoughtCoprocessor with a library.

    Duck-typed scan (``array_head`` + attached ``program_library``) so tests
    can substitute a lightweight stand-in. Returns the *last* match — for
    ``target_layers=[-2,-1]`` that is the final wrapped layer, whose hidden
    states are the ones library lookup keys on at generation time.
    """
    found = None
    for module in model.modules():
        if (
            getattr(module, "array_head", None) is not None
            and getattr(module, "program_library", None) is not None
        ):
            found = module
    return found


def _render_capture_prompt(tokenizer, prompt: str) -> str:
    """Mirror ``humaneval_runner.generate_solution``'s prompt rendering.

    The hidden state must be captured under the same chat template the eval
    runs use, otherwise the recorded signature won't match the hidden states
    the library sees at the next eval.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a Python code generator. Output ONLY executable Python code. No explanation, no markdown prose, no comments about approach."},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    return prompt


def _capture_target_hidden_state(model, tokenizer, device, prompt: str, module):
    """Forward the prompt once; capture the last token's hidden state entering
    ``module`` (the wrapped array-thought layer). Returns a rank-1 tensor or
    None if the module never fired."""
    import torch

    text = _render_capture_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    captured: dict = {}

    def _hook(_mod, args):
        if args and isinstance(args[0], torch.Tensor) and args[0].ndim == 3:
            captured["hidden"] = args[0].detach()[0, -1, :].clone()

    handle = module.register_forward_pre_hook(_hook)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    return captured.get("hidden")


def _first_usable_pair(io_pairs, max_len: int):
    """First io_pair shaped like (int array fitting max_len, finite scalar)."""
    import math

    for pair in io_pairs:
        arr = io_pair_int_array(pair)
        if arr is None or len(arr) > max_len:
            continue
        try:
            expected = float(pair.expected)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(expected):
            continue
        return arr, expected
    return None


def grow_library_from_solve(
    *,
    model,
    tokenizer,
    device: str,
    work_item: Optional[WorkItem],
    solved_item: SolvedItem,
    library_path: Path,
) -> str:
    """FIX-4 hook: turn one verified cascade solve into a keyed library entry.

    1. Translate the solved Python into a 5-tuple (refuse non-array shapes).
    2. Forward the model on the item's prompt; capture the target-layer
       hidden state (same capture point library lookup uses at inference).
    3. ``record_successful_generation`` keys the entry on that hidden state;
       if the head's own crystallized program doesn't reproduce the ground
       truth, fall back to recording the *translated* program directly —
       it is probe-verified against the solved behavior, so the entry is
       immediately usable at the next eval either way.
    4. Save the library JSON.

    Returns a one-line growth/refusal report. Exceptions propagate; the
    caller wraps this so distillation failure never kills the solve loop.
    """
    import torch

    from ncpu.self_optimizing.array_program_library import DiscreteArrayProgram
    from ncpu.self_optimizing.continual_library import record_successful_generation

    task_id = solved_item.task_id
    if work_item is None:
        return (
            f"library unchanged: {task_id} → no WorkItem in queue "
            f"(cannot recover prompt/io_pairs)"
        )
    coproc = _find_array_thought_coprocessor(model)
    if coproc is None:
        return (
            f"library unchanged: {task_id} → no array-thought coprocessor "
            f"with attached library"
        )

    five = translate_to_5tuple(
        solved_item,
        work_item.io_pairs,
        prompt=work_item.prompt,
        entry_point=work_item.entry_point,
    )
    if five is None:
        return (
            f"library unchanged: {task_id} → not translatable to 5-tuple "
            f"(not an int-array→scalar reduction)"
        )
    solved_item.program_5tuple = five

    hidden = _capture_target_hidden_state(
        model, tokenizer, device, work_item.prompt, coproc
    )
    if hidden is None:
        return f"library unchanged: {task_id} → hidden-state capture failed"

    library = coproc.program_library
    head = coproc.array_head
    before = len(library)
    param = next(head.parameters())
    max_len = int(getattr(head.config, "array_max_len", 8))

    recorded_via = None
    pair = _first_usable_pair(work_item.io_pairs, max_len)
    if pair is not None:
        arr, expected = pair
        array_inputs = torch.zeros(max_len, dtype=param.dtype, device=param.device)
        array_inputs[: len(arr)] = torch.tensor(
            arr, dtype=param.dtype, device=param.device
        )
        lengths = torch.tensor(
            float(len(arr)), dtype=param.dtype, device=param.device
        )
        report = record_successful_generation(
            library,
            head,
            hidden_state=hidden.to(device=param.device, dtype=param.dtype),
            array_inputs=array_inputs,
            lengths=lengths,
            ground_truth_scalar=expected,
            task_name=task_id,
        )
        if report.grew or report.reason == "refreshed existing entry":
            recorded_via = f"record_successful_generation ({report.reason})"
    if recorded_via is None:
        program = DiscreteArrayProgram.from_dict(five)
        library.record(hidden, program, task_name=task_id, convergence_gap=0.0)
        recorded_via = "direct record of translated 5-tuple"

    after = len(library)
    library.save(Path(library_path).expanduser())
    verb = "grew" if after > before else "updated"
    return f"library {verb}: {task_id} → entries {before}→{after} via {recorded_via}"


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

    # FIX-4: with NPCoT loaded, every cascade solve feeds the library so the
    # next eval run can fire it. The queue is loaded once for WorkItem lookup
    # (the runner's callback only carries the CascadeResult).
    work_items_by_id = (
        {it.task_id: it for it in load_queue(args.queue)} if use_npcot else {}
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
        if use_npcot and result.solved and result.solved_item is not None:
            try:
                msg = grow_library_from_solve(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    work_item=work_items_by_id.get(result.task_id),
                    solved_item=result.solved_item,
                    library_path=args.library,
                )
            except Exception as exc:  # noqa: BLE001 — growth must never kill the loop
                msg = (
                    f"library growth failed: {result.task_id} → "
                    f"{type(exc).__name__}: {exc}"
                )
            print(f"[driver] {msg}", flush=True)

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
