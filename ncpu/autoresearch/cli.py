"""Command-line entry point for the autoresearch loop.

Verbs
-----

* ``mine`` — read an eval JSON, append hard-fails to the work queue.
* ``run-once`` — consume the queue once under a budget.
* ``status`` — print session + artifact summary.
* ``distill`` — offline 5-tuple translation pass over solved_programs.jsonl
  (pure Python, no torch; pending entries land next to the library).

All artifacts default to ``.nCPU_autoresearch/`` in the current working
directory.

Usage::

    python -m ncpu.autoresearch.cli mine \\
        --eval training_results/realworld_vastai/humaneval_agent_4B.json \\
        --benchmark humaneval

    python -m ncpu.autoresearch.cli run-once \\
        --wall-seconds 600 \\
        --max-problems 30

    python -m ncpu.autoresearch.cli status
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from ncpu.autoresearch.cascade import CascadeConfig, run_cascade
from ncpu.autoresearch.distiller import (
    dedupe_solved,
    distill_solved,
    load_solved,
    summarize_solved,
)
from ncpu.autoresearch.miner import load_queue, mine
from ncpu.autoresearch.runner import run_session
from ncpu.autoresearch.types import Budget, DEFAULT_ARTIFACT_DIR


def _queue_path(art: Path, benchmark: str) -> Path:
    return art / f"{benchmark}_queue.jsonl"


def _solved_path(art: Path) -> Path:
    return art / "solved_programs.jsonl"


def _status_path(art: Path) -> Path:
    return art / "status.json"


def cmd_mine(args: argparse.Namespace) -> int:
    art = Path(args.artifact_dir)
    queue = _queue_path(art, args.benchmark)
    counters = mine(
        eval_json_path=args.eval,
        benchmark=args.benchmark,
        out_path=queue,
        min_io_pairs=args.min_io_pairs,
        task_filter=set(args.task) if args.task else None,
    )
    print(json.dumps(counters, indent=2))
    return 0


def cmd_mine_registry(args: argparse.Namespace) -> int:
    """Mine rejected verified-skill registry submissions into the driver
    queue. Implements the registry source described in
    ``docs/autoresearch_continuous.md`` §4. Round-trips through
    :class:`WorkItem` so the cascade can run on it like any other
    benchmark.
    """
    from ncpu.autoresearch.sources.registry import mine_registry_misses

    art = Path(args.artifact_dir)
    queue = _queue_path(art, "registry")
    counters = mine_registry_misses(args.misses, queue)
    print(json.dumps(counters, indent=2))
    return 0


def cmd_run_once(args: argparse.Namespace) -> int:
    art = Path(args.artifact_dir)
    queue = _queue_path(art, args.benchmark)
    solved = _solved_path(art)
    status = _status_path(art)
    if not queue.exists():
        print(f"queue {queue} not found; run `mine` first", file=sys.stderr)
        return 2
    budget = Budget(
        wall_seconds=args.wall_seconds,
        max_cost_usd=args.max_cost_usd,
        max_problems=args.max_problems,
        per_problem_seconds=args.per_problem_seconds,
    )
    cfg = CascadeConfig(
        solver_names=args.solver,
        per_solver_seconds=args.per_problem_seconds,
    )

    def progress(result, report):
        tag = "SOLVED" if result.solved else "unsolved"
        print(
            f"[autoresearch] {report.problems_attempted}: "
            f"{result.task_id}: {tag} "
            f"(solved={report.problems_solved})",
            flush=True,
        )

    report = run_session(
        queue_path=queue,
        solved_path=solved,
        budget=budget,
        cascade_config=cfg,
        status_path=status,
        on_result=progress,
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    art = Path(args.artifact_dir)
    status = _status_path(art)
    solved_path = _solved_path(art)
    out: dict = {}
    if status.exists():
        out["last_session"] = json.loads(status.read_text())
    solved = load_solved(solved_path)
    out["artifact_dir"] = str(art)
    out["solved_programs"] = summarize_solved(solved)
    for f in art.glob("*_queue.jsonl") if art.exists() else []:
        items = load_queue(f)
        out.setdefault("queues", {})[f.stem] = {"items": len(items)}
    print(json.dumps(out, indent=2))
    return 0


def cmd_dedupe(args: argparse.Namespace) -> int:
    path = Path(args.artifact_dir) / "solved_programs.jsonl"
    n = dedupe_solved(path)
    print(f"kept {n} unique solved items in {path}")
    return 0


def cmd_distill(args: argparse.Namespace) -> int:
    """Offline 5-tuple distillation over an existing solved_programs.jsonl.

    Pure Python (no torch): translatable solves are parked in
    ``pending_distill.json`` next to the library — see
    :func:`distiller.distill_solved` for why signature-less entries can't
    go straight into the library JSON.
    """
    if not args.solved.exists():
        print(f"solved log {args.solved} not found", file=sys.stderr)
        return 2
    items = load_solved(args.solved)
    work_items = None
    if args.queue is not None:
        work_items = {it.task_id: it for it in load_queue(args.queue)}
    summary = distill_solved(items, args.library, work_items=work_items)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_user(args: argparse.Namespace) -> int:
    """Solve a single user prompt end-to-end via the cascade.

    Pipeline: read prompt → extract tests via
    :func:`prompt_parser.build_work_item` → run cascade → persist
    result into the compounding store.
    """
    from ncpu.autoresearch.compounding_store import (
        CompoundingStore, CompoundingStoreConfig,
    )
    from ncpu.autoresearch.prompt_parser import build_work_item

    prompt = args.prompt
    if prompt is None and args.prompt_file is not None:
        prompt = args.prompt_file.read_text()
    if prompt is None:
        prompt = sys.stdin.read()
    if not prompt.strip():
        print("error: empty prompt", file=sys.stderr)
        return 2

    wi = build_work_item(
        prompt, task_id="user/cli", entry_point=args.entry_point,
    )
    if wi is None:
        print("error: could not infer entry_point from prompt; "
              "pass --entry-point or include a `def` line.", file=sys.stderr)
        return 2
    print(f"[user] entry_point={wi.entry_point}  "
          f"io_pairs={len(wi.io_pairs)}  "
          f"sources={wi.provenance.get('extraction_sources')}",
          flush=True)

    solvers = args.solver or ["template_match"]
    cfg = CascadeConfig(
        solver_names=solvers,
        per_solver_seconds=args.per_problem_seconds,
    )
    result = run_cascade(wi, config=cfg)

    store = CompoundingStore(CompoundingStoreConfig(
        artifact_dir=Path(args.artifact_dir),
    ))
    if result.solved and result.solved_item is not None:
        store.record(result.solved_item, work_item=wi)
        print(f"\n[user] SOLVED by {result.solver} in {result.wall_seconds:.2f}s\n")
        print(wi.prompt + result.solved_item.program_python)
    else:
        print(f"\n[user] unsolved (tried: {','.join(solvers)})")
        print(f"last error: {result.error}")
    return 0 if result.solved else 1


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="ncpu.autoresearch",
                                description=__doc__.splitlines()[0])
    p.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    sub = p.add_subparsers(dest="verb", required=True)

    m = sub.add_parser("mine", help="mine an eval JSON into the work queue")
    m.add_argument("--eval", type=Path, required=True)
    m.add_argument("--benchmark", choices=["humaneval", "mbpp"], required=True)
    m.add_argument("--min-io-pairs", type=int, default=2)
    m.add_argument("--task", action="append", default=None)
    m.set_defaults(func=cmd_mine)

    mr = sub.add_parser(
        "mine-registry",
        help="mine rejected registry submissions into the driver queue",
    )
    mr.add_argument("--misses", type=Path, required=True,
                    help="JSONL file of registry misses (one per line)")
    mr.set_defaults(func=cmd_mine_registry)

    r = sub.add_parser("run-once", help="consume the queue once under a budget")
    r.add_argument("--benchmark", choices=["humaneval", "mbpp", "registry"],
                   default="humaneval")
    r.add_argument("--wall-seconds", type=float, default=1800.0)
    r.add_argument("--max-cost-usd", type=float, default=1.0)
    r.add_argument("--max-problems", type=int, default=50)
    r.add_argument("--per-problem-seconds", type=float, default=30.0)
    r.add_argument("--solver", action="append",
                   default=None,
                   help="solver name(s) to run, in order; repeatable. "
                        "Default: template_match.")
    r.set_defaults(func=cmd_run_once)

    st = sub.add_parser("status", help="print artifact + session summary")
    st.set_defaults(func=cmd_status)

    d = sub.add_parser("dedupe", help="dedupe the solved_programs.jsonl")
    d.set_defaults(func=cmd_dedupe)

    di = sub.add_parser(
        "distill",
        help="offline 5-tuple distillation pass over a solved_programs.jsonl",
    )
    di.add_argument("--solved", type=Path, required=True)
    di.add_argument("--library", type=Path, required=True)
    di.add_argument("--queue", type=Path, default=None,
                    help="optional WorkItem queue JSONL — recovers "
                         "prompts/entry points/io_pairs for body-style programs")
    di.set_defaults(func=cmd_distill)

    u = sub.add_parser("user",
                       help="extract tests from a user prompt and solve it end-to-end")
    u.add_argument("--prompt", type=str, default=None,
                   help="user prompt; defaults to reading stdin")
    u.add_argument("--prompt-file", type=Path, default=None)
    u.add_argument("--entry-point", type=str, default=None,
                   help="override entry point (otherwise inferred from def)")
    u.add_argument("--per-problem-seconds", type=float, default=10.0)
    u.add_argument("--solver", action="append", default=None,
                   help="solver name(s) to run, in order; repeatable. "
                        "Default: template_match.")
    u.set_defaults(func=cmd_user)

    args = p.parse_args(argv)
    # Fill solver default (argparse can't default a 'append' action).
    if getattr(args, "solver", None) is None and args.verb == "run-once":
        args.solver = ["template_match"]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
