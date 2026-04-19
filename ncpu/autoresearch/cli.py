"""Command-line entry point for the autoresearch loop.

Verbs
-----

* ``mine`` — read an eval JSON, append hard-fails to the work queue.
* ``run-once`` — consume the queue once under a budget.
* ``status`` — print session + artifact summary.

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

from ncpu.autoresearch.cascade import CascadeConfig
from ncpu.autoresearch.distiller import dedupe_solved, load_solved, summarize_solved
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

    r = sub.add_parser("run-once", help="consume the queue once under a budget")
    r.add_argument("--benchmark", choices=["humaneval", "mbpp"], default="humaneval")
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

    args = p.parse_args(argv)
    # Fill solver default (argparse can't default a 'append' action).
    if getattr(args, "solver", None) is None and args.verb == "run-once":
        args.solver = ["template_match"]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
