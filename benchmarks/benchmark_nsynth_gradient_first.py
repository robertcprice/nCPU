#!/usr/bin/env python3
"""Resumable nSynth gradient-first coverage survey.

The default nSynth pipeline puts enumerative synthesis before native gradient,
so the default-mode breakdown (60 gradient / 25 enumerative / 10 search in
artifacts/nsynth_per_problem_coverage.jsonl) understates what gradient descent
*alone* can solve. A `--prefer-differentiable` pipeline run inverts that
ordering but is impractical to run in one shot: each hard problem consumes
minutes of gradient search before failing over to enumerative. Full 95-problem
wall time lands in the 2-5 hour range.

This driver runs the gradient-first survey one problem at a time and
checkpoints after every problem, so it can be killed and resumed across
sessions without losing work. Usage:

    # Start (or resume) the survey; writes to artifacts/nsynth_gradient_first/
    python benchmarks/benchmark_nsynth_gradient_first.py

    # Limit to 5 problems per invocation (for local poking)
    python benchmarks/benchmark_nsynth_gradient_first.py --max-problems 5

    # Summarize what's been collected so far
    python benchmarks/benchmark_nsynth_gradient_first.py --summarize

    # Per-problem time budget (default 600 s); problems that run past budget
    # are recorded as `timeout` and can be retried with a larger budget.
    python benchmarks/benchmark_nsynth_gradient_first.py --budget 1200
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_BINARY = PROJECT_ROOT / "nsynth" / "target" / "release" / "mog_synth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "nsynth_gradient_first"

GRADIENT_METHODS = {
    "synth_gradient",
    "univ_arr_gradient",
    "arr_gradient",
    "arr_gradient_binary_search",
    "arr_gradient_count_distinct",
    "arr_gradient_kth_smallest",
    "arr_gradient_two_sum_exists",
}


def _classify(method: str) -> str:
    if method in GRADIENT_METHODS:
        return "gradient"
    if method.startswith("enumerative"):
        return "enumerative"
    if method.startswith("search_"):
        return "search"
    if method.startswith("timeout"):
        return "timeout"
    return "other"


def list_problem_names(binary: Path) -> list[str]:
    """Use the reference artifact, if present, to enumerate problem names.

    Falls back to invoking the binary with --help-style introspection if the
    artifact is missing.
    """
    artifact = PROJECT_ROOT / "artifacts" / "nsynth_per_problem_coverage.jsonl"
    if artifact.exists():
        names: list[str] = []
        for line in artifact.read_text().splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("summary"):
                continue
            name = obj.get("name")
            if isinstance(name, str):
                names.append(name)
        if names:
            return names

    # Fallback: run the binary in default mode, parse problem names out of
    # --per-problem-json. This is slower but self-contained.
    result = subprocess.run(
        [str(binary), "--per-problem-json"],
        capture_output=True,
        text=True,
        check=False,
    )
    names: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("summary"):
            continue
        n = obj.get("name")
        if isinstance(n, str):
            names.append(n)
    return names


def _load_completed(output_dir: Path) -> dict[str, dict]:
    completed: dict[str, dict] = {}
    for file in output_dir.glob("*.json"):
        try:
            row = json.loads(file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        name = row.get("name")
        if isinstance(name, str):
            completed[name] = row
    return completed


def solve_one(binary: Path, name: str, budget: float) -> dict:
    """Invoke mog_synth --prefer-differentiable --problem <name>, bounded.

    The binary's --problem mode emits a k=v formatted body (problem:, method:,
    family:, memory_records:, then the synthesized source code). We parse
    method/family/success from stdout and report a uniform row shape."""
    cmd = [
        str(binary),
        "--prefer-differentiable",
        "--problem",
        name,
    ]
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=budget,
        )
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "success": False,
            "method": "timeout",
            "seconds": round(time.perf_counter() - t0, 4),
            "family": "timeout",
            "budget_exceeded": True,
        }

    method = "unknown"
    family = ""
    for line in result.stdout.splitlines():
        if line.startswith("method:"):
            method = line.split(":", 1)[1].strip()
        elif line.startswith("family:"):
            family = line.split(":", 1)[1].strip()
    success = result.returncode == 0
    row = {
        "name": name,
        "success": success,
        "method": method,
        "family": family or _classify(method),
        "seconds": round(time.perf_counter() - t0, 4),
    }
    if not success:
        row["error"] = (result.stderr or "").strip().splitlines()[-1][:200] if result.stderr else ""
    return row


def run_survey(
    *,
    binary: Path,
    output_dir: Path,
    max_problems: int | None,
    budget: float,
    retry_timeouts: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list_problem_names(binary)
    if not names:
        raise SystemExit("could not enumerate problem names")

    completed = _load_completed(output_dir)
    pending: list[str] = []
    for name in names:
        if name not in completed:
            pending.append(name)
        elif retry_timeouts and completed[name].get("budget_exceeded"):
            pending.append(name)

    print(
        f"gradient-first survey: {len(completed)}/{len(names)} recorded, "
        f"{len(pending)} pending (budget={budget}s)"
    )
    if max_problems is not None:
        pending = pending[:max_problems]

    stop_requested = [False]

    def _stop(_signum, _frame):
        stop_requested[0] = True
        print("\n[signal] stopping after current problem — state is checkpointed", flush=True)

    signal.signal(signal.SIGINT, _stop)

    for i, name in enumerate(pending, 1):
        if stop_requested[0]:
            break
        print(f"  [{i}/{len(pending)}] {name} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        row = solve_one(binary, name, budget)
        elapsed = time.perf_counter() - t0
        (output_dir / f"{name}.json").write_text(json.dumps(row, indent=2))
        if row.get("success") and row.get("family") == "gradient":
            status = "OK (gradient)"
        elif row.get("success"):
            status = f"OK ({row.get('family')})"
        elif row.get("budget_exceeded"):
            status = "TIMEOUT"
        else:
            status = "FAIL"
        print(f"{status} in {elapsed:.1f}s — {row.get('method')}")


def summarize(output_dir: Path) -> None:
    if not output_dir.exists():
        print(f"no output dir yet: {output_dir}")
        return
    completed = _load_completed(output_dir)
    total = len(completed)
    if not total:
        print(f"no rows yet at {output_dir}")
        return
    by_family: dict[str, int] = {}
    timeouts: list[str] = []
    slowest = sorted(completed.values(), key=lambda r: -r.get("seconds", 0))[:10]
    for row in completed.values():
        fam = row.get("family", _classify(row.get("method", "")))
        by_family[fam] = by_family.get(fam, 0) + 1
        if row.get("budget_exceeded"):
            timeouts.append(row["name"])
    success = sum(1 for r in completed.values() if r.get("success"))
    print(f"gradient-first survey: {success}/{total} solved")
    for family in sorted(by_family, key=lambda k: -by_family[k]):
        print(f"  {family:12s} {by_family[family]}")
    if timeouts:
        print(f"timeouts ({len(timeouts)}): {', '.join(timeouts)}")
    print("\nslowest completed problems:")
    for row in slowest:
        s = row.get("seconds", 0)
        print(f"  {s:7.1f}s  {row.get('method','?'):30s}  {row['name']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--budget",
        type=float,
        default=600.0,
        help="per-problem wall-time budget in seconds (default 600)",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="stop after this many new problems (useful for incremental runs)",
    )
    parser.add_argument(
        "--retry-timeouts",
        action="store_true",
        help="re-attempt previously timed-out problems (useful after --budget increase)",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="print current survey summary and exit without running anything",
    )
    args = parser.parse_args()

    if args.summarize:
        summarize(args.output_dir)
        return 0

    if not args.binary.exists():
        print(
            f"nsynth binary not found at {args.binary}. "
            f"Build it with: (cd nsynth && cargo build --release)",
            file=sys.stderr,
        )
        return 2

    run_survey(
        binary=args.binary,
        output_dir=args.output_dir,
        max_problems=args.max_problems,
        budget=args.budget,
        retry_timeouts=args.retry_timeouts,
    )
    summarize(args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
