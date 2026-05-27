"""Compare multiple benchmark-run JSON artifacts into one table.

Takes a directory of `humaneval_*.json` / `mbpp_*.json` files (the output
format written by `humaneval_runner` and `mbpp_runner`) and emits either a
human-readable table or a machine-readable JSON summary.

Used to turn a multi-model sweep into a headline number table:

    python3 -m ncpu.self_optimizing.compare_runs \\
        ~/reports/ --benchmark humaneval
    python3 -m ncpu.self_optimizing.compare_runs \\
        ~/reports/ --benchmark mbpp --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def collect_reports(directory: Path, benchmark: str) -> list[dict]:
    pattern = f"{benchmark}_*.json"
    reports: list[dict] = []
    for path in sorted(directory.glob(pattern)):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "results" not in payload:
            continue
        reports.append({"file": path.name, "payload": payload})
    return reports


def summarize(reports: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for r in reports:
        cfg = r["payload"].get("config", {})
        res = r["payload"]["results"]
        rows.append({
            "file": r["file"],
            "model": cfg.get("model", "?"),
            "use_npcot": cfg.get("use_npcot", False),
            "library": cfg.get("library_path"),
            "problems": res.get("total_problems", 0),
            "pass_count": res.get("pass_count", 0),
            "pass_at_1": res.get("pass_at_1", 0.0),
            "total_seconds": res.get("total_seconds", 0.0),
        })
    # Sort: by model name, then baseline before +NPCoT within the same model.
    rows.sort(key=lambda r: (r["model"], r["use_npcot"]))
    return rows


def render_table(rows: list[dict]) -> str:
    lines = [
        f"{'model':36s} {'config':16s} {'problems':>9s} {'pass@1':>9s} {'wall':>7s}",
        "-" * 84,
    ]
    for row in rows:
        cfg = "+NPCoT" if row["use_npcot"] else "baseline"
        pct = f"{row['pass_at_1'] * 100:.1f}%"
        wall = f"{row['total_seconds']:.0f}s"
        lines.append(
            f"{row['model'][:36]:36s} {cfg:16s} {row['problems']:>9d} "
            f"{pct:>9s} {wall:>7s}"
        )
    return "\n".join(lines)


def render_markdown(rows: list[dict], benchmark: str) -> str:
    lines = [
        f"# {benchmark} multi-model sweep",
        "",
        f"| model | config | problems | pass@1 | wall |",
        f"|-------|--------|----------|--------|------|",
    ]
    for row in rows:
        cfg = "+NPCoT" if row["use_npcot"] else "baseline"
        pct = f"{row['pass_at_1'] * 100:.1f}%"
        wall = f"{row['total_seconds']:.0f}s"
        lines.append(
            f"| `{row['model']}` | {cfg} | {row['problems']} | "
            f"**{pct}** | {wall} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--benchmark", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args(argv)

    if not args.directory.exists():
        print(f"directory not found: {args.directory}", file=sys.stderr)
        return 2

    reports = collect_reports(args.directory, args.benchmark)
    if not reports:
        print(f"no {args.benchmark}_*.json files in {args.directory}", file=sys.stderr)
        return 3

    rows = summarize(reports)
    if args.markdown:
        print(render_markdown(rows, args.benchmark))
    else:
        print(render_table(rows))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"benchmark": args.benchmark, "rows": rows}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
