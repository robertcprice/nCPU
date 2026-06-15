#!/usr/bin/env python3
"""Tier-A benchmark: measure the new search teachers on a curated
held-out set and produce a publishable results table.

Runs `mog_synth --per-problem-json` against the full benchmark,
then aggregates the per-problem JSONL output into:

  1. Method coverage table — which methods solve how many problems
  2. Method × category matrix — which categories use which methods
  3. New-teacher coverage — specifically the 6 teachers added in
     this batch
  4. Latency distribution (mean / median / p99) per method

The output is both human-readable text and machine-parseable JSON
(`--out /path/to/results.json`).

Run from the repo root:
  python3 nsynth/scripts/bench_tier_a.py

Requires the release binary at nsynth/target/release/mog_synth
(`cargo build --release` in nsynth/).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = REPO_ROOT / "nsynth" / "target" / "release" / "mog_synth"

# The 6 search teachers added in this batch (2026-06-14 work). Used
# to compute "new-teacher coverage" — a high number here means the
# new teacher surface is being used.
NEW_TEACHERS = {
    "search_strictly_increasing",
    "search_has_strictly_increasing_run",
    "search_first_index_of",
    "search_last_index_of",
    "search_stateful_reducer",
    "search_intersects",
    "search_is_anagram",
    "search_longest_run",
}


def _run_benchmark(variants: int) -> tuple[list[dict], dict]:
    """Run `mog_synth --per-problem-json --variants N` and return
    (per_problem_rows, summary)."""
    proc = subprocess.run(
        [
            str(BACKEND),
            "--per-problem-json",
            "--variants",
            str(variants),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    rows: list[dict] = []
    summary: dict = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("summary"):
            summary = obj
        else:
            rows.append(obj)
    return rows, summary


def _format_table(
    headers: list[str],
    rows: list[list[str]],
    title: str | None = None,
) -> str:
    out = []
    if title:
        out.append(f"\n{title}\n" + "=" * len(title))
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = "  "
    out.append(sep.join(h.ljust(w) for h, w in zip(headers, widths)))
    out.append(sep.join("-" * w for w in widths))
    for r in rows:
        out.append(sep.join(c.ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


def _print_method_coverage(rows: list[dict]) -> None:
    by_method = Counter(r["method"] for r in rows if r["success"])
    total = len(rows)
    succeeded = sum(1 for r in rows if r["success"])
    headers = ["method", "solved", "of_total", "share"]
    table_rows = []
    for method, n in by_method.most_common():
        share = f"{100 * n / total:.1f}%"
        table_rows.append([method, str(n), str(total), share])
    # add a "FAILED" row if any
    failed = total - succeeded
    if failed:
        table_rows.append(["(failed)", str(failed), str(total), f"{100 * failed / total:.1f}%"])
    print(_format_table(headers, table_rows, title="Method coverage (--variants 1)"))


def _print_method_by_category(rows: list[dict]) -> None:
    matrix: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if r["success"]:
            matrix[r["category"]][r["method"]] += 1
    # Pick top-8 categories by problem count to keep the table readable
    cat_counts = sorted(matrix.items(), key=lambda kv: -sum(kv[1].values()))
    top_cats = [c for c, _ in cat_counts[:8]]
    methods = sorted({m for c in top_cats for m in matrix[c]}, key=lambda m: -sum(matrix[c][m] for c in top_cats))
    # Truncate methods to the top 6 for readability
    methods = methods[:6]
    headers = ["category"] + methods + ["row_total"]
    table_rows = []
    for cat in top_cats:
        row = [cat]
        total = 0
        for m in methods:
            n = matrix[cat][m]
            row.append(str(n))
            total += n
        row.append(str(total))
        table_rows.append(row)
    print(_format_table(headers, table_rows, title="Method × category (top 8 categories, top 6 methods)"))


def _print_new_teacher_coverage(rows: list[dict]) -> None:
    by_method = Counter(r["method"] for r in rows if r["success"])
    new_used = sum(n for m, n in by_method.items() if m in NEW_TEACHERS)
    new_teachers_seen = sorted(m for m in NEW_TEACHERS if m in by_method)
    new_teachers_missing = sorted(m for m in NEW_TEACHERS if m not in by_method)
    print()
    print("New-teacher coverage (this batch, 2026-06-14)")
    print("=" * 50)
    print(f"  new-teachers used: {new_used} of {len(rows)} problems ({100 * new_used / max(1, len(rows)):.1f}%)")
    print(f"  new-teachers seen: {len(new_teachers_seen)} of {len(NEW_TEACHERS)}")
    for m in new_teachers_seen:
        n = by_method[m]
        print(f"    ✓ {m:42s}  {n}")
    for m in new_teachers_missing:
        print(f"    ✗ {m:42s}  (not exercised)")


def _print_latency(rows: list[dict]) -> None:
    by_method: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["success"]:
            by_method[r["method"]].append(r["seconds"] * 1000.0)  # ms
    # Top 8 methods by call count
    top = sorted(by_method.items(), key=lambda kv: -len(kv[1]))[:8]
    headers = ["method", "calls", "mean_ms", "median_ms", "p99_ms", "max_ms"]
    table_rows = []
    for method, lats in top:
        lats_sorted = sorted(lats)
        n = len(lats)
        mean = sum(lats) / n
        median = statistics.median(lats)
        p99_idx = max(0, min(n - 1, int(math.ceil(0.99 * n)) - 1))
        p99 = lats_sorted[p99_idx]
        maxv = lats_sorted[-1]
        table_rows.append([
            method,
            str(n),
            f"{mean:.3f}",
            f"{median:.3f}",
            f"{p99:.3f}",
            f"{maxv:.3f}",
        ])
    print(_format_table(headers, table_rows, title="Latency per method (top 8 by call count)"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="write machine-parseable JSON to this path")
    args = ap.parse_args()

    if not BACKEND.is_file():
        print(f"missing binary: {BACKEND}\nrun: cargo build --release in nsynth/", file=sys.stderr)
        return 2

    print(f"Running tier-A benchmark: --variants {args.variants}")
    t0 = time.time()
    rows, summary = _run_benchmark(args.variants)
    wall = time.time() - t0
    total = len(rows)
    succeeded = sum(1 for r in rows if r["success"])
    print(f"\nRan {total} problems in {wall:.2f}s — {succeeded}/{total} solved ({100 * succeeded / max(1, total):.1f}%)")
    if summary:
        print(f"  binary wall_seconds: {summary.get('wall_seconds', '?')}")
        print(f"  binary problem_count: {summary.get('problem_count', '?')}")
        print(f"  binary passed: {summary.get('passed', '?')}")

    if total == 0:
        print("ERROR: no problems returned from the binary", file=sys.stderr)
        return 1

    _print_method_coverage(rows)
    _print_method_by_category(rows)
    _print_new_teacher_coverage(rows)
    _print_latency(rows)

    if args.out is not None:
        args.out.write_text(json.dumps({
            "wall_seconds": wall,
            "problem_count": total,
            "succeeded": succeeded,
            "summary": summary,
            "rows": rows,
        }, indent=2))
        print(f"\nWrote JSON to {args.out}")

    return 0 if succeeded == total else 1


if __name__ == "__main__":
    sys.exit(main())
