#!/usr/bin/env python3
"""Reproducible harness for the Rust nSynth program synthesizer.

Invokes nsynth/target/release/mog_synth --per-problem-json, parses the
JSONL stream (one row per problem plus a trailing summary object), and
reports solver portfolio breakdown grouped by family (gradient /
enumerative / search). Suitable for paper tables and CI regression gates.

Exit code is non-zero when observed coverage falls below --min-coverage.

Usage:
    python benchmarks/benchmark_nsynth.py
    python benchmarks/benchmark_nsynth.py --variants 3 --json out.json
    python benchmarks/benchmark_nsynth.py --binary /path/to/mog_synth
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_BINARY = PROJECT_ROOT / "nsynth" / "target" / "release" / "mog_synth"

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
    if method.startswith("template"):
        return "template"
    return "other"


def run_harness(*, binary: Path, variants: int, extra_args: list[str]) -> dict:
    if not binary.exists():
        raise SystemExit(
            f"nsynth binary not found at {binary}. "
            f"Build it with: (cd nsynth && cargo build --release)"
        )
    cmd = [str(binary), "--per-problem-json", "--variants", str(variants), *extra_args]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 and not result.stdout:
        raise SystemExit(
            f"nsynth exited with code {result.returncode}\nstderr:\n{result.stderr}"
        )

    rows: list[dict] = []
    summary: dict | None = None
    for line in result.stdout.splitlines():
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

    if summary is None:
        raise SystemExit("nsynth output did not contain a summary line")

    # Family breakdown.
    family_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("success"):
            family_counts[_classify(row.get("method", ""))] += 1

    # Timing.
    times = [row.get("seconds", 0.0) for row in rows]
    timing = {
        "mean_seconds": round(statistics.mean(times), 4) if times else 0.0,
        "median_seconds": round(statistics.median(times), 4) if times else 0.0,
        "max_seconds": round(max(times), 4) if times else 0.0,
    }
    if len(times) >= 20:
        timing["p95_seconds"] = round(statistics.quantiles(times, n=20)[-1], 4)

    slowest = sorted(rows, key=lambda r: -r.get("seconds", 0.0))[:10]
    return {
        "summary": {
            **summary,
            "family_counts": dict(family_counts),
            "timing": timing,
            "slowest": [
                {
                    "name": r["name"],
                    "method": r["method"],
                    "seconds": r["seconds"],
                }
                for r in slowest
            ],
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY, help="path to mog_synth release binary")
    parser.add_argument("--variants", type=int, default=1, help="variants per factory")
    parser.add_argument("--json", type=Path, default=None, help="write full result as JSON")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=1.0,
        help="fail with non-zero exit when coverage falls below this fraction (default 1.0)",
    )
    parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="extra args forwarded to mog_synth (separate with --)",
    )
    args = parser.parse_args()

    extra = [a for a in args.passthrough if a != "--"]

    report = run_harness(binary=args.binary, variants=args.variants, extra_args=extra)
    s = report["summary"]
    print(
        f"nsynth: {s['passed']}/{s['problem_count']} "
        f"({s['coverage']*100:.2f}%) in {s['wall_seconds']}s "
        f"| variants={s['variants_per_factory']} binary={args.binary.name}"
    )
    print("family breakdown (gradient/enumerative/search/template/other):")
    for family in ("gradient", "enumerative", "search", "template", "other"):
        count = s["family_counts"].get(family, 0)
        if count:
            print(f"  {family:12s} {count}")
    print("timing:")
    for k, v in s["timing"].items():
        print(f"  {k}: {v}")
    print("slowest:")
    for row in s["slowest"][:5]:
        print(f"  {row['seconds']:6.2f}s  {row['method']:30s}  {row['name']}")
    if s["failures"]:
        print(f"\nfailures ({len(s['failures'])}):")
        for name in s["failures"]:
            print(f"  {name}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nwrote {args.json}")

    if s["coverage"] < args.min_coverage:
        print(
            f"\nCOVERAGE REGRESSION: {s['coverage']*100:.2f}% < "
            f"{args.min_coverage*100:.2f}%",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
