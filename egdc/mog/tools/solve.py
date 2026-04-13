"""CLI for the unified Mog orchestrator.

Usage:
    python -m egdc.mog.tools.solve --num_problems 25
    python -m egdc.mog.tools.solve --variants_per_factory 5 --use_real_compiler --output report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from egdc.mog.benchmark import get_benchmark
from egdc.mog.routing.orchestrator import MogOrchestrator


def main():
    ap = argparse.ArgumentParser(description="Unified Mog solver")
    ap.add_argument("--variants_per_factory", type=int, default=1)
    ap.add_argument("--num_problems", type=int, default=None)
    ap.add_argument("--memory_root", type=str, default="egdc/mog/routing/pathway_memory")
    ap.add_argument("--completion_checkpoint", type=str, default=None)
    ap.add_argument("--use_real_compiler", action="store_true")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    problems = get_benchmark(seed=42, variants_per_factory=args.variants_per_factory)
    if args.num_problems is not None:
        problems = problems[:args.num_problems]

    orch = MogOrchestrator(
        memory_root=args.memory_root,
        completion_checkpoint=args.completion_checkpoint,
        use_real_compiler=args.use_real_compiler,
    )
    summary = orch.evaluate(problems)

    print(f"Solved {summary['num_solved']}/{summary['num_problems']} ({summary['pass_rate']:.3f})")
    print(f"By method: {summary['by_method']}")
    if summary.get("family_scores"):
        print("Family scores:")
        for fam, score in sorted(summary["family_scores"].items()):
            print(f"  {fam}: {score:.3f}")
    print(f"Total pathway successes stored: {summary['total_successes']}")

    if summary.get("induced_patterns"):
        print(f"\nInduced structural patterns:")
        for p in summary["induced_patterns"][:10]:
            print(f"  {p['shared_structure']} (freq={p['frequency']}, members={p['member_functions'][:5]})")

    if summary.get("num_regressions", 0) > 0:
        print(f"\nAuto-regressions recorded: {summary['num_regressions']}")

    failed = [r for r in summary["results"] if not r["success"]]
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for r in failed:
            print(f"  {r['problem']}: {r['method']} / {r['error']}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved report to {out}")


if __name__ == "__main__":
    main()
