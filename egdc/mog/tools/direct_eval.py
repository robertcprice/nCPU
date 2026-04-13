"""CLI for direct/adaptive Mog synthesis benchmark evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from egdc.mog.benchmark import get_benchmark
from egdc.mog.routing.adaptive_router import AdaptiveMogRouter


def main():
    ap = argparse.ArgumentParser(description="Evaluate direct Mog synthesis on the benchmark")
    ap.add_argument("--variants_per_factory", type=int, default=5)
    ap.add_argument("--num_problems", type=int, default=None)
    ap.add_argument("--memory_root", type=str, default="egdc/mog/routing/pathway_memory")
    ap.add_argument("--use_real_compiler", action="store_true")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    problems = get_benchmark(seed=42, variants_per_factory=args.variants_per_factory)
    if args.num_problems is not None:
        problems = problems[:args.num_problems]

    router = AdaptiveMogRouter(memory_root=args.memory_root)
    summary = router.evaluate(problems, use_real_compiler=args.use_real_compiler)

    print(f"Solved {summary['num_solved']}/{summary['num_problems']} ({summary['pass_rate']:.3f})")
    if summary.get("family_scores"):
        print("Family scores:")
        for fam, score in sorted(summary["family_scores"].items()):
            print(f"  {fam}: {score:.3f}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"Saved report to {out}")


if __name__ == "__main__":
    main()
