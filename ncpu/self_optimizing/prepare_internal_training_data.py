"""Build supervised training data from hidden SOME trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ncpu.self_optimizing.trajectory_dataset import (
    build_distillation_dataset,
    write_distillation_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare SFT-style training data from hidden SOME trajectory JSONL files."
    )
    parser.add_argument(
        "--trajectory-root",
        required=True,
        help="Directory containing hidden trajectory JSONL files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path for prepared training examples.",
    )
    parser.add_argument(
        "--summary-output",
        help="Optional JSON path for dataset summary metadata. Defaults to <output>.summary.json.",
    )
    parser.add_argument(
        "--include-failed-steps",
        action="store_true",
        help="Include steps whose immediate verification failed.",
    )
    parser.add_argument(
        "--exclude-think-steps",
        action="store_true",
        help="Exclude hidden planning steps from the exported dataset.",
    )
    parser.add_argument(
        "--allow-unverified-commits",
        action="store_true",
        help="Include trajectories that never reached a verified commit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = build_distillation_dataset(
        args.trajectory_root,
        include_failed_steps=args.include_failed_steps,
        include_think_steps=not args.exclude_think_steps,
        require_verified_commit=not args.allow_unverified_commits,
    )
    summary = write_distillation_dataset(examples, args.output)

    summary_path = Path(args.summary_output) if args.summary_output else Path(f"{args.output}.summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote {summary['num_examples']} training examples to {args.output}")
    print(f"Summary written to {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
