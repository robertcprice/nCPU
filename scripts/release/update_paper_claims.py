#!/usr/bin/env python3
"""Update measured benchmark claims in the paper from an artifact directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.utils.paper_claims import write_updated_paper  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Update paper benchmark claims from artifact outputs")
    parser.add_argument("--artifact-dir", type=Path, required=True, help="Artifact directory to read")
    parser.add_argument(
        "--paper-path",
        type=Path,
        default=PROJECT_ROOT / "paper" / "ncpu_paper.md",
        help="Paper markdown file to update",
    )
    parser.add_argument("--output-path", type=Path, help="Write the updated paper to a separate path")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the target paper file in place",
    )
    args = parser.parse_args()

    target = write_updated_paper(
        args.artifact_dir,
        args.paper_path,
        output_path=args.output_path,
        in_place=args.in_place,
    )
    print(f"[paper-update] Wrote updated paper to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
