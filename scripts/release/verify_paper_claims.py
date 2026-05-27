#!/usr/bin/env python3
"""Verify that the paper's measured benchmark sections match an artifact bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.utils.paper_claims import verify_paper_claims  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify paper benchmark claims against artifact outputs")
    parser.add_argument("--artifact-dir", type=Path, required=True, help="Artifact directory to read")
    parser.add_argument(
        "--paper-path",
        type=Path,
        default=PROJECT_ROOT / "paper" / "ncpu_paper.md",
        help="Paper markdown file to verify",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Optional path for the rendered expected paper (written on both pass/fail if set)",
    )
    args = parser.parse_args()

    matches, preview_path = verify_paper_claims(
        args.artifact_dir,
        args.paper_path,
        output_path=args.output_path,
    )

    if matches:
        print("[paper-verify] Paper claims match artifact outputs")
        if preview_path is not None:
            print(f"[paper-verify] Wrote rendered preview to {preview_path}")
        return 0

    print("[paper-verify] Paper claims differ from artifact outputs")
    if preview_path is not None:
        print(f"[paper-verify] Wrote expected paper preview to {preview_path}")
    else:
        print("[paper-verify] Re-run with --output-path to capture the rendered expected paper")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
