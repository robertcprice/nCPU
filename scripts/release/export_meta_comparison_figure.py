#!/usr/bin/env python3
"""Export a stable paper-facing alias for the meta comparison demo artifact."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "paper" / "generated" / "meta_comparison_demo_latest"


def _copy_tree(src: Path, dst: Path) -> None:
    for path in sorted(src.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the meta comparison demo to a stable paper-facing path")
    parser.add_argument("--source-dir", type=Path, required=True, help="Source artifact directory to export")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Stable destination directory (default: paper/generated/meta_comparison_demo_latest)",
    )
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"Source directory not found: {source_dir}")
    if not (source_dir / "final.png").is_file():
        raise SystemExit(f"Expected final.png under source directory: {source_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_tree(source_dir, output_dir)

    print(f"[meta-compare-export] copied {source_dir} -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
