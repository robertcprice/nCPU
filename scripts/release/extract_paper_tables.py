#!/usr/bin/env python3
"""Render paper-ready markdown tables and key metrics from an artifact run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.utils.paper_tables import write_paper_tables  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract paper-ready tables from artifact JSON files")
    parser.add_argument("--artifact-dir", type=Path, required=True, help="Artifact directory to read")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory for generated tables (default: <artifact-dir>/paper_tables)",
    )
    args = parser.parse_args()

    paths = write_paper_tables(args.artifact_dir, args.output_dir)
    print(f"[paper-tables] Wrote metrics to {paths['metrics']}")
    print(f"[paper-tables] Wrote markdown tables to {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
