"""Compliance report CLI (NV5).

Emits a machine-readable or human-readable compliance report for any
`ArrayProgramLibrary` JSON file. Use this as the final step of an audit
pipeline, or to check a shipped library before deploying it into a
regulated workflow.

Usage::

    python3 -m scripts.cli.npcot_compliance path/to/library.json
    python3 -m scripts.cli.npcot_compliance --json path/to/library.json
    python3 -m scripts.cli.npcot_compliance \\
        --markdown --library-name sales_agent_v1 \\
        --input-lower -10 --input-upper 10 --max-length 32 \\
        path/to/library.json > report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ncpu.self_optimizing.array_program_library import ArrayProgramLibrary
from ncpu.self_optimizing.compliance_report import (
    ComplianceReportConfig,
    build_compliance_report,
    render_markdown,
)
from ncpu.self_optimizing.program_verifier import (
    RangeBound,
    VerifierConfig,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("library_path", type=Path)
    parser.add_argument("--library-name", type=str, default=None)
    parser.add_argument("--input-lower", type=float, default=-10.0)
    parser.add_argument("--input-upper", type=float, default=10.0)
    parser.add_argument("--max-length", type=int, default=16)
    parser.add_argument("--overflow-threshold", type=float, default=1e6)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--markdown", action="store_true", help="Render as markdown."
    )
    group.add_argument(
        "--json", action="store_true", help="Emit raw JSON report."
    )
    args = parser.parse_args(argv)

    if not args.library_path.exists():
        print(
            f"library file not found: {args.library_path}", file=sys.stderr
        )
        return 2

    library = ArrayProgramLibrary.load(args.library_path)
    config = ComplianceReportConfig(
        verifier=VerifierConfig(
            input_bound=RangeBound(args.input_lower, args.input_upper),
            max_length=args.max_length,
            overflow_threshold=args.overflow_threshold,
        ),
        library_name=args.library_name
        or args.library_path.stem,
    )
    report = build_compliance_report(library, config=config)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        # Default: markdown (flag kept for parity with explorer CLI).
        print(render_markdown(report))
    return 0 if report["aggregate"]["aggregate_risk"] != "high" else 3


if __name__ == "__main__":
    raise SystemExit(main())
