"""CLI explorer for NPCoT array program libraries (NV1).

Loads an `ArrayProgramLibrary` JSON file and prints, per cached skill:

* The Rust-like pseudocode that will execute on a library hit.
* Hit count, convergence gap, task name.
* A top-line summary (entry count, average gap, capacity).

It can also emit the markdown audit report for compliance review:

    python3 -m scripts.cli.npcot_skill_explorer path/to/library.json
    python3 -m scripts.cli.npcot_skill_explorer --markdown path/to/library.json
    python3 -m scripts.cli.npcot_skill_explorer --json path/to/library.json

The library itself is plain JSON, so any editor or `jq` works too — this CLI
just applies the canonical human-facing rendering.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ncpu.self_optimizing.array_program_library import ArrayProgramLibrary


def _format_text(library: ArrayProgramLibrary) -> str:
    report = library.audit_report()
    summary = report["summary"]
    lines: list[str] = []
    lines.append("NPCoT Array Program Library")
    lines.append("=" * 40)
    lines.append(f"Entries:              {summary['entry_count']}")
    lines.append(f"Unique program shapes: {summary['unique_program_shapes']}")
    lines.append(f"Total library hits:   {summary['total_hits']}")
    avg_gap = summary["avg_convergence_gap"]
    if avg_gap is not None:
        lines.append(f"Avg convergence gap:  {avg_gap:.4f}")
    max_gap = summary["max_convergence_gap"]
    if max_gap is not None:
        lines.append(f"Max convergence gap:  {max_gap:.4f}")
    lines.append(f"Similarity threshold: {summary['config']['similarity_threshold']:.3f}")
    lines.append(f"Capacity:             {summary['config']['max_entries']}")

    if not report["entries"]:
        lines.append("")
        lines.append("(library is empty)")
        return "\n".join(lines)

    for index, entry in enumerate(report["entries"], start=1):
        lines.append("")
        lines.append("-" * 40)
        lines.append(
            f"[{index}] task={entry['task_name'] or '(unnamed)'}  "
            f"hits={entry['hit_count']}"
        )
        gap = entry["convergence_gap"]
        if gap is not None:
            lines.append(f"    convergence_gap={gap:.4f}")
        cached_at = entry["cached_at_step"]
        if cached_at is not None:
            lines.append(f"    cached_at_step={cached_at}")
        lines.append(f"    signature_dim={entry['signature_dim']}")
        lines.append("")
        for program_line in entry["program"]["program_text"].splitlines():
            lines.append(f"    {program_line}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pretty-print an NPCoT array program library.",
    )
    parser.add_argument("library_path", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--markdown",
        action="store_true",
        help="Emit markdown audit report (suitable for compliance review).",
    )
    group.add_argument(
        "--json",
        action="store_true",
        help="Emit the raw audit JSON report.",
    )
    parser.add_argument(
        "--sort-by-hits",
        action="store_true",
        help="Sort entries by hit count (descending) before rendering.",
    )
    args = parser.parse_args(argv)

    if not args.library_path.exists():
        print(f"library file not found: {args.library_path}", file=sys.stderr)
        return 2

    library = ArrayProgramLibrary.load(args.library_path)

    if args.sort_by_hits:
        library._entries.sort(key=lambda entry: entry.hit_count, reverse=True)

    if args.markdown:
        print(library.audit_markdown())
    elif args.json:
        print(json.dumps(library.audit_report(), indent=2))
    else:
        print(_format_text(library))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
