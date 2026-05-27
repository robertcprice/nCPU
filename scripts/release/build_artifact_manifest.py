#!/usr/bin/env python3
"""Build a reproducible manifest for generated publication artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.utils.provenance import collect_provenance, file_record


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an artifact manifest for publication outputs")
    parser.add_argument("--output-dir", type=Path, required=True, help="Artifact directory to scan")
    parser.add_argument("--label", default="publication-artifacts", help="Human-readable label for this run")
    parser.add_argument(
        "--manifest-name",
        default="artifact_manifest.json",
        help="Filename for the generated manifest within the output directory",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / args.manifest_name

    files = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        if path == manifest_path:
            continue
        files.append(file_record(path, root=output_dir))

    manifest = {
        "label": args.label,
        "output_dir": str(output_dir),
        "provenance": collect_provenance(
            PROJECT_ROOT,
            argv=[sys.argv[0], *sys.argv[1:]],
            extra={"artifact_manifest_for": str(output_dir)},
        ),
        "files": files,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[manifest] Wrote {manifest_path}")
    print(f"[manifest] Indexed {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
