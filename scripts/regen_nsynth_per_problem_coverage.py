#!/usr/bin/env python3
"""Regenerate artifacts/nsynth_per_problem_coverage.jsonl from the authoritative
per-problem coverage in artifacts/nsynth_coverage.json.

The canonical record of the nSynth portfolio run is nsynth_coverage.json (a
{"rows": [...]} object, one entry per problem). The reporting/regression layer
(tests/test_nsynth_coverage.py) consumes the flattened JSONL form plus the
nsynth_per_problem_summary.json. This script derives that JSONL from the
authoritative rows so the two never drift apart.

It does NOT re-run synthesis — it reshapes already-produced results. To
re-run the Rust portfolio end to end, use benchmarks/benchmark_nsynth.py.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COVERAGE_JSON = ROOT / "artifacts" / "nsynth_coverage.json"
SUMMARY_JSON = ROOT / "artifacts" / "nsynth_per_problem_summary.json"
OUT_JSONL = ROOT / "artifacts" / "nsynth_per_problem_coverage.jsonl"


def main() -> int:
    cov = json.loads(COVERAGE_JSON.read_text())
    rows = cov["rows"]
    summary = json.loads(SUMMARY_JSON.read_text())

    passed = sum(1 for r in rows if r["success"])
    if not (len(rows) == summary["problem_count"] == passed == summary["passed"]):
        raise SystemExit(
            f"refusing to write: coverage rows ({len(rows)}, {passed} passed) "
            f"disagree with summary ({summary['problem_count']}, {summary['passed']} passed)"
        )

    lines = [json.dumps(r, sort_keys=True) for r in rows]
    lines.append(json.dumps({**summary, "summary": True}, sort_keys=True))
    OUT_JSONL.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSONL.relative_to(ROOT)}: {len(rows)} rows, {passed}/{len(rows)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
