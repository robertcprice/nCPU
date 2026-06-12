#!/usr/bin/env python3
"""Cold/warm negative-memory benchmark for the nSynth rejected-program cache.

Automates the REPRODUCIBILITY Claim 7 measurement ("failed search never
repeats"): runs the solver twice on a NOVEL out-of-space problem with every
persistent nsynth bank isolated under a temp directory, and asserts that the
warm rerun adds ZERO new rejection hashes to the rejected-program TSV
(``nsynth/src/rejected_cache.rs``) — i.e. the entire failed search
deduplicated across runs.

The problem (``sum_above_first``: sum of elements strictly greater than the
first element) is outside the current solver portfolio: the run must end in
an honest refusal after exhausting the cascade. If the portfolio ever grows
to solve it, this harness exits non-zero telling you to pick a harder
problem — a solved problem records no rejections and proves nothing.

Each solver run grinds the full gradient cascade to exhaustion (~2-3 min).

Usage:
    python benchmarks/benchmark_negative_memory.py
    python benchmarks/benchmark_negative_memory.py --out artifacts/negative_memory_benchmark.json
    python benchmarks/benchmark_negative_memory.py --binary /path/to/mog_synth

Artifact schema (validated by tests/test_negative_memory_benchmark.py):
    {cold_rejections, warm_new_rejections, converged, problem,
     wall_cold_s, wall_warm_s, ...}
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BINARY = PROJECT_ROOT / "nsynth" / "target" / "release" / "mog_synth"
DEFAULT_ARTIFACT = PROJECT_ROOT / "artifacts" / "negative_memory_benchmark.json"

# Novel out-of-space problem: sum of elements strictly greater than xs[0].
# Requires a conditional fold against a scalar derived from the array itself,
# which no current solver family expresses. Verified to refuse (success:
# false) after full cascade exhaustion.
PROBLEM = {
    "name": "sum_above_first",
    "signature": "fn sum_above_first(xs: &[i64]) -> i64",
    "examples": [
        {"inputs": [[3, 1, 4, 1, 5]], "expected": 9},
        {"inputs": [[10, 2, 3]], "expected": 0},
        {"inputs": [[1, 2, 3, 4]], "expected": 9},
        {"inputs": [[5, 5, 5]], "expected": 0},
        {"inputs": [[2, 7, 1, 8]], "expected": 15},
        {"inputs": [[0, -1, 3, 0, 2]], "expected": 5},
    ],
    "holdouts": [
        {"inputs": [[4, 9, 9, 1]], "expected": 18},
        {"inputs": [[7]], "expected": 0},
    ],
}

# Every env var that points the Rust backend at persistent state (same list
# as tests/mcp_server/conftest.py::_BANK_ENV). All banks live under one temp
# directory shared by the cold and warm runs, so the rejected TSV persists
# across runs (that persistence IS the feature under test) while nothing
# reads or pollutes the user's real ~/.nsynth_* banks. The solved cache is
# disabled outright ('' sentinel) so a cache hit can never skip the search.
_BANK_FILES = {
    "NSYNTH_CACHE_PATH": "",  # '' disables — must never short-circuit the search
    "NSYNTH_BIAS_BANK_PATH": "learned_biases.jsonl",
    "NSYNTH_REJECTED_PATH": "rejected_programs.tsv",
    "NSYNTH_METHOD_ROUTER_PATH": "method_router.json",
    "NSYNTH_SEARCH_FAMILY_ROUTER_PATH": "search_family_router.json",
    "NSYNTH_TEACHER_FAILURES_PATH": "teacher_failures.json",
    "NSYNTH_META_WEIGHTS_PATH": "meta_weights.json",
    "NSYNTH_BOOTSTRAP_STATE_PATH": "bootstrap_state.json",
    "NSYNTH_BOOTSTRAP_MARKER_PATH": "bootstrap_marker.json",
    "NSYNTH_AUTOTUNE_CONFIG": "autotune_config.json",
}


def isolated_env(bank_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    for var, fname in _BANK_FILES.items():
        env[var] = str(bank_dir / fname) if fname else ""
    return env


def count_rejection_hashes(tsv_path: Path) -> int:
    """Total rejection hashes in the bank.

    TSV row format (see nsynth/src/rejected_cache.rs):
        last_used \\t comma-joined-hashes \\t examples-fingerprint
    """
    if not tsv_path.is_file():
        return 0
    total = 0
    for line in tsv_path.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) < 3 or not fields[1]:
            continue
        total += len(fields[1].split(","))
    return total


def run_solver(binary: Path, env: dict[str, str]) -> tuple[dict, float]:
    """One full solve attempt. Returns (result_json, wall_seconds)."""
    start = time.monotonic()
    proc = subprocess.run(
        [str(binary), "--problem-json", "-"],
        input=json.dumps(PROBLEM),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    wall = time.monotonic() - start
    if proc.returncode != 0:
        raise SystemExit(
            f"solver exited {proc.returncode}\nstderr tail:\n{proc.stderr[-2000:]}"
        )
    # The result object is the last JSON line on stdout (stderr carries logs).
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line), wall
    raise SystemExit(f"no JSON result on solver stdout:\n{proc.stdout[-2000:]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_ARTIFACT, help="artifact JSON path"
    )
    args = parser.parse_args(argv)

    if not args.binary.is_file():
        raise SystemExit(
            f"nsynth binary not found at {args.binary}. "
            f"Build it with: (cd nsynth && cargo build --release)"
        )

    with tempfile.TemporaryDirectory(prefix="negmem_banks_") as tmp:
        bank_dir = Path(tmp)
        env = isolated_env(bank_dir)
        rejected_tsv = bank_dir / _BANK_FILES["NSYNTH_REJECTED_PATH"]

        print(f"[negmem] problem: {PROBLEM['name']} (banks: {bank_dir})")
        print("[negmem] cold run (full cascade to exhaustion, ~2-3 min)...")
        cold_result, wall_cold = run_solver(args.binary, env)
        if cold_result.get("success"):
            raise SystemExit(
                "PROBLEM NO LONGER NOVEL: the solver synthesized "
                f"{PROBLEM['name']} via {cold_result.get('method')!r}. A solved "
                "problem records no rejections — pick a harder problem."
            )
        cold_rejections = count_rejection_hashes(rejected_tsv)
        print(
            f"[negmem] cold: refused in {wall_cold:.1f}s, "
            f"{cold_rejections} rejection hashes persisted"
        )
        if cold_rejections == 0:
            raise SystemExit(
                "cold run persisted zero rejections — the refusal happened "
                "before any candidate search ran; pick a problem that reaches "
                "the gradient cascade."
            )

        print("[negmem] warm run (identical problem, banks retained)...")
        warm_result, wall_warm = run_solver(args.binary, env)
        if warm_result.get("success"):
            raise SystemExit("warm run unexpectedly solved the problem")
        warm_total = count_rejection_hashes(rejected_tsv)
        warm_new = warm_total - cold_rejections
        converged = warm_new == 0
        print(
            f"[negmem] warm: refused in {wall_warm:.1f}s, "
            f"{warm_new} NEW rejection hashes (total {warm_total})"
        )

    artifact = {
        "claim": "nSynth persistent negative memory: failed search never repeats",
        "problem": PROBLEM,
        "cold_rejections": cold_rejections,
        "warm_new_rejections": warm_new,
        "warm_total_rejections": warm_total,
        "converged": converged,
        "wall_cold_s": round(wall_cold, 2),
        "wall_warm_s": round(wall_warm, 2),
        "cold_refusal_error": cold_result.get("error"),
        "binary": str(args.binary.relative_to(PROJECT_ROOT))
        if args.binary.is_relative_to(PROJECT_ROOT)
        else str(args.binary),
        "generated_unix": int(time.time()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n")

    print(
        f"\n[negmem] SUMMARY: cold={cold_rejections} rejections in "
        f"{wall_cold:.1f}s | warm added {warm_new} in {wall_warm:.1f}s | "
        f"converged={converged}"
    )
    print(f"[negmem] artifact written: {args.out}")
    if not converged:
        print("[negmem] FAIL: warm run grew the rejection bank", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
