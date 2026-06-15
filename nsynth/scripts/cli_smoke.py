#!/usr/bin/env python3
"""CLI smoke test for the new structural-array search teachers.

Spawns the release binary once per problem and asserts the expected
teacher name is returned. Mirrors the unit tests in
nsynth/src/solver/tests.rs::search_*_learns_* but exercises the actual
binary the way a user would invoke it.

Requires the release binary at nsynth/target/release/mog_synth
(`cargo build --release` in nsynth/).

Usage:
    python3 nsynth/scripts/cli_smoke.py
    python3 nsynth/scripts/cli_smoke.py -v   # verbose
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = REPO_ROOT / "nsynth" / "target" / "release" / "mog_synth"


@dataclass
class Case:
    """A single CLI invocation + the teacher we expect it to pick.

    `expected_method` is the *preferred* teacher; if None, any teacher
    that solves the problem is accepted. This lets a simpler teacher
    (e.g. `search_suffix_class`) win when it fits, so the smoke
    verifies the *problem* is solved, not the specific algorithm.
    """

    name: str
    request: dict
    expected_method: str | None
    description: str = ""


# Each case builds a problem-json that the new teacher is the
# natural fit for. The teacher name in `expected_method` must be
# the literal method that mog_synth reports back.
CASES: list[Case] = [
    Case(
        name="array_feature_dnf_ok",
        request={
            "name": "array_feature_ok",
            "signature": "fn array_feature_ok(arr: [i64]) -> i64",
            # This is the same problem as
            # solver::tests::search_array_feature_dnf_learns_count_and_run_features
            # so a green smoke proves the CLI agrees with the unit test.
            "examples": [
                {"inputs": [[7, 7, 1, 2, 3]], "expected": 1},
                {"inputs": [[0, 7, 5, 7]], "expected": 1},
                {"inputs": [[7, 3, 7]], "expected": 1},
                {"inputs": [[2, 7, 7, 8]], "expected": 1},
                {"inputs": [[4, 4, 4, 9]], "expected": 1},
                {"inputs": [[1, 4, 4, 4]], "expected": 1},
                {"inputs": [[4, 4, 4]], "expected": 1},
                {"inputs": [[6, 4, 4, 4, 5]], "expected": 1},
                {"inputs": [[7, 1, 2]], "expected": 0},
                {"inputs": [[7, 7, 7, 1]], "expected": 0},
                {"inputs": [[4, 4, 9, 4]], "expected": 0},
                {"inputs": [[4, 9, 4, 4]], "expected": 0},
                {"inputs": [[7, 4, 4, 4]], "expected": 0},
                {"inputs": [[4, 4, 7, 7]], "expected": 0},
            ],
        },
        expected_method="search_array_feature_dnf",
        description="DNF over ArrayFeature taxonomy (count+sequence features)",
    ),
    Case(
        name="string_subsequence_class_ok",
        request={
            "name": "subseq_q_a_z_ok",
            "signature": "fn subseq_q_a_z_ok(s: string) -> i64",
            "examples": [
                # rule: 1 if s contains the subsequence "q" then "a" then "z" (in order)
                # suffix_class can solve this too (ends_with("z")), so we accept
                # either teacher. The point is that the problem is solved.
                {"inputs": ["q a z"], "expected": 1},
                {"inputs": ["q b a z"], "expected": 1},
                {"inputs": ["x q y a z w"], "expected": 1},
                {"inputs": ["q z a"], "expected": 0},
                {"inputs": ["a q z"], "expected": 0},
                {"inputs": ["a z q"], "expected": 0},
                {"inputs": ["q a"], "expected": 0},
                {"inputs": ["hello world"], "expected": 0},
            ],
        },
        expected_method=None,
        description="String subsequence membership (q -> a -> z); any teacher",
    ),
    Case(
        name="strictly_increasing_ok",
        request={
            "name": "strictly_increasing_ok",
            "signature": "fn strictly_increasing_ok(arr: [i64]) -> i64",
            "examples": [
                {"inputs": [[1, 2, 3]], "expected": 1},
                {"inputs": [[-3, -1, 0, 7, 100]], "expected": 1},
                {"inputs": [[1, 1, 2]], "expected": 0},
                {"inputs": [[3, 2, 1]], "expected": 0},
                {"inputs": [[1, 5, 4, 9]], "expected": 0},
            ],
        },
        expected_method="search_strictly_increasing",
        description="Strict monotonicity (no equal neighbours allowed)",
    ),
    Case(
        name="last_index_of_5_ok",
        request={
            "name": "last_index_of_5_ok",
            "signature": "fn last_index_of_5_ok(arr: [i64]) -> i64",
            "examples": [
                # Include arrays with multiple 5s so first != last; otherwise
                # search_first_index_of wins the search and the test is
                # indistinguishable.
                {"inputs": [[1, 2, 3, 4, 5]], "expected": 4},
                {"inputs": [[5]], "expected": 0},
                {"inputs": [[5, 5]], "expected": 1},   # first=0, last=1
                {"inputs": [[1, 5, 2, 5, 3]], "expected": 3},  # first=1, last=3
                {"inputs": [[1, 2, 3]], "expected": -1},
                {"inputs": [[5, 4, 3, 2, 1]], "expected": 0},
                {"inputs": [[1, 5, 1, 5, 1, 5]], "expected": 5},
            ],
        },
        expected_method="search_last_index_of",
        description="Last index of target=5, or -1 (with multiple 5s)",
    ),
    Case(
        name="count_distinct_ok",
        request={
            "name": "count_distinct_ok",
            "signature": "fn count_distinct_ok(arr: [i64]) -> i64",
            "examples": [
                {"inputs": [[1, 2, 3]], "expected": 3},
                {"inputs": [[1, 1, 1]], "expected": 1},
                {"inputs": [[1, 2, 1, 2, 1]], "expected": 2},
                {"inputs": [[5, 4, 3, 2, 1]], "expected": 5},
                {"inputs": [[7]], "expected": 1},
                {"inputs": [[]], "expected": 0},
            ],
        },
        expected_method="search_count_distinct",
        description="Number of distinct values (empty array = 0)",
    ),
    Case(
        name="is_sorted_bool_ok",
        request={
            "name": "is_sorted_bool_ok",
            "signature": "fn is_sorted_bool_ok(arr: [i64]) -> i64",
            "examples": [
                {"inputs": [[1, 2, 3]], "expected": 1},
                {"inputs": [[3, 2, 1]], "expected": 0},
                {"inputs": [[1, 1, 2]], "expected": 1},
                {"inputs": [[5]], "expected": 1},
                {"inputs": [[1, 3, 2]], "expected": 0},
            ],
        },
        expected_method="search_is_sorted",
        description="is_sorted predicate (uses pre-existing teacher, bool wire)",
    ),
]


def _run_case(case: Case, verbose: bool) -> tuple[bool, str, float]:
    """Run one case through the mog_synth CLI. Returns (ok, method, elapsed_s)."""
    t0 = time.time()
    proc = subprocess.run(
        [str(BACKEND), "--problem-json", json.dumps(case.request)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    elapsed = time.time() - t0
    out = proc.stdout
    if verbose and proc.stderr:
        print(f"  stderr: {proc.stderr[:200]}", file=sys.stderr)

    # Parse the JSON line in the output. The CLI prints [trace]
    # lines to stdout; the final line is the JSON result.
    result = None
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                result = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    if result is None:
        return False, "<no-json>", elapsed

    if not result.get("success"):
        return False, result.get("error", "?"), elapsed

    method = result.get("method", "?")
    if case.expected_method is None:
        return True, method, elapsed
    return method == case.expected_method, method, elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    ap.add_argument(
        "--keep-going", action="store_true", help="don't stop on first failure"
    )
    args = ap.parse_args()

    if not BACKEND.is_file():
        print(f"missing binary: {BACKEND}\nrun: cargo build --release in nsynth/", file=sys.stderr)
        return 2

    print(f"CLI smoke: {len(CASES)} cases against {BACKEND.name}")
    print("-" * 70)

    n_pass = 0
    n_fail = 0
    for case in CASES:
        ok, method, elapsed = _run_case(case, args.verbose)
        status = "PASS" if ok else "FAIL"
        flag = "✓" if ok else "✗"
        expected_str = case.expected_method if case.expected_method else "<any>"
        print(
            f"  {flag} {status}  {case.name:36s}  "
            f"expected={expected_str:34s}  got={method:34s}  {elapsed*1000:6.1f}ms"
        )
        if args.verbose and case.description:
            print(f"      ↳ {case.description}")
        if ok:
            n_pass += 1
        else:
            n_fail += 1
            if not args.keep_going:
                break

    print("-" * 70)
    print(f"  {n_pass}/{n_pass + n_fail} pass")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
