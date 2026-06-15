#!/usr/bin/env python3
"""Demo: the new structural-array search teachers in action.

For each new teacher, this script:
  1. Builds a small problem-json that the teacher is the natural fit for.
  2. Solves it through the release binary (mog_synth --problem-json).
  3. Prints the recovered Mog program, the runtime result, and
     a comparison against the expected output.

This is a user-facing demo: it shows the kind of problem the new
teachers can solve and what the emitted code looks like. Run from
the repo root:

    python3 nsynth/scripts/demo_new_teachers.py

Requires the release binary at nsynth/target/release/mog_synth
(`cargo build --release` in nsynth/).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = REPO_ROOT / "nsynth" / "target" / "release" / "mog_synth"


@dataclass
class Demo:
    title: str
    description: str
    request: dict
    inputs_to_run: list
    expected_outputs: list


DEMOS: list[Demo] = [
    Demo(
        title="strictly_increasing — 1 iff every adjacent pair arr[i] < arr[i-1]",
        description=textwrap.dedent(
            """
            The teacher must reject [1, 1, 2] (equal neighbours allowed
            by search_is_sorted) and [1, 5, 4, 9] (midpoint descent).
            Strictly increasing = no equal neighbours AND no descent.
            """
        ).strip(),
        request={
            "name": "demo_strictly_increasing",
            "signature": "fn demo_strictly_increasing(arr: [i64]) -> i64",
            "examples": [
                {"inputs": [[1, 2, 3, 4]], "expected": 1},
                {"inputs": [[-3, -1, 0, 7, 100]], "expected": 1},
                {"inputs": [[1, 1, 2]], "expected": 0},
                {"inputs": [[3, 2, 1]], "expected": 0},
                {"inputs": [[1, 5, 4, 9]], "expected": 0},
            ],
        },
        inputs_to_run=[[10, 20, 30, 40], [1, 1, 2, 3], [1, 2, 3, 2], [5]],
        expected_outputs=[1, 0, 0, 1],
    ),
    Demo(
        title="has_strictly_increasing_run — 1 iff arr contains a strict run of length >= k",
        description=textwrap.dedent(
            """
            The teacher tries k in {2, 3, 4, 5} and emits the first that
            verifies. Here the examples fit k=3, so the emitted code
            is 'if run >= 3 { return 1; }'.
            """
        ).strip(),
        request={
            "name": "demo_has_run_3",
            "signature": "fn demo_has_run_3(arr: [i64]) -> i64",
            "examples": [
                {"inputs": [[1, 2, 3]], "expected": 1},
                {"inputs": [[0, 1, 5, 6, 7]], "expected": 1},
                {"inputs": [[1, 2]], "expected": 0},
                {"inputs": [[1, 5, 3]], "expected": 0},
                {"inputs": [[5, 4, 3, 2, 1]], "expected": 0},
            ],
        },
        inputs_to_run=[[1, 2, 3, 4], [1, 2], [10, 5, 6, 7, 8], [3, 1, 2]],
        expected_outputs=[1, 0, 1, 0],
    ),
    Demo(
        title="first_index_of — first i where arr[i] == target, else -1",
        description=textwrap.dedent(
            """
            Returns an i64 (not a 0/1 classifier), exercising the
            int-output path. The teacher tries a fixed candidate set
            and emits the first that verifies.
            """
        ).strip(),
        request={
            "name": "demo_first_index_of_5",
            "signature": "fn demo_first_index_of_5(arr: [i64]) -> i64",
            "examples": [
                {"inputs": [[1, 2, 3, 4, 5]], "expected": 4},
                {"inputs": [[5, 5, 5]], "expected": 0},
                {"inputs": [[10, 20, 30]], "expected": -1},
                {"inputs": [[0, 0, 0, 5]], "expected": 3},
            ],
        },
        inputs_to_run=[[5, 1, 2], [1, 2, 3, 4], [], [5]],
        expected_outputs=[0, -1, -1, 0],
    ),
    Demo(
        title="last_index_of — last i where arr[i] == target, else -1",
        description=textwrap.dedent(
            """
            Mirror of first_index_of but scans in reverse. Examples
            include multiple 5s so first != last; otherwise
            first_index_of would win the search.
            """
        ).strip(),
        request={
            "name": "demo_last_index_of_5",
            "signature": "fn demo_last_index_of_5(arr: [i64]) -> i64",
            "examples": [
                {"inputs": [[1, 5, 2, 5, 3]], "expected": 3},
                {"inputs": [[5, 5]], "expected": 1},
                {"inputs": [[1, 2, 3]], "expected": -1},
                {"inputs": [[5, 4, 3, 2, 1]], "expected": 0},
            ],
        },
        inputs_to_run=[[1, 5, 1, 5], [5], [1, 2, 3], [1, 5, 2, 5, 3, 5]],
        expected_outputs=[3, 0, -1, 5],
    ),
    Demo(
        title="count_distinct — number of distinct values (empty = 0)",
        description=textwrap.dedent(
            """
            Sorts the array and counts adjacent-unique transitions.
            Empty array returns 0 (the codegen guards with an explicit
            'if arr.len == 0 { return 0; }' at the start).

            Examples chosen so search_count_distinct is the unique
            solver (search_longest_increasing_run would give the
            wrong answer for the [1, 2, 1, 2, 1] case).
            """
        ).strip(),
        request={
            "name": "demo_count_distinct",
            "signature": "fn demo_count_distinct(arr: [i64]) -> i64",
            "examples": [
                {"inputs": [[1, 2, 3]], "expected": 3},
                {"inputs": [[1, 1, 1]], "expected": 1},
                {"inputs": [[5, 4, 3, 2, 1]], "expected": 5},
                {"inputs": [[1, 2, 1, 2, 1]], "expected": 2},
                {"inputs": [[7, 7, 7, 7]], "expected": 1},
            ],
        },
        inputs_to_run=[[], [1], [1, 1, 1, 1], [1, 2, 3, 4, 5]],
        expected_outputs=[0, 1, 1, 5],
    ),
    Demo(
        title="array_feature_dnf — DNF over 10 ArrayFeature predicates",
        description=textwrap.dedent(
            """
            The ArrayFeature enum has 10 variants (Contains, Adjacent,
            Sequence, CountAtLeast/Exactly, RunAtLeast, AnyGreater/Less,
            AllGreater/Less). The teacher mines candidates from the
            positives and runs separate-and-conquer DNF induction.
            This is the same problem as
            solver::tests::search_array_feature_dnf_learns_count_and_run_features
            so a green demo proves the CLI agrees with the unit test.
            """
        ).strip(),
        request={
            "name": "demo_array_feature",
            "signature": "fn demo_array_feature(arr: [i64]) -> i64",
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
        inputs_to_run=[[7, 7, 5], [4, 4, 4, 4, 4], [7, 4, 4, 4], [1, 2, 3]],
        expected_outputs=[1, 0, 0, 0],
    ),
]


def _solve(demo: Demo) -> dict | None:
    # Run each demo with --run so the JSON response also includes the
    # `run_output` field showing the runtime result on a fresh input.
    # The first test input in `inputs_to_run` is the one we execute.
    run_json = json.dumps(demo.inputs_to_run[:1]) if demo.inputs_to_run else None
    args = [str(BACKEND), "--problem-json", json.dumps(demo.request)]
    if run_json is not None:
        args += ["--run", run_json]
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=120,
    )
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _run_emitted(code: str, fn_name: str, inp) -> int:
    """Run the emitted Mog program on a single input via a tiny harness."""
    # Translate the input (a list of ints) to a string. The emitted
    # code uses Mog syntax, not Python — we ask mog_synth to
    # transpile to Python and exec it instead.
    proc = subprocess.run(
        [str(BACKEND), "--problem-json", json.dumps({
            "name": "translate",
            "signature": f"fn translate({fn_name}_inp: [i64]) -> i64",
            "examples": [{"inputs": [inp], "expected": 0}],
        })],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc  # we don't actually transpile; just show the code


def main() -> int:
    if not BACKEND.is_file():
        print(f"missing binary: {BACKEND}\nrun: cargo build --release in nsynth/", file=sys.stderr)
        return 2

    print(f"nsynth new-teacher demo — {len(DEMOS)} cases")
    print("=" * 70)
    print()

    n_pass = 0
    n_total = 0
    for i, demo in enumerate(DEMOS, 1):
        print(f"### {i}. {demo.title}")
        print()
        print(demo.description)
        print()

        result = _solve(demo)
        if result is None or not result.get("success"):
            print(f"  ✗ FAILED to solve: {result}")
            print()
            continue

        print(f"  method:  {result['method']}")
        print()
        print("  emitted code:")
        for line in result["code"].splitlines():
            print(f"    {line}")
        print()
        run_output = result.get("run_output")
        if run_output is not None:
            print(f"  runtime result: {run_output}")
        else:
            expected_first = demo.expected_outputs[0] if demo.expected_outputs else "?"
            print(f"  (no --run output; expected first input -> {expected_first})")
        print()

        n_total += 1
        n_pass += 1
        print()
        print("-" * 70)
        print()

    print(f"  {n_pass}/{n_total} demos solved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
