#!/usr/bin/env python3
"""Language-understanding benchmark suite (nsynth + LinguaGenesis).

Cross-project end-to-end: every task takes its data from the
LinguaGenesis curriculum (not hand-authored). The bridge script
emits a Problem JSON, the synthesis API solves it, and we record
the recovered Mog program, the method used, and the holdout
accuracy. This file is the SUITE RUNNER — it iterates over a
fixed task list and prints a summary table.

Tasks (all are 100% curriculum-sourced; see
LINGUAGENESIS_BRIDGE.md and the comments in linguagenesis_bridge.py
for the per-task oracle):

  Word-level (string -> i64):
    verb_3sg_es       Classify verbs: takes -es in 3sg? (ch/sh/ss/x/z/o)
    verb_past_form    Generate past-tense form of a verb (string -> string)
    verb_gerund_form  Generate -ing form of a verb (string -> string)
    verb_3sg_form     Generate 3sg present form (string -> string)
    pluralize_gen     Generate plural form of a noun (string -> string)

  Sentence-level (string -> i64):
    sentence_3sg      "The dog walks." vs "The dog walk." (3sg agreement)
    sentence_past     Past-tense agreement
    sentence_gerund   Subject-verb agreement under gerund
    sentence_full     Full multi-feature grammaticality
    formal_logic      Stage 1 logic: quantifier/formula classification

  Semantic (string -> i64):
    semantic_roles    Subject/object role labeling

Usage:
    python3 language_benchmark_suite.py
    python3 language_benchmark_suite.py --tasks verb_3sg_es,pluralize_gen
    python3 language_benchmark_suite.py --url http://127.0.0.1:8093
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

NSYNTH = Path(__file__).resolve().parent.parent
BRIDGE = NSYNTH / "scripts" / "linguagenesis_bridge.py"
MOG_SYNTH = NSYNTH / "target" / "release" / "mog_synth"

# All language tasks.  Each tuple is (task name, expected holdout > 0).
TASKS = [
    # Word-level classification (string -> i64)
    "verb_3sg_es",
    # Word-level generation (string -> string)
    "verb_3sg_form",
    "verb_past_form",
    "verb_gerund_form",
    "pluralize_gen",
    # Sentence-level
    "sentence_3sg",
    "sentence_past",
    "sentence_gerund",
    "sentence_full",
    "formal_logic",
    # Semantic
    "semantic_roles",
]


def run_task(task: str, url: str) -> dict:
    """Run one task through the bridge + mog_synth.  Returns dict
    with name, method, success, holdout_accuracy (if available),
    duration."""
    t0 = time.time()
    try:
        # 1. Build the problem via the bridge script.
        bridge_proc = subprocess.run(
            [sys.executable, str(BRIDGE), "--task", task],
            capture_output=True,
            text=True,
            check=True,
        )
        problem_json = bridge_proc.stdout
    except subprocess.CalledProcessError as e:
        return {
            "task": task,
            "success": False,
            "error": f"bridge failed: {e.stderr.strip()[:200]}",
            "duration": time.time() - t0,
        }

    # 2. Run mog_synth on the problem JSON. We pass the JSON via stdin
    # so the bridge can be piped directly.  mog_synth uses --problem-json
    # with "-" for stdin.
    try:
        synth_proc = subprocess.run(
            [str(MOG_SYNTH), "--problem-json", "-"],
            input=problem_json,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        # Find the last JSON-looking line in stdout.
        result_line = None
        for line in synth_proc.stdout.splitlines()[::-1]:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    result_line = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        if result_line is None:
            return {
                "task": task,
                "success": False,
                "error": f"no JSON in mog_synth stdout. stderr: {synth_proc.stderr.strip()[:200]}",
                "duration": time.time() - t0,
            }
    except subprocess.TimeoutExpired:
        return {
            "task": task,
            "success": False,
            "error": "mog_synth timeout (>60s)",
            "duration": time.time() - t0,
        }

    return {
        "task": task,
        "success": result_line.get("success", False),
        "method": result_line.get("method", "?"),
        "error": result_line.get("error"),
        "duration": time.time() - t0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", default=",".join(TASKS), help="comma-separated task names"
    )
    parser.add_argument("--url", default="http://127.0.0.1:8093")
    parser.add_argument(
        "--quiet", action="store_true", help="only print the summary table"
    )
    args = parser.parse_args()
    selected = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not selected:
        print("no tasks selected", file=sys.stderr)
        return 1
    if not MOG_SYNTH.exists():
        print(f"mog_synth not found at {MOG_SYNTH}; build first", file=sys.stderr)
        return 1
    if not BRIDGE.exists():
        print(f"bridge not found at {BRIDGE}", file=sys.stderr)
        return 1

    results = []
    print(f"running {len(selected)} language-understanding tasks")
    for task in selected:
        r = run_task(task, args.url)
        results.append(r)
        if not args.quiet:
            tag = "PASS" if r["success"] else "FAIL"
            method = r.get("method", "?")
            dur = r.get("duration", 0)
            print(f"  [{tag}] {task:25s} method={method:35s} {dur:.2f}s")
            if not r["success"] and r.get("error"):
                print(f"        error: {r['error'][:120]}")

    # Summary
    n_pass = sum(1 for r in results if r["success"])
    n_total = len(results)
    print()
    print("=" * 70)
    print(f"Language-understanding benchmark suite: {n_pass}/{n_total} pass")
    print("=" * 70)
    print(f"{'task':25s} {'method':35s} {'time':>6s}")
    print("-" * 70)
    for r in results:
        tag = "PASS" if r["success"] else "FAIL"
        method = r.get("method", "?")[:35]
        dur = r.get("duration", 0)
        print(f"  {r['task']:25s} {method:35s} {dur:5.2f}s  [{tag}]")
    print("=" * 70)
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
