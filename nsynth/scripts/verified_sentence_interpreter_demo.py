#!/usr/bin/env python3
"""Verified sentence interpreter demo (nsynth + LinguaGenesis).

End-to-end demonstration that nsynth can synthesize verified Mog
programs for each of the three core English-subset agreement
features, then conceptually shows how to combine them into a
full sentence interpreter via AND-of-features over the token stream.

The three features (all from the LinguaGenesis Stage 3 morphology
generator):

  3sg agreement       "The dog walks." vs "The dog walk."
  past agreement      "The dog walked." vs "The dog walk."
  gerund agreement    "The dog is walking." vs "The dog are walking."

Each feature is learned independently as a binary classifier over
a token-id array (the morpheme-tokenized sentence). nsynth
recovers a small DNF-style Mog program for each. The full
interpreter is the AND of the three features.

Run:
    # 1. Start the synthesis API server in another shell:
    #      python3 ncpu/synthesis_api/server.py
    # 2. Run the demo:
    #      python3 nsynth/scripts/verified_sentence_interpreter_demo.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

NSYNTH = Path(__file__).resolve().parent.parent
BRIDGE = NSYNTH / "scripts" / "linguagenesis_bridge.py"
MOG_SYNTH = NSYNTH / "target" / "release" / "mog_synth"

# Three agreement features, each is a sentence-level task in the bridge.
FEATURES = [
    ("sentence_3sg",     "3sg agreement",      "valid_3sg"),
    ("sentence_past",    "past agreement",     "valid_past"),
    ("sentence_gerund",  "gerund agreement",   "valid_gerund"),
]


def run_feature(bridge_task: str) -> dict:
    """Run one feature through bridge + mog_synth. Returns {task, success, method, code, duration}."""
    t0 = time.time()
    bridge_proc = subprocess.run(
        [sys.executable, str(BRIDGE), "--task", bridge_task],
        capture_output=True,
        text=True,
        check=True,
    )
    synth_proc = subprocess.run(
        [str(MOG_SYNTH), "--problem-json", "-"],
        input=bridge_proc.stdout,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    # Pull last JSON line.
    result = None
    for line in synth_proc.stdout.splitlines()[::-1]:
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                result = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if result is None:
        return {
            "task": bridge_task,
            "success": False,
            "error": synth_proc.stderr.strip()[:200] or "no JSON",
            "duration": time.time() - t0,
        }
    return {
        "task": bridge_task,
        "success": result.get("success", False),
        "method": result.get("method"),
        "code": result.get("code"),
        "error": result.get("error"),
        "duration": time.time() - t0,
    }


def main() -> int:
    print("=" * 70)
    print("VERIFIED SENTENCE INTERPRETER DEMO")
    print("nsynth + LinguaGenesis fusion, end-to-end")
    print("=" * 70)
    print()
    print("Concept: each English agreement feature is learned as a binary")
    print("classifier over a token-id array. The full interpreter is the")
    print("AND of all three features.")
    print()

    results = []
    for bridge_task, label, fn_name in FEATURES:
        print(f"--- {label} ({bridge_task}) ---")
        r = run_feature(bridge_task)
        results.append((label, fn_name, r))
        if r["success"]:
            print(f"  PASS in {r['duration']:.2f}s via {r['method']}")
            code = r["code"]
            # Show just the body of the function (not the signature line)
            if code:
                for line in code.split("\n"):
                    print(f"    {line}")
        else:
            print(f"  FAIL in {r['duration']:.2f}s — {r.get('error', '?')[:120]}")
        print()

    print("=" * 70)
    print("FULL INTERPRETER (conceptual)")
    print("=" * 70)
    print()
    print("The full Stage-3 interpreter is the AND of the three features:")
    print()
    print("  fn interpreter(sent: [i64]) -> i64 {")
    print("      if valid_3sg(sent) == 0 { return 0; }")
    print("      if valid_past(sent) == 0 { return 0; }")
    print("      if valid_gerund(sent) == 0 { return 0; }")
    print("      return 1;")
    print("  }")
    print()
    print("Each `valid_*` is a Mog function recovered by nsynth from the")
    print("LinguaGenesis curriculum. The conjunction is mechanical: at")
    print("runtime, every token array is checked against all three; the")
    print("sentence is grammatical iff all three classifiers return 1.")
    print()
    n_pass = sum(1 for _, _, r in results if r["success"])
    print(f"summary: {n_pass}/{len(FEATURES)} features verified")
    return 0 if n_pass == len(FEATURES) else 1


if __name__ == "__main__":
    sys.exit(main())
