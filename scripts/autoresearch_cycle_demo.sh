#!/usr/bin/env bash
# Smallest end-to-end continuous-autoresearch cycle that runs without a GPU.
#
# Demonstrates every stage of the loop documented in
# docs/autoresearch_continuous.md (ROADMAP.md Rung 5):
#
#   mine        — real hard-fails from the vast.ai eval JSON → work queue
#   extract     — prompt→WorkItem extraction of 4 user-shaped requests
#   run-once #1 — cascade (template_match) solves the 4, distills to store
#   re-ask      — same prompts again under NEW task ids
#   run-once #2 — compounding-store prompt-cache hits, zero cascade calls
#   status      — before/after store sizes
#
# Usage:
#   scripts/autoresearch_cycle_demo.sh [ARTIFACT_DIR]
#
# ARTIFACT_DIR defaults to a fresh mktemp dir. Pure CPU; the only network
# dependency is the HuggingFace `openai_humaneval` dataset for the `mine`
# step (skipped gracefully when unavailable — the rest of the cycle still
# runs on the 4 extracted user items).
#
# Expected result (measured 2026-06-11, Apple Silicon, CPU only):
#   session 1: 55 attempted, 4 solved by template_match, ~7s wall
#   session 2: store_hits=2, already_solved_skipped=4, by_solver={}
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ART="${1:-$(mktemp -d -t autoresearch_demo.XXXX)}"
mkdir -p "$ART"
EVAL_JSON="training_results/realworld_vastai/humaneval_agent_4B.json"

echo "[demo] artifact dir: $ART"
echo
echo "=== step 1: mine real hard-fails (53 fails -> 51 work items) ==="
if python3 -m ncpu.autoresearch.cli --artifact-dir "$ART" mine \
    --eval "$EVAL_JSON" --benchmark humaneval 2>/dev/null; then
  echo "[demo] mined $(wc -l < "$ART/humaneval_queue.jsonl" | tr -d ' ') queue items"
else
  echo "[demo] WARN: mine skipped (datasets / HF Hub unavailable);"
  echo "[demo] continuing with extracted user items only."
fi

echo
echo "=== step 2: prompt -> WorkItem extraction (4 user requests) ==="
python3 - "$ART" <<'PYEOF'
import json, sys
from pathlib import Path
from ncpu.autoresearch.prompt_parser import build_work_item

art = Path(sys.argv[1])
# One prompt per supported extraction pattern: arrow, arrow, doctest,
# fenced-assert. (Plain-prose asserts only parse inside ``` fences.)
prompts = [
    ("demo/square",
     "Write def square(x): that squares a number. "
     "square(3) -> 9, square(-4) -> 16"),
    ("demo/abs_diff",
     "def abs_diff(a, b): returns the absolute difference. "
     "abs_diff(3, 10) -> 7, abs_diff(10, 3) -> 7"),
    ("demo/add",
     "Implement def add(a, b):\n>>> add(2, 3)\n5\n>>> add(-1, 1)\n0"),
    ("demo/mul",
     "Need def mul(a, b): multiply two ints.\n"
     "```python\nassert mul(3, 4) == 12\nassert mul(5, 0) == 0\n```"),
]
with open(art / "humaneval_queue.jsonl", "a") as fh:
    for tid, prompt in prompts:
        wi = build_work_item(prompt, task_id=tid)
        assert wi is not None, f"extraction failed for {tid}"
        fh.write(json.dumps(wi.to_dict()) + "\n")
        print(f"[demo] extracted {tid}: entry={wi.entry_point} "
              f"io_pairs={len(wi.io_pairs)} "
              f"sources={wi.provenance['extraction_sources']}")
PYEOF

echo
echo "=== BEFORE: store state ==="
python3 -m ncpu.autoresearch.cli --artifact-dir "$ART" status

echo
echo "=== step 3: run-once session 1 (cascade: template_match) ==="
python3 -m ncpu.autoresearch.cli --artifact-dir "$ART" run-once \
  --benchmark humaneval --wall-seconds 300 --max-problems 60 \
  --per-problem-seconds 5 2>&1 | tail -14

echo
echo "=== step 4: re-ask 2 of the same prompts under NEW task ids ==="
python3 - "$ART" <<'PYEOF'
import json, sys
from pathlib import Path
from ncpu.autoresearch.prompt_parser import build_work_item
art = Path(sys.argv[1])
prompts = [
    ("demo/square_again",
     "Write def square(x): that squares a number. "
     "square(3) -> 9, square(-4) -> 16"),
    ("demo/add_again",
     "Implement def add(a, b):\n>>> add(2, 3)\n5\n>>> add(-1, 1)\n0"),
]
with open(art / "humaneval_queue.jsonl", "a") as fh:
    for tid, prompt in prompts:
        fh.write(json.dumps(build_work_item(prompt, task_id=tid).to_dict()) + "\n")
print("[demo] appended 2 re-asked items (same prompt hash, new task_id)")
PYEOF

echo
echo "=== step 5: run-once session 2 (expect store_hits=2, by_solver={}) ==="
python3 -m ncpu.autoresearch.cli --artifact-dir "$ART" run-once \
  --benchmark humaneval --wall-seconds 300 --max-problems 60 \
  --per-problem-seconds 5 2>&1 | tail -14

echo
echo "=== AFTER: store state ==="
python3 -m ncpu.autoresearch.cli --artifact-dir "$ART" status
echo
echo "[demo] solved log:    $(wc -l < "$ART/solved_programs.jsonl" | tr -d ' ') rows ($ART/solved_programs.jsonl)"
echo "[demo] prompt cache:  $(python3 -c "import json;print(len(json.load(open('$ART/prompt_cache.json'))))") entries ($ART/prompt_cache.json)"
echo "[demo] done. Unsolved hard-fails remain in the queue for a GPU"
echo "[demo] llm_resample session (python -m ncpu.autoresearch.driver)."
