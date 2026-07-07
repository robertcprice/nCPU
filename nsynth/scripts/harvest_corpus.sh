#!/usr/bin/env bash
# Self-sufficiency harvester. Run the LLM-FREE engine (mbpp_solve_one) over a corpus
# of {text, examples} specs and append every VERIFIED (task -> Mog) pair to a JSONL
# training file (mlx_lm.lora chat format, via the engine's own NSYNTH_HARVEST hook).
#
# The corpus is nsynth's OWN verified output: rejection-sampling / STaR — only
# programs the verifier accepted are harvested, so the training data is correct by
# construction. The local model later trains on this (finetune_local.sh), learning
# to drive nsynth from its own proofs — NO external teacher. This is the mechanism
# by which the agent becomes self-sufficient.
#
# Usage: harvest_corpus.sh <corpus.jsonl> [out.jsonl] [per_task_timeout_s]
set -u
CORPUS="${1:?usage: harvest_corpus.sh <corpus.jsonl> [out.jsonl] [timeout_s]}"
OUT="${2:-harvest.jsonl}"
TMO="${3:-8}"
BIN="${MBPP_BIN:-target/release/mbpp_solve_one}"
[ -x "$BIN" ] || BIN="target/debug/mbpp_solve_one"
[ -x "$BIN" ] || { echo "build first: cargo build --release --bin mbpp_solve_one"; exit 1; }

# Snapshot the binary so a sibling agent's `cargo clean` can't break a long run.
SNAP="$(mktemp)"; cp "$BIN" "$SNAP"; chmod +x "$SNAP"
: > "$OUT"
solved=0; total=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  total=$((total+1))
  # Each task in its OWN process; NSYNTH_HARVEST appends the verified pair on SOLVE.
  res=$(printf '%s' "$line" | NSYNTH_HARVEST="$OUT" NSYNTH_ENUM_BUDGET_MS=$((TMO*1000)) "$SNAP" 2>/dev/null)
  case "$res" in SOLVED*) solved=$((solved+1));; esac
done < "$CORPUS"
rm -f "$SNAP"
echo "[harvest] $solved/$total verified pairs -> $OUT ($(wc -l < "$OUT" 2>/dev/null | tr -d ' ') lines)"
