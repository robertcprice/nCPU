#!/usr/bin/env bash
# MBPP benchmark driver: run the LLM-free engine on each representable MBPP task in
# its OWN process under a per-task OS timeout, so a pathological search is killed
# cleanly. Usage: run_mbpp_bench.sh <bench.jsonl> [per_task_timeout_s] [limit]
set -u
BENCH="${1:?usage: run_mbpp_bench.sh <bench.jsonl> [timeout_s] [limit]}"
TMO="${2:-5}"
LIMIT="${3:-100000}"
BIN="target/release/mbpp_solve_one"
[ -x "$BIN" ] || BIN="target/debug/mbpp_solve_one"
[ -x "$BIN" ] || { echo "build first: cargo build --release --bin mbpp_solve_one"; exit 1; }

solved=0; unsolved=0; skip=0; killed=0; n=0
solved_ids=""
while IFS= read -r line; do
  n=$((n+1)); [ "$n" -gt "$LIMIT" ] && break
  out=$(printf '%s' "$line" | timeout "$TMO" "$BIN" 2>/dev/null)
  rc=$?
  if [ "$rc" -eq 124 ]; then killed=$((killed+1)); continue; fi
  case "$out" in
    SOLVED*)   solved=$((solved+1)); solved_ids="$solved_ids ${out#SOLVED }";;
    UNSOLVED*) unsolved=$((unsolved+1));;
    SKIP*)     skip=$((skip+1));;
    *)         killed=$((killed+1));;
  esac
done < "$BENCH"

attempted=$((solved+unsolved+killed))
echo "[MBPP] attempted=$attempted  solved=$solved  unsolved=$unsolved  timeout_killed=$killed  skipped=$skip"
if [ "$attempted" -gt 0 ]; then
  awk "BEGIN{printf \"[MBPP] solve-rate = %.1f%% of representable (%d/%d); per-task timeout=${TMO}s\n\", 100*$solved/$attempted, $solved, $attempted}"
fi
echo "[MBPP] solved_ids:$solved_ids"
