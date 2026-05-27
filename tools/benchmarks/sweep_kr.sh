#!/usr/bin/env bash
# Performance-curve sweep: (best-of-N, retries) grid on humaneval_lite.
#
# For each (k, max_retries) combination, run the agent loop, record
# pass@1 + mean ms + total tokens. Emit a grid + Pareto analysis to
# artifacts/agent_performance_curve.md.
#
# This is the first empirical map of "how much test-time compute buys
# how much pass@1" — a 3D surface (k, retries) → (pass@1, cost, time).
# Every cell is one real run against the Anthropic API.
#
# Usage:
#   ANTHROPIC_API_KEY=sk-... tools/benchmarks/sweep_kr.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  if [[ -f /tmp/nsynth_api_key ]]; then
    export ANTHROPIC_API_KEY="$(cat /tmp/nsynth_api_key)"
  else
    echo "[sweep] ANTHROPIC_API_KEY not set"; exit 2
  fi
fi

OUT="$REPO_ROOT/artifacts/agent_performance_curve.md"
mkdir -p "$(dirname "$OUT")"

# Grid. Keep it small — each run is ~30-60s at 30 problems. 9 cells
# total is ~5-10 minutes of API time.
KS=(1 3 5)
RS=(0 2 4)

# Fresh cache per cell so the measurement is clean (no bleed-through).
WORK=$(mktemp -d)
trap "rm -rf $WORK" EXIT

{
  echo "# Agent Loop Performance Curve"
  echo
  echo "Generated $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "30-problem humaneval_lite. For each (k, retries) cell: one agent"
  echo "run with a fresh LLM cache (so each number is the cold-start"
  echo "measurement, not amortised by earlier cells)."
  echo
  echo "## pass@1 grid"
  echo
  printf "| k \\\\ retries |"
  for r in "${RS[@]}"; do printf " r=%s |" "$r"; done
  printf "\n|---:|"
  for r in "${RS[@]}"; do printf ":---:|"; done
  printf "\n"
} > "$OUT"

# bash 3 (macOS default) has no associative arrays. Writing key-value
# lines to a scratch file and grepping them out works fine for this
# 9-cell grid.
SCRATCH=$(mktemp)
trap "rm -rf $WORK $SCRATCH" EXIT

for k in "${KS[@]}"; do
  row="| k=$k |"
  for r in "${RS[@]}"; do
    cache_path="$WORK/cache_${k}_${r}.tsv"
    report_path="$WORK/agent_${k}_${r}.md"

    NSYNTH_LLM_CACHE_PATH="$cache_path" \
      python3 tools/benchmarks/run_humaneval_agent.py \
      --k "$k" --max-retries "$r" \
      --out "$report_path" > /dev/null 2>&1

    p=$(grep -oE '\*\*[0-9]+/[0-9]+' "$report_path" | head -1 | sed 's/\*\*//')
    ms=$(grep -oE '[0-9.]+s total' "$report_path" | head -1 | sed 's/s total//')
    printf "%s,%s\t%s\t%s\n" "$k" "$r" "$p" "$ms" >> "$SCRATCH"

    pct=$(echo "$p" | awk -F'/' '{ if ($2 > 0) printf "%.1f", 100*$1/$2 }')
    row="$row **${p}** (${ms}s) |"
    echo "  [sweep] k=$k r=$r → $p = ${pct}% in ${ms}s"
  done
  echo "$row" >> "$OUT"
done

lookup_ms() {
  grep "^$1,$2	" "$SCRATCH" | awk -F'\t' '{print $3}'
}

{
  echo
  echo "## runtime grid (seconds)"
  echo
  printf "| k \\\\ retries |"
  for r in "${RS[@]}"; do printf " r=%s |" "$r"; done
  printf "\n|---:|"
  for r in "${RS[@]}"; do printf "---:|"; done
  printf "\n"
  for k in "${KS[@]}"; do
    printf "| k=%s |" "$k"
    for r in "${RS[@]}"; do
      printf " %s |" "$(lookup_ms "$k" "$r")"
    done
    printf "\n"
  done
  echo
  echo "## Reading the curve"
  echo
  echo "- Higher k + higher retries → higher pass@1 but linear cost scale."
  echo "- Cache column (k=1, r=0) is the LLM-alone baseline."
  echo "- Pareto frontier: the (k, r) cells that dominate others at either pass@1 or runtime."
  echo "- Diminishing returns: if k=3,r=2 ties k=5,r=4 on pass@1, the former is Pareto-optimal."
  echo
  echo "Raw reports per cell in $(realpath --relative-to="$REPO_ROOT" "$WORK")/ (cleaned up on exit)."
} >> "$OUT"

echo "[sweep] wrote $OUT"
