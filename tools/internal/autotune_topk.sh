#!/usr/bin/env bash
# Auto-tune TEACHER_TOPK from measured data.
#
# Runs tools/diversity_pareto.sh, reads the resulting CSV, picks the
# Pareto-dominant K value, and writes it to
# tools/config/nsynth_autotune.tsv. The solver's teacher_topk() reads
# that file as a fallback when NSYNTH_TEACHER_TOPK isn't set, so the
# next production run picks up the newly-tuned value automatically.
#
# Selection rule: among K ≠ 0 entries, find the one with the highest
# win_pct. If multiple tie, pick the one with the lowest mean_ms. This
# is a simple realization of "dominant in the (accuracy → speed)
# Pareto order".
#
# Usage:
#   tools/autotune_topk.sh [--offset N] [--limit M]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OFFSET=40
LIMIT=20
while [[ $# -gt 0 ]]; do
  case "$1" in
    --offset) OFFSET="$2"; shift 2 ;;
    --limit)  LIMIT="$2"; shift 2 ;;
    -h|--help) sed -n '1,15p' "$0"; exit 0 ;;
    *) echo "[autotune] unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Delegate to the Pareto sweep (it builds the binaries if needed).
"$REPO_ROOT/tools/diversity_pareto.sh" --offset "$OFFSET" --limit "$LIMIT" > /dev/null

CSV="$REPO_ROOT/artifacts/diversity_pareto.csv"
if [[ ! -s "$CSV" ]]; then
  echo "[autotune] $CSV missing or empty — aborting"
  exit 1
fi

# Parse the CSV: k,wins,attempts,win_pct,mean_ms. Find the K (excluding
# 0, which is the reference "no cap" mode) with highest win_pct; break
# ties by lowest mean_ms. awk handles both the float comparison and the
# tie-break rule without needing jq/python in CI.
BEST=$(
  awk -F',' '
    NR == 1 { next }          # header
    $1 == 0 { next }           # exclude reference
    {
      pct = $4 + 0
      ms  = $5 + 0
      if (pct > best_pct || (pct == best_pct && ms < best_ms)) {
        best_pct = pct
        best_ms  = ms
        best_k   = $1
      }
    }
    END {
      if (best_k != "") print best_k, best_pct, best_ms
    }
  ' "$CSV"
)

if [[ -z "$BEST" ]]; then
  echo "[autotune] no candidate K values in $CSV — nothing to autotune"
  exit 1
fi

BEST_K=$(echo "$BEST" | awk '{print $1}')
BEST_PCT=$(echo "$BEST" | awk '{print $2}')
BEST_MS=$(echo "$BEST" | awk '{print $3}')

CONFIG_DIR="$REPO_ROOT/tools/config"
CONFIG_FILE="$CONFIG_DIR/nsynth_autotune.tsv"
mkdir -p "$CONFIG_DIR"

{
  echo "# Autotuned nsynth config. Generated $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# by tools/autotune_topk.sh from artifacts/diversity_pareto.csv."
  echo "# Format: key<TAB>value. One key per line. Read by strategy.rs"
  echo "# when NSYNTH_TEACHER_TOPK env var is unset."
  printf "topk\t%s\n" "$BEST_K"
  printf "measured_win_pct\t%s\n" "$BEST_PCT"
  printf "measured_mean_ms\t%s\n" "$BEST_MS"
  printf "measured_offset\t%s\n" "$OFFSET"
  printf "measured_limit\t%s\n" "$LIMIT"
} > "$CONFIG_FILE"

echo "[autotune] best K = $BEST_K  (win_pct=$BEST_PCT, mean_ms=$BEST_MS)"
echo "[autotune] wrote $CONFIG_FILE"

# Append one row to the autotune history TSV. Every run preserves its
# measurement — plotting the history answers "how has the Pareto-optimal
# K moved as the cache grew?" over time. First column is epoch seconds so
# downstream analysis is trivial.
HISTORY="$REPO_ROOT/artifacts/autotune_history.tsv"
mkdir -p "$(dirname "$HISTORY")"
if [[ ! -s "$HISTORY" ]]; then
  printf "ts\tcache_size\tchosen_k\twin_pct\tmean_ms\toffset\tlimit\n" > "$HISTORY"
fi
# Cache size: line-count of the live cache file, matching what
# measure_self_improvement uses.
CACHE_PATH="${NSYNTH_CACHE_PATH:-$HOME/.nsynth_solved_programs.json}"
if [[ -f "$CACHE_PATH" ]]; then
  CACHE_SIZE=$(wc -l < "$CACHE_PATH" | tr -d ' ')
else
  CACHE_SIZE=0
fi
TS=$(date -u +%s)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "$TS" "$CACHE_SIZE" "$BEST_K" "$BEST_PCT" "$BEST_MS" "$OFFSET" "$LIMIT" >> "$HISTORY"
echo "[autotune] appended row to $HISTORY"
