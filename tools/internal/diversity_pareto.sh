#!/usr/bin/env bash
# Sweep NSYNTH_TEACHER_TOPK across a grid of K values, measuring the
# (wins, mean_ms) pair at each. Produces the Pareto frontier that
# characterizes the ranker's (accuracy ⇄ speed) tradeoff.
#
# Output: artifacts/diversity_pareto.csv + artifacts/diversity_pareto.md
#
# Usage:
#   tools/diversity_pareto.sh [--offset N] [--limit M]
#
# The resulting CSV has one row per K:
#   k,wins,attempts,win_pct,mean_ms
# And the MD file renders the same data as a table any reader can scan.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT/nsynth"

OFFSET=40
LIMIT=20

while [[ $# -gt 0 ]]; do
  case "$1" in
    --offset) OFFSET="$2"; shift 2 ;;
    --limit)  LIMIT="$2"; shift 2 ;;
    -h|--help) sed -n '1,15p' "$0"; exit 0 ;;
    *) echo "[pareto] unknown arg: $1" >&2; exit 2 ;;
  esac
done

if ! [[ -x target/release/diversity_ab ]]; then
  cargo build --release --bin diversity_ab > /dev/null 2>&1
fi

CSV="$REPO_ROOT/artifacts/diversity_pareto.csv"
MD="$REPO_ROOT/artifacts/diversity_pareto.md"
mkdir -p "$(dirname "$CSV")"

# Sweep grid. K=0 is the "no cap / exhaustive" reference; the other K
# values progressively loosen the cap. Mode A in each run is always K=0 so
# we have a stable baseline.
KS=(4 8 16 32 48 64 0)

# Fresh CSV + MD header each run.
printf "k,wins,attempts,win_pct,mean_ms\n" > "$CSV"
{
  echo "# Diversity Pareto Sweep"
  echo
  echo "Offset: ${OFFSET}, Limit: ${LIMIT}, generated $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "K=0 is the reference (no cap, full rank). The other K values cap"
  echo "top-K with the diversity pass active. Lower win_pct + lower mean_ms"
  echo "means the cap is too tight; ~equal win_pct with lower mean_ms is a"
  echo "Pareto improvement."
  echo
  echo "| K | wins | attempts | win_pct | mean_ms |"
  echo "|--:|-----:|---------:|--------:|--------:|"
} > "$MD"

for K in "${KS[@]}"; do
  # Mode A is fixed at 0 (no cap / baseline); mode B is the swept K.
  echo "[pareto] sweeping K=$K ..."
  JSON=$(./target/release/diversity_ab \
    --offset "$OFFSET" --limit "$LIMIT" \
    --topk-a 0 --topk-b "$K" \
    --json 2>/dev/null || true)

  # Extract B's stats — A's stats are the reference so we emit them once at K=0.
  WINS_B=$(echo "$JSON" | sed 's/.*"b":{"wins":\([0-9]*\).*/\1/')
  ATT_B=$(echo "$JSON" | sed 's/.*"b":{"wins":[0-9]*,"attempts":\([0-9]*\).*/\1/')
  PCT_B=$(echo "$JSON" | sed 's/.*"b":{"wins":[0-9]*,"attempts":[0-9]*,"win_pct":\([^,]*\).*/\1/')
  MS_B=$(echo "$JSON" | sed 's/.*"b":{"wins":[0-9]*,"attempts":[0-9]*,"win_pct":[^,]*,"mean_ms":\([^}]*\).*/\1/')

  printf "%s,%s,%s,%s,%s\n" "$K" "$WINS_B" "$ATT_B" "$PCT_B" "$MS_B" >> "$CSV"
  printf "| %s | %s | %s | %s | %s |\n" "$K" "$WINS_B" "$ATT_B" "$PCT_B" "$MS_B" >> "$MD"
done

echo
echo "[pareto] done. CSV=$CSV MD=$MD"
echo
echo "Quick summary:"
cat "$MD"
