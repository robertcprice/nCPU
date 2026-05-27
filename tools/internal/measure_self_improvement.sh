#!/usr/bin/env bash
# Run a scoped cumulative sweep, compute self-improvement metrics, append a
# dated row to artifacts/SELF_IMPROVEMENT_RATE.md.
#
# Designed to be committed from a weekly cron. The resulting markdown file
# becomes a published trajectory of the system's learning progress — the
# answer to "does this thing keep getting smarter?" as a releasable artifact.
#
# Usage:
#   tools/measure_self_improvement.sh [--limit N] [--offset M]
#
# Columns emitted:
#   date | cache_size | joined | median_ratio | instant_hits | slowdowns | transfer_pct | top_teacher_success

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT/nsynth"

LIMIT=30
OFFSET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)  LIMIT="$2"; shift 2 ;;
    --offset) OFFSET="$2"; shift 2 ;;
    -h|--help) sed -n '1,15p' "$0"; exit 0 ;;
    *) echo "[measure] unknown arg: $1" >&2; exit 2 ;;
  esac
done

if ! [[ -x target/release/transfer_curve ]] || \
   ! [[ -x target/release/curve_analysis ]] || \
   ! [[ -x target/release/top_teachers ]]; then
  cargo build --release \
    --bin transfer_curve --bin curve_analysis --bin top_teachers > /dev/null 2>&1
fi

TMPDIR="$(mktemp -d -t nsynth_self_improvement.XXXX)"
trap 'rm -rf "$TMPDIR"' EXIT

CUM_FILE="$TMPDIR/curve_cum.jsonl"

echo "[measure] running cumulative sweep: offset=$OFFSET limit=$LIMIT rounds=2"
NSYNTH_TEACHER_BUDGET_SEC=10 ./target/release/transfer_curve \
  --rounds 2 \
  --offset "$OFFSET" \
  --limit "$LIMIT" \
  --out "$CUM_FILE" \
  --quiet > /dev/null 2>&1

ANALYSIS=$(./target/release/curve_analysis \
  --baseline "$CUM_FILE" \
  --treatment "$CUM_FILE" \
  --baseline-round 0 \
  --treatment-round 1 \
  --json)

MEDIAN=$(echo "$ANALYSIS" | sed 's/.*"improvement_rate_median":\([^,}]*\).*/\1/')
INSTANT_HITS=$(echo "$ANALYSIS" | sed 's/.*"instant_hits":\([^,}]*\).*/\1/')
SLOWDOWNS=$(echo "$ANALYSIS" | sed 's/.*"slowdowns":\([^,}]*\).*/\1/')
TRANSFER_PCT=$(echo "$ANALYSIS" | sed 's/.*"via_cached_teachers_pct":\([^,}]*\).*/\1/')
JOINED=$(echo "$ANALYSIS" | sed 's/.*"joined":\([0-9]*\).*/\1/')

# Cache size: read the file's length instead of spinning up another binary.
CACHE_PATH="${NSYNTH_CACHE_PATH:-$HOME/.nsynth_solved_programs.json}"
if [[ -f "$CACHE_PATH" ]]; then
  CACHE_SIZE=$(wc -l < "$CACHE_PATH" | tr -d ' ')
else
  CACHE_SIZE=0
fi

# Top teacher's success_count (via JSONL output, grabbed safely with sed).
TOP_SUCCESS=$(./target/release/top_teachers --top 1 --json 2>/dev/null | \
  sed -n '1{s/.*"success_count":\([0-9]*\).*/\1/;p;}')
: "${TOP_SUCCESS:=0}"

OUT_FILE="$REPO_ROOT/artifacts/SELF_IMPROVEMENT_RATE.md"
mkdir -p "$(dirname "$OUT_FILE")"

if [[ ! -f "$OUT_FILE" ]]; then
  cat > "$OUT_FILE" <<'HDR'
# Self-Improvement Rate

Published trajectory of the nsynth solver's cross-run learning progress.
Each row is a scoped cumulative transfer_curve sweep; median_ratio is
round-1-over-round-0 time-per-problem. Lower is better (learning working);
> 1.0 means the cache is not being useful.

| date (UTC) | cache_size | joined | median_ratio | instant_hits | slowdowns | transfer_pct | top_teacher_success |
|------------|-----------:|-------:|-------------:|-------------:|----------:|-------------:|--------------------:|
HDR
fi

DATE_UTC=$(date -u +%Y-%m-%d)
ROW=$(printf "| %s | %s | %s | %s | %s | %s | %s | %s |" \
  "$DATE_UTC" "$CACHE_SIZE" "$JOINED" "$MEDIAN" "$INSTANT_HITS" "$SLOWDOWNS" "$TRANSFER_PCT" "$TOP_SUCCESS")

echo "$ROW" >> "$OUT_FILE"
echo "[measure] appended to $OUT_FILE:"
echo "$ROW"
