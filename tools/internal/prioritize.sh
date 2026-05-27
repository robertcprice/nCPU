#!/usr/bin/env bash
# Surface near-miss program families as a committed prioritization list.
#
# Runs transfer_curve with NSYNTH_LOG_TEACHER_FAILURES=1 so the cached-
# teachers stage writes artifacts/transfer_failures.jsonl whenever it
# exhausts its top-K without a win. Then feeds that file into
# near_miss_clusters to produce a human-readable ranked list of program
# families the solver keeps trying and keeps failing on.
#
# Output: artifacts/SOLVER_PRIORITIZATION.md — the "where should solver
# development focus next?" list, committed and grep-able.
#
# Usage:
#   tools/prioritize.sh [--offset N] [--limit M] [--k K]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT/nsynth"

OFFSET=0
LIMIT=50
K=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --offset) OFFSET="$2"; shift 2 ;;
    --limit)  LIMIT="$2"; shift 2 ;;
    --k)      K="$2"; shift 2 ;;
    -h|--help) sed -n '1,20p' "$0"; exit 0 ;;
    *) echo "[prioritize] unknown arg: $1" >&2; exit 2 ;;
  esac
done

if ! [[ -x target/release/transfer_curve ]] || ! [[ -x target/release/near_miss_clusters ]]; then
  cargo build --release --bin transfer_curve --bin near_miss_clusters > /dev/null 2>&1
fi

# Isolated artifacts so repeated runs don't pollute each other. The failure
# log gets reset first so the list reflects only this run's misses.
FAILURES="$REPO_ROOT/artifacts/transfer_failures.jsonl"
mkdir -p "$(dirname "$FAILURES")"
: > "$FAILURES"

TMPCURVE=$(mktemp -t prioritize_curve.XXXX.jsonl)
trap 'rm -f "$TMPCURVE"' EXIT

echo "[prioritize] running transfer_curve with failure logging enabled..."
# Cumulative mode (no --fresh-cache) so the cache grows during the sweep and
# CachedTeachers actually fires. Isolated cache path keeps this from
# polluting the user's real ~/.nsynth_solved_programs.json.
TMP_CACHE=$(mktemp -t prioritize_cache.XXXX.json)
trap 'rm -f "$TMPCURVE" "$TMP_CACHE"' EXIT
NSYNTH_LOG_TEACHER_FAILURES=1 \
NSYNTH_TEACHER_FAILURES_PATH="$FAILURES" \
NSYNTH_TEACHER_BUDGET_SEC=8 \
NSYNTH_CACHE_PATH="$TMP_CACHE" \
  ./target/release/transfer_curve \
    --rounds 1 \
    --offset "$OFFSET" \
    --limit "$LIMIT" \
    --out "$TMPCURVE" \
    --quiet > /dev/null 2>&1

FAIL_COUNT=$(wc -l < "$FAILURES" | tr -d ' ')
echo "[prioritize] captured $FAIL_COUNT near-miss rows"

MD="$REPO_ROOT/artifacts/SOLVER_PRIORITIZATION.md"

if [[ "$FAIL_COUNT" -eq 0 ]]; then
  {
    echo "# Solver Prioritization"
    echo
    echo "Generated $(date -u +%Y-%m-%dT%H:%M:%SZ)."
    echo
    echo "No cached-teachers misses captured — CachedTeachers either did"
    echo "not fire (problems solved by earlier pipeline stages) or every"
    echo "top-K attempt transferred. Consider a harder problem slice:"
    echo "\`tools/prioritize.sh --offset 90 --limit 15\`."
  } > "$MD"
  echo "[prioritize] no misses — wrote placeholder to $MD"
  cat "$MD"
  exit 0
fi

CLUSTERS=$(./target/release/near_miss_clusters --in "$FAILURES" --k "$K" --min-cluster-size 1 2>&1 \
  | grep -v '^\[' || true)

{
  echo "# Solver Prioritization"
  echo
  echo "Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) from a sweep of"
  echo "offset=${OFFSET}, limit=${LIMIT} with CachedTeachers misses logged."
  echo
  echo "## Near-miss clusters (k=$K)"
  echo
  echo "These are program families the solver attempted (top-K teacher"
  echo "distillation) but never converged on. A tight cluster = shape the"
  echo "system *keeps* failing on = concrete solver-work target."
  echo
  echo '```'
  echo "$CLUSTERS"
  echo '```'
  echo
  echo "Total near-miss rows: $FAIL_COUNT"
} > "$MD"

echo "[prioritize] wrote $MD"
echo
cat "$MD"
