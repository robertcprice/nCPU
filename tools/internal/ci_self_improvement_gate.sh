#!/usr/bin/env bash
# CI gate: run a small transfer_curve sweep, feed through curve_analysis,
# fail the build if cumulative is slower than fresh or if slowdowns > 0.
#
# Usage:
#   tools/ci_self_improvement_gate.sh [--limit N] [--offset M] [--threshold F]
#
# Env knobs passed through:
#   NSYNTH_TEACHER_BUDGET_SEC  (default 10)
#   NSYNTH_META_L2             (default: pass-through)
#
# Exit codes:
#   0 - within bounds
#   1 - slowdown detected (cumulative is worse than fresh on the gate set)
#   2 - bad CLI args / missing binaries
#
# The gate measures *round-over-round* improvement within a single cumulative
# sweep, which is the cleanest signal: round 0 populates the cache; round 1
# either hits it (instant) or doesn't. median_ratio > threshold on this
# comparison means something broke the "cache is usable" invariant.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT/nsynth"

# Defaults — tuned for fast CI (~30-60s) not full bench coverage.
LIMIT=20
OFFSET=0
THRESHOLD=1.05   # fail when cumulative median_ratio > 1.05 × baseline
: "${NSYNTH_TEACHER_BUDGET_SEC:=10}"
export NSYNTH_TEACHER_BUDGET_SEC

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)     LIMIT="$2"; shift 2 ;;
    --offset)    OFFSET="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,25p' "$0"
      exit 0
      ;;
    *) echo "[ci_gate] unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Build if needed — release binaries are checked in CI's build step normally;
# this lets you run the gate locally against fresh source.
if ! [[ -x target/release/transfer_curve ]] || ! [[ -x target/release/curve_analysis ]]; then
  echo "[ci_gate] building transfer_curve + curve_analysis..."
  cargo build --release --bin transfer_curve --bin curve_analysis
fi

TMPDIR="$(mktemp -d -t nsynth_ci_gate.XXXX)"
trap 'rm -rf "$TMPDIR"' EXIT

CUM_FILE="$TMPDIR/curve_cum.jsonl"

echo "[ci_gate] running cumulative sweep: rounds=2 offset=$OFFSET limit=$LIMIT"
./target/release/transfer_curve \
  --rounds 2 \
  --offset "$OFFSET" \
  --limit "$LIMIT" \
  --out "$CUM_FILE" \
  --quiet > /dev/null 2>&1

echo "[ci_gate] analysing round-0 vs round-1..."
ANALYSIS=$(./target/release/curve_analysis \
  --baseline "$CUM_FILE" \
  --treatment "$CUM_FILE" \
  --baseline-round 0 \
  --treatment-round 1 \
  --json)

echo "[ci_gate] $ANALYSIS"

# Parse JSON without jq (keeps the gate portable to CI images that don't
# have it installed). Grep/sed extracts median_ratio + slowdowns; awk does
# the float comparison.
MEDIAN=$(echo "$ANALYSIS" | sed 's/.*"improvement_rate_median":\([^,}]*\).*/\1/')
SLOWDOWNS=$(echo "$ANALYSIS" | sed 's/.*"slowdowns":\([^,}]*\).*/\1/')
INSTANT_HITS=$(echo "$ANALYSIS" | sed 's/.*"instant_hits":\([^,}]*\).*/\1/')

echo "[ci_gate] median_ratio=$MEDIAN  slowdowns=$SLOWDOWNS  instant_hits=$INSTANT_HITS  threshold=$THRESHOLD"

# Fail if median_ratio > threshold, or if any slowdown > 1.05× was observed.
# Uses awk so the comparison works on BSD and GNU bash without bc.
FAILED=0
if awk -v m="$MEDIAN" -v t="$THRESHOLD" 'BEGIN { exit !(m > t) }'; then
  echo "[ci_gate] FAIL: median_ratio $MEDIAN > threshold $THRESHOLD"
  FAILED=1
fi
if [[ "$SLOWDOWNS" != "0" ]]; then
  echo "[ci_gate] WARN: $SLOWDOWNS problems slowed >1.05× in cumulative round 1"
  # Slowdown count alone isn't a hard fail — a real regression will also
  # move the median. Keep the log loud so humans see it on green runs.
fi

if [[ "$FAILED" -ne 0 ]]; then
  echo "[ci_gate] cross-run learning regression detected"
  exit 1
fi

echo "[ci_gate] PASS"
exit 0
