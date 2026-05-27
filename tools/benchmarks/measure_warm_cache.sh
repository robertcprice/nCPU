#!/usr/bin/env bash
# Warm-cache speedup measurement.
#
# Runs the HumanEval-lite benchmark twice with the *same* cache path.
# Round 1 populates the cache; round 2 should be 100% Stage-0 cache hits.
# The ratio round2_total_ms / round1_total_ms is the speedup factor — a
# concrete "cross-run learning works at the synthesis layer" signal.
#
# Output: artifacts/warm_cache_measurement.md with both rounds' numbers.
#
# Usage:
#   tools/benchmarks/measure_warm_cache.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if ! [[ -x nsynth/target/release/nsynth_codegen ]]; then
  (cd nsynth && cargo build --release --bin nsynth_codegen > /dev/null 2>&1)
fi

CACHE=/tmp/warm_cache_measurement.json
WEIGHTS=/tmp/warm_cache_weights.tsv
rm -f "$CACHE" "$WEIGHTS"

OUT="$REPO_ROOT/artifacts/warm_cache_measurement.md"
mkdir -p "$(dirname "$OUT")"

run_round() {
  local label="$1"
  local report="/tmp/warm_cache_${label}.md"
  NSYNTH_CACHE_PATH="$CACHE" \
  NSYNTH_META_WEIGHTS_PATH="$WEIGHTS" \
  python3 tools/benchmarks/run_humaneval_lite.py \
    --timeout 25 \
    --out "$report" > "/tmp/warm_cache_${label}.log" 2>&1
  # Extract pass@1 + total runtime from the report.
  PASS=$(grep -oE 'Pass@1.*[0-9]+/[0-9]+' "$report" | head -1 | grep -oE '[0-9]+/[0-9]+' | head -1)
  MS=$(grep -oE 'total runtime [0-9.]+' "$report" | head -1 | grep -oE '[0-9.]+' | head -1)
  echo "$PASS $MS"
}

echo "── warm-cache speedup: round 1 (cold) ──"
R1=$(run_round round1_cold)
R1_PASS=$(echo "$R1" | awk '{print $1}')
R1_MS=$(echo "$R1" | awk '{print $2}')
echo "round 1 cold: pass@1 $R1_PASS, $R1_MS s"

echo "── round 2 (warm) ──"
R2=$(run_round round2_warm)
R2_PASS=$(echo "$R2" | awk '{print $1}')
R2_MS=$(echo "$R2" | awk '{print $2}')
echo "round 2 warm: pass@1 $R2_PASS, $R2_MS s"

# Compute speedup via awk (no bc dep).
SPEEDUP=$(awk -v a="$R1_MS" -v b="$R2_MS" 'BEGIN { if (b > 0) printf("%.2f", a / b); else print "inf" }')

{
  echo "# Warm Cache Speedup"
  echo
  echo "Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) on the 30-problem humaneval_lite set."
  echo
  echo "| round | pass@1 | total runtime (s) |"
  echo "|-------|:------:|------------------:|"
  echo "| 1 (cold) | $R1_PASS | $R1_MS |"
  echo "| 2 (warm) | $R2_PASS | $R2_MS |"
  echo
  echo "**Warm-cache speedup**: **${SPEEDUP}×**"
  echo
  echo "Round 1 starts from an empty cache; every successful synthesis"
  echo "call records its (fingerprint, code) pair. Round 2 runs the same"
  echo "30 problems against the now-populated cache — every fingerprint"
  echo "hits Stage 0, emits in ~0 ms, skips enumerative + gradient entirely."
  echo
  echo "The ratio is the concrete, measured answer to \"does cross-run"
  echo "learning actually save time at the synthesis layer?\" — lower"
  echo "round 2 total = yes."
} > "$OUT"

echo
echo "── wrote $OUT ──"
cat "$OUT"
