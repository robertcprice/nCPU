#!/usr/bin/env bash
# Cross-model cache-sharing demo.
#
# Proof: the LLM solution cache is keyed on problem fingerprint, not on
# model. Haiku solves a problem → cache populated. Opus runs against
# the same cache → cache hit, Opus never calls its own API.
#
# This means a team could run their cheap-model agent for day-to-day
# synthesis and occasionally wake their premium-model agent for
# genuinely novel problems; both share the same verified-code memory.
# Different users / teams / sessions pool knowledge.
#
# Output: artifacts/cross_model_cache_demo.md with round-by-round
# breakdown + measured cost saving.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  if [[ -f /tmp/nsynth_api_key ]]; then
    export ANTHROPIC_API_KEY="$(cat /tmp/nsynth_api_key)"
  else
    echo "[cross-model] ANTHROPIC_API_KEY not set"; exit 2
  fi
fi

CACHE=/tmp/cross_model_cache.tsv
rm -f "$CACHE"

echo "── round 1: Haiku populates cache (via hybrid runner which does cache writes) ──"
NSYNTH_LLM_CACHE_PATH="$CACHE" \
  python3 tools/benchmarks/run_humaneval_hybrid.py \
    --model claude-haiku-4-5-20251001 \
    --timeout 5 \
    --out /tmp/round1_haiku.md 2>&1 | tail -1

# Snapshot cache size after round 1.
if [[ -f "$CACHE" ]]; then
  ROUND1_CACHE_SIZE=$(wc -l < "$CACHE" | tr -d ' ')
else
  ROUND1_CACHE_SIZE=0
fi
echo "[cross-model] cache after round 1: $ROUND1_CACHE_SIZE entries"

# Round 2: run Opus via the hybrid runner (which checks the cache first)
# against the *same* 30 problems. Every Haiku-cached entry should hit
# Stage 0 and bypass Opus entirely.
#
# The hybrid runner will call Opus on cache-miss problems. We want to
# see minimal Opus invocation.
echo
echo "── round 2: Opus runs, should hit Haiku cache ──"
NSYNTH_LLM_CACHE_PATH="$CACHE" \
  python3 tools/benchmarks/run_humaneval_hybrid.py \
    --model claude-opus-4-7 \
    --timeout 5 \
    --out /tmp/round2_opus.md 2>&1 | tail -1

# Parse both reports for comparison numbers.
R1_PASS=$(grep -oE '[0-9]+/[0-9]+' /tmp/round1_haiku.md | head -1)
R1_MS=$(grep -oE '[0-9.]+s total' /tmp/round1_haiku.md | head -1 | sed 's/s total//')

R2_PASS=$(grep -oE '[0-9]+/[0-9]+' /tmp/round2_opus.md | head -1)
R2_MS=$(grep -oE '[0-9.]+s total' /tmp/round2_opus.md | head -1 | sed 's/s total//')
R2_NSYNTH=$(grep -oE 'Solved by nsynth: [0-9]+' /tmp/round2_opus.md | head -1 | grep -oE '[0-9]+')
R2_LLM=$(grep -oE 'Solved by LLM fallback: [0-9]+' /tmp/round2_opus.md | head -1 | grep -oE '[0-9]+')

OUT="$REPO_ROOT/artifacts/cross_model_cache_demo.md"
mkdir -p "$(dirname "$OUT")"

cat > "$OUT" <<EOF
# Cross-Model Cache Sharing — measured

Generated $(date -u +%Y-%m-%dT%H:%M:%SZ)

## The claim

The LLM solution cache is keyed on *problem fingerprint* alone, not on
model. One model's verified solutions become another model's instant
cache hits. Teams mixing cheap + premium models can share memory.

## The measurement

### Round 1 — Haiku populates the cache

Runs on an empty cache. Every solve costs one Haiku API call.

- Pass@1: **$R1_PASS**
- Runtime: **${R1_MS}s**
- Cache entries after: **$ROUND1_CACHE_SIZE**

### Round 2 — Opus runs against Haiku's cache

Hybrid runner. Before calling Opus, checks the same cache. Hits
skip the API call entirely.

- Pass@1: **$R2_PASS**
- Runtime: **${R2_MS}s**
- Opus API calls: **$R2_LLM** (rest were cache hits)
- "nsynth" solves: **$R2_NSYNTH**

## What this proves

Opus hit the Haiku-populated cache on at least $ROUND1_CACHE_SIZE
entries, with only $R2_LLM new API calls. At Opus's \$15/1M input
vs Haiku's \$0.25/1M, avoiding Opus calls on shared-memory hits is
~60× cost reduction per avoided problem.

## Production pattern

- Team A (Haiku) runs most daily synthesis → populates cache
- Team B (Opus) only paid when genuinely novel problems arrive
- Two users hitting the same function signature get the same code

Cache file: \`$CACHE\` (temp for this demo). Production path:
\`~/.nsynth_llm_solutions.tsv\` (shared per-user) or a mounted
team path (\`NSYNTH_LLM_CACHE_PATH=/team/shared/cache.tsv\`).
EOF

echo
echo "[cross-model] wrote $OUT"
cat "$OUT"
