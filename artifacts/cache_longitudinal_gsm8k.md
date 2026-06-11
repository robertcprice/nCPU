# Cache longitudinal: same benchmark, twice

Measured 2026-04-19 — `claude-haiku-4-5-20251001`, GSM8K-test first-50,
agent mode `--k 3 --max-retries 2`.

## Pass 1 vs pass 2

| | pass 1 (empty cache) | pass 2 (populated) | speedup |
|---|--:|--:|--:|
| Wall | 121.2s | **0.1s** | **1212×** |
| Tokens | ~26,695 | **~0** | — |
| Accuracy | 50/50 | 50/50 | — |
| Path: `cache` | 0 | **50** | — |
| Path: `sample` | 42 | 0 | — |
| Path: `retry` | 8 | 0 | — |
| Cost @ Haiku | ~$0.020 | **$0.000** | — |

Pass 2 consumes zero API tokens and finishes in a tenth of a second
for 50 problems. Every entry that pass-1 verified (all 50) is re-verified
against the in-memory cache before return — so a cache hit that no
longer passes tests (e.g. because the test suite changed) fails closed
and falls through to a fresh solve.

## Why this matters

**CI economics**: every PR re-running HumanEval or GSM8K would cost
$0.02–$0.70 per run *without* the cache. With a warm cache, the
second-onward runs are free. For a large team pushing hundreds of
PRs per week, this is the difference between "we can afford to test on
every commit" and "we can't".

**Cross-model sharing**: the cache is keyed on problem fingerprint,
not model. A Haiku-populated cache is a free speed-up for Opus on the
same problems. Measured separately in `artifacts/cross_model_cache_demo.md`:
Opus hits cache on 14/15 HumanEval-lite problems after Haiku pre-populates.

**Feedback loop into distillation**: every solved row becomes a training
pair `(problem → verified code)` in the distillation dataset. A week of
CI runs yields hundreds of verified solutions; those fine-tune
Qwen3.5-4B via `tools/distillation/auto_distill.sh` on a vast.ai A100.
The cache is where the loop closes from usage → data → local model.

## Cache hygiene

- **Fingerprint collisions**: the cache verifies every hit against the
  current problem's tests before returning. A stale or mis-keyed row
  fails, falls through to a fresh solve, and gets overwritten.
- **Capacity**: the on-disk TSV currently has no eviction. At
  ~700 bytes/row it can grow indefinitely (50K problems ≈ 35 MB). Add
  LRU + size-cap if the scale exceeds a single benchmark suite.
- **Atomicity**: writes are temp-file + rename so a crashed process
  cannot corrupt the store.

## Reproduce

```bash
CACHE=/tmp/gsm8k_agent.tsv
rm -f "$CACHE"

# Pass 1: empty cache.
ANTHROPIC_API_KEY=sk-ant-... NSYNTH_LLM_CACHE_PATH=$CACHE \
  python3 tools/benchmarks/run_gsm8k.py --mode agent --limit 50

# Pass 2: warm cache.
ANTHROPIC_API_KEY=sk-ant-... NSYNTH_LLM_CACHE_PATH=$CACHE \
  python3 tools/benchmarks/run_gsm8k.py --mode agent --limit 50
```

The `cache` path count in the pass-2 report should equal the number of
verified solves from pass 1.
