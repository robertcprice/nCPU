# Cross-Model Cache Sharing — measured

Generated 2026-04-19T00:24:22Z

## The claim

The LLM solution cache is keyed on *problem fingerprint* alone, not on
model. One model's verified solutions become another model's instant
cache hits. Teams mixing cheap + premium models can share memory.

## The measurement

### Round 1 — Haiku populates the cache

Runs on an empty cache. Every solve costs one Haiku API call.

- Pass@1: **29/30**
- Runtime: **97.4s**
- Cache entries after: **15**

### Round 2 — Opus runs against Haiku's cache

Hybrid runner. Before calling Opus, checks the same cache. Hits
skip the API call entirely.

- Pass@1: **29/30**
- Runtime: **13.3s**
- Opus API calls: **1** (rest were cache hits)
- "nsynth" solves: **13**

## What this proves

Opus hit the Haiku-populated cache on at least 15
entries, with only 1 new API calls. At Opus's $15/1M input
vs Haiku's $0.25/1M, avoiding Opus calls on shared-memory hits is
~60× cost reduction per avoided problem.

## Production pattern

- Team A (Haiku) runs most daily synthesis → populates cache
- Team B (Opus) only paid when genuinely novel problems arrive
- Two users hitting the same function signature get the same code

Cache file: `/tmp/cross_model_cache.tsv` (temp for this demo). Production path:
`~/.nsynth_llm_solutions.tsv` (shared per-user) or a mounted
team path (`NSYNTH_LLM_CACHE_PATH=/team/shared/cache.tsv`).
