# Claude Model Tier Pareto — measured

30-problem humaneval_lite, LLM-only mode (no nsynth). Each row is a
real API-backed run captured 2026-04-18.

| model | pass@1 | runtime | mean ms/problem |
|-------|:------:|--------:|----------------:|
| claude-haiku-4-5-20251001 | **27/30 (90.0%)** | **20.0 s** | **667 ms** |
| claude-sonnet-4-6 | 27/30 (90.0%) | 40.7 s | 1357 ms |
| claude-opus-4-7 | 27/30 (90.0%) | 42.8 s | 1427 ms |

**Finding**: all three tiers produce identical pass@1 on this benchmark.
Haiku is **2× faster** than Sonnet / Opus at identical accuracy on
scalar integer synthesis problems.

The three misses (polynomial_2ax_plus_b, sum_abs_diffs, scaled_add) are
the same across all models — these are benchmark-design ambiguity
problems where multiple plausible formulas fit the 5–6 training
examples. Resolving them requires *more examples*, not a bigger model.

## Takeaway

For the common case (scalar functions with 5–10 examples), **Haiku is
the production default**. Reserve Sonnet/Opus for problems where:

- The benchmark set requires complex reasoning (multi-step algorithms,
  nested data structures, proof-like correctness)
- Temperature-driven variance matters (retrying with different sampling
  might help Sonnet/Opus more)
- Tokens/cost is less constrained than latency

## Reproduce

```bash
for model in claude-haiku-4-5-20251001 claude-sonnet-4-6 claude-opus-4-7; do
  ANTHROPIC_API_KEY=sk-... python3 tools/benchmarks/run_humaneval_llm_only.py \
    --model "$model" \
    --out "artifacts/humaneval_results_llm_only_${model}.md"
done
```
