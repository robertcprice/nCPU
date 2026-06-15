# Lifelong Rule Memory — Monotonic Pillar

`nsynth/scripts/rule_memory_experiment.py` is the headline experiment for
the "rule-compressed memory = infinite context" claim. It now satisfies
a **monotonic correctness guarantee** that earlier versions did not.

## The claim

A lifelong learning system that has seen `n` stream items should:

1. **Be correct on every item it has ever seen.** A re-synthesized rule
   must not silently change the answer for an earlier-correct word just
   because a new miss needs a different rule.
2. **Converge to compact form.** Repeated exception families should be
   promoted from instance memory into verified executable code so the
   total memory footprint grows sublinearly.
3. **Self-improve.** An error, once corrected, is never made again
   (Hamilton mistake memory).

## What changed in this rewrite

### 1. Train on every observation, not just misses

Earlier: `pending = misses; regular_pool.extend(pending)`. This made the
rule pillar forget earlier-correct answers when a new miss forced a
re-synthesis that happened to also fit the new constraints.

Now: `integrate_regular_pairs([(w, oracle[w]) for w in chunk], ...)`
attempts to fold the **entire chunk** into the regular pool. If a bulk
integration is infeasible (some chunk item is irreducible), the
integration is retried with the *suspect* words (those the prior rule
missed) excluded; the fallback path admits new pairs one at a time so
that the rule can only *grow*, never *forget*.

The result: `cov_seen_rule` is now a flat **1.0** across every checkpoint
(was dipping to 0.80–0.99 in the old version).

### 2. Abstaining exception micro-rules

Large exception tables are now partitioned into verified
**abstaining micro-rules**. Each micro-rule:

- Fires on a suffix or exact guard.
- Returns the corrected output *only if* it is correct for every regular
  word and every exception it touches.
- **Returns `""`** (empty string) on no match, so the hierarchical
  wrapper can fall through to the next micro-rule and finally the main
  rule.

This is conservative by design. The micro-rule never claims a word it
hasn't been told about. The progressive fallback chain is:

```
fn pluralize(s: string) -> string {
    res_0 = pluralize_micro_0(s); if res_0 != "" { return res_0; }
    res_1 = pluralize_micro_1(s); if res_1 != "" { return res_1; }
    ... (all micro rules) ...
    return main_synthesized_rule(s);
}
```

The micro-rule emitter (`_emit_guarded_micro_rule`) groups exceptions by
edit action (strip N chars, append S) and by suffix length, and emits
the **first** candidate whose verification pass succeeds. This makes the
`Hamilton` table monotonic: each committed micro-rule strictly reduces
the exception count, never increases it.

### 3. Partition threshold

While `len(exceptions) >= 8`, the experiment calls
`synth_exception_micro_rule` to mine a new micro-rule. If no safe
micro-rule exists, the exception set is left as-is (the instance memory
still has them).

## What the new CSV shows

| n | seen_rule_coverage (old → new) | unseen_real_coverage (old → new) | nonce_coverage (old → new) |
|---|---|---|---|
| 200 | 0.99 → **1.0** | 0.99 → 0.99 | 0.99 → 0.99 |
| 600 | 0.99 → **1.0** | 0.96 → **0.997** | 0.90 → 0.958 |
| 1200 | 0.98 → **1.0** | 0.96 → **0.997** | 0.78 → **1.0** |

Resynth count roughly doubles at the end (20 → 30) because the rule now
has to satisfy *every* observation, not just misses. Rule bytes grow
moderately (3196 → 793 + 3 micro-rules + 3 exception tail), but
`rule + micro_rules` total stays below the instance-memory byte count
across the whole stream.

## Reproducibility

```bash
cd nsynth && python3 scripts/rule_memory_experiment.py --stream 1200 --chunk 50
```

Output: per-n metrics + final summary printed to stdout; full curve
written to `nsynth/scripts/rule_memory_results.csv` (committed).

## Files

- `nsynth/scripts/rule_memory_experiment.py` — the experiment.
- `nsynth/scripts/rule_memory_results.csv` — committed curve.
- `nsynth/scripts/lifelong_library_experiment.py` — related: the -z
  verb lexicon was previously missing from the regular-plural training
  set, which biased the rule pillar toward "ends-with-h" overgeneralization.
  The `bucket()` function now splits sibilants into `es_s`, `es_x`, `es_z`,
  `es_ch`, `es_sh` so the synthesized rule can't collapse them.
- `nsynth/scripts/lifelong_library.json` — the recovered library, with
  `ends_with("z") → +es` added (was missing).
