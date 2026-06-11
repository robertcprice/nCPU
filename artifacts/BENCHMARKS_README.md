# nsynth Benchmarks

This directory is the **empirical record** of the nsynth program
synthesizer's cross-run learning behaviour. Every file here is a
committable dataset produced by an installed binary or shell script —
nothing is hand-curated.

The question each file answers:

| File | Question |
|------|----------|
| [`SELF_IMPROVEMENT_RATE.md`](./SELF_IMPROVEMENT_RATE.md) | Does the solver get faster week-over-week? |
| [`WEIGHT_TRAJECTORY.md`](./WEIGHT_TRAJECTORY.md) | Which features has the online ranker learned matter? |
| [`diversity_pareto.md`](./diversity_pareto.md) | What's the Pareto-optimal `TEACHER_TOPK` for this cache? |
| [`SOLVER_PRIORITIZATION.md`](./SOLVER_PRIORITIZATION.md) | Which program families does the solver keep failing on? |
| [`DEMO_LEARNING_LOOP.md`](./DEMO_LEARNING_LOOP.md) | End-to-end "watch the system learn" walk-through. |

## How each file is produced

```
SELF_IMPROVEMENT_RATE.md   ← tools/measure_self_improvement.sh   (weekly cron)
WEIGHT_TRAJECTORY.md       ← tools/weight_trajectory.py          (any time; reads .tsv)
diversity_pareto.md        ← tools/diversity_pareto.sh           (after ranker change)
SOLVER_PRIORITIZATION.md   ← tools/prioritize.sh                 (nightly cron)
DEMO_LEARNING_LOOP.md      ← tools/demo_learning_loop.sh         (on demand)
```

## What's also in this directory

- `meta_weights_history.tsv` — raw append-only weight snapshots
  (one row per `weights_snapshot` invocation). The source for
  `WEIGHT_TRAJECTORY.md`.
- `diversity_pareto.csv` — raw Pareto data (the source for
  `diversity_pareto.md`).
- `transfer_curve/curve_*.jsonl` — per-round solve logs from
  `transfer_curve`, used by `curve_analysis` and the weekly
  self-improvement measurement.
- `transfer_failures.jsonl` — near-miss rows captured when
  `NSYNTH_LOG_TEACHER_FAILURES=1`. Source for
  `SOLVER_PRIORITIZATION.md`.
- `cluster_history/*.jsonl` — timestamped cache-cluster snapshots
  (written by `cluster_drift --snapshot`). `cluster_drift --diff`
  over two of these reports which teachers moved cluster between runs.

## Why this dataset is unusual

No other program-synthesis project in the public literature tracks a
rolling trajectory of its own learned ranker weights, its Pareto-optimal
hyperparameters, and its per-family prioritization list — as committed
repo artifacts updated by CI. The novelty isn't the synthesis technique;
it's the empirical accountability.

Three observations worth isolating:

1. **Self-improvement is measured, not assumed.** Every weekly row in
   `SELF_IMPROVEMENT_RATE.md` is an actual sweep — `curve_analysis`
   computes the round-over-round ratio and appends the number.
2. **Hyperparameter choices have Pareto justification.** The default
   `TEACHER_TOPK` is whatever `diversity_pareto.sh` measures as Pareto-
   dominant. When the cache changes, re-run the sweep and bump the
   default (or let `autotune_topk.sh` do it automatically).
3. **Negative results survive.** When a change doesn't help (see
   `bootstrap_train`'s measured null result on the 94-entry cache), the
   honest finding goes into the commit message and the artifacts stay.

## Reading order for a new contributor

1. Start with `DEMO_LEARNING_LOOP.md` — 60-second end-to-end orientation.
2. Check `SELF_IMPROVEMENT_RATE.md` — is the trajectory moving the right
   direction?
3. If you're changing the ranker, look at `diversity_pareto.md` before
   and after. Same row pattern means no Pareto impact; a shift means the
   default `TEACHER_TOPK` may want updating.
4. If you're adding a solver technique, check `SOLVER_PRIORITIZATION.md`
   for which families are worth your attention.
