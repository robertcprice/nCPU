# artifacts/

Committed benchmark results and analysis reports — the reproducible evidence the
paper and regression tests cite. These are generated outputs kept under version
control so claims stay anchored.

Key entries:
- `nsynth_coverage.json`, `nsynth_per_problem_coverage.jsonl`,
  `nsynth_per_problem_summary.json` — the nSynth solver-portfolio coverage
  (95/95; gradient 60 / enumerative 25 / search 10). The `.jsonl` is regenerated
  from `nsynth_coverage.json` via `scripts/regen_nsynth_per_problem_coverage.py`.
- `mog_synthesis_coverage.json` — Mog grammar-constrained synthesis (315/315).
- `BENCHMARK_RESULTS.md`, `BENCHMARKS_README.md` — human-readable result writeups.
- `archive/` — superseded result snapshots.

Consumed by `tests/test_nsynth_coverage.py`, `tests/test_paper_claims.py`,
`tests/test_artifact_integrity.py`.
