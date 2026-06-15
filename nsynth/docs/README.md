# nsynth docs

Documentation for the nsynth Rust synthesis crate. Read in order
if you're new; jump straight to the per-feature doc for the
piece you're working on.

## Index

| Doc | Audience | What it covers |
|---|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Everyone | The 13-feature surface, the preemption invariant, the test pyramid, and how the pieces fit together. Read this first. |
| [STRUCTURAL_ARRAY_TEACHERS.md](STRUCTURAL_ARRAY_TEACHERS.md) | Solver contributors | The `ArrayFeature` enum, the 3 codegen helpers, the 6 new teachers (`search_array_sequence`, `search_array_feature_dnf`, `search_string_subsequence_class`, `search_strictly_increasing`, `search_has_strictly_increasing_run`, `search_first_index_of`, `search_last_index_of`, `search_is_anagram`, `search_longest_run`, `search_intersects`, `search_stateful_reducer`). |
| [RULE_MEMORY.md](RULE_MEMORY.md) | Pipeline contributors | The monotonic-pillar rule-memory rewrite, the abstaining micro-rule pattern, and the before/after coverage numbers. |
| [REGISTRY_V3.md](REGISTRY_V3.md) | Registry / NPCoT contributors | `ProgramV3` (per-data-point running state with reset), `execute_program_v3`, library format 3 schema. |
| [AUTORESEARCH_SOURCES.md](AUTORESEARCH_SOURCES.md) | Driver contributors | `ncpu.autoresearch.sources.*` adapters, the `mine-registry` CLI command, the round-trip test coverage. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | New teachers | The 10-step cookbook for adding a new search teacher: module selection, helper writing, codegen, registration, preemption, regression tests, unit tests, benchmark factories, CLI/API integration, plus a 'common pitfalls' section. |

## Test commands

```bash
# All teacher + preemption regression tests
cd nsynth && cargo test --release --lib --no-fail-fast -- \
  solver::tests::routing_cases::new_teacher_preemption_cases \
  solver::tests::search solver::tests::new_teacher

# CLI smoke test (no Python needed; uses the release binary directly)
python3 nsynth/scripts/cli_smoke.py

# User-facing demo of the new teachers
python3 nsynth/scripts/demo_new_teachers.py

# Tier-A benchmark — publishable results table
python3 nsynth/scripts/bench_tier_a.py
python3 nsynth/scripts/bench_tier_a.py --variants 4  # fuller
python3 nsynth/scripts/bench_tier_a.py --out results.json

# Go client example (requires the synthesis server running)
python3 ncpu/synthesis_api/server.py &
cd examples/go_client && go run main.go --run
```

## Per-test-suite notes

| Suite | What it tests | How to run |
|---|---|---|
| `solver::tests::routing_cases::new_teacher_preemption_cases` | 14 invariants: every new search teacher is both registered in `SEARCH_CANDIDATES` and listed in the preemption whitelist. | `cargo test --release --lib --no-fail-fast -- new_teacher_preemption` |
| `solver::tests::search` | Per-teacher unit tests: each teacher learns a hand-curated example set. | `cargo test --release --lib --no-fail-fast -- solver::tests::search` |
| `solver::tests::new_teacher` | New-teacher infrastructure: benchmark factory wiring, edge cases (empty + single-element arrays). | `cargo test --release --lib --no-fail-fast -- solver::tests::new_teacher` |
| `solver::tests::routing_cases::teacher_distillation_cases` | Routing: the gradient / search-pipeline interplay. | `cargo test --release --lib --no-fail-fast -- teacher_distillation` |
| `solver::tests::exact_cases` | Exact-benchmark integration: every factory problem is solved by the search pipeline. | `cargo test --release --lib --no-fail-fast -- exact_cases` (slow, ~140s) |

## See also

- [CHANGELOG.md](../../CHANGELOG.md) at the repo root for the per-commit summary of work shipped on this branch.
- [docs/stateful_synthesis_status.md](../../docs/stateful_synthesis_status.md) for the honest status of where stateful synthesis is — the Stage 1 (scalar, array) -> scalar reducer is now shipped, but the larger Stages 2-4 are scoped for future work.
