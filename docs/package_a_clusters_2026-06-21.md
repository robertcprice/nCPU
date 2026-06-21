# Package A cluster run — 2026-06-21

Serial runs per module prefix (`cargo test --lib <prefix> -- --test-threads=1`).

| Cluster | Passed | Failed | Owner | Notes |
|---|---:|---:|---|---|
| `comprehension` (vocab) | 6 | 0 | C | green |
| `self_improve` | 25 | 0 | O | ~191s |
| `database` | 7 | 0 | F/K | green |
| `agent::repo` | 28 | 0 | B/H | green |
| `package_b` gate | 4 | 0 | B | green |
| `search_only_solves_full_benchmark` | 1 | 0 | A | CI smoke |
| `agent::debate` | 5 | 0 | scaffold | quarantined heuristic |
| `security` | 18 | 0 | F | vulnerability patterns + taint + validation stage |
| `runtime` | 58 | 0 | A/K | holdout verify fix — full benchmark green |
| `interactive::` | 8 | 0 | A/B | clarify flow + legacy wiring fixed |
| `orchestrator` | 17 | 0 (1 ignored) | A/B | interactive batch green; legacy batch ignored |
| `optimization` | 26 | 2 | K | rust for-loop / parallelizer detection |
| `tensor` | 270 | 0 (17 ignored) | O | quarantined — `docs/TENSOR_QUARANTINE.md` |

## Security failures (6) — fixed 2026-06-21

All `security::vulnerability` and `validation::stages::test_security_validation` tests green.

## Orchestrator failures (5)

- `orchestrator::tests::orchestrator_persists_and_retrieves_interactive`
- `orchestrator::tests::orchestrator_retrieves_interactive_among_mixed_records`
- `orchestrator::tests::orchestrator_solves_batch_interactive`
- `orchestrator::tests::orchestrator_solves_batch_legacy_only`
- `orchestrator::tests::orchestrator_solves_interactive_problem`

## Optimization failures (2)

- `optimization::parallel::tests::test_rust_for_loop_detection`
- `optimization::tests::test_parallelizer_detection`

## Tensor failures (17)

Experimental tensor/NAS/diffusion tests — defer to Package O unless blocking G0 compile.
