# Package A Baseline — 2026-06-21

**Snapshot date:** 2026-06-21
**Package/Gate:** A / G0
**State:** in progress (G0 largely closed); compile-green; cluster runs green; tensor quarantined
**Authority:** root `MASTER_ROADMAP.md`
**Cluster detail:** `docs/package_a_clusters_2026-06-21.md`

## CI gates (green)

| Command | Result |
|---|---|
| `cargo test 'agent::repo' --lib -- --test-threads=1` | **29 passed**, 0 ignored |
| `cargo test package_b --lib` | **4 passed** (Package B gate) |
| `cargo test search_only_solves_full_benchmark --lib` | **1 passed** (140 tasks; per-problem holdout verify after `solve_problem_search_only`) |
| `cargo test comprehension --lib` (vocab) | **6 passed** |
| `cargo test self_improve --lib` | **25 passed** (~191s) |
| `cargo test database --lib` | **7 passed** |

## Failure clusters (serial prefix runs, 2026-06-21)

Detail: `../docs/package_a_clusters_2026-06-21.md`

| Cluster | Failed | Owner | Action |
|---|---:|---|---|
| `comprehension` (vocab) | 0 | C | green |
| `self_improve` | 0 | O | green (~191s) |
| `database` | 0 | F/K | green |
| `agent::repo` | 0 | B/H | green (29 tests) |
| `package_b` gate | 0 | B | green |
| `search_only_solves_full_benchmark` | 0 | A | green (140 + holdout verify) |
| `security` | 0 | F | vulnerability + taint + validation stage fixed |
| `runtime` | 0 | A/K | green (holdout verify fix) |
| `interactive::` | 0 | A/B | green |
| `orchestrator` | 0 (1 ignored) | A/B | interactive batch green |
| `optimization` | 2 | K | rust for-loop / parallelizer detection |
| `tensor` | 0 (17 ignored) | O | quarantined — `docs/TENSOR_QUARANTINE.md` |
| `cargo fmt --check` | green | A | formatting baseline |

**Clustered failures remaining:** 2 (`optimization`); tensor experimental stack excluded via `#[ignore]`.

Full serial suite with complete failure list: use per-cluster commands above; `--no-fail-fast` unavailable in this toolchain.

## P0 stub status (2026-06-21)

| Item | Status |
|---|---|
| `agent/orchestrator.rs` simulate paths | **no `simulate` in `agent/`** — tests use real search solver |
| `agent/planning.rs` simulate_execution | **not found** |
| `nl/mod.rs` NotImplemented | **quarantined** — production NL delegates to `linguigenesis_bridge` |

## Package A exit checklist

- [x] no unmerged index entries
- [x] debug/release library checks pass
- [x] repo-agent focused tests pass (**29**)
- [x] Package B gate closed (`docs/PACKAGE_B_GATE.md`)
- [x] search-only benchmark smoke green
- [x] failure clusters re-run with owners (`package_a_clusters_2026-06-21.md`)
- [x] tensor experimental cluster quarantined (`docs/TENSOR_QUARANTINE.md`, 17 `#[ignore]`)
- [ ] full library suite single-command summary (toolchain lacks `--no-fail-fast`)
- [x] formatting baseline green (`cargo fmt --check`)
- [ ] P0 stubs outside `nl/` closed or quarantined with CI guard

---

# Package A Baseline — 2026-06-20 (historical)

**Starting revision:** `45accfc40e96573575b074d0705ea12204aac612`
**Package/Gate:** A / G0
**State:** in progress; compile-green, test-red
**Authority:** root `MASTER_ROADMAP.md`

## What This Baseline Proves

- The merged library module graph is syntactically coherent.
- Legacy single-function problems conform to the new multi-function-capable `Problem` shape.
- Debug and release libraries compile.
- The repo-agent scaffold's focused unit tests pass.
- The previous search-only stack overflow is removed and all 140 search-only benchmark tasks solve. This was reverified after public solver/synthesis entrypoints began stripping holdouts and reference implementations, reference fallback routes were disabled, and the benchmark scorer's Python warmstart fallback was removed.
- `search_output_is_invariant_to_evaluation_oracles` poisons both holdout outputs and reference code, proves synthesis returns byte-identical code/method, then proves the independent evaluator detects the poisoned oracle.
- Planner and collaborative-orchestrator fake success paths are removed: tasks require a supplied real result, integrated plans mark unexecuted nodes skipped, collaborative proposals come from the Rust search solver, and reviews run runtime/call-graph/forbidden-construct checks.
- `agent/debate.rs` remains quarantined as heuristic-only scaffold; its passing unit tests do not count as verified code-hardening capability.
- The complete test and formatting baselines are not green, so Package A and G0 remain open.

## Commands and Results

| Command | Result |
|---|---|
| `git ls-files -u` | no unmerged entries |
| `cargo check --lib` | passed; 360 `mog_synth` warnings plus Linguigenesis warnings |
| `RUSTFLAGS='-Awarnings' cargo check --release --lib` | passed in 14.74s |
| `RUSTFLAGS='-Awarnings' cargo test 'agent::repo' --lib -- --test-threads=1` | 16 passed, 0 failed, 2,096 filtered |
| `cargo test preemptive_search_teacher_solves_slow_exact_search_cases --lib` | 1 passed; confirms ordinary parity no longer enters mutual recursion |
| `cargo test orchestrator::tests::orchestrator_solves_batch_search_only --lib -- --exact --test-threads=1` | 1 passed; all 140 tasks solved in 3.36s |
| `cargo test runtime::resource::tests --lib` | 8 passed, 0 failed |
| `cargo test solver::search_time_families::tests::test_gaussian_elimination_2x2 --lib` | 1 passed, 0 failed |
| `cargo fmt --check` | failed: 473 hunks across 34 files |
| `cargo test --lib -- --test-threads=1` | CI subset: `search_only_solves_full_benchmark` + agent NL tests; full portfolio benchmark `#[ignore]` nightly |

The full-suite count is a partial observation, not a final test summary. It must not be presented as 1,211/2,112 passing.

## Baseline Defects Fixed in This Package

1. `nsynth/src/lib.rs` contained an unresolved Git index conflict. The working copy was verified as the exact union of all 47 modules from both sides before resolution.
2. Four legacy `Problem` initializers omitted `functions`; single-function problems now use the existing `functions: vec![]` pattern.
3. The mutual even/odd teacher generated a helper and wrapper with the same `is_even` name, making the wrapper call itself forever.
4. Recursive search teachers claimed ordinary non-recursive problems; they now require recursion/explicit-stack opt-in.
5. Factorial, Fibonacci, Tribonacci, and integer polynomial recognition could panic on unrelated large inputs; their arithmetic is bounded or checked.
6. Resource tests registered descriptors 1 and 2 as owned resources, so `ResourceManager::drop` closed the test harness output. Tests now transfer descriptors they actually own.
7. The Gaussian test claimed an integer solution for a system whose real solution was fractional; its fixture now matches its documented solution.
8. The temporal teacher evaluated a different expression than it generated, passed codegen arguments in the wrong order, and stopped after verifier-rejected overfits; evaluator and generator now agree and continue safely.
9. Two temporal benchmark holdouts contradicted their reference programs (`count_rate` and `tick_every_2`); the oracle data now matches the declared behavior.

## Current Failure Clusters

The 61 failures observed before the stopped full-suite run cluster as follows. Re-run each cluster independently before editing it.

| Cluster | Representative failures | Likely package owner |
|---|---|---|
| comprehension/self-improvement | creature classification, entailment suite, extension/gate/store cases | A for baseline regressions; C/D for new semantics |
| database/HTTP generators | SQL rendering, GraphQL, WebAuthn, PWA, realtime, Vue, workers | K validation/backend graduation |
| interactive/orchestrator | clarification flow, interactive retrieval/batches, legacy batches | A for broken existing assertions; B/H for canonical runtime |
| runtime | full-benchmark execution and several program parsers | A/K |
| security scanner | command/SQL/XSS/taint tests | F/K |
| solver/routing | default route expectations and legacy/full benchmark tests | A/K |
| optimization/transpilation | Rust-loop detection and multi-language transpilation | K |

Do not fix all 61 opportunistically. First identify shared root causes with focused commands and close one cluster at a time.

## Stub and Simulation Inventory

The broad audit found 149 textual hits across 41 Rust files. Many are legitimate test mocks, semantic placeholders, or functions whose domain verb is “simulate”; the production blockers below require action.

| Severity | Location | Current behavior | Required action |
|---|---|---|---|
| P0 | `src/agent/orchestrator.rs` | simulated synthesis/review and generated `todo!()` body | remove from production path; replace with real typed actor/tool calls in B/H |
| P0 | `src/agent/planning.rs` | `simulate_execution` marks hinted tasks complete | replace with real executor result propagation in B/H |
| P0 | `src/nl/mod.rs`, `src/nl/dialogue.rs` | reachable `NLError::NotImplemented` | quarantine legacy path and make Linguigenesis bridge canonical in C/D |
| P0 | `src/main.rs` | target-language transpilation advertised as placeholder | implement or report target absent in K |
| P0 | `src/prob/mod.rs` | returns placeholder program source | mark capability absent or implement in K |
| P1 | tensor meta/probabilistic/NAS/loss modules | dummy gradients, scalar zeros, simplified objectives | isolate as experimental until real math and conformance tests in O |
| P1 | HTTP security/compression/workers/WASM modules | placeholder hashing, compression, worker outputs, partial WASM | remove capability claims or implement during F/K |
| P1 | DB ORM | placeholder query behavior | implement behind the real database boundary in F/K |
| P1 | understanding discourse/QA | skeleton handling for newer meanings | finish in C before native coding-intent claims |
| P2 | tests and semantic AST comments | test dummy values or intentionally unknown slots | retain when they are explicit test/representation terminology |

## Full-Suite Runtime Blocker

`solves_full_benchmark` runs the standard synthesis pipeline over all 140 factories inside the ordinary unit-test binary without a package-level deadline. In the observed serial run it remained active for several minutes after the suite had already run for roughly seven minutes. Package A must establish a bounded verification strategy:

1. preserve the exhaustive benchmark as a nightly or explicitly budgeted test;
2. add a deterministic fast smoke subset for ordinary CI;
3. enforce per-problem and whole-benchmark budgets in production code, not merely in the test runner;
4. report timeouts as failures with problem names;
5. never ignore or silently skip the exhaustive benchmark.

## Next Safe Actions

1. Add budget/reporting to benchmark execution so the full suite can finish with an honest summary.
2. Re-run the observed failing tests by cluster and identify root-cause reductions.
3. Resolve the 34-file formatting baseline intentionally after concurrent semantic changes stabilize.
4. Close every P0 stub before Package H can claim an MVP loop.

## Package A Exit Checklist

- [x] no unmerged index entries;
- [x] debug/release library checks pass;
- [x] repo-agent focused tests pass;
- [x] previous stack overflow has a passing regression;
- [ ] full library suite completes within a declared budget;
- [ ] remaining failures have stable reproducers and owners;
- [ ] formatting baseline is green or explicitly isolated by owned commits;
- [x] production stub inventory exists;
- [ ] all P0 production stubs are quarantined or replaced.
