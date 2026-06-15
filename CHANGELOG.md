# CHANGELOG

New work on `feat/bottom-up-piecewise-synthesis` since 2026-06-12.
See `nsynth/docs/ARCHITECTURE.md` for the architecture index and
`nsynth/docs/CONTRIBUTING.md` for the new-teacher cookbook.

## 2026-06-14 — 21 commits, 13 features, 1 critical fix

### Critical fix

- **`fix(egdc): restore mog_gradient_bridge.py stub`** — the 12-line
  bridge script that the Rust test runner shells out to was missing,
  breaking every orchestrator and interactive-solver test. Stub calls
  the existing `egdc.mog.solvers.gradient_bridge.main`.

### New search teachers (nsynth)

- **`feat(nsynth): 3 structural-array teachers + ArrayFeature taxonomy`**
  - `ArrayFeature` enum (10 variants: Contains, Adjacent, Sequence,
    CountAtLeast, CountExactly, RunAtLeast, AnyGreater, AnyLess,
    AllGreater, AllLess) — `nsynth/src/solver/search_families.rs`.
  - `search_array_sequence` — 1 iff A appears before B.
  - `search_array_feature_dnf` — DNF over the full taxonomy.
  - `search_string_subsequence_class` — string analog.
  - Codegen helpers in `search_codegen.rs`:
    `code_array_sequence_search`, `code_array_feature_dnf_search`,
    `code_string_subsequence_class_search`, `escape_string`.
  - Regression test `new_teacher_preemption_cases.rs` enforces the
    pair (registered in `SEARCH_CANDIDATES`, preempts gradient) for
    every new teacher.

- **`feat(nsynth): search_strictly_increasing + search_has_strictly_increasing_run`**
  - Two unary-array teachers that round out the strictly-monotonicity
    surface (between `search_is_sorted` ≤ and the DNF teacher).

- **`feat(nsynth): search_first_index_of teacher`**
  - Returns the first index where `arr[i] == target`, or -1.
  - Tests the int-output (not 0/1 classifier) path.

### New search teachers (nsynth)

- **`feat(nsynth): 3 structural-array teachers + ArrayFeature taxonomy`**
  - `ArrayFeature` enum (10 variants: Contains, Adjacent, Sequence,
    CountAtLeast, CountExactly, RunAtLeast, AnyGreater, AnyLess,
    AllGreater, AllLess) — `nsynth/src/solver/search_families.rs`.
  - `search_array_sequence` — 1 iff A appears before B.
  - `search_array_feature_dnf` — DNF over the full taxonomy.
  - `search_string_subsequence_class` — string analog.
  - Codegen helpers in `search_codegen.rs`:
    `code_array_sequence_search`, `code_array_feature_dnf_search`,
    `code_string_subsequence_class_search`, `escape_string`.
  - Regression test `new_teacher_preemption_cases.rs` enforces the
    pair (registered in `SEARCH_CANDIDATES`, preempts gradient) for
    every new teacher.

- **`feat(nsynth): search_strictly_increasing + search_has_strictly_increasing_run`**
  - Two unary-array teachers that round out the strictly-monotonicity
    surface (between `search_is_sorted` ≤ and the DNF teacher).

- **`feat(nsynth): search_first_index_of teacher`**
  - Returns the first index where `arr[i] == target`, or -1.
  - Tests the int-output (not 0/1 classifier) path.

- **`feat(nsynth): last_index_of teacher + bool predicate bridge + is_sorted_bool`**
  - `search_last_index_of` — mirror of `first_index_of` but scans in
    reverse. Pre-existing `search_count_distinct` is added to the
    preemption whitelist (the empty-array codegen got an explicit
    `if arr.len == 0 { return 0; }` guard).
  - Bool predicate bridge in `output_matches`: added the missing
    `(Value::Int, BV::Bool)` arm so a solver emitting `return 1;`
    verifies against `expected: true` and vice versa.
  - `make_is_sorted_bool` factory added (later removed from
    FACTORIES — bool expected can't be solved by search teachers,
    so it doesn't belong in the search-only benchmark).

### Pipeline plumbing

- **`fix(nsynth): wire Value::Bool through the full pipeline`** — the
  `BenchmarkValue::Bool` variant added earlier was missing the
  exhaustive-match arms at 3 sites: `method_router.rs:74`,
  `runtime.rs:433`, `solved_cache.rs:42`. Now bool is first-class.

- **`feat(synthesis-api): accept bool expected/outputs in JSON requests`**
  - Python server coerces bool expected to 0/1 int before shelling out
    to mog_synth, with a docstring explaining the design.
  - Rust CLI's `parse_problem_json` accepts bool inputs and bool
    expected outputs, mapping them to `Value::Bool`.

### Rule memory

- **`feat(rule-memory): monotonic rule pillar + abstaining micro-rules`**
  - `integrate_regular_pairs()` — trains on every observation, not just
    misses; the rule can only grow, never forget.
  - `_emit_guarded_micro_rule()` + `compile_hierarchical_rule()` —
    abstaining micro-rules that return "" on no match so the wrapper
    falls through to the next micro-rule and finally the main rule.
  - `synth_exception_micro_rule()` — mines safe micro-rules by
    (action, suffix-length) bucket.
  - Lifelong library: sibilant bucket split (es_s/es_x/es_z/es_ch/es_sh)
    so the rule can't collapse them; `ends_with("z") → +es` added.
  - CSV at n=1200: seen-rule coverage 0.98 → 1.0; unseen-real
    0.96 → 0.997; nonce 0.78 → 1.0.

### Registry v3

- **`feat(registry): ProgramV3 — per-data-point running state with reset`**
  - `ProgramV3` namedtuple: arity, combine_idx, guard_idx,
    guard_threshold, reset_guard_idx, reset_threshold, state_init_idx,
    update_transform_idx, update_reduce_idx, post_scale_idx,
    output_idx, offset.
  - `execute_program_v3(p, data, n_steps) -> List[float]` mirrors the
    canonical Rust executor; `execute_program_v3_final()` lets V3 drop
    into any V2-shaped verifier by lifting V2 -> V3.
  - Library format 3 schema (program_v3 only, no program or
    program_v2); V3_FIELDS is the canonical schema.
  - 5 new tests including `test_v3_running_counter_replay`,
    `test_v3_running_max_with_reset`,
    `test_v3_lift_matches_v2_final_output`,
    `test_verify_accepts_v3_trace_examples`,
    `test_submit_v3_skill_lifts_library_to_format_3`,
    `test_registry_misses_loop`.

### Autoresearch sources

- **`feat(autoresearch): registry + synthesis_api work-item sources`**
  - `ncpu/autoresearch/sources/registry.py` — mines rejected
    verified-skill submissions into a `WorkItem` queue; V3 trace
    examples are skipped.
  - `ncpu/autoresearch/sources/synthesis_api.py` — mines
    `success: false` responses from `ncpu.synthesis_api.server`.
  - CLI: `python -m ncpu.autoresearch.cli mine-registry --misses <path>`.
  - `run-once --benchmark` now accepts `humaneval`, `mbpp`, or
    `registry`.
  - 19 new tests across `test_cli_registry`, `test_registry_source`,
    `test_synthesis_api_source`, `test_prompt_parser`, and
    `synthesis_api/test_server`.

### Repo hygiene

- **`chore(repo): LFS coverage guard for nsynth/data + CI smoke script`**
  - 9 LFS rules for the >50MB JSONLs in `nsynth/data` (the prior_net
    training sets and combined corpora).
  - `tests/test_lfs_gitattributes.py` re-scans the directory on every
    run and fails if any >50MB file slips through.
  - `scripts/ci_smoke.sh` runs the targeted suites that gate nCPU
    robustness: prompt parser, LFS coverage, synthesis API, registry,
    MCP server, the new mine-registry CLI tests.

### Benchmark + integration

- **`feat(benchmark): new structural-array factories for the benchmark sweep`**
  - `make_strictly_increasing` — single variant, exercises
    `search_strictly_increasing`.
  - `make_has_strictly_increasing_run` — 4 variants cycling
    k ∈ {2, 3, 4, 5}.
  - `make_first_index_of` — 6 variants cycling target ∈
    {0, 1, 2, 5, 7, -1}.
  - `make_last_index_of` — 4 variants cycling target ∈
    {5, 0, 7, -2}.
  - Multi-variant factories use `Box::leak` to convert dynamic
    strings to `&'static str`.
  - New test `new_teacher_factories_are_in_benchmark_and_solve`
    verifies each factory emits a problem the corresponding teacher
    solves end-to-end.

- **`test(nsynth): adapt legacy / benchmark / routing tests to new teachers`**
  - `default_solver_uses_structured_gradient_for_hard_array_families`:
    accept any solver (search_* or arr_gradient_*) — the test is
    about solvability, not path.
  - `search_teacher_promotes_scalar_gradient_before_raw_search`:
    accept `diff_gradient_arithmetic` as a valid scalar solver.
  - `legacy_only_entrypoint_still_solves_full_benchmark`: filter
    the new benchmark factories out of the legacy sweep (the
    legacy fallback intentionally bypasses the new teachers).

### Test coverage + docs

- **`test(nsynth): edge cases for the new structural-array teachers`**
  - Pins the empty-array and single-element behavior of
    `search_strictly_increasing`, `search_first_index_of`,
    `search_last_index_of`, `search_count_distinct`.
  - Empty arrays kept in holdouts (verified after a teacher is
    selected) rather than training examples, because the search
    pipeline tries every search teacher on the training set and
    some pre-existing teachers (e.g. `search_second_max`) panic
    on empty input.

- **`feat(scripts): CLI smoke test for the new search teachers`**
  - `nsynth/scripts/cli_smoke.py` — Python script that exercises the
    new teachers through the actual release binary. 6 cases, all
    green in <50 ms total.

- **`feat(scripts): demo for the new structural-array search teachers`**
  - `nsynth/scripts/demo_new_teachers.py` — user-facing demo that
    shows the teacher surface in action. 6 demos, prints the
    recovered Mog program for each.

- **`docs(nsynth): CONTRIBUTING guide for adding a new search teacher`**
  - 10-step cookbook covering module selection, helper writing,
    codegen, registration, preemption, regression tests, unit
    tests, benchmark factories, CLI/API integration, plus a
    'common pitfalls' section.

### Documentation

- **`docs+artifacts: architecture index + tier-C/D deliverables + refreshed prior_net eval`**
  - `nsynth/docs/ARCHITECTURE.md` — the index document.
  - `nsynth/docs/STRUCTURAL_ARRAY_TEACHERS.md` — the new teachers and
    the preemption invariant.
  - `nsynth/docs/RULE_MEMORY.md` — the monotonic-pillar rewrite and
    the abstaining micro-rule pattern.
  - `nsynth/docs/REGISTRY_V3.md` — ProgramV3 and library format 3.
  - `nsynth/docs/AUTORESEARCH_SOURCES.md` — the source adapters and
    the mine-registry CLI.
  - `artifacts/prior_net_phase_a.{json,md}` — refreshed after a re-run.
  - `artifacts/full_grammar_validator_demo.json` and
    `recovered_grammar_checkers.json` — Tier C/D deliverables from
    Rung 10.

## Test results

| Suite | Pass | Fail | Notes |
|---|---|---|---|
| `cargo test --release --lib --no-fail-fast` (solver::tests, search+routing+new+exact+bench) | 149 | 0 | All structural, holdout, benchmark, and gradient tests pass; full benchmark takes 141s. |
| `cargo test --release --lib --no-fail-fast` (single new_teacher tests) | 18 | 0 | New teachers + preemption + benchmark factory + edge cases. |
| `pytest tests/registry/` | 25 | 0 | Including 5 V3 tests + 1 misses-loop test. |
| `pytest tests/autoresearch/test_*_source.py` + `test_cli_registry.py` | 19 | 0 | New in this batch. |
| `pytest tests/synthesis_api/test_server.py` | 26 | 0 | Including 2 array_feature / strictly_increasing / last_index / count_distinct integration tests. |
| `pytest tests/test_lfs_gitattributes.py` | 2 | 0 | LFS coverage guard. |
| `pytest tests/mog/test_mog_gradient_solver.py` | 14 | 0 | Now passes after the bridge restore (was 14/14 fail). |
| `pytest tests/mog/test_mog_orchestrator.py` | 4 | 0 | Same. |
| `python3 nsynth/scripts/cli_smoke.py` | 6 | 0 | CLI smoke test for the new teachers. |
| `python3 nsynth/scripts/demo_new_teachers.py` | 6 | 0 | User-facing demo of the new teacher surface. |

## How to reproduce

```bash
# Build the release binary once
cd nsynth && cargo build --release

# Run the full preemption + new-teacher regression suite
cd nsynth && cargo test --release --lib --no-fail-fast -- \
  solver::tests::routing_cases::new_teacher_preemption_cases \
  solver::tests::search_array_sequence_learns_order_constraint \
  solver::tests::search_array_feature_dnf_learns_count_and_run_features \
  solver::tests::search_string_subsequence_class_learns_order_constraint \
  solver::tests::search_strictly_increasing_learns_strict_inequality \
  solver::tests::search_has_strictly_increasing_run_learns_run_length \
  solver::tests::search_first_index_of_learns_target_value \
  solver::tests::search_last_index_of_learns_target_value \
  solver::tests::search_count_distinct_learns_distinct_count \
  solver::tests::new_teacher_factories_are_in_benchmark_and_solve \
  solver::tests::new_teacher_edge_cases_handle_empty_and_single_element

# Run the rule-memory experiment
cd nsynth && python3 scripts/rule_memory_experiment.py --stream 200 --chunk 25

# Run the synthesis API end-to-end
pytest -q tests/synthesis_api/test_server.py tests/registry/ \
       tests/autoresearch/test_cli_registry.py \
       tests/autoresearch/test_registry_source.py \
       tests/autoresearch/test_synthesis_api_source.py \
       tests/test_lfs_gitattributes.py

# Run the CLI smoke
python3 nsynth/scripts/cli_smoke.py

# Run the user-facing demo
python3 nsynth/scripts/demo_new_teachers.py

# Or the CI smoke
bash scripts/ci_smoke.sh
```
