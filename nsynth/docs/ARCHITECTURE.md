# nsynth — Architecture Map

This is the index of architecture documents for the `nsynth` Rust
synthesis crate. Read these to understand what's in the tree and how the
parts fit together.

## Solver teachers (search pipeline)

The solver iterates `SEARCH_CANDIDATES` (`nsynth/src/solver/search.rs`),
each of which is a *teacher* that looks at a `Problem` and either
returns a verified `SolveResult` or `None`. The pipeline prefers
teachers that pre-empt the slow native-gradient path.

- [STRUCTURAL_ARRAY_TEACHERS.md](STRUCTURAL_ARRAY_TEACHERS.md) — the
  `ArrayFeature` taxonomy and the three new teachers
  (`search_array_sequence`, `search_array_feature_dnf`,
  `search_string_subsequence_class`). Includes the preemption invariant
  regression test (`new_teacher_preemption_cases.rs`).

- [REGISTRY_V3.md](REGISTRY_V3.md) — `ProgramV3`, `execute_program_v3`,
  and the library format 3 schema. Verified-skill registry for the
  NPCoT WASM kernel.

- [RULE_MEMORY.md](RULE_MEMORY.md) — the lifelong rule-memory
  experiment, the monotonic-pillar guarantee, and the abstaining
  micro-rule pattern that turns exception tables into verified code.

- [AUTORESEARCH_SOURCES.md](AUTORESEARCH_SOURCES.md) — the
  `ncpu.autoresearch.sources.*` adapters that mine rejected
  registry submissions and synthesis-API refusals into the driver
  queue.

## Solver surface

- `nsynth/src/solver/` — the search pipeline (`search.rs`), 6
  family modules (`scalar_search`, `array_compose`, `text_families`,
  `search_affine`, `search_bitwise`, `search_float`, `search_codegen`,
  `search_families`, `search_numeric_families`, `search_scalar_families`),
  the post-enumerative router (`post_enumerative.rs`), the dispatch
  table (`solver.rs`), and the test corpus (`tests.rs` + the cases
  submodules under `tests/search_cases/`).
- `nsynth/src/synthesis/` — gradient + universal-array synthesis
  fallback paths, used only when no search teacher solves the problem.
- `nsynth/src/morph_transduce.rs` — generative string-to-string
  morphology (the morpheme-transduce teacher used for English
  pluralization and 3sg inflection).
- `nsynth/src/runtime.rs` — the Mog VM that executes the
  verified programs.
- `nsynth/src/mog_transpile.rs` — Mog → Python / Rust / TypeScript
  transpilers.

## Commands

- `nsynth_codegen` — CLI for emitting Mog from problems.
- `nsynth_serve` — the synthesis-API server (see
  `tests/synthesis_api/test_server.py` for the integration tests).
- `transfer_curve`, `top_teachers`, `warm_start_bench`, `mog_synth`,
  `mog_to_python`, `mog_to_rust`, `mog_to_typescript` — additional
  utilities.

## Reproducing

```bash
cd nsynth
cargo build --release
cargo test --release --lib --no-fail-fast -- \
  solver::tests::exact_cases \
  solver::tests::routing_cases::new_teacher_preemption_cases

# Lifelong rule-memory experiment
python3 scripts/rule_memory_experiment.py --stream 1200 --chunk 50

# Verified-skill registry
pytest -q tests/registry/

# Autoresearch sources
pytest -q tests/autoresearch/test_cli_registry.py \
        tests/autoresearch/test_registry_source.py \
        tests/autoresearch/test_synthesis_api_source.py
```

See `nsynth/scripts/LINGUAGENESIS_BRIDGE.md` for the cross-project
nsynth ↔ LinguaGenesis integration reference.
