# Reproducibility

Every numerical claim in `paper/` is backed by (1) a committed artifact under `artifacts/`, (2) a reproducibility harness under `benchmarks/`, and (3) a regression test under `tests/`. This document enumerates them so a reviewer can verify each claim in one command.

## One-Shot Verification

```bash
# Full publication-gate regression suite.
pytest tests/neural/test_gpu_only_engine.py \
       tests/mog/test_mog_diff_compiler.py \
       tests/test_nsynth_coverage.py -q
```

Expected: **92 passed, 1 skipped** in ~90 s. The skipped test is the live nSynth harness (`NCPU_NSYNTH_FULL_RUN=1`, ~15 min).

## Publishable Claims and Their Reproduction

### Claim 1: Differentiable Mog synthesis, 315/315 (100%) coverage

| Component | Path |
|---|---|
| Paper section | `paper/section_mog_synthesis.md` |
| Harness | `benchmarks/benchmark_mog_synthesis.py` |
| Artifact | `artifacts/mog_synthesis_coverage.json` |
| Regression | `tests/mog/test_mog_diff_compiler.py::test_search_solves_all_factories_multi_variant` |

```bash
# Reproduce the artifact (writes JSON with per-problem rows + summary):
python benchmarks/benchmark_mog_synthesis.py --variants 5 --seed 42 \
  --json /tmp/mog.json

# Verify agreement with committed artifact:
jq .summary.coverage /tmp/mog.json artifacts/mog_synthesis_coverage.json
# Both print 1.0; the wall time and method distribution match.
```

Expected: `315/315 (100.00%) in ~17 s`, 43 distinct solver methods in portfolio.

### Claim 2: nSynth (Rust) synthesizer, 95/95 coverage

| Component | Path |
|---|---|
| Paper section | `paper/section_solver_portfolio.md` |
| Rust CLI | `nsynth/target/release/mog_synth --per-problem-json` |
| Harness | `benchmarks/benchmark_nsynth.py` |
| Artifact | `artifacts/nsynth_coverage.json` (live) + `artifacts/nsynth_per_problem_coverage.jsonl` (persistent) + `artifacts/nsynth_per_problem_summary.json` |
| Regression | `tests/test_nsynth_coverage.py` (4 fast + 1 opt-in slow) |

```bash
# Build the Rust binary if needed:
(cd nsynth && cargo build --release)

# Artifact-based regression (fast, ~0.2 s):
pytest tests/test_nsynth_coverage.py -q

# Live reproduction (~15 min, writes fresh JSON):
python benchmarks/benchmark_nsynth.py --json /tmp/nsynth.json
```

Expected: `95/95 (100%) in ~900 s`, family breakdown `{gradient: 60, enumerative: 25, search: 10}`.

### Claim 3: Region-guard minimization, 50% pre-sync reduction

| Component | Path |
|---|---|
| Paper section | `paper/section_superblock_cache.md` |
| Implementation | `ncpu/neural/cpu/engines/gpu_only.py` (`_simulate_superblock_path`, `_plan_hotloop_memory_sync`) |
| Regressions | `tests/neural/test_gpu_only_engine.py` (5 read/write split tests) |
| Demo benchmark | `NCPU_GPU_BENCH_WORKLOAD=bytecopy NCPU_GPU_ONLY_AUTO_VALUE_THRESHOLD=0.0 python benchmarks/benchmark_gpu_only.py --json` |

Specific regressions:
- `test_simulator_pre_windows_excludes_store_only_ranges`
- `test_simulator_pre_windows_shadows_read_after_write`
- `test_simulator_pre_windows_keeps_pure_load_ranges`
- `test_superblock_trace_cache_ignores_store_only_byte_changes`
- `test_superblock_trace_cache_invalidates_on_load_byte_change`

Measured wins on `bytecopy` and `adjacent-bytecopy` workloads: `pre_sync_bytes` halved (4000 → 2000 and 8000 → 4000 respectively).

### Claim 4: Adaptive trace-level promotion, 17% per-lookup speedup

| Component | Path |
|---|---|
| Paper section | `paper/section_superblock_cache.md` §16.8 item 3 |
| Implementation | `ncpu/neural/cpu/engines/gpu_only.py` (`_superblock_trace_miss_counter`, threshold at `NCPU_GPU_ONLY_SUPERBLOCK_TRACE_PROMOTION`) |
| Regressions | `test_adaptive_trace_promotion_skips_trace_lookup_after_misses`, `test_adaptive_trace_promotion_disabled_at_zero_threshold` |
| Benchmark | `benchmarks/benchmark_superblock_promotion.py` |
| Artifact | `artifacts/superblock_promotion_benchmark.txt` |

```bash
python benchmarks/benchmark_superblock_promotion.py --iterations 2000
```

Expected: threshold=0 produces ~24.7 μs/iter; threshold=3 produces ~20.4 μs/iter (17% faster) with identical template hit counts.

### Claim 5: Solver portfolio taxonomy (methods-breakdown)

| Component | Path |
|---|---|
| Paper section | `paper/section_solver_portfolio.md` |
| Supporting artifacts | see Claims 1 and 2 |
| Regression | `tests/test_nsynth_coverage.py::test_artifact_solver_family_breakdown` (pins family-level counts at >=55 gradient, >=20 enumerative, 95 total) |

The portfolio argument is self-verifying: the per-method counts in both artifacts add up to the total coverage, so any third party can recompute family proportions directly from the committed JSON.

### Claim 6: NPCoT skill formats v2 + v3 (fail-closed versioning, 32 tests)

| Component | Path |
|---|---|
| Paper section | `paper/sections/section_synthesized_software.md` §20.1 |
| Harness | `cargo test` in `kernels/npcot_wasm/` |
| Artifact | `kernels/npcot_wasm/pkg/npcot_wasm_bg.wasm` (133,262 bytes, wasm-pack release build) |
| Regression | the full cargo suite: v1/v2 equivalence sweep (all 216 v1 programs), v2/v3 lift grid, v1/v2 fail-closed rejection of higher formats, lowest-format export demotion, mined-threshold guard + reset discovery, honest refusals, 5-language rendering |

```bash
(cd kernels/npcot_wasm && cargo test)
```

Expected: **32 passed, 0 failed** in <1 s (24 as of format v2, commit `b0622a5`; 32 as of format v3, commit `d4bb6a5`).

### Claim 7: nSynth persistent negative memory (failed search never repeats)

| Component | Path |
|---|---|
| Paper section | `paper/sections/section_synthesized_software.md` §20.3 |
| Implementation | `nsynth/src/rejected_cache.rs` (commit `1d2ed2f`), wired via RAII `RejectionRecorder` in `nsynth/src/synthesis/universal_array.rs` |
| Harness | `benchmarks/benchmark_negative_memory.py` — automates the cold/warm measurement: runs a novel out-of-space problem (`sum_above_first`, embedded in the script and artifact) twice with every nsynth bank isolated under a temp dir, verifies both runs end in honest refusals, and counts rejection hashes in the TSV after each run |
| Artifact | `artifacts/negative_memory_benchmark.json` (`cold_rejections` / `warm_new_rejections` / `converged`, plus the exact problem, wall times, and refusal reason) |
| Regression | `cargo test rejected_cache` in `nsynth/` (6 unit tests: persistence round-trip, cap eviction, per-fingerprint isolation) · `pytest tests/test_negative_memory_benchmark.py` (4 fast tests pin artifact shape + `converged == true`; the full ~5 min cold/warm rerun is opt-in via `NCPU_NEGMEM_FULL=1`) |

```bash
(cd nsynth && cargo test rejected_cache)
python3 -m pytest tests/test_negative_memory_benchmark.py        # fast: validates the committed artifact
python3 benchmarks/benchmark_negative_memory.py                  # regenerate the artifact (2 solver runs, ~2-3 min each)
NCPU_NEGMEM_FULL=1 python3 -m pytest tests/test_negative_memory_benchmark.py  # end-to-end rerun under pytest
```

Measured by the harness on 2026-06-11 (`sum_above_first`: sum of elements strictly greater than the first — outside the solver portfolio, refused after full cascade exhaustion): cold run persisted **1,483** rejections in 126.1 s; the warm rerun added **zero** in 60.6 s — the entire failed search deduplicated across runs. (Original manual measurement at commit `1d2ed2f`: 1,415 cold / 0 new warm.)

### Claim 8: Synthesized Pong — 22 rules, zero domain-sweep mismatches

| Component | Path |
|---|---|
| Paper section | `paper/sections/section_synthesized_software.md` §20.4 |
| Harness | `node tools/pong_synthesis/finalize_pong_rules.mjs` (domain-sweeps every rule, CEGIS retry on failure, verifies all 8 compositions, fails loudly on any mismatch) |
| Artifact | `tools/pong_synthesis/pong_rules_final.json` (per rule: method, Mog + TypeScript source, the exact training examples, swept case count) |
| Regression | byte-identical regeneration of the site's `synthesized.ts`; sweeps span 4 (exhaustive boolean tables) to 160,801 cases per rule, bounded to reachable game physics |

```bash
(cd nsynth && cargo build --release)   # provides mog_synth + --transpile
node tools/pong_synthesis/finalize_pong_rules.mjs
```

Expected: 22/22 rules verified (14 synthesized + 8 composed), zero sweep mismatches. With the solver memory banks populated, the rerun is near-instant.

### Claim 9: Delivery surfaces — synthesis API, verified-skill registry, MCP server

| Component | Path |
|---|---|
| Paper section | `paper/sections/section_synthesized_software.md` §20.5 |
| Harnesses | `pytest tests/synthesis_api/` (21 tests, live HTTP server) · `pytest tests/registry/` (19 tests, incl. `--verify-all` trust sweep + corruption detection) · `pytest tests/mcp_server/` (24 tests, real stdio subprocess; count as committed at `dd650ae` — the server is under active extension) |
| Implementations | `ncpu/synthesis_api/` (commit `9061d14`) · `tools/registry/` (commit `18ede07`) · `ncpu/mcp_server/` (commit `dd650ae`) |
| Regression | all three suites isolate the three nsynth memory banks under tmp paths; the MCP suite cross-checks the Python and Rust examples-fingerprints |

```bash
python3 -m pytest tests/synthesis_api/ tests/registry/ tests/mcp_server/ -q
```

Expected (at the commits above): 21 + 19 + 24 all green. Requires `nsynth/target/release/mog_synth` to be built.

## Optional Long-Running Claims (User-Green-Lit)

### Gradient-first nSynth survey (overnight)

The default nSynth pipeline runs enumerative before gradient. The true gradient-only coverage is bounded below by 60/95 but has not been fully measured because per-problem gradient search can consume minutes.

```bash
# Resumable, per-problem checkpoint driver (~3-6 h wall):
python benchmarks/benchmark_nsynth_gradient_first.py --budget 1200

# Check progress at any time:
python benchmarks/benchmark_nsynth_gradient_first.py --summarize
```

Writes to `artifacts/nsynth_gradient_first/<name>.json` one file per problem; survives interruption.

### Coprocessor real-world evaluation on vast.ai

```bash
# Dry run (prints plan + cost estimate without spending):
scripts/gpu/deploy_coprocessor_realworld_vastai.sh

# Actually launch (requires vastai CLI + API key; ~$11-$24 for 4B + 9B sweep):
scripts/gpu/deploy_coprocessor_realworld_vastai.sh --confirm
```

Pre-flight status: imports verified (`torch 2.10`, `transformers 5.5.4`, `ncpu.coprocessor.inject`), weights files structurally valid, HumanEval download URL 200 OK. Script targets full HumanEval-164 and GSM8K-500 on Qwen3.5-4B and -9B with and without the coprocessor.

## Reproducibility Contract

Every artifact in this document includes:

1. **Seed fixed and recorded.** Re-running the harness with the same seed produces byte-identical method counts.
2. **Method or solver identity per problem.** Aggregated coverage numbers without per-problem attribution are not accepted as evidence.
3. **Non-zero exit on regression.** Harnesses accept a `--min-coverage` floor and exit with code 2 below it, so CI wiring is trivial.
4. **Source-path traceable.** Every solver method name emitted by the harnesses corresponds to a discoverable function in the source tree.

A claim that cannot be expressed in this form should not appear in `paper/`.

## Known Limitations

- **Seed sensitivity.** All reported numbers are at seed 42. A systematic seed sweep is future work; it is not part of the current publication claim.
- **Portfolio co-evolution.** Both synthesis benchmarks (Mog and nSynth) were iteratively refined alongside their solvers. Out-of-distribution evaluation on a sealed third-party benchmark is future work.
- **Sample size on real-world benchmarks.** The existing instruct-sweep real-world results are at N=10 (too noisy to draw transfer conclusions). The vast.ai script addresses this with N=164 (HumanEval) + N=500 (GSM8K) but has not been run yet.
- **Confidence-aware gating gate value.** All instruct sweep runs report `mean_gate = 0.0` at evaluation time. This is expected under confidence-aware gating with `max_gate = 0.1`, but a direct ablation against hard-capped non-confidence gating is future work.

## Layout

```
paper/
  ncpu_paper.md                  # main paper (older sections)
  section_differentiable_programs.md  # §14: register-level differentiable engine
  section_mog_synthesis.md            # §15: grammar-constrained synthesis (315/315)
  section_superblock_cache.md         # §16: region-guard minimization + promotion
  section_solver_portfolio.md         # §17: portfolio taxonomy across engines

benchmarks/
  benchmark_gpu_only.py                 # GPU engine throughput (IPS, sync bytes)
  benchmark_mog_synthesis.py            # Mog compiler coverage harness
  benchmark_nsynth.py                   # nSynth Rust harness (Python wrapper)
  benchmark_nsynth_gradient_first.py    # Resumable gradient-only survey
  benchmark_superblock_promotion.py     # Adaptive promotion microbenchmark
  benchmark_coprocessor_realworld.py    # HumanEval + GSM8K baseline/coprocessor eval

tests/
  neural/test_gpu_only_engine.py      # 53 tests: ALU dispatch, superblock cache, promotion
  mog/test_mog_diff_compiler.py       # 35 tests: grammar, beam, multi-variant 315/315
  test_nsynth_coverage.py             # 5 tests: artifact breakdown + live run (opt-in)

artifacts/
  mog_synthesis_coverage.json         # 315 rows + summary (seed 42, variants 5)
  nsynth_coverage.json                # live-run snapshot with family breakdown
  nsynth_per_problem_coverage.jsonl   # 95 rows (line-oriented)
  nsynth_per_problem_summary.json     # method counts + totals
  superblock_promotion_benchmark.txt  # Before/after measurements

scripts/gpu/
  deploy_coprocessor_realworld_vastai.sh   # Opt-in vast.ai deploy (dry-run default)
```
