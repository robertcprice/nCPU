# nCPU Rust/GPU Next Iterations

Goal: continue advancing nCPU through Rust-first, GPU-side work, with the Rust Metal launcher and runtime as the primary user-facing execution path.

## Iteration 1: Benchmark mode in `ncpu_run`

Objective:
- add a Rust-native benchmark mode to `kernels/rust_metal/bin/ncpu_run.rs`

Suggested behavior:
- `--benchmark`
- `--repeat N`
- run the same ELF/boot image repeatedly
- print aggregate timing summary
- optionally emit JSON aggregate report

Validation:
- `cargo check --bin ncpu_run`
- run benchmark on a small ELF if available

## Iteration 2: Aggregate reporting

Objective:
- extend JSON reporting to include repeated-run aggregate statistics

Suggested fields:
- runs
- mean_cycles
- mean_elapsed_secs
- mean_ips
- min/max elapsed
- stop_reason histogram if multiple runs differ

Validation:
- benchmark mode + `--json-report`
- ensure stable parseable JSON

## Iteration 3: Rust-side docs/examples

Objective:
- add a dedicated Rust launcher workflow guide

Suggested location:
- `kernels/rust_metal/README.md`

Include:
- inspect mode
- json report mode
- benchmark mode
- ELF example
- boot-image example
- expected output snippets

## Iteration 4: Tighten launcher UX

Objective:
- make `ncpu_run` easier to use without reading source

Possible improvements:
- better `Usage:` output
- `--help`
- subcommand-like organization or clearer flag groups
- consistent human-readable summary formatting

## Iteration 5: Small correctness/support fixes local to Rust path

Objective:
- fix Rust-side issues encountered during launcher/benchmark work

Examples:
- local compile blockers
- report formatting issues
- inspect/report edge cases
- obvious correctness bugs in launcher-adjacent code

## Hard constraints

- Rust-first
- GPU-side where possible
- avoid Python-side feature work unless it is minimal glue
- preserve existing working behavior
- keep commits small and focused
- validate with `cargo check --bin ncpu_run`
