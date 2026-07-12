## Cursor Cloud specific instructions

Operating notes for this Linux cloud VM (deps are refreshed automatically by the startup update script — Python venv + `pip install -e ".[demo,dev]"`).

- **Core product = the Python `ncpu` package** (see `MASTER_ROADMAP.md`, README "Start in 60 seconds"). Activate the venv first: `source .venv/bin/activate`. Sanity check: `python -m ncpu doctor`. Hello-world: `python -m ncpu discover` (differentiable program synthesis; it is a *stochastic gradient search*, so accuracy varies per run — 85–100% is normal).
- **Tests:** `python -m pytest tests/ -q`. ~3240 tests collect; **8 collection errors are GPU-only** (`RuntimeError: No GPU backend available`) and are expected on this CPU-only VM. Lightweight CPU subset for a quick check: `python -m pytest tests/test_cli_entrypoints.py tests/test_artifact_integrity.py -q`. CI parity: `bash scripts/ci_smoke.sh`.
- **The `nsynth` Rust engine (crate `mog_synth`) does NOT build on Linux as-is.** `nsynth/src/runtime/syscall.rs` sets `sin_family = AF_INET as u8`, which only type-checks on macOS (Linux `libc::sockaddr_in.sin_family` is `u16`). So `cargo build --release` in `nsynth/`, the `nsynth_codegen` binary, and the synthesis HTTP API (`ncpu.synthesis_api.server`) are effectively macOS-only unless that cast is fixed. `nsynth` also depends on the sibling path crate `../../linguigenesis/rust/linguigenesis-core` (present in this multi-repo workspace — do not relocate it).
- **No lint tooling / no git hooks** are configured in this repo.
- The Mog compiler/runtime toolchain (`python -m ncpu doctor` reports it) is optional and not installed; only needed for compiler-backed Mog tests.

<claude-mem-context>
# Memory Context

# [nCPU] recent context, 2026-06-28 5:57pm EDT

No previous sessions found.
</claude-mem-context>