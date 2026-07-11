<claude-mem-context>
# Memory Context

# [nCPU] recent context, 2026-06-28 5:57pm EDT

No previous sessions found.
</claude-mem-context>

## Cursor Cloud specific instructions

This VM is Linux x86 (no macOS, no Apple Metal, no guaranteed CUDA). Two Python
stacks are runnable here; the Rust and Metal stacks are not (see blockers).

### What runs on this VM
- **`ncpu` Python package (flagship)** — `python -m ncpu`. Env check:
  `python -m ncpu doctor`. Demos are interactive REPLs; drive them
  non-interactively by piping stdin, e.g. `python -m ncpu discover < /dev/null`
  runs a scripted gradient-descent ADD synthesis, and
  `python -m ncpu text --interactive` for the neural text machine. These need
  `torch` (CPU is fine); they train fresh models and do NOT load checkpoints.
- **Synthesis HTTP API (stdlib-only)** — `python3 -m ncpu.synthesis_api.server
  --port 8093`. It always starts; `GET /health` works. `POST /synthesize`
  returns **503** here because it shells out to the Rust `mog_synth` binary,
  which cannot be built (see blocker). `GET /stats` and template-only `/prompt`
  paths still work.

### Blockers (external, not fixable via setup)
- **`nsynth/` Rust crate does NOT build.** `nsynth/Cargo.toml` has a path dep
  `linguigenesis-core = { path = "../../linguigenesis/rust/linguigenesis-core" }`
  which resolves to `/linguigenesis/...` — a **private sibling repo not in this
  checkout**. `cargo build` fails immediately. Consequence: no `mog_synth`
  binary, so synthesis_api `/synthesize`, the MCP synthesis tools, and any
  Rust-portfolio tests (`NCPU_NSYNTH_FULL_RUN=1`, `tests/synthesis_api`,
  `tests/mcp_server`) are unavailable. To unblock, clone `linguigenesis` as a
  sibling so `/linguigenesis/rust/linguigenesis-core` exists.
- **Metal / GPU stack is macOS-only.** `python -m ncpu gpu` / `systems`,
  `tests/gpu/`, and `tests/neural/test_gpu_only_engine.py` import the
  `ncpu_metal` extension → they fail/skip on Linux. This is expected, not a bug.

### Gotchas
- **Model weights are git-LFS.** The checkout ships LFS pointer stubs. The
  update script runs `git lfs pull --include="models/**,ncpu/**"`. WITHOUT the
  real `.pt` files, torch tests fail with `torch.load` / unpickling errors
  (e.g. `invalid load key 'v'`) — that means missing LFS, NOT a code bug. The
  ~1 GB bulk of LFS is `nsynth/data/*.jsonl` (only used by the un-buildable
  Rust crate) and is intentionally not pulled.
- **No Python linter is configured** (no ruff/flake8/black). CI is
  `scripts/ci_smoke.sh` = pytest + `cargo test` (the cargo parts need the
  linguigenesis sibling). Use `pytest tests/ -q` for the fast, artifact-backed
  suite.
- Pre-existing `SyntaxError` in `ncpu/self_optimizing/mog_integration.py` (not
  on the flagship demo/test path); ignore unless working in that module.