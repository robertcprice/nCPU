# rust_metal Module Organization

This crate was previously a flat collection of ~50 `.rs` files directly in `src/`.

It has been restructured into a proper layered module system that makes the different kinds of kernels and responsibilities explicit.

## Current Structure

| Module      | Responsibility |
|-------------|----------------|
| `core/`     | Fundamental ARM64 CPU emulation (fetch/decode/execute, micro-ops, instruction semantics) |
| `os/`       | Process model, launcher, scheduling, and multi-process UNIX runtime (BusyBox, Alpine, self-hosting C compiler) |
| `neural/`   | Neural ALU (in-shader), trained weights, dispatch logic, neural CPU variants, hybrid modes |
| `jepa/`     | **High-signal current work**: JEPA Neural Kernel + learned Neural OS (3 decision levers for scheduling, recency-aware churn, fairness telemetry) |
| `execution/`| All the different execution engine variants (pure_gpu, optimized, ultra, OOO, differentiable, JIT, fusion, parallel, continuous, async, etc.) |
| `loader/`   | ELF loading, boot images, root filesystem, VFS |
| `cache/`    | Blackboard / working-set caches used by the neural OS layer |
| `support/`  | Small supporting utilities (native ABI helpers, NPCoT bridge) |

The root of `src/` is now clean — only `lib.rs` and this README remain at the top level.

## Top-level Re-exports (for ergonomics)

```rust
use ncpu_metal::{
    GpuLauncher,
    Process, ProcessManager,
    GpuFullArm64,
    NeuralJepaKernel,
    PreparedElf,
    // plus full module paths: ncpu_metal::os::..., ncpu_metal::jepa::..., etc.
};
```

## Why This Matters

The old flat structure hid the architecture. It was hard to answer:
- Where is the deterministic GPU computer implemented?
- Where is the neural ALU in the shader?
- Where is the live learned JEPA kernel that actually steers real scheduling on BusyBox workloads?

The new layout makes the hero thesis ("The GPU *is* the computer") and the current high-signal JEPA direction immediately visible in the filesystem.

## Status (as of this session)

**Reorganization substantially complete.**

All significant code has been moved out of the flat `src/` root into logical modules:

- `core/` — ARM64 CPU emulation engine
- `os/` — Process model + launcher + UNIX runtime
- `neural/` — Neural ALU + weights + dispatch
- `jepa/` — Learned JEPA Neural Kernel + Neural OS (current high-signal work)
- `execution/` — All execution engine variants
- `loader/` — ELF, boot, rootfs, VFS
- `cache/` — Blackboard caches
- `support/` — Small utilities

The crate compiles cleanly (`cargo check --release`) after every phase. Sensible `pub use` re-exports have been added throughout for ergonomic usage.

## Future Polish Work (lower priority)

- Move the 26-command deterministic debugging toolkit into a proper `debug/` module
- Consider extracting the large embedded Metal shader sources into a `metal/` module
- Further split very large files inside `core/` or `execution/` if they grow
- Improve inline documentation comments in the largest files

The current structure already delivers the main goal: making the different kinds of kernels and the hero architecture obvious and maintainable.

- Extract Metal shader sources more cleanly into `metal/`
- Move the 26-command deterministic debugging toolkit into `debug/`
- Add more `pub use` re-exports if needed for specific workflows
- Possibly split very large files inside `core/` and `execution/` further

This organization makes it much safer and clearer to extend specific areas (new learned predictors, new execution modes, new in-shader neural operations, new debug commands) without touching unrelated code.
