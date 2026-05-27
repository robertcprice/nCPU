# nCPU Architecture Layering & Ownership

**Purpose**: Make it obvious to every contributor and user which code owns which part of the story, what the tradeoffs are, and where new work belongs.

## The Two Primary Layers

### 1. Performance + Determinism Layer (The GPU Computer)
**Home**: `kernels/rust_metal/`

**What it is**:
- High-performance ARM64 CPU emulator implemented as Metal compute shaders (Rust + objc2-metal + PyO3).
- ~200 instructions (integer + floating-point).
- Full multi-process UNIX semantics (fork/pipe/wait, 28+ syscalls, 25-command shell).
- Real ELF64 loader → BusyBox (264 KB) + Alpine Linux v3.20 on GPU.
- Self-hosting C compiler running entirely on the GPU.
- **Optional neural ALU weights embedded directly in the Metal shader** (cooperative threadgroup evaluation of the Kogge-Stone CLA carry-combine MLP, byte-pair MUL LUT, truth tables, precomputed shift LUTs).
- Zero-copy StorageModeShared memory, GPU-side syscall buffering.
- 26-command deterministic post-mortem debugging toolkit (trace, replay, breakpoints, watchpoints, reverse dataflow, constant-time verification, memory sanitization, fuzzing, etc.).
- Bit-identical determinism (sigma=0.0 cycle variance).

**Performance**: ~1.9M IPS sustained for real C workloads. ~500× faster compilation than the older Python paths.

**When to use / extend this layer**:
- You want maximum speed or real OS behavior.
- You want the determinism superpowers (debugging, security proofs, education).
- You are adding new ARM64 instructions, improving the ELF loader, adding new GPU superpower commands, or embedding more neural operations in-shader.
- You are building the "GPU is a self-sufficient computer" experience.

**Key files**:
- `src/neural_cpu.rs`, `src/neural_alu.rs` — the in-shader neural execution heart.
- `src/full_arm64.rs`, `src/launcher.rs`, `src/elf_loader.rs`.
- `bin/ncpu_run.rs` — the standalone binary.

**Status**: The production path for the hero thesis. This is where the unique "CPU modeled on GPU with neural networks and kernels" claim lives in its strongest form.

### 2. Research + Differentiability Layer (The Neural / Programmable Computer Lab)
**Home**: `ncpu/` (Python package)

**What it is**:
- Multiple CPU implementations for research:
  - `ncpu.neural.*` — Full GPU-resident NeuralCPU with trained `.pt` models (Kogge-Stone via neural full adders + carry-combine, byte-pair MUL, attention shifts, etc.).
  - `ncpu.model.*` — Text-assembly model CPU, primarily for training/testing the neural ops.
  - `ncpu.tensor.*` — Pure tensor implementation (fast mode, maximum differentiability with native autograd).
- The differentiable execution engine (`ncpu/differentiable/`) — backprop through programs, gradient-based synthesis, neural ISA discovery, self-modifying differentiable programs, diff compiler.
- The coprocessor system (`ncpu/coprocessor/`) — inject the neural ALU (hard or soft bilinear truth tables + STE) into transformer forward passes as a routed expert.
- Neural OS prototypes (`ncpu/os/neuros/`) — 11 trained models for MMU, TLB, cache, scheduler, assembler, compiler, etc. (the "neurOS" research line).
- Execution-guided training signals for code models (`ncpu/execution_training/`).
- Self-optimizing / hidden controller work (`ncpu/self_optimizing/`, the SOME runtime).

**Performance**: Much slower than the Rust kernel (thousands of IPS). Designed for gradient flow and experimentation, not production speed.

**When to use / extend this layer**:
- You are training or studying the neural ALU models.
- You want to backpropagate through execution.
- You are working on the coprocessor or differentiable synthesis.
- You are exploring "what if the entire OS or compiler was learned?"
- You need the exact same ISA semantics in a fully differentiable setting.

**Key principle**: This layer should be able to produce weights and insights that improve the Performance Layer (e.g., new neural op architectures that later get ported into the Rust Metal shader).

## Current Live High-Value Work Spanning Both Layers (as of this session)

The bottom-up JEPA Neural CPU (JNC) + live substrate integration is the clearest current example of the two layers working together at the highest signal:

- Performance Layer (Rust Metal) supplies the real deterministic multi-process UNIX substrate (BusyBox + Alpine, 50+ syscalls, perfect post-mortem state).
- Research + Differentiability Layer supplies the learned JEPA predictors and churn model.
- Integration point: `jepa_observer` wired into `GpuLauncher` — real snapshots at every schedule point and memory syscall → shadow structured memory model (dirty + access + true recency via `current_step` / `_last_touch_step`) → three decision levers (context-switch bias, immediate syscall bias, adaptive persistent yield skips driven by churn delta) → fairness telemetry (`times_scheduled`, `per_process_scheduled`, `get_all_deprios`).
- Concrete results on real guest code: 69–80 overrides per short multi-process BusyBox run, visible differentiation (0.04 spreads), adaptive de-prio that makes small relative deltas produce multi-turn yield.

This pattern (exact fast substrate + learned predictive layer observing and lightly steering it) is the direct path to the Neural OS vision. New work should be evaluated against whether it strengthens this loop.

## Supporting Systems (Clearly Scoped)

| System | Home | Relationship to Hero Thesis | Recommended Scope |
|--------|------|-----------------------------|-------------------|
| nsynth / Mog synthesis | `nsynth/` (Rust) + `benchmarks/` harnesses | Verified execution-based code generation engine. Excellent for producing training data and for the "program by examples" interactive demos. | Supporting tool for the differentiable layer and for practical LLM code generation. Not the core "GPU computer" claim. |
| SOME / Weight CPU | `ncpu/self_optimizing/` + `some/` | Hidden controller + task-local fast weights + latent heads for internal reasoning before visible output. Long-term "CPU inside the model" vision. | High-potential future work. Currently a parallel research thread. Should eventually feed the coprocessor or differentiable layers. |
| EGDC | `egdc/` | Earlier differentiable compiler / execution-guided debug surface. | Historical / parallel research. Evaluate for integration or archival during reorganization. |

## Explicit Tradeoff Table (What Users Should See)

| Mode / Path | Speed | Differentiable? | Determinism | Best For | Primary Home |
|-------------|-------|------------------|-------------|----------|--------------|
| Rust Metal (GPU Computer) | ~1.9M IPS | No (or limited via future diff JIT) | Perfect (sigma=0.0) | Real OS, debugging superpowers, production demos, constant-time proofs | `kernels/rust_metal/` |
| Rust Metal + Neural ALU in-shader | High (slightly slower than native Metal ops) | No (weights are frozen in shader) | Perfect | The full hero experience: GPU computer with neural logic | `kernels/rust_metal/` (neural_cpu.rs + weights) |
| Python Neural (research) | ~5K IPS | Yes (full) | Good (model determinism) | Training, coprocessor research, backprop through execution | `ncpu/neural/`, `ncpu/differentiable/` |
| Python Fast Tensor | Higher than neural models | Yes (native autograd) | Good | Rapid prototyping of differentiable programs | `ncpu/tensor/` |
| neurOS (Python neural OS models) | Low | Yes | Research-grade | Exploring fully learned operating systems | `ncpu/os/neuros/` |

All unified user-facing commands (`ncpu gpu`, `ncpu discover`, etc.) must surface this table or a one-line version of it.

## Ownership Rules (For Future Work)

- **New ARM64 instruction, syscall, or GPU superpower command** → `kernels/rust_metal/`.
- **New neural op architecture or training recipe for the ALU** → `ncpu/` research layer first (prove it), then port the weights/architecture into the Rust shader if it wins.
- **Coprocessor improvements, new routing, scaling studies** → `ncpu/coprocessor/`.
- **Gradient-based program synthesis or diff compiler work** → `ncpu/differentiable/`.
- **Hidden controller / internal reasoning / fast weights** → `ncpu/self_optimizing/` (coordinate with SOME docs).
- **Verified synthesis engine improvements** → `nsynth/`.
- **Anything that makes the "GPU is the computer with optional neural ALU" story stronger** → treat as hero work and land in the Performance Layer (or produce artifacts that clearly feed it).

## Historical / Legacy Code (Do Not Confuse New Readers)

- `kernels/mlx/` — older Python Metal kernels. Largely superseded by `rust_metal/`. Label as legacy in docs.
- `scripts/cli/ncpu.py` — old direct launcher. Being superseded by the unified `ncpu lab` / `python -m ncpu` surface. Add deprecation notices.
- `ncpu/demo.py` legacy flags (`--live`, `--headless`, etc.) — route through the new surface or label.
- `some/`, `egdc/` top-level presence — document as historical research threads.

## How This Layering Serves the Hero Thesis

The hero claim only holds if the Performance Layer (Rust Metal + optional in-shader neural ALU) is clearly the star, and the Research Layer is understood as the laboratory that produces better neural components and new usage modes (coprocessor, differentiable synthesis) for that star.

When in doubt, ask: "Does this change make the GPU feel more like a complete, self-sufficient, optionally-neural computer with superpowers that CPUs cannot have?"

If yes → it strengthens the hero thesis.

---

*Maintainers: Update this document whenever ownership or layering decisions change. It is the contract between the two primary layers.*