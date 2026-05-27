# nCPU World Models (JEPA-style)

This is the home for abstract predictive world models of the nCPU machine.

**Primary document:** `docs/architecture/JEPA_MACHINE_WORLD_MODEL.md`

## Current Status (April 2026)
- Skeleton + core classes implemented (`JEWorldModel`, encoders, predictor).
- First integration target: cheap latent-space speculation inside `ExecutableThoughtHead` / the latent controller before falling back to exact differentiable execution.
- Training data source: unlimited perfect traces from `ncpu.differentiable.execution.DifferentiableEngine` + GPU kernel.

## Philosophy
- Low-level substrate = **exact** (neural ALU in Python or in-shader in Rust Metal).
- High-level reasoning = **fast + abstract** in a learned latent dynamics model (JEPA style).
- The combination gives us both correctness guarantees *and* the ability to imagine many futures cheaply.

This is one of the highest-novelty directions on top of the core "GPU is a neural computer" thesis.

See the design doc for use cases (speculation, robustness/anomaly detection, better hot-loop policy, richer signals for SOME, future distillation into the kernel, etc.).
