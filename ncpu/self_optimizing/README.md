## nCPU Self-Optimizing

Runtime and training code for the SOME effort while it still lives inside the shared hub repo.

High-level layout:

- `core/` — main engine, verification, model integration, and experiment orchestration
- `controller/` — controller bundle/runtime/training code
- `architecture/` — weight-CPU and related architecture definitions
- `latent_heads/` — latent action, halt, descriptor, and memory heads
- `benchmarks/` — benchmark runners and evaluation helpers
- `training/` — training-specific helpers and data handling
- `data/` — tracked small data/assets needed by this stack

Top-level modules that remain here are the operational glue around those packages while the code is still being consolidated.
