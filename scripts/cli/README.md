## Scripts CLI (Legacy / Direct)

User-facing launcher scripts.

**⚠️ Most of this surface is legacy.** The unified, recommended entry point is now:

- `python -m ncpu` / `ncpu-lab` (see `ncpu/lab.py`)
- `python -m ncpu gpu` (the hero Rust Metal GPU computer path)

- `ncpu.py` — direct/legacy nCPU launcher for neural, fast, and older compute execution modes.
  Add deprecation notices have been added; prefer the unified `ncpu gpu` / Rust `ncpu_run` path for the GPU-as-computer experience.
