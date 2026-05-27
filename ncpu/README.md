## nCPU

The core Python package for the repository.

Major areas:

- `model/` — model-based CPU for text assembly and verified operation routing
- `neural/` — full neural CPU stack and GPU-backed neural execution path
- `tensor/` — tensor-only ARM64 execution kernel
- `differentiable/` — differentiable execution, synthesis, and diff-compiler work
- `coprocessor/` — LLM coprocessor integration and training
- `execution_training/` — execution-as-training-signal stack
- `self_optimizing/` — SOME runtime/training code while that effort still lives in this repo
- `os/` — neural/GPU operating-system experiments
- `distributed/` — multi-GPU and IPC scheduling work
- `crypto/` — constant-time crypto implementations and verification helpers
- `utils/` — smaller shared utilities used across the paper/release surface

Top-level entrypoints:

- `lab.py` — curated interactive launcher and demo guide
- `demo.py` — compact demo entrypoint
- `__main__.py` — package entrypoint for `python -m ncpu`
