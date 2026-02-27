# nCPU

`nCPU` is a model-native CPU research project that explores two execution strategies:

1. Full-model execution: learned components are wired directly into execution paths.
2. Tensor-optimized execution: execution is dominated by tensor operations rather than LLM inference in the hot loop.

This repository is organized to be understandable, reproducible, and publishable.

## What This Project Is

nCPU is not a single script. It is a structured workspace for:
- CPU/runtime experimentation
- neural and tensor execution prototypes
- workload validation (including DOOM and Snake)
- benchmark tooling and analysis

## Core Runtime Variants

### Full-Model Runtime
- Path: `runtimes/full_model/`
- Entry: `python3 neural_cpu.py`
- Includes broader learned subsystem integration.

### Tensor-Optimized Runtime
- Path: `runtimes/tensor_optimized/`
- Entries:
  - `python3 neural_cpu_tensor_native.py`
  - `python3 neural_cpu_tensor_integrated.py`
- Focuses on tensor-first execution behavior.

## Snake GPU Workload

- Primary demo: `games/snake_gpu_tensor.py`
- Compatibility entry: `python3 snake_neural.py`

Example headless run:

```bash
python3 games/snake_gpu_tensor.py --steps 500 --render-every 0
```

## Repository Layout

```text
nCPU/
├── src/                    # Core package baseline (kvrm_cpu)
├── models/                 # Pre-trained models (ALU, 64-bit kernels, decoders)
├── kernels/                # Metal and Rust optimized MLX CPU kernels
├── runtimes/               # Runtime implementations by strategy
├── games/                  # Interactive workloads/demos (snake)
├── workloads/              # Large domain workloads (doom/linux/neural)
├── tools/                  # Runners + analysis scripts
├── artifacts/              # Generated outputs, model blobs, result files
├── benchmarks/             # Benchmark scripts and related docs
├── training/               # Training pipelines and experiments
├── programs/               # Assembly/program fixtures
├── docs/                   # Architecture and reference documentation
├── tests/                  # Test suite
└── archive/                # Non-core/legacy material (git-ignored)
```

## Backward-Compatible Entrypoints

These root scripts are intentionally kept for convenience:
- `neural_cpu.py`
- `neural_cpu_tensor_native.py`
- `neural_cpu_tensor_integrated.py`
- `snake_neural.py`
- `snake_neural_cpu.py`

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Testing

Fast smoke tests:

```bash
pytest tests/test_runtime_variants.py -q
```

Broader tests (excluding slow-marked):

```bash
pytest tests -m "not slow"
```

Note: Some real-model tests require optional packages and model artifacts (for example `peft`).

## Why The Archive Exists

This project had accumulated many ad-hoc scripts and experimental directories at root.
Those are now moved into `archive/` so the main tree stays clean and easier to maintain.
The `archive/` folder is excluded from Git via `.gitignore`.
