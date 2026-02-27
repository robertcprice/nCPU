# Project Structure and Conventions

## Goals of the reorganization

- Keep root directory readable.
- Split runtime code by execution philosophy.
- Keep compatibility with old script names.
- Isolate noisy experimental clutter into `legacy/`.

## Conventions

- Runtime code lives under `runtimes/<variant>/`.
- Game/demo workloads live under `games/`.
- Large experimental families live under `workloads/`.
- Execution/launch scripts live under `tools/runners/`.
- Analysis scripts live under `tools/analysis/`.
- Generated outputs and model blobs live under `artifacts/`.
- Core stable package remains `src/kvrm_cpu/`.
- Tests live in `tests/` (avoid root-level `test_*.py`).

## What moved

- `neural_cpu.py` source -> `runtimes/full_model/neural_cpu_full.py`
- `neural_cpu_tensor_native.py` source -> `runtimes/tensor_optimized/tensor_native_cpu.py`
- `neural_cpu_tensor_integrated.py` source -> `runtimes/tensor_optimized/tensor_native_kernel.py`
- `snake_neural.py` source -> `games/snake_gpu_tensor.py`
- `snake_neural_cpu.py` source -> `games/snake_full_model.py`
- `benchmark_*.py` scripts -> `benchmarks/scripts/`
- `train_*.py` scripts -> `training/experiments/`
- `doom_*`, Linux boot, and NeuralOS families -> `workloads/`
- `run_*.py`/`run_*.sh` operational launchers -> `tools/runners/`
- trace/result/model blobs from root -> `artifacts/`

## Compatibility

Original top-level filenames are now wrappers so external scripts still run.
