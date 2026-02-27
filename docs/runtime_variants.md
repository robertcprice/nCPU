# Runtime Variants

## 1) Full-Model Runtime (`runtimes/full_model`)

This path wires in learned model components and related neural subsystems.

Primary files:
- `neural_cpu_full.py`
- `memory_oracle.py`
- `semantic_dispatcher.py`

Entry point (compat):
- `python3 neural_cpu.py`

Use this when you want the full learned-system behavior.

## 2) Tensor-Optimized Runtime (`runtimes/tensor_optimized`)

This path prioritizes tensor-native execution and minimizes inference overhead
in the execution hot path.

Primary files:
- `tensor_native_cpu.py`
- `tensor_native_kernel.py`

Entry points (compat):
- `python3 neural_cpu_tensor_native.py`
- `python3 neural_cpu_tensor_integrated.py`

Use this when you want execution throughput and tensor-kernel behavior.

## Demo Workloads

- `games/snake_gpu_tensor.py`: tensor-first snake workload on GPU/MPS/CPU.
- `games/snake_full_model.py`: snake driven through the full-model runtime.
