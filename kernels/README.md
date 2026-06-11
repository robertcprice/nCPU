# kernels/

Low-level execution kernels — the compiled/accelerated backends under the Python
layer.

- `rust_metal/` — the Rust + Metal GPU compute kernel (~50K LOC): ARM64
  emulation, Turing-complete VM, the GPU-native execution engine, JEPA hooks.
- `mlx/` — Apple MLX kernels.
- `npcot_wasm/` — WebAssembly build of the Neural-Physical CoT path.

These are the performance-critical substrate; the `ncpu/` Python package calls
into them.
