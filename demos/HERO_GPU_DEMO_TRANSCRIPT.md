# ★ HERO GPU DEMO TRANSCRIPT
**The "GPU *is* the Computer" + Neural ALU in Shader Experience**

This is the single highest-signal way to understand what makes nCPU unique.

**Prerequisites**
```bash
pip install -e ".[demo,dev]"
# For the full Rust Metal performance + neural-in-shader path:
cd kernels/rust_metal
maturin develop --release
# (or cargo build --release if using the standalone binary)
```

---

## 1. Launch the Hero Experience (GPU as Complete Computer)

```bash
python -m ncpu gpu
# or explicitly:
python -m ncpu gpu shell --interactive
```

**What you should see / feel**:
- A real UNIX shell prompt running *entirely on the Apple Silicon GPU* via Metal compute shaders.
- ~1.9M instructions per second for real C workloads.
- You can run `ls`, `cat`, `cc`, compile programs, fork processes, use pipes, etc.
- Every instruction is executing on the GPU with zero ongoing CPU involvement after bootstrap (except for Python-mediated syscalls in the current integration).

**Key mental shift**: This is not "a program running on GPU". This is the *operating system + hardware* of a computer implemented on the GPU.

---

## 2. Experience the Determinism Superpowers (the 26-command Toolkit)

While inside the GPU shell (or via the debug mode), try these (availability depends on exact backend integration):

```bash
# Inside the GPU shell or via ncpu gpu debug
gpu-trace
gpu-history
gpu-replay
gpu-diff

gpu-break <address or symbol>
gpu-watch <address>
gpu-step

gpu-profile
gpu-stack
gpu-heat
gpu-coverage

gpu-taint
gpu-reverse
gpu-sanitize
gpu-const-time   # Critical: prove AES or other crypto has zero timing leakage
gpu-timing-proof
```

**Why this is impossible on a normal CPU**:
- Every run is bit-identical (σ=0.0 cycle variance).
- Full machine state (registers, memory, trace buffer) persists after the program exits.
- Breakpoints and watchpoints have zero overhead (checked every cycle in the shader).
- You can diff two executions instruction-by-instruction, including the exact cycle counts.
- You can do post-mortem analysis on a process that has already terminated.

This is the "super debugger" that only exists because of the GPU substrate + explicit state.

---

## 3. The Neural ALU Inside the Shader (the "CPU modeled on GPU with neural networks" part)

```bash
python -m ncpu gpu --neural-alu
# or from inside a shell that supports it
```

When the neural weights are active in the shader:
- ADD/SUB/CMP go through the Kogge-Stone neural carry-lookahead (cooperative 64-thread MLP evaluation of the carry-combine function).
- MUL goes through the byte-pair lookup tensor.
- Bitwise ops use the learned truth tables.
- All of this happens *inside the Metal kernel*, not as Python PyTorch calls.

**The performance inversion you can observe**:
- In the neural path, multiplication is dramatically faster than addition (because the LUT is O(1) while carry propagation is O(log n) passes through the MLP).

This is not an approximation. The integer ALU models are exhaustively verified to 100% accuracy on all inputs for their bit-width.

---

## 4. Run Real Linux Userspace on the "Neural-ish" GPU Computer

```bash
python -m ncpu gpu alpine --demo
# or
python demos/gpu/alpine_gpu.py --demo
```

Watch a real Alpine Linux v3.20 environment boot and run on the GPU:
- BusyBox as the multi-call binary.
- Pipes, scripting, package manager stubs, /proc, etc.
- The GPU superpower commands (the 26 debugging tools) are available as first-class commands inside this environment.

This is the concrete existence proof of the thesis.

---

## 5. (Advanced) Drop into the Raw Rust Launcher

```bash
# After building in kernels/rust_metal
cargo run --bin ncpu_run -- --elf ../../demos/gpu/busybox.elf --rootfs --interactive
# or with neural weights if the build includes them
cargo run --bin ncpu_run -- --elf ... --neural-weights path/to/weights
```

This bypasses Python entirely for maximum performance and is the purest form of "the GPU is the computer."

---

## What Makes This Insanely Novel

1. **Exact neural logic gates** running at useful speed inside a real instruction set emulator on commodity GPU hardware.
2. **The entire computer** (not just an accelerator) lives on the GPU, with determinism properties that conventional CPUs cannot provide by construction.
3. The combination enables capabilities (perfect constant-time proofs, zero-overhead full-history debugging, post-mortem everything) that change what is possible in systems security, education, and reliable computing.

Everything else in the repo (differentiable synthesis, coprocessor injection into LLMs, SOME latent controllers, JEPA-style world models of execution, etc.) is either training data for these components, ways to *use* these components from higher-level AI systems, or research into making the next generation of this substrate even more powerful.

---

**Next after this transcript**:
- Run the hero: `python -m ncpu gpu` (or `--native` for raw speed)
- Read `docs/architecture/HERO_THESIS.md` + `LAYERING.md`
- Try the live JEPA direction: `python -m ncpu.world_model.quickstart`
- Explore the ultimate horizon: `docs/architecture/BOTTOM_UP_JEPA_NEURAL_CPU.md` + `ncpu/jepa_neural_cpu/`

We're in deep continuous grind mode: repo hygiene improving, hero path becoming more delightful, and real forward progress on JEPA machine world models (fast latent simulation + robustness) built directly on the exact neural GPU computer thesis.

The long-term bet: a complete bottom-up Neural CPU where the *entire machine* is a learned (cross-)JEPA predictive world model.

Early prototype now exists and runs:
```bash
python -m ncpu.jepa_neural_cpu.demo
```