<p align="center">
  <img src="assets/logo.png" alt="nCPU" width="400">
</p>

<p align="center">
  <strong>An end-to-end AI computer. Every layer --- from arithmetic to OS to compiler --- is either a trained neural network or runs entirely on GPU.</strong><br>
  The AI doesn't run <em>on</em> a computer. The AI <em>is</em> the computer.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/tests-1%2C544%20passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/models-24%20trained-blue" alt="Models">
  <img src="https://img.shields.io/badge/accuracy-100%25%20integer-green" alt="Accuracy">
  <img src="https://img.shields.io/badge/verified-exhaustive-blueviolet" alt="Verified">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License">
</p>

---

## Three Big Ideas

### 1. A Fully Differentiable CPU

Every ALU operation is a trained neural network --- addition, subtraction, multiplication, bitwise, shifts, division. Because the entire computation graph is differentiable, this opens the door to **optimizing programs via gradient descent**: backpropagating through execution to discover better algorithms, instruction schedules, or hardware configurations. No conventional CPU can do this.

### 2. A Complete AI Computer

Not "AI running on a computer" --- an AI that **is** the computer, end to end. The neural ALU computes. The neural OS (neurOS) manages memory, schedules processes, compiles code. The GPU executes compiled C programs, boots a UNIX shell, runs a self-hosting compiler, serves HTTP, plays games, runs VMs. From the silicon to the inference layer, every component is either learned or GPU-native. This is what a complete AI computational apparatus looks like.

### 3. GPU as Self-Sufficient Computer

A single GPU chip running an entire computer --- no CPU required beyond initial bootstrap. The Metal compute shader executes ARM64 natively at 4M+ IPS, boots a multi-process UNIX OS with fork/pipe/wait, compiles C, loads and runs real Linux ELF binaries (BusyBox), and even runs a 2-instruction Turing-complete VM (MUXLEQ) that boots eForth. The GPU isn't an accelerator here. It's the whole machine.

> See the [research paper](paper/ncpu_paper.md), the standalone [GPU debugging toolkit paper draft](paper/gpu_debugging_toolkit_paper.md), and the [wiki](../../wiki) for detailed analysis.

## Quick Start

```bash
pip install -e ".[dev]"

# Neural mode --- all arithmetic through trained neural networks
python main.py --program programs/fibonacci.asm

# GPU compute mode --- Metal shader, ~4M IPS
python main.py --program programs/fibonacci.asm --compute

# GPU UNIX OS --- 25-command shell with fork/pipe/wait on Metal
python ncpu/os/gpu/demo.py --multiproc

# Run real BusyBox on the GPU
python demos/busybox_gpu_demo.py

# Rust-native launcher --- standalone Rust path (ELF or boot image)
cd kernels/rust_metal
cargo run --bin ncpu_run -- --elf ../../demos/busybox.elf --rootfs -- echo hello
cargo run --bin ncpu_run -- ../../path/to/image.bin
```

`cargo check --bin ncpu_run` currently passes in this workspace. Direct `cargo run` is still subject to the local PyO3/Python link environment.

## The Stack

| Layer | Implementation | What It Proves |
|-------|---------------|----------------|
| **ALU** | 13 trained `.pt` models | Neural nets do exact integer arithmetic (exhaustively verified) |
| **OS** | 11 neural models (neurOS) | Learned MMU, TLB, cache, scheduler, compiler --- zero fallbacks |
| **Compute** | Rust Metal kernel (139 ARM64 insns) | GPU executes arbitrary programs at ~1.9M IPS, zero-copy StorageModeShared |
| **UNIX OS** | Compiled C on Metal | Fork/pipe/wait, 25-command shell, 28 syscalls |
| **Compiler** | cc.c self-hosting on GPU | GPU hosts a complete software development toolchain |
| **ELF Loader** | Real Linux binaries on GPU | BusyBox (264KB, 30+ applets) runs on Metal |
| **MUXLEQ** | 2-instruction Turing-complete VM | If neural nets handle 2 instructions exactly, the principle is universal |

```python
# Neural mode --- every operation is a trained model
from ncpu.model import CPU
cpu = CPU(neural_execution=True)
cpu.load_program("MOV R0, 7\nMOV R1, 6\nMUL R2, R0, R1\nHALT")
cpu.run()
print(cpu.get_register("R2"))  # 42 --- computed by neural byte-pair LUT

# GPU compute mode
from kernels.mlx.ncpu_kernel import NCPUComputeKernel
kernel = NCPUComputeKernel()
kernel.load_program_from_asm("MOV R0, 7\nMOV R1, 6\nMUL R2, R0, R1\nHALT")
result = kernel.execute()  # ~4M IPS on Metal
```

## What's Running on the GPU

### GPU-Native Multi-Process UNIX OS

A 25-command UNIX shell running as compiled C on Apple Silicon Metal with full multi-process support:

```
gpu:/home/user$ ls | grep .c | sort
fib.c
fork_test.c
hello.c
gpu:/home/user$ cc fork_test.c && run /bin/fork_test
Parent PID: 1
Forked child PID: 2
Child process (PID 2, parent 1)
Child exited, parent done
```

- **25 shell commands** including pipes (`|`), background (`&`), chaining (`;`/`&&`/`||`), redirect (`>`/`>>`)
- **Multi-process**: fork/wait/pipe/dup2/kill via memory swapping, up to 15 concurrent processes
- **28 syscalls**, freestanding C runtime with malloc/printf/fork/pipe/qsort/strtol
- **Robustness**: fork bomb protection, SIGTERM/SIGKILL, orphan reparenting, per-process resource limits

### Self-Hosting C Compiler on Metal GPU

A ~4,200-line self-hosting C compiler (`cc.c`) that compiles C source into ARM64 machine code **entirely on the Metal GPU**, then executes the result on the same GPU:

```
Host GCC compiles cc.c -> compiler₀
  GPU runs compiler₀, self-compiles cc.c -> compiler₁
    GPU runs compiler₁, compiles test.c -> binary
      GPU runs test binary -> correct result
```

Supports 18 C features: structs (`.`/`->`), pointers, arrays, recursion, for/while/do-while, ternary, sizeof, compound assignment, bitwise ops, short-circuit `&&`/`||`, `enum`, `typedef`, `switch`/`case`/`default`, `#ifdef`/`#ifndef`/`#elif`/`#endif`, global initializers, function pointers, `union`, function-like macros, `goto`/labels, multi-dimensional arrays. **73/73 test programs verified, 18 bugs fixed, full self-compilation verified.**

### BusyBox on Metal GPU

Real BusyBox (Alpine Linux core utils, 264KB static binary) running on the Metal GPU shader via an ELF64 loader:

- Cross-compiled with `aarch64-linux-musl-gcc -static`
- ELF64 parser loads PT_LOAD segments, sets up Linux stack (argc/argv/envp/auxv)
- 50+ Linux syscalls handled: exit, read, write, brk, mmap, ioctl, writev, uname, symlink, etc.
- 34+ verified commands: echo, uname, cat, ls, printf, basename, dirname, head, tail, wc, cut, sort, uniq, grep, expr, touch, mkdir, rm, cp, stat, mv, chmod, sleep, tr, find, tee, readlink, ln, and more
- GPUFilesystem wired via syscalls --- `cat /etc/motd` reads from Python-side filesystem

### Alpine Linux on Metal GPU

Full Alpine Linux v3.20 distribution running on Metal GPU compute shader with a comprehensive POSIX shell:

- BusyBox (321 KB, musl libc) as multi-call binary behind every command
- Pipes (`|`), chaining (`;`/`&&`/`||`), redirection (`>`/`>>`), command substitution (`$(cmd)`)
- Shell scripting: for/while/if/elif/case, functions, local variables, parameter expansion, brace expansion
- 35+ builtins, here-documents, glob expansion, aliases, history
- 26 novel GPU superpower commands spanning post-mortem forensics, replay/diff, state snapshots, tracing, breakpoints, watchpoints, profiling, disassembly, sanitization, fuzzing, reverse data flow, constant-time verification, memory visualization, and entropy analysis. Full reference: [GPU debugging toolkit](docs/gpu_debugging_toolkit.md)

### Rust Metal Kernel

The primary execution backend: Rust + Metal with StorageModeShared for zero-copy GPU↔Python communication. [Architecture docs](docs/rust_metal_kernel.md).

- **139 ARM64 instructions**, ~1.9M IPS sustained
- **~500x faster compilation** than the Python MLX kernel (~44ms vs ~22s)
- **Zero-copy SVC handling** via unified memory (no 16MB copies per syscall)
- GPU-side SVC buffer for SYS_WRITE, SYS_BRK, SYS_CLOSE, SYS_EXIT
- **GPU-native debugging toolkit** (26 commands: trace, breakpoints, watchpoints, disassembler, sanitizer, fuzzer, reverse analysis, constant-time verification, and more)
- Built with `maturin develop --release`, exposed to Python via PyO3
- Rust-native runtime modules for boot-image loading, ELF loading, VFS/rootfs, syscall handling, native ABI experiments, and standalone launching live in `kernels/rust_metal/src/{boot_image,elf_loader,vfs,rootfs,process,native_abi,launcher}.rs`
- The live standalone launcher path now includes `ProcessManager`-backed scheduling, fork/wait/pipe/dup/exec, and Linux `clone(220)` interception
- Remaining gaps are mostly cutover and cleanup work: Python still owns many repo-facing orchestration surfaces, and direct `cargo run` still depends on a healthy local PyO3/Python linker setup

### GPU-Native Debugging Toolkit

A finished debugging platform **impossible on conventional CPUs**. The Metal
kernel provides a verified 26-command toolkit for instruction tracing,
breakpoints, watchpoints, replay/diff, disassembly, fuzzing, reverse data flow,
constant-time verification, and memory analysis. Full reference:
[GPU debugging toolkit](docs/gpu_debugging_toolkit.md). Standalone paper draft:
[GPU debugging toolkit paper](paper/gpu_debugging_toolkit_paper.md).

Core capabilities:

- **Instruction tracing**: 4096-entry circular buffer capturing PC, instruction word, x0-x3, NZCV flags, and SP (56 bytes/entry) at 0x3B0000
- **Breakpoints**: Up to 4 PC breakpoints checked every GPU cycle at zero overhead (debug control block at 0x3A0000)
- **Conditional breakpoints**: Fire only when PC matches AND a register equals a specific value
- **Memory watchpoints**: Shadow-comparison write-watch on up to 4 addresses, fires the instant a value changes
- **Time-travel debugging**: Browse instruction-by-instruction execution history with register/flag diffs
- **Instruction coverage**: 88+ ARM64 instruction type classifier
- **Performance profiling**: Instruction mix, call graph, compute/memory ratio analysis
- **Call stack reconstruction**: BL/RET tracking with call tree visualization
- **Single-step debugging**: First N instructions with register change diffs per step
- **Instruction frequency heatmap**: Visual block-character display of hot/cold code regions
- **Comparative execution**: Diff traces of the same program under different inputs
- **Deterministic replay**: Bit-identical execution (σ=0.0000)
- **ARM64 disassembler**: Built-in disassembly of all 139 instructions in traces
- **Memory sanitizer**: Zero-overhead memory safety checking (vs ASan's 2-5x overhead)
- **Automated fuzzing**: Crash detection with instant post-mortem traces (no reproduction needed)
- **Reverse data flow**: Trace backwards to find where a value originated
- **Constant-time verification**: Exact verification of constant-time crypto (impossible on noisy CPUs)
- **Memory map visualization**: Visual GPU memory layout with access patterns
- **Cross-execution comparison**: GPU vs QEMU trace diffing

**Why this matters**: On a CPU, process state is destroyed after exit, breakpoints require ptrace overhead, watchpoints are limited by hardware debug registers, and non-deterministic microarchitecture prevents replay. On GPU, ALL execution state persists, breakpoints and watchpoints are free (checked every cycle in the shader), and every run is deterministic.

```python
cpu.enable_trace()
cpu.set_breakpoint(0, 0x10040)            # Break at address
cpu.set_conditional_breakpoint(1, 0x10080, 0, 42)  # Break when x0==42
cpu.set_watchpoint(0, 0x50000)            # Watch memory address
result = cpu.execute(100000)              # Stops at first trigger
trace = cpu.read_trace()
for pc, inst, x0, x1, x2, x3, flags, sp in trace[-10:]:
    nzcv = f"{'N' if flags & 8 else '.'}{'Z' if flags & 4 else '.'}{'C' if flags & 2 else '.'}{'V' if flags & 1 else '.'}"
    print(f"PC=0x{pc:08X} [{nzcv}] SP=0x{sp:X}")
# Watchpoint info: which address changed, old/new values
wp_info = cpu.read_watchpoint_info()  # (index, addr, old_val, new_val)
```

**Alpine shell commands**: all 26 commands from [GPU debugging toolkit](docs/gpu_debugging_toolkit.md), including `gpu-help`, `gpu-xray`, `gpu-replay`, `gpu-diff`, `gpu-freeze`, `gpu-thaw`, `gpu-timing-proof`, `gpu-strace`, `gpu-trace`, `gpu-break`, `gpu-watch`, `gpu-history`, `gpu-coverage`, `gpu-taint`, `gpu-bisect`, `gpu-step`, `gpu-profile`, `gpu-stack`, `gpu-heat`, `gpu-diff-input`, `gpu-asm`, `gpu-sanitize`, `gpu-fuzz`, `gpu-reverse`, `gpu-const-time`, `gpu-map`, and `gpu-entropy`

### 13+ Compiled C Applications on Metal

| Category | Programs |
|----------|----------|
| **Crypto** | SHA-256, AES-128 ECB+CBC (6/6 FIPS pass), password vault |
| **Games** | Tetris, Snake, roguelike dungeon crawler, text adventure |
| **VMs** | Brainfuck interpreter, Forth REPL, CHIP-8 emulator |
| **Networking** | HTTP/1.0 server (TCP via Python proxy) |
| **Neural net** | MNIST classifier (Q8.8 fixed-point, 784->128->10) |
| **Tools** | ed line editor, Game of Life, self-hosting compiler |

### MUXLEQ: Turing-Complete in 2 Instructions

A minimal proof of universality: SUBLEQ + MUX running on nCPU in three modes (neural, fast, compute). Loads `.dec` images, boots eForth. If neural nets exactly execute a 2-instruction OISC, the principle extends to any instruction set.

### neurOS: Fully Neural Operating System

Every OS component is a trained neural network --- 11 models, zero fallbacks:

| Component | Accuracy | Component | Accuracy |
|-----------|----------|-----------|----------|
| MMU | 100% | Assembler codegen | **100%** |
| TLB | 99.6% | Assembler tokenizer | 99.4% |
| Cache | 99.7% | Compiler optimizer | 95.2% |
| Scheduler | 99.2% | Watchdog | 100% |
| Prefetch | 97.8% | Block allocator | 98.4% |

Self-compilation verified: nsl source -> neural compiler -> neural assembler -> neural CPU -> correct results.

### Timing Side-Channel Immunity

GPU execution produces **zero cycle-count variance** (sigma=0.0 across 270 runs). Same code on native Apple Silicon shows 47-73% timing variance. AES-128 T-table attacks are structurally impossible --- no data cache, no cache lines, no cache-miss penalty.

## Self-Optimizing Machine Engine (SOME)

SOME is the repo's execution-grounded code and reasoning stack. The current implementation is not "magic full dense CPU inside a frozen transformer," and the docs now state that plainly. What exists today is a benchmarkable bridge toward that goal:

- **Buffered hidden controller**: `think -> write -> verify -> patch -> commit`, with only committed output exposed
- **Latent control heads**: learned latent action, halt, descriptor, state-patch, and recurrent memory heads
- **Task-local fast weights**: descriptor-driven per-task weight updates during inference
- **Segmented decode path**: recent exact token window plus compressed committed-history descriptors
- **Trajectory-first training loop**: hidden-controller trajectories feed SFT, latent-head, and patch-head training
- **Async API and bundles**: bundle-backed buffered inference, async task streaming, and benchmark integration

Current evidence worth paying attention to:

- **HumanEval+**: `qwen3.5:4b 147 -> 154`, `9b 144 -> 156`, `27b 153 -> 156`
- **BigCodeBench-Hard**: finished `qwen3.5:9b 33 -> 49`
- **Latent-memory proof**: on a fresh synthetic memory-aware corpus, the learned memory head improved validation MSE by `83.26%` over the zero-delta baseline

Read these first:

- [SOME Complete Guide](docs/SOME_COMPLETE_GUIDE.md)
- [SOME Architecture](docs/SOME_ARCHITECTURE.md)
- [Weight CPU Architecture](docs/WEIGHT_CPU_ARCHITECTURE.md)
- [SOME Results](docs/SOME_RESULTS.md)

## Neural Arithmetic

| Instruction | Neural Model | Strategy | Latency |
|-------------|-------------|----------|---------|
| ADD/SUB/CMP | arithmetic.pt + carry_combine.pt | Kogge-Stone CLA (8 passes) | 248 us |
| MUL | multiply.pt | Byte-pair LUT (65,536 entries) | 21 us |
| AND/OR/XOR | logical.pt | Vectorized truth table | 21 us |
| SHL/SHR | lsl.pt / lsr.pt | Attention-based bit routing | 434 us |
| DIV | arithmetic.pt | Restoring division (neural subtraction) | varies |

**Multiplication is 12x faster than addition** --- inverting the conventional CPU hierarchy. Addition requires a sequential carry chain (Kogge-Stone CLA, 8 neural passes). Multiplication decomposes into parallel byte-pair lookups (one pass). Classical hardware algorithms transfer to neural architectures, but the performance hierarchy flips.

All sub-components **exhaustively verified** --- every possible input tested, not sampled.

## Project Structure

```
ncpu/
  os/
    neuros/     # Neural OS: 17 modules (MMU, TLB, cache, scheduler, compiler, ...)
    gpu/        # GPU UNIX OS: runner, filesystem, shell, ELF loader
      src/      # C source (shell, libc, syscalls, linker script)
      programs/ # Compiled C apps (crypto, games, vms, net, nn, tools, graphics)
  self_optimizing/  # SOME runtime, hidden controller, fast weights, benchmark stack
  neural/       # NeuralCPU: 12K-line CPU with neural ALU bridge
  model/        # Model-based CPU (neural_ops, assembler, architectures)
  tensor/       # Tensor-based ARM64 emulator
kernels/
  mlx/          # Metal compute kernels (ARM64 V2 + nCPU ISA + MUXLEQ)
  rust_metal/   # Rust + Metal ARM64 kernel (primary backend, ~500x faster)
models/         # 24 trained .pt models (alu, shifts, math, os, decode)
programs/       # 62 assembly programs
tests/          # 1,544 tests across 21 files
benchmarks/     # Benchmark scripts; generated result dumps are kept local and gitignored
demos/          # Standalone demos (BusyBox, DOOM raycaster, pipeline, meta-compilation)
training/       # Local controller bundles, synthetic corpora, and proof runs (gitignored)
paper/          # Research paper
```

## Tests

```bash
pytest tests/ -v   # 1,544 passed
```

1,544 tests across 21 files: exhaustive formal verification, neural ops, neurOS, compute mode, multi-process, MUXLEQ, BusyBox/Alpine, GPU debugging toolkit, and more.

## Documentation

- **[Wiki](../../wiki)** --- comprehensive documentation (architecture, models, demos, ISA reference)
- **[Research Paper](paper/ncpu_paper.md)** --- detailed analysis and findings
- **[Model Index](models/MODEL_INDEX.md)** --- complete trained model inventory
- **[Rust Metal Kernel](docs/rust_metal_kernel.md)** --- architecture, zero-copy design, build instructions
- **[Compilation Pipeline](docs/compilation_pipeline.md)** --- end-to-end C-to-GPU flow
- **[GPU Debugging Toolkit](docs/gpu_debugging_toolkit.md)** --- the 26-command GPU-native "super debugger"
- **[SOME Complete Guide](docs/SOME_COMPLETE_GUIDE.md)** --- hidden controller, fast weights, latent heads, and training pipeline
- **[Weight CPU Architecture](docs/WEIGHT_CPU_ARCHITECTURE.md)** --- the "CPU in the model" roadmap and current prototype boundaries
- **[SOME Results](docs/SOME_RESULTS.md)** --- benchmark evidence and latent-memory proof summary

## License

MIT
