<p align="center">
  <img src="assets/logo.png" alt="nCPU" width="400">
</p>

<p align="center">
  <strong>An end-to-end AI computer. Every layer --- from arithmetic to OS to compiler --- is either a trained neural network or runs entirely on GPU.</strong><br>
  The AI doesn't run <em>on</em> a computer. The AI <em>is</em> the computer.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/tests-939%20passing-brightgreen" alt="Tests">
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

> See the [research paper](paper/ncpu_paper.md) and [wiki](../../wiki) for detailed analysis.

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
```

## The Stack

| Layer | Implementation | What It Proves |
|-------|---------------|----------------|
| **ALU** | 13 trained `.pt` models | Neural nets do exact integer arithmetic (exhaustively verified) |
| **OS** | 11 neural models (neurOS) | Learned MMU, TLB, cache, scheduler, compiler --- zero fallbacks |
| **Compute** | Metal shader (135+ ARM64 insns) | GPU executes arbitrary programs at ~4M IPS, no CPU needed |
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

A ~3,500-line self-hosting C compiler (`cc.c`) that compiles C source into ARM64 machine code **entirely on the Metal GPU**, then executes the result on the same GPU:

```
Host GCC compiles cc.c -> compiler₀
  GPU runs compiler₀, self-compiles cc.c -> compiler₁
    GPU runs compiler₁, compiles test.c -> binary
      GPU runs test binary -> correct result
```

Supports: structs (`.`/`->`), pointers, arrays, recursion, for/while/do-while, ternary, sizeof, compound assignment, bitwise ops, short-circuit `&&`/`||`, `enum`, `typedef`, `switch`/`case`/`default`, `#ifdef`/`#ifndef`/`#endif`, global initializers, function pointers, `union`. **40/40 test programs verified, 14 bugs fixed, self-compilation verified.**

### BusyBox on Metal GPU

Real BusyBox (Alpine Linux core utils, 264KB static binary) running on the Metal GPU shader via an ELF64 loader:

- Cross-compiled with `aarch64-linux-musl-gcc -static`
- ELF64 parser loads PT_LOAD segments, sets up Linux stack (argc/argv/envp/auxv)
- 28+ Linux syscalls handled: exit, read, write, brk, mmap, ioctl, writev, uname, etc.
- 30+ applets: echo, uname, basename, dirname, cat, ls, grep, printf
- GPUFilesystem wired via syscalls --- `cat /etc/motd` reads from Python-side filesystem

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
  neural/       # NeuralCPU: 12K-line CPU with neural ALU bridge
  model/        # Model-based CPU (neural_ops, assembler, architectures)
  tensor/       # Tensor-based ARM64 emulator
kernels/mlx/    # Metal compute kernels (ARM64 V2 + nCPU ISA + MUXLEQ)
models/         # 24 trained .pt models (alu, shifts, math, os, decode)
programs/       # 62 assembly programs
tests/          # 939 tests across 17 files
benchmarks/     # Neural, neurOS, compute, ARM64, side-channel, multi-process
demos/          # Standalone demos (BusyBox, DOOM raycaster, pipeline, meta-compilation)
paper/          # Research paper
```

## Tests

```bash
pytest tests/ -v   # 939 tests passing
```

939 tests across 17 files: exhaustive formal verification, neural ops, neurOS (258), compute mode (138), multi-process (41), MUXLEQ (32), BusyBox (23), and more.

## Documentation

- **[Wiki](../../wiki)** --- comprehensive documentation (architecture, models, demos, ISA reference)
- **[Research Paper](paper/ncpu_paper.md)** --- detailed analysis and findings
- **[Model Index](models/MODEL_INDEX.md)** --- complete trained model inventory

## License

MIT
