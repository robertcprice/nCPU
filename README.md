<p align="center">
  <img src="assets/logo.png" alt="nCPU" width="400">
</p>

<p align="center">
  <strong>An end-to-end AI computer. Every layer --- from arithmetic to OS to compiler --- is either a trained neural network or runs entirely on GPU.</strong><br>
  The AI doesn't run <em>on</em> a computer. The AI <em>is</em> the computer.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/interactive-differentiable%20program%20discovery-brightgreen" alt="Interactive discovery">
  <img src="https://img.shields.io/badge/models-24%20trained-blue" alt="Models">
  <img src="https://img.shields.io/badge/accuracy-100%25%20integer-green" alt="Accuracy">
  <img src="https://img.shields.io/badge/verified-exhaustive-blueviolet" alt="Verified">
  <img src="https://img.shields.io/badge/coprocessor-11%20model%20sweep-orange" alt="Coprocessor">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License">
</p>

---

## Start in 60 Seconds

nCPU is most compelling when you treat it as a program-by-examples and text-by-examples machine first, then explore the deeper GPU and coprocessor stack.

```bash
# Best first-time install
pip install -e ".[demo,dev]"

# See the guided demo map
python -m ncpu.lab demos --verbose

# Flagship interactive experiences
python -m ncpu.lab discover
python -m ncpu.lab text --interactive
```

What works today:

| Experience | Status | Best platform |
|------------|--------|---------------|
| Interactive program discovery | Ready now | Cross-platform |
| Neural text machine | Ready now | Cross-platform |
| GPU BusyBox / Alpine demos | Ready now | macOS / Apple Silicon |
| Coprocessor demo | Available with heavier deps | Cross-platform with model stack |

Recommended path:
1. Discover a program from examples
2. Discover a text transform or cipher
3. Try the GPU systems demos
4. Explore the coprocessor and deeper research modules

Tiny terminal preview:

```text
$ python -m ncpu.lab discover
ncpu> preset fib
ncpu> synthesize
ncpu> summary
ncpu> test 13, 21

$ python -m ncpu.lab text --interactive
text> cipher hello khoor
text> summary
text> apply world
```

Further guides:
- `demos/README.md` — curated demo map and starter transcripts
- `docs/REPO_HYGIENE.md` — what should stay in git vs stay local
- `docs/MAINTAINER_CLEANUP_CHECKLIST.md` — pre-push cleanup checklist

## Four Big Ideas

### 1. A Fully Differentiable CPU

Every ALU operation is a trained neural network --- addition, subtraction, multiplication, bitwise, shifts, division. Because the entire computation graph is differentiable, you can **backpropagate through execution**: optimizing programs via gradient descent, discovering better algorithms, tuning instruction schedules. No conventional CPU can do this. The trained neural ALU achieves 100% accuracy on 32-bit integer arithmetic, exhaustively verified over every possible input --- not sampled, *proven*.

### 2. A Complete AI Computer --- Fully Differentiable from Source Code to Execution

Not "AI running on a computer" --- an AI that **is** the computer, end to end, and every layer supports gradient flow. The neural ALU computes. The neural OS (neurOS) manages memory, schedules processes, compiles code --- 11 trained models, zero fallbacks, 93.7--100% accuracy. The full pipeline is differentiable: **source code -> neural compiler -> neural assembler -> neural CPU -> result**, all through trained models. This means you can optimize not just programs but the *OS itself* via gradient descent.

### 3. GPU as Self-Sufficient Computer

A single GPU chip running an entire computer --- no CPU required beyond initial bootstrap. The Metal compute shader executes ARM64 natively at 1.9M+ IPS, boots a multi-process UNIX OS with fork/pipe/wait, compiles C, loads and runs real Linux ELF binaries (BusyBox/Alpine Linux), and even runs a 2-instruction Turing-complete VM (MUXLEQ) that boots eForth. The GPU isn't an accelerator here. It's the whole machine --- complete with a self-hosting C compiler, 13+ compiled applications, and debugging tools impossible on conventional hardware.

### 4. Teaching Transformers to Compute --- The Differentiable Coprocessor

nCPU's trained neural ALU can be **injected directly into any transformer's forward pass** as a differentiable coprocessor. The coprocessor replaces MLP sublayers with a routed mixture: a learned per-token gate decides whether each token flows through the original MLP or through nCPU's neural ALU. Neural truth tables provide differentiable logic (AND/OR/XOR) via bilinear soft indexing, tensor ops provide differentiable arithmetic (ADD/SUB/MUL), and a confidence-aware gating mechanism modulates routing based on the model's own uncertainty. The entire path --- including the discrete logic operations --- supports gradient flow, so the transformer learns *when* to use the coprocessor through standard backpropagation.

An 11-model scaling sweep across the Qwen 2.5/3/3.5 families demonstrates the effect:

| Model | Synthetic Arithmetic Gain | Best Result |
|-------|--------------------------|-------------|
| Qwen3.5-2B (instruct) | 14.5% -> **71.0%** (+56.5%) | Best overall |
| Qwen3.5-2B (base) | 15.5% -> **63.0%** (+47.5%) | 100% on ADD/SUB/MUL/DIV |
| Qwen3.5-4B | +51.0% delta | Largest base sweep gain (tied) |
| Qwen3.5-9B | +51.0% delta | Largest base sweep gain (tied) |
| Qwen3.5-9B (instruct) | 8.0% -> **58.5%** (+50.5%) | |

Real-world transfer on matched models (Qwen3.5-2B): coding preserved (60%), reasoning improved (0% -> 10%), +5% average with no degradation.

> See the [research paper](paper/ncpu_paper.md), the standalone [GPU debugging toolkit paper draft](paper/gpu_debugging_toolkit_paper.md), and the [wiki](../../wiki) for detailed analysis.

## Three CPU Modes

nCPU provides three complete execution modes --- each a different point in the design space, each fully functional:

| Mode | What Runs | Backend | Differentiable? | Speed |
|------|-----------|---------|-----------------|-------|
| **Neural** | 13 trained `.pt` models | PyTorch on GPU | **Yes** --- full gradient flow through every operation | ~5K IPS |
| **Fast** | Native tensor ops | PyTorch tensors | **Yes** --- standard autograd | ~5K IPS |
| **Compute** | Rust + Metal shader | Apple Silicon GPU | No (discrete hardware) | **~1.9M IPS** |

**Neural mode** is the research core: every arithmetic operation, every OS decision, every compiler pass flows through trained neural networks. Addition uses a Kogge-Stone carry-lookahead adder built from neural full adders (8 passes). Multiplication uses a 256x256 byte-pair lookup tensor. Bitwise logic uses learned truth tables. The entire pipeline from source assembly to computed result is differentiable.

**Fast mode** skips the trained models and uses native PyTorch tensor operations for the same ISA --- same differentiability guarantees, without the overhead of model inference. Useful for rapid prototyping and as a correctness oracle.

**Compute mode** is the performance path: a Rust + Metal kernel executes ~200 ARM64 instructions (integer + floating-point) on the GPU at ~1.9M IPS with zero-copy StorageModeShared memory. This is where the UNIX OS boots, the compiler self-hosts, BusyBox runs, and Alpine Linux comes alive. ~500x faster compilation than the Python path.

All three modes execute the same programs and produce the same results. The neural and fast modes are fully differentiable; the compute mode trades gradient flow for raw speed.

## Quick Start

Install paths:

```bash
# Best first-time install for the flagship interactive demos
pip install -e ".[demo,dev]"

# Broader local environment for coprocessor / training work
pip install -e ".[demo,model,train,dev]"
```

First commands to try:

```bash
# Unified launcher
python -m ncpu.lab demos
python -m ncpu.lab discover
python -m ncpu.lab text --interactive

# Direct demo entrypoints
PYTHONPATH=. python demos/showcase/interactive_discovery.py
PYTHONPATH=. python demos/showcase/neural_text_machine.py --interactive

# Neural mode --- all arithmetic through trained neural networks
python main.py --program programs/arithmetic/fibonacci.asm

# GPU compute mode --- Metal shader, ~1.9M IPS
python main.py --program programs/arithmetic/fibonacci.asm --compute

# GPU UNIX OS --- 25-command shell with fork/pipe/wait on Metal
python ncpu/os/gpu/demo.py --multiproc

# Run real BusyBox on the GPU
python demos/gpu/busybox_gpu_demo.py --interactive

# Alpine Linux on GPU
python demos/gpu/alpine_gpu.py --demo

# Rust-native launcher --- standalone Rust path (ELF or boot image)
cd kernels/rust_metal
cargo run --bin ncpu_run -- --elf ../../demos/gpu/busybox.elf --rootfs -- echo hello
cargo run --bin ncpu_run -- --elf ../../demos/gpu/busybox.elf --inspect --json-report
cargo run --bin ncpu_run -- ../../path/to/image.bin

# Benchmark mode --- run 3x with aggregate statistics
cargo run --bin ncpu_run -- --elf ../../demos/gpu/busybox.elf --benchmark --rootfs -- echo hello
# Custom repeat count with JSON aggregate output
cargo run --bin ncpu_run -- --elf ../../demos/gpu/busybox.elf --repeat 10 --json-report --rootfs -- echo hello

# Differentiable coprocessor --- inject nCPU into a transformer
python ncpu/coprocessor/train.py  # Train on synthetic arithmetic + GSM8K
```

`cargo check --bin ncpu_run` currently passes in this workspace. Direct `cargo run` is still subject to the local PyO3/Python link environment.

## The Full Stack

| Layer | Implementation | What It Proves |
|-------|---------------|----------------|
| **ALU** | 13 trained `.pt` models (neural) or native tensor ops (fast) | Neural nets do exact 32-bit integer arithmetic --- exhaustively verified, 100% accuracy |
| **OS** | 11 neural models (neurOS), zero fallbacks | Learned MMU, TLB, cache, scheduler, assembler, compiler --- the OS is differentiable |
| **GPU Compute** | Rust Metal kernel, ~200 ARM64 insns (int + FP) | GPU executes arbitrary programs at ~1.9M IPS, zero-copy StorageModeShared |
| **UNIX OS** | Compiled C on Metal | Fork/pipe/wait, 25-command shell, 28 syscalls, multi-process |
| **Compiler** | cc.c, ~4,200 lines, self-hosting on GPU | GPU hosts a complete toolchain; compiler compiles itself then compiles and runs programs |
| **ELF Loader** | Real Linux binaries on GPU | BusyBox (264KB) and Alpine Linux v3.20 run on Metal |
| **Coprocessor** | nCPU ALU injected into transformer forward pass | Transformers learn to route tokens through neural arithmetic --- +56.5% on best model |
| **MUXLEQ** | 2-instruction Turing-complete VM | If neural nets handle 2 instructions exactly, the principle is universal |
| **Program Optimization** | Backprop through execution (ncpu/differentiable/) | Gradient descent optimizes programs, discovers algorithms, learns ISAs |
| **Self-Modifying Programs** | Differentiable self-modification (ncpu/differentiable/) | Programs rewrite own instructions during execution with gradient flow |
| **Diff Compiler** | Neural Transformer compiler (ncpu/differentiable/) | Source code -> compilation -> execution, end-to-end differentiable |
| **Constant-Time Crypto** | Provably secure AES-128 (ncpu/crypto/) | sigma=0.0 timing; FIPS 197 + NIST SP 800-38A verified |
| **Multi-GPU** | Distributed cores with shared memory (ncpu/distributed/) | Fork/pipe/wait across GPUs; parallel + pipeline execution |
| **SOME** | Hidden controller with latent heads and fast weights | Self-optimizing inference: HumanEval+ and BigCodeBench-Hard improvements |

```python
# Neural mode --- every operation is a trained model
from ncpu.model import CPU
cpu = CPU(neural_execution=True)
cpu.load_program("MOV R0, 7\nMOV R1, 6\nMUL R2, R0, R1\nHALT")
cpu.run()
print(cpu.get_register("R2"))  # 42 --- computed by neural byte-pair LUT

# GPU compute mode --- same program, ~1.9M IPS on Metal
from kernels.mlx.ncpu_kernel import NCPUComputeKernel
kernel = NCPUComputeKernel()
kernel.load_program_from_asm("MOV R0, 7\nMOV R1, 6\nMUL R2, R0, R1\nHALT")
result = kernel.execute()

# Differentiable coprocessor --- inject nCPU into any Hugging Face model
from ncpu.coprocessor import inject_ncpu_coprocessor, NCPUCoprocessorConfig
config = NCPUCoprocessorConfig(confidence_aware=True, deterministic_alu=True)
inject_ncpu_coprocessor(model, config)  # Model now routes tokens through neural ALU
```

## The Differentiable Coprocessor

The coprocessor (`ncpu/coprocessor/`) embeds nCPU's neural ALU inside a transformer as a routed expert. Key innovations:

- **Bilinear soft truth table indexing**: nCPU's hard-indexed truth tables (zero gradient) are replaced with soft bilinear interpolation that provides gradients through both input bits AND truth table parameters
- **Straight-through estimators**: Hard thresholds in bit decomposition use STE for gradient flow through discrete operations
- **Confidence-aware gating**: An MLP variance-based uncertainty signal modulates the coprocessor gate --- high model confidence means less coprocessor intervention, preventing disruption of already-correct predictions
- **Deterministic ALU mode**: Exact integer arithmetic via STE (100% correctness, fully differentiable) --- the neural ALU computes the exact right answer while still supporting backpropagation
- **Per-layer gate scaling**: Later transformer layers can receive less coprocessor influence, respecting the model's own learned representations

The coprocessor is trained with the transformer backbone frozen --- only the router, projection layers, and coprocessor internals update. This means you can add differentiable arithmetic to any pretrained language model without catastrophic forgetting.

**Training data**: synthetic arithmetic expressions, 22K GSM8K-extracted math problems, MATH dataset problems.

## Gradient-Based Program Optimization (NEW)

The `ncpu/differentiable/` package delivers on the central promise of a differentiable CPU: **backpropagating through program execution** to optimize programs, discover algorithms, and learn instruction sets via gradient descent.

### Program Optimization --- Backprop Through Execution

Given a fixed program structure, gradient descent finds the parameter values that produce a desired output. Gradients flow backward through every instruction:

```python
from ncpu.differentiable import DifferentiableEngine, ProgramOptimizer

engine = DifferentiableEngine()
prog = engine.assemble("MOV R1, #3\nMUL R2, R0, R1\nHALT")

# Question: what value of R0 makes R0 * 3 = 42?
opt = ProgramOptimizer(engine)
result = opt.optimize_inputs(prog, target_registers={2: 42.0},
                             input_registers=[0], initial_values={0: 1.0})
# Gradient descent discovers R0 = 14.0 in ~34 steps
```

**Polynomial fitting** also works: `f(x) = ax^2 + bx + c` with unknown coefficients is assembled as a program, and gradient descent through execution discovers a=2, b=3, c=5 to fit target points --- verified on held-out inputs.

### Program Synthesis --- Discover Algorithms via Gradient Descent

Programs are represented as continuous parameters (Gumbel-softmax over opcodes, soft attention over registers). Gradient descent searches program space by backpropagating through the differentiable CPU:

```python
from ncpu.differentiable import ProgramSynthesizer, SynthesisSpec

# Specification: I want a program where R2 = R0 + R1
spec = SynthesisSpec(examples=[
    ({0: 3.0, 1: 5.0}, {2: 8.0}),
    ({0: 7.0, 1: 2.0}, {2: 9.0}),
    # ... more examples
])
synth = ProgramSynthesizer(max_program_len=6)
result = synth.synthesize(spec, max_iters=2000, verbose=True)
# Gradient descent discovers: ADD R2, R0, R1; HALT
```

Temperature annealing transitions from soft/exploratory to hard/discrete, converging on an actual executable program. Length regularization biases toward shorter programs.

### Neural ISA Discovery --- Learn Instruction Sets

Instead of implementing ARM64, gradient descent **discovers the optimal instruction set** from benchmark pressure:

```python
from ncpu.differentiable import NeuralISADiscovery

isa = NeuralISADiscovery()
result = isa.discover([arithmetic_benchmark, bitwise_benchmark])
# Op0 learns addition, Op1 learns multiplication, Op2 learns subtraction
# Each operation's "cost" is a learned parameter --- gradient descent finds
# the cheapest set of operations that achieves correctness
```

### Neural Floating-Point ALU

Extends the integer neural ALU to IEEE 754 floating-point. Each operation (FADD, FMUL, FDIV, FSQRT) is a dedicated neural network trained to match ground-truth arithmetic:

```python
from ncpu.differentiable import NeuralFloatALU

alu = NeuralFloatALU(hidden_dim=128)
alu.train_from_ground_truth("add", epochs=200)   # Learns addition
alu.train_from_ground_truth("mul", epochs=300)   # Learns multiplication
# Fully differentiable: gradients flow through every float operation
```

**28 tests passing** across gradient flow verification, optimization convergence, synthesis, ISA discovery, and float ALU training.

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

**What makes this compiler special**: it runs on a GPU compute shader, not a CPU. Every instruction executes deterministically (sigma=0.0 cycle variance), meaning the compilation process itself is immune to timing side-channels. The GPU retains complete execution state after every compilation --- you can inspect every instruction the compiler executed, set breakpoints inside the compiler's own code, replay compilation runs bit-identically, and diff two compilations instruction-by-instruction. No CPU-hosted compiler can offer any of this.

Supports 18 C features: structs (`.`/`->`), pointers, arrays, recursion, for/while/do-while, ternary, sizeof, compound assignment, bitwise ops, short-circuit `&&`/`||`, `enum`, `typedef`, `switch`/`case`/`default`, `#ifdef`/`#ifndef`/`#elif`/`#endif`, global initializers, function pointers, `union`, function-like macros, `goto`/labels, multi-dimensional arrays. **73/73 test programs verified, 18 bugs fixed, full self-compilation verified.** 8 meta-compilation programs verified (stack evaluator, ARM64 encoder, Ackermann A(3,4)=125, matrix determinant, prime sieve, Towers of Hanoi, DJB2 hash, Collatz).

### BusyBox on Metal GPU

Real BusyBox (Alpine Linux core utils, 264KB static binary) running on the Metal GPU shader via an ELF64 loader:

- Cross-compiled with `aarch64-linux-musl-gcc -static`
- ELF64 parser loads PT_LOAD segments, sets up Linux stack (argc/argv/envp/auxv)
- 50+ Linux syscalls handled: exit, read, write, brk, mmap, ioctl, writev, uname, symlink, etc.
- 34+ verified commands: echo, uname, cat, ls, printf, basename, dirname, head, tail, wc, cut, sort, uniq, grep, expr, touch, mkdir, rm, cp, stat, mv, chmod, sleep, tr, find, tee, readlink, ln, and more
- GPUFilesystem wired via syscalls --- `cat /etc/motd` reads from Python-side filesystem

### Alpine Linux on Metal GPU

Full Alpine Linux v3.20 distribution running on Metal GPU compute shader with a comprehensive POSIX shell:

- BusyBox (264KB, musl libc) as multi-call binary behind every command
- Pipes (`|`), chaining (`;`/`&&`/`||`), redirection (`>`/`>>`), command substitution (`$(cmd)`)
- Shell scripting: for/while/if/elif/case, functions, local variables, parameter expansion, brace expansion
- Here-documents, glob expansion, aliases, history, 35+ builtins
- 109-file Alpine rootfs with /proc, /dev, /etc, init stubs, user databases, package manager
- 26 novel GPU superpower commands spanning post-mortem forensics, replay/diff, state snapshots, tracing, breakpoints, watchpoints, profiling, disassembly, sanitization, fuzzing, reverse data flow, constant-time verification, memory visualization, and entropy analysis. Full reference: [GPU debugging toolkit](docs/gpu/gpu_debugging_toolkit.md)

### Rust Metal Kernel

The primary execution backend: Rust + Metal with StorageModeShared for zero-copy GPU<->Python communication. [Architecture docs](docs/gpu/rust_metal_kernel.md).

- **~200 ARM64 instructions** (integer + floating-point), ~1.9M IPS sustained
- **Floating-point support**: single-precision FADD, FSUB, FMUL, FDIV, FSQRT, FABS, FNEG, FMADD, FMSUB, FCMP, FCSEL, FRINT*, FMAX, FMIN, SCVTF, UCVTF, FCVTZS, FCVTZU, FMOV, FCVT, plus all FP load/store addressing modes. Double-precision (D-register) instructions decode and execute but operate at single-precision accuracy — Apple Silicon GPU has no FP64 hardware.
- **~500x faster compilation** than the Python MLX kernel (~44ms vs ~22s)
- **Zero-copy SVC handling** via unified memory (no 16MB copies per syscall)
- GPU-side SVC buffer for SYS_WRITE, SYS_BRK, SYS_CLOSE, SYS_EXIT
- **GPU-native debugging toolkit** (26 commands: trace, breakpoints, watchpoints, disassembler, sanitizer, fuzzer, reverse analysis, constant-time verification, and more)
- Built with `maturin develop --release`, exposed to Python via PyO3
- Rust-native runtime modules for boot-image loading, ELF loading, VFS/rootfs, syscall handling, native ABI experiments, and standalone launching
- The live standalone launcher path includes `ProcessManager`-backed scheduling, fork/wait/pipe/dup/exec, and Linux `clone(220)` interception

### GPU-Native Debugging Toolkit

A debugging platform **impossible on conventional CPUs**. The Metal kernel provides a verified 26-command toolkit:

- **Instruction tracing**: 4096-entry circular buffer capturing PC, instruction word, x0-x3, NZCV flags, and SP
- **Breakpoints & watchpoints**: Up to 4 each, checked every GPU cycle at zero overhead; conditional breakpoints fire on PC + register value match
- **Time-travel debugging**: Browse instruction-by-instruction execution history with register/flag diffs
- **Deterministic replay**: Bit-identical execution (sigma=0.0000) --- every run reproduces exactly
- **Memory sanitizer**: Zero-overhead memory safety checking (vs ASan's 2-5x CPU overhead)
- **Automated fuzzing**: Crash detection with instant post-mortem traces (no reproduction needed)
- **Reverse data flow**: Trace backwards to find where a value originated
- **Constant-time verification**: Exact verification of constant-time crypto (impossible on noisy CPUs)
- Full reference: [GPU debugging toolkit](docs/gpu/gpu_debugging_toolkit.md). Paper draft: [GPU debugging toolkit paper](paper/gpu_debugging_toolkit_paper.md)

**Why this matters**: On a CPU, process state is destroyed after exit, breakpoints require ptrace overhead, watchpoints are limited by hardware debug registers, and non-deterministic microarchitecture prevents replay. On GPU, ALL execution state persists, breakpoints and watchpoints are free, and every run is deterministic.

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

A minimal proof of universality: SUBLEQ + MUX running on nCPU in three modes (neural, fast, compute). Loads `.dec` images, boots eForth. Neural mode: SUB via Kogge-Stone CLA (~248us), MUX via neural AND/OR/NOT (~63us). If neural nets exactly execute a 2-instruction OISC, the principle extends to any instruction set.

### neurOS: Fully Neural Operating System

Every OS component is a trained neural network --- 11 models, zero fallbacks. The entire pipeline is differentiable: source code passes through the neural compiler, then the neural assembler, then executes on the neural CPU.

| Component | Accuracy | Component | Accuracy |
|-----------|----------|-----------|----------|
| MMU | 100% | Assembler codegen | **100%** |
| TLB | 99.6% | Assembler tokenizer | 99.4% |
| Cache | 99.7% | Compiler optimizer | 95.2% |
| Scheduler | 99.2% | Watchdog | 100% |
| Prefetch | 97.8% | Block allocator | 98.4% |

Self-compilation verified: nsl source -> neural compiler -> neural assembler -> neural CPU -> correct results.

### Timing Side-Channel Immunity

GPU execution produces **zero cycle-count variance** (sigma=0.0 across 270 runs). Same code on native Apple Silicon shows 47-73% timing variance. AES-128 T-table attacks are structurally impossible --- no data cache, no cache lines, no cache-miss penalty. This is a security property that is architecturally impossible on conventional CPUs.

### Provably Constant-Time Cryptographic Library

Built on the GPU's timing immunity, `ncpu/crypto/` provides a complete constant-time AES-128 implementation (ECB + CBC) where every operation is provably free of timing side-channels:

- **19 constant-time primitives**: ct_select, ct_equal, ct_xor, ct_byte_lookup (full 256-entry table scan), ct_memcmp, ct_swap, ct_rotate, etc. --- no data-dependent branches anywhere
- **AES-128**: SubBytes via full S-box scan (not indexed lookup), MixColumns via algebraic GF(2^8) (no T-tables), ShiftRows as fixed permutation
- **FIPS 197 and NIST SP 800-38A test vectors**: all passing (ECB + CBC)
- **Timing verification framework**: measures cycle counts across diverse inputs, verifies sigma=0.0, generates formal verification reports
- **88 tests passing** including FIPS vectors, NIST vectors, avalanche effect, key sensitivity, full S-box/inverse-S-box verification

### Self-Modifying Differentiable Programs

Programs that rewrite their own instruction memory during execution, with gradients flowing through the self-modification (`ncpu/differentiable/self_modifying.py`):

- **STORE_INST**: soft-write new instruction content at a target position via Gaussian attention + learned projection (register values -> instruction features)
- **LOAD_INST**: soft-read instruction identity into registers
- Gradient descent optimizes not just what a program computes, but **how it modifies itself** during execution
- This is the differentiable analogue of self-modifying machine code, with full gradient flow

### Differentiable Compilation Pipeline

End-to-end gradient flow from source code through compilation to execution (`ncpu/differentiable/diff_compiler.py`):

```
Source tokens → Neural Compiler (Transformer) → SoftProgram → Execution → Loss
                     ↑ gradients flow all the way back ↑
```

- **DifferentiableCompiler**: Transformer encoder + linear decoder heads mapping source tokens to instruction sequences
- The compiler learns instruction encodings from scratch --- purely from execution feedback, no supervised instruction labels
- Training verified: loss converges, compiled programs produce correct outputs on test inputs

### Multi-GPU Distributed nCPU

Multiple GPUs as multiple cores of a single neural computer (`ncpu/distributed/`):

- **GPUCore**: each core wraps its own DifferentiableEngine with private registers, PC, flags, local memory
- **Parallel execution**: independent programs run on multiple cores simultaneously
- **Pipeline execution**: staged dataflow where core N's output feeds core N+1's input
- **Fork/pipe**: UNIX process model across GPU boundaries (clone state, create channels)
- **Shared memory**: inter-core communication with atomic operations (add, compare-and-swap)
- **Distributed scheduler**: round-robin, load-balanced, and affinity-based policies
- **Device-aware dispatch**: automatic CPU/MPS/CUDA discovery, round-robin or mirrored core assignment, rebalance/report APIs
- **78 tests passing** across the distributed execution suite

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

## Self-Optimizing Machine Engine (SOME)

SOME is the repo's execution-grounded code and reasoning stack --- a self-improving hidden controller that tries to turn part of the neural machine into an internal coprocessor for code generation and reasoning:

- **Buffered hidden controller**: `think -> write -> verify -> patch -> commit`, with only committed output exposed
- **Latent control heads**: learned latent action, halt, descriptor, state-patch, and recurrent memory heads
- **Task-local fast weights**: descriptor-driven per-task weight updates during inference
- **Segmented decode path**: recent exact token window plus compressed committed-history descriptors
- **Trajectory-first training loop**: hidden-controller trajectories feed SFT, latent-head, and patch-head training

Current evidence:

- **HumanEval+**: `qwen3.5:4b 147 -> 154`, `9b 144 -> 156`, `27b 153 -> 156`
- **BigCodeBench-Hard**: `qwen3.5:9b 33 -> 49`
- **Latent-memory proof**: learned memory head improved validation MSE by `83.26%` over zero-delta baseline

Docs: [SOME Complete Guide](docs/some/SOME_COMPLETE_GUIDE.md) | [Architecture](docs/some/SOME_ARCHITECTURE.md) | [Weight CPU Architecture](docs/some/WEIGHT_CPU_ARCHITECTURE.md) | [Results](docs/some/SOME_RESULTS.md)

### 5. Differentiable Execution as a Training Signal for Code Models

The `ncpu/execution_training/` package makes nCPU's differentiable CPU a **training signal source** for language models. Instead of sparse pass/fail rewards from external execution, the model receives dense, per-operation gradient signal from running its code through the differentiable engine.

```python
# Parse Python → nCPU ISA → execute differentiably → backprop execution error
from ncpu.execution_training import CodeToISAParser, ExecutionLoss
from ncpu.differentiable import DifferentiableEngine

parser = CodeToISAParser()
engine = DifferentiableEngine()
loss_fn = ExecutionLoss(engine=engine)

result = parser.parse_block("result = a * b + c", arg_names=["a", "b", "c"])
soft_prog = result.to_soft_program()
exec_result = loss_fn.compute_soft(soft_prog, inputs={0: 3, 1: 5, 2: 2}, expected={3: 17.0})
exec_result.total_loss.backward()  # Gradients through every ALU operation!
```

Three training modes:

| Mode | Method | Gradient Source |
|------|--------|----------------|
| **Coprocessor + Execution Loss** | Parse reference code, execute, add to LM loss | Per-operation MSE through differentiable engine |
| **Differentiable Compilation** | Map LM hidden states → DiffCompiler → execution | End-to-end: execution → compilation → embeddings |
| **Generated Code Training** | Model generates code, parse, execute, REINFORCE | Execution rewards + optional policy gradient |

96 tests passing across the full pipeline. See [architecture doc](docs/mog/DIFFERENTIABLE_EXECUTION_TRAINING.md) and the [module README](ncpu/execution_training/README.md).

```bash
# Smoke test (no model needed)
python -m ncpu.execution_training.train --synthetic-only --steps 200

# Full training
python -m ncpu.execution_training.train --model Qwen/Qwen3.5-0.8B --steps 2000

# Scaling sweep
python -m ncpu.execution_training.run_sweep --quick
```

## Project Structure

```
ncpu/
  differentiable/ # Differentiable execution, program optimization, synthesis,
                  # ISA discovery, float ALU, self-modifying programs, diff compiler
  coprocessor/  # Differentiable coprocessor: inject nCPU into transformer forward passes
  execution_training/  # Differentiable execution as training signal for code LMs (3 modes, 96 tests)
  crypto/       # Provably constant-time crypto (AES-128 ECB/CBC, timing verification)
  distributed/  # Multi-GPU distributed nCPU (cores, shared memory, scheduler)
  os/
    neuros/     # Neural OS: 17 modules (MMU, TLB, cache, scheduler, compiler, ...)
    gpu/        # GPU UNIX OS: runner, filesystem, shell, ELF loader
      src/      # C source (shell, libc, syscalls, linker script)
      programs/ # Compiled C apps (crypto, games, vms, net, nn, tools, graphics)
  self_optimizing/  # SOME runtime, hidden controller, fast weights, benchmark stack
  neural/       # NeuralCPU: 12K-line CPU with neural ALU bridge (differentiable)
  model/        # Model-based CPU (neural_ops, assembler, architectures)
  tensor/       # Tensor-based ARM64 emulator (differentiable, no trained models)
kernels/
  mlx/          # Metal compute kernels (ARM64 V2 + nCPU ISA + MUXLEQ)
  rust_metal/   # Rust + Metal ARM64 kernel (primary backend, ~500x faster)
models/         # 24 trained .pt models (alu, shifts, math, os, decode)
programs/       # 62 assembly programs
tests/          # ~1,840 tests across 25+ files
benchmarks/     # Benchmark scripts; generated result dumps are kept local and gitignored
demos/          # Standalone demos (BusyBox, Alpine, DOOM raycaster, meta-compilation)
training_results/  # Coprocessor scaling sweeps, ablation studies, instruct sweeps
paper/          # Research paper + new section on differentiable programs
```

## Tests

```bash
pytest tests/ -v   # ~1,840 passed
```

~1,840 tests across 25+ files: exhaustive formal verification, neural ops, neurOS, compute mode, multi-process, MUXLEQ, BusyBox/Alpine, GPU debugging toolkit, coprocessor, differentiable execution (36), constant-time crypto (88), self-modifying programs + diff compiler (99), multi-GPU distributed (74), and more.

## Documentation

- **[Wiki](../../wiki)** --- comprehensive documentation (architecture, models, demos, ISA reference)
- **[Research Paper](paper/ncpu_paper.md)** --- detailed analysis and findings
- **[Model Index](models/MODEL_INDEX.md)** --- complete trained model inventory
- **[Rust Metal Kernel](docs/gpu/rust_metal_kernel.md)** --- architecture, zero-copy design, build instructions
- **[Compilation Pipeline](docs/gpu/compilation_pipeline.md)** --- end-to-end C-to-GPU flow
- **[GPU Debugging Toolkit](docs/gpu/gpu_debugging_toolkit.md)** --- the 26-command GPU-native "super debugger"
- **[SOME Complete Guide](docs/some/SOME_COMPLETE_GUIDE.md)** --- hidden controller, fast weights, latent heads, and training pipeline
- **[Weight CPU Architecture](docs/some/WEIGHT_CPU_ARCHITECTURE.md)** --- the "CPU in the model" roadmap and current prototype boundaries
- **[SOME Results](docs/some/SOME_RESULTS.md)** --- benchmark evidence and latent-memory proof summary
- **[Differentiable Programs](paper/section_differentiable_programs.md)** --- paper section on program optimization, synthesis, ISA discovery
- **[Differentiable Execution Training](docs/mog/DIFFERENTIABLE_EXECUTION_TRAINING.md)** --- architecture doc for execution-grounded code model training
- **[Execution Training Paper Section](paper/section_execution_training.md)** --- paper section on dense execution gradients for code LMs

## License

MIT
