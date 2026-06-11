<p align="center">
  <img src="assets/logo.png" alt="nCPU" width="400">
</p>

<p align="center">
  <strong>A complete neural computer. Every layer — from arithmetic to OS to compiler to display — is a trained neural network or runs entirely on GPU.</strong><br>
  The AI doesn't run <em>on</em> a computer. The AI <em>is</em> the computer.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/interactive-differentiable%20program%20discovery-brightgreen" alt="Interactive discovery">
  <img src="https://img.shields.io/badge/models-24%20trained-blue" alt="Models">
  <img src="https://img.shields.io/badge/accuracy-100%25%20integer-green" alt="Accuracy">
  <img src="https://img.shields.io/badge/coprocessor-11%20model%20sweep-orange" alt="Coprocessor">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License">
</p>

---

## Five Pillars

### 1. The Neural Computer

Every ALU operation — addition, subtraction, multiplication, bitwise logic, shifts, division — is a trained neural network. The neural OS (neurOS) manages memory, schedules processes, compiles code: 11 trained models, zero fallbacks. The neural display renders characters through char→glyph MLPs→ConvNet→pixels (143K params). The full pipeline is differentiable: source code → neural compiler → neural assembler → neural CPU → neural display, all through trained models.

The neural ALU achieves **100% accuracy on 32-bit integer arithmetic**, exhaustively verified over every possible input. Multiplication is 12x faster than addition — inverting the conventional CPU hierarchy because addition needs an 8-pass carry chain while multiplication decomposes into parallel byte-pair lookups.

| Instruction | Strategy | Latency |
|-------------|----------|---------|
| ADD/SUB/CMP | Kogge-Stone CLA (8 passes) | 248 us |
| MUL | Byte-pair LUT (65,536 entries) | 21 us |
| AND/OR/XOR | Vectorized truth table | 21 us |
| SHL/SHR | Attention-based bit routing | 434 us |
| DIV | Restoring division (neural subtraction) | varies |

neurOS accuracy:

| Component | Accuracy | Component | Accuracy |
|-----------|----------|-----------|----------|
| MMU | 100% | Assembler codegen | **100%** |
| TLB | 99.6% | Assembler tokenizer | 99.4% |
| Cache | 99.7% | Compiler optimizer | 95.2% |
| Scheduler | 99.2% | Watchdog | 100% |
| Prefetch | 97.8% | Block allocator | 98.4% |

### 2. The GPU Computer

A complete self-sufficient computer running on a single GPU chip — no CPU required beyond bootstrap. The Rust + Metal kernel executes ~200 ARM64 instructions (integer + floating-point) at **~1.9M IPS** with zero-copy StorageModeShared memory and **zero cycle-count variance** (σ=0.0).

What runs on it:
- **Multi-process UNIX OS**: fork/pipe/wait, 25-command shell, 28 syscalls, up to 15 concurrent processes
- **Self-hosting C compiler**: ~4,200 lines, compiles itself then compiles and runs programs — entirely on GPU
- **Real Linux binaries**: BusyBox (264KB, 34+ commands) and Alpine Linux v3.20 via ELF64 loader
- **13+ compiled C applications**: SHA-256, AES-128, Tetris, Snake, Brainfuck interpreter, Forth REPL, CHIP-8 emulator, HTTP server, MNIST classifier, and more
- **26-command deterministic debugger**: instruction tracing, breakpoints/watchpoints, time-travel, memory sanitizer, automated fuzzing, reverse data flow, constant-time verification — structurally impossible on conventional CPUs

### 3. Differentiable Program Synthesis

Given input/output examples, gradient descent discovers executable programs by backpropagating through the differentiable CPU. Programs are continuous parameters (Gumbel-softmax over opcodes, soft attention over registers) that converge on discrete executable code via temperature annealing.

The Rust synthesizer (`nsynth/`) solves **105/105 benchmark problems** across five solver families:

| Family | Solved | Method |
|--------|--------|--------|
| Gradient | 66/105 | Differentiable search with learned restart bank |
| Enumerative | 21/105 | Bottom-up expression enumeration |
| Search | 13/105 | Single-branch, struct-pair, string teachers |
| Template | 5/105 | Pattern-matching for hardest problems |

Key optimizations: persistent solved-program memoization (5000x on cache hit), learned bias bank with warm-refine cross-problem transfer, emergent constant vocabulary mined from examples.

### 4. The Differentiable Coprocessor

nCPU's neural ALU injected directly into any transformer's forward pass as a routed expert. A learned per-token gate decides whether each token flows through the original MLP or through the neural ALU. Bilinear soft truth tables provide differentiable logic (AND/OR/XOR), tensor ops provide differentiable arithmetic, and confidence-aware gating modulates routing based on model uncertainty.

**11-model scaling sweep** across Qwen 2.5/3/3.5 families:

| Model | Gain | Best Result |
|-------|------|-------------|
| Qwen3.5-2B (instruct) | 14.5% → **71.0%** (+56.5%) | Best overall |
| Qwen3.5-2B (base) | 15.5% → **63.0%** (+47.5%) | 100% on ADD/SUB/MUL/DIV |
| Qwen3.5-4B | +51.0% delta | Largest base gain (tied) |
| Qwen3.5-9B | +51.0% delta | Largest base gain (tied) |

Real-world transfer (Qwen3.5-4B, A100): 62.2% → **64.6%** (+4 problems solved on full HumanEval).

### 5. JEPA Predictive Machine Dynamics

A bottom-up predictive world model of the computer itself. Instead of just executing instructions, a JEPA-style (Joint Embedding Predictive Architecture) network learns to predict machine state transitions in a compressed latent space:

```
latent_state_t + instruction → predictor → latent_state_{t+1}
```

This runs at two levels:
- **Python** (`ncpu/jepa_neural_cpu/`): A watchable demo where real programs execute alongside an untrained JEPA predictor — prediction error becomes a live anomaly/robustness signal
- **Rust Metal** (`kernels/rust_metal/src/jepa/`): 2,858 lines of JEPA neural kernel + neural OS models that observe deterministic GPU execution and actively steer scheduling via learned bias override, Ready demotion, and adaptive de-prio

The JEPA layer sits on top of the exact neural ALU substrate, giving it two properties most world models lack: **perfect ground truth is free** (just run more programs through the engine) and **exact vs. predicted can be mixed at will** (cheap latent speculation for exploration, exact execution when precision matters).

Long-term target: a hierarchical cross-JEPA where bit-level, instruction-level, program-level, and task-level predictors form a complete neural machine whose dynamics are entirely learned.

```bash
python3 -m ncpu.jepa_neural_cpu.demo    # Bottom-up JEPA neural computer demo
python -m ncpu.world_model.quickstart    # JEPA machine world model quickstart
```

---

## Start in 60 Seconds

```bash
pip install -e ".[demo,dev]"

# Hero: GPU as complete computer (macOS / Apple Silicon)
python -m ncpu gpu                 # GPU IS the computer
python -m ncpu gpu --neural-alu    # With neural ALU in Metal shader
python -m ncpu gpu debug           # 26-command deterministic super-debugger

# Cross-platform (no heavy deps)
python -m ncpu discover            # Program by examples via differentiable synthesis
python -m ncpu text --interactive  # Neural text / cipher machine

# Full neural pipeline (with model stack)
python -m ncpu full-neural         # Bottom-up neural CPU + neural display
python -m ncpu meta-compare        # Side-by-side proof demo

# JEPA predictive layer
python3 -m ncpu.jepa_neural_cpu.demo    # JEPA neural computer demo
python -m ncpu.world_model.quickstart   # World model quickstart

# Rust-native standalone (no Python needed)
cd kernels/rust_metal
cargo run --bin ncpu_run -- --elf ../../demos/gpu/busybox.elf --rootfs -- echo hello
```

---

## Three Execution Modes

| Mode | What Runs | Differentiable? | Speed |
|------|-----------|-----------------|-------|
| **Neural** | 13 trained `.pt` models | Yes — full gradient flow | ~5K IPS |
| **Fast** | Native tensor ops | Yes — standard autograd | ~5K IPS |
| **Compute** | Rust + Metal shader | No (discrete hardware) | **~1.9M IPS** |

Neural mode: every operation flows through trained neural networks. Fast mode: native tensors, same ISA, same differentiability. Compute mode: trades gradient flow for raw speed — this is where the UNIX OS boots, the compiler self-hosts, and BusyBox/Alpine run.

All three modes execute the same programs and produce the same results.

```python
# Neural mode — every operation is a trained model
from ncpu.model import CPU
cpu = CPU(neural_execution=True)
cpu.load_program("MOV R0, 7\nMOV R1, 6\nMUL R2, R0, R1\nHALT")
cpu.run()
print(cpu.get_register("R2"))  # 42 — computed by neural byte-pair LUT

# Differentiable coprocessor — inject into any Hugging Face model
from ncpu.coprocessor import inject_ncpu_coprocessor, NCPUCoprocessorConfig
config = NCPUCoprocessorConfig(confidence_aware=True, deterministic_alu=True)
inject_ncpu_coprocessor(model, config)

# Differentiable program synthesis
from ncpu.differentiable import ProgramSynthesizer, SynthesisSpec
spec = SynthesisSpec(examples=[
    ({0: 3.0, 1: 5.0}, {2: 8.0}),
    ({0: 7.0, 1: 2.0}, {2: 9.0}),
])
synth = ProgramSynthesizer(max_program_len=6)
result = synth.synthesize(spec, max_iters=2000)
# Discovers: ADD R2, R0, R1; HALT

# JEPA world model — predict machine state transitions
from ncpu.world_model.je_world_model import JEWorldModel, JEWMConfig
model = JEWorldModel(JEWMConfig(state_dim=22, action_dim=8))
pred = model.predict_next_latent(model.encode_state(state), model.encode_action(action))
```

---

## The Full Stack

| Layer | Implementation | What It Proves |
|-------|---------------|----------------|
| **ALU** | 13 trained `.pt` models | Neural nets do exact 32-bit integer arithmetic — exhaustively verified |
| **OS** | neurOS — 11 neural models, zero fallbacks | Learned MMU, TLB, cache, scheduler, compiler — the OS is differentiable |
| **GPU Compute** | Rust Metal kernel, ~200 ARM64 insns | GPU executes arbitrary programs at ~1.9M IPS |
| **UNIX OS** | Compiled C on Metal | Fork/pipe/wait, 25-command shell, 28 syscalls |
| **Compiler** | cc.c, ~4,200 lines, self-hosting | Compiles itself then compiles programs — on GPU |
| **ELF Loader** | Real Linux binaries on GPU | BusyBox + Alpine Linux v3.20 run on Metal |
| **Coprocessor** | nCPU ALU in transformer forward pass | Transformers learn to route tokens through neural arithmetic |
| **JEPA** | Predictive world model of machine dynamics | Fast latent speculation + anomaly detection on top of exact substrate |
| **Program Synthesis** | Backprop through execution | Gradient descent discovers programs from I/O examples |
| **Constant-Time Crypto** | AES-128 ECB/CBC (ncpu/crypto/) | σ=0.0 timing; FIPS 197 + NIST SP 800-38A verified |
| **Multi-GPU** | Distributed cores with shared memory | Fork/pipe/wait across GPUs; parallel + pipeline execution |
| **SOME** | Hidden controller with latent heads | Self-optimizing inference: HumanEval+ and BigCodeBench improvements |

---

## Timing Side-Channel Immunity

GPU execution produces **zero cycle-count variance** (σ=0.0 across 270 runs). Same code on native Apple Silicon shows 47-73% timing variance. AES-128 T-table attacks are structurally impossible — no data cache, no cache lines, no cache-miss penalty.

Built on this: `ncpu/crypto/` provides provably constant-time AES-128 (ECB + CBC) with 19 constant-time primitives, FIPS 197 and NIST SP 800-38A test vectors all passing.

---

## Self-Optimizing Machine Engine (SOME)

A hidden controller that turns part of the neural machine into an internal coprocessor for code generation and reasoning:

- **Buffered hidden controller**: think → write → verify → patch → commit
- **Latent control heads**: learned action, halt, descriptor, state-patch, and recurrent memory heads
- **Task-local fast weights**: descriptor-driven per-task weight updates during inference
- **Latent-memory proof**: learned memory head improved validation MSE by 83.26% over baseline

Results: HumanEval+ qwen3.5:4b 147→154, 9b 144→156. BigCodeBench-Hard qwen3.5:9b 33→49.

---

## MUXLEQ: Turing-Complete in 2 Instructions

SUBLEQ + MUX running in all three modes. Neural mode: SUB via Kogge-Stone CLA (~248us), MUX via neural AND/OR/NOT (~63us). Loads `.dec` images, boots eForth. If neural nets exactly execute a 2-instruction OISC, the principle extends to any instruction set.

---

## Program Synthesis from Examples (nsynth_codegen)

```bash
cargo build --release --bin nsynth_codegen
./target/release/nsynth_codegen --lang python --examples '{
  "name":"square","signature":"fn square(x: i64) -> i64",
  "examples":[{"inputs":[0],"expected":0},{"inputs":[3],"expected":9}]
}'
# → def square(x: int): return (0 * x * x) + (1 * x * x) + 0
```

---

## Project Structure

```
ncpu/
  differentiable/    # Differentiable execution, program synthesis, ISA discovery
  coprocessor/       # Inject nCPU into transformer forward passes
  execution_training/# Differentiable execution as training signal for code LMs
  crypto/            # Provably constant-time crypto (AES-128)
  distributed/       # Multi-GPU distributed execution
  jepa_neural_cpu/   # Bottom-up JEPA neural computer demo
  world_model/       # JEPA machine world model (predictive dynamics)
  autoresearch/      # Automated research + compounding NPCoT loop
  os/
    neuros/          # Neural OS: 17 modules (MMU, TLB, cache, scheduler...)
    gpu/             # GPU UNIX OS: shell, filesystem, ELF loader, C source
  self_optimizing/   # SOME: hidden controller, fast weights
  neural/            # NeuralCPU: neural ALU bridge, weave pipeline
  model/             # Model-based CPU (neural_ops, assembler)
  tensor/            # Tensor-based ARM64 emulator (differentiable)

# Compiled / accelerated backends
kernels/             # rust_metal (Rust+Metal ARM64 kernel), mlx, npcot_wasm
nsynth/              # Rust program synthesizer (gradient + enumerative + search)
packages/            # Companion packages (metal_mlp)

# Models & synthesis corpus
models/              # Trained neural-component weights (see models/MODEL_INDEX.md)
programs/            # Synthesis benchmark corpus (arithmetic, bitwise, algorithms, ...)

# Evidence, paper, experiments
artifacts/           # Committed benchmark results cited by the paper + tests
paper/               # Research paper + modular sections
benchmarks/          # Benchmark driver scripts
experiments/         # Exploratory experiment runs

# Usage & ops
examples/            # Minimal runnable demos (one per execution path)
demos/               # Larger showcase walkthroughs (BusyBox, Alpine, compiler)
scripts/             # Entry points + maintainer automation
tools/               # Developer tooling
training/            # Training pipelines
packaging/           # Deployment scaffolding (Homebrew, Modal, DEPLOYMENT.md)

# Tests, docs, assets
tests/               # Test suite (see tests/README.md)
docs/                # Documentation
assets/              # Logos / static assets

# Build & runtime output (gitignored — regenerable, not committed)
checkpoints/         # Large .pt weight checkpoints
training_results/    # Coprocessor scaling sweeps, ablation studies
dist/                # Build distributions
logs/  outputs/      # Run logs and scratch outputs
```

Every top-level directory carries its own `README.md` describing its purpose.

---

## Tests

```bash
python -m ncpu doctor
pytest tests/ -q   # 2,500+ tests across the stack
```

Covers: exhaustive formal verification, neural ops, neurOS, compute mode, multi-process, MUXLEQ, BusyBox/Alpine, GPU debugging toolkit, coprocessor, Mog synthesis, differentiable execution, constant-time crypto, self-modifying programs, diff compiler, multi-GPU distributed, SOME, and JEPA predictive models.

---

## Documentation

- **[Research Paper](paper/ncpu_paper.md)** — detailed analysis and findings
- **[GPU Debugging Toolkit Paper](paper/gpu_debugging_toolkit_paper.md)** — the 26-command GPU-native debugger
- **[GPU Debugging Toolkit Reference](docs/gpu/gpu_debugging_toolkit.md)** — full command reference
- **[Rust Metal Kernel](docs/gpu/rust_metal_kernel.md)** — architecture, zero-copy design, build instructions
- **[Compilation Pipeline](docs/gpu/compilation_pipeline.md)** — end-to-end C-to-GPU flow
- **[JEPA Neural CPU](docs/architecture/BOTTOM_UP_JEPA_NEURAL_CPU.md)** — bottom-up neural computer vision and architecture
- **[JEPA Machine World Model](docs/architecture/JEPA_MACHINE_WORLD_MODEL.md)** — predictive dynamics design and use cases
- **[Model Index](models/MODEL_INDEX.md)** — complete trained model inventory
- **[SOME Complete Guide](some/docs/guides/SOME_COMPLETE_GUIDE.md)** — hidden controller and training pipeline
- **[Differentiable Programs](paper/section_differentiable_programs.md)** — program optimization, synthesis, ISA discovery
- **[Benchmark Results](artifacts/BENCHMARK_RESULTS.md)** — pass@1 numbers for every mode and model tier

---

## License

MIT
