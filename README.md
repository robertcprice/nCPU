<p align="center">
  <img src="assets/logo.png" alt="nCPU" width="400">
</p>

<p align="center">
  <strong>A complete computer in which every layer — arithmetic, OS, compiler, display — is either a trained neural network or runs entirely on GPU.</strong><br>
  The model doesn't run <em>on</em> the computer. The model <em>is</em> the computer.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/interactive-differentiable%20program%20discovery-brightgreen" alt="Interactive discovery">
  <img src="https://img.shields.io/badge/models-24%20trained-blue" alt="Models">
  <img src="https://img.shields.io/badge/accuracy-100%25%20integer-green" alt="Accuracy">
  <img src="https://img.shields.io/badge/coprocessor-11%20model%20sweep-orange" alt="Coprocessor">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License">
</p>

---

nCPU is one repository pursuing one thesis from five directions: a computer
can be built out of learned components, and once the whole execution stack is
differentiable, programs stop being things you write and become things you can
search for by gradient descent. Each subsystem below stands on its own
measurements; together they cover the stack from individual ALU operations to
an operating system to program synthesis.

## The five subsystems

### 1. The neural computer

Every ALU operation — addition, subtraction, multiplication, bitwise logic,
shifts, division — is a trained neural network. The neural OS (neurOS) manages
memory, schedules processes, and compiles code through 11 trained models with
no hand-written fallbacks. A neural display renders characters through
char→glyph MLPs and a ConvNet (143K parameters). The full pipeline — source
code → neural compiler → neural assembler → neural CPU → neural display — is
differentiable end to end.

The neural ALU reaches 100% accuracy on 32-bit integer arithmetic, verified
exhaustively over every possible input. One result inverts the conventional
hardware hierarchy: multiplication is 12x *faster* than addition here, because
addition needs an 8-pass carry chain while multiplication decomposes into
parallel byte-pair table lookups.

| Instruction | Strategy | Latency |
|-------------|----------|---------|
| ADD/SUB/CMP | Kogge-Stone carry-lookahead (8 passes) | 248 µs |
| MUL | Byte-pair LUT (65,536 entries) | 21 µs |
| AND/OR/XOR | Vectorized truth table | 21 µs |
| SHL/SHR | Attention-based bit routing | 434 µs |
| DIV | Restoring division (neural subtraction) | varies |

neurOS component accuracy:

| Component | Accuracy | Component | Accuracy |
|-----------|----------|-----------|----------|
| MMU | 100% | Assembler codegen | 100% |
| TLB | 99.6% | Assembler tokenizer | 99.4% |
| Cache | 99.7% | Compiler optimizer | 95.2% |
| Scheduler | 99.2% | Watchdog | 100% |
| Prefetch | 97.8% | Block allocator | 98.4% |

### 2. The GPU computer

A self-sufficient computer on a single GPU — the CPU is involved only at
bootstrap. The Rust + Metal kernel executes about 200 ARM64 instructions
(integer and floating-point) at roughly 1.9M instructions per second, with
zero-copy shared memory and zero cycle-count variance across runs (σ = 0.0).

What runs on it:

- A multi-process UNIX OS: fork/pipe/wait, a 25-command shell, 28 syscalls,
  up to 15 concurrent processes
- A self-hosting C compiler (~4,200 lines) that compiles itself, then compiles
  and runs other programs — entirely on the GPU
- Real Linux binaries via an ELF64 loader: BusyBox (264KB, 34+ commands) and
  Alpine Linux v3.20
- 13+ compiled C applications: SHA-256, AES-128, Tetris, Snake, a Brainfuck
  interpreter, a Forth REPL, a CHIP-8 emulator, an HTTP server, an MNIST
  classifier, and others
- A 26-command deterministic debugger: instruction tracing, breakpoints and
  watchpoints, time-travel debugging, a memory sanitizer, automated fuzzing,
  reverse data-flow analysis, and constant-time verification. Deterministic
  execution is what makes time-travel and exact replay possible; conventional
  CPUs, with cache- and speculation-induced timing noise, can't offer the same
  guarantees.

### 3. Differentiable program synthesis

Given input/output examples, gradient descent discovers executable programs by
backpropagating through the differentiable CPU. A candidate program is a set
of continuous parameters — Gumbel-softmax distributions over opcodes, soft
attention over registers — that temperature annealing collapses into discrete,
runnable code.

Two synthesizers cover two benchmark suites, both at full coverage:

- **Mog** (grammar-constrained, differentiable compiler): 315/315 problems.
- **nSynth** (Rust solver portfolio): 105/105 problems on the expanded suite.
  (The paper's canonical suite is the earlier 95-problem version; the expanded
  suite adds a template-solver family.)

nSynth's coverage by solver family:

| Family | Solved | Method |
|--------|--------|--------|
| Gradient | 66/105 | Differentiable search with a learned restart bank |
| Enumerative | 21/105 | Bottom-up expression enumeration |
| Search | 13/105 | Single-branch, struct-pair, and string teachers |
| Template | 5/105 | Pattern matching for the hardest problems |

No single family gets close to full coverage alone; the portfolio does. Key
optimizations: persistent solved-program memoization (about 5000x on a cache
hit), a learned bias bank with warm-refine transfer across problems, and a
constant vocabulary mined from the examples themselves.

### 4. The differentiable coprocessor

The neural ALU injected into a transformer's forward pass as a routed expert.
A learned per-token gate decides whether each token flows through the original
MLP or through the neural ALU. Bilinear soft truth tables provide
differentiable logic, tensor ops provide differentiable arithmetic, and
gating is modulated by model confidence.

Results from an 11-model sweep across the Qwen 2.5/3/3.5 families, on
arithmetic tasks:

| Model | Arithmetic accuracy | Note |
|-------|--------------------|------|
| Qwen3.5-2B (instruct) | 14.5% → 71.0% (+56.5 pp) | best overall |
| Qwen3.5-2B (base) | 15.5% → 63.0% (+47.5 pp) | 100% on ADD/SUB/MUL/DIV |
| Qwen3.5-4B | +51.0 pp | largest base-model gain (tied) |
| Qwen3.5-9B | +51.0 pp | largest base-model gain (tied) |

Real-world transfer is measured, not extrapolated: on full HumanEval
(Qwen3.5-4B, A100), 62.2% → 64.6% — four additional problems solved.

### 5. JEPA predictive machine dynamics

A predictive world model of the computer itself. Alongside exact execution, a
JEPA-style network (Joint Embedding Predictive Architecture) learns to predict
machine state transitions in a compressed latent space:

```
latent_state_t + instruction → predictor → latent_state_{t+1}
```

It runs at two levels. A Python demo (`ncpu/jepa_neural_cpu/`) executes real
programs next to the predictor, turning prediction error into a live anomaly
signal. A Rust Metal implementation (`kernels/rust_metal/src/jepa/`, 2,858
lines) observes deterministic GPU execution and actively steers scheduling
through learned bias overrides.

Because the substrate underneath is exact, this world model has two properties
most lack: unlimited free ground truth (run more programs), and the ability to
mix predicted and exact execution at will — cheap latent speculation when
exploring, exact execution when it matters. The long-term direction is a
hierarchy of predictors at the bit, instruction, program, and task levels.

```bash
python3 -m ncpu.jepa_neural_cpu.demo     # bottom-up JEPA neural computer demo
python -m ncpu.world_model.quickstart    # JEPA machine world model quickstart
```

---

## Start in 60 seconds

```bash
pip install -e ".[demo,dev]"

# The headline demo: the GPU as a complete computer (macOS / Apple Silicon)
python -m ncpu gpu                 # boot it
python -m ncpu gpu --neural-alu    # with the neural ALU inside the Metal shader
python -m ncpu gpu debug           # 26-command deterministic debugger

# Cross-platform, no heavy dependencies
python -m ncpu discover            # program by examples, via differentiable synthesis
python -m ncpu text --interactive  # neural text / cipher machine

# Full neural pipeline (requires the model stack)
python -m ncpu full-neural         # bottom-up neural CPU + neural display
python -m ncpu meta-compare        # side-by-side comparison demo

# JEPA predictive layer
python3 -m ncpu.jepa_neural_cpu.demo
python -m ncpu.world_model.quickstart

# Rust-native, no Python required
cd kernels/rust_metal
cargo run --bin ncpu_run -- --elf ../../demos/gpu/busybox.elf --rootfs -- echo hello
```

---

## Three execution modes

| Mode | What runs | Differentiable? | Speed |
|------|-----------|-----------------|-------|
| Neural | 13 trained `.pt` models | yes — full gradient flow | ~5K IPS |
| Fast | native tensor ops | yes — standard autograd | ~5K IPS |
| Compute | Rust + Metal shader | no (discrete hardware) | ~1.9M IPS |

All three execute the same programs and produce the same results. Neural mode
sends every operation through trained networks. Fast mode uses native tensors
with the same ISA and the same differentiability. Compute mode trades gradient
flow for speed — it is where the UNIX OS boots, the compiler self-hosts, and
BusyBox runs.

```python
# Neural mode — every operation is a trained model
from ncpu.model import CPU
cpu = CPU(neural_execution=True)
cpu.load_program("MOV R0, 7\nMOV R1, 6\nMUL R2, R0, R1\nHALT")
cpu.run()
print(cpu.get_register("R2"))  # 42 — computed by the neural byte-pair LUT

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
# discovers: ADD R2, R0, R1; HALT

# JEPA world model — predict machine state transitions
from ncpu.world_model.je_world_model import JEWorldModel, JEWMConfig
model = JEWorldModel(JEWMConfig(state_dim=22, action_dim=8))
pred = model.predict_next_latent(model.encode_state(state), model.encode_action(action))
```

---

## The full stack

| Layer | Implementation | Result |
|-------|---------------|--------|
| ALU | 13 trained `.pt` models | Exact 32-bit integer arithmetic, exhaustively verified |
| OS | neurOS — 11 neural models, no fallbacks | Learned MMU, TLB, cache, scheduler, compiler |
| GPU compute | Rust Metal kernel, ~200 ARM64 instructions | Arbitrary programs at ~1.9M IPS |
| UNIX OS | Compiled C on Metal | fork/pipe/wait, 25-command shell, 28 syscalls |
| Compiler | cc.c, ~4,200 lines, self-hosting | Compiles itself, then compiles programs — on GPU |
| ELF loader | Real Linux binaries on GPU | BusyBox and Alpine Linux v3.20 on Metal |
| Coprocessor | Neural ALU in a transformer forward pass | Tokens routed through neural arithmetic, measured gains |
| JEPA | Predictive world model of machine dynamics | Latent speculation + anomaly detection over an exact substrate |
| Program synthesis | Backprop through execution | Programs discovered from I/O examples |
| Constant-time crypto | AES-128 ECB/CBC (`ncpu/crypto/`) | σ = 0.0 timing; FIPS 197 + NIST SP 800-38A vectors pass |
| Multi-GPU | Distributed cores with shared memory | fork/pipe/wait across GPUs; parallel and pipeline execution |
| SOME | Hidden controller with latent heads | Self-optimizing inference; HumanEval+ and BigCodeBench gains |

---

## Timing side-channel immunity

GPU execution here produces zero cycle-count variance — σ = 0.0 across 270
runs, where the same code on native Apple Silicon shows 47–73% timing
variance. With no data cache there are no cache lines and no cache-miss
penalty, so AES T-table attacks have nothing to measure.

Built on that property, `ncpu/crypto/` provides constant-time AES-128 (ECB and
CBC) from 19 constant-time primitives, passing all FIPS 197 and NIST SP
800-38A test vectors.

---

## Self-Optimizing Machine Engine (SOME)

A hidden controller that turns part of the neural machine into an internal
coprocessor for code generation: a buffered think → write → verify → patch →
commit loop, learned action/halt/descriptor/state-patch/memory heads, and
task-local fast weights updated during inference. The learned memory head
improved validation MSE by 83.26% over baseline.

Measured end to end: HumanEval+ for qwen3.5:4b improved 147 → 154 and for
qwen3.5:9b 144 → 156; BigCodeBench-Hard for qwen3.5:9b improved 33 → 49.

---

## MUXLEQ: Turing-complete in two instructions

SUBLEQ plus MUX, running in all three execution modes. In neural mode, SUB
goes through the Kogge-Stone carry-lookahead (~248 µs) and MUX through neural
AND/OR/NOT (~63 µs). It loads `.dec` images and boots eForth. The point: if
trained networks exactly execute a two-instruction one-instruction-set
computer, the construction extends to any instruction set.

---

## Program synthesis from examples (nsynth_codegen)

```bash
cargo build --release --bin nsynth_codegen
./target/release/nsynth_codegen --lang python --examples '{
  "name":"square","signature":"fn square(x: i64) -> i64",
  "examples":[{"inputs":[0],"expected":0},{"inputs":[3],"expected":9}]
}'
# → def square(x: int): return (0 * x * x) + (1 * x * x) + 0
```

---

## Project structure

```
ncpu/
  differentiable/    # Differentiable execution, program synthesis, ISA discovery
  coprocessor/       # Inject nCPU into transformer forward passes
  execution_training/# Differentiable execution as training signal for code LMs
  crypto/            # Constant-time crypto (AES-128)
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

Every top-level directory has its own `README.md` describing its purpose.

---

## Tests

```bash
python -m ncpu doctor
pytest tests/ -q   # 2,500+ tests across the stack
```

Coverage spans exhaustive formal verification of the ALU, neural ops, neurOS,
compute mode, multi-process execution, MUXLEQ, BusyBox/Alpine, the GPU
debugging toolkit, the coprocessor, Mog synthesis, differentiable execution,
constant-time crypto, self-modifying programs, the diff compiler, multi-GPU
distribution, SOME, and the JEPA predictive models.

---

## Documentation

- [Research paper](paper/ncpu_paper.md) — the full analysis and findings
- [GPU debugging toolkit paper](paper/gpu_debugging_toolkit_paper.md) — the 26-command GPU-native debugger
- [GPU debugging toolkit reference](docs/gpu/gpu_debugging_toolkit.md) — command reference
- [Rust Metal kernel](docs/gpu/rust_metal_kernel.md) — architecture, zero-copy design, build instructions
- [Compilation pipeline](docs/gpu/compilation_pipeline.md) — end-to-end C-to-GPU flow
- [JEPA neural CPU](docs/architecture/BOTTOM_UP_JEPA_NEURAL_CPU.md) — bottom-up neural computer architecture
- [JEPA machine world model](docs/architecture/JEPA_MACHINE_WORLD_MODEL.md) — predictive dynamics design
- [Model index](models/MODEL_INDEX.md) — complete trained-model inventory
- [SOME complete guide](some/docs/guides/SOME_COMPLETE_GUIDE.md) — hidden controller and training pipeline
- [Differentiable programs](paper/section_differentiable_programs.md) — program optimization, synthesis, ISA discovery
- [Benchmark results](artifacts/BENCHMARK_RESULTS.md) — pass@1 numbers for every mode and model tier

---

## License

MIT
