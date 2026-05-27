# Bottom-Up JEPA Neural CPU: A Complete Neural Machine

**Vision**: A full CPU (not just ALU) built entirely from neural networks, bottom-up, using JEPA-style predictive world models as the fundamental primitive for state, control, memory, and execution.

This is the natural evolution of nCPU's hero thesis ("The GPU *is* the computer") + the JEPA Machine World Model work.

## Contrast with Top-Down "Neural Computers" (e.g., Microsoft-inspired work)

Top-down approaches (Microsoft Research, DeepMind DNC, Neural GPUs, Neural Execution Engines, etc.):
- Start with high-level abstract architecture (controller + memory + interfaces).
- Learn to *use* the architecture to execute algorithms.
- Often trained on program synthesis or algorithmic tasks.
- Strengths: Good at learning high-level reasoning and algorithms.
- Weaknesses: The "computer" substrate is still hand-designed; the NNs are *on top* of a traditional computer metaphor.

nCPU + JEPA bottom-up direction:
- Start from the lowest level (neural gates / truth tables / predictors).
- Build registers, ALU, memory, pipeline, control logic, branch prediction, etc., as learned predictive models.
- The entire machine *is* a JEPA world model (or hierarchy of them): current latent state + instruction-as-action → predicted next latent state + side effects.
- "Cross-JEPA": Multi-scale / hierarchical predictors (bit-level, instruction-level, program-level, task-level).
- Run on the existing deterministic GPU substrate (Rust Metal + optional in-shader neural weights) for speed + superpowers (exact replay, constant-time proofs, post-mortem debugging).

This gives us a **complete neural computer** whose dynamics are learned, predictive, and differentiable (or fast deterministic when desired).

## Proposed Architecture: JEPA Neural CPU (JNC)

### Core Primitive: The JEPA Machine Step
Instead of traditional fetch-decode-execute:
```
latent_state_t, instruction → predictor(latent_state_t, instruction) → latent_state_{t+1}, outputs, side_effects
```

- **Latent State**: Compressed predictive representation of the entire machine (registers, memory summary, flags, PC context, recent history, "mental model" of program intent).
- **Action**: Instruction (or sequence / micro-op / SoftProgram).
- **Predictor**: JEPA-style (or cross-JEPA hierarchical) network trained to predict the next latent state accurately.
- Training: Self-supervised on massive traces from the existing DifferentiableEngine + GPU kernel. Use real JEPA objectives (contrastive, variance regularization, etc.) so the latent space is rich and useful for downstream reasoning.

### Layers (Hierarchical / Cross-JEPA)
1. **Low-level (Bit / Gate level)**: Extend current neural ALU (truth tables, Kogge-Stone carry, byte-pair MUL) into a full predictive model of bit transitions.
2. **Mid-level (Instruction / Micro-arch)**: Predict effects of individual instructions on latent register/memory state. Include learned branch predictor, hazard detector, etc., as JEPA heads.
3. **High-level (Program / Algorithm)**: Predict effects of short instruction sequences or full SoftPrograms. This is where the current ExecutableThoughtHead + JEPA speculation already points.
4. **Meta-level (Task / Intent)**: Higher JEPA layers that predict what a program "means" or what the overall goal state should be. Ties into SOME latent controllers and Weight CPU ideas.

### Execution Modes
- **Fast Deterministic Mode**: Distilled / compiled version runs inside the Rust Metal kernel (like current neural ALU weights in shader). Gives ~1.9M IPS + all determinism superpowers.
- **Differentiable / Research Mode**: Full JEPA predictors with gradients. Enables backprop through the entire machine dynamics.
- **Hybrid / Speculative Mode**: Use cheap high-level JEPA rollouts for speculation (current speculate_with_world_model direction), fall back to lower levels or exact substrate only when needed. This is where massive speed + robustness comes from.

### Memory
Not a traditional addressable array, but a learned predictive memory model (JEPA memory heads that predict what reading/writing at a "location" (latent key) will do). Can still interface with traditional memory when needed for I/O or compatibility.

### Registers / State
Latent vectors + sparse explicit values when precision is required. The model learns when to keep things explicit vs compressed/predictive.

## Why This Is Extremely Valuable and Novel

- **True bottom-up neural computer**: Not "NNs running on a computer" or "learned controller on top of hand-designed machine". The machine *itself* is neural and predictive.
- **Direct evolution of existing work**:
  - Current neural ALU + in-shader execution (hero GPU path).
  - Differentiable execution engine.
  - ExecutableThoughtHead + latent controllers (SOME).
  - JEPA Machine World Model (already in progress).
- **Enormous implications**:
  - Learning algorithms by optimizing the machine dynamics directly.
  - Extremely robust execution (prediction error = anomaly / attack / bug detection).
  - Speculative execution at every level (massive effective speedups).
  - New security properties (predictive constant-time, learned side-channel resistance).
  - Bridges systems + representation learning in a fundamental way.
  - On the GPU substrate: we get determinism + debugging superpowers "for free" on a fully neural machine.

This positions nCPU as the credible bottom-up counterpart to top-down neural computer research (Microsoft, DeepMind, etc.).

## Immediate Next Steps (High-Value Bets)

1. **Prototype** — Done and *significantly hardened* (see `ncpu/jepa_neural_cpu/`).
2. **Watchable + trainable demo of a real program** — Extremely strong now. Run:
   ```bash
   python3 -m ncpu.jepa_neural_cpu.demo
   ```
   It now demonstrates four clear phases:
   - PHASE 1: Observe a real unrolled sum-1-to-N loop with full per-step register visibility (before / executed / predictor guess + error at *every* instruction).
   - PHASE 2: The JNC *trains its own internal JEPA predictor* on the exact 10 observed transitions (tiny online learning loop, loss visibly drops).
   - PHASE 3: Replay the identical program with the now-trained predictor → final per-step error drops from ~1.0 to <0.01, guesses become visually identical to reality.
   - PHASE 4: Cross-check the exact same instruction sequence against the *real* DifferentiableEngine → identical r0=15.00 result.
3. **Live integration with the real deterministic substrate** (Rust Metal GpuLauncher) — achieved. The JEPA Neural Kernel now receives real process snapshots at every context switch and memory syscall (brk/mmap/munmap), maintains a structured shadow memory model (dirty pages + access counts + mutation counts + true recency via current_step / _last_touch_step), and returns scheduling bias.

4. **Three decision levers active on real guest code**:
   - Lever 1: `on_context_switch` bias (prefer low-churn peer at every schedule point).
   - Lever 2: Immediate `on_syscall` bias on memory operations (high-signal moment).
   - Lever 3: Adaptive persistent yield (model computes churn delta and sets `jepa_deprio_remaining` 1–7; launcher + schedule_next respect and age the skips). This makes small relative spreads (0.04) produce multi-turn de-prioritization.

5. **Fairness telemetry** (`times_scheduled` per-process counter + `per_process_scheduled` in LaunchResult + `get_all_deprios`). The model is now measured not just by override count but by actual change in slice distribution.

6. **Concrete results on real BusyBox** (aarch64, multi-process sh -c workloads with heavy dd + lights): 69–80 real scheduling overrides per short run, visible churn differentiation (0.042–0.047 spreads), 140+ syscalls + mutations observed live, 550+ observation steps. Cycles remain identical on pure memory-bound cases (expected); the value is the learned model actively steering a deterministic UNIX scheduler.

7. **Adaptive delta-driven deprio** (latest): skip duration is proportional to the exact relative churn delta at the moment of bias. Hotter pressure = longer forced yield.

This is the credible bottom-up Neural OS direction on a substrate that already boots real multi-process UNIX with determinism superpowers.

See also:
- docs/architecture/JEPA_MACHINE_WORLD_MODEL.md (foundation)
- docs/architecture/HERO_THESIS.md + LAYERING.md (the GPU computer substrate)
- docs/architecture/NEURAL_OS_VISION.md (full roadmap to hosting a real OS)
- kernels/rust_metal/src/neural_jepa_kernel.rs (compute_churn_score with true recency, on_context_switch, on_syscall, adaptive deprio logic)
- kernels/rust_metal/src/launcher.rs (live observation + bias application sites)
- /tmp/test_real_jepa_busybox.py (canonical A/B + fairness measurement harness)
- ncpu/jepa_neural_cpu/jepa_neural_cpu.py (Python research surface + status)