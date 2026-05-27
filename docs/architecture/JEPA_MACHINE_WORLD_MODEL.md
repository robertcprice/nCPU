# JEPA Machine World Model (J-MWM)
**A Predictive World Model of Computation Itself — For Speed, Robustness, and New Capabilities on the Neural GPU Computer**

**Status**: Design sketch + implementation roadmap (April 2026)  
**Goal**: Make the core nCPU thesis ("CPU modeled on GPU with neural networks and kernels") *insanely more powerful* by adding a fast, abstract, self-supervised predictive model of machine dynamics on top of the exact neural ALU substrate.  
**Key constraints from user**: Must be **deliberate + novel + robust + FAST**.

---

## The Core Insight

nCPU already has two extraordinary things:

1. **Exact low-level neural logic** (the ALU models + in-shader versions) that compute 32-bit integer operations with 100% verified correctness via memorization-by-decomposition.
2. **Rich latent predictive machinery** already scattered across the system:
   - `LatentMemoryHead`, `LatentControllerState`, `StatePatchHead`, action/halt/descriptor heads (SOME)
   - `ExecutableThoughtHead` + `ArrayExecutableThoughtHead` (hidden state → SoftProgram → nCPU execution → latent state patch)
   - `HotloopValueModel` (predicts when fast GPU path is valuable)
   - `NeuralBranchPredictor`, `NeuralDependencyPredictor`, `NeuralHazardPredictor`, `MemoryOracle`, `BranchTraceBuffer`, instruction scheduler, etc.
   - `hotloop_value_model` for GPU-only auto-handoff policy.

These are *proto world models* of execution. They are mostly hand-designed features + small MLPs or trained on specific narrow objectives.

**JEPA** (Joint Embedding Predictive Architecture, LeCun et al.) gives us a principled, scalable, self-supervised way to turn these into a general, robust, abstract **world model of the entire machine**.

Instead of predicting exact bits or tokens, we learn to predict *in a learned abstract representation space*. The predictor is trained so that its prediction of "what the machine will look like after this action" matches the actual future representation (with clever stop-gradient / EMA target encoder tricks for stability and information content).

Applied here: a **JEPA Machine World Model** learns the *dynamics of a programmable computer* whose gates can themselves be neural.

This is novel. Most JEPA work is vision (I-JEPA, V-JEPA) or robotics. We are doing it for *von Neumann machine state transitions* on top of an already-neural substrate.

---

## Why This Is Especially Powerful on nCPU

Because the low-level substrate is *exact and differentiable* (or fast deterministic in the Rust kernel), the JEPA world model gets two superpowers most world models don't have:

- **Ground truth is perfect and cheap to generate** at scale (just run more programs through the differentiable engine or the GPU kernel and record (state, action, next_state) tuples).
- **You can mix exact and abstract reasoning** at will: use the fast latent JEPA predictor for exploration/speculation/robustness checks, then fall back to the exact neural ALU (Python or in-shader) only when you need precise bits or side effects.

This combination (exact verified neural gates + fast abstract predictive world model of the machine) is extremely rare and defensible.

---

## Proposed Architecture: JEPA Execution World Model (J-EWM)

### Components

1. **State Encoder** (small, fast)
   - Input: Current nCPU machine state (registers [32] int64, memory summary, flags, PC context, recent trace window).
   - Memory can be summarized (hash of pages, small conv over a working set window, or even the existing `neural_memory` components).
   - Output: compact latent vector `z_t` (e.g., 64–256 dim).

2. **Action Encoder**
   - Next instruction(s), or a short proposed `SoftProgram` / code fragment, or even a high-level "intent" embedding from the SOME controller.
   - Can be a small transformer or just a learned embedding of the opcode + operands (we already have excellent opcode embeddings in several places).

3. **Predictor** (the JEPA heart — can be very small for speed)
   - Takes `(z_t, action_embedding)` → predicted `ẑ_{t+k}` (k-step or multi-step).
   - Can be a lightweight MLP, small GRU/LSTM, or even a tiny transformer over a short action sequence.
   - **Goal for speed**: This must be *much* cheaper per step than full exact execution.

4. **Target Encoder** (EMA of the state encoder, stop-gradient)
   - Encodes the *actual* next state after running the action on the real engine (differentiable execution engine or fast tensor path or even sampled GPU kernel traces).
   - The predictor is trained to make its output close (in a chosen similarity metric) to the target encoder's output on the real future.

5. **Optional: Variance / Uncertainty Head**
   - Predict not just the mean latent, but also a confidence or variance. High uncertainty → fall back to exact execution. This is key for robustness + speed (speculative execution only when safe).

**Training is almost entirely self-supervised** on execution traces we can generate in unlimited quantity using the existing `DifferentiableEngine`, `NeuralCPU`, and GPU kernel paths.

---

## Concrete Novel + Fast + Robust Use Cases

### 1. Latent Speculation & Fast "Mental Simulation" (Biggest Speed Win)
In the SOME / ExecutableThoughtHead / coprocessor loops:
- Instead of executing every candidate thought/program exactly on the differentiable engine (expensive, even if differentiable), first run 8–32 steps of cheap J-EWM rollouts in latent space.
- Only the top-k most promising trajectories (by predicted reward or by reaching a desired latent goal state) get promoted to expensive exact execution.
- This directly attacks the sample/compute cost of program synthesis and hidden reasoning.

**Fast path**: The predictor can be tiny enough to run on CPU or in the same GPU context as the main model.

### 2. Robustness / Anomaly Detection / Self-Repair (The "Robust" Part)
- During real execution (especially in the GPU kernel or when using the neural ALU), occasionally compare actual state evolution against the J-EWM prediction.
- Large divergence → something interesting or wrong is happening.
  - Could trigger the existing `neural_error_recovery`, `neural_security_monitor`, or the `train_error_recovery.py` paths.
  - Could be used for constant-time verification hardening or side-channel detection (the world model expects "normal" timing behavior in latent space).
- In the coprocessor setting: the LLM can trust the world model prediction for "what will this code do?" most of the time, and only pay for full emulation on high-uncertainty cases.

This makes the whole system more robust to its own neural components being slightly imperfect or to adversarial inputs.

### 3. Better Hot-Loop / Phase Detection and GPU Handoff (Directly Improves the Hero Path)
The existing `HotloopValueModel` is already doing a narrow version of this.
A J-EWM can learn much richer representations of "what kind of computation phase we are in" (compute-bound, memory-bound, branchy, regular, etc.) from raw execution traces and predict when the fast Rust Metal path or a specific neural optimization will win.
This can feed back into the GPU kernel scheduling and the `gpu_only.py` auto-handoff logic.

### 4. Training Signal for the Latent Controller Stack (SOME)
The JEPA loss provides a dense, self-supervised training signal for the latent memory, action, halt, and descriptor heads. "Did my internal latent prediction of what executing this thought would do match reality?" becomes a powerful auxiliary loss.

This is exactly in the spirit of making the hidden controller more "deliberate" (it has an explicit world model it is trying to be accurate about).

### 5. (Longer-term, High Novelty) JEPA inside the GPU Kernel or as a Coprocessor Expert
Distill a tiny version of the world model predictor into weights that can live in the Metal shader (similar to how we already embed the neural ALU weights).
This gives the GPU computer a fast "imagination" capability for scheduling, prefetching, or even lightweight speculative execution of hot regions — all while the exact path remains available for correctness.

Even more ambitious: expose the J-EWM as another expert in the differentiable coprocessor router. The LLM can route tokens through "ask the machine world model what this code will do" in addition to the exact neural ALU.

---

## Current Reality (the live substrate integration has changed the game)

The original "build a tiny J-EWM from scratch" roadmap has been overtaken by the concrete live JEPA Neural Kernel work on the real Rust Metal substrate.

What already exists and is running on real guest code today:
- A fast, always-available, predictor-free churn scorer using structured page/dirty model + true recency (`current_step` vs `_last_touch_step`).
- Live observation at every context switch and memory syscall with rich snapshots pushed from the actual GpuLauncher.
- Three decision levers that the learned model is already using to bias real scheduling (69–80 overrides observed on real BusyBox multi-process workloads).
- Adaptive de-prio skips driven directly by churn delta at the moment of bias.
- Full fairness telemetry (`times_scheduled`, `per_process_scheduled`, `get_all_deprios`).

The "world model" work is no longer purely speculative research — parts of it are already in production on the deterministic GPU computer, delivering measurable scheduling influence.

## Updated Implementation Priorities (Grounded)

1. **Make the existing fast churn scorer even stronger** (the thing that is actually firing the levers today).
   - Richer page features, better recency decay, working-set awareness.
   - Feed the same signals into a small learned head when we want probabilistic speculation.

2. **Use the live substrate as the source of truth for training data**
   - The Rust Metal execution (plus the existing DifferentiableEngine) can generate essentially unlimited, perfectly labeled (state, action, next_state) traces from real multi-process UNIX workloads.
   - This is gold for training a proper J-EWM or for distilling the current heuristic scorer.

3. **Add cheap latent speculation on top of the existing levers**
   - Use a small learned world model (or even just the current churn scorer + simple rollouts) to look a few steps ahead before committing to a bias decision or a de-prio skip value.
   - High-uncertainty cases fall back to the exact deterministic substrate (which we already have).

4. **Distillation into the shader (the real long-term win)**
   - The most valuable learned components (recency-aware pressure model, adaptive yield policy, small hazard/branch predictor) get lowered into Metal shader weights exactly like the current neural ALU.
   - The GPU computer then has *learned* scheduling and memory pressure logic that was never hand-written, while the exact path remains for post-mortem and verification.

5. **Coprocessor + hidden controller integration**
   - Expose the live J-EWM (or the fast churn signals) as another expert the LLM can route through ("ask the machine world model what this code will actually do to memory pressure and scheduling").
   - Feed the same signals into the SOME hidden controller as a dense, self-supervised training signal ("did my internal prediction of what this thought would do to the machine match reality?").

This is the correct evolution: the live substrate integration gave us something real and measurable much faster than the original pure-research roadmap. Now we harden the fast path that is already working, generate better training data from it, and distill the valuable pieces back into the shader.

**Phase 3 (maximum novelty)**
- Multi-step latent rollouts + goal-conditioned prediction ("I want the machine to reach this latent state — what short program gets me there?").
- Hierarchical JEPA (fast low-level predictor + slower higher-level abstract program semantics predictor).
- Close the loop with the neural ALU training: use divergence between JEPA prediction and actual neural execution as a hard negative or curriculum signal.

---

## Specific Files / Modules That Are Natural Homes

- New module: `ncpu/world_model/` or `ncpu/jepa/` (keep it clean and small at first).
  - `state_encoder.py`
  - `action_encoder.py`
  - `predictor.py`
  - `je_world_model.py` (main class)
  - `data.py` (trace collection from existing engines)
- Strong integration points:
  - `ncpu/self_optimizing/executable_thought_head.py` and array version (add latent rollout before exact execution).
  - `ncpu/self_optimizing/latent_heads/` (especially memory + controller state).
  - `ncpu/differentiable/execution.py` and `DifferentiableEngine` (primary data source + place to inject cheap prediction).
  - `ncpu/execution_training/` (add world model as an additional training signal source).
  - `ncpu/neural/cpu/prediction.py` and the various `*Predictor` classes (the J-EWM can subsume or augment many of them).
  - Longer term: `kernels/rust_metal/` (distilled predictor weights + shader integration, analogous to current neural_alu.rs).

The `ExecutableThoughtHead` + `DifferentiableCompiler` + `SoftProgram` machinery is *already* doing a form of "latent program execution." JEPA makes the *prediction* of that execution abstract and cheap.

---

## Why This Makes the Whole Project More "Insanely Novel + Robust + Fast"

- **Novelty**: First (to our knowledge) application of modern JEPA-style predictive world models to the dynamics of a *programmable computer with neural gates*. "A world model of computation itself."
- **Robustness**: Explicit predictive model + uncertainty gives anomaly detection, better error recovery, and safer use of the neural components (including when they are injected into LLMs via the coprocessor).
- **Speed (deliberate + fast)**: Cheap latent rollouts let the higher-level agents (SOME, executable thought, future Weight CPU) explore far more possibilities per unit of expensive exact execution. This is the same philosophy that makes MCTS + fast value networks powerful, but applied to program synthesis and machine control.
- **Strengthens the hero thesis**: The exact neural ALU (in Python *and* in the Rust Metal shader) becomes the reliable "physics engine" that the abstract world model is trained against and can fall back to. The combination is greater than either alone.

This is not "add JEPA because it's trendy." It is a natural, high-leverage completion of the latent predictive machinery that is *already* one of the most sophisticated parts of the repo, applied to the unique substrate nCPU provides.

## Extension: Full Bottom-Up JEPA Neural CPU

The current JEPA Machine World Model is the foundation for a much more ambitious goal: a *complete* CPU (registers, memory model, control logic, pipeline, everything) built bottom-up as a hierarchy of JEPA predictors.

See the dedicated document `docs/architecture/BOTTOM_UP_JEPA_NEURAL_CPU.md` and the early prototype in `ncpu/jepa_neural_cpu/`.

This is the bottom-up neural computer counterpart to top-down work (Microsoft Neural Computer ideas and similar). The machine dynamics themselves become learned and predictive.

---

## Risks & Mitigations (Be Deliberate)

- Risk: The latent state is too lossy and the world model learns nothing useful.
  - Mitigation: Start with rich enough state (include recent trace window + memory working set summary). Use the existing predictors as feature sources.
- Risk: Training is unstable (classic JEPA problem).
  - Mitigation: Use proven tricks from the I-JEPA / V-JEPA papers (multi-scale, stop-grad, EMA, variance regularization). We have the luxury of unlimited perfect synthetic data.
- Risk: Adds complexity and slows things down.
  - Mitigation: Keep the initial predictor *tiny* and optional. The first version should demonstrably *increase* throughput of synthesis or hidden reasoning, not decrease it.

---

**Recommended Immediate Next Action**

Implement Phase 0 as a small, self-contained prototype under `ncpu/world_model/je_world_model.py` + a training script that re-uses the existing differentiable execution traces.

Once a working (even if simple) J-EWM exists and shows it can predict useful things in latent space, the integration points (especially into executable thought and the latent controller) become obvious and high-ROI.

This direction has the potential to be one of the most distinctive and defensible contributions on top of the already-strong "exact neural gates on GPU computer" foundation.

---

*Write the code. Measure the speed/robustness win. Then decide how deeply to embed it into the hero Rust kernel path and the coprocessor.*