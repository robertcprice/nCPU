# Game-Scale Synthesis Stages 3-5 — Design Sketches

Stages 1 and 2 of the roadmap (§19 in the paper) are tractable in-session:
scaling slot counts and adding multi-loop phases. Stages 3-5 are architectural
additions that require focused research sprints. This document is a design
sketch detailed enough that a future session can pick up with clear entry
points, not a promise of immediate implementation.

## Stage 3 — Hierarchical Library-Augmented Synthesis

### Problem Statement

Current synthesis solves one function at a time. For app-scale code we need to
synthesize a library of N functions where later functions can call earlier
ones. A naive "synthesize all at once" approach blows up the search space
multiplicatively. The practical approach is sequential synthesis with a
growing call vocabulary.

### Approach

1. **Function registry.** Introduce `SynthesisLibrary` with methods:
   - `register(name: &str, code: String, io_behavior: Fn)` — store a solved function.
   - `candidates(n_args: usize) -> Vec<&LibraryFn>` — library entries that match
     the given arity.

2. **Soft pool extension.** `SoftUniversalProgram` currently has a register pool
   of `[arg, const, slot]` entries. Extend to include
   `[arg, const, slot, library_call_result]`. Each library-call entry has its
   own soft selector over which library function to call (Gumbel-softmax over
   library indices) and soft selectors for each function's arguments from the
   current pool.

3. **Training loop.**
   - Synthesize functions in benchmark-declared order (or a dependency order).
   - After each success, register in the library.
   - When synthesizing function k, the soft pool includes library entries for
     functions 0..k-1.

4. **Loss extension.** A library call's forward pass must actually execute the
   cached function on the soft-selected arguments. Easy case: store the cached
   function as an `fn(Vec<f32>) -> f32` closure. Hard case: the cached function
   has array inputs or control flow — we'd need a soft-executable version.

### Key Design Decisions

- **Pure-gradient calls only at first.** Only library functions whose soft
  form was successfully gradient-solved can be called softly. Template-solved
  or search-solved functions become discrete, un-differentiable calls. This
  keeps gradient flow clean.

- **Argument-count matching.** A library entry `helper(x: i64) -> i64` can only
  be called from positions where we want a scalar result computed from a
  scalar input. The pool extension must enforce this.

- **Recursion forbidden in the library.** Recursive calls during synthesis
  diverge. If a function's specification implies recursion (e.g., factorial),
  we synthesize an iterative form.

### Benchmark Milestones

- **Tier 3a**: `g(x) = 2 * f(x)` where `f(x) = x + 1`, both to be synthesized.
  Success = gradient discovers the call structure in one training run.
- **Tier 3b**: a 3-function compositional problem (e.g., `max3(a,b,c) =
  max2(max2(a,b), c)` where `max2` was synthesized earlier).

### Entry Points

- New file: `nsynth/src/synthesis/library.rs` with `SynthesisLibrary` struct.
- Modify `universal.rs`: extend `univ_pool` to optionally include
  `n_library_entries`. Add `library_soft_call` helper.
- Solver pipeline change: `solve_benchmark_library_mode` iterates problems and
  updates the library.

### Estimated Effort

~2 weeks of focused research. The differentiable call execution is the
unknown; everything else is engineering.

---

## Stage 4 — State-Bearing Programs

### Problem Statement

A game's state (score, position, entities) persists across frames. Current
programs are pure `(input) -> output` functions. To synthesize stateful code we
need `(state_in, event) -> (state_out, output)`.

### Approach

1. **I/O specification trace.** Replace
   `examples: Vec<(Inputs, Output)>` with
   `traces: Vec<Vec<(StateIn, Event, StateOut, Output)>>`. Each trace is a
   sequence of transitions. Loss aggregates per-transition error across the
   trace.

2. **Soft program state threading.** `SoftUniversalProgram::forward` currently
   takes inputs and returns a scalar. Extend to
   `forward(state_in: &[f32], event: f32) -> (state_out: Vec<f32>, output: f32)`.
   The soft program must allocate M state components (small number,
   configurable, e.g., 4-8).

3. **Specification shape.** A trace of length T gives T transitions. The loss
   is the sum of per-transition squared errors on both `state_out` and
   `output`.

### Key Design Decisions

- **State size is a hyperparameter.** Pick M at problem-registration time.
  Over-provisioning is fine (unused state components stay at 0).

- **Traces, not examples.** The specification shape changes fundamentally.
  All existing Tier 1 and 2 benchmarks still work (they're 1-step traces with
  no state).

- **State is scalar for now.** Array-valued state (e.g., a game grid) is
  Stage 4b, not 4a.

### Benchmark Milestones

- **Tier 4a**: "counter" — event 0 increments, event 1 resets. Specification is
  a 10-event trace with state ∈ {0, 1, 2, ..., 5}. Success = gradient discovers
  the state transition function.
- **Tier 4b**: "accumulator with decay" — event adds a value, decay rate 0.9.

### Entry Points

- New file: `nsynth/src/synthesis/stateful.rs`.
- New benchmark category: `stateful` with the trace-valued spec.
- `SoftStatefulProgram::forward_trace(trace: &[...])` computes per-step loss.

### Estimated Effort

~1-2 weeks. Threading state through the soft forward pass is moderate; trace
loss aggregation is straightforward.

---

## Stage 5 — Event Dispatch with Per-Handler Composition

### Problem Statement

A game dispatches on input events (key press, tick, collision). Each event
type invokes a different handler. Synthesizing this as a monolithic soft
program forces the soft router to choose between N mutually exclusive paths,
and gradient on the wrong paths is zero.

### Approach (builds on Stages 3-4)

1. **Explicit dispatcher.** Synthesize a pure function `event_code -> handler_id`
   first. This is usually a small multi-branch scalar function, easily
   gradient-solvable.

2. **Per-handler synthesis.** For each handler, synthesize independently using
   the Stage-3 library: handlers can call previously-synthesized utilities.

3. **Stateful dispatcher loop.** Wrap dispatcher + handlers in a Stage-4
   stateful wrapper: `(state, event) -> (state, output)` calls
   `handler[dispatch(event)](state, event)`.

### Key Design Decisions

- **Dispatcher is a pure function.** No state dependence. Keeps the dispatch
  path cheap to synthesize.

- **Handlers see full state.** Each handler gets `(state, event_args) -> 
  (new_state, output)`. Handler specifications come from sliced traces: trace
  entries where the dispatched event matches.

### Benchmark Milestones

- **Tier 5a**: 3-state game state machine (menu, play, game-over) with
  transitions on START, TICK, DIE events. A 30-step trace pins each state
  transition explicitly.

### Entry Points

- Reuses `SynthesisLibrary` (Stage 3) and `SoftStatefulProgram` (Stage 4).
- New orchestrator function: `solve_event_dispatched_program`.

### Estimated Effort

~1-2 weeks on top of Stages 3-4. The architectural pieces are in place;
synthesis is composition of existing primitives.

---

## What Remains Out of Reach After Stage 5

A full playable game has:
- Rendering to a frame buffer (requires array-valued state, gradient over
  spatial writes).
- Physics (continuous-time integration with sub-step stability).
- Entity management (dynamic allocation, collection iteration).
- Audio and asset loading.

None of these are addressed by Stages 1-5. The honest position: after Stage 5,
we will be able to synthesize game *logic* (state machines, dispatch, small
loops). Rendering and physics would need further research, probably 6 months+.

## Prioritization

Of Stages 3-5, **Stage 4 (state-bearing) is the highest-value single
addition.** It opens up an entire class of problems (counters, accumulators,
finite state machines) without requiring cross-function composition. Stage 3
(library) is second-highest because it unlocks genuine compositional synthesis
in a way nothing else does. Stage 5 (event dispatch) is the smallest
increment on top, so it naturally lands last.

## Out-of-Scope for This Document

- Concrete Rust module layouts beyond the stubs above.
- Specific parameter counts for the scaled architectures.
- Benchmark corpus for stages 3-5 — deferred until implementation begins.
- Integration with the existing orchestrator and memory system.

These are implementation details that belong in the session that actually
begins Stage 3/4/5 work.
