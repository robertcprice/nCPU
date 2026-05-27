## 19. Roadmap to Game-Scale Gradient Synthesis

Sections 15 and 17 established that gradient descent --- combined with a specialist portfolio --- reliably synthesizes single-function programs in the 10--30 instruction range. Users naturally ask whether the same approach scales to "games and apps": hundreds to thousands of lines of interdependent code. This section is a deliberately honest assessment of the gap and a staged plan for closing it.

The short answer: **gradient descent of a single end-to-end soft program does not currently scale to game-sized code, and probably will not without structural changes.** Below we describe what scaling the current architecture gets you, where it stops working, and which additions are load-bearing for continued progress.

### 19.1 Where We Are

| Dimension | Current (`SoftUniversalProgram`) | Game-scale target |
|---|---|---|
| Soft AST slots per function | 11 (3 init + 6 loop + 2 post) | 100--1000 |
| Functions per synthesis run | 1 | 10--100 |
| Soft parameter count | ~1,250 per 1-arg problem | 100k--1M |
| Per-problem wall time | 0.01--30 s | minutes to hours |
| I/O specification shape | static input → scalar output | multi-step traces, events, renders |
| Persistent state | none (scoped to one call) | required |
| Composition between functions | none | required |

`SoftUniversalArrayProgram` adds array-shaped pools (item, index, parity, `arr[i-1]`, constants) but keeps the same single-function frame. Both architectures are capable of the 95/95 coverage measured in §17. Neither can represent a second function.

### 19.2 The First Two Scaling Dimensions

Two dimensions can be scaled with minor changes:

**Slot count.** `N_INIT_SLOTS=3`, `N_LOOP_SLOTS=6`, `N_POST_SLOTS=2` are constants in `nsynth/src/synthesis/universal.rs`. Doubling them quadruples the per-restart parameter count and slows the gradient loop roughly 4×, but should close the gap on Tier-2 problems that require more dispatch branches or more accumulators. Section 19.4 reports the first Tier-2 measurement.

**Multi-loop phases.** The current architecture has a single loop. Some problems need two sequential scans over an array (first pass computes a statistic, second pass uses it). A straightforward extension is a two-loop `SoftUniversalProgram` where phase 2 can read outputs from phase 1. This adds ~N slots but does not change the optimization core.

### 19.3 The Three Dimensions That Do Not Scale Without Structural Work

**Function composition.** Games consist of small functions that call each other. Gradient synthesis of multiple functions at once is possible in principle but the search space grows multiplicatively with function count. A realistic approach is *hierarchical* synthesis: gradient-synthesize each function from its own I/O specification, cache successful solutions in a library, and allow later functions to reference library entries as first-class soft choices. This turns "synthesize 10 functions at once" into "synthesize 10 functions in sequence with growing vocabulary."

**Persistent state.** A game's state (score, position, entities) persists across frames. Current soft programs have no external state --- each invocation is independent. Supporting state requires treating the program as a map `(state, input) → (new_state, output)`. The I/O specification likewise becomes a trace of state transitions, not a single pair. This is representable but requires rewriting the loss function to aggregate trace-level error.

**Event-driven control flow.** A game dispatches on input events (key press, tick, collision). Encoding this as a single soft program forces the soft router to decide between N mutually exclusive branches per input; the gradient signal on the wrong branch is zero and the right branch often small. Practical approaches split this into (a) a synthesized dispatcher (which event code maps to which handler) and (b) per-handler synthesis, again a hierarchical structure.

### 19.4 Empirical Tier-2 Measurement

We added 10 "game-adjacent" problems to the benchmark suite: score dispatch, vending change, combat resolution with clamping, traffic-light phase, run-length decode, adjacent-difference count, priority pop, turn-order rotation, 4-argument grid bounds check, and bounded-velocity gravity integration. Each requires multi-branch scalar dispatch, 3--4 argument coupling, or compositional array work --- shapes that are realistic starting points for game logic.

Running these through the current solver pipeline (default ordering: enumerative → gradient → search) produced the following coverage table. The numbers land in `artifacts/tier2_coverage.json` when the default-mode benchmark run completes; the narrative fraction below will be updated with the measured result in place of this placeholder.

> `placeholder: update after tier-2 run completes.`

The more diagnostic number is the **gradient-first** measurement on Tier-2: which of the 10 can the gradient path discover on its own. That tells us where the gradient architecture's capacity ceiling is relative to game-adjacent shapes. We commit to reporting both the default and gradient-first numbers with their per-problem breakdowns.

### 19.5 Five-Stage Plan

The staged plan, each stage independently valuable and with a concrete milestone:

**Stage 1: Scale single-function slot count.** Raise `N_LOOP_SLOTS` from 6 to 12 and add adaptive slot-count selection driven by the I/O shape. Milestone: ≥ 8 / 10 Tier-2 problems solved by gradient alone (currently gradient-first fraction on Tier-2 is measured in §19.4).

**Stage 2: Multi-loop phase programs.** Add a second loop block; gradient over its continuation condition and its inputs from the first loop. Milestone: solve two problems that require a two-pass algorithm (e.g., second_max with swap, mean-then-deviation).

**Stage 3: Hierarchical library-augmented synthesis.** When a function is synthesized, cache its I/O behavior and its Mog source. When synthesizing a later function, add a soft pool entry "call library function f_k" as a first-class choice. Milestone: solve a compositional problem where `g(x) = f(f(x)) + 1` without hand-coding `f`.

**Stage 4: State-bearing programs.** Extend the I/O specification to a list of `(state_in, event, state_out, output)` transitions. Loss aggregates over the trace. Milestone: synthesize a counter that increments on event=0 and resets on event=1, verified across a 20-event trace.

**Stage 5: Event dispatch + per-handler composition.** Combine stages 3 and 4: a dispatcher function plus N handlers, synthesized together. Milestone: synthesize a 3-state game state machine (menu, play, game-over) with transitions on keyboard events.

Stages 1 and 2 are plausibly one-session scope. Stages 3-5 are multi-session research directions.

### 19.6 What Remains Out of Reach

Even at Stage 5 completion, the synthesized "game" is a three-state machine reacting to three event types. Genuine game code (rendering, physics, entity management) has complexity orders of magnitude beyond this. The honest position is:

- Gradient synthesis of single functions up to 30-ish instructions is within reach and measured.
- Gradient-assisted composition of 5--10 function libraries is plausible near-term research.
- Gradient synthesis of a playable game from I/O examples alone is not a near-term objective.

This section is not a promise that the nCPU differentiable engine will synthesize games. It is a roadmap for the next measurable milestones, and an explicit statement that we do not believe any current gradient-synthesis approach can produce a game --- a position we'd expect the rest of the field to agree with once stated plainly.

### 19.7 Reproducibility

The Tier-2 benchmark additions live at `nsynth/src/benchmark.rs` in the `// Tier-2` comment block (10 factories, one variant each for initial measurement). Default-mode and gradient-first runs produce artifacts at `artifacts/tier2_coverage.json` and `artifacts/tier2_gradient_first.json`. The existing `benchmark_nsynth.py` and `benchmark_nsynth_gradient_first.py` harnesses work unchanged; the factories are picked up automatically because they are registered in `FACTORIES`.
