# Stateful Synthesis Status — what works, what doesn't, what it would take

> Authored after the Rung 10/30× expansion. Honest about scope: the
> "memory cell" + "register" + "tensor 3D/4D" ask is partially delivered
> today and partially blocked on the search teachers.

## What works today (verified end-to-end through `mog_synth`)

| Surface | Status | How it's used |
|---|---|---|
| Soft-register machine | `synthesis::synthesize_register_machine` — a pool of `f64` registers, soft-write blending, soft-select on return | Discovered via `synthesis::register_machine` route; lives in the post-enumerative pipeline |
| 2-arg `(state, event) -> state` reducer | `score_tracker` and `game_tick` benchmarks, plus the new `ema_state` and `memory_cell` | Solved by `search_single_branch` / `search_two_branch` in 0.0s on the per-tick reducers |
| Per-tick frame with `array_max` | `tensor_3d_per_frame` (renamed-shape: 1-arg `frame`, max over components) | Solved by `arr_gradient` |
| Memory primitives in the canonical lore | Pre-existing `interactive_two_register_accum_*` family (kadane, branched accumulators) | Discovered by the `interactive_*` teachers — used for streaming accumulation problems |
| ProgramV3 v3 stateful synthesis (executive side) | `tools/registry/executor.py::execute_program_v3` (running counter, running max with reset) | Trustless registry; the v1/v2/v3 *executors* all work; only the **synthesizer** for v3 lives in kernels/npcot_wasm, not in nsynth |

## What is **not** in scope today

- **A `(state, frame_array) -> state` reducer that uses array ops inside the body.** The current search teachers cover `(scalar, scalar) -> scalar` and `(array) -> scalar` but not the cross product. The user's "3D coord per-frame" problem lives at this signature and the 2-arg shape needs a new search space (covered below).
- **A true `tensor<3, 4>` type in the Mog AST/executor.** Today the Mog surface has `i64`, `f64`, `bool`, `string`, `[i64]`, `Point`/`Rectangle` (structs), and that's it. A 3D/4D tensor type would mean a new `Value` variant, new executor ops (broadcast, dot, matmul), and new teachers.
- **Per-component state registers** — e.g. `state: [f64; 3]` where each component persists independently across ticks. The `tensor_3d_per_frame` benchmark above is the *outer* shape (one scalar memory cell per call) but the per-axis state is out of reach without a new search space.
- **A "time" lane in the registry** — there's no `t` argument in the current benchmark shape, and no search teacher for it.

## What it would take to close the gap (sequenced, with effort estimates)

### Stage 1 — `(scalar, array) -> scalar` reducer (small)

* The existing `search_array_compose` and `search_array_feature_dnf` teachers cover the array half. Add a small wrapper teacher that for a 2-arg `(i64, [i64]) -> i64` problem enumerates `f(state, arr) = state op g(arr)` for `g` from a small set (`sum`, `max`, `min`, `count_positive`, `count_zeros`, …) and `op` from `{+, -, *, /, max, min}`.
* Effort: 1 day. Solves a class of "state + per-frame array contribution" problems in one shot.
* Coverage: ~10–20 new benchmark problems.

### Stage 2 — `tensor<N, dtype>` type in the AST/executor (medium)

* Add a `Value::Tensor { dims: Vec<usize>, data: Vec<f64> }` variant.
* Add executor ops: `tensor_zeros`, `tensor_add` (broadcast), `tensor_get`, `tensor_set`, `tensor_dot`.
* Add a `Tensor<f64, N>` Mog syntax that parses to a call into those builtins.
* Effort: 1 week minimum. The grammar is the easy part; the verifier, the array gradient teacher, and the search teachers all need a tensor lane to learn over.
* The `(state: Tensor, frame: Tensor) -> Tensor` shape then becomes a 1-arg-tensor + 1-arg-tensor problem — well within Stage 1's pattern.

### Stage 3 — Per-component state registers (medium)

* The executor already supports struct state. Extend the search teachers to describe state machines as struct-update rules with carry-forward.
* Effort: 1 week. Most of the lift is the verifier's understanding of struct state, not the teachers.
* The `(state: Point, frame: [i64; 3]) -> Point` shape (3D position + 3D velocity = 3D position per tick) becomes one enumeration of struct-update rules.

### Stage 4 — Time / temporal index lane (large)

* Add a synthetic time argument `t: i64` to the canonical per-tick signature and a search teacher for the differential update `state = f(state, frame, t)`.
* Effort: 1–2 weeks. The frontend (signature, executor) is easy; the search teachers have to enumerate across `t` as well as the existing state args.
* The output unlocks periodic functions (`state = (t % 60) * 6` for a clock), event-triggered updates (`if t % 7 == 0 { state += 1 }`), and time-aware animations.

## Why we did not ship these in the 30× expansion

* The nsynth search teacher catalogue is the bottleneck: each new
  "shape" is gated on a teacher that can enumerate that shape and verify
  on the I/O examples. Adding 3D/4D tensors without first growing a
  `(scalar, array) -> scalar` teacher would put expensive new types in
  front of a search space that can't even synthesize the existing 2-arg
  stateful problems.
* The user's "as much as it wants to allocate" budget is a real
  constraint — pushing Stage 2+ to "ship a real 3D game-loop
  synthesizer" is the kind of project that warrants a dedicated
  research sprint, not a follow-up in a robustness session.

## What to do next

1. Land Stage 1 (the wrapper teacher). It's a half-day of Rust and
   unblocks a real class of stateful games.
2. Write a Stage-2 plan that names a single concrete target (e.g.
   "a synthesized 3D position-from-velocity program that the GPU
   executes on the ARM64 cores") so the work is scoped to a
   shippable artifact.
3. While Stage 1/2 are in flight, add Stage 3/4 as parallel tracks
   — they share the `state` register shape but the search spaces
   are independent.
