# Synthesized Software: A Verified-Synthesis Stack from Skill Formats to a Running Game

## 20. Synthesized Software

Sections 14--17 established the solvers: gradient descent over soft programs, grammar-constrained synthesis, and a portfolio router that reaches 100% coverage on its benchmarks. A solver, however, is not yet a system. This section documents the layer that turns verified synthesis into deployable software: a family of portable skill formats with strict versioning discipline (v1/v2/v3), vocabularies mined from data rather than hardcoded, three persistent memory banks --- including a *negative* memory that makes the search physically unable to repeat a failed program --- four delivery surfaces (browser WASM, HTTP API, a trustless skill registry, and an MCP server), and a case study that exercises the whole stack end to end: a complete, playable Pong whose every rule of game logic was discovered from input/output example pairs and verified by exhaustive domain sweep, with no human writing a single line of the logic.

The unifying contract across every layer is the same one introduced in Section 15: **a synthesis answer is either proof-carrying or an explicit refusal.** Every component below either returns a program that has been executed against every provided example, or says it cannot --- no layer ever emits unverified code.

### 20.1 The Skill Format Family: v1 → v2 → v3

The NPCoT browser runtime (`kernels/npcot_wasm`, Section 15) stores discovered programs as a tiny typed IR rather than as source text. Two format upgrades shipped this cycle, each widening the program space while preserving a versioning discipline that earlier formats' loaders can rely on.

**v1 (baseline).** A scalar fold pipeline: `init → transform → reduce → post-scale → offset` over a sequence of numbers. The full v1 space is 216 programs --- small enough to enumerate exhaustively, which is exactly what the equivalence tests below exploit.

**v2: multi-field records and mined predicate guards** (commit `b0622a5`). ProgramV2 prepends two stages to the v1 pipeline:

* **combine** --- a k-field record (arity 1--4) is collapsed to a scalar by field select, sum, product, difference, absolute difference, min, or max. This is what lets a skill consume *structured* observations (a 2-field point, a 4-field row) rather than bare numbers; dot products and Manhattan distances become discoverable.
* **guard** --- a per-element include/skip predicate (`always`, `> t`, `< t`, `|v| > t`, `== t`) whose threshold candidates are **mined from the training examples** rather than drawn from a fixed list (Section 20.2). A guarded mean divides by the *included* count, not the total.

The synthesizer `synthesize_program_v2` searches combine × guard × threshold × transform × reduce × post × init --- roughly 43k--112k candidate programs depending on arity --- in milliseconds, with the same honest-refusal contract as v1. Any discovered program can be rendered by `program_source_v2` as a complete working function in **five languages** (Rust, Python, JavaScript, C, TypeScript): the artifact is an IR, and source code is a deterministic rendering of it.

**v3: stateful skills synthesized from traces** (commit `d4bb6a5`). ProgramV3 implements skill = (state, input) → (state′, output): a v2-style per-step pipeline plus one persistent `f32` state cell,

```
v  = combine(fields)                  (v2 combine vocabulary, arity 1..4)
if reset_guard fires on v:            (v2 guard comparisons; thresholds mined;
    s ← init, included ← 0             "never" is the stateless default)
s  ← reduce(s, transform(v))          (v1 op vocabularies)
y  = output_select(s, v) + offset     (s, v, s+v, s·v, |s|; offset fitted
                                       closed-form on the outputs)
```

This unlocks counters, debouncers, running statistics, and resettable aggregates --- interactive programs, not just stateless folds. Training examples for v3 are **traces**: sequences of (input record, expected output) steps. Synthesis enumerates the bounded space --- at most 8 combines × 49 reset options (1 + 4 comparisons × ≤12 mined thresholds) × 3 inits × 6 transforms × 4 reduces × 5 output selectors = **141,120 candidates** (17,640 at arity 1) --- and accepts only candidates that reproduce *every step of every trace* within tolerance, with state restarting between traces. A two-step-delay trace that no single state cell can express pins the honest-refusal path.

**Versioning discipline.** Three invariants are enforced across the family, each pinned by a test:

1. **The lower format is an exact special case of the higher one.** v1 lifts to v2 with arity = 1, combine = field-select, guard = always; an exhaustive test proves v1/v2 engine equivalence across all 216 v1 programs. v2 lifts to v3 with reset = never, output = state; a grid test pins final-step v3 output equal to `execute_program_v2` across all combine × guard × init × transform × reduce × post combinations.
2. **Old loaders fail closed.** A library that uses v2 capabilities is serialized with `"format": 2` and a `program_v2` key, which a v1 loader rejects cleanly (its parser requires `program`) instead of silently mis-executing a guarded program; v3 exports use `"format": 3` and `program_v3`, rejected by both earlier loaders. Wrong answers are structurally impossible; the failure mode is a load error, never a miscomputation.
3. **Exports demote to the lowest loadable format.** A pure-v1 library still exports as v1 and loads everywhere; a v3 program with reset = never and output = state demotes back to v2 on export. Capability is paid for only when used.

The runtime's test suite grew from 14 to 24 tests with v2 (dot-product and Manhattan-distance discovery, mined-threshold guards, guarded-mean semantics, the v1/v2 equivalence sweep, round-trips, fail-closed rejection, refusal on cross-field conditionals, 5-language rendering) and to **32 tests** with v3 (running-counter discovery, running-max-with-mined-reset across multiple traces, the v2/v3 lift grid, format demotion, arity-mismatch refusal). The full runtime --- v1+v2+v3 execution, synthesis, serialization, and library management --- compiles to a **133,262-byte** WASM binary (`wasm-pack build --target web --release`), small enough that the browser tier of the cascade ships as a single fetch.

### 20.2 Mined Vocabularies: Nothing Hardcoded

A recurring failure mode in template-based synthesis is the hidden hand-tuned constant: the magic threshold, the blessed list of "useful" numbers. Three places in the stack replace such lists with vocabularies mined from the problem's own data:

* **Constants** (nsynth): `discover_useful_consts` builds the per-problem constant pool from example inputs, outputs, scalar arguments, and pairwise differences, anchoring only {0, 1, −1}. Gradient descent starts with constants matched to the problem's own scale.
* **Guard thresholds** (v2): predicate thresholds are candidates mined from the example values --- the dot-product and guarded-mean tests discover thresholds no fixed list would contain.
* **Reset thresholds** (v3): the same `mine_thresholds` machinery feeds the reset guard. The running-max-with-reset regression test pins a threshold of −8/−10 mined from trace data, in a setting where the anchored zero provably cannot produce the right resets --- the test fails unless mining works.

The principle is the one that runs through the whole stack: the synthesizer's vocabulary is data-driven, so its competence grows with the problems it sees rather than with the foresight of whoever wrote the templates.

### 20.3 Three Persistent Memory Banks, Including Negative Memory

The nsynth solver carries cross-run memory in three on-disk banks, each independently overridable (and disableable) by environment variable:

| Bank | Direction | Default path | Env var |
|---|---|---|---|
| Solved programs | positive | `~/.nsynth_solved_programs.json` | `NSYNTH_CACHE_PATH` |
| Learned biases | procedural | `~/.nsynth_learned_biases.jsonl` | `NSYNTH_BIAS_BANK_PATH` |
| Rejected programs | **negative** | `~/.nsynth_rejected_programs.tsv` | `NSYNTH_REJECTED_PATH` |

The **solved bank** maps an examples-fingerprint to a verified program; hits are re-verified against the current examples before being returned (fingerprint collisions and stale rows fail closed), so repeat requests come back in milliseconds with the proof obligation re-discharged. The **bias bank** stores the gradient-solver initializations that led to past successes and replays the highest-scoring ones first on new problems; warm replays turn cold ~76 s gradient solves into ~29 ms answers (`ncpu/mcp_server/README.md`; measured in the bank's commit history). Both were described in earlier engineering notes; what is new this cycle is the third bank.

**Negative memory** (commit `1d2ed2f`, `nsynth/src/rejected_cache.rs`). During gradient synthesis the discretizer emits thousands of candidate programs; each one that fails strict verification is *deterministic dead weight* for that exact example set --- verification is a pure function of (code, examples), so a rejected candidate can never become correct on a rerun. Previously those rejections lived in a per-call `HashSet` and died with the process; every rerun re-parsed and re-executed the same dead programs. The `rejected_cache` module persists rejection fingerprints across runs, keyed by the same examples-fingerprint the solved bank uses:

* Candidates are stored as 128-bit double-FNV hashes of their source, not as source text --- thousands of rejections per problem stay cheap on disk, and a hash collision merely skips one candidate (the restart cascade explores neighbors; coverage risk at $2^{-128}$ per pair is ignorable).
* A known-bad candidate is skipped **before the Mog parser ever sees it**.
* Because verification is deterministic, rejections never expire. Growth is bounded instead by caps: 4,096 hashes per fingerprint and 512 fingerprint rows with LRU eviction.
* The bank is wired into the universal-array gradient path via an RAII `RejectionRecorder` that flushes on every exit path, so partial searches still contribute their rejections.

**Measurement.** On a novel out-of-space problem (one the portfolio cannot solve, so the search runs to exhaustion), the cold run persisted **1,415 rejections**; the warm rerun added **zero** --- the entire failed search was deduplicated across runs. The module carries 6 unit tests (persistence round-trip, cap eviction, per-fingerprint isolation), and the universal-array suite stays green with the bank active.

Together the three banks give the synthesizer the three kinds of memory a learning system needs: *what worked* (solved), *how to look* (biases), and *what to never try again* (rejected). Section 20.5's registry sketches the federated version of the same triple.

### 20.4 Case Study: Pong, Synthesized from Example Pairs

The centerpiece demonstration is the `/pong` page on ncpu.ai: a playable Pong in which **every game-logic function was discovered by the synthesizer rather than written**. The human-authored part is only the canvas/event loop. The complete toolchain, provenance, and verified artifact are committed at `tools/pong_synthesis/` (the page itself lives in the sms-hub repository, commit `7c46bc9`).

**The inputs were example pairs only.** For each rule, a contract was designed --- a name, an integer signature, and 6--16 input/output examples --- and fed to `mog_synth --problem-json`. No code, no pseudo-code, no natural-language description of the rule. Two of the actual contracts, verbatim from the committed artifact `pong_rules_final.json`:

| Rule | Training input (literally all of it) | Discovered program |
|---|---|---|
| `next_pos` | (10, 2)→12, (5, −3)→2, (0, 4)→4, (100, −7)→93, (42, 0)→42, (799, 11)→810, (−5, −11)→−16, (600, 1)→601 | `return (b + a)` via scalar-expression search |
| `hit_top` | (0)→1, (−1)→1, (−5)→1, (1)→0, (3)→0, (100)→0, (600)→0, (−20)→1 | a gradient-discovered ≤ 0 test |

**Final tally: 22 rules --- 14 synthesized directly, 8 composed** --- every one swept with zero mismatches over its rule's full reachable game domain, from the exhaustive 4-case boolean tables (`flag_and`, `flag_or`) up to 160,801-case two-argument integer grids (`sub2`, `min2`, `max2`). Sweeps are bounded to the game's reachable physics (|ball velocity| ≤ 11, an 800×600 field), and every per-rule case count is recorded in the artifact.

**CEGIS catches the impostors.** Training examples alone are not the verification story. After a candidate verifies on its seed pairs, a counterexample-guided loop (CEGIS) sweeps it over the rule's full reachable domain against a reference; mismatches become new training examples and the synthesizer runs again. This loop caught two overfit programs that would otherwise have shipped:

* `sub2` first came back as a branchy program fitting only its 6 seed pairs --- **40,100 of 160,801** domain cases disagreed with true subtraction. With 10 counterexamples folded into the example set (visible as the −300-family rows in the committed artifact), the gradient solver produced true `a − b` on the next iteration (0.7 s for the repair solve; see `tools/pong_synthesis/README.md`).
* An overfit `gte` was caught the same way --- and was ultimately not re-synthesized at all, but *composed* (below).

The impostor catches are the point, not an embarrassment: they are direct evidence that example-fit alone is an inadequate acceptance criterion and that the domain-sweep discipline does real work. A synthesis pipeline without the sweep would have shipped a subtraction function that is wrong on 25% of its domain.

**Composition as emergent skill reuse.** Eight of the 22 rules were never synthesized as fresh programs; they are pure wiring of already-verified skills, discovered when the direct solver exceeded budget:

```
gte(a, b)              = hit_top(sub2(b, a))          (a >= b  iff  b - a <= 0)
max2(a, b)             = neg(min2(neg(a), neg(b)))
abs2(v)                = neg(min2(neg(v), v))
reflect_x(vx, hit)     = select(hit, neg(vx), vx)
grow(v)                = select(gte(v,1), next_pos(v,1), next_pos(v,-1))
crossed_right(p, n, c) = flag_and(gte(c, next_pos(p,1)), gte(n, c))
score_if_out_right     = next_pos(score, exited_right(ball_x, w))
score_if_out_left      = next_pos(score, exited_left(ball_x))
```

The `gte` composition deserves emphasis: the system replaced a failed direct synthesis with an algebraic identity over two skills it had already verified --- a ≥ b iff b − a ≤ 0 --- exactly the kind of library reuse the NPCoT skill-cache design (Section 15.3) postulates. Composed rules are swept over the same domains as direct ones; the page labels them `composed:` and lists which primitives they reuse.

**What the discovered programs look like.** Several directly synthesized rules are *not* the program a human would write. `hit_top` is a digit-extraction loop (`while x > 0 { x = x / 11; acc = acc / 2 }`) that happens to implement "≤ 0" exactly, over the entire swept domain. This is the verification semantics stated honestly: the claim for each rule is extensional correctness over the rule's reachable domain, established by exhaustive sweep --- not intensional match with a human-intended algorithm. Within the game, the two are indistinguishable by construction; outside the swept domain, no claim is made.

**Toolchain hardening as a by-product.** Pushing synthesized Mog through to running TypeScript surfaced two transpiler correctness bugs, both fixed with regression tests in `nsynth/src/mog_transpile.rs`: Mog's `/` is truncating i64 division while JavaScript's `/` is float division, so every emitted division is now wrapped as `Math.trunc(A / B)` with left-associative folding for chains (commit `54582b7`); and the gradient backend's inline single-line `if cond { stmt; }` form is now handled in all three target languages (`inline_if_tests`). Synthesized code is adversarial input for a transpiler in a way hand-written code is not --- it explores the grammar without stylistic priors.

**Reproduction.** `node tools/pong_synthesis/finalize_pong_rules.mjs` re-merges the solved shards, re-transpiles, re-sweeps every rule (with CEGIS retry on failure), re-verifies the 8 compositions, and regenerates the site's `synthesized.ts`; it fails loudly on any domain mismatch. With the memory banks of Section 20.3 populated, the rerun is near-instant.

### 20.5 Delivery Surfaces

Four surfaces, shipped this cycle, expose the same verified-synthesis core at different points of the cost/capability cascade (browser tier → heavy synthesizer → LLM, with refusals escalating and verified wins distilling back down):

| Surface | Path | Contract | Tests |
|---|---|---|---|
| Browser WASM runtime | `kernels/npcot_wasm` | v1/v2/v3 synthesis + execution in 133 KB; refuses anything outside its space | 32 (cargo) |
| Synthesis API | `ncpu/synthesis_api` | stdlib-only HTTP server over the nsynth binary; solves or refuses, transpiles wins to Python/Rust/TS; `/stats` exposes the three memory banks | 21 (pytest, live server) |
| Verified-skill registry | `tools/registry` | "crates.io for synthesized programs": the server **re-executes every submission against its claimed examples** before storage; rejection returns the concrete counterexample; `--verify-all` replays verification over the whole database | 19 (pytest) |
| MCP server | `ncpu/mcp_server` | natural language → mined I/O examples → verified program, over stdio MCP for Claude Code/Desktop, Cursor, etc.; refusals carry guidance, never code | 24 (pytest, real subprocess) |

Two properties are worth singling out. The **registry's trust model is verification, not reputation**: because every NPCoT program is a microseconds-cheap deterministic IR, the server can afford to re-run each submission over all of its claimed examples (via a pure-Python executor mirror ported from the canonical Rust executor, with pinned-output tests guarding the port); wrong code physically cannot enter, anonymous and maintainer submissions pass the identical gate, and the entire store is re-checkable forever --- even against direct database tampering. Its `/library.json` endpoint emits the registry as a loadable NPCoT library with the same format discipline as Section 20.1 (pure-v1 stays v1; any v2 skill lifts everything to format 2, which v1 loaders reject rather than mis-execute), so the browser WASM runtime, the native executor, and the Python server all consume it directly --- the cross-runtime interoperability is itself pinned by a Python↔Rust fingerprint cross-check in the MCP suite.

The **MCP server closes the natural-language loop without surrendering the contract**: its prompt parser mines I/O pairs from arrow notation, doctests, asserts, and "returns" prose, echoes the extracted examples back, and synthesizes from those; a prompt with no examples gets guidance ("provide concrete examples like f(2,3) → 5"), not guessed code. Every response is one of exactly two shapes --- `verified: true` with a program that reproduced every example inside the synthesizer, or `verified: false` with a reason and **no code field, ever**. An answer from this tool never needs review for hallucination.

### 20.6 Limitations

1. **Domain-bounded verification.** The Pong sweep establishes correctness over reachable game states, not over all of i64. Programs like the digit-loop `hit_top` are exactly correct on the swept domain and undefined-by-claim outside it. Widening the claim requires widening the sweep (or a proof, which the IR's small size makes plausible future work).
2. **Co-designed contracts.** The 22 rule contracts (signatures, seed examples, domain bounds) were designed by a human; the synthesizer discovered the programs, not the decomposition of Pong into rules. Automating the decomposition is open.
3. **Single state cell.** v3 covers counters, debouncers, and resettable aggregates; programs needing two interacting state variables (the pinned two-step-delay refusal) are out of space by design and escalate up the cascade.
4. **Executor mirroring.** The registry's Python executor is a hand-ported mirror of the canonical Rust executor. Pinned-output tests guard the port, but the duplication is a standing risk; generating both from one definition is preferable.
5. **One-shot measurements.** The negative-memory numbers (1,415 cold / 0 warm) are from a single novel-problem run, and bank speedups vary with problem mix. The claims here are existence proofs of the mechanism, with the regression tests as the durable artifact.

### 20.7 Outlook: A Program-Native Model

The stack above is deliberately model-free: enumeration, gradient descent, memory banks, and verification. The design document `docs/native_synthesis_model.md` (ROADMAP Rung 9) describes where it points --- a coding model in which **reasoning happens in program space, and natural language and source code are input/output renderings, not the medium of thought**. The model's loop is encode task → propose a program in the typed IR → execute it (differentiably in training, exactly at inference) → verify → transpile; the output path has no generative freedom, so it cannot hallucinate. Nearly every component already exists in embryo in this section's stack: the example encoder (the mog_synth meta-learner), the program prior (the gradient solvers, whose learned-bias bank is literally an amortized proposal distribution being collected today), the executor and transpiler (zero learned parameters), and the library attention (the solved bank and registry). The document's further themes --- JEPA-style speculation in program-latent space ("imagination before execution") and recursive, tree-structured networks matched to the tree structure of programs --- are future work; the present section is the substrate they would train on. The bet is that a sub-1B model whose every answer is verified or refused can, on the program space it covers, beat any token-predicting model at any size.

### 20.8 Reproducibility

| Claim | Command | Artifact / pin |
|---|---|---|
| v2 + v3 formats, 32 tests, fail-closed versioning | `cd kernels/npcot_wasm && cargo test` | commits `b0622a5`, `d4bb6a5`; 133,262-byte `pkg/npcot_wasm_bg.wasm` |
| Negative memory dedupes failed search | `cargo test rejected_cache` (in `nsynth/`) | commit `1d2ed2f`; 1,415-cold/0-warm measurement procedure in `REPRODUCIBILITY.md` |
| Pong: 22 rules, zero sweep mismatches | `node tools/pong_synthesis/finalize_pong_rules.mjs` | `tools/pong_synthesis/pong_rules_final.json` (byte-identical regeneration) |
| Synthesis API solves-or-refuses | `python3 -m pytest tests/synthesis_api/ -q` | 21 tests, live server |
| Registry trustless gate | `python3 -m pytest tests/registry/ -q` | 19 tests; `--verify-all` trust sweep |
| MCP NL→verified program | `python3 -m pytest tests/mcp_server/ -q` | 24 tests, real stdio subprocess |
