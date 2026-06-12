# A Program-Native Coding Model (bottom-up, not bolt-on)

Design document for ROADMAP Rung 9. Status: design, with Phase A
immediately buildable from existing components.

## The inversion

Every current code LLM — and NPCoT-as-addition — keeps token prediction
as the substrate and treats programs as text to be emitted. The bottom-up
architecture inverts this:

> **Reasoning happens in program space. Natural language and source code
> are input/output renderings, not the medium of thought.**

The model's core loop is: encode the task → propose a program in a typed
internal IR → execute it (differentiably during training, exactly at
inference) → check against evidence → emit. The emitted code is a
*deterministic transpile* of the internal program — `mog_transpile`
already renders one IR to Python/Rust/TypeScript with zero learned
parameters. Nothing in the output path can hallucinate, because the
output path has no generative freedom.

## Architecture

```
 NL prompt ──► [Text Encoder]──┐
                               ▼
 I/O examples ─► [Example     [Task Latent z]
                  Encoder] ──► (shared space) ──► [Program Prior Net] ──► soft program p̃
                               ▲                        │
              [Library Attention] ◄─── verified         ▼
              (external memory of      skills    [Differentiable Executor]
               discrete programs,                 run p̃ on examples
               cosine over signatures)                  │
                                                        ▼
                                            execution-match loss / verify
                                                        │
                                        discretize (STE/Gumbel) ──► exact verify
                                                        │
                                         [Transpiler — deterministic, 0 params]
                                                        ▼
                                              Python / Rust / TS source
```

Learned components (the entire parameter budget):
1. **Example encoder** — I/O pairs → task latent. Already exists in
   embryo: `mog_synth` meta-learner v5 (TransformerEncoder mapping I/O →
   UniversalProgramDescription, 23/24 on its bench).
2. **Text encoder** — NL → the *same* latent space. Small (100M-class);
   trained on a synthetic NL↔program corpus (below).
3. **Program prior net** — latent → distribution over the typed program
   space (ops, structure, constants). The Gumbel-softmax machinery exists
   in `ncpu/differentiable/` (opcode/register distributions, temperature
   annealing); the gradient solvers in nsynth are the non-amortized
   version of this network.
4. **Confidence gate** — execution-variance → answer/refuse. Exists
   (confidence-aware gating, Section 11.10 of the paper).

Non-learned components (free):
- Executor semantics (the canonical NPCoT executor — v1/v2/v3 — plus the
  Mog register machine), exact at inference, soft during training.
- Transpiler (mog_transpile).
- Library + the three memory banks (solved / biases / rejected) — the
  bank of learned restart biases is literally an amortized proposal
  distribution being collected already.

## Why this can be tiny

A 7B code LLM spends most of its capacity memorizing surface code and
human idiom. Here, surface form costs zero (transpiler), execution costs
zero (interpreter), and correctness is enforced by verification rather
than learned. The model only learns the *mapping from task description to
program-space region* — a far smaller function. Plausible target: a
sub-1B model that, on the program space it covers, beats any LLM at any
size — because its answers are verified, and refuses outside it.

## Training: self-generated, perfect ground truth

The data engine requires no human labels and no scraped code:
1. Sample random programs from the DSL (the `random_bias_init` emergent
   generator already does this for the universal-array space).
2. Execute on random inputs → perfect I/O pairs (this is the repo's
   "perfect ground truth advantage").
3. Render NL descriptions: template-based first ("a function that returns
   the sum of squares of the inputs"), LLM-paraphrased for diversity.
   The teacher LLM contributes *linguistic variety only* — semantics come
   from execution, so this stays bottom-up: no LLM weights, logits, or
   architecture enter the model.
4. Train: encoder(s) → program prior → soft-execute → execution-match
   loss + program-reconstruction loss; discretize with straight-through;
   hard-verify; failures feed the negative bank.
5. Compounding: every real-world solve (autoresearch, MCP server,
   registry) adds (NL, examples, program) triples back into training.

## Phases

- **Phase A (buildable now):** scale the meta-learner into the Program
  Prior Net over the v1/v2/v3 + universal-array spaces. Use as tier-0
  proposer in nsynth: predict program region, verify, fall back to search.
  Metric: % solved with zero search; mean search-steps saved on the
  105-problem bench. Components: meta-learner v5 + synthetic generator.
- **Phase B:** text encoder into the same latent space (synthetic NL
  corpus). Result: NL → verified program with no LLM in the loop. The MCP
  server swaps its extraction step for this encoder.
- **Phase C:** end-to-end differentiable stack with library attention as
  external memory; joint training; confidence gate. This is the native
  model. Evaluate against the matched-size LLM baseline on
  DSL-expressible tasks.
- **Phase D:** grow the DSL (typed values, richer data structures,
  bounded recursion) with the format-versioning discipline; curriculum
  over program size; honest published scoping of what fraction of e.g.
  HumanEval is DSL-expressible at each stage.

## Risks, stated plainly

- **The DSL ceiling becomes the model ceiling.** No internal LLM fallback
  — by design. Mitigation: the external cascade remains for everything
  outside the space; the model's job is to own its space perfectly.
- **NL understanding without web-scale pretraining is the hard part.**
  Synthetic-paraphrase corpora may not cover real user phrasing.
  Mitigation: the MCP/agent deployments collect real (prompt, examples,
  program) triples continuously — the data engine targets exactly the
  distribution that matters.
- **Amortization may not generalize** beyond training program sizes.
  Mitigation: the prior proposes, search + verification dispose — worst
  case degrades to today's (already 100%-coverage) search, never below.

## Reasoning in code — the core principle (no semantic fallback)

Clarified intent: the model does not "fall back to an LLM" when a task is
outside its library, because the model has no word-thinking component to
fall back to. **Every reasoning step is a program — an object with
executable semantics — never a sentence.** Natural language exists only
at the boundary (the text encoder maps a prompt into task-latent space);
from that point inward, thought is program-space manipulation:

1. **Composition search.** A novel task is first attacked as a
   composition of known skills. Pong's `gte = hit_top ∘ sub2` was found
   mechanically; the recursive library embeddings make that search
   guided rather than blind. Composition over a verified library is the
   cheapest form of out-of-domain generalization — and it produces
   artifacts that are correct by construction of their parts plus a
   final sweep.
2. **Program rewriting as chain of thought.** Intermediate "thoughts"
   are program transformations with defined semantics: specialize an
   argument, generalize a constant into a parameter, invert a known
   skill, fuse two passes, wrap with a guard. Each step is executable,
   so each step is CHECKABLE — a chain of thought that cannot drift,
   because every link either runs correctly on the evidence or is
   discarded. This is what "Neural-Physical" in NPCoT means, promoted
   from cache-lookup to the reasoning loop itself.
3. **Sketch-and-fill.** The prior net proposes program *sketches* —
   structure with typed holes — and holes are closed by local search /
   gradient descent (the nsynth machinery). Thinking = refining an
   executable hypothesis, not narrating one.
4. **Abstraction growth (wake-sleep).** When several library programs
   share structure, anti-unify them into a new parameterized primitive
   and add it to the DSL — the system grows its own language from its
   own solutions (the DreamCoder insight, here with NPCoT's verification
   discipline and persistent banks). This is the deep answer to
   "out of known domain": **the domain is not fixed.** Every solved
   frontier task becomes vocabulary; the reachable space expands
   bottom-up, permanently, with proofs.
5. **JEPA intuition over semantics of code, not words** (next section):
   predicted execution outcomes prune the search — the system "imagines
   running it" rather than "talks itself through it."

The external LLM cascade (MCP verify_candidate tier) remains as
*bootstrap scaffolding* while the native model matures — every
client-LLM draft that passes verification becomes training data and
library content for the code-native reasoner that replaces it.

## JEPA integration — imagination before execution

The repo's JEPA machinery (ncpu/jepa_neural_cpu, ncpu/world_model,
kernels/rust_metal/src/jepa) predicts machine-state transitions in latent
space. In the native model it becomes the **executor's fast approximate
twin**:

- **Latent speculation:** before running a candidate program exactly, a
  JEPA predictor estimates its execution outcome embedding from (task
  latent, program embedding). Candidates whose predicted outcome is far
  from the target are pruned without execution — thousands of candidates
  scored for the cost of one forward pass. The exact executor only runs
  finalists; verification stays exact, so soundness is untouched (JEPA
  prunes, never approves).
- **Stateful traces (format v3):** JEPA's native job is next-state
  prediction — exactly the (state, input) → state' shape of v3 skills.
  Train the predictor on traces of soft-program execution; prediction
  error doubles as an anomaly detector flagging library entries whose
  behavior drifts from their fingerprint.
- **Perfect ground truth:** JEPA training data is free and unlimited —
  the exact executor generates as many (state, program, next-state)
  triples as wanted. This is the paper's bottom-up JEPA thesis applied to
  the synthesis stack.

## Recursive (tree-structured) networks — programs are trees

Programs are ASTs; recursive networks are their natural geometry, and the
Pong build surfaced exactly the phenomenon they model:

- **Recursive program decoder:** instead of emitting a flat op-vector,
  the Program Prior Net grows the AST recursively — each node's
  distribution conditioned on its parent's embedding. Nested structure
  (if-inside-while, composed calls) gets first-class treatment instead of
  fixed slots; this is the principled path past the hardcoded
  `N_ARR_BODY = 4` body-slot architecture.
- **Compositional library embeddings:** a recursive encoder makes a
  composed skill's embedding a *function of its parts* —
  embed(gte) = g(embed(hit_top), embed(sub2)) for the Pong composition
  gte(a,b) = hit_top(sub2(b,a)). Retrieval then generalizes
  compositionally: a task near "comparison" can surface a composition of
  cached pieces that was never explicitly taught. The 8 composed Pong
  rules are the hand-built existence proof; the recursive encoder
  automates discovering such reuse.
- **Bounded recursion in the DSL (Phase D)** pairs with this decoder:
  self-similar programs handled by a self-similar network.

## Relation to the rest of nCPU

This is where the two halves of the project meet: the neural computer
supplies differentiable execution substrates (neural ALU, soft truth
tables, JEPA dynamics for latent speculation); the synthesis stack
supplies the program space, verification discipline, and memory banks.
The native model is both pillars in one artifact.

## Phase A implementation notes (2026-06-11)

Phase A is built and measured. The Program Prior Net runs as an
env-gated tier-0 proposer inside nsynth's universal-array fallback.

**What was built, where:**

- **A1 — data generator** (`nsynth/src/synthesis/universal_array/prior_gen.rs`,
  bin `nsynth/src/bin/gen_prior_data.rs`): samples programs from the
  universal-array space — the 25 hand-coded restart shapes (extracted
  verbatim into `apply_handcoded_restart_bias`, now shared with the solver
  cascade) and the `random_bias_init` emergent sampler, 40% of rows with
  extra random spike mutations — executes the *discretized* Mog code through
  the exact verifier runtime, and emits `(examples -> discrete description)`
  JSONL. `describe()`/`from_description()` give a lossless discrete round
  trip over the soft program space (unit-tested for all 25 shapes ×
  n_scalar 0..2). Generated 100,000 clean rows in 105 s (712,798 attempts:
  218,551 exec-error, 109,106 constant-output, 285,129 dup-cap, 12 magnitude
  rejections; 41,688 distinct programs; 76% hand / 24% random after
  filtering; n_scalar 31k/32k/37k). Stats:
  `artifacts/prior_net_gen_stats.json`. Dataset gitignored (regenerable).
- **A2 — prior net v0** (`nsynth/scripts/prior_net/prior_net_model.py`,
  `training/prior_net/train.py`): 3,461,961-param TransformerEncoder
  (linear per-example embed → 4 layers, d_model 256 → 60 classification
  heads over slots/body-inits/return/constants; lineage: v5 meta-learner).
  Trained on MPS, 95k/5k split, early stop epoch 18 (~12 min). Held-out
  slot accuracy: op 91.2%, cmp 89.2%, else 88.7%, gate 85.7%, src 85.6%,
  body_init 75.0%, ret 67.9%, const 41.9%; full 60-head exact match 1.9%.
  Checkpoint: `training/prior_net/prior_net_v0.pt` (13.9 MB) + eval report.
- **A3 — tier-0 wiring** (`nsynth/scripts/prior_net/propose.py` +
  `synthesize_universal_array_fallback`): with `NSYNTH_PRIOR_NET=1`, the
  solver subprocesses propose.py once per problem (K=4 proposals: argmax +
  3 temperature samples, n_scalar-masked), tries each zero-step
  (discretize+verify, method `prior_net`) then warm-refines ≤120 Adam steps
  (method `prior_net_warm`). Verified-or-discarded; all failures fall
  through to the existing cascade; flag unset = byte-identical default.
  Stub-bridge unit test covers the full subprocess round trip.
- **A4 — eval** (`training/prior_net/eval_phase_a.py`,
  `artifacts/prior_net_phase_a.{json,md}`,
  `tests/test_prior_net_phase_a.py`): 105-problem bench OFF/ON with fresh
  isolated banks, plus a direct head-to-head on the 16 universal-array
  problems with the search stages bypassed
  (`gen_prior_data --eval-fallback`).

**Measured result (honest, mixed):**

- Full bench: **105/105 both runs** (coverage cannot regress — pinned by
  regression test). The tier-0 never fires on the standard bench: the 2026
  search-teacher catalog pre-empts the universal-array fallback on all 105
  problems (bench wall 57.3 s OFF vs 61.3 s ON, delta within contention
  noise). The integration result on this bench is **null by pre-emption**,
  not by failure.
- Direct fallback head-to-head (16 problems, fresh banks): 16/16 solved
  both ways. **Zero-search solves: 2/16** — the net read raw I/O examples
  and emitted verbatim-correct programs for `longest_increasing_run`
  (9.04 s → 1.56 s) and `count_peaks` (7.12 s → 3.26 s). The other 14
  proposals missed; each miss costs ~2-4 s (bridge subprocess + 4 × 120
  warm-refine steps), netting +33.3 s total (83.5 → 116.9 s). The prior
  *works* — amortized I/O→program inference verifiably solves real bench
  problems with zero search — but at v0 hit rate (12.5%) the overhead
  exceeds the savings.

**What's next (Phase A follow-ups before Phase B):** raise the hit rate
(train on the bench-problem distribution via solved-cache replay, predict
constants conditioned on `discover_useful_consts`, K>4 with batched
inference), cut the overhead (persistent bridge process instead of
per-problem model load; skip warm-refine when proposal confidence is low),
and gate tier-0 on predicted-success probability so misses cost ~0.

## Phase A v1 notes (2026-06-12): confidence gate + persistent server

v1 took v0's +33 s overhead loss and flipped it to a measured net win,
with the same 2 zero-search wins and full coverage. Three changes:

- **Persistent async proposer server.** `propose.py --serve` loads torch +
  the checkpoint once and speaks one-line-JSON request/response over
  stdin/stdout. The Rust side (`prior_gen.rs`) keeps the child in a
  process-global state machine (`Untried -> Spawning -> Running/Failed`);
  the spawn + ready handshake runs on a background thread, and requests
  arriving before ready return `None` (cascade proceeds), so startup costs
  ~0 wall time. Measured (`artifacts/prior_net_proposer_cost.json`):
  startup 2.55 s idle (12 s contended), then **4.2 ms median per request**
  vs v0's one-shot subprocess at 0.68 s idle / ~7 s contended per problem —
  a ~160-1,600x per-problem cut. Fail-soft: any protocol failure demotes
  the server to `Failed` for the process lifetime (no respawn storm).
- **Calibrated confidence gate.** `calibrate.py` evaluates four candidate
  signals over the 60 n_scalar-masked heads (mean max-softmax, mean margin,
  mean log max-softmax, min max-softmax) on the 10k held-out rows of the
  300k generated split; `mean_logp` wins the exact-vs-miss rank-AUC contest
  (0.743). Deployment tau comes from the **hit-recall rule** — the largest
  tau keeping >= 90% of held-out exact hits firing (tau = -0.2473, baked
  into `DEFAULT_PRIOR_TAU`/`DEFAULT_PRIOR_SIGNAL`, regression-pinned
  against `confidence_calibration.json`). The utility-max rule is reported
  as a diagnostic only: on the generated holdout (~0.7% base exact rate) it
  degenerates to an extreme-tail tau that fires on ~nothing real, because
  that distribution badly understates the bench fallback population's hit
  rate (12.5%). At the chosen tau the gate fires on 16/42 array bench
  problems — including both historically verified winners.
- **Warm refine removed from tier-0.** v0 measured 0 conversions from 64
  warm-refine attempts at ~0.4-1 s each; every win came from zero-step
  verification. A gate-open miss now costs ~0.2 s (round-trip + K=4
  zero-step verifies); a gated miss ~5-10 ms.

**Retrain = honest null result.** A 300k-row dataset (94,857 distinct
programs; stats post-hoc in `artifacts/prior_net_gen_stats_300k.json`) was
trained for the 1 h time-box (best epoch 11): held-out exact rate 0.65% vs
v0's 0.66% on the identical 10k holdout — no lift — and its bench fire set
drops `longest_increasing_run`. The v0 checkpoint stays deployed; the
rejected checkpoint is kept untracked.

**Measured result (A4 protocol, fresh isolated banks, idle machine):**
full bench 105/105 both runs (tier-0 still pre-empted by the search-teacher
catalog). Direct fallback head-to-head on the 16 universal-array problems:
16/16 both ways, zero-search wins **2** (`longest_increasing_run` 3.15 s ->
**8 ms**, `count_peaks` 3.02 s -> **23 ms**), wall **OFF 36.3 s -> ON
31.6 s (-4.7 s)**. v0 history preserved in
`artifacts/prior_net_phase_a.{json,md}`; success criteria are pinned by
`tests/test_prior_net_phase_a.py`.

**What's next (before Phase B):** hit rate is still the binding constraint
— 2/16 verbatim. The gate and server made misses nearly free, so the next
lever is training on the bench-problem distribution (solved-cache replay),
conditioning constants on `discover_useful_consts`, and K>4 batched
proposals now that each extra proposal costs ~15 ms to verify rather than
a warm refine.
