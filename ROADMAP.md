# nCPU / NPCoT Execution Roadmap

Status date: 2026-06-11. This is the committed execution plan. Each rung has
an owner track, a definition of done, and a verification requirement. Nothing
ships without its verification step passing.

## Context

NPCoT = discover a program that provably produces the right answer, verify
it, cache it in a similarity-keyed library, reuse it forever. The browser
runtime (kernels/npcot_wasm, 113 KB WASM) now synthesizes v2 programs
(multi-field records, mined predicate guards) and exports discovered programs
in 5 languages. The heavy synthesizer (nsynth) carries three cross-run memory
banks: solved programs, learned gradient biases, rejected programs
(negative memory). The LLM tier (autoresearch) lifted Qwen3.5-4B from 58.5%
to 85.98% HumanEval for $0.39 GPU via verified compounding.

The cascade: browser tier (instant, exhaustive, refuses honestly) → nsynth
(loops/branches/strings/algorithms) → LLM (anything) → verified wins distill
back down into the cheap tiers.

## Rung 1 — Synthesized-Pong demo (proof of wow)

A playable game on ncpu.ai where every rule of game logic was synthesized
from I/O examples and verified — physics step, paddle bounce, scoring,
bounds checks. The human-written part is only the dumb canvas/event loop.

- Game-logic programs come from problems the nsynth benchmark already
  solves: simulate_gravity, combat_resolve, score_tracker, grid_bounds_check,
  turn_order_rotate (105/105 coverage artifact).
- Transpile Mog → TypeScript via nsynth/src/mog_transpile.rs (to_typescript).
- New page: sms-hub/apps/ncpu-site/src/app/pong/page.tsx. Banner: "No human
  wrote this game's logic — it was discovered from examples and verified."
- Each rule displayed alongside the game with its source + the examples it
  was synthesized from.

DoD: `pnpm build` green; Playwright drives a paddle, ball bounces, score
increments; page shows per-rule synthesized source. Screenshot artifact.

STATUS: ✅ shipped (sms-hub `7c46bc9`). 22 rules: 14 synthesized + 8
composed from synthesized primitives, all domain-swept. Full provenance,
the exact training inputs, and the reproduction harness live in
`tools/pong_synthesis/` (committed artifact `pong_rules_final.json`;
`finalize_pong_rules.mjs` regenerates the site's synthesized.ts
byte-identically and fails loudly on any sweep mismatch).

## Rung 2 — Stateful skills (format v3)

Today's programs are stateless folds. v3 adds a persistent state vector:
skill = (state, input) → (state', output). Unlocks counters, debouncers,
running statistics, state machines — i.e. interactive programs.

- Extend kernels/npcot_wasm: ProgramV3 owns `n_state` floats; per-step
  execution reads/writes state. v2 remains the exact special case n_state=0.
- Serialization: `"format": 3` + `program_v3` key; v1/v2 loaders reject
  cleanly (same versioning discipline as v2: old runtimes must fail closed,
  never mis-execute).
- Synthesis: enumerate state-update/output pairs over the existing op
  vocabulary; thresholds and constants mined from examples (no hardcoded
  vocabulary — emergent only).
- Trace examples: a v3 training example is a sequence of (input, expected
  output) steps; verification replays the trace.

DoD: cargo tests cover discovery of a running-counter, a running-max with
reset signal, and an honest refusal on a non-expressible trace; v2/v3
equivalence test at n_state=0; WASM exports synthesize_v3/insert/consult;
all existing 24 tests stay green.

## Rung 3 — Synthesis API endpoint (the cascade, served)

nsynth behind HTTP so browser refusals escalate to the heavy synthesizer.

- New module: ncpu/synthesis_api/server.py — stdlib-only HTTP server in the
  style of ncpu/self_optimizing/npcot_server.py (zero deps, <100 MB).
- POST /synthesize {name, signature?, examples: [{inputs, expected}]} →
  shells out to nsynth/target/release/mog_synth --problem-json - (already
  supported), returns {success, method, code, transpiled: {python, rust,
  typescript}} using mog_transpile via the CLI or a --transpile flag.
- GET /health, GET /stats (bank sizes: solved/bias/rejected counts).
- Caching automatic: nsynth's solved_cache makes repeat requests ~instant;
  rejected_cache prevents re-grinding failures.

DoD: pytest hitting a live server instance: solve a loop-class problem
(e.g. fibonacci-shaped examples), assert verified code + python transpile
returned; second request returns in <100 ms (solved-cache hit); unsolvable
request returns success:false (refusal preserved end-to-end).

## Rung 4 — Verified-skill registry MVP (community engine seed)

crates.io for synthesized programs. Trustless contribution: server re-runs
verification before accepting; spam/wrong code physically cannot enter.

- New: tools/registry/server.py (stdlib or FastAPI — match repo norms),
  SQLite storage.
- POST /skills {name, author, examples, program_v1|v2} → server re-executes
  the program against the examples with a pure-Python mirror of the
  canonical executor (v1+v2 semantics, ported from kernels/npcot_wasm) →
  accept iff max_err <= 1e-3 → assign fingerprint, dedupe by examples
  fingerprint.
- GET /skills (list, with author attribution), GET /skills/{fp},
  GET /library.json (whole registry as a loadable NPCoT library, format
  auto: v1 if pure-v1 else v2).
- Re-verification sweep: a --verify-all mode re-checks every stored skill
  (CI-able trust guarantee).

DoD: pytest: submit valid skill → accepted + retrievable in library.json
and loadable shape; submit subtly-wrong program → rejected with the
verification error; duplicate fingerprint → deduped; --verify-all green.

## Rung 5 — Continuous autoresearch (self-improving loop)

The daemon already exists (mine hard-fails → cascade-solve → distill →
CompoundingStore). Continuous = scheduled + fed by real usage.

- Document the loop: docs/autoresearch_continuous.md — how to run nightly
  on vast.ai (~$0.10/hr), how the prompt→WorkItem extractor turns arbitrary
  coding requests into verifiable work items, how wins land in the
  CompoundingStore and misses in negative memory.
- Wire registry misses (Rung 4) as a work-item source.
- CI self-improvement gate already enforces cross-run learning on commits.

DoD: documented runbook with exact commands; one demonstrated end-to-end
cycle on a small problem set with before/after cache sizes.

## Rung 6 — Publish

- Fresh paper PDF build script (pandoc), arXiv submission.
- python -m build → PyPI wheel.
- Deploy ncpu.ai (Vercel, root dir apps/ncpu-site) + pricing/contact page.
- Pre-public hygiene: nsynth/data ~180 MB tracked JSONL → LFS or prune.

DoD: arXiv ID exists; pip install ncpu works; ncpu.ai serves the demo.

## Rung 7 — Natural-language → verified-program engine (MCP server)

The end-state product: a coding tool where a natural-language request is
converted to I/O examples, the cascade synthesizes a verified program, and
the answer ships with its proof. Delivery vehicle: an **MCP server** — it
plugs into Claude Code, Claude Desktop, Cursor, and any MCP client
instantly, which beats building a bespoke agent UI.

Pieces that already exist:
- NL → work item: `ncpu/autoresearch/prompt_parser.py` (the prompt→WorkItem
  extractor: mines I/O pairs from arrow notation, doctests, fenced asserts).
  Known bug to fix: `_accept` crashes on list-valued args (unhashable
  tuple; `_freeze` applied to expected but not args).
- Examples → program: nsynth via `mog_synth --problem-json` (solver
  portfolio + three persistent memory banks) and the HTTP wrapper in
  `ncpu/synthesis_api/`.
- Program → user's language: `--transpile {python,rust,typescript}`.
- Verified cache: solved bank + registry (`tools/registry/`).

MCP tools to expose:
1. `synthesize_from_examples(name, examples, language?)` → verified code +
   method + proof metadata (examples checked, sweep counts), or honest
   refusal.
2. `synthesize_from_prompt(prompt, language?)` → extract I/O pairs from the
   prompt text (and confirm them back to the caller), then tool 1. When
   extraction finds no pairs, return the pairs it needs — the LLM client
   supplies them; the human-in-the-loop never writes code, only examples.
3. `consult_library(examples)` → instant answer when a cached verified
   skill already matches (fingerprint or signature similarity).
4. `library_stats()` → solved/bias/rejected bank sizes (observable learning).

DoD: stdio MCP server (`ncpu/mcp_server/`), registered in a
`.mcp.json`-style config, pytest suite covering all four tools incl. an
end-to-end "NL prompt with arrow examples → verified Python function"
case and an honest-refusal case; prompt_parser list-args bug fixed with a
regression test; README with client-setup instructions.

## Rung 8 — Reunify with the neural computer

NPCoT/synthesis is one pillar of nCPU, not the whole project. Once Rungs
1-7 are online, re-center the public story so the neural computer (neural
ALU, neurOS, GPU-as-computer running Alpine/self-hosting C compiler, JEPA
machine dynamics) is co-headline, not background:
- Site: dedicated /computer page (GPU hero demo transcript, neural OS
  accuracy tables, MUXLEQ Turing proof) + homepage rebalanced to the
  five-pillar structure the README already uses.
- Bridge demo: a synthesized program compiled and executed ON the GPU
  ARM64 computer — connects the two halves in one artifact.
  STATUS (2026-06-12): SHIPPED — `demos/bridge/synthesized_on_gpu.py`; nsynth
  → Mog → C → self-hosting cc ON rust_metal GPU → GPU execution; 10/10 outputs
  match incl. 2 never-seen inputs (see `docs/bridge_demo.md`, `artifacts/bridge_demo_result.json`).
- Paper remains the umbrella: synthesis sections already live alongside
  the neural-computer sections.

## Rung 9 — Program-native coding model (bottom-up)

Build the model where NPCoT is the substrate, not an attachment: reasoning
in program space, NL/source code as I/O renderings, transpiler-only output
path (cannot hallucinate), verification in the loop, JEPA latent
speculation for candidate pruning, recursive networks for tree-structured
program generation and compositional library embeddings. Full design:
`docs/native_synthesis_model.md`. Phase A (meta-learner → Program Prior
Net as nsynth tier-0 proposer) is buildable now from existing components.

Phase A work breakdown (IN PROGRESS):
- A1 data generator: sample random programs from the universal-array space
  (random_bias_init + discover_useful_consts already exist in
  nsynth/src/synthesis/universal_array.rs), execute on random arrays via
  the exact executor, emit (examples -> program description) JSONL.
  Perfect ground truth, unlimited volume.
- A2 prior net v0: TransformerEncoder over tokenized I/O examples ->
  program-slot logits (architecture lineage: the v5 meta-learner that hit
  23/24 on the 1-arg scalar space). Train locally (MPS), ~100k samples.
- A3 tier-0 wiring: extend the existing Python warm-start bridge pattern
  (nsynth py_warmstart subprocess; try_python_warmstart call site in
  solver.rs) so the prior proposes parameter initializations / discrete
  programs BEFORE the hand-coded restart cascade. Verified-or-discarded:
  coverage can never regress below the search baseline.
- A4 eval: 105-problem bench with prior on/off — % solved zero-search,
  mean gradient steps saved, wall time; artifact JSON + regression test.
  Honest reporting either way.

DoD: A4 artifact committed showing the measured delta; bench coverage
stays 105/105; all existing nsynth tests green.

## Rung 10 — Rule-Compressed Memory ("infinite context") + recovered grammar

Thesis: storing knowledge as VERIFIED SYNTHESIZED PROGRAMS (rules) instead of
instances yields a memory with **bounded converging storage, unbounded reach
(generalizes to never-seen items), and zero forgetting** — the precise,
defensible form of "infinite context". Two pillars: (1) POSITIVE — rules +
learned_biases compress the regular part; (2) NEGATIVE / Hamilton — mistake
memory (HamiltonGuard in linguigenesis; rejected_cache + exceptions in nsynth)
remembers the irreducible residue and never repeats an error → monotonically
self-improving. Built on existing infra: solved_cache (cross-run persistence),
learned_biases bank, rule synthesizers (search_array_dnf, morph_transduce,
string_synth), morpheme tokenizer (perception layer).

### Tier A — prove infinite-context (DONE, committed 92e9632)
`nsynth/scripts/rule_memory_experiment.py` streams (word→plural) and compares
RULE vs INSTANCE(RAG) vs WINDOW-W(LLM). Measured @500: RULE 731 B FLAT (8
re-synths, 0 exceptions) vs INSTANCE 6972 B linear; recall-on-seen RULE 99.6%
vs WINDOW 30% (forgot rest); unseen-real 99.0%, nonce-wug 100% (baselines 0%);
9.5× compression. Hamilton metric: 65 distinct mistakes (converges, finite), 0
repeated (self-improving). Paper: `paper/sections/section_rule_memory.md` (3
properties: compression, persistence-without-drift, composition + Hamilton
second-pillar section). DoD met.

### Tier B — lifelong multi-domain library, zero catastrophic forgetting (IN PROGRESS — workflow wcdzcat00)
`nsynth/scripts/lifelong_library_experiment.py`: learn 3-4 domains IN SEQUENCE
into ONE persistent rule library (pluralization → string transform → 3sg
inflection). After learning ALL, re-test every EARLIER domain → show 100%
retention (a discrete program added for domain N cannot degrade N-1; this is
the structural impossibility-of-forgetting that weights lack). Report per-domain
accuracy after each stage + total library bytes. DoD: early-domain accuracy
stays 100% after later domains added; artifact + numbers committed.

### Tier C — recovered grammar + rule-memory over SENTENCES (IN PROGRESS — workflow wcdzcat00)
- Recover the remaining agreement checkers as verified DNF programs via the
  feature+DNF recipe that produced sentence_3sg_general (suffix tokens 100-108 +
  structural feature tokens 900-904, search_array_dnf teacher):
  - number_agreement: valid iff (sing ∧ 3sg-suffix) ∨ (plural ∧ ¬3sg-suffix)
  - copular_agreement: valid iff (sing ∧ is ∧ ¬are) ∨ (plural ∧ are ∧ ¬is)
  - past_sentence (general, all verb classes): valid iff <+ied> ∨ (<+ed> ∧ ¬y) ∨ <+d>
- `nsynth/scripts/sentence_memory_experiment.py`: same 3-memory comparison as
  Tier A but over grammaticality-labeled sentences — DNF grammaticality rule
  storage flattens while it accepts/rejects held-out sentences (incl. unseen
  verbs) that window/instance cannot.
DoD: each checker verified on holdouts (target ≥95%); sentence-memory artifact
committed; recovered programs wired as native validators (extend
`linguigenesis/v2/nsynth_validator.py` beyond for_3sg).

### Tier D — rules into the LLM / coprocessor (NEXT, not yet started)
Wire recovered verified rules into a generation loop as RETRIEVED VERIFIED
SKILLS — the validator-in-the-loop pattern (`nsynth_validator.py`
nsynth_validated_generation already exists for 3sg) generalized to the full
recovered grammar, and longer-term as a retrieval source for the differentiable
coprocessor (ncpu/coprocessor) so the transformer can CALL a verified rule
instead of approximating it in weights. DoD: draft→validate→revise loop over the
full recovered checker set measurably fences faulty generations (repeat the 8/8
fencing result in kvrm_constrained_speak.py across number/copular/past, not just
3sg).

### Active workflow wcdzcat00 (background)
Script: `…/workflows/scripts/infinite-memory-fanout-wf_5141238c-5e5.js`. Phases:
(1) parallel recover number/copular/past checkers; (2) parallel Tier B lifelong
+ Tier C sentence-memory experiments; (3) adversarial verify every claimed
number (reproduced=false → omit/flag); (4) synthesize
`paper/sections/section_lifelong_rule_memory.md` from VERIFIED results only.
On completion: review verdicts, commit confirmed artifacts, update MEMORY.md.
Risk: checker-recovery agents may hit Mog-synthesis feature-encoding walls →
they report success=false rather than burn budget; re-do those inline if so.

### Novel opportunities (innovation / publication / monetization)
- PAPER: "Rule-Compressed Memory: Bounded Storage, Unbounded Reach, Zero
  Forgetting" — a real alternative framing to context-window scaling and RAG;
  the measured compression + generalization + no-forgetting triple is novel and
  defensible. Pairs with the two-pillar (positive rules + Hamilton negative)
  self-improvement story.
- PRODUCT: "verified skill memory" — a persistent, auditable, never-forgetting
  knowledge store for agents where each memory is executable+verifiable code,
  not a vector. Differentiator vs vector DBs: generalizes to unseen queries,
  storage converges, every answer is provably correct or honestly refused.
- COPROCESSOR TIE-IN: recovered rules as a retrieval source the differentiable
  transformer can call (Rung 9 / ncpu coprocessor) — symbolic verified skills
  inside the forward pass.
- BENCHMARK: publish the rule-vs-instance-vs-window harness as a standard
  "memory generalization" benchmark (storage curve × seen-recall × unseen-reach
  × repeat-mistake-rate).

DoD (Rung 10 overall): Tier A committed (done); Tier B + C artifacts committed
with verified numbers and regression tests; Tier D fencing result reproduced
across the full checker set; `section_lifelong_rule_memory.md` merged; MEMORY.md
+ this roadmap reflect final measured numbers.

## Standing rules

1. Verification gates everything: no rung ships without its DoD test green.
2. Honest refusal is the product: never trade it for coverage.
3. No hardcoded vocabularies: thresholds/constants mined from data.
4. Format changes are versioned: old runtimes must fail closed, never
   mis-execute.
5. Every numerical claim gets a harness + artifact + regression test
   (REPRODUCIBILITY.md discipline).
