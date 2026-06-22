# MASTER ROADMAP — Linguigenesis-Native nCPU Coding Agent

**Status:** Active execution; pre-MVP
**Execution authority:** This file
**Architecture companion:** [`nsynth/docs/LINGUIGENESIS_NATIVE_CODING_AGENT_PLAN.md`](nsynth/docs/LINGUIGENESIS_NATIVE_CODING_AGENT_PLAN.md)
**Low-level runtime companion:** [`nsynth/docs/PHASE_2_4_EXECUTION_PLAN.md`](nsynth/docs/PHASE_2_4_EXECUTION_PLAN.md)
**Scope:** `nCPU/nsynth` and required Rust work in sibling `../linguigenesis`
**Core rule:** Linguigenesis is the native comprehension/reasoning layer. External LLMs are optional, untrusted proposers—not the agent core.
**Implementation rule:** production orchestration is Rust. No simulated capability, stub success, benchmark theater, or hidden Python agent runtime.

### 0.6 North Star — Universal program synthesis

The finished **nsynth** layer must synthesize **any** program shape expressible as a typed `Problem` (examples + signature + verifier), then extend to multi-function modules and full repositories. Capability grows by:

1. **KVRM data** (operations, constraints, workflows) — not Rust `if text.contains` or per-op tables;
2. **Learned / evidence-ranked portfolios** (`method_router`, `search_family_router`) — not fixed category→stage tables;
3. **UTBUS / structural composition** ([`nsynth/docs/SYNTHESIS_NEXT_STEPS.md`](nsynth/docs/SYNTHESIS_NEXT_STEPS.md) Part 2) — collapse scattered teachers into one typed bottom-up engine.

**No hard-coded routing rule:** production paths must not select semantics or solver stages via keyword lists, factory name tables, or `category == "…"` dispatch. Allowed: (a) registry graph walks, (b) type/shape features of `Problem`, (c) learned routers fed only by verified outcomes.

### 0.7 Impact-ordered execution queue (do in this order)

| Priority | Package / work | Why it matters |
|---:|---|---|
| 1 | **C** — finish emergent NL → `SynthesisRequirement` | Unblocks all NL-driven synthesis without keyword tables |
| 2 | **C** — expand `coding_registry.json`; remove `populate_code_entities` production fallback | Single data source for operations |
| 3 | **D** — `Requirement` → `Problem` → `solve_problem` (universal entry) | Any registry-described op synthesizes through one API |
| 4 | **A** — baseline truth (bounded tests, failure clusters) | Stops regressions from masking synthesis work |
| 5 | **B** — `AgentRun`, `CodingIntent`, capability registry | One contract for agent + synthesis |
| 6 | **Synthesis** — type-shape portfolio dispatch (replace category routing) | ✅ confirmed: solver dispatches by type/shape + learned `method_router`, not `category` strings (§1.0); repair proposer now uses real synthesis, not keyword tables |
| 7 | **E–H** — repo agent loop | Programs beyond isolated functions |
| 8 | **UTBUS** — compositional structural synthesizer | True "any program" from typed composition |

> **Every agent must read Sections 0, 0.6, 0.7, 3, 6, 7, and the current phase card before editing code.**
> This roadmap replaces historical “completed” checklists. A module existing does not mean the capability works.

## 0. Agent Start Protocol

Follow these steps in order at the beginning of every coding-agent session:

1. Run `git status --short` and `git ls-files -u`. Never overwrite an unmerged or concurrently edited file without understanding all sides.
2. Read this roadmap, then the current phase's referenced source/docs. Do not infer APIs from names.
3. Check `graphify-out/GRAPH_REPORT.md` and `graphify-out/wiki/index.md` when present before architecture work.
4. Confirm the current package and gate in the Execution Ledger below. Do not start a later package because it looks easier.
5. Record the exact starting revision and dirty-tree state in the handoff. This repository often has concurrent work.
6. Run the package's preflight commands before modifying code. A prior agent's green result is not transferable to a changed tree.
7. Make the smallest coherent implementation that closes a real acceptance criterion.
8. Add success, failure, adversarial, and no-false-success tests. Wiring-only tests do not prove capability.
9. Run the package verification commands. Report exact counts and failures.
10. Update this ledger and the current phase card. Do not mark a gate complete without executable evidence.
11. Save architectural decisions and non-obvious discoveries to persistent memory.
12. End with the Handoff Template in Appendix C so the next agent can continue without archaeology.

### 0.1 Status vocabulary

Use only these labels:

- **ABSENT:** no implementation exists.
- **SCAFFOLD:** types/modules exist, but the real boundary or workflow does not execute.
- **EXPERIMENTAL:** executes real work but lacks full safety, determinism, or acceptance evidence.
- **IMPLEMENTED:** passes the phase conformance suite, including failure and adversarial paths.
- **VERIFIED:** implemented and demonstrated on isolated unseen tasks with saved artifacts.
- **BLOCKED:** cannot proceed without a named prerequisite; include the reproducer and owner.

Never use “done,” “complete,” or a checkmark for SCAFFOLD code.

### 0.2 No-cheating contract

An agent must not:

- return success without a real acceptance oracle;
- substitute metadata validation for benchmark execution;
- hard-code expected benchmark patches or answers;
- train, retrieve, or prompt from sealed evaluation solutions;
- expose holdout outputs or reference implementations to candidate generation,
  routing, ranking, acceptance, repair, memory, or credit assignment;
- count an external Python/model fallback as an nsynth solve;
- replace Linguigenesis with an external model call and call it native cognition;
- use regex/keywords as the final semantic representation;
- bypass policy through direct `std::fs`, shell, network, git, or database calls in production workflows;
- write memory or credit before verification establishes what happened;
- weaken/delete tests to make a patch pass;
- suppress failures globally or omit a failing command from the report;
- invent APIs, configuration fields, or benchmark results.

### 0.3 Current truth snapshot — update after every package

**Snapshot date:** 2026-06-21 (evening)
**Overall readiness:** roughly 42% of a complete repo coding agent
**MVP target:** completion through Package H / Gate G5
**Benchmark-readiness target:** completion through Package M / Gate G7
**Active priority:** G5 sign-off; Package I durable workflows + CLI session resume

| Package | Gate | State | Evidence / blocker | Next owner action |
|---|---|---|---|---|
| A — baseline truth | G0 | IN PROGRESS (G0 largely closed) | compile-green; `agent::repo` 45/45; `agent::session` 7/7; `agent::tools` 32/32; `optimization::parallel` 23/23; search-only holdout verify green; `cargo fmt --check` green | G0 sign-off: full serial lib suite summary |
| B — runtime contracts | G1 prerequisite | IMPLEMENTED | Phase 1 contracts + conformance suite — `docs/PACKAGE_B_GATE.md` | — |
| C — Linguigenesis coding semantics | G1 | IMPLEMENTED (breadth-limited; *universal* NL claim EXPERIMENTAL) | emergent NL + negation + robustness corpus; `coding_registry.json` = **25 proven ops** (integrity gate `every_registry_operation_is_synthesizable`), 1 vocabulary-only gap (`reduce`); registry now git-tracked. See §0.8 | type/shape `Problem` targeting for unseen ops (queue item 6) |
| D — grounded bridge | G1 | IMPLEMENTED | `nl_to_requirement`, clarification (non-synthesis workflows exempt), `solve_from_description` | — |
| E — repository model | G2 | IMPLEMENTED | `RepoIndex` + retrieval benchmark tests green | — |
| F — secure tools | G3 | IMPLEMENTED | `SecureToolRuntime` deny-by-default + `for_general_agent` / `for_repo_repair`; HTTP host allowlist; verification cargo-only oracle | docs + HTTP CLI allowlist examples |
| G — transactional edits | G4 | IMPLEMENTED | `IsolatedRepoSession` git worktree + temp-copy fallback; promote/discard; repair loop isolated | — |
| H — closed repair loop | G5 | EXPERIMENTAL (near G5) | NL fixture suite (add/sub/mult/div/max/reverse/multifile/gcd + **unseen `triple` via inline examples**); cargo-test oracles; **real verified synthesis is now the primary repair path** (`try_real_synthesis_patch`, keyword table demoted to fallback) — see §1.0; budget-gated + plain-Rust validity gate; `repo_agent_repairs_gcd_via_general_synthesis`; supervisor budget sync | G5 sign-off: solver-IR→Rust lowering so safe-div/loop ops also use real synthesis; widen unseen corpus |
| I — workflows/supervision | G6 | EXPERIMENTAL | `RepoWorkflowRunner` + `run_query` session router; workflow JSON persist | durable typed workflow resume across sessions |
| J — durable memory/resume | G6 | EXPERIMENTAL | `CodingAgentSession` + `.nsynth/sessions/` snapshots; clarify resume API | end-to-end CLI `--clarify` on real ambiguous synthesis |
| K — project-scale generation/validation | G6 | SCAFFOLD | strong function synthesis; project/multilanguage claims exceed evidence | graduate each backend independently |
| L — CLI/session/telemetry | G6 | EXPERIMENTAL | `coding_agent` binary builds; `--root`, `--session`, `--tool`, `--json`, `--clarify` | telemetry + primary entry vs legacy paths |
| M — executable local benchmark | G7 | SCAFFOLD | 20 task manifests; no fixtures/runner/scoring | convert each to isolated executable task |
| N — sealed external benchmarks | G8 | NOT STARTED | local harness not yet trustworthy | do not start before G7 |
| O — evidence-driven self-improvement | G8 | EXPERIMENTAL PARTS | synthesis/meta components exist without repo-agent trace gate | mine only verified held-out traces |
| P — CI/release/publication | G9 | NOT STARTED | no trustworthy release gate | last package |

### 0.4 Immediate Package A work card

**Objective:** create a stable, truthful, reproducible starting point. Do not add agent features during this package.

**Prerequisites to read**

- `nsynth/Cargo.toml` and workspace Cargo configuration.
- `nsynth/src/lib.rs`, including Git index stages 1/2/3 when unmerged.
- `nsynth/src/benchmark.rs::Problem` and all direct `Problem { ... }` initializers.
- `nsynth/docs/PHASE_2_4_EXECUTION_PLAN.md` Phase 0.

**Allowed existing patterns**

- For a legacy single-function `Problem`, copy the `functions: vec![]` initialization used by `nsynth/src/benchmark.rs::problem`.
- Resolve `nsynth/src/lib.rs` as the reviewed union of valid modules from both sides; do not choose ours/theirs wholesale.
- Use Cargo's actual targets/features from `Cargo.toml`; do not invent package names.

**Steps**

1. Capture `git status --short`, `git ls-files -u`, and the module lists from all conflict stages.
2. Verify the working `lib.rs` contains the required union without duplicate modules or conflict markers.
3. Stage only `nsynth/src/lib.rs` to mark that reviewed conflict resolved.
4. Add `functions: vec![]` to legacy single-function `Problem` initializers that fail compilation.
5. Run `cargo check --lib`; repeat only for concrete compiler errors, not warnings.
6. Run focused `cargo test 'agent::repo' --lib -- --test-threads=1`.
7. Run the complete library suite serially and record every failure/abort. Fix only baseline integration defects with clear ownership; do not mask semantic failures.
8. Run `cargo fmt --check`. If formatting drift is broad and concurrent, inventory it instead of mass-reformatting unrelated work.
9. Scan production Rust for `todo!`, `unimplemented!`, `NotImplemented`, `simulate`, placeholder success, and dummy backends; classify every hit.
10. Write/update the baseline artifact with commands, revision, counts, failures, and owners.

**Verification**

```bash
cd nsynth
cargo check --lib
cargo test 'agent::repo' --lib -- --test-threads=1
cargo test --lib -- --test-threads=1
cargo fmt --check
rg -n 'todo!|unimplemented!|NotImplemented|simulate_|placeholder|dummy' src
cd ..
git diff --check
git ls-files -u
```

**Package A exit criteria**

- no unmerged index entries;
- debug and release library checks complete;
- full library test process completes without stack overflow or hang;
- all remaining failures have reproducible commands and owners;
- stub audit exists and production-reachable hits are classified;
- no capability status is inflated to hide baseline failures.

### 0.5 Immediate Package C work card — emergent NL → nsynth requirements

**Objective:** replace all keyword/hard-coded NL→example routing with registry-derived `SynthesisRequirement`. No new standalone plan documents; this card is authoritative.

**Prerequisites**

- `linguigenesis/rust/linguigenesis-core/src/registry.rs` (JSON relations loader).
- `linguigenesis/rust/linguigenesis-core/src/coding_requirements.rs`.
- `linguigenesis/data/coding_registry.json` (coding ontology seed).
- `nsynth/src/linguigenesis_bridge.rs` (migration target).

**Build (linguigenesis)**

1. Extend `EntityType` with `Workflow` and `ConstraintMarker`.
2. Load `relations` and typed `attributes` from registry JSON; support `merge_from_json`.
3. Ship `coding_registry.json`: operations (`example_cases`, `arity`, `input_types`, `output_type`, `nsynth_category`), workflows (`signal_lemmas`, `maps_to_intent`, `coding_workflow`), constraint markers.
4. Implement `RequirementDeriver::derive(BeliefState, text) -> SynthesisRequirement`.
5. Implement `CodingComprehension` (parse + derive in one call).
6. Replace `comprehension.rs` keyword `classify_intent` with workflow-entity matching from registry signals.
7. Tests: paraphrase via synonym (`sum` → `add` examples); negation constraint; zero-network.

**Build (nsynth bridge — Package D slice)**

1. `nl_to_examples` / `nl_to_requirement` call `CodingComprehension` only.
2. Delete `generate_examples_from_text` and migrate `populate_code_entities` knowledge into JSON.
3. `solve_from_nl` uses emergent path when `nl` feature enabled.
4. Map `SynthesisRequirement` → `Problem` with evidence entity IDs in metadata.

**Verification**

```bash
cd ../linguigenesis/rust/linguigenesis-core
cargo test coding_requirements --lib
cargo test coding_comprehension --lib
rg -n 'text\.contains|lower\.contains' src/comprehension.rs ../ncpu/nsynth/src/linguigenesis_bridge.rs

cd ../../ncpu/nsynth
cargo test linguigenesis_bridge --lib
```

**Package C exit criteria** — ✅ DONE (2026-06-21)

- unseen paraphrase resolves to identical `example_cases` via synonym relations;
- no hard-coded `Example { inputs: ... }` in bridge for NL routing;
- `comprehension.rs` intent from workflow entities, not substring lists;
- `coding_registry.json` is the sole source of operation examples for the native path.

**Package D exit criteria (follows C)** — ✅ DONE (2026-06-21)

- bridge produces `Problem` with provenance via `problem_from_requirement` / `synthesize_from_description`;
- `solve_from_description` is the universal solver entry (no `nl` feature required);
- low-confidence or non-empty `unresolved` triggers clarification via `CodingDialogue` / `BridgeError::ClarificationNeeded`;
- quarantine `nl/mod.rs` from production (TODO — separate from emergent NL path).

### 0.6 Package C2 — Truly emergent NL resolution (COMPLETE 2026-06-21)

**Objective:** paraphrases not literally present in `coding_registry.json` still resolve via graph walks, morphology, fuzzy lemma match, definition overlap, and `AnalogyReasoner` — never via Rust keyword tables.

**Architecture**

```text
surface token
  → EntityResolver (direct → morphology → synonym/hypernym/similar/entailment hops
                     → reverse relations → fuzzy lemma → definition overlap)
  → matched Entity set
  → RequirementDeriver::select_primary_operation
       → if empty: infer_operation_from_text
            → resolver per-token operation match
            → AnalogyReasoner::find_analogies(anchor noun/domain entity)
            → AnalogyReasoner::rank_operations_for_surface(token)
  → canonical_operation (synonym walk to example_cases bearer)
  → SynthesisRequirement (confidence, unresolved, evidence_entity_ids)
  → if needs_clarification: CodingDialogue typed questions
  → apply_clarification(answer) → updated requirement
```

**Implemented modules**

| Module | Role |
|--------|------|
| `entity_resolution.rs` | Multi-hop KVRM surface→entity resolution |
| `coding_dialogue.rs` | Typed clarification questions + `apply_clarification` |
| `computing_knowledge_import.rs` | Merge `computing_knowledge.json` definitions/relations |
| `reasoning.rs` | `rank_operations_for_surface` for unknown verbs |
| `coding_registry.json` | `grammar_stop_words`, `combine`, `double`, `sub`, `fold` |

**Registry scale**

- `coding_registry.json` merged at bridge load (operations + workflows + grammar markers).
- `computing_knowledge.json` merged for supplemental definitions (does not overwrite `example_cases`).

**Verification gates**

```bash
cd ../linguigenesis/rust/linguigenesis-core
cargo test entity_resolution coding_dialogue coding_comprehension coding_requirements computing_knowledge_import --lib
rg -n 'text\.contains|lower\.contains' src/comprehension.rs src/coding_requirements.rs
rg -n '"map" \| "filter"' src/comprehension.rs  # must be empty (no hard-coded op lists)

cd ../../ncpu/nsynth
cargo test linguigenesis_bridge --lib
```

**Exit criteria** — all satisfied:

- `combine two numbers` → `add` `example_cases` (synonym + analogy path);
- `double the array` → `map` `example_cases` (morphology + synonym);
- `sum two integers` → `add` (synonym reverse walk);
- gibberish → `ClarificationNeeded` with registry operation options, not guessed synthesis;
- zero-network (no external APIs in NL path);
- production NL path has no `text.contains` keyword routing.

**Stretch (deferred — compositional NL)**

- Multi-operation utterances → DAG of requirements (not needed for single-function synthesis entry).

### 0.7 Package C3 — Open-ended emergent comprehension (IN PROGRESS 2026-06-21)

**Objective:** vague or unseen NL surfaces resolve via analogy enrichment and ranked clarification — not blind tool explore or flat registry option lists.

**Implemented**

| Module | Role |
|--------|------|
| `emergent_nl.rs` | `rank_operation_options` — analogy-ranked clarification options |
| `coding_requirements.rs` | `enrich_from_analogy`, matched-entity synonym walk in `infer_operation_from_text` |
| `coding_comprehension.rs` | Analogy enrich pass before clarification gate |
| `coding_dialogue.rs` | Emergent-ranked operation options in questions |
| `session.rs` | `handle_explore` tries emergent comprehension before blind repo scan |
| `coding_registry.json` | `invert`, `larger` surface lemmas linked to operations |

**Verification**

```bash
cd ../linguigenesis/rust/linguigenesis-core
cargo test emergent_nl nl_robustness coding_dialogue coding_comprehension --lib
cd ../../ncpu/nsynth
cargo test agent::session --lib -- --test-threads=1
```

### 0.8 Package C4 — Registry breadth + proven-capability gate (2026-06-21 evening)

**Motivation (re-evaluation finding):** the synthesis engine is wide (~30 search
families; `search_only_solves_full_benchmark` = 140 tasks with holdout verify),
but the NL → agent path could only express **16** registry operations in 2
categories. "Universal NL synthesis" was bottlenecked at the registry, not the
solver. C3 polished the *front door* over a 16-word vocabulary.

**Done**

| Change | Effect |
|--------|--------|
| `coding_registry.json` +11 ops (increment, decrement, negate, square, bit_and/or/xor, length, first, last) | NL vocabulary 16 → **25 proven** ops across arithmetic/bitwise/array |
| `SynthesisRequirement::from_operation_entity` (`coding_requirements.rs`) | reusable registry-seed → requirement path |
| `linguigenesis_bridge::every_registry_operation_is_synthesizable` test | **integrity gate**: every declared op must solve through the real solver (declared == proven, per §0.2) |
| force-added `data/coding_registry.json` to git | the sole NL data source was **gitignored/untracked** — now versioned |

**Defects the gate surfaced (both pre-existing):**

- `mod` — `default_fn_name: "mod"` collided with the Rust `mod` keyword and
  emitted invalid code ("expected identifier, got Module"). Fixed → `modulo`.
- `reduce` — array→scalar fold is **not synthesizable**: the router sends it to
  the scalar differentiable path which rejects array input. Demoted to
  `synthesis_status: vocabulary_only` (honest: array→array transforms map/filter/
  sort/reverse work; array→scalar fold does not). **Open gap:** add a reduce/fold
  search family and restore its `example_cases`.

**Verification**

```bash
cd ncpu/nsynth
cargo test --lib every_registry_operation_is_synthesizable   # 25 ops solve, 0.4s
cargo test --lib agent::session linguigenesis_bridge          # 10/10 + 10/10
cd ../../linguigenesis/rust && cargo test -p linguigenesis-core --lib   # 82/82
```

**Honest status correction:** Package C is breadth-limited; with a 25-op
vocabulary the *universal* NL claim remains **EXPERIMENTAL**, not done. The
next goal-critical step (queue item 6) is **type/shape `Problem` targeting** —
let NL with inline I/O examples synthesize *unseen* ops directly through the
140-task engine, instead of only registry-named ops.

### 0.9 Package D2 — inline-example type/shape targeting (2026-06-21 night)

**This is the mouth→engine connection** (queue item 6 first cut): NL prompts
that *demonstrate* behaviour with explicit I/O examples now synthesize **any**
function — including operations with no registry entry — straight through the
typed solver. The registry is no longer the ceiling on NL synthesis.

**Done**

| Change | Effect |
|--------|--------|
| `inline_examples.rs` (new, zero-dep parser) | parses `add(2,3)=5`, `f(2,3) -> 5`, `triple(4) returns 12`, `[1,2,3] -> [1,4,9]`, multi-example, negatives, arrays, bools; arity-consistency + dedup |
| `coding_requirements::apply_inline_examples` | inline examples become authoritative evidence → workflow=Synthesize, name from call form, array/arith category, high confidence |
| end-to-end (bridge) | unseen `quux(1)=10…` (×10) and unseen array map `[1,2,3]->[2,3,4]` both **synthesize successfully** |
| conflict handling | contradictory examples (`bad(1)=1, bad(1)=2`) → **clarification, not a false success** (probabilistic sampler path was emitting a `rand` program and calling it solved) |

**Latent defects fixed while getting it to "perfect":**

- **Nondeterministic comprehension** — three score-only sorts (`reasoning.rs`
  ×2, `entity_resolution.rs`, `coding_requirements::select_primary_operation`)
  left ties to HashMap iteration order, so the same prompt could resolve to
  different ops across runs. Added deterministic id tie-breaks. Verified stable
  over repeated full-suite runs.
- **Vague prompts over-resolving** — function words slipped past the stop-word
  filter (the belief-entity loop in `collect_matched_entities` skipped it), and
  `definition_overlap_score` gave "with" a 0.51 match to `add`'s definition, so
  "help with something" silently became "synthesize add". Added a built-in
  baseline function-word stop list and filtered it in both match loops; vague
  prompts now correctly clarify.

**Verification**

```bash
cd linguigenesis/rust && cargo test -p linguigenesis-core --lib   # 93/93, deterministic
cd ncpu/nsynth
cargo test --lib linguigenesis_bridge        # 13/13 (incl. integrity gate + 3 inline e2e)
cargo test --lib agent::session              # 11/11 (incl. unseen-op-from-inline e2e)
cargo test --lib agent::repo                 # 46/46
```

**Still open:** inline parser covers the common forms; tuple/struct/tree literal
inputs and `f(x)=y where ...` natural phrasing are future work. `reduce`
(array→scalar fold) remains the one vocabulary-only registry gap (§0.8).

---

### 1.0 Package H — real verified synthesis in the repair loop (2026-06-21 late)

**Removed benchmark theater from a production path + closed-loop generalization
(queue items 6 & 7).** Audit of the repo repair proposer found the common case
was *not synthesis at all*: `synthesis_proposer::scalar_i64_body_for_nl` was a
hard-coded keyword→canned-code table (`desc.contains("subtract") → "a - b"`) and
`repo_rust_body_for_nl` hard-templated a 2-arg `i64` signature. This violated the
North Star "no hard-coded routing" rule and the no-cheating contract, and capped
repair at ~5 canned ops.

**Queue item 6 — confirmed satisfied at the engine level (evidence):** the solver
already dispatches by **type/shape**, not category strings. `solve_problem_inner`
gates `search_float_affine` on `-> f64`, `solve_multi_arg_affine` on arg shape,
branches on `has_non_scalar_input`, and uses the learned `method_router`
(verified wins/misses) — never `category == "…"`. In the bridge,
`problem_from_requirement` derives the signature from example **types**
(`infer_signature`); `category` is metadata only, never a dispatch key.

**Done (this slice)**

| Change | Effect |
|--------|--------|
| `synthesis_proposer::try_real_synthesis_patch` (new) | NL/example description → `CodingIntent::from_nl` → `to_problem` → `solve_problem` → **verified** Rust; reshaped to preserve the repo function's exact signature (body swap + positional param rename) so the failing test's call convention holds |
| wired as **primary** path in `nl_synthesis_proposer` and `nl_synthesis_proposer_with_run` | real synthesis precedes the keyword fast-patch; keyword table demoted to last-resort fallback |
| plain-Rust validity gate (`is_plain_rust_body`) | declines when the solver emits abstract IR (`ok(..)`/`err(..)`, unlowered `:=`) that won't compile for the concrete signature → safe fallback (no non-compiling repairs written) |
| synthesis-budget gating | real synthesis *is* a synthesis candidate: skipped when the persisted `AgentRun` budget is exhausted, and records a candidate when used (keeps `max_synthesis_candidates=0` rejection semantics intact) |
| new fixture `nl_fixture_triple` (×3, unseen) | no keyword-table entry exists for it — the **only** way to repair it is to actually synthesize `a*3` from the inline examples in the issue |

**Proof (cargo-test oracles, the acceptance contract):**

- `real_synthesis_repairs_unseen_inline_example_op`: `triple` fixture **fails
  before**, the proposer returns a patch tagged `proposer=nl_real_synthesis`, and
  cargo test **passes after** — the closed repair loop now fixes an arbitrary
  demonstrated function, not just the canned vocabulary.
- existing add/divide repair tests converted to behavior-based oracles
  (fail-before / cargo-green-after); divide correctly falls back (its solver
  output uses Result-style wrappers) so no regression.

**Verification**

```bash
cd ncpu/nsynth
cargo test --lib agent::synthesis_proposer -- --test-threads=1   # 7/7
cargo test --lib agent::repo::run_supervisor -- --test-threads=1 # 10/10 (budget semantics intact)
cargo test --lib agent::repo -- --test-threads=1                 # full repair regression
```

**Still open (proper IR lowering):** ops whose solver output is non-plain Rust
(safe-div `ok/err`, loop bodies with `:=`) still take the keyword fallback. The
real fix is a faithful solver-IR → Rust lowering (handle Result wrappers and Mog
assignment for concrete return types) so those ops also flow through real
synthesis. Tracked for the next slice.

### 1.0.1 Package H follow-up — solver-IR → Rust lowering + multi-function reshape (2026-06-21 night)

**Closed the "still open" item above and fixed a latent multi-function
correctness bug.** Cataloged the actual solver IR shapes the repair path
receives (via a throwaway diag over divide/gcd/multiply/max):

| Op | method | shape |
|----|--------|-------|
| divide | `search_safe_div_or_neg1_branch` | 2 fns: `helper_div -> Result<i64>` with `ok/err` + `fn divide` with multi-line `match r { ok(v)=>v, err(e)=>-1 }` and `:=` |
| gcd | `search_gcd_loop` | 1 fn: `while` loop + `:=` (already lowered cleanly) |
| multiply | `search_lcm_formula` | 2 fns: `gcd_inner` (loop, `:=`) + `multiply` calling it |
| max | `search_max2_formula` | 1 fn: plain `if/else` |

**Done (this slice)**

| Change | Effect |
|--------|--------|
| `lower_result_tokens` | `Result<` → `Option<`, `ok(X)` → `Some(X)`, `err(..)` → `None` — the safe-result idiom maps 1:1 onto `Option` (the error payload is never inspected) |
| `fold_result_match_idiom` | folds the boilerplate `match VAR { ok(v)=>v, err(e)=>CONST }` → `VAR.unwrap_or(CONST)`; the line-based mog transpiler can't handle a multi-line match, so this runs as a pre-pass. Only fires for the identity-ok / constant-err shape |
| `mog_source_for_rust_transpile` `:=` fix | now emits `let mut x = rhs` (type **inference**) instead of a hard-coded `: i64` annotation, so an `Option<i64>` binding from a lowered Result helper type-checks |
| `split_top_level_functions` + multi-function `reshape_to_repo_signature` | when the solver emits a main fn **plus helpers**, emit the helpers verbatim and the main fn renamed to the repo fn (`pub`), replacing the repo definition wholesale. The main is the name-matching fn (else the last). **Fixes a latent bug**: the old single-body reshape grabbed the *first* function (`gcd_inner`/`helper_div`) for multi-fn output, producing wrong code that only passed by luck of which method won |

**Proof (cargo-test oracles, fail-before / cargo-green-after):**

- `real_synthesis_repairs_divide_result_idiom`: the 2-fn `Result`/`match`
  template lowers to compilable plain Rust; `divide(12,4)==3` passes after,
  via `try_real_synthesis_patch` (no keyword fallback).
- `real_synthesis_repairs_multiply_multifunction`: LCM-formula
  (`gcd_inner` + `multiply`) repaired through multi-function reshape;
  `multiply(3,4)==12` passes after.
- `real_synthesis_repairs_unseen_inline_example_op`: `triple` still green.

**Verification**

```bash
cd ncpu/nsynth
cargo test --lib agent::synthesis_proposer -- --test-threads=1   # 9/9
cargo test --lib mog_transpile -- --test-threads=1               # 15/15
cargo test --lib agent::repo::run_supervisor -- --test-threads=1 # 10/10 (budget intact)
cargo test --lib agent::repo -- --test-threads=1                 # 46/46
```

Divide and multiply now flow through **real verified synthesis** instead of the
keyword fallback — the last of the safe-div / multi-function theater is retired.

### 1.0.2 Package H follow-up — broadened unseen-NL repair corpus (2026-06-21 night)

**G5 sign-off item "widen the unseen NL repair corpus" — addressed.** Before
adding fixtures, an empirical breadth probe ran each candidate shape through the
real path (`CodingIntent::from_nl` → `to_problem` → `solve_problem` → lowering)
and was read **skeptically** to avoid benchmark theater:

| Shape | Probe result | Decision |
|-------|--------------|----------|
| square `x*x` | `enumerative` / `search_polynomial_multi`, correct | **added** |
| negate `-x`, sum3 `a+b+c` | affine, correct | **added** |
| abs `if a<0 {-a} else {a}` | `loop_template`, correct | **added** |
| array sum / max / count | `enumerative-array` fold, correct | **added** |
| min, modulo | `search_affine` **overfit** 3 pts (`-6a-7b+70`, `a/9`) — not the real op | **excluded** (honest: underdetermined affine) |
| `[i64]->[i64]` map (double-each) | `diff_gradient_unsupported` | **excluded** |
| string length | string arg misparsed as `Vec<i64>` | **excluded** |

**Done (this slice)** — 7 new fixtures in `nl_fixture_wrong_stub` /
`nl_fixture_test_module`, each described **only by inline I/O examples** (no
registry op, no keyword-table entry), so repair can succeed *only* via genuine
example-driven synthesis:

| Fixture | Family exercised | Holdout asserts (not in examples) |
|---------|------------------|-----------------------------------|
| `nl_fixture_square` | nonlinear scalar (enumerative) | `square(7)=49, square(10)=100` |
| `nl_fixture_negate` | single-arg affine | `negate(100)=-100` |
| `nl_fixture_abs` | predicate branch | `absval(-100)=100` |
| `nl_fixture_sum3` | 3-arg affine | `add3(5,5,5)=15` |
| `nl_fixture_arrsum` | array fold-add | `total(vec![10,20,30])=60` |
| `nl_fixture_arrmax` | array fold-max | `biggest(vec![100,2,50])=100` |
| `nl_fixture_arrlen` | array fold-count | `howmany(vec![10,20,30,40])=4` |

Every fixture's cargo test asserts **holdout inputs** absent from the synthesis
examples, so a green run proves generalization rather than example overfit — in
keeping with the no-cheating contract.

**Proof:** `cargo test --lib agent::synthesis_proposer -- --test-threads=1` →
**16/16** (9 prior + 7 new), each new test fails-before / cargo-green-after via
`try_real_synthesis_patch` (`proposer=nl_real_synthesis`).

**Honest gaps surfaced (tracked, not papered over):** underdetermined affine
search overfits when examples are few (min/modulo) — needs holdout-aware
acceptance or example sufficiency checks; `[i64]->[i64]` map and string
transduction are not yet synthesizable through the repair path.

---
## 1. Product Definition

The finished product accepts a natural-language software request, determines what the user means, inspects a real repository, plans bounded work, edits files transactionally, invokes real tools, verifies the result, learns only from verified evidence, and returns an auditable answer.

It must support at least these workflows:

1. diagnose and repair a failing build or test;
2. implement a feature spanning multiple files;
3. write or strengthen tests;
4. perform a behavior-preserving refactor;
5. update dependencies or configuration;
6. migrate an API or data shape;
7. review a patch and explain risks;
8. explain unfamiliar code without modifying it;
9. generate a new, buildable project from a specification;
10. resume an interrupted task from durable state.

“Working” means succeeding on unseen, executable repository tasks. A type, metadata record, prompt, hard-coded answer, mocked command, or test that only confirms internal wiring does not count as capability.

## 2. Non-Negotiable Truth Rules

### 2.1 Capability rules

- No method named `simulate_*`, no fake success result, no placeholder body, and no generated `todo!()` may be reachable from a production workflow.
- A capability is **Implemented** only when it crosses a real boundary and is covered by a failure-path test.
- Otherwise it is **Experimental**, **Deferred**, or **Absent**. Documentation must use one of those labels.
- No benchmark task counts unless it provides a reproducible starting repository, executable oracle, isolated run, and machine-readable result.
- No learning signal may be recorded before the verifier establishes the claimed outcome.
- No self-modification may bypass the same guardrails and benchmark gates applied to ordinary patches.

### 2.2 Linguigenesis-first rule

The default request path is:

```text
user text
  → tokenize + KVRM entity lookup (lemma, synonym, domain, entailment)
  → BeliefState (comprehension, intent, constraints, evidence entity IDs)
  → RequirementDeriver (registry graph walk — no Rust keyword tables)
  → SynthesisRequirement (signature, examples, category, confidence, unresolved)
  → CodingIntent + CodeTaskSpec (Package B/D)
  → nsynth plan and synthesis
  → policy-gated tools
  → transactional patch
  → real verifier
  → verified memory and response
```

**Emergent NL rule:** operation examples, workflow signals, intent mapping, and constraints must live in the KVRM registry (`linguigenesis/data/coding_registry.json` and merged registries). Rust may parse JSON and walk relations; it must not use `text.contains` or hard-coded `Example` values to decide what the user asked for. New capabilities are added by extending registry data, not by growing keyword branches in `comprehension.rs` or `linguigenesis_bridge.rs`.

An external model adapter may propose an intent, plan, or patch only when explicitly enabled. Every proposal must be converted into native typed structures and pass the same policy, synthesis, and verification gates. The system must remain useful with all external-model features disabled.

### 2.3 Rust-only orchestration rule

Production orchestration, tool execution, persistence, benchmark running, and reporting remain Rust. Python may not become a hidden agent runtime. If a benchmark distributes a Python harness, invoke it as an isolated external oracle and record that fact.

## 3. Current-State Verdict (2026-06-20)

| Area | Current reality | Readiness |
|---|---|---:|
| Function synthesis | Large, real solver portfolio with many tests | ~70% |
| NLP-to-code | Linguigenesis bridge exists, but coding intent/spec grounding is shallow and the legacy NL path contains `NotImplemented` | ~30% |
| Planning | Typed decomposition/dependencies require real executor evidence; integrated solving records one real outcome and skips unexecuted nodes | ~55% |
| Tool use | Rust tool modules exist; policy, isolation, timeouts, and real boundary semantics are incomplete | ~30% |
| Repository understanding | Hardness scanner and knowledge structures exist; no canonical repository index/context engine | ~10–15% |
| Repo repair loop | New typed scaffold exists, but no executing propose/apply/test/revise loop | ~10% |
| Memory | Several local representations exist; no single durable, evidence-gated task memory | ~30% |
| Security | Initial allow/deny checks exist; canonical-path, symlink, process, network, and secret defenses are incomplete | ~20% |
| Benchmarks | 20 metadata task descriptions exist; no executable local repo-agent benchmark harness | ~10% |
| Product UX | No canonical coding-agent CLI/session protocol | ~10% |

Overall: the architecture is substantially mapped, but only about one quarter of an end-to-end coding agent is operational. The new `agent::repo` work is a useful seam, not a finished agent.

### Point-in-time baseline blockers

- The reviewed 47-module union in `nsynth/src/lib.rs` is now merged and the Git index has no unmerged entries.
- All known legacy `Problem` initializers now include the required `functions` field; debug and release library checks pass.
- The focused `agent::repo` suite passes all 16 tests. That verifies its unit wiring, not an end-to-end repair.
- The search-only stack overflow and six temporal synthesis misses are fixed.
  Public solver and exported synthesis entrypoints now strip evaluator holdouts
  and reference implementations before synthesis; reference/template fallback
  routes and the default benchmark's Python warmstart fallback are disabled.
  An oracle-poisoning invariance regression passes, and the search-only
  orchestrator still solves all 140 tasks in about 3.10s without seeing those
  oracles.
- The serial 2,112-test library run remains too slow and red: after roughly eleven minutes it had 1,211 passes, 61 failures, one ignored test, and was manually stopped inside the unbounded full-benchmark test.
- `cargo fmt --check` reports 473 hunks across 34 files. This is inventoried rather than mass-formatted while semantic baseline work remains active.
- The legacy collaborative orchestrator and planner no longer fabricate proposals,
  reviews, or task completion: synthesis uses the real Rust search solver, reviews
  run executable/static checks, and executor-less tasks fail closed. Placeholder
  behavior still remains in the NL frontend, project transpilation, and several
  advertised library modules. `agent/debate.rs` is specifically heuristic-only
  (keyword critiques, canned confidence, non-empty validation, prose claims) and
  is excluded from capability claims until every proposal/critique is executed
  and evidence-gated.
- `Problem` still physically co-locates synthesis examples and evaluator fields;
  `Problem::synthesis_view()` enforces the current boundary, but Package M must
  split these into separate serialized task/evaluation types and run sealed
  evaluation in a process that cannot expose answers to the agent.

These are Phase 0 blockers. Because the working tree is shared and changing, every implementation package must publish its revision and rerun its gates rather than inherit a previous agent's success claim.

## 4. Canonical Architecture and Ownership

```text
Session API / CLI / future MCP
        |
        v
Linguigenesis Coding Comprehension ----> Clarification dialogue
        |                                (when evidence is insufficient)
        v
CodeTaskSpec + AcceptanceOracle + PolicyEnvelope
        |
        v
Repository Index ---> Context Selector ---> Goal DAG / Workflow
        |                                      |
        +----------------------+---------------+
                               v
                    Agent Runtime / Supervisor
                               |
               +---------------+---------------+
               |               |               |
          nsynth solver   edit engine      tool registry
               |               |               |
               +---------------+---------------+
                               v
                    isolated transaction/worktree
                               |
                               v
              format -> build -> tests -> lint -> security
                               |
                success -------+------- failure parser
                   |                        |
                   v                        +--> bounded revision
          verified trace/memory
                   |
                   v
              user response
```

### 4.1 One owner per concern

| Concern | Canonical owner |
|---|---|
| language understanding and clarification | `../linguigenesis/rust/*` |
| coding-domain conversion | `nsynth/src/linguigenesis_bridge.rs` and a new `agent/intent/` module |
| task/plan/runtime contracts | new `nsynth/src/agent/runtime/` |
| repository graph and context | new `nsynth/src/agent/repository/` |
| tool registry and policy enforcement | `nsynth/src/agent/tools/` plus `agent/policy/` |
| patch transactions | new `nsynth/src/agent/edit/` |
| workflow execution | `nsynth/src/agent/repo/` |
| verification | `nsynth/src/validation/` with repo-aware adapters |
| durable task memory | new `nsynth/src/agent/memory/` |
| synthesis | existing `nsynth/src/solver*` and `synthesis/` |
| benchmark harness | new `nsynth/benches/repo_agent/` or binary plus fixtures |

Duplicate types should be migrated into these owners and deprecated, not wrapped indefinitely.

## 5. Canonical Contracts

Implement these contracts before adding more actors or benchmark names:

- `CodingIntent`: workflow kind, requested behavior, explicit exclusions, target language/framework, confidence, evidence spans, unresolved questions.
- `CodeTaskSpec`: repository root, base revision, intent, acceptance criteria, allowed scope, resource budget, required approvals, oracle definitions.
- `RepositorySnapshot`: revision, tracked/untracked state, manifests, languages, targets, symbols, dependencies, test topology, diagnostics, content hashes.
- `ExecutionPlan`: dependency DAG of typed steps with preconditions, effects, rollback, budgets, and verification.
- `ToolRequest` / `ToolOutcome`: typed arguments, policy decision, timing, capped output, exit/status data, redaction metadata.
- `PatchCandidate`: base hash, structured file operations, unified diff, provenance, expected effect, affected tests.
- `VerificationReport`: exact commands, environment fingerprint, results, regressions, changed coverage, security findings, oracle verdict.
- `AgentRun`: stable run/task/attempt IDs, state-machine status, budgets, snapshots, events, terminal outcome.
- `ExperienceRecord`: immutable link to a successful or failed verified run; never merely a self-rating.

All persisted contracts require explicit schema versions and migration tests.

## 6. Completion Gates

The project is benchmark-ready only when all gates pass:

| Gate | Required evidence |
|---|---|
| G0 Truthful baseline | clean merge state, deterministic commands, failures classified and recorded |
| G1 Native comprehension | unseen paraphrases produce correct typed specs or clarification, with provenance |
| G2 Safe repository model | deterministic index; ignored files, symlinks, generated trees, and large/binary files handled |
| G3 Real tools | filesystem/process/git/network boundaries execute under deny-by-default policy |
| G4 Transactional edits | apply/rollback/retry are hash-checked and cannot escape the workspace |
| G5 Closed repair loop | seeded failures are diagnosed, changed, verified, and bounded without manual patching |
| G6 Durable agency | interruption/resume, budgets, supervision, and verified memory work |
| G7 Executable local benchmark | all fixtures reset, run, score, and emit auditable artifacts |
| G8 External evaluation hygiene | sealed tasks, no leakage, reproducible environment, pass/fail plus cost metrics |
| G9 Release | CI, security review, docs, install path, and zero reachable stubs |

## 7. Phased Implementation Plan

Each phase is blocked by all earlier completion gates. Do not parallelize downstream feature work across an unstable contract.

### Phase 0 — Freeze, Reconcile, and Establish Truth

**Build**

- Resolve the current unmerged `nsynth/src/lib.rs` state and checkpoint the feature branch without discarding other agents' work.
- Inventory every `todo!`, `unimplemented!`, `NotImplemented`, `simulate`, fake success, placeholder transpilation, and ignored failing test.
- Produce one command manifest for format, build, focused tests, full tests, clippy, and release build.
- Split pre-existing failures from new regressions and record deterministic reproductions.

**Documentation/API references**

- Existing `PHASE_2_4_EXECUTION_PLAN.md`, `phase2_agentic_architecture.md`, root `MASTER_ROADMAP.md`.
- Cargo workspace/package definitions only; no new runtime API.

**Verification**

- `git diff --check`; `cargo fmt --check`; debug and release `cargo check`.
- Full test run completes without process abort or stack overflow, even if known failures remain quarantined with owned issues.
- A generated stub audit fails CI when forbidden production tokens are reachable.

**Do not** hide failures with global warning suppression, delete tests, or relabel simulations as experimental without isolating them from production.

### Phase 1 — Canonical Runtime and Capability Registry

**Build**

- Introduce the canonical contracts from Section 5 and a single explicit `AgentRun` state machine: `Created -> Understanding -> Planning -> Executing -> Verifying -> Revising -> Succeeded|Failed|Cancelled`.
- Add budget accounting for wall time, attempts, commands, bytes read/written, and synthesized candidates.
- Add a capability registry whose states are `Implemented`, `Experimental`, `Deferred`, or `Absent` and whose `Implemented` entries link to executable conformance tests.
- Migrate `TaskDecomposer`, `GoalTree`, `DependencyResolver`, executor, and repo types to the shared contracts.

**References**

- Existing `agent/{planning,hierarchy,dependencies,executor}.rs` and `agent/repo/*.rs` public APIs.
- Preserve standard solver entrypoints in `solver.rs`; remove alternate agent runtime definitions after migration.

**Verification**

- State transition property tests, cancellation tests, budget exhaustion tests, schema round trips, old-schema migration tests.
- Compile-fail tests prevent illegal transitions and unverified experience recording.

**Do not** store important state in free-form strings or maintain two “canonical” task models.

### Phase 2 — Finish Linguigenesis Coding Comprehension Upstream

**Build in `../linguigenesis`**

- Add a coding ontology in KVRM data (`data/coding_registry.json`): repository, file, symbol, dependency, diagnostic, behavior, constraint, workflow, acceptance oracle, and risk entities with relations—not Rust keyword tables.
- Implement `RequirementDeriver` and `CodingComprehension`: BeliefState + registry graph → `SynthesisRequirement` (signature, `example_cases`, category, confidence, evidence IDs, `unresolved`).
- Extend comprehension from keyword intent to registry workflow entities (`signal_lemmas`, `maps_to_intent`, `coding_workflow`) and compositional coding requests, negation, scope, constraints, references, and follow-up resolution.
- Add a typed coding dialogue state that preserves evidence spans and explicitly represents ambiguity when `unresolved` is non-empty.
- Finish or isolate the deferred `lg-communicator` environment/training adapters; no deferred training API may be advertised as active capability.
- Add coding utterance corpora with paraphrase, adversarial ambiguity, conflicting instructions, and multi-turn corrections.

**Registry schema (operations)**

| Attribute | Purpose |
|---|---|
| `arity`, `input_types`, `output_type` | signature synthesis |
| `example_cases` | JSON `[{inputs, expected}]` — sole source of synthesis examples |
| `nsynth_category` | solver routing hint |
| `signature_template` | `fn {name}({params}) -> {return}` |

**Relations:** `synonym`, `domain`, `entailment`, `code_pattern`, `type_of`.

**Allowed existing APIs**

- `linguigenesis_core::Registry::{new,from_json,from_json_auto,merge_from_json,get_by_lemma,query,get_related}`.
- `Comprehension::{new,parse}` and `BeliefState`/`IntentType`.
- `RequirementDeriver`, `CodingComprehension`, `SynthesisRequirement` (Package C).
- `AnalogyReasoner`, `MultiHopReasoner`, and `KnowledgeQA` for grounded relations.
- `lg-communicator` comprehension/policy/dialogue only after its environment contract is unified and tested.

**Verification**

- Frozen train/dev/test split with unseen vocabulary and paraphrases.
- Exact or structural match for `SynthesisRequirement`; paraphrase via synonym → identical examples.
- Calibration error for confidence; clarification precision/recall when `unresolved` non-empty.
- Mutation tests for negation, file scope, version constraints, and “do not change” clauses.
- `rg` audit: zero `text.contains` semantic routing in `comprehension.rs` production path.
- A zero-network test proves the native path works without any model API.

**Do not** replace understanding with regex routing, keyword-only classification, hard-coded examples in Rust, or an external-model call hidden behind an interface.

### Phase 3 — Typed Linguigenesis-to-nsynth Bridge

**Build**

- Convert `SynthesisRequirement` into `CodingIntent`, then ground against `RepositorySnapshot` to produce `CodeTaskSpec`.
- `linguigenesis_bridge.rs` must call `CodingComprehension` only; delete `generate_examples_from_text` and migrate `populate_code_entities` into registry JSON.
- Map `SynthesisRequirement` → `Problem` / `ParsedRequirements` with evidence entity IDs and confidence.
- Reject unsupported or low-confidence conversions with typed clarification questions.
- Preserve source text spans, registry entities, reasoning path, and conversion diagnostics.
- Retire or quarantine `nl/mod.rs` paths that return `NotImplemented`; the canonical CLI must never select them.

**References**

- Existing `nsynth/src/linguigenesis_bridge.rs` is the migration seed.
- Use only stable upstream APIs named in Phase 2; pin the sibling path revision for benchmark runs.

**Verification**

- End-to-end request-to-spec fixtures across all ten workflows.
- Multi-turn tests where a clarification changes only the ambiguous field.
- Repository grounding tests prevent hallucinated files, symbols, commands, and frameworks.

**Do not** silently fill missing acceptance criteria or convert `Unknown` intent into a guessed edit.

### Phase 4 — Deterministic Repository Model and Context Selection

**Build**

- Discover manifests, workspace roots, languages, targets, test layouts, generated paths, and repository instructions.
- Parse symbols/imports/calls/types with language adapters; start with Rust, then add languages only with conformance suites.
- Build a content-addressed repository graph with incremental invalidation.
- Select context by task evidence, dependency reachability, diagnostics, tests, and change history—not global directory scanning.
- Respect `.gitignore`, tool-specific ignores, size/binary limits, symlinks, submodules, and nested repositories.

**References**

- Reuse `knowledge/{graph,api_graph}.rs` concepts where correct.
- Replace the global `agent/repo/hardness.rs` traversal with repository-index queries.

**Verification**

- Golden repository maps, incremental update tests, cross-platform path tests, symlink-loop tests, binary/large-file tests.
- Retrieval benchmark measures relevant-file recall, irrelevant bytes, and symbol/dependency coverage.

**Do not** read `target/`, `.git/`, secrets, or the whole repository to answer every task.

### Phase 5 — Deny-by-Default Tool Runtime

**Build**

- Define one async typed tool trait and registry; every call passes through policy, audit, timeout, cancellation, output caps, and redaction.
- Filesystem: descriptor/canonical-root confinement, symlink-safe create/read/write/rename/delete, atomic writes.
- Process: argv-based execution without implicit shell, minimal environment, cwd confinement, process-group termination, resource limits.
- Git: read operations by default; mutations require scoped approval and cannot publish automatically.
- Network: disabled by default; allowlisted scheme/domain/IP/port, DNS rebinding and redirect checks, private-address denial.
- Database: real configured drivers with parameter binding; remove in-memory emulation from production capability claims.

**References**

- Migrate `agent/tools/{fs,shell,git,http,database,registry}.rs` behind the canonical trait.
- Centralize `agent/repo/guardrails.rs` decisions in a new policy engine; no tool-local policy forks.

**Verification**

- Traversal, symlink race, Unicode path, prefix confusion, command injection, timeout, output flood, secret exfiltration, SSRF, redirect, DNS rebinding, and SQL-injection tests.
- Every denied operation produces a redacted audit event and no side effect.

**Do not** use substring command checks, shell out to `curl`, or treat lexical path normalization as containment.

### Phase 6 — Transactional Structured Edit Engine

**Build**

- Execute changes in an isolated git worktree or equivalent content-addressed transaction.
- Support exact replacements, structured symbol edits, file create/delete/rename, dependency manifest edits, and unified-diff import.
- Require base hashes and detect stale/conflicting edits.
- Format changed files through policy-gated tools, show the resulting diff, and rollback atomically on rejection/failure.
- Upgrade `PatchGate` to understand `/dev/null`, renames, modes, binary/submodule/symlink edits, patch size, generated files, secrets, immutable paths, and no-op patches.

**Verification**

- Property tests for apply/reverse round trips and crash-safe rollback.
- Adversarial diff corpus and concurrent-edit tests.
- Tests prove no operation escapes the transaction or modifies the caller's base revision.

**Do not** infer safety solely from `--- a/` and `+++ b/` text prefixes.

### Phase 7 — Real Closed-Loop Repository Agent

**Build**

- Implement `RepoAgent::run(CodeTaskSpec) -> AgentRunResult` as a bounded state machine.
- Establish a baseline, select context, form hypotheses, plan, synthesize or construct edits, apply transactionally, run the narrowest informative oracle, parse evidence, revise, then run the full acceptance oracle.
- Make `test_command`, acceptance criteria, iteration/time/token budgets, and allowed paths operational.
- Parse structured compiler/test output where available and retain raw capped evidence.
- Assign credit only after comparing baseline and final verifier reports.

**References**

- `agent/repo/repo_agent.rs` becomes the loop owner.
- `failure_parser.rs`, `credit.rs`, and `trace.rs` become evidence adapters, not independent in-memory demos.
- Standard `solve_problem`/`solve_problem_agentic` remain real synthesis providers, never fake task completion.

**Verification**

- Seeded single-file bug fixtures, timeout, compile error, test failure, runtime panic, regression, invalid patch, stale base, and exhausted-budget tests.
- A negative-control fixture must remain unsolved rather than produce a false success.

**Do not** count “proposal generated,” “patch accepted,” or “command ran” as success; only the final oracle decides.

### Phase 8 — Workflow Library and Supervision

**Build**

- Encode bug-fix, feature, test, refactor, dependency, migration, review, explanation, and greenfield workflows as typed DAG templates.
- Add a supervisor that schedules independent steps, detects blocked/deadlocked work, cancels dependents, and enforces global budgets.
- Introduce specialist actors only where their tools and output schemas differ materially; all outputs are evidence-backed proposals.
- Implement human approval checkpoints for scope expansion, destructive actions, credentials, network, publishing, and ambiguous requirements.

**Verification**

- Workflow conformance fixtures for each template.
- Scheduling property tests, cancellation propagation, deadlock detection, approval resume/deny, and deterministic seeded replay.

**Do not** create role-playing “agents” that merely emit different prose or simulated reviews.

### Phase 9 — Durable Memory, Context, and Resume

**Build**

- Persist runs, events, repository/version fingerprints, verified outcomes, failed hypotheses, decisions, and user constraints.
- Separate episodic task history, semantic repository knowledge, and reusable verified solution patterns.
- Add retention, compaction, provenance, redaction, schema migration, and repository/revision invalidation.
- Resume interrupted runs from the last durable state only after confirming repository hashes and policy.

**Verification**

- Crash/restart tests at every state boundary, corrupt-record recovery, migration tests, secret-redaction tests, stale-revision rejection.
- Retrieval evaluations measure whether memory improves success without importing irrelevant or cross-project data.

**Do not** persist self-authored claims as facts or share memory across projects without explicit scope.

### Phase 10 — Project-Scale Code Generation

**Build**

- Extend synthesis from isolated functions to modules, public interfaces, manifests, dependency choices, tests, and buildable project layouts.
- Replace placeholder non-Rust transpilation with real language backends or mark those targets absent.
- Add interface-first decomposition, contract generation, incremental compilation feedback, and cross-file consistency checks.
- Start with high-quality Rust support; graduate each additional language independently.

**Verification**

- Greenfield projects build and run from empty directories.
- Multi-file API evolution fixtures catch missing call sites, imports, serialization, docs, and tests.
- Language conformance matrix requires formatter, compiler/typechecker, unit test, and packaging checks.

**Do not** claim language support because code-shaped text can be emitted.

### Phase 11 — Validation, Recovery, and Security as Mandatory Gates

**Build**

- Connect existing `ValidationPipeline`, test generation, coverage, vulnerability checks, and recovery to every modifying workflow.
- Define validation profiles per repository/language and change risk.
- Detect flaky tests through bounded reruns and distinguish infra failures from patch failures.
- Require full-scope regression validation before success.

**Verification**

- Injected syntax, type, logic, race, security, flaky-test, and environment failures.
- Tests prove validation cannot be bypassed by an actor or optional configuration for release profiles.

**Do not** leave validation components as library-only modules used solely by their own unit tests.

### Phase 12 — Product Interfaces and Integration

**Build**

- A canonical Rust CLI with `ask`, `plan`, `run`, `resume`, `inspect`, `approve`, `benchmark`, and `capabilities` commands.
- Streaming typed events and stable JSON output for automation.
- Optional MCP/IDE/GitHub adapters as thin clients over the same session API.
- Clear diff, evidence, approvals, budget, and final-verification reporting.

**Verification**

- CLI golden tests, JSON schema compatibility, Ctrl-C cancellation, terminal resize/error handling, adapter contract tests.
- End-to-end tests invoke the installed binary in clean temporary repositories.

**Do not** implement separate agent logic in each interface.

### Phase 13 — Instrumentation and Reproducibility

**Build**

- Record stage timings, tool latency, bytes/context, candidates, retries, changed lines, verifier commands, environment fingerprint, and outcome.
- Use stable run IDs and content hashes; export JSONL plus human-readable reports.
- Support deterministic seeds where algorithms allow them and state unavoidable nondeterminism.

**Verification**

- Schema tests, replay tests, metric invariants, redaction audits, and report regeneration from raw events.

**Do not** optimize or train on metrics that cannot be reproduced from saved evidence.

### Phase 14 — Replace the 20-Task Metadata List with an Executable Local Benchmark

**Build**

- Keep the existing task descriptions only as manifests.
- For every task, provide a versioned fixture or deterministic mutation that creates the failure, a setup/reset command, acceptance and regression oracles, allowed scope, timeout, and expected difficulty evidence.
- Run each task in a fresh isolated worktree/container and emit patch, trace, commands, timings, and score.
- Include at minimum five single-file repairs, five test tasks, five multi-file features, and five regressions/refactors, plus negative controls and security adversaries.
- Compute hardness from observed solver/tool outcomes, not identical preassigned dimension values.

**Metrics**

- resolved-task rate; oracle pass rate; regression-free rate; false-success rate; unsafe-action rate; clarification quality; first-pass yield; attempts; wall time; commands; bytes read; patch size; reproducibility.

**Verification**

- Harness self-tests prove reset isolation, deliberate broken baseline, oracle discrimination, timeout enforcement, and score reproducibility.
- Every fixture fails before the agent and passes only after a valid repair.

**Do not** count metadata validation as a benchmark run.

### Phase 15 — External Benchmarks and Sealed Evaluation

**Build**

- Add adapters only after G0–G7: function-level sanity sets, SWE-bench-style repository tasks, multilingual repo tasks, and terminal/tool-use tasks.
- Pin dataset version, container image, dependency cache policy, base revision, and scorer.
- Separate development tasks from sealed evaluation tasks and log all contamination risks.
- Compare native-only, ablated, and optional-external-proposer configurations.

**Required ablations**

- no Linguigenesis reasoning; no repository graph; no memory; no iterative repair; no specialized synthesis; optional external proposer off/on.

**Verification**

- Re-run samples twice; reconcile nondeterminism; retain complete artifacts.
- Publish resolved counts, denominator, failures, unsafe actions, time, compute, and configuration—not a single cherry-picked percentage.

**Do not** tune against sealed answers, import patches into memory, or benchmark before the local harness is trustworthy.

### Phase 16 — Evidence-Driven Curriculum and Self-Improvement

**Build**

- Mine difficulty, failure clusters, useful primitives, routing decisions, and reusable verified transformations from benchmark traces.
- Promote learned patterns only through held-out improvement and regression/security gates.
- Apply synthesis portfolio search, analogy, world modeling, meta-learning, and recursive improvement only to measured bottlenecks.
- Treat every self-generated change as an ordinary untrusted patch.

**Verification**

- Held-out A/B evaluation, overfit-rate reporting, rollback, catastrophic-regression checks, provenance and reproducibility.

**Do not** reward self-ratings, training-set wins, or unverifiable “novelty.”

### Phase 17 — CI, Release, and Publication

**Build**

- CI tiers: fast format/check/unit; security/adversarial; fixture E2E; nightly full/regression/benchmark.
- Release profiles default to native Linguigenesis, network disabled, safe policy, and explicit capability report.
- Reproducible install, versioned schemas, migration policy, threat model, operator runbook, benchmark card, and architecture docs.
- Prepare research artifacts around verified program synthesis plus native learned language grounding, with ablations and limitations.

**Verification**

- Clean-machine install and smoke task; release build; artifact checksum; docs command verification; zero reachable-stub audit; security sign-off.

**Do not** release on unit-test count alone.

## 8. Repo-Agent Scaffold Review and Required Corrections

The new `nsynth/src/agent/repo/` layer is worth keeping because its types create clear seams. Its current tests are honest unit tests for those seams, but the roadmap language must not imply an executing agent.

### Keep

- separation of task spec, guardrails, hardness, failure parsing, patch gate, credit, trace, and coordinator;
- Rust-native implementation;
- explicit allowed paths, acceptance criteria, budgets, and expected hardness;
- focused unit-test module and 20-task manifest seed.

### Correct before G5

1. Add the real `RepoAgent::run` loop and make all task fields operational.
2. Replace lexical path/pattern checks with root-confined filesystem policy.
3. Fix decision precedence so secrets always deny even when an unsafe-token check also matches.
4. Replace global recursive hardness scanning with indexed, task-local evidence; detect inline tests.
5. Parse structured diagnostics and order classifications so generic `error` does not swallow test/runtime failures.
6. Make patch parsing structured and cover creates/deletes/renames/modes/binary/symlink/submodule/no-op/size/secrets.
7. Persist traces with IDs, timestamps, hashes, redaction, tool outcomes, and verifier evidence.
8. Validate/clamp credit and link it to baseline/final reports; prohibit pre-verification credit.
9. Convert all 20 task manifests into real isolated fixtures and add negative/security tasks.

## 9. Execution Packages

Use these as review-sized implementation blocks. A package closes only when its gate evidence is attached.

| Package | Phases | Deliverable |
|---|---|---|
| A | 0 | reconciled tree, baseline, stub inventory |
| B | 1 | canonical contracts/runtime/capabilities |
| C | 2 | coding registry, `RequirementDeriver`, emergent NL comprehension |
| D | 3 | bridge → `SynthesisRequirement` → `Problem`; quarantine legacy NL |
| E | 4 | repository index/context benchmark |
| F | 5 | secure tool runtime |
| G | 6 | transaction/edit/rollback engine |
| H | 7 | first real closed-loop repairs |
| I | 8 | workflow templates and supervision |
| J | 9 | durable memory and resume |
| K | 10–11 | project generation and mandatory verification |
| L | 12–13 | CLI/session API and telemetry |
| M | 14 | executable 20+ task local benchmark |
| N | 15 | sealed external benchmark adapters |
| O | 16 | gated learning/self-improvement |
| P | 17 | CI/release/publication package |

Rough effort from the current tree, assuming focused implementation and no major solver regressions: **12–18 focused sessions to a credible native MVP (through H), 24–35 to benchmark readiness (through M), and additional work for external evaluation/release.** Progress is gate-based, not calendar-based.

## 10. Definition of Done

The coding agent is fully functioning only when one clean install can:

1. understand an unseen natural-language repository task through Linguigenesis;
2. ask a precise clarification when required;
3. produce a repository-grounded, reviewable plan;
4. execute real, policy-gated tools in isolation;
5. construct and apply a scoped multi-file patch transactionally;
6. diagnose failures and revise within declared budgets;
7. pass real acceptance and regression oracles;
8. rollback or stop safely when it cannot succeed;
9. persist a redacted, replayable evidence trail and resume safely;
10. report capabilities and limitations truthfully;
11. pass the executable local benchmark with zero false-success and unsafe-action events;
12. run sealed external benchmarks without task leakage.

Until then, call it an experimental coding-agent stack—not a completed autonomous coding agent.

## Appendix A — Documentation and API Discovery Index

Agents must read implementations and tests, not merely this list.

| Topic | Source of truth | Known valid entry points |
|---|---|---|
| Linguigenesis registry | `../linguigenesis/rust/linguigenesis-core/src/registry.rs` | `Registry::{new,from_json,from_json_auto,get_entity,get_by_lemma,query,get_by_type,get_related}` |
| Linguigenesis coding requirements | `../linguigenesis/rust/linguigenesis-core/src/coding_requirements.rs` | `RequirementDeriver::{new,derive}`, `SynthesisRequirement`, `ExampleSpec` |
| Linguigenesis coding registry | `../linguigenesis/data/coding_registry.json` | operations, workflows, constraints (data — not Rust keyword tables) |
| Linguigenesis comprehension | `../linguigenesis/rust/linguigenesis-core/src/comprehension.rs` | `Comprehension::{new,parse}`; workflow-based intent only after Package C |
| Belief and intent | `../linguigenesis/rust/linguigenesis-core/src/belief.rs` | `BeliefState`, `IntentBelief`, `IntentType`, `Constraint` |
| Native reasoning | `../linguigenesis/rust/linguigenesis-core/src/reasoning.rs` | `AnalogyReasoner`, `MultiHopReasoner`, `KnowledgeQA` |
| Linguigenesis dialogue | `../linguigenesis/rust/lg-communicator/src/` | use only tested `comprehend`, policy, and dialogue APIs; training/env parity is deferred |
| nCPU synthesis | `nsynth/src/solver.rs`, `nsynth/src/solver/`, `nsynth/src/synthesis/` | `solve_problem` and verified solver routes |
| Existing planning | `nsynth/src/agent/{planning,hierarchy,dependencies,executor}.rs` | migrate into canonical runtime; do not preserve simulated execution |
| Repo scaffold | `nsynth/src/agent/repo/` | seams only until `RepoAgent::run` closes G5 |
| Existing tools | `nsynth/src/agent/tools/` | scaffolds to migrate behind one policy-gated trait |
| Existing validation | `nsynth/src/validation/`, `nsynth/src/testing/` | must become mandatory modifying-workflow gates |
| Detailed closure plan | `nsynth/docs/LINGUIGENESIS_NATIVE_CODING_AGENT_PLAN.md` | architectural rationale and same phase ordering |

## Appendix B — Required Evidence Artifact per Package

Every package handoff must contain:

1. starting and ending revision plus dirty/unmerged state;
2. exact scope and files intentionally changed;
3. APIs/documentation read before implementation;
4. acceptance criteria and anti-pattern checks;
5. exact commands and exit codes;
6. test counts: passed, failed, ignored, filtered, aborted;
7. known failures with shortest reproducer and assigned next package;
8. security/adversarial results where applicable;
9. capability-state changes with evidence links;
10. next safe action and files that concurrent agents must avoid.

Store machine-readable run evidence under a dedicated artifact directory once Package B defines the schema. Until then, record it in the session summary and the Execution Ledger.

## Appendix C — Mandatory Agent Handoff Template

```markdown
## Package / Gate
[package, gate, state before -> state after]

## Objective
[one concrete capability or blocker closed]

## Starting State
[revision, dirty files, unmerged entries, known failing command]

## Documentation and APIs Read
- [exact file and relevant symbol/section]

## What Changed
[behavior in plain English; do not provide a file-by-file changelog]

## Verification
- `[exact command]` -> [exit code, counts]

## Anti-Pattern Audit
- [forbidden patterns checked and results]

## Remaining Risks
- [specific reproducible issue]

## Next Safe Action
[one bounded next step, dependencies, and files to avoid]
```

## Appendix D — Roadmap Maintenance Rules

- Only this file controls package/gate state.
- Update the snapshot date and exact evidence whenever a package changes state.
- Architecture changes require updating both this roadmap and the companion plan in the same patch.
- Never remove a failed result; mark it superseded with the newer revision and evidence.
- Estimated percentages are informational. Gates, tests, and artifacts decide readiness.
- Benchmark readiness means G0–G7 all pass; it does not mean external benchmark success.
- “Ultimate agent” research begins after the MVP loop is real, safe, and measured—not instead of it.
