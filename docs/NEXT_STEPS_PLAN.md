# nCPU / nSynth — Next-Steps & Novel-Opportunities Plan

_Authored 2026-06-19. Grounded in actual repository state, not the aspirational roadmap._

## 0. The One Insight That Governs Everything

Across this codebase the recurring reality is **capability built as an island, then left unwired**:

- Phase 2 `src/agent/` had 98 passing tests but nothing called it → fixed this session (`--agentic`, compositional synthesis).
- `Problem.functions` existed but was dead → made live this session.
- `src/learning/experience.rs` has a full `ExperienceDB` (record / find-similar / effective-actions / Lessons) — but is it on the live solve path?
- `src/knowledge/api_graph.rs` is a multi-language API knowledge graph — consumed by the solver?
- `src/meta_learner.rs` is a real meta-learner (weights, persistence, teacher ranking, `record_transfer_success`) — already partly wired into routing.

**Therefore the highest-leverage work is rarely "write a new module." It is "connect an existing island to the live synthesis loop, then prove the connection with a before/after metric."** Every phase below is framed as *what exists → the wiring gap → the concrete change → how we prove it*. Greenfield is called out explicitly when it is genuinely new.

---

## 1. Immediate (close Phase 2 fully) — ~3-5 days

### 1.1 CLI-demoable compositional synthesis
- **Exists**: `solve_compositional` (executor-driven, non-fabricating), reachable via API + 3 unit tests.
- **Gap**: no benchmark populates `Problem.functions`, so `--agentic` on the CLI always takes the single-shot branch. Compositional is invisible from the command line.
- **Change**: (a) add a `multi_function` benchmark fixture (e.g. `compose_add_mul`, a small pipeline) to `benchmark.rs`; (b) teach `--problem-json` to parse a `functions[]` array into `FunctionDef`s.
- **Prove**: `mog_synth --problem compose_add_mul --agentic` emits both function bodies; method tag `agentic_compositional[...]`.
- **Effort** S · **Risk** low · **Dep** none.

### 1.2 Wire `ToolRegistry` into an agent loop
- **Exists**: 5 real tools + registry (25 tests). `orchestrator.rs` / `debate.rs` agents that currently cannot act on the world.
- **Gap**: agents reason but cannot read files, run git, or persist state.
- **Change**: give `Orchestrator` an optional `ToolRegistry`; add a `ToolUsingAgent` capability so a synthesis result can be written to disk, formatted via shell, and committed via git — all behind the existing sandbox/allowlist.
- **Prove**: integration test — orchestrator synthesizes → writes `out.rs` via `FsTool` → `git status` via `GitTool` shows it.
- **Effort** M · **Risk** low (boundaries already enforced) · **Dep** 1.1 optional.

### 1.3 Persist + replay agent runs
- **Exists**: `Executor` progress/`ExecutionSummary`; `learning/experience.rs` `ExperienceDB`.
- **Gap**: agentic runs are not recorded as experiences.
- **Change**: on every `solve_problem_agentic`, call `ExperienceDB::record_experience` with the decomposition + outcome.
- **Prove**: after N agentic solves, `find_similar_problems` returns them.
- **Effort** S · **Risk** low · **Dep** 1.2.

---

## 2. Phase 3 — World Model & Knowledge Integration — ~4-6 weeks

**Theme**: turn the existing knowledge islands into a queryable world model the solver consults *before* search.

### 3.1 Unify the knowledge graph (build on `knowledge/api_graph.rs` + Linguigenesis)
- **Exists**: `api_graph.rs` (APFunction, UsagePattern, MigrationPath, multi-language); `linguigenesis_bridge.rs` (Entity/EntityType/RelationType, `query_knowledge`, 1435+ code entities, CodePattern relations).
- **Gap**: two graphs, no shared `CodeKnowledgeGraph`, no embedding similarity, no query language.
- **Change**: `src/knowledge/graph.rs` — a `CodeKnowledgeGraph` that ingests both sources; entity types Function/Pattern/Concept/Solution; relations Uses/Implements/SimilarTo/Solves; cosine similarity over feature vectors (reuse `meta_learner::extract_problem_features`, no new embedding model needed).
- **Prove**: `graph.query("solves: sort")` returns sort solutions ranked by similarity; <100ms lookup over 10K relations.
- **Effort** L · **Risk** med (graph scale) · **Dep** none.

### 3.2 Analogy-driven synthesis (the killer app of the KG)
- **Exists**: `solved_cache.rs`, `meta_learner::record_transfer_success`, `find_similar_problems`.
- **Gap**: similar solved problems are not *adapted* to new targets — only exact-cache hits help.
- **Change**: `src/solver/analogy.rs` — `AnalogySolver` finds A:B::C:D structural mappings between a new `Problem` and solved ones, transplants the solved program with variable/constant remapping, then **verifies against the new examples** (non-fabricating: an adapted solution that fails examples is discarded).
- **Prove**: a problem absent from cache but analogous to a solved one is solved by transfer; measure transfer success rate on a held-out analogy set.
- **Effort** L · **Risk** med · **Dep** 3.1.
- **This is the single most novel near-term capability** — see §7.1.

### 3.3 Causal / behavioral model (lighter than roadmap)
- **Reframe**: full causal inference is over-scoped. The useful slice is a **behavioral effect model**: predict whether an edit to a synthesized program preserves the examples (a cheap pre-check before re-running the solver).
- **Change**: `src/reasoning/effects.rs` — track which AST mutations historically broke vs preserved correctness; gate the optimizer with it.
- **Effort** M · **Risk** low · **Dep** 1.3 (needs experience data).

---

## 3. Phase 4 — Meta-Learning & Adaptation — ~3-4 weeks (mostly wiring!)

**Theme**: the meta-learner and experience DB largely exist; close the loop so the system measurably improves with use.

### 4.1 Close the learning loop
- **Exists**: `meta_learner.rs` (weights, gradient update, teacher ranking via meta); `experience.rs` (`get_effective_actions`, `Lesson`).
- **Gap**: lessons are recorded but not fed back into routing decisions at solve time.
- **Change**: in the solver pipeline, before choosing a strategy, query `ExperienceDB::get_effective_actions(problem)` and bias `rank_teachers_with_meta` toward historically-winning strategies for similar problems.
- **Prove**: A/B — synthesis success rate and median time, learning-loop ON vs OFF, over the benchmark suite run twice (cold vs warmed experience DB). Target: measurable lift on the second pass.
- **Effort** M · **Risk** low · **Dep** 1.3.

### 4.2 Prioritized experience replay + curriculum
- **Exists**: `ExperienceDB` storage.
- **Gap**: no prioritization, no curriculum ordering for the meta-learner's gradient updates.
- **Change**: `src/learning/replay.rs` — priority = surprise (residual between predicted and actual difficulty); sample hard-but-solvable first.
- **Prove**: meta-weights converge faster (fewer updates to same success rate) with prioritized vs uniform replay.
- **Effort** M · **Risk** med · **Dep** 4.1.

### 4.3 Strategy auto-acquisition
- **Change**: when a novel teacher/strategy combination repeatedly wins, promote it to a first-class routing option (data-driven, logged). Hook into the existing `search_family_router.rs`.
- **Effort** M · **Risk** med · **Dep** 4.1, 4.2.

---

## 4. Phase 5 — AGI Precursors — ~6-8 weeks (gate on safety)

**Theme**: the tool framework + executor + meta-learner make *bounded* recursive self-improvement actually implementable — but only with hard safety rails.

### 5.1 Recursive self-improvement (the headline, done safely)
- **Exists**: `ToolRegistry` (fs/shell/git), `Executor` (rollback!), `meta_learner` (measurable objective), benchmark suite (the fitness function).
- **Change**: `src/meta/recursive.rs` — a loop that (1) proposes a change to its *own* meta-weights or routing config, (2) **runs the full benchmark suite as the safety/fitness gate**, (3) keeps the change only if success-rate strictly improves and no regression, else `Executor` rolls back. Self-modification is restricted to *data/config* (weights, routing tables), **never** arbitrary code edits, and every proposal is logged.
- **Safety rails** (non-negotiable, written explicitly): allowlist of mutable targets; mandatory full-suite pass; monotonic-improvement constraint; bounded improvement budget; full audit log; human-readable diff of every accepted change.
- **Prove**: starting from cold weights, N self-improvement cycles raise benchmark success rate monotonically with zero regressions; every cycle's diff is auditable.
- **Effort** XL · **Risk** HIGH (this is the dangerous one — gate hard) · **Dep** 4.1-4.3.

### 5.2 Transfer learning across domains
- **Exists**: multi-language transpile (Phase 19), `api_graph` per-language, analogy (3.2).
- **Change**: `src/meta/transfer.rs` — reuse a Rust-domain solution as a prior for the JS/Python version of the same problem; measure positive transfer.
- **Effort** L · **Risk** med · **Dep** 3.2.

### 5.3 Symbol grounding (scoped to code semantics)
- **Reframe**: ground synthesized symbols in *executable behavior* (run the candidate, observe I/O) rather than perception. Much of this is the existing verifier; formalize it as grounding-confidence per symbol.
- **Effort** M · **Risk** low.

---

## 5. Phase 6 — Production Deployment — ~4-6 weeks

- **Exists**: `bin/nsynth_serve.rs` (server), MCP server (`mcp__ncpu-synth__*`), vast.ai deploy scripts, SLURM suite.
- **Gaps / changes**:
  - **Concurrency**: connection pool + work-stealing scheduler for parallel solves (the `Executor` semaphore is the seed).
  - **Observability**: structured metrics (success rate, p50/p99 latency, overfit-rate) exported from the serve binary.
  - **Caching tier**: promote `solved_cache` to a shared/persistent store for multi-instance.
  - **Resource limits**: per-request synthesis budget + the disk-pressure guard (this session lost time to a full disk — add a pre-build/pre-run free-space check + auto-clean of regenerable caches).
- **Effort** L · **Risk** low-med · **Dep** none (parallelizable with Phase 3-4).

---

## 6. Cross-Cutting (do continuously)

- **Wire-the-island audit**: for every `pub mod`, assert it is reachable from a live entry point or a test that exercises the live path. Prevents new islands.
- **Non-fabrication invariant**: keep the property that *all emitted code originates from the gated solver* across analogy/transfer/self-improvement. It is the project's defensible core — never let an "agent" path mint unverified code.
- **Metric discipline**: every capability ships with a before/after number (success rate, time, overfit-rate, transfer rate). No "it works" without a count.
- **Disk/CI hygiene**: free-space precheck; `cargo clean` of stale targets in CI; the machine sits near-full.

---

## 7. Novel Opportunities (research, OSS, monetization)

### 7.1 Research papers (each has a real, measurable claim)
1. **"Verifiably Non-Hallucinating Agentic Program Synthesis."** The composition/analogy/self-improvement layers can never emit unverified code — every body comes from a gated solver. Pair with the existing **overfit-rate** metric for a generalization-bounded story. _This is the flagship and it is nearly demonstrable today._
2. **"Analogy Transfer as Cache Generalization."** §3.2 — quantify how structural A:B::C:D transfer extends a solved-cache beyond exact hits. Clean ablation: exact-cache vs analogy-cache success on held-out analogues.
3. **"Bounded Recursive Self-Improvement with a Benchmark Fitness Gate."** §5.1 — self-improvement restricted to weights/config, full-suite gate, monotonic constraint, full audit. A *safe* RSI demonstration is publishable precisely because it is bounded and reproducible.
4. **"Decompose-Synthesize-Compose"** as a general harness: swap the per-task synthesizer (search / ML / probabilistic) and report heterogeneous synthesis results.

### 7.2 Open-source surfaces
- **`mog-agent-tools`**: the dependency-light (std + shell-out), safety-boundaried Rust tool crate (fs-sandbox, shell-allowlist, git, http, in-mem db). Genuinely useful standalone for the Rust-agent ecosystem.
- **`mog-executor`**: the async task-graph executor (ordering, bounded parallelism, retry, rollback, progress) as a small reusable crate.
- **The benchmark + overfit-rate harness** as a synthesis-evaluation standard.

### 7.3 Monetization / product
- **Hosted synthesis API** (the serve binary + MCP already exist): NL/examples → verified multi-language code, billed per solve, with the non-fabrication guarantee as the differentiator vs LLM codegen.
- **"Verified codegen" enterprise tier**: every emitted function comes with the examples it provably satisfies — auditable, regression-safe; sells into regulated/safety-critical shops where LLM hallucination is disqualifying.
- **IDE plugin**: compositional synthesis of helper functions from inline examples.

---

## 8. Recommended Sequence (opinionated)

1. **Finish Phase 2 wiring (§1.1-1.3)** — small, unlocks demos + experience capture. _Do first._
2. **Phase 4.1 learning loop (§4.1)** — mostly wiring existing parts; first measurable self-improvement number. _High ROI._
3. **Phase 3.1-3.2 knowledge graph + analogy (§3.1-3.2)** — the flagship novel capability + paper #1/#2.
4. **Phase 5.1 bounded RSI (§5.1)** — only after 3-4 give a stable fitness signal; gate on safety.
5. **Phase 6 productionize** — in parallel once Phase 3-4 stabilize.

Rationale: maximize measurable improvement per unit effort early (wiring > greenfield), front-load the defensible research claims (non-fabrication + analogy), and defer the high-risk self-modification until the fitness signal and rollback machinery are proven.
