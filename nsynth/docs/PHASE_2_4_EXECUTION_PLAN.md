# nCPU Phase 2–4 Completion Plan

**Status:** Implementation-ready plan  
**Prepared:** 2026-06-19  
**Scope:** Finish Phase 2 agentic infrastructure, then build the data foundation, curriculum learning, synthesis-portfolio NAS, world-model integration, meta-learning, and publication-grade experiments.

## 1. Why This Plan Exists

The previous workflow completed the Phase 2 architecture document and the first planning block:

- adaptive task decomposition;
- goal hierarchy management;
- dependency resolution;
- asynchronous execution;
- the existing debate and collaborative-orchestration prototypes.

That is not the whole Phase 2 roadmap. Tool execution, unified memory, real specialized agents, solver/CLI integration, curriculum learning, synthesis-portfolio NAS, and paper automation remain. The repository also already contains partial implementations that must be reused rather than duplicated.

This plan sequences the work so that later learning systems consume real traces from a working agentic runtime instead of training on simulated agent outputs.

## 2. Non-Negotiable Constraints

1. The legacy `solve_problem` path remains behaviorally unchanged unless `--agentic` is explicitly selected.
2. Generated programs must still pass existing runtime/holdout verification before being accepted or stored.
3. File, shell, Git, HTTP, and database tools are default-deny and policy-gated.
4. No agent may claim success from a simulated result.
5. Learning/NAS experiments use separate train, validation, and sealed evaluation sets.
6. Every phase ends with tests, a release build, documentation, and an artifact containing measured results.
7. Existing infrastructure is wrapped or extended before introducing a second implementation of the same concern.
8. Work begins only after the current large working tree is checkpointed and reproducibly baselined.

## 3. Phase 0 — Documentation Discovery and Baseline Freeze

### What to do

1. Create a checkpoint commit or dedicated worktree containing the current Phase 2 planning foundation.
2. Record:
   - `git status --short`;
   - `cargo test --lib` results;
   - `cargo test --tests` results;
   - `cargo build --release` results;
   - current benchmark coverage and wall time from `--per-problem-json`.
3. Generate `artifacts/phase2_baseline.json` containing the commit, Rust version, OS, test totals, benchmark totals, warnings, and environment variables that affect routing.
4. Resolve the architectural naming collision between:
   - `src/orchestrator.rs`, the real pathway-memory wrapper used by the CLI; and
   - `src/agent/orchestrator.rs`, the collaborative prototype that still simulates synthesis/review.
5. Treat `docs/phase2_agentic_architecture.md` as a design source, not compilable API documentation.

### Allowed APIs confirmed in the repository

- `agent::planning::TaskDecomposer::{decompose, integrate_solver}` — `src/agent/planning.rs`.
- `agent::hierarchy::GoalTree` — `src/agent/hierarchy.rs`.
- `agent::dependencies::DependencyResolver` — `src/agent/dependencies.rs`.
- `agent::executor::Executor::{with_executor, execute}` — `src/agent/executor.rs`.
- `solver::solve_problem` and current cascade — `src/solver/pipeline.rs:1-230`.
- `solved_cache::{lookup, record, flush, prune}` — `src/solved_cache.rs:141-666`.
- `learning::ExperienceDB::{record_experience, find_similar_problems, get_effective_actions}` — `src/learning/experience.rs:192-440`.
- `execution::Sandbox::{with_config, verify, execute}` — `src/execution/sandbox.rs:339-899`.
- `validation::ValidationPipeline` — `src/validation/pipeline.rs`.
- `db::sql` query builders and `db::pool` connection bookkeeping — `src/db/sql.rs`, `src/db/pool.rs`.
- `tensor::nas::{DARTSCell, ENAS, SearchSpace, NasOptimizer}` — `src/tensor/nas.rs`.
- `tensor::rl_core::{PolicyGradient, PPOClip, ValueFunction}` — `src/tensor/rl_core.rs`.

### Known unavailable or incomplete APIs

- `http::Client::send_request` is a stub that always returns an error (`src/http/client.rs:161-166`).
- Database pooling/query builders do not constitute a real database execution backend.
- `agent::orchestrator` generates simulated code/reviews and contains literal placeholder output.
- `TaskDecomposer::execute_task` simulates individual task execution; only `integrate_solver` reaches the real solver.
- The architecture document references `reqwest`, `sqlx`, `problem.hash()`, `Arc<Solver>`, and async trait methods that do not currently exist in those forms.

### Verification

- Baseline artifact can be regenerated from one command.
- Every uncommitted file is classified as intentional, generated, temporary, or removable.
- No later phase begins with unknown baseline failures.

### Anti-pattern guards

- Do not stack more work on the current dirty tree without a checkpoint.
- Do not add `reqwest`, `sqlx`, `dashmap`, or `anyhow` merely because the design document mentions them.
- Do not treat architecture pseudocode as copy-ready Rust.

---

## 4. Phase 1 — Canonical Agent Runtime and Shared Types

**Depends on:** Phase 0  
**Goal:** Make the existing planning, hierarchy, dependency, and executor modules compose without translation glue scattered across the codebase.

### What to implement

1. Add `src/agent/types.rs` with canonical types:
   - `AgentTaskId`;
   - `AgentRole`;
   - `AgentPriority`;
   - `AgentTaskStatus`;
   - `AgentError`;
   - `AgentEvent`;
   - `AgentRunId`.
2. Add `src/agent/config.rs`:
   - concurrency limits;
   - per-task timeout;
   - total plan budget;
   - retry policy;
   - tool and memory policy references;
   - deterministic seed.
3. Add `src/agent/plan.rs` as the canonical executable-plan representation.
4. Add explicit conversions from planning/hierarchy/dependency structures into `AgentPlan`.
5. Rename the collaborative prototype to avoid collision, for example `CollaborativeCoordinator`; retain `src/orchestrator.rs` as `PathwayOrchestrator` until migration is complete.
6. Add schema versions to serializable plans and events.

### Documentation references

- Planning types: `src/agent/planning.rs`.
- Executor plan/task model: `src/agent/executor.rs:16-230`.
- Existing agent messages and roles: `src/agent/orchestrator.rs:11-246`.
- Existing pathway memory: `src/orchestrator.rs:20-180`.

### Verification checklist

- Round-trip serialization tests for plans/events.
- Conversion tests prove dependency direction is preserved.
- Cycle detection remains green.
- Existing 98 agent tests remain green.
- No public type named only `TaskId` is re-exported ambiguously from multiple modules.

### Anti-pattern guards

- Do not maintain four unrelated task-ID representations indefinitely.
- Do not break stored pathway or cache formats without versioned migration.

---

## 5. Phase 2 — Secure Tool Framework Core

**Depends on:** Phase 1  
**Goal:** Give agents real, auditable capabilities without giving them unrestricted process access.

### What to implement

Create `src/agent/tools/`:

- `types.rs` — `ToolId`, `ToolCapability`, `ToolRequest`, `ToolResult`, `ToolError`, `SideEffect`, `ResourceUsage`;
- `trait.rs` — async `Tool` trait using the existing `async-trait` dependency;
- `registry.rs` — registration, capability lookup, typed dispatch, timeout enforcement;
- `policy.rs` — path roots, command allowlists, domain allowlists, read/write/network/process permissions;
- `audit.rs` — append-only JSONL execution records tied to run/task IDs;
- `filesystem.rs` — canonicalized read/list/write/mkdir with symlink-escape protection;
- `shell.rs` — executable-plus-argument API, never raw shell interpolation;
- `git.rs` — status/diff/log first; commit/branch only behind explicit mutation permission;
- `sandbox.rs` — adapter over `execution::Sandbox`.

Implement tools in two safety tiers:

1. **Tier A, read-only:** file read/list, Git status/diff/log, sandbox verification.
2. **Tier B, mutating:** file write/mkdir and allowlisted commands with audit records and rollback metadata.

### Documentation references

- Sandbox behavior: `src/execution/sandbox.rs:339-525`.
- Architecture intent: `docs/phase2_agentic_architecture.md:749-1190`.
- Existing rollback model: `src/agent/executor.rs`.

### Verification checklist

- Unit tests for every operation and denied operation.
- Property tests for `..`, absolute path, symlink, Unicode, and prefix-confusion escapes.
- Timeout/process cleanup tests.
- Concurrency tests for registry dispatch and audit logging.
- No test writes outside a temporary directory.
- Target: at least 30 focused tool-policy tests.

### Anti-pattern guards

- Never execute `sh -c` with model-generated text.
- Never authorize a path with string-prefix matching; canonicalize first.
- Never expose push, reset, force, credential, or remote mutation operations by default.
- Never report a tool success without exit status and captured output.

---

## 6. Phase 3 — Complete HTTP and Database Capability Boundaries

**Depends on:** Phase 2  
**Goal:** Avoid wrapping stubs and calling them production tools.

### What to implement

#### HTTP decision gate

Choose one documented backend and record the decision:

- finish the existing runtime socket/TLS transport; or
- add a maintained HTTP dependency behind an `agent-http` feature.

Then implement:

- URL parsing with explicit scheme/host/port;
- DNS/IP allow/deny rules, including loopback and link-local protection;
- redirect re-validation;
- response-size and timeout caps;
- GET/POST/PUT/DELETE;
- deterministic mock-server tests.

#### Database decision gate

Split database capability into:

1. `SqlBuildTool`, which safely produces SQL from existing typed builders and performs no I/O.
2. Optional `DatabaseExecuteTool`, behind a feature with a real backend and parameter binding.

Start read-only; schema/mutations require separate permissions and transactions.

### Documentation references

- Current HTTP stub: `src/http/client.rs:13-179`.
- Query builders: `src/db/sql.rs:7-536`.
- Pool bookkeeping: `src/db/pool.rs:11-291`.

### Verification checklist

- HTTP integration tests use only local deterministic servers.
- SSRF tests cover redirects, numeric IPs, IPv6, and DNS rebinding boundaries.
- SQL tests prove values are bound/escaped and identifiers validated.
- Feature-disabled builds remain green.

### Anti-pattern guards

- Do not wrap `http::Client` until `send_request` is real.
- Do not describe query-string generation as database execution.
- Do not permit arbitrary SQL strings in the default agent policy.

---

## 7. Phase 4 — Unified Persistent Agent Memory

**Depends on:** Phase 1; can run alongside Phases 2–3  
**Goal:** Present one memory API while reusing the three existing persistence systems.

### What to implement

Create `src/agent/memory/`:

- `mod.rs` — `AgentMemory` facade;
- `working.rs` — bounded current-run context and event summaries;
- `episodic.rs` — adapter over `learning::ExperienceDB`;
- `solution.rs` — adapter over `solved_cache`;
- `pathway.rs` — adapter/migration for `PathwayMemory`;
- `patterns.rs` — index over `ExperienceDB::lessons()` and measured effectiveness;
- `query.rs` — typed exact/similar/pattern/context requests;
- `consolidation.rs` — deduplication, confidence decay, pruning, schema migration, and atomic persistence;
- `stats.rs` — hits, latency, bytes, evictions, transfer outcomes.

Memory retrieval order:

1. exact verified solution;
2. effective method/pattern hints;
3. similar episodes/pathways;
4. empty response—not a fabricated memory.

Every retrieved program must be re-verified against the current problem before use.

### Documentation references

- `src/solved_cache.rs:141-666`.
- `src/learning/experience.rs:192-440`.
- `src/orchestrator.rs:20-132`.
- Architecture intent: `docs/phase2_agentic_architecture.md:1190-1648`.

### Verification checklist

- Persistence/reload/migration tests.
- Concurrent query/store tests.
- Corruption recovery and atomic-write tests.
- Re-verification tests reject stale or colliding entries.
- Benchmarks report p50/p95 lookup latency and warm-vs-cold solve time.
- Target: at least 25 memory tests.

### Anti-pattern guards

- Do not create a fourth unrelated successful-solution database.
- Do not hold async locks across disk I/O.
- Do not learn only from successes; preserve typed failure/timeout evidence.

---

## 8. Phase 5 — Real Specialized Agents and Supervision

**Depends on:** Phases 1, 2, and 4  
**Goal:** Replace the simulated collaborative prototype with agents that call real nCPU components.

### What to implement

Create `src/agent/actors/`:

- `trait.rs` — async `AgentActor` contract;
- `planner.rs` — emits an `AgentPlan` from `TaskDecomposer`;
- `synthesizer.rs` — calls the existing solver cascade and records actual results;
- `validator.rs` — performs strict runtime verification and optional AST validation;
- `optimizer.rs` — proposes changes but accepts only re-verified improvements;
- `learner.rs` — records complete run traces and outcomes;
- `supervisor.rs` — owns lifecycle, cancellation, retry, escalation, and health;
- `bus.rs` — bounded `mpsc` requests plus `oneshot` responses and correlation IDs;
- `scheduler.rs` — priority queue, dependency release, concurrency caps, and fairness.

Required execution flow:

`Planner → Scheduler → Synthesizer → Validator → optional Optimizer → Validator → Learner`

Failures flow to the supervisor and recovery policy, never directly into a success result.

### Documentation references

- Existing agent roles/messages: `src/agent/orchestrator.rs:11-320`.
- Existing executor: `src/agent/executor.rs`.
- Existing recovery policies: `src/solver/recovery.rs`.
- Strict runtime verification used by `src/orchestrator.rs`.

### Verification checklist

- End-to-end success, solver failure, validation failure, timeout, cancellation, retry, and shutdown tests.
- Supervisor detects and reports disconnected agents.
- Bounded channels demonstrate backpressure.
- No literal placeholder program appears in production paths.
- Target: at least 30 actor/coordination tests.

### Anti-pattern guards

- Remove `simulate_agent_synthesis` and `simulate_agent_review` from executable paths.
- Do not use unbounded channels for task payloads.
- Do not call blocking solver/disk work directly on Tokio worker threads; use controlled blocking tasks where needed.

---

## 9. Phase 6 — Agentic Solver, Pipeline, Library, and CLI Integration

**Depends on:** Phase 5  
**Goal:** Deliver a real opt-in product surface without regressing the legacy cascade.

### What to implement

1. Add `src/solver/agentic.rs`:
   - `AgenticSolver`;
   - `AgenticConfig`;
   - `AgenticSolveResult` with plan, trace, tool calls, memory hits, validation, and final `SolveResult`.
2. Legacy cascade remains the synthesizer's primary callable; agentic mode orchestrates around it.
3. Add library APIs:
   - `solve_problem_agentic`;
   - `plan_problem`;
   - `execute_agent_plan`.
4. Extend the current manual CLI parser—without introducing fictional `clap` structs—with:
   - `--agentic`;
   - `--plan-only`;
   - `--plan-file PATH`;
   - `--agent-trace PATH`;
   - `--agent-max-parallel N`;
   - `--agent-timeout SEC`;
   - `--tool-policy PATH`;
   - `--memory-root PATH` reuse.
5. Add machine-readable JSON output for plans and traces.
6. Preserve all existing `--problem`, `--problem-json`, `--orchestrate`, interactive, target-language, and legacy flags.

### Documentation references

- Current CLI parser/dispatch: `src/main.rs:21-900`.
- Current solver wrapper/cache placement: `src/solver/pipeline.rs:1-230`.
- Architecture intent: `docs/phase2_agentic_architecture.md:2130-2254`.

### Verification checklist

- CLI golden tests for every new flag and invalid combination.
- Legacy benchmark result/method parity with agentic mode disabled.
- Agentic end-to-end tests for scalar, array, string, stateful, and failure problems.
- `--plan-only` performs no tool mutation and no synthesis.
- Release binary and library builds succeed.

### Anti-pattern guards

- Do not replace the default solver cascade with agentic orchestration.
- Do not recursively call agentic mode from the synthesizer agent.
- Do not silently ignore invalid policy or plan files.

---

## 10. Phase 7 — Portfolio Instrumentation and Experiment Framework

**Depends on:** Phase 6  
**Goal:** Produce the trustworthy data required for curriculum learning and NAS.

### What to implement

1. Add a `PortfolioConfig` describing enabled solver families, ordering, budgets, teacher top-k, cache policy, and agentic settings.
2. Add CLI flags:
   - `--portfolio PATH|NAME`;
   - `--portfolio-report PATH`;
   - `--portfolio-ablation FAMILY`.
3. Instrument every attempted solver stage with:
   - applicability;
   - start/end time;
   - success/failure/timeout/skipped;
   - validation result;
   - cache/memory involvement;
   - resource estimates.
4. Add `experiments/` manifests with fixed seed, commit, benchmark split, configuration, environment, and expected artifact schema.
5. Extend the existing `--per-problem-json` output rather than inventing an incompatible benchmark format.
6. Generalize existing Pareto scripts (`tools/internal/diversity_pareto.sh`) to coverage/time/memory/config comparisons.
7. Add failure-aware JSONL capture and transfer-matrix aggregation.

### Documentation references

- Existing per-problem harness: `src/main.rs`.
- Existing Pareto sweep: `tools/internal/diversity_pareto.sh`.
- Existing transfer failures: `tools/internal/prioritize.sh`.
- Reproducibility contract: `REPRODUCIBILITY.md` and `artifacts/BENCHMARKS_README.md`.

### Verification checklist

- Schema tests and deterministic replay from a manifest.
- Method counts sum exactly to solved totals.
- Failure rows include stage and concrete reason.
- Pareto frontier recomputes from raw committed JSON/CSV.
- Baseline portfolio remains reproducible.

### Anti-pattern guards

- Do not claim improvement from one seed or the training benchmark.
- Do not overwrite canonical paper artifacts during exploratory runs.
- Do not aggregate failures into an untyped string bucket.

---

## 11. Phase 8 — Curriculum Learning and Prioritized Replay

**Depends on:** Phase 7  
**Goal:** Learn a training/solve progression from measured difficulty and transfer—not hand-authored labels.

### What to implement

Create `src/learning/curriculum/`:

- `features.rs` — problem size, input types, argument count, statefulness, teacher applicability, historical solve time, failures, and route entropy;
- `difficulty.rs` — calibrated easy/medium/hard score with confidence;
- `buckets.rs` — adaptive quantile boundaries with minimum bucket sizes;
- `replay.rs` — prioritized replay with diversity and recency controls;
- `sequencer.rs` — five-stage progression: exact/easy → compositional → stateful → transfer-hard → frontier failures;
- `evaluation.rs` — forgetting, transfer gain, and hard-domain improvement metrics.

Use `ExperienceDB` records and Phase 7 traces as inputs. Keep this distinct from the LinguaGenesis linguistic curriculum, although linguistic tasks can be one evaluation domain.

### Documentation references

- Existing learning database: `src/learning/experience.rs`.
- Existing router/strategy outcomes: `src/method_router.rs`, `src/strategy.rs`.
- Curriculum roadmap: `MASTER_ROADMAP.md:394-480`.
- Existing LinguaGenesis curriculum paper: `paper/sections/section_curriculum_rule_learning.md`.

### Verification checklist

- Difficulty monotonicity and calibration tests.
- Stable adaptive-boundary tests on sparse/skewed data.
- Replay diversity and catastrophic-forgetting tests.
- Five-stage sequencing tests.
- At least three seeds on train/validation/sealed evaluation splits.
- Report hard-domain delta, total coverage delta, time delta, and forgetting delta.
- Target: at least 25 curriculum tests.

### Anti-pattern guards

- Do not use benchmark names as difficulty labels.
- Do not tune boundaries on sealed evaluation results.
- Do not report only hard-task improvement if easy-task forgetting offsets it.

---

## 12. Phase 9 — NAS for Synthesis Portfolios

**Depends on:** Phases 7 and 8  
**Goal:** Search solver portfolios, not generic neural layer graphs.

### What to implement

Create `src/learning/portfolio_nas/`:

- `encoding.rs` — versioned fixed-dimensional encoding for stage enablement, ordering groups, time budgets, top-k, cache policy, curriculum policy, and agent limits;
- `search_space.rs` — validity constraints and mutation/crossover operations;
- `fitness.rs` — multi-objective coverage, p50/p95 time, memory, tool cost, and robustness;
- `surrogate.rs` — predicts configuration outcomes from Phase 7 traces;
- `controller.rs` — policy/value controller using existing tensor RL primitives only after offline data is sufficient;
- `baseline.rs` — random, greedy, and evolutionary baselines;
- `runner.rs` — deterministic train/validation evaluation and resumable checkpoints;
- `pareto.rs` — nondominated frontier and deployment selection policy.

Milestone order:

1. Validate the encoding/fitness with random and greedy search.
2. Establish evolutionary baseline.
3. Train RL controller against the same budget.
4. Compare sample efficiency and final Pareto frontier.

The existing `tensor::nas` module may supply tensor/search primitives, but its layer-oriented `SearchSpace` must not be presented as a synthesis-portfolio controller.

### Documentation references

- Tensor NAS primitives: `src/tensor/nas.rs`.
- RL primitives: `src/tensor/rl_core.rs`.
- Existing Pareto methodology: `artifacts/diversity_pareto.*`.
- Self-improvement roadmap: `MASTER_ROADMAP.md:394-480`.

### Verification checklist

- Encoding round-trip and invalid-configuration property tests.
- Deterministic sampling by seed.
- Fitness agrees with raw benchmark records.
- Random/evolution/RL comparisons use equal evaluation budgets.
- Winning configuration is re-run from scratch on sealed problems.
- Target: at least 20 NAS tests plus reproducible experiment artifacts.

### Anti-pattern guards

- Do not call generic DARTS/ENAS layer search “portfolio NAS.”
- Do not reward coverage alone; it will select arbitrarily slow portfolios.
- Do not train and evaluate the controller on identical problems.

---

## 13. Phase 10 — Phase 3 World Model and Analogy Layer

**Depends on:** Stable Phase 7–9 data  
**Goal:** Use structured knowledge and analogical transfer after exact memory and learned routing are reliable.

### What to implement

1. Extend `src/knowledge/` with code-specific entities and relations rather than replacing `api_graph.rs`.
2. Add schema-versioned relations: `Uses`, `Implements`, `Transforms`, `SolvedBy`, `FailsOn`, `SimilarTo`, `TransfersTo`.
3. Import selected LinguaGenesis entities through an explicit bridge and provenance records.
4. Add `src/solver/analogy.rs`:
   - retrieve structurally similar solved problems;
   - map signatures/features;
   - adapt candidate programs/patterns;
   - assign confidence;
   - strictly verify transfers.
5. Add `src/reasoning/causal.rs` only after failure/intervention data exists:
   - portfolio configuration as intervention;
   - solve outcome/time as effects;
   - counterfactual configuration queries.

### Verification checklist

- Graph import idempotency and provenance tests.
- Analogy transfers never bypass verification.
- Compare analogy retrieval against lexical and ExperienceDB baselines.
- Causal claims require intervention data, not correlations alone.
- Sealed-domain transfer report with confidence calibration.

### Anti-pattern guards

- Do not embed all source code into one unversioned graph blob.
- Do not call nearest-neighbor similarity causal reasoning.
- Do not let Linguigenesis become a mandatory dependency for core synthesis builds.

---

## 14. Phase 11 — Phase 4 Meta-Learning and Guarded Self-Improvement

**Depends on:** Phases 8–10  
**Goal:** Learn routing, curriculum, and tool policies while preventing unverified self-modification.

### What to implement

- `src/learning/replay.rs` — shared prioritized replay facade;
- `src/meta/policy.rs` — selects portfolio/curriculum/tool policy;
- `src/meta/transfer.rs` — cross-domain lesson applicability and confidence;
- `src/meta/improvement.rs` — proposes configuration changes;
- `src/meta/gate.rs` — accepts changes only after validation, sealed evaluation, and rollback checkpoint;
- `src/meta/report.rs` — before/after statistics and provenance.

Improvement loop:

`Measure → diagnose bottleneck → propose one bounded change → evaluate → compare → accept/reject → record`

Code changes remain human-reviewed. Initial self-improvement is configuration/policy adaptation, not autonomous source rewriting.

### Verification checklist

- Replay/curriculum/NAS policies can be frozen and replayed.
- Every accepted change has baseline, candidate, sealed evaluation, and rollback metadata.
- Negative transfer is detected and rejected.
- Longitudinal benchmark demonstrates improvement without catastrophic forgetting.

### Anti-pattern guards

- No self-editing source code in the first production version.
- No acceptance based on training-set score.
- No unbounded recursive agent-improvement loop.

---

## 15. Phase 12 — Paper Infrastructure and Publication Packages

**Depends on:** Phase 7 onward; evolves with later phases  
**Goal:** Make every claim reproducible from raw records.

### What to implement

1. `experiments/schema.json` — manifest and output schemas.
2. `experiments/run.py` or Rust runner — executes a manifest without hidden defaults.
3. `experiments/aggregate.py` — confidence intervals, paired comparisons, Pareto fronts, transfer matrices, and failure taxonomy.
4. `experiments/check_artifact.py` — validates totals, provenance, hashes, and missing runs.
5. `paper/tables/` and `paper/figures/` generated only from validated artifacts.
6. Three claim packages:
   - **Agentic synthesis:** planning/tool/memory ablations, success, latency, recovery, safety denials.
   - **Curriculum portfolios:** hard-task transfer, total coverage, forgetting, and cross-domain matrix.
   - **Portfolio NAS:** random/evolution/RL sample efficiency and sealed Pareto frontier.
7. Update `REPRODUCIBILITY.md` with one command per claim.

### Required experimental design

- fixed pre-registered hypotheses;
- at least three seeds for learning claims;
- paired problems/configurations where possible;
- confidence intervals and effect sizes;
- sealed third-party/OOD benchmark;
- complete failure accounting;
- hardware/software/commit metadata;
- ablations for planning, tools, memory, curriculum, and NAS separately.

### Documentation references

- Existing paper build: `paper/build_pdf.sh`.
- Existing paper index: `paper/ncpu_paper.md`.
- Existing portfolio section: `paper/sections/section_solver_portfolio.md`.
- Existing reproducibility contract: `REPRODUCIBILITY.md`.

### Verification checklist

- Delete generated tables/figures and rebuild them from raw data.
- Counts in paper equal counts in artifacts.
- PDF build succeeds.
- Artifact checker rejects incomplete/edited data.
- Each claim lists limitations and negative results.

### Anti-pattern guards

- Do not hand-edit generated paper tables.
- Do not call three implementation tracks “three papers” until independent hypotheses and results exist.
- Do not reuse exploratory benchmark results as sealed evaluation.

---

## 16. Phase 13 — Documentation, CI, Performance, and Release Gate

**Depends on:** All intended release phases  
**Goal:** Turn the work into a maintainable product rather than a research branch.

### What to implement

- API docs for agent runtime, tools, memory, curriculum, and NAS;
- CLI guide with safe-policy examples;
- updated architecture/data-flow diagrams;
- migration guide from `--orchestrate` to `--agentic`;
- threat model for tool execution;
- artifact/reproducibility guide;
- CI jobs for formatting, library tests, integration tests, policy tests, release build, docs, and small deterministic benchmark;
- nightly jobs for full benchmark, learning, NAS, and artifact validation.

### Final verification matrix

1. `cargo fmt --check`.
2. `cargo clippy --all-targets` with an agreed warning baseline.
3. `cargo test --lib`.
4. `cargo test --tests`.
5. Security-policy and sandbox tests.
6. Legacy benchmark parity.
7. Agentic benchmark and failure recovery.
8. Curriculum multi-seed evaluation.
9. NAS sealed evaluation.
10. `cargo build --release`.
11. Rustdoc/documentation build.
12. Paper PDF and artifact checker.

### Release acceptance criteria

- zero test failures;
- zero known hangs/deadlocks;
- no tool escape from allowed roots/commands/domains;
- no simulated solution in production execution;
- legacy behavior unchanged by default;
- agentic traces reproduce each final decision;
- paper claims regenerate from committed raw artifacts.

---

## 17. Dependency Graph and Recommended Execution Order

```text
Phase 0 Baseline
    ↓
Phase 1 Shared Runtime Types
    ├──→ Phase 2 Tool Core ──→ Phase 3 HTTP/DB
    └──→ Phase 4 Unified Memory
              \               /
               → Phase 5 Real Agents
                         ↓
                 Phase 6 Agentic Integration
                         ↓
                 Phase 7 Instrumentation
                    ┌────┴────┐
                    ↓         ↓
             Phase 8       Phase 12 begins
             Curriculum    Paper infrastructure
                    ↓
             Phase 9 Portfolio NAS
                    ↓
             Phase 10 World Model
                    ↓
             Phase 11 Meta-Learning
                    ↓
             Phase 13 Release Gate
```

## 18. Session-Sized Execution Packages

Each package is designed to fit one focused implementation session and end green:

1. Baseline/checkpoint and naming decision.
2. Canonical types and conversion tests.
3. Tool types/registry/policy.
4. Filesystem and Git read-only tools.
5. Shell and sandbox adapters.
6. HTTP transport decision/implementation.
7. Database capability boundary.
8. Memory facade and solution-cache adapter.
9. Experience/pathway/pattern adapters.
10. Consolidation and memory benchmarks.
11. Agent trait/bus/scheduler.
12. Real planner/synthesizer/validator.
13. Optimizer/learner/supervisor and failure recovery.
14. Agentic solver/library API.
15. CLI and trace output.
16. Portfolio schema/instrumentation.
17. Benchmark/Pareto/failure matrices.
18. Curriculum features/difficulty/buckets.
19. Replay/sequencer/evaluation.
20. NAS encoding/fitness/baselines.
21. NAS controller and sealed evaluation.
22. Knowledge graph/analogy.
23. Causal and meta-learning gates.
24. Publication automation and final release validation.

## 19. Effort and Test Budget

These are engineering ranges, not promises:

| Track | Sessions | New focused tests |
|---|---:|---:|
| Baseline + canonical runtime | 2 | 15–25 |
| Tool framework + transports | 5 | 45–65 |
| Unified memory | 3 | 25–40 |
| Real agents + integration | 5 | 40–60 |
| Portfolio instrumentation | 2 | 15–25 |
| Curriculum learning | 2 | 25–35 |
| Portfolio NAS | 3 | 20–30 |
| World model + meta-learning | 3–5 | 30–50 |
| Paper/CI/release | 2–3 | artifact validators + integration gates |

Expected total: roughly **24 focused sessions** and **215–330 new focused tests**, while preserving the existing suite.

## 20. First Concrete Implementation Slice

The next execution session should perform only the following:

1. checkpoint and baseline the working tree;
2. write the naming/type decision record;
3. add canonical agent task/event/config types;
4. add conversion tests for planning, hierarchy, dependencies, and executor;
5. run the full library/integration tests and release build;
6. update this plan with measured baseline counts.

This slice removes the largest architectural risk—the incompatible duplicate task and orchestrator models—before any new tool, memory, or learning code is added.
