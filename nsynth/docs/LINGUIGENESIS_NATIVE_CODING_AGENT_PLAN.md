# Linguigenesis-Native Coding Agent: Definitive Implementation Plan

**Status:** architecture companion to the execution authority
**Scope:** `nCPU/nsynth` plus the required upstream work in `../linguigenesis`
**Execution language:** Rust
**Core intelligence:** Linguigenesis; external LLMs are optional, untrusted proposal sources only
**Execution authority:** root `MASTER_ROADMAP.md` owns package state, ordering, and completion gates.
**Companion:** `PHASE_2_4_EXECUTION_PLAN.md` remains useful for lower-level agent/runtime work.

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

**Emergent NL rule:** operation examples, workflow signals, and constraints live in `linguigenesis/data/coding_registry.json`. Rust parses and walks relations; it must not use `text.contains` or hard-coded examples in `linguigenesis_bridge.rs`. See root `MASTER_ROADMAP.md` §0.5 Package C work card.

An external model adapter may propose an intent, plan, or patch only when explicitly enabled. Every proposal must be converted into native typed structures and pass the same policy, synthesis, and verification gates. The system must remain useful with all external-model features disabled.

### 2.3 Rust-only orchestration rule

Production orchestration, tool execution, persistence, benchmark running, and reporting remain Rust. Python may not become a hidden agent runtime. If a benchmark distributes a Python harness, invoke it as an isolated external oracle and record that fact.

Synthesis receives an oracle-free task view: evaluator holdouts and reference implementations may not enter generation, routing, ranking, acceptance, repair, memory, or credit. The current `Problem::synthesis_view()` boundary is defense in depth; the benchmark-ready harness must place sealed evaluation data in a separate process-owned type. External fallbacks must be reported separately and never counted as native nsynth solves.

## 3. Current-State Verdict (2026-06-20)

| Area | Current reality | Readiness |
|---|---|---:|
| Function synthesis | Large, real solver portfolio with many tests | ~70% |
| NLP-to-code | Linguigenesis bridge exists, but coding intent/spec grounding is shallow and the legacy NL path contains `NotImplemented` | ~30% |
| Planning | Typed decomposition/dependencies exist; some execution paths still simulate completion | ~45% |
| Tool use | Rust tool modules exist; policy, isolation, timeouts, and real boundary semantics are incomplete | ~30% |
| Repository understanding | Hardness scanner and knowledge structures exist; no canonical repository index/context engine | ~10–15% |
| Repo repair loop | New typed scaffold exists, but no executing propose/apply/test/revise loop | ~10% |
| Memory | Several local representations exist; no single durable, evidence-gated task memory | ~30% |
| Security | Initial allow/deny checks exist; canonical-path, symlink, process, network, and secret defenses are incomplete | ~20% |
| Benchmarks | 20 metadata task descriptions exist; no executable local repo-agent benchmark harness | ~10% |
| Product UX | No canonical coding-agent CLI/session protocol | ~10% |

Overall: the architecture is substantially mapped, but only about one quarter of an end-to-end coding agent is operational. The new `agent::repo` work is a useful seam, not a finished agent.

### Point-in-time baseline blockers

- The reviewed 47-module `lib.rs` union is merged; debug and release library checks pass.
- The focused repo-agent suite passes 16/16, and the search-only orchestrator batch solves all 140 tasks after evaluator-oracle stripping; an oracle-poisoning invariance regression verifies the generated method/code cannot change when holdouts or reference code are altered.
- The older planning/collaboration layer now fails closed without a bound executor, records only the real solver node as completed, skips decorative nodes, generates proposals with the native Rust search solver, and performs executable/static reviews instead of role-played approval.
- The serial 2,112-test library run remains too slow and red: a partial run observed 1,211 passes, 61 failures, and one ignored test before manual termination inside an unbounded full-benchmark test.
- Formatting reports 473 hunks across 34 files.
- Production-reachable simulation/placeholder paths remain in the legacy agent orchestrator, planner, NL frontend, project transpilation, and advertised library modules.
- Exact evidence and cluster ownership live in `docs/PACKAGE_A_BASELINE.md`.

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

- Coding ontology in `data/coding_registry.json` (not Rust keyword tables).
- `RequirementDeriver` + `CodingComprehension` → `SynthesisRequirement`.
- Registry workflow entities for intent (`signal_lemmas`, `maps_to_intent`).
- Compositional requests, negation, scope, dialogue ambiguity via `unresolved`.
- Coding utterance corpora with paraphrase and adversarial cases.

**Verification:** paraphrase via synonym → identical `example_cases`; zero `text.contains` in comprehension production path.

**Do not** hard-code examples in Rust or use keyword-only classification.

### Phase 3 — Typed Linguigenesis-to-nsynth Bridge

**Build**

- `linguigenesis_bridge.rs` calls `CodingComprehension` only; delete `generate_examples_from_text`.
- `SynthesisRequirement` → `Problem` / `CodingIntent` → `CodeTaskSpec`.
- Quarantine `nl/mod.rs` from production.

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
| C | 2 | coding ontology and native comprehension upstream |
| D | 3 | grounded bridge and clarification |
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
