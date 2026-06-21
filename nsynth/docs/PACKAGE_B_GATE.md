# Package B Gate — Runtime Contracts (G1 prerequisite)

**Status:** ✅ DONE (2026-06-21)
**Authority:** `MASTER_ROADMAP.md` Phase 1 / Section 5

## Gate evidence

| Contract | Location | Conformance test |
|---|---|---|
| `CodingIntent` | `agent/coding_intent.rs` | `coding_intent_from_add_nl` |
| `CodeTaskSpec` | `agent/runtime/code_task_spec.rs` | `code_task_spec_roundtrip_json` |
| `AgentRun` state machine | `agent/runtime/state_machine.rs` | `legal_nl_synthesis_path`, `package_b_state_machine_has_terminal_guard` |
| `AgentRun` lifecycle | `agent/agent_run.rs` | `agent_run_comprehend_and_synthesize_add` |
| Budget accounting | `agent/runtime/budget.rs` | `agent_run_budget_blocks_extra_synthesis` |
| Cancellation | `agent/agent_run.rs` | `agent_run_cancel_from_understanding` |
| JSON snapshot v1 | `agent/agent_run_persistence.rs` | `agent_run_save_load_roundtrip`, `legacy_snapshot_migration` |
| Capability registry | `agent/capability_registry.rs` | `implemented_capabilities_link_conformance_tests` |
| `RepoRunSupervisor` | `agent/repo/run_supervisor.rs` | `supervisor_executes_nl_add_task` |
| `RepoWorkflowRunner` | `agent/repo/workflow_runner.rs` | `workflow_runner_executes_nl_fixture_suite` |
| Gate suite | `agent/package_b_gate.rs` | `cargo test package_b --lib` |

## Verify

```bash
cd nsynth
cargo test package_b --lib
cargo test agent_run --lib
cargo test capability_registry --lib
cargo test run_supervisor --lib
cargo test workflow_runner --lib
cargo test synthesis_proposer --lib
```

## Schema

- `SCHEMA_VERSION = 1` (`agent/runtime/state_machine.rs`)
- Snapshots: `schema_version`, `run_id`, `status`, `budget`, `events`, synthesis fields
- Legacy `phase` snapshots migrate via `legacy_phase` when `schema_version == 0`

## Out of scope (deferred)

- Full `Created→Planning→Verifying→Revising` on every NL path (repair loop uses proposer; verifying/revising wired in state machine for Package H)
