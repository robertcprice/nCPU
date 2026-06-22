# nCPU — agent entry point (READ THIS FIRST)

**Single source of truth: [`MASTER_ROADMAP.md`](MASTER_ROADMAP.md).** Read its **§0.01 (operating rules)** and **§0.05 (current synthesis state + plan)** before doing anything. Do not create new plan/roadmap `.md` files — edit MASTER_ROADMAP.md.

## Hard rules (the cause of every wasted-time failure if ignored)
1. **Canonical tree = `nCPU/nsynth`** (pkg `mog_synth`). It is the ONLY tree you edit/build/audit.
   **`nCPU/ncpu-learned-parser/` is a STALE, gitignored, SEPARATE git repo** (Jun-19 fork, 244 vs 322 `.rs` files, 54 unmerged files pending salvage). **Do NOT read, edit, build, or cite it as current state.** Ignore any older note calling it the "active crate."
2. **Read-manifest-first:** before auditing/changing, enumerate ALL files (`find nsynth/src -name '*.rs' | wc -l`) and cover every one. No discovery-based scoping → that is why files get skipped.
3. **Verify, don't assume:** cite `file:line`; read code not names; "works" needs a test you ran.
4. **No duplicate work:** grep for existing impl + check MASTER_ROADMAP §0.05 + memory before building. Most things exist partially.
5. **Isolate writes** in a git worktree (`git rev-parse --show-toplevel` must be your worktree); never edit a file another agent is touching.

Note: root `AGENTS.md` is an auto-generated memory dump (claude-mem hook), NOT the operating rules — these are.
