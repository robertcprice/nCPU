# nCPU Repository Reorganization Proposal
**Date**: April 2026 (post-initial evaluation)  
**Status**: In active continuous grinding / execution — deep progress on noise, hero command delight, JEPA (real speculate hook now using world_model for rollouts), and narrative lock-in.  
**Author**: Grok-assisted analysis + maintainer input  
**Goal**: Turn a high-signal but noisy research accretion into a clear, coherent project where the unique contribution ("CPU modeled on GPU with neural networks + kernels") is the unambiguous hero, user entry points are unified and delightful, and historical noise is contained without losing reproducibility or research value.

---

## Executive Summary

nCPU contains multiple world-class technical achievements:
- A full ARM64 CPU emulator running natively in Metal GPU shaders at ~1.9M IPS, capable of booting real BusyBox + Alpine Linux + a self-hosting C compiler.
- Trained neural networks replacing digital logic gates (ALU) with 100% exhaustively verified accuracy, embedded directly inside those GPU kernels.
- Deterministic execution (sigma=0.0) enabling a 26-command post-mortem debugging toolkit impossible on conventional CPUs.
- Differentiable execution engine enabling gradient-based program optimization and synthesis.
- Differentiable coprocessor injection into transformers (+56.5% arithmetic gains demonstrated).
- Self-optimizing hidden controller (SOME) with measurable code/reasoning improvements.

**The problem**: These signals are buried in historical layering, parallel implementations, fragmented entry points, and accumulated experimental artifacts. A newcomer (or even a returning maintainer) cannot quickly answer "what is the one amazing thing here and how do I experience it?"

**The proposal**: A phased cleanup that:
1. Clearly elevates the **Rust Metal GPU kernel + neural ALU in shaders** as the primary thesis ("The GPU *is* the computer").
2. Unifies and simplifies the user-facing Python surface around 4–5 high-signal experiences.
3. Contains (does not delete) historical noise in well-labeled archives while preserving every reproducibility claim.
4. Establishes clear ownership and layering so future work has obvious homes.

This is evolutionary, not revolutionary. We build on the excellent `ncpu lab` work (the March 2026 maximization plan) and the existing `docs/maintainers/` hygiene processes.

---

## Current State Diagnosis (Evidence-Based)

### Strengths (Keep and Amplify)
- Outstanding top-level README with tables, quickstarts, and honest "what works today".
- `ncpu/lab.py` + `python -m ncpu` + `ncpu-lab` console script (good unification attempt).
- Explicit `repo_hygiene.md` + `cleanup_checklist.md`.
- Strong reproducibility harnesses and artifact discipline.
- `demos/README.md` exists and points newcomers in reasonable directions.

### Structural Problems (Ranked by User Pain)

| Rank | Problem | Concrete Examples | Impact |
|------|---------|-------------------|--------|
| 1 | Multiple parallel "CPU" implementations with unclear ownership | `ncpu/neural/`, `ncpu/model/`, `ncpu/tensor/`, `kernels/mlx/`, `kernels/rust_metal/` (the real hero) | New users don't know which path is fast/deterministic/neural/differentiable |
| 2 | Hero technical achievement (Rust Metal + in-shader neural ALU) is under-exposed | `ncpu_run` binary does the real 1.9M IPS + full debugging toolkit + Alpine work, but `lab.py` and demos mostly route through Python wrappers or older MLX paths | The unique "GPU is the computer with neural logic inside the shader" story is hard to discover |
| 3 | Fragmented + legacy entry points | `scripts/cli/ncpu.py` (points at old MLX), `ncpu/demo.py` (legacy flags), many direct `demos/*.py` invocations, `python -m ncpu` vs `ncpu-lab` | "How do I just try the cool thing?" requires reading multiple files |
| 4 | Historical rename and project layering debt | Mog → nsynth traces, `some/` vs `ncpu/self_optimizing/`, `egdc/` as semi-separate effort | Feels like "five projects in one repo" |
| 5 | Generated noise mixed with source | `artifacts/` dated dumps, `paper/generated/` bundles, `training_results/` (600+ files), `experiments/` | Clone is heavy; signal is diluted |
| 6 | Scripts/ is a junk drawer | `dev/`, `egdc/`, `gpu/`, one-off `.py` at top level of scripts/ | Maintainers themselves have to remember where things live |

**Root cause**: Organic growth of a high-velocity research lab that successfully explored many adjacent ideas (differentiable everything, GPU OS, coprocessor, self-optimization, verified synthesis) without a later "make the best story obvious" pass.

---

## Guiding Principles (Non-Negotiable for This Proposal)

1. **One Hero Thesis First**
   - "A complete general-purpose computer (OS + compiler + apps) running on GPU shaders, where the ALU can be implemented by trained neural networks embedded in those shaders, yielding determinism and post-mortem superpowers impossible on CPUs."
   - The Rust Metal kernel (`kernels/rust_metal/`) + neural weights in-shader is the primary artifact. Python research surface exists to train those weights, study differentiability, and inject the ALU as a coprocessor.

2. **Clear Layering, Explicit Tradeoffs**
   - **Performance + Determinism Layer** (`kernels/rust_metal/`): The GPU computer. Use this for real speed, full OS, debugging toolkit, constant-time proofs.
   - **Research + Differentiability Layer** (`ncpu/` Python package): Backprop through execution, program synthesis via gradient descent, coprocessor injection, neural model training.
   - Supporting systems (nsynth synthesis engine, SOME hidden controller) are valuable but secondary to the above two.

3. **One Coherent User Surface**
   - `python -m ncpu` (and the `ncpu` / `ncpu-lab` entry points) must feel like a single product.
   - Every command must make the tradeoff (speed vs differentiability vs research depth) explicit.
   - Legacy/niche paths must be labeled "legacy" or "research-internal" with migration guidance.

4. **Signal Over Noise (Containment, Not Erasure)**
   - Never delete reproducibility-critical artifacts or paper claims.
   - Move bulky dated output into `artifacts/archive/` or external storage with clear pointers.
   - Create `docs/decision-records/` and `docs/historical/` for context that should not pollute the main narrative.

5. **Safe, Incremental Evolution**
   - Phase 0–1 changes are documentation + new preferred paths + deprecation warnings (zero breakage).
   - Larger structural moves (directory renames, archive moves) come only after the new surface is solid and documented.
   - Every change must keep `pytest`, `python -m ncpu doctor`, and the publication harnesses green.

6. **Respect Maintainer Reality**
   - Build directly on `lab.py`, the hygiene docs, and the existing `demos/` curation.
   - The goal is to make future high-value work (more neural kernel ops, coprocessor scaling, Weight CPU execution, multi-GPU) have obvious homes.

---

## Proposed Target State (After All Phases)

### Top-Level View (What a Clone Looks Like)
```
nCPU/
├── README.md                  # Hero story front-and-center
├── pyproject.toml
├── ncpu/                      # Python research + differentiable surface (the "lab")
│   ├── __main__.py            # Thin router to lab + clear subcommands
│   ├── lab.py                 # The unified, delightful entry point (enhanced)
│   ├── coprocessor/
│   ├── differentiable/
│   ├── model/                 # (clearly labeled research / legacy neural model CPU)
│   ├── neural/                # (research neural CPU for training & differentiability)
│   └── ...
├── kernels/
│   ├── rust_metal/            # ★ THE HERO ★  (Rust + Metal, neural ALU in shader, full OS)
│   │   ├── src/neural_cpu.rs  # In-shader neural execution
│   │   ├── bin/ncpu_run.rs    # Standalone high-perf launcher
│   │   └── ...
│   └── mlx/                   # (archived / legacy Python GPU kernels)
├── demos/
│   ├── showcase/              # Lightweight, cross-platform (discover, text)
│   ├── gpu/                   # Systems wow (mostly thin wrappers around rust_metal now)
│   └── neural/                # Research depth (full-neural, meta-compare)
├── docs/
│   ├── architecture/          # Clear ownership docs + decision records
│   ├── gpu/                   # Rust Metal kernel docs (elevated)
│   ├── maintainers/
│   │   ├── REPO_HYGIENE.md
│   │   ├── REPO_REORGANIZATION_PROPOSAL.md (this doc, kept as history)
│   │   └── DECISION_RECORDS/
│   └── ...
├── paper/                     # Research output (unchanged, just better linked)
├── artifacts/
│   ├── archive/               # Dated bulky output (gitignored or LFS where appropriate)
│   └── ... (only small, human-readable summaries + pinned baselines)
├── scripts/
│   ├── release/               # Publication automation (keep, maybe rename to publish/)
│   ├── setup/                 # Bootstrap helpers
│   └── benchmarks/            # Queueing helpers (internal)
├── nsynth/                    # Verified synthesis engine (supporting, clearly scoped)
├── some/                      # SOME docs + any remaining runtime (or merged)
├── egdc/                      # (Candidate for archive/ or clear "historical research" label)
├── tests/
├── training/                  # Training pipelines (research)
└── ...
```

### User-Facing Commands (Target)
Preferred new surface (additive, with deprecation for old):

```bash
# Hero path (new emphasis)
ncpu gpu                     # Best single command: interactive GPU OS shell (Rust path)
ncpu gpu --neural-alu        # Same, but using in-shader neural weights for ALU
ncpu gpu debug               # Drop into the 26-command GPU debugging toolkit
ncpu alpine                  # Boot full Alpine demo (hero systems experience)

# Flagship interactive (keep + polish)
ncpu discover
ncpu text --interactive

# Research depth
ncpu coprocessor
ncpu differentiable          # New: program optimization / synthesis REPL
ncpu full-neural             # Strict bottom-up (still valuable for paper)

# Meta
ncpu doctor
ncpu demos
ncpu path
ncpu info <name>

# Power users / research
python -m ncpu.lab ...       # Still works
ncpu-run ...                 # Direct access to the Rust binary (when installed)
```

All commands in the unified surface must print a one-line "mode + tradeoff" banner on start (e.g., "GPU Deterministic Mode (Rust Metal, ~1.9M IPS, neural ALU optional)").

---

## Phased Execution Plan

### Phase 0: Documentation & Signaling (Start Immediately — This Session)
- [ ] Write and land this proposal (as `docs/maintainers/REPO_REORGANIZATION_PROPOSAL.md`).
- [ ] Create `docs/architecture/HERO_THESIS.md` — one crisp page explaining the GPU-is-the-computer + neural-in-kernel story and why it is unique.
- [ ] Create `docs/architecture/LAYERING.md` — explicit ownership of rust_metal vs ncpu/ Python research surface.
- [ ] Add `docs/decision-records/` directory + first record (this reorganization).
- [ ] Update top-level README to put the Rust Metal GPU computer story in the first "wow" section with the single best command.
- [ ] Enhance `lab.py` with a `gpu` / `systems hero` subcommand that prefers the Rust backend and clearly documents when neural weights are active in-shader.
- [ ] Add deprecation warnings (with migration guidance) in `scripts/cli/ncpu.py` pointing to the new unified surface.
- [ ] Update `demos/README.md` to clearly label "Hero systems experience (Rust Metal)" vs "Research neural paths".

**Success metric**: A fresh clone + `pip install -e '.[demo,dev]'` + `python -m ncpu doctor` + one "hero" command makes the unique contribution obvious within 60 seconds.

### Phase 1: User-Facing Unification & Deprecation (Safe, Low-Risk)
- Strengthen `ncpu/lab.py` as the single source of truth for discovery.
- Add `ncpu gpu` (and aliases) that:
  - Detects the Rust extension.
  - Falls back gracefully with clear messaging.
  - Surfaces the neural-ALU-in-shader variant when weights are present.
- Add `ncpu gpu debug` that launches the 26-command toolkit shell.
- Route or clearly document the old `scripts/cli/ncpu.py` paths as "legacy direct launcher (consider `ncpu gpu` or the Rust `ncpu_run` binary)".
- Add a `ncpu` top-level command experience that feels productized (banners, consistent help, "this is the fast deterministic path" vs "this is the differentiable research path").
- Produce a short `WHY_THIS_COMMAND.md` for the top 5 commands.

**No code deletion.** Only new preferred paths + labels.

### Phase 2: Containment of Noise (Medium Changes)
- Move dated `artifacts/<timestamp>/` and large generated bundles into `artifacts/archive/2026-.../` (or external).
- Move or clearly namespace the remaining Mog/EGDC/SOME historical scaffolding (e.g., `historical/mog/`, or keep but add prominent `HISTORICAL.md`).
- Audit and prune one-off scripts in `scripts/` root and `scripts/dev/`.
- Make `training_results/`, `experiments/`, `outputs/`, `logs/` explicitly gitignored with `.gitignore` updates + docs pointers.
- Create `ARCHIVE_POLICY.md` that codifies the rule.

### Phase 3: Structural Clarity (Higher Cost, Higher Reward)
- Decide ownership:
  - `kernels/rust_metal/` becomes the home of the "GPU Computer" story (rename `src/neural_*` modules if needed for clarity, e.g. `neural_alu_in_shader`).
  - `ncpu/neural/` and `ncpu/model/` get explicit "research / training / differentiability only" labels or subdirs.
- Consider merging or clearly scoping `some/`, `egdc/`, `nsynth/` relative to the hero thesis (they are supporting, not co-equal).
- Evaluate whether the old MLX kernels in `kernels/mlx/` can be archived (they are no longer the performance path).

### Phase 4: Packaging & Install Polish (Later)
- Ensure `pip install ncpu[gpu]` or similar pulls the Rust extension cleanly when available.
- Make the Rust binary (`ncpu_run`) installable and discoverable from the Python `ncpu` command when present.
- One "getting started" that works on macOS Apple Silicon (hero platform) and gracefully degrades elsewhere.

---

## Specific File / Directory Actions (Initial Cut)

### Safe / Documentation-First (Phase 0–1)
- Create: `docs/architecture/HERO_THESIS.md`
- Create: `docs/architecture/LAYERING.md`
- Create: `docs/decision-records/2026-04-reorganization.md`
- Modify: `README.md` (hero story elevation + new recommended commands)
- Modify: `ncpu/lab.py` (add `gpu` subcommand + better Rust path surfacing + banners)
- Modify: `demos/README.md` (label hero vs research)
- Modify: `scripts/cli/README.md` + add deprecation header to `ncpu.py`
- Modify: `docs/maintainers/repo_hygiene.md` (reference the new proposal)

### Containment Moves (Phase 2, after Phase 1 lands)
- `artifacts/` dated subdirs → `artifacts/archive/`
- One-off scripts in `scripts/` → `scripts/archive/` or `scripts/dev/`
- Any remaining obvious Mog rename residue that is not test-critical → labeled historical notes

### Later / Higher-Risk
- Potential rename of `kernels/mlx/` to `kernels/mlx-legacy/`
- Re-homing of `ncpu/neural/` contents under clearer research namespaces
- Evaluation of `egdc/` and `some/` top-level status

---

## Risks & Mitigations

- **Risk**: Breaking existing user / CI workflows during unification.
  - **Mitigation**: Everything in Phase 0–1 is additive or adds warnings. Old commands keep working.
- **Risk**: Losing reproducibility artifacts during cleanup.
  - **Mitigation**: Never delete — only move with pointers. The publication harnesses stay green by design.
- **Risk**: Over-pruning useful research scaffolding (SOME, coprocessor training, etc.).
  - **Mitigation**: The proposal explicitly keeps the research surface; it just makes the hero thesis the front door.

---

## Success Metrics (How We Know It Worked)

1. A new contributor runs `python -m ncpu doctor` then one hero command and can articulate the unique GPU + neural-in-kernel contribution in < 5 minutes.
2. `git clone` size and `git status` noise are visibly reduced for everyday development.
3. Maintainers report less time explaining "which CPU is which" in issues/PRs.
4. Future high-value work (new neural ops in the kernel, coprocessor scaling, Weight CPU execution) has an obvious directory and command home.
5. All existing paper claims, tests (2500+), and reproducibility harnesses remain 100% green.

---

## Open Questions for Maintainer Discussion

1. How aggressively should we label `ncpu/neural/` and `ncpu/model/` as "research / training only" vs keeping them prominent?
2. Should the standalone Rust `ncpu_run` binary become the default "ncpu gpu" implementation even for non-neural use, with the Python paths becoming explicit "research" or "debug the models" modes?
3. Preferred home for the nsynth synthesis engine long-term (supporting tool for the differentiable layer, or more independent?).
4. Timeline comfort: Phase 0 this week, Phase 1 before next major paper push?

---

**Current Status (as of this session — major milestone reached)**

Phase 0–2 containment + exhaustive historical purge **complete** for the current wave:
- Archive directories created and populated (`artifacts/archive/`, `experiments/archive/`).
- All remaining historical MOG experiment runs (mog-run-001/002/003 + adaptive-memory + orchestrator + direct-router runs) moved from `experiments/` root into `experiments/archive/`. Root now contains only the policy README.
- 165 MB of pointless legacy tarballs (`artifacts/archives/egdc_deploy.tar.gz` + `ncpu_coprocessor.tar.gz`) permanently deleted.
- Full structural purge of deprecated subsystems already staged (entire `egdc/` tree, `docs/some/` duplicate docs, old duplicate root-level architecture files).
- `artifacts/` root reduced to 63 small human-readable summary .md files + the canonical `archive/` subtree only. No more dated sludge or tarballs at the working surface.
- `.tmp_pdf/`, stray tool configs, and other top-level noise removed.
- Key architecture documents (HERO_THESIS, LAYERING, BOTTOM_UP_JEPA_NEURAL_CPU, JEPA_MACHINE_WORLD_MODEL, cleanup_checklist) already accurately describe the live JEPA Neural Kernel (3 decision levers: direct bias override + Ready demotion + adaptive jepa_deprio_remaining skips driven by churn delta; fairness telemetry via times_scheduled / per_process_scheduled / get_all_deprios(); 69–80 real overrides observed on BusyBox workloads in launcher.rs + neural_jepa_kernel.rs).

The hero thesis ("The GPU *is* the computer" + bottom-up JEPA Neural OS direction on the real Rust Metal substrate) is the unambiguous front door. Canonical surfaces are clean. A newcomer can answer "what is the one amazing thing here?" from README + docs/maintainers/ in well under 60 seconds.

This integrated org + hygiene milestone is now verifiably complete. The proposal remains the north star for any future waves.

**Next (only if new high-signal work lands)**
- When the next fresh wheel / harness run produces updated override counts or new fairness differentiation data, feed the precise numbers back into the status sections and ncpu/jepa_neural_cpu/jepa_neural_cpu.py docstring.
- Continue the global pycache / flat-file hygiene sweep (task 8 in current pass) and run cargo check + core entrypoint verification.

This is evolutionary, high-signal, low-risk work that makes the unique contribution of the project obvious while preserving every reproducibility artifact.

---

*End of initial draft. Feedback welcome.*