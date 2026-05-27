# Decision Record: Repository Reorganization Initiation (April 2026)

**Date**: 2026-04  
**Status**: Accepted (initial phase)  
**Deciders**: Maintainer + analysis  
**Related**: `docs/maintainers/REPO_REORGANIZATION_PROPOSAL.md`, `docs/architecture/HERO_THESIS.md`, `docs/architecture/LAYERING.md`

## Context

After deep evaluation of the nCPU repository, it became clear that the project contains multiple high-value technical contributions, but they are difficult for newcomers (and even returning contributors) to discover and prioritize because of:

- Multiple parallel CPU implementations (`ncpu/neural/`, `ncpu/model/`, `kernels/mlx/`, `kernels/rust_metal/`).
- The strongest unique claim ("CPU modeled on GPU with neural networks and kernels" — the Rust Metal kernel with in-shader neural ALU weights, full deterministic OS, and 26-command debugging superpowers) was under-exposed in the user-facing surface.
- Fragmented entry points (`python -m ncpu`, `ncpu-lab`, `scripts/cli/ncpu.py`, direct `demos/*.py` calls, legacy flags in `ncpu/demo.py`).
- Historical project layering (Mog/nsynth, some/ vs self_optimizing, egdc, etc.) creating a "multiple projects sharing a repo" feel.
- Generated artifacts and experimental scaffolding mixed with source.

The March 2026 maximization plan had already identified part of this (lack of unified flagship launcher) and begun work with `ncpu/lab.py`. The hygiene docs (`repo_hygiene.md`, `cleanup_checklist.md`) showed good self-awareness but were not yet sufficient to make the hero signal obvious.

## Decision

We initiated a phased reorganization with the following non-negotiable principles (codified in the proposal):

1. Elevate the **Rust Metal GPU kernel + neural ALU in shaders** as the primary hero thesis ("The GPU *is* the computer").
2. Create clear layering documentation (`HERO_THESIS.md` + `LAYERING.md`) so ownership and tradeoffs are explicit.
3. Unify and improve the user-facing surface (starting with enhancements to the existing `lab.py` work) while adding safe deprecation guidance for legacy paths.
4. Contain (rather than delete) historical noise and generated artifacts.
5. All changes must preserve 100% of existing tests, reproducibility harnesses, and paper claims.

We chose an evolutionary approach: documentation + new preferred paths + labels first (Phase 0–1), larger structural moves only after the new surface proves itself.

## Consequences

### Positive
- New contributors will be able to articulate the unique contribution within minutes instead of hours.
- Future high-value work has obvious homes (Performance Layer vs Research Layer).
- The excellent existing `lab.py` unifier and hygiene processes are respected and extended.
- The hero technical achievement (in-shader neural execution + determinism superpowers) becomes the front door.

### Negative / Tradeoffs
- Short-term increase in documentation surface (three new docs landed in this session).
- Maintainers must be disciplined about not letting new parallel implementations appear without clear layering justification.
- Some legacy paths will carry deprecation notices for a transition period (temporary user confusion is accepted as the cost of clarity).

### Neutral / To Watch
- The research surface (`ncpu/differentiable/`, coprocessor, SOME, nsynth) remains fully supported but is now explicitly positioned as supporting/enabling the hero thesis rather than co-equal.

## Next Actions (Recorded at Time of Decision)

1. Land the proposal + HERO_THESIS + LAYERING + this decision record.
2. Enhance `ncpu/lab.py` with a `gpu` hero subcommand that prefers the Rust backend and surfaces the neural-in-shader option.
3. Update top-level README and `demos/README.md` to point at the new hero framing.
4. Add deprecation notices to the most confusing legacy entry points (`scripts/cli/ncpu.py`).
5. Iterate the proposal based on maintainer feedback before larger moves.

## References
- Full proposal: `docs/maintainers/REPO_REORGANIZATION_PROPOSAL.md`
- Hero framing: `docs/architecture/HERO_THESIS.md`
- Layering contract: `docs/architecture/LAYERING.md`

---

This decision record will be updated as phases complete.