# Noisy Directories Audit & Containment Plan
**Goal**: Reduce clone size, `git status` noise, and cognitive load while preserving every reproducibility artifact and paper claim.

**Date**: April 2026 (during reorganization push)

## Measured Sizes (as of this session)

| Directory              | Size   | Notes |
|------------------------|--------|-------|
| `experiments/`         | 876 MB | Largest offender. Mostly one-off experiment outputs. |
| `artifacts/`           | 350 MB | Mix of useful summaries + many dated full benchmark dumps, images, logs, mcp_sessions, vastai runs, adapters, etc. |
| `training_results/`    | 20 MB  | Coprocessor sweeps, instruct sweeps, real_memory proofs — some are still referenced. |
| `paper/generated/`     | ~8 MB  | Mostly timestamped submission bundles and meta-compare runs. Some are pinned baselines. |

Total "generated sludge" easily exceeds 1.2 GB and grows with every benchmark or paper run.

## Recommended Containment Strategy (Phased, Safe)

### Tier 1 — Move Immediately (Low Risk, High Win)

**`experiments/` → `artifacts/archive/experiments/` (or external if > few GB)**
- Almost everything here is transient experiment scratch.
- Keep a `experiments/README.md` that says "All historical experiment outputs have been archived. Reproduce via the scripts in `experiments/` or the relevant benchmark."
- Add to `.gitignore` patterns for future `experiments/*` (except the README).

**`artifacts/` dated / bulky subdirs**
Move the following classes (keep only small human summaries + the `archives/` dir which already seems intentional):

- `artifacts/nsynth_gradient_first/` and similar large dated runs
- `artifacts/vast_ai_run/`, `artifacts/vastai/`
- `artifacts/mcp_sessions/` (30+ files — these are probably session logs; decide if they belong in a separate non-git store)
- `artifacts/adapters/` (many dated qwen3 adapters — these are heavy; consider LFS or external model registry for the ones that are still active)
- `artifacts/universal_codegen_mix*`, `artifacts/codegen_*`, `artifacts/transfer_curve/`, `artifacts/mlx_distill_*`

**Rule**: If it has a timestamp in the name or is >50 MB and not explicitly a "pinned baseline", it moves to `artifacts/archive/YYYY-MM/`.

### Tier 2 — After the New Surface Is Solid

**`training_results/`**
- Audit which subdirs are still referenced in current benchmarks, `some/docs/`, or paper sections.
- Keep the active ones (e.g. recent coprocessor sweeps that feed current claims).
- Archive the rest (many are from 2026-03 or earlier specific model versions).
- Update any hard-coded paths in scripts that point into here.

**`paper/generated/`**
- The submission bundles and meta_compare runs are valuable for reproducibility of the paper.
- Strategy:
  - Keep the *latest canonical* ones that the `scripts/release/` harnesses actually use as baselines.
  - Move older timestamped ones (`20260416T...`, `rust-only-preview-...`, etc.) to `paper/archive/generated/`.
  - Strongly consider putting the big zips and full capture directories on Git LFS or an external artifact store, with only the manifest + summary md in git.

### Tier 3 — Process & Tooling (Prevents Regrowth)

1. Strengthen `.gitignore` (or use a `docs/maintainers/.gitignore.research` that people can cat into the root).
2. Make the `scripts/release/paper_artifacts.sh` and benchmark scripts automatically write heavy output under `artifacts/archive/<run-id>/` by default, with only a small manifest landing in the tracked locations.
3. Add a pre-commit or `make clean` target that warns about large untracked files in the noisy dirs.
4. Update `repo_hygiene.md` and `cleanup_checklist.md` with the new archive locations and a one-line command to do a "sludge sweep".

## Concrete First Moves (Safe to Do in One PR)

1. Create `artifacts/archive/` and `experiments/archive/` (or just move under `artifacts/archive/experiments/`).
2. Move the largest obvious dated trees from `artifacts/` (the ones listed in Tier 1).
3. Move `experiments/` contents (or the whole tree) under archive.
4. Add a top-level `ARTIFACTS.md` (or expand the existing one) that explains:
   - Where to find current baselines.
   - How to reproduce anything that was archived.
   - The policy for new runs.
5. Update any CI or nightly scripts that might be writing directly into the noisy locations.

## What Must Never Be Deleted or Made Hard to Reproduce

- Anything referenced by a committed `artifacts/*.md` summary that is part of a paper claim.
- The pinned baseline directories that `scripts/release/compare_artifacts.py`, `promote_publication_baseline.sh`, etc. use.
- Small, human-readable result files that tell the story without the raw GB of logs/images.

## Current Status (as of this session — major wave complete)

Containment + exhaustive purge executed to completion for the current cycle:
- `artifacts/archive/` and `experiments/archive/` directories created and populated.
- All remaining non-archived historical experiment runs (4 additional mog-* routing/memory runs + ablation_results) moved from `experiments/` root into archive/. Root now strictly follows the "scripts + README only" contract.
- 165 MB of pointless legacy tarballs in the anomalous `artifacts/archives/` (plural) dir permanently deleted (egdc_deploy + ncpu_coprocessor — deprecated subsystems with no reproducibility value in the tracked tree).
- `artifacts/` root reduced to 63 compact human-readable .md summaries + the canonical archive/ subtree. No more dated bulky dumps or tarballs at the working surface.
- Top-level noise (`.tmp_pdf/`, stray tool configs) removed. Local caches (`.hermes-lab*`, `.mog_synth_memory`, `.nCPU_autoresearch`, etc.) already correctly gitignored.
- 501+ deletions of deprecated code/docs (egdc/ full tree, docs/some/, duplicate old architecture files) staged from prior pass.
- Policy documents themselves updated to reflect the integrated milestone.

The "if pointless just delete / move to archive" rule is now the lived reality of the repo, not aspirational text.

## Updated Success Metric

- Root `artifacts/` and `experiments/` (outside the new `archive/` subdirs) should contain only small, human-readable summaries, current pinned baselines, and the scripts that produce them.
- `git status` after normal development is free of surprise 100 MB+ generated trees.
- A fresh clone (without LFS) is comfortable for day-to-day work.
- New contributors can find the hero path and current high-signal work without wading through dated experiment sludge.

This document (together with REPO_REORGANIZATION_PROPOSAL.md) is the living guide. Update the "Current Status" section as major containment waves complete.

---

**Owner**: Whoever drives the next phase of the reorganization. Can be done incrementally over a few low-risk PRs.