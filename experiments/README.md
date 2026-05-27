# Experiments (Archived)

All historical one-off experiment runs live under `archive/`.

## Policy

- `experiments/` root should contain only:
  - This README
  - Reproducible benchmark / training scripts that are actively maintained
  - Tiny fixtures required for those scripts
- Any run that produces weights, large logs, JSONL traces, per-step checkpoints, or dated ablation outputs belongs in `archive/<run-name>/` or is summarized in `artifacts/`.

## Why

The project generates high volumes of experimental sludge (see `docs/maintainers/NOISY_DIRS_AUDIT_AND_CONTAINMENT.md`). Keeping the root clean makes the hero path (GPU-as-computer + JEPA Neural Kernel) and current high-signal work obvious to newcomers and maintainers.

## Reproducing

Historical runs are preserved for reproducibility of paper claims. Re-run via the original scripts in `training/`, `benchmarks/`, or `scripts/` with the same seeds/configs noted in the archived `run.txt` or report files.

## Current Archive Contents

- `mog-run-001`, `mog-run-002-overfit`, `mog-run-003-completion`
- `mog-adaptive-memory-run1`, `mog-direct-router`, `mog-orchestrator-run1`, `mog-orchestrator-run2`
- Older ablation and routing memory experiments

Large dated artifacts from `artifacts/` also follow the same containment pattern (see `artifacts/archive/`).

If you are adding a new experiment that will produce bulky output, direct it to `artifacts/archive/<run-id>/` or an external store by default.
