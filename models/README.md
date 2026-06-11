# models/

Trained neural-component weights, grouped by the CPU layer each replaces:
`alu/`, `decode/`, `decoder/`, `math/`, `memory/`, `register/`, `neural/`.

See **`MODEL_INDEX.md`** for the authoritative per-model catalog (architecture,
accuracy, provenance). Large weight files (`*.pt`) are gitignored under
`checkpoints/`; this directory holds the curated, tracked model set.
