# Benchmarks

This directory contains the benchmark harnesses and scripts for nCPU.

## Structure
- `*.py` at top level: individual benchmark scripts (keep short and focused).
- `results/`: JSON and data outputs from runs (gitignored or archived for large ones).
- `archive/`: Old or large historical results.
- `coprocessor/`, `gpu/`, etc. (future grouping for related families).

Run via `python -m ncpu.lab` or directly. See top-level benchmarks/README.md for overview.

New benchmarks should be added as focused .py files or in logical subdirs. Avoid dumping long output here — use results/ or external storage.
