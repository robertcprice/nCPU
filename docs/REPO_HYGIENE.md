# Repository Hygiene Guide

This repository mixes research code, runnable demos, benchmark harnesses, papers, and experimental outputs. To keep the project readable, only durable source artifacts should live in git.

## Commit these

- source code under `ncpu/`, `demos/`, `tests/`, `benchmarks/`, `kernels/`, `training/`
- documentation and papers
- small deterministic fixtures that are required to reproduce tests
- benchmark scripts and helper code
- concise result summaries written for humans

## Do not commit these by default

- `training_results/` trees
- raw benchmark output dumps
- `*.json.progress.jsonl`
- remote snapshots
- ad-hoc large evaluation datasets copied locally
- tarballs and exported archives
- one-off logs generated during local experimentation

## If a result matters, summarize it

When a run is worth keeping:

1. leave the raw artifact local or in external storage
2. summarize the important numbers in a tracked markdown report
3. link the script/command that produced it
4. keep the summary small and reviewable

Good pattern:
- tracked: `benchmarks/benchmark_coprocessor_realworld.py`
- tracked: `benchmarks/README.md` or a paper/report with summarized findings
- untracked/local: raw JSON dumps, logs, progress streams, checkpoint folders

## Preferred install paths

For the flagship demo surface:

```bash
pip install -e ".[demo,dev]"
```

For a broader local research environment:

```bash
pip install -e ".[demo,model,train,dev]"
```

## Preferred first-run flow

1. `python -m ncpu.lab demos`
2. `python -m ncpu.lab discover`
3. `python -m ncpu.lab text --interactive`
4. then move on to systems or coprocessor demos as needed

## Why this matters

nCPU is strongest when a newcomer can immediately find:
- the flagship interactive demos
- the systems wow demos
- the research depth demos

Keeping generated artifacts out of the main repo surface makes that much easier.

For a quick pre-push review, see `docs/MAINTAINER_CLEANUP_CHECKLIST.md`.
