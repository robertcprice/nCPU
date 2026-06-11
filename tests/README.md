# tests/

Test suite for nCPU. Subdirectories mirror the package layers:
`differentiable/`, `mog/`, `neural/`, `gpu/`, `autoresearch/`,
`self_optimizing/`, plus top-level `test_*.py` covering paper claims, artifact
integrity, packaging, and CLI entry points.

## Running
```bash
python -m pytest tests/ -q                      # fast suite (artifact-backed)
NCPU_NSYNTH_FULL_RUN=1 python -m pytest tests/test_nsynth_coverage.py  # opt-in: re-run Rust portfolio (~15 min)
```

The publication-claim regression layer (`test_paper_claims.py`,
`test_nsynth_coverage.py`, `test_artifact_integrity.py`, `test_paper_tables.py`)
asserts the headline numbers against the committed `artifacts/`; keep it green.
