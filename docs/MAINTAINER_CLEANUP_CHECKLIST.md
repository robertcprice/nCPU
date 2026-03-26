# Maintainer Cleanup Checklist

Use this checklist before a public push, release, or README-heavy announcement.

## 1. Check the working tree

```bash
git status --short
```

Review every untracked path and ask:
- is this source code or documentation that should live in git?
- or is it a generated artifact, local result dump, weight file, or temporary export?

## 2. Keep only durable artifacts tracked

Usually keep:
- code
- tests
- docs
- benchmark scripts
- tiny fixtures needed for tests

Usually do not keep:
- `training_results/`
- raw benchmark outputs
- ad-hoc JSON dumps
- progress JSONL files
- tarballs
- copied local datasets
- one-off logs

## 3. Summarize, do not dump

If an experiment matters:
- keep the script tracked
- keep the command or invocation documented
- summarize results in markdown or a paper
- keep the large/raw output local or external

## 4. Re-run the flagship checks

```bash
pytest -q tests/test_package_metadata.py tests/test_lab_cli.py
python3 -m ncpu.lab demos
python3 -m ncpu.lab doctor
```

## 5. Verify the newcomer path still makes sense

A first-time visitor should be able to answer all of these from the repo root:
- what is nCPU?
- what should I run first?
- what works cross-platform?
- what is Apple-Silicon-specific?
- what is the flagship text/program-discovery experience?

## 6. Check the docs funnel

Make sure these are still aligned:
- `README.md`
- `demos/README.md`
- `benchmarks/README.md`
- `docs/REPO_HYGIENE.md`

## 7. Preserve the project shape

The intended top-level journey is:
1. interactive discovery
2. neural text machine
3. GPU systems demos
4. coprocessor / research-depth demos

If a new addition fights that flow, document it carefully or move it deeper into the tree.
