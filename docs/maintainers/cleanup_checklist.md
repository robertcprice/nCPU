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
- `docs/maintainers/repo_hygiene.md`

## 7. Preserve the project shape (updated with current hero thesis)

The intended top-level journey is now:
1. ★ HERO: `python -m ncpu gpu` (or `ncpu gpu`) — the GPU *is* the computer (Rust Metal + optional neural ALU in-shader + the live JEPA Neural Kernel steering real scheduling on that computer).
2. Interactive discovery / neural text machine (cross-platform, lightweight).
3. GPU systems depth (debugging toolkit, self-hosting, determinism superpowers).
4. Research depth (coprocessor, differentiable synthesis, bottom-up JEPA Neural OS direction).

The canonical surfaces are:
- Performance + Determinism Layer: `kernels/rust_metal/`
- Research + Differentiability Layer: `ncpu/`

All new work should make the tradeoff explicit and should strengthen (or at least not fight) the hero path. If something fights the flow, document the tradeoff clearly or move it deeper.

## 8. Current high-signal work (update this section as major milestones land)

As of this session the live JEPA Neural Kernel integration with the real Rust Metal substrate is the clearest current example of the two layers working together at the highest signal:
- 3 decision levers (context-switch bias, immediate syscall bias on mem ops, adaptive persistent yield via model-computed de-prio skips).
- True recency-aware relative churn scoring producing visible differentiation on real BusyBox workloads.
- Fairness telemetry (`times_scheduled`, `per_process_scheduled`, `get_all_deprios`).
- Real results: 69–80 scheduling overrides per short multi-process run on actual guest code.

Keep this section (and the architecture docs it points to) current. A newcomer should be able to answer "what is the most exciting thing happening right now?" from the README + this checklist + the key architecture documents in under 5 minutes.
