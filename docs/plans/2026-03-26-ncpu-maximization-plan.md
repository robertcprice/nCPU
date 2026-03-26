# nCPU Maximization Implementation Plan

> For Hermes: execute this plan in order, prioritizing flagship demo coherence, installability, and clear user-facing workflows over adding more subsystems.

Goal: turn nCPU from an impressive research repo with many entrypoints into a coherent, installable flagship project centered on interactive differentiable computing and text/program discovery.

Architecture: keep the existing research modules intact, but add one thin product layer on top: a unified `ncpu lab` launcher, a clean docs funnel, and reliable packaging metadata. The differentiable CPU, text machine, GPU demos, and coprocessor stay as the core; the new layer simply makes them easier to discover and run.

Tech Stack: Python, setuptools/pyproject packaging, pytest, existing `demos/` and `ncpu/differentiable/` modules.

Status snapshot:
- Interactive demos landed: `demos/interactive_discovery.py`, `demos/neural_text_machine.py`
- Frontier work landed: STE hard branches, per-example PC for batched branching, discrete verification
- Current blockers still visible in repo metadata:
  - malformed `pyproject.toml` author field
  - malformed `ncpu/__init__.py` author field
  - no unified installable flagship command
  - top-level README does not foreground the new interactive demos

---

## Phase 1: Immediate repo health and flagship entrypoint

### Task 1: Repair package metadata
Objective: make `pyproject.toml` and package metadata parse cleanly.

Files:
- Modify: `pyproject.toml`
- Modify: `ncpu/__init__.py`
- Test: `tests/test_package_metadata.py`

Steps:
1. Fix malformed author fields.
2. Ensure extras remain valid and consistent.
3. Add at least one console script entrypoint for the flagship launcher.
4. Add a regression test that parses `pyproject.toml` and imports `ncpu.__version__`.

Verification:
- `python -c "import tomllib, pathlib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['name'])"`
- `python -c "import ncpu; print(ncpu.__version__)"`
- `pytest -q tests/test_package_metadata.py`

### Task 2: Add unified `ncpu lab` launcher
Objective: expose the best interactive experiences behind one command.

Files:
- Create: `ncpu/lab.py`
- Possibly create: `tests/test_lab_cli.py`
- Modify: `pyproject.toml`

Behavior:
- `python -m ncpu.lab` opens a simple terminal menu.
- `ncpu-lab discover` runs the interactive discovery REPL.
- `ncpu-lab text` runs the neural text machine in interactive mode.
- `ncpu-lab demos` prints a curated ranked list of high-value demos.
- `ncpu-lab doctor` prints environment/platform guidance.

Verification:
- `python -m ncpu.lab --help`
- `python -m ncpu.lab demos`
- `python -m ncpu.lab doctor`
- `pytest -q tests/test_lab_cli.py`

### Task 3: Preserve reuse instead of rewriting demos
Objective: keep the launcher thin and stable.

Files:
- Modify only if needed: `demos/interactive_discovery.py`
- Modify only if needed: `demos/neural_text_machine.py`

Rules:
- Import and call existing functions where possible.
- Do not duplicate REPL logic in `ncpu/lab.py`.
- If needed, expose tiny helper functions from demo modules instead of copying code.

Verification:
- Both old commands still work.
- New launcher delegates correctly.

---

## Phase 2: Documentation funnel

### Task 4: Add `demos/README.md`
Objective: create a ranked demo index with platform expectations and “run this first” guidance.

Files:
- Create: `demos/README.md`

Sections:
1. Start here
2. Interactive demos
3. GPU/system demos
4. Coprocessor demos
5. Platform matrix
6. Expected runtime/dependency notes

The top two demos should be:
- Interactive Discovery REPL
- Neural Text Machine

### Task 5: Rewrite top README opening funnel
Objective: help newcomers understand what to run in under 30 seconds.

Files:
- Modify: `README.md`

Required top-of-file structure:
1. One-sentence thesis
2. “Run this first” block with exact commands
3. Three flagship experiences
4. Platform notes
5. Link to `demos/README.md`

Must explicitly feature:
- `demos/interactive_discovery.py`
- `demos/neural_text_machine.py --interactive`
- one GPU-native demo

Also update stale numbers if they are no longer accurate.

Verification:
- README top 120 lines answers:
  - what is nCPU?
  - what should I run first?
  - what works on Apple Silicon only?
  - what can do text/program discovery now?

---

## Phase 3: Product coherence around the flagship story

### Task 6: Define the flagship story as “program by examples”
Objective: keep repo messaging coherent.

Files:
- Modify: `README.md`
- Modify: `demos/README.md`
- Optional: add `docs/flagship-story.md`

Message:
- nCPU is the differentiable computer where you can specify behavior by examples and discover executable programs.
- Text transformation and cipher discovery are first-class manifestations of that thesis.
- GPU-native OS and coprocessor demos are deep supporting evidence, not the first impression.

### Task 7: Curate the demo taxonomy
Objective: prevent cognitive overload.

Recommended categories:
- Flagship interactive: discovery, text machine
- Systems wow: BusyBox, Alpine, GPU shell
- Research depth: coprocessor, ISA discovery, diff compiler, crypto verification

---

## Phase 4: Tight regression coverage

### Task 8: Add minimal tests for new user-facing workflows
Objective: keep the polished entrypoints stable.

Files:
- Create: `tests/test_package_metadata.py`
- Create: `tests/test_lab_cli.py`

Test cases:
- pyproject parses
- `ncpu` imports cleanly
- `python -m ncpu.lab --help` exits 0
- `python -m ncpu.lab demos` exits 0 and mentions both flagship demos
- `python -m ncpu.lab doctor` exits 0

Avoid brittle tests that depend on heavy models or GPU availability.

---

## Phase 5: Frontier work ordering after polish

### Task 9: Prioritize frontier work that improves the flagship
Do next:
1. soft branching quality improvements
2. per-example PC robustness/perf
3. better discrete extraction/verification UX

Do after that:
4. autoregressive compiler scale-up
5. real multi-device dispatch
6. full AES C -> Metal timing proof

Reason: these later items are valuable, but they do not improve first-run user experience as much as synthesis quality and clarity do.

---

## Suggested execution order for this repo now

1. Fix metadata and packaging
2. Add `ncpu lab`
3. Add demo index docs
4. Rewrite README top funnel
5. Add regression tests
6. Reassess demo UX and then continue frontier work

---

## Success criteria

The repo is “maximized” for the next pass when all of these are true:
- `pip install -e .` or `pip install -e ".[dev]"` works without malformed metadata
- a newcomer can run one flagship command and see the best interactive experience
- README clearly points to interactive discovery and text machine first
- GPU demos are still showcased, but not competing for top-level attention
- basic CLI/help/metadata tests protect the polished surface

---

## Immediate next implementation slice

1. Repair `pyproject.toml` and `ncpu/__init__.py`
2. Add `ncpu/lab.py` plus console script
3. Add `demos/README.md`
4. Update README quick-start block
5. Add `tests/test_package_metadata.py` and `tests/test_lab_cli.py`
