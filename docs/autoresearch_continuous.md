# Continuous Autoresearch — Runbook

**Spec:** [ROADMAP.md — Rung 5](../ROADMAP.md) (continuous autoresearch:
scheduled + fed by real usage).
**Result it produced:** Qwen3.5-4B **85.98% HumanEval (141/164)** =
compounding stack 111 + autoresearch 30 rescues, **$0.39** GPU cost —
paper sections 15.12/15.13 in
[`paper/sections/section_npcot.md`](../paper/sections/section_npcot.md),
artifacts in
[`training_results/realworld_vastai/`](../training_results/realworld_vastai/)
(`status.json`, `solved_programs.jsonl`, `log_autoresearch.txt`,
`humaneval_agent_4B.json`).

The daemon lives in [`ncpu/autoresearch/`](../ncpu/autoresearch/) (12
modules, ~2,200 lines). Continuous = run it nightly against whatever the
day's evals and user prompts failed on; every verified solve permanently
strengthens the next run.

---

## 1. The loop, precisely

```
attempt ──▶ verify ──▶ mine hard-fails ──▶ cascade-solve ──▶ distill
   ▲                                                            │
   └──────────── next run is strictly stronger ◀────────────────┘
```

### attempt
An eval runner produces a per-problem JSON:
`ncpu/self_optimizing/npcot_agent_runner.py` (the compounding-stack
agent) or `ncpu/self_optimizing/humaneval_runner.py` / the MBPP runner.
Each row in `per_problem` carries `task_id` and `passed`. The 85.98% run
started from `training_results/realworld_vastai/humaneval_agent_4B.json`
(the agent run that reached 111/164).

### verify
One correctness criterion everywhere:
`ncpu.autoresearch.cascade.verify_python_solution(item, candidate_code)`.
For HumanEval-style suites (test source contains `def check(`) it
delegates to `ncpu.self_optimizing.humaneval_runner._check_solution` —
the *same* function the eval runs use. For MBPP-style raw top-level
asserts it builds a subprocess harness (5 s timeout) and requires exit 0.
Nothing enters the store unverified, and cached programs are
**re-verified on every hit** before being accepted (stale entries fail
closed — see `runner.run_session`).

### mine hard-fails
`ncpu/autoresearch/miner.py::mine()`:

1. Load the eval JSON, select `per_problem` rows with `passed == False`.
2. Re-join with the benchmark dataset (`openai_humaneval` or `mbpp` via
   HuggingFace `datasets`) to recover prompt, `entry_point`, and test
   source. MBPP rows are reconstructed into runnable HumanEval-shaped
   records by `_mbpp_task_from_row` (prompt stub + entry-point guess from
   the canonical code).
3. `extract_io_pairs()` AST-parses the test source for
   `assert candidate(args) == expected` (both orientations) and keeps
   only compile-time-literal triples — deliberately conservative.
4. Items with fewer than `--min-io-pairs` (default 2) extractable pairs
   are skipped; the rest are appended as `ncpu.autoresearch.types.WorkItem`
   rows to `<artifact-dir>/<benchmark>_queue.jsonl`, deduped by
   `task_id`, priority `1.0 + 0.1 × len(io_pairs)`.

`WorkItem` = `{task_id, source_benchmark, prompt, entry_point,
test_source, io_pairs, canonical_solution?, priority, provenance}`
(`ncpu/autoresearch/types.py`). I/O pairs serialize losslessly via
`repr`/`ast.literal_eval` round-trip (`IoPair.to_dict/from_dict`).

### cascade-solve
`ncpu/autoresearch/cascade.py::run_cascade(item, config)` tries solvers
in order and returns the first **verified** `SolvedItem`:

| solver | where | what |
|---|---|---|
| `template_match` | `solvers.py` | brute-force over ~30 Python templates (array reductions, scalar 1-arg/2-arg ops) against the extracted I/O pairs. Pure CPU, milliseconds. |
| `nsynth_fast` | `solvers.py` | stub today (returns `None`); the integration point for the nsynth Rust synthesizer. |
| `llm_resample` | `llm_resample.py::make_llm_resampler` | re-runs the *same* target LLM with a bigger sampling budget: default 4 temperatures (0.3, 0.5, 0.7, 0.9) × 4 samples = 16 attempts, each verified against the full test suite. Injected at runtime via `CascadeConfig.extra_solvers` by `driver.py` (needs a model handle). On a win it stamps `winning_temperature` / `winning_sample_idx` into provenance. Optional `temperature_priority` hook reorders temps by historical wins. |
| `llm_teacher` | `solvers.py` | stub for a stronger-model API escalation. |

This is strictly more powerful than what ran at eval time: the agent
runner gave up after its retry schedule (the 4B run used
`retry_gates=[0.0, 0.02, 0.05]`, `retry_temperatures=[0,0,0,0.5]`,
`max_retries=4` — see the `config` block of `humaneval_agent_4B.json`);
the resampler explores 16 fresh samples across 4 temperatures. That is
where all 30 rescues came from (`by_solver: {"llm_resample": 30}` in
`training_results/realworld_vastai/status.json`).

### distill
On a verified solve, `runner.run_session` calls
`ncpu.autoresearch.compounding_store.CompoundingStore.record()`, which
updates three indices in the artifact dir:

1. `solved_programs.jsonl` — append-only fact log, one `SolvedItem` per
   line (source of truth; rebuildable indices).
2. `prompt_cache.json` — `hash_prompt(prompt, entry_point)`
   (sha256/32 hex) → `{task_id, program_python, solver, provenance}` for
   exact-match re-runs.
3. `temperature_stats.json` — per-temperature win counts across all
   sessions, so future resample sweeps can try historically winning
   temperatures first.

A fourth, best-effort output: when the solution is translatable to a
`DiscreteArrayProgram` (array-reduction shape), the NPCoT library is
grown in place via `continual_library.record_successful_generation`
(see `compounding_store.py` module docstring and
`distiller.py`). `distiller.py` also provides `load_solved`,
`dedupe_solved` (keep latest per task_id), and `summarize_solved`.

**Misses / negative memory.** Unsolved items stay in the queue (the
session report records the last cascade error per item), so the next
GPU session re-attempts exactly the residual set. At the synthesis tier,
negative memory is the nsynth rejected-programs bank
(`~/.nsynth_rejected_programs.tsv`, env `NSYNTH_REJECTED_PATH`) plus the
per-problem rejected-codes cache inside the gradient solver — see
[`ncpu/synthesis_api/README.md`](../ncpu/synthesis_api/README.md) for the
bank table. There is no dedicated negative store in
`ncpu/autoresearch/` yet; wiring one is part of §4.

### next run is strictly stronger
`runner.run_session` short-circuits, in order:

1. **Legacy-log skip** — `task_id` already in `solved_programs.jsonl` →
   `problems_already_solved_skipped`.
2. **Prompt-cache hit** — `CompoundingStore.check_prompt(item)` matches
   the exact prompt hash (works across *different* task ids asking the
   same thing) → re-verify, then count as `store_hits` with zero cascade
   cost.
3. Only then does the cascade run, under a `Budget` (wall seconds, max
   problems, symbolic USD at `cost_per_gpu_hour_usd = 0.19`).

The same store feeds live evals: `npcot_agent_runner.py` accepts
`--compounding-store DIR` (config field `compounding_store_dir`, "AR-6")
and consults `store.check_prompt` *before generating anything*, then
records its own wins back — so eval runs and autoresearch sessions
compound into one shared memory. `run_livecodebench.py` does the same.

---

## 2. Exact commands

All verbs share `--artifact-dir` (default `.nCPU_autoresearch/` in the
cwd — `DEFAULT_ARTIFACT_DIR` in `types.py`).

### One cycle locally (CPU or MPS, no GPU)

```bash
# 0) one-command repro of the whole local cycle (verified, see §6):
scripts/autoresearch_cycle_demo.sh /tmp/ar_demo

# -- or step by step --

# 1) mine hard-fails from a real eval JSON into the work queue
python3 -m ncpu.autoresearch.cli --artifact-dir /tmp/ar_demo mine \
    --eval training_results/realworld_vastai/humaneval_agent_4B.json \
    --benchmark humaneval            # also: mbpp; --min-io-pairs N; --task ID

# 2) consume the queue once under a budget (CPU-only solver)
python3 -m ncpu.autoresearch.cli --artifact-dir /tmp/ar_demo run-once \
    --benchmark humaneval --wall-seconds 300 --max-problems 60 \
    --per-problem-seconds 5          # --solver NAME repeatable; default template_match

# 3) inspect store + queues + last session
python3 -m ncpu.autoresearch.cli --artifact-dir /tmp/ar_demo status

# 4) housekeeping: keep latest solve per task_id
python3 -m ncpu.autoresearch.cli --artifact-dir /tmp/ar_demo dedupe

# 5) one-shot: arbitrary coding request end-to-end (see §3)
python3 -m ncpu.autoresearch.cli --artifact-dir /tmp/ar_demo user \
    --prompt 'def cube(x): returns the cube. cube(2) -> 8, cube(-3) -> -27'
```

The `mine` step needs the `datasets` package and HF Hub access (cached
after first download; set `HF_TOKEN` to avoid rate-limit warnings).
Everything else is offline.

### GPU session (the binary the vast.ai VM actually runs)

`ncpu/autoresearch/driver.py` is the production driver — it loads the
model, installs the real `llm_resample` solver, and runs the session.
It is factored out of `cli.py` so the lightweight CLI never imports
torch/transformers.

```bash
python3 -m ncpu.autoresearch.driver \
    --model Qwen/Qwen3.5-4B \
    --queue .nCPU_autoresearch/humaneval_queue.jsonl \
    --solved .nCPU_autoresearch/solved_programs.jsonl \
    --status .nCPU_autoresearch/status.json \
    --wall-seconds 1800 --max-problems 30 --max-cost-usd 1.0 \
    --per-problem-seconds 120 \
    --temperatures 0.3,0.5,0.7,0.9 --samples-per-temp 4 \
    --include-templates-first \
    # optional NPCoT coprocessor (both flags or neither):
    --library /workspace/checkpoints/npcot_qwen3.5-4B_library.json \
    --coprocessor-checkpoint /workspace/checkpoints/npcot_qwen3.5-4B.pt
```

Other flags: `--device auto|cuda|mps|cpu`, `--target-layers -2,-1`,
`--array-max-len 8`, `--trust-remote-code`.

### Nightly on vast.ai (~$0.10–0.19/hr)

The validated provision/run/teardown pattern lives in
[`packaging/scripts/vast_run.sh`](../packaging/scripts/vast_run.sh)
(search offers ≤ `GPU_BUDGET` $/hr with ≥ `GPU_RAM_MIN` GB VRAM, rsync
the repo to `/workspace/nCPU`, run, pull artifacts, destroy instance on
exit). Its current modes are `tests | bench3 | humaneval | mbpp`; an
`autoresearch` mode is a natural addition, but the manual sequence the
85.98% run used is:

```bash
# rent + upload (use vast_run.sh's offer search, or by hand):
vastai search offers 'reliability > 0.99 num_gpus=1 gpu_ram>=16 verified=true rentable=true dph_total<=0.19' -o dph_total
vastai create instance <OFFER_ID> --image pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel --disk 40
rsync -az --exclude='.git/' --exclude='*/target/' --exclude='models/' \
      --exclude='training_results/' -e "ssh -p $PORT" ./ root@$HOST:/workspace/nCPU/

# on the box:
pip install -q transformers datasets torch
cd /workspace/nCPU
python3 -m ncpu.autoresearch.cli mine --eval <eval.json> --benchmark humaneval
python3 -m ncpu.autoresearch.driver --model Qwen/Qwen3.5-4B \
    --queue .nCPU_autoresearch/humaneval_queue.jsonl \
    --solved .nCPU_autoresearch/solved_programs.jsonl \
    --status .nCPU_autoresearch/status.json \
    --wall-seconds 7200 --max-problems 60 | tee log_autoresearch.txt

# pull the store back and destroy the instance:
rsync -az -e "ssh -p $PORT" root@$HOST:/workspace/nCPU/.nCPU_autoresearch/ \
      ./training_results/realworld_vastai/
vastai destroy instance <ID>
```

Real cost of the measured run: 7,329.86 s wall on an RTX 3090 ≈
**$0.3869** (`Budget.cost_per_gpu_hour_usd` default 0.19 matches the
rented rate). The budget guard stops the session before
`--max-cost-usd` is exceeded.

**Outputs land in the artifact dir**: `<benchmark>_queue.jsonl`
(residual queue), `solved_programs.jsonl`, `prompt_cache.json`,
`temperature_stats.json`, `status.json` (live-updated each problem).
Commit the pulled store into `training_results/` (the 85.98% run's
store is checked in) and point the next eval's
`--compounding-store` at it.

---

## 3. The prompt→WorkItem extractor

`ncpu/autoresearch/prompt_parser.py` turns an arbitrary coding request
into a *verifiable* work item — the coding-assistant path into the
cascade. `build_work_item(prompt, task_id=..., entry_point=None,
extra_io_pairs=None)`:

1. **Entry point**: explicit arg > last `def name(...)` line in the
   prompt (`extract_entry_point`). No entry point → returns `None`
   (honest refusal: nothing to solve for).
2. **I/O pair extraction** (`extract_from_prompt`), deduped across
   patterns, tagged in `provenance["extraction_sources"]`:
   - `doctest` — `>>> f(args)` + next-line expected value
   - `assert` — `assert f(args) == expected` *in fully-parseable Python
     text*; `assert_fenced` — the same inside ``` fences (prose
     containing asserts must be fenced, since the assert extractor
     `ast.parse`s the whole text)
   - `arrow` — `f(args) -> expected` (or `→`), tolerates prose
   - `returns` — `f(args) returns expected` prose, per sentence
   All values must be Python literals; non-literal expressions are
   skipped. No tests are invented — when the prompt has no examples, a
   caller can pass LLM-proposed candidates via `extra_io_pairs`.
3. **Harness synthesis**: emits a minimal `def check(candidate):` block
   of asserts plus a clean `def <entry>(<params>):` runtime prompt with
   a docstring distilled from the surrounding prose — so
   `verify_python_solution` runs the identical path it uses for
   HumanEval/MBPP.

CLI surface: `python3 -m ncpu.autoresearch.cli user --prompt '...'`
(or `--prompt-file`, or stdin; `--entry-point` override; `--solver`
repeatable). On a solve it prints the verified function and records it
to the CompoundingStore; exit code 1 on honest refusal.

Known limitations (observed while building §6):
- plain-prose `assert` lines extract only inside fenced code blocks;
- **bug**: extracted pairs with list/dict-valued *arguments* crash the
  dedupe in `prompt_parser._accept` (`prompt_parser.py:268` —
  `tuple(p.args)` is unhashable when an arg is a list; the `_freeze`
  helper is applied to `expected` but not to `args`). Scalar and string
  args work. Fix is a one-liner; until then keep user-path examples
  scalar or route list-shaped problems through the miner.

---

## 4. Wiring future work-item sources (TODO design — not implemented)

Both new Rung 3/4 services are verifier-gated, so their *misses* are
exactly the shape the cascade wants: a name + I/O examples that some
cheaper tier failed to satisfy.

### Registry misses → queue
[`tools/registry/server.py`](../tools/registry/server.py) (verified-skill
registry, SQLite) rejects submissions whose program fails re-execution:
`handle_submission()` returns 422 with the concrete `first_failure`
counterexample (verification by `tools/registry/executor.py`, the
pure-Python mirror of the canonical executor).

- **Capture point**: in `handle_submission()`'s rejection branch, append
  `{name, author, examples, error, ts}` to a
  `registry_misses.jsonl` next to the SQLite DB. Optionally also log
  `do_GET` 404s on `/skills/{fp}` (demand for a skill nobody has
  contributed yet) — name-only, no examples, so lower value.
- **Adapter**: new `ncpu/autoresearch/sources/registry.py` with
  `mine_registry_misses(misses_path, out_path) -> counters`, mapping
  each miss to a `WorkItem`: `entry_point` from the sanitized skill
  name, `io_pairs` from the registry `examples`
  (`{data, n_points, target}` → `IoPair(args=[data], expected=target)`),
  `source_benchmark="registry"`, and a synthesized `def check(candidate)`
  harness (reuse `prompt_parser.build_work_item`'s harness emitter).
- **Close the loop**: when the cascade solves one and the solution is
  expressible as an NPCoT program, POST it back to `/skills` — the
  registry re-verifies, so the loop stays trustless.

### Synthesis-API refusals → queue
[`ncpu/synthesis_api/server.py`](../ncpu/synthesis_api/server.py) returns
`success: false` plus the backend error when nsynth cannot find a
program (`handle_synthesize_request()`); per its README this honest
refusal is what escalates to the LLM tier.

- **Capture point**: in `handle_synthesize_request()`'s failure branch,
  append the validated request (`name`, `examples`, error, elapsed) to
  `refusals.jsonl` alongside the solved/bias/rejected banks
  (`read_bank_stats()` already reports bank sizes; add the refusal count
  there).
- **Adapter**: `mine_synthesis_refusals(refusals_path, out_path)` in the
  same `ncpu/autoresearch/sources/` module — `entry_point` from
  `_sanitize_fn_name(name)`, signature via `_build_signature`, examples
  → `IoPair`s, `source_benchmark="synthesis_api"`.
- **Distill back**: a cascade win on a refusal is a labelled training
  example for exactly the problem family nsynth currently misses — feed
  it to the nsynth learned-bias bank / teacher data, not only the
  prompt cache.

Driver integration for both: `cli.py cmd_run_once` currently derives the
queue path from `--benchmark {humaneval,mbpp}`; extend the choices with
`registry` and `synthesis_api` (queue files `registry_queue.jsonl`,
`synthesis_api_queue.jsonl`) so one nightly driver invocation per source
works unchanged.

---

## 5. The CI self-improvement gate

[`tools/internal/ci_self_improvement_gate.sh`](../tools/internal/ci_self_improvement_gate.sh)
enforces the *cross-run learning* invariant on the nsynth side of the
house: a solve in round 0 must make round 1 strictly cheaper.

What it does:

1. Builds (if needed) the nsynth release binaries `transfer_curve` and
   `curve_analysis`.
2. Runs a 2-round **cumulative** sweep over a small problem slice
   (`--limit 20` default, `--offset` to shift the slice): round 0
   populates the solved-program cache, round 1 should hit it.
3. Feeds both rounds to `curve_analysis --baseline-round 0
   --treatment-round 1 --json` and extracts `improvement_rate_median`,
   `slowdowns`, `instant_hits`.
4. **Fails (exit 1)** when `median_ratio > threshold` (default 1.05 —
   i.e. round 1 is materially *slower* than round 0, meaning the cache
   isn't being used). Nonzero `slowdowns` alone is a loud WARN, not a
   hard fail (a real regression also moves the median). Exit 2 = bad
   args / missing binaries.

Env knobs: `NSYNTH_CACHE_PATH`, `NSYNTH_META_WEIGHTS_PATH`,
`NSYNTH_TEACHER_BUDGET_SEC` (the CI job pointed these at `/tmp` paths so
the gate never touches real user banks).

Reading a failure: the gate prints the full analysis JSON, then
`FAIL: median_ratio X > threshold Y` and
`cross-run learning regression detected`. That means round-1 cache hits
stopped landing — first suspects are the solved-cache path
(`nsynth/src/solved_cache.rs`, fingerprint or re-verify logic) or an
env override pointing the cache somewhere unwritable. `instant_hits`
near zero with a green build is the smoking gun.

History note: this gate ran in GitHub Actions as the
`self_improvement_gate` job of `.github/workflows/ci.yml`
(`tools/ci_self_improvement_gate.sh --limit 15`) until all workflows
were removed on 2026-06-11 (commit `2514102`, "nightly/weekly runs
spamming failure emails"). The script remains the CI-able contract —
run it locally or wire it into whatever scheduler replaces Actions:

```bash
NSYNTH_CACHE_PATH=/tmp/nsynth_ci_gate_cache.json \
NSYNTH_TEACHER_BUDGET_SEC=8 \
tools/internal/ci_self_improvement_gate.sh --limit 15
```

The autoresearch-side equivalent of this invariant is regression-tested
in `tests/autoresearch/test_compounding_end_to_end.py`
(`TestAlwaysCompounding`: a session-1 solve must return as a session-2
store hit with `problems_attempted == 0`, even after the legacy log is
deleted).

---

## 6. A demonstrated cycle (real numbers, measured 2026-06-11)

Everything below was actually executed on this machine (Apple Silicon,
CPU only — no GPU, no model load), via
[`scripts/autoresearch_cycle_demo.sh`](../scripts/autoresearch_cycle_demo.sh)
on a fresh artifact dir. Reproduce with:

```bash
scripts/autoresearch_cycle_demo.sh /tmp/ar_demo
```

**Step 1 — mine** the real vast.ai eval JSON
(`training_results/realworld_vastai/humaneval_agent_4B.json`, the
111/164 agent run):

```json
{"hard_fails_total": 53, "written": 51, "skipped_no_task": 0, "skipped_no_io_pairs": 2}
```

51 work items — the *same* 51 the $0.39 GPU session attempted.

**Step 2 — prompt→WorkItem extraction** of 4 user-shaped requests, one
per extraction pattern:

```
demo/square   entry=square   io_pairs=2 sources={'arrow': 2}
demo/abs_diff entry=abs_diff io_pairs=2 sources={'arrow': 2}
demo/add      entry=add      io_pairs=2 sources={'doctest': 2}
demo/mul      entry=mul      io_pairs=2 sources={'assert_fenced': 2}
```

**Before**: `solved_programs.jsonl` 0 rows, `prompt_cache.json` 0
entries, queue 55 items.

**Step 3 — run-once session 1** (`--solver template_match`, CPU):

```json
{"problems_attempted": 55, "problems_solved": 4,
 "problems_already_solved_skipped": 0, "store_hits": 0,
 "wall_seconds": 3.12, "estimated_cost_usd": 0.0002,
 "by_solver": {"template_match": 4}, "stopped_reason": "done"}
```

The 4 extracted items were solved and verified; the 51 mined hard-fails
were honestly *not* solved by the cheap tier — they are precisely the
residual queue a nightly GPU `driver.py` session consumes (on the real
run, `llm_resample` solved 30 of these 51 in 7,330 s / $0.39).

**After session 1**: `solved_programs.jsonl` **4 rows**,
`prompt_cache.json` **4 entries**.

**Step 4 — re-ask** two of the same prompts under *new* task ids
(`demo/square_again`, `demo/add_again`) — simulating a user asking for
the same function tomorrow.

**Step 5 — run-once session 2**:

```json
{"problems_attempted": 51, "problems_solved": 2,
 "problems_already_solved_skipped": 4, "store_hits": 2,
 "wall_seconds": 4.88, "estimated_cost_usd": 0.0003,
 "by_solver": {}, "stopped_reason": "done"}
```

Reading: the 4 session-1 task ids were skipped via the legacy log; the
2 re-asked prompts were answered from the **prompt cache**
(`store_hits: 2`, re-verified before acceptance) with **zero cascade
invocations** (`by_solver: {}`); the 51 hard-fails were re-attempted and
remain queued. The store only grows: that is the always-compounding
contract.

A one-shot `user`-verb run was also exercised for §3:

```
$ python3 -m ncpu.autoresearch.cli --artifact-dir /tmp/ar_demo user \
    --prompt 'def cube(x): returns the cube. cube(2) -> 8, cube(-3) -> -27'
[user] entry_point=cube  io_pairs=2  sources={'arrow': 2}
[user] SOLVED by template_match
def cube(x):
    """Implement cube."""
    return x ** 3
```

bringing the store to **5 solved rows / 5 prompt-cache entries**.

**Test suite** (same day, same machine):

```
$ python3 -m pytest tests/autoresearch/ -v
============================== 44 passed in 3.79s ==============================
```

**Not exercised locally**: the GPU `llm_resample` tier (`driver.py` with
a loaded Qwen3.5 model) — its measured behaviour is the checked-in
vast.ai artifact set (`status.json`: 51 attempted, 30 solved,
7,329.86 s, $0.3869).

---

## Cross-references

- [ROADMAP.md](../ROADMAP.md) — Rung 5 spec and DoD.
- [`paper/sections/section_npcot.md`](../paper/sections/section_npcot.md)
  §15.12 (autoresearch + 85.98% composite), §15.13 (generalization
  beyond coding benchmarks / three-tier verifier degradation).
- [`ncpu/synthesis_api/README.md`](../ncpu/synthesis_api/README.md) —
  HTTP synthesis tier, persistent solved/bias/rejected banks.
- [`tools/registry/README.md`](../tools/registry/README.md) —
  trustless verified-skill registry (future miss source).
- `tests/autoresearch/` — invocation patterns and the always-compounding
  regression tests.
