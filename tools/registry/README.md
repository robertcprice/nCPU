# Verified-Skill Registry

**crates.io for synthesized programs.** A community registry for NPCoT
skills where contribution is *trustless*: the server re-executes every
submitted program against its claimed I/O examples before anything is
stored. Wrong code, approximate code, and spam physically cannot enter —
there is no moderation queue because there is nothing to moderate.

## The trustless-contribution model

Every NPCoT program is a tiny deterministic IR (init / transform / reduce /
post-scale / offset, plus combine + predicate guards in v2). Executing one
costs microseconds. That changes the economics of a package registry:

1. A submission carries its own proof obligation: the `examples` it claims
   to solve.
2. The server re-runs the program over **all** examples with
   [`executor.py`](executor.py) — a pure-Python mirror ported exactly from
   the canonical executor in `kernels/npcot_wasm/src/lib.rs`.
3. Accept iff `max abs error <= 1e-3 * max(1, max|target|)` — the same
   acceptance rule the browser-tier synthesizer uses, so anything it
   discovered and verified is accepted here verbatim.
4. Rejection returns the concrete counterexample (`first_failure`), not a
   vague error.

Verification is the gate, not reputation. An anonymous submission and a
maintainer submission go through the identical check.

The trust guarantee is also *re-checkable forever*: `--verify-all` replays
verification over every stored skill, so CI (or any skeptical user) can
prove the whole registry is still sound — even against direct database
tampering.

## Running

```bash
# Start the server (stdlib-only: http.server + sqlite3, zero deps)
python3 -m tools.registry.server --port 8430 --db registry.sqlite

# Trust sweep — exits nonzero listing any skill that fails re-verification
python3 -m tools.registry.server --db registry.sqlite --verify-all
```

## Endpoints

### `POST /skills` — submit a skill

```bash
curl -s -X POST http://127.0.0.1:8430/skills \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "sum",
    "author": "alice",
    "examples": [
      {"data": [1.0, 2.0, 3.0], "n_points": 3, "target": 6.0},
      {"data": [10.0, -4.0],    "n_points": 2, "target": 6.0}
    ],
    "program": {"init_idx": 0, "transform_idx": 0, "reduce_idx": 0,
                "post_scale_idx": 0, "offset": 0.0}
  }'
# → {"accepted": true, "duplicate": false,
#    "fingerprint": "9c40…", "skill_id": 1, "max_err": 0.0}
```

v2 skills (multi-field records + guards) use `program_v2` instead:

```bash
curl -s -X POST http://127.0.0.1:8430/skills \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "dot_product", "author": "bob",
    "examples": [{"data": [2.0, 3.0, 4.0, 0.5], "n_points": 2, "target": 8.0}],
    "program_v2": {"arity": 2, "combine_idx": 3, "guard_idx": 0,
                   "guard_threshold": 0.0, "init_idx": 0, "transform_idx": 0,
                   "reduce_idx": 0, "post_scale_idx": 0, "offset": 0.0}
  }'
```

A wrong program is rejected with the proof of wrongness:

```bash
# Claims "sum" but transform_idx=1 computes sum of squares
# → HTTP 422
# {"accepted": false, "max_err": 110.0,
#  "first_failure": {"example_index": 0, "expected": 6.0, "got": 14.0}, ...}
```

Deduplication is by **examples fingerprint** (sha256 over a sorted stable
serialization of the canonical examples):

* same fingerprint + identical program → `200 {"duplicate": true}`, count
  unchanged;
* same fingerprint + *different* program → both kept — they are
  alternative verified solutions to the same task.

### `GET /skills` — list with attribution

```bash
curl -s http://127.0.0.1:8430/skills
# → {"skills": [{"id": 1, "name": "sum", "author": "alice",
#     "fingerprint": "9c40…", "format": 1, "max_err": 0.0,
#     "created_at": "2026-06-11T17:02:11+00:00"}], "count": 1}
```

### `GET /skills/<id>` — full record (program + examples)

```bash
curl -s http://127.0.0.1:8430/skills/1
```

### `GET /library.json` — the registry as a loadable NPCoT library

```bash
curl -s http://127.0.0.1:8430/library.json > library.json
```

This is the payload that plugs into **every NPCoT runtime**: the browser
WASM runtime (`NpcotRuntime::new(library_json)` in `kernels/npcot_wasm`),
the Metal/native executor, and the Python `ArrayProgramLibrary` server all
load this exact format. Format discipline mirrors the canonical emitter:

* pure-v1 registry → v1 format (no `"format"` key, entries carry
  `"program"`) so every existing runtime loads it;
* any v2 skill → `"format": 2` with **all** entries lifted to
  `"program_v2"` (the v1→v2 lift is exact). v1 loaders fail closed on
  format 2 — old runtimes can never silently mis-execute a guarded
  program.

Entry signatures are deterministic unit vectors (dim 8) derived from the
skill id via sha256, so the library is consultable through the standard
similarity lookup out of the box.

### `GET /health`

```bash
curl -s http://127.0.0.1:8430/health   # → {"status": "ok", "skills": 2}
```

## The community flywheel (future work)

The registry is the seed of a shared learning loop, mirroring nsynth's
three cross-run memory banks:

* **Shared solved bank** — `/library.json` already is one: every accepted
  skill becomes instantly reusable by every runtime. A miss in your
  browser can be a hit because someone else solved it last week.
* **Shared bias bank** — accepted gradient-solved skills carry the initial
  parameter vectors that found them; pooling those biases warm-starts
  everyone's synthesis (nsynth's learned-restart bank, federated).
* **Shared negative bank** — verified *rejections* are valuable too:
  publishing "this program does NOT solve these examples" prevents the
  whole community from re-grinding known failures.
* **Registry misses as work items** — unsolved example sets feed the
  continuous autoresearch loop (ROADMAP Rung 5) as a task source: the
  cascade solves them offline and the verified wins land back here.

## Files

| file | role |
|---|---|
| `executor.py` | pure-Python mirror of the canonical v1+v2 executor (ported from `kernels/npcot_wasm/src/lib.rs`) |
| `server.py` | stdlib HTTP server + SQLite storage + `--verify-all` trust sweep |
| `../../tests/registry/test_registry.py` | pinned executor outputs, verification gate, dedupe, format lifting, corruption detection |
