# Synthesis API — nsynth behind HTTP (cascade Rung 3)

A stdlib-only HTTP server (`server.py`) that puts the nsynth Rust
synthesizer behind a JSON API. Zero third-party dependencies — plain
`http.server`, in the same style as `ncpu/self_optimizing/npcot_server.py`.

Each request shells out to the release binary
(`nsynth/target/release/mog_synth --problem-json -`), which synthesizes a
Mog program that reproduces the supplied input/output examples, verifies
it, and returns it. Successful programs are additionally transpiled to
Python, Rust, and TypeScript via the binary's `--transpile` flag.

**Refusals are honest.** When the synthesizer cannot find a program, the
response is `success: false` with the backend's error — the server never
fabricates code. That property is what makes this a trustworthy middle
tier of the cascade:

```
browser WASM tier (instant, tiny)        — refuses anything non-trivial
        │ refusal
        ▼
this endpoint (heavy nsynth synthesizer) — solves or refuses, verified
        │ refusal
        ▼
LLM tier (expensive, unverified)         — last resort
```

A WASM-tier refusal escalates here; a refusal here escalates to the LLM
tier. Anything answered at this tier is *verified against the examples*,
not generated text.

## Running

```bash
# Build the backend once (if not already present):
cd nsynth && cargo build --release && cd ..

# Start the server (defaults: 127.0.0.1:8093, repo-relative backend):
python3 -m ncpu.synthesis_api.server --port 8093
```

Flags:

| flag | default | meaning |
|---|---|---|
| `--backend PATH` | `<repo>/nsynth/target/release/mog_synth` | synthesizer binary |
| `--host` / `--port` | `127.0.0.1` / `8093` | bind address |
| `--timeout SECONDS` | `120` | default per-request solver timeout (hard cap 300) |
| `--solved-cache PATH` | inherit env / `$HOME` | override `NSYNTH_CACHE_PATH` |
| `--bias-bank PATH` | inherit env / `$HOME` | override `NSYNTH_BIAS_BANK_PATH` |
| `--rejected-cache PATH` | inherit env / `$HOME` | override `NSYNTH_REJECTED_PATH` |

## Endpoints

### `POST /synthesize`

Request body:

```json
{
  "name": "add_two",
  "signature": "fn add_two(a: i64, b: i64) -> i64",
  "examples": [
    {"inputs": [1, 2], "expected": 3},
    {"inputs": [5, 7], "expected": 12},
    {"inputs": [0, 0], "expected": 0}
  ],
  "holdouts": [{"inputs": [10, 20], "expected": 30}],
  "timeout_s": 60
}
```

- `name` (required): problem name. Used for the generated function name
  when no `signature` is given.
- `examples` (required, non-empty): each `inputs` entry is an i64, an
  array of i64, or a string; `expected` is an i64.
- `signature` (optional): Mog signature; auto-built from `name` and the
  first example's input types when omitted.
- `holdouts` (optional): extra examples the solution must also satisfy.
- `timeout_s` (optional): per-request solver budget, capped at 300.

```bash
curl -X POST http://localhost:8093/synthesize \
     -H 'Content-Type: application/json' \
     -d '{"name": "add_two", "examples": [
           {"inputs": [1, 2], "expected": 3},
           {"inputs": [5, 7], "expected": 12},
           {"inputs": [0, 0], "expected": 0}]}'
```

Response (200):

```json
{
  "success": true,
  "method": "search_scalar_expr",
  "code": "fn add_two(a: i64, b: i64) -> i64 {\n    return (a + b);\n}\n",
  "error": null,
  "transpiled": {
    "python": "def add_two(a: int, b: int):\n    return (a + b)",
    "rust": "fn add_two(a: i64, b: i64) -> i64 {\n    return (a + b);\n}",
    "typescript": "function add_two(a: number, b: number): number {\n    return (a + b);\n}"
  },
  "elapsed_ms": 24.1
}
```

Refusal (still 200 — a refusal is a valid answer, not a server error):

```json
{
  "success": false,
  "method": "diff_gradient_unsupported",
  "code": null,
  "error": "differentiable solver currently supports scalar numeric problems only",
  "transpiled": null,
  "elapsed_ms": 8.3
}
```

A solver that exceeds the request budget is also an honest refusal:
`{"success": false, "error": "timeout", ...}`.

Malformed input (bad JSON, missing/ill-typed fields, out-of-i64-range
numbers) returns **400** with `{"error": "<what is wrong>"}` — never 500.

### `GET /health`

```json
{"status": "ok", "backend": "/path/to/mog_synth", "backend_present": true}
```

### `GET /stats`

Sizes of the backend's three persistent memory banks (missing file → 0):

```json
{"solved_entries": 111, "bias_entries": 19, "rejected_rows": 4, "rejected_hashes": 57}
```

## Caching & memory banks

The Rust backend carries persistent cross-run memory in three files,
which is why repeat requests for the same examples come back in
milliseconds (the solved cache is keyed by an examples fingerprint):

| bank | default path | env var | format |
|---|---|---|---|
| solved programs | `~/.nsynth_solved_programs.json` | `NSYNTH_CACHE_PATH` | one record per line |
| learned biases | `~/.nsynth_learned_biases.jsonl` | `NSYNTH_BIAS_BANK_PATH` | JSONL |
| rejected programs | `~/.nsynth_rejected_programs.tsv` | `NSYNTH_REJECTED_PATH` | TSV `ts\thashes\tfp` |

Setting an env var (or the corresponding server flag) to the empty
string disables that bank. Tests point all three at tmp paths for full
isolation — see `tests/synthesis_api/test_server.py`.

## Embedding

The request handler is a plain function, usable without the HTTP layer:

```python
from ncpu.synthesis_api import SynthConfig, handle_synthesize_request

status, body = handle_synthesize_request(
    {"name": "add_two", "examples": [{"inputs": [1, 2], "expected": 3}]},
    SynthConfig(),
)
```

## Tests

```bash
python3 -m pytest tests/synthesis_api/ -v
```

Spawns a live server on a free port with isolated banks and checks:
health, easy solve + transpiles, solved-cache hit (< 1 s repeat),
honest refusal on a patternless mapping, timeout-as-refusal, and
400-on-malformed-input (13 bad-body variants).
