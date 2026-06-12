# ncpu-synth — natural language → verified program (MCP server)

The synthesis cascade as a coding tool: **I/O examples in, proof-carrying
code out**. A natural-language request is converted to input/output
examples, the nsynth solver portfolio synthesizes a program that is
*executed against every example inside the synthesizer*, and the answer
ships with its proof metadata (method, examples checked) transpiled into
Python, Rust, or TypeScript. When no program reproduces the examples, the
server says so — it never fabricates code. Out-of-domain requests are not
dead ends: the refusal carries the protocol for the client to draft the
code itself and admit it through the **same example-verification gate**
(`verify_candidate`), and `run_program` executes verified code on new
inputs — NL → code → actual program output.

Delivery vehicle: a stdio [Model Context Protocol](https://modelcontextprotocol.io)
server (2024-11-05 revision), implemented with **zero dependencies beyond
the Python stdlib** (newline-delimited JSON-RPC 2.0 on stdin/stdout). It
plugs into Claude Code, Claude Desktop, Cursor, and any MCP client.

```
NL prompt ──prompt_parser──▶ I/O examples ──mog_synth──▶ verified Mog ──transpile──▶ your language
                                  ▲                            │
                                  │                            ▼
                       client supplies more examples   three persistent memory banks
```

## Requirements

- Python 3.9+ (stdlib only — no `pip install` needed).
- The nsynth release binary: `cd nsynth && cargo build --release`
  (expected at `nsynth/target/release/mog_synth`, override with `--backend`).

## Client setup

### Claude Code

```bash
claude mcp add ncpu-synth -- python3 -m ncpu.mcp_server
```

Run from the repo root (or add `--directory`-style cwd handling via your
shell). Then ask Claude: *"use ncpu-synth to write double_plus_one given
double_plus_one(3) -> 7, double_plus_one(0) -> 1, double_plus_one(10) -> 21"*.

### Generic `.mcp.json`

```json
{
  "mcpServers": {
    "ncpu-synth": {
      "command": "python3",
      "args": ["-m", "ncpu.mcp_server"],
      "cwd": "/path/to/nCPU"
    }
  }
}
```

Useful flags (all optional): `--backend PATH`, `--timeout SECONDS`
(default 120, max 300), `--solved-cache PATH`, `--bias-bank PATH`,
`--rejected-cache PATH`, `--log-level LEVEL`. Logs go to stderr; stdout
is the protocol channel.

## Tools

### 1. `synthesize_from_examples(name, examples, language="python", timeout_s?)`

Examples in, verified code out.

```json
{
  "name": "add_two",
  "examples": [
    {"inputs": [1, 2], "expected": 3},
    {"inputs": [5, 7], "expected": 12},
    {"inputs": [0, 0], "expected": 0}
  ],
  "language": "python"
}
```

Success:

```json
{
  "verified": true,
  "method": "search_scalar_expr",
  "mog": "fn add_two(a: i64, b: i64) -> i64 {\n    return (a + b);\n}\n",
  "code": "def add_two(a: int, b: int):\n    return (a + b)",
  "language": "python",
  "examples_checked": 3,
  "elapsed_ms": 41.2
}
```

Input values are ints, lists of ints, or strings; `expected` is always an
int (the solver's output domain is i64). `language` is one of `python`,
`rust`, `typescript`.

### 2. `synthesize_from_prompt(prompt, language="python", timeout_s?)`

Natural language in. The parser mines I/O pairs from arrow notation
(`f(2,3) -> 5`), doctests (`>>> f(2,3)` / `5`), asserts
(`assert f(2,3) == 5`), and "returns" prose, echoes the extracted
examples back (`extracted_examples`, `function_name`,
`extraction_sources`), and proceeds to tool 1.

When the prompt contains **no** examples, the response closes the loop
instead of guessing:

```json
{
  "verified": false,
  "reason": "no I/O examples found",
  "guidance": "Provide concrete input/output examples like: f(2,3) -> 5. ..."
}
```

The conversational client relays that to the user — the human in the
loop never writes code, only examples.

Every other refusal (synthesis exhausted, unrepresentable examples,
timeout) carries the full out-of-domain protocol plus the examples in
`verify_candidate`-ready form, so the client can act without re-parsing:

```json
{
  "verified": false,
  "reason": "no program found",
  "guidance": "synthesis refused; draft the function yourself and submit it through verify_candidate with these same examples; only verified code should be shown to the user. ...",
  "examples": [{"inputs": ["hello"], "expected": "olleh"}, ...]
}
```

### 3. `consult_library(examples)`

Instant answer when these exact examples were already solved in any past
session. Computes the same deterministic examples-fingerprint the Rust
solver uses for its solved bank and returns the cached verified program
on a hit (`{"hit": true, "method", "mog", "fingerprint", ...}`), or
`{"hit": false, "fingerprint"}` on a miss — no subprocess, no search.

### 4. `library_stats()`

```json
{"solved_entries": 132, "bias_entries": 47, "rejected_rows": 12, "rejected_hashes": 880}
```

### 5. `verify_candidate(name, code, examples, language="python", timeout_s=10)`

The out-of-domain gate. The client drafts the function; the server
executes it against **every** example in a subprocess sandbox and only
then calls it verified:

```json
{
  "name": "reverse_string",
  "code": "def reverse_string(s):\n    return s[::-1]\n",
  "examples": [
    {"inputs": ["hello"], "expected": "olleh"},
    {"inputs": ["abc"], "expected": "cba"}
  ]
}
```

All examples reproduce → `{"verified": true, "examples_checked": 2}`.
Any mismatch → `{"verified": false, "failures": n, "first_failure":
{"example_index", "inputs", "expected", "got"}}` (or `"error"` instead of
`"got"` when that example raised). A crash, syntax error, missing
function, or timeout → `{"verified": false, "error": "..."}` with the
captured detail. Results are compared with `==`, plus a
`math.isclose(rel_tol=1e-6)` fallback for floats. Unlike the synthesis
tools, inputs and expected values may be **any JSON values** — this tier
exists precisely for what the solver domain can't express.

`language` is `python` (always available, run via `python3 -I`) or
`javascript` (run via `node`; if node is not installed the tool returns
a clean unsupported message rather than failing).

### 6. `run_program(name, code, inputs, language="python", timeout_s=10, batch=false)`

Execute a verified program on new inputs in the same sandbox: calls
`name(*inputs)` once and returns `{"ok": true, "output": ...}` or
`{"ok": false, "error": "..."}`. With `batch: true`, `inputs` is a list
of argument lists and the result is `{"ok": true, "outputs": [{"ok",
"output"|"error"}, ...]}` — one entry per call, errors isolated per call.

## The full cascade

Two tiers, one proof standard — code reaches the user only after it has
reproduced every example in an executed check:

- **Tier 1 — verified synthesis** (`synthesize_from_examples`,
  `synthesize_from_prompt`, `consult_library`): machine-discovered,
  proof-carrying. The nsynth portfolio searches for a program and
  verifies it against every example inside the synthesizer. Strongest
  guarantee, limited domain (int / [int] / str inputs, int outputs).
- **Tier 2 — client-LLM draft + `verify_candidate` gate**: out-of-domain
  requests are refused with the protocol and the echoed examples; the
  MCP client (itself a capable code generator) drafts the function and
  submits it through the same example-verification gate. Same proof
  shape: `verified: true` means *executed and reproduced every example*,
  never "looks right".
- **`run_program`** then turns verified code into actual outputs on new
  inputs, completing NL → code → program output.

```
Client (Claude / Cursor)                    ncpu-synth server
        |                                          |
        |--- synthesize_from_examples(ex) -------->|   tier 1: solver search
        |                                          |   + in-synthesizer proof
        |<-- verified:true + code  ... DONE -------|
        |        ... or ...                        |
        |<-- verified:false + guidance + ex -------|   honest refusal
        |                                          |
   [client drafts candidate code itself]           |
        |                                          |
        |--- verify_candidate(name, code, ex) ---->|   tier 2: sandbox runs
        |                                          |   code on EVERY example
        |<-- verified:true (or first_failure) -----|
        |                                          |
        |--- run_program(name, code, new_inputs) ->|   execution on demand
        |<-- ok:true + output ---------------------|
        |                                          |
   [only verified code is shown to the user]
```

### Sandbox trust model — stated plainly

`verify_candidate` and `run_program` **execute client-written code**.
There is deliberately no pattern-matching blocklist of "dangerous" code:
naive blocklists are trivially bypassed and create false confidence. The
control is the sandbox itself:

- fresh process per call — `python3 -I` (isolated mode: no user site,
  `PYTHON*` env vars ignored, script dir off `sys.path`) or `node`;
- scrubbed environment: minimal `PATH`, no `HOME`, nothing inherited;
- cwd is a fresh temporary directory, deleted after the run;
- hard wall-clock timeout (default 10 s, max 60 s) — on expiry the whole
  process tree is killed (own session + `killpg`).

This runs with **equivalent trust to any local coding agent executing
the code it just wrote** — no more, no less. It is not an OS-level jail:
do not feed it code you would not let your coding agent run locally.

## The honest-refusal contract

Every synthesis result is one of exactly two shapes:

- `verified: true` — the returned program **reproduced every provided
  example inside the synthesizer** before being returned. The `mog`
  field is the verified source of truth; `code` is its transpilation.
- `verified: false` + `reason` — no program found (search exhausted,
  unsupported shape, or `timeout`). **No code field, ever.**

This is the property that makes the tool composable with LLM clients:
an answer from `ncpu-synth` never needs review for hallucination — it is
either proof-carrying or an explicit refusal the client can act on
(supply more examples, or follow the tier-2 protocol: draft the code and
push it through `verify_candidate`). The same contract holds in tier 2:
`verify_candidate` says `verified: true` only after the sandbox executed
the candidate against every example.

## How it learns across sessions

The Rust backend maintains three persistent memory banks (defaults under
`~/`, each overridable by env var):

| Bank | Default path | Env var | Role |
|------|--------------|---------|------|
| Solved programs | `~/.nsynth_solved_programs.json` | `NSYNTH_CACHE_PATH` | Examples-fingerprint → verified program. Every successful solve is recorded; repeat requests return in milliseconds (`consult_library` reads this bank directly). |
| Learned biases | `~/.nsynth_learned_biases.jsonl` | `NSYNTH_BIAS_BANK_PATH` | Gradient-solver initializations that led to past successes, replayed first on new problems — cold 76s solves become ~29ms warm. |
| Rejected programs | `~/.nsynth_rejected_programs.tsv` | `NSYNTH_REJECTED_PATH` | Hashes of candidate programs already disproven for a fingerprint, so retries never re-verify dead ends. |

`library_stats()` makes the learning observable: solve something new,
call it again, and watch `solved_entries` grow. Set any env var to the
empty string to disable that bank (useful for tests — the pytest suite
isolates all of them plus the method-router state under a tmp dir).

## Tests

```bash
python3 -m pytest tests/mcp_server/ -v
```

The suite spawns the real server subprocess over stdio and covers the
protocol handshake, all six tools, the end-to-end "NL prompt → verified
Python function" path, the honest-refusal case (now including the
protocol guidance + echoed examples), the no-examples guidance path, the
Python↔Rust fingerprint cross-check, the prompt_parser list-args
regression, and the sandbox tier: correct/subtly-wrong/raising
candidates, timeout enforcement (process-tree kill, no hang), batch
`run_program`, the JavaScript path (clean skip without node), and one
full cascade walk — synthesis refuses → client drafts → `verify_candidate`
verifies → `run_program` produces output.
