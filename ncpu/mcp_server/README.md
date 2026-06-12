# ncpu-synth — natural language → verified program (MCP server)

The synthesis cascade as a coding tool: **I/O examples in, proof-carrying
code out**. A natural-language request is converted to input/output
examples, the nsynth solver portfolio synthesizes a program that is
*executed against every example inside the synthesizer*, and the answer
ships with its proof metadata (method, examples checked) transpiled into
Python, Rust, or TypeScript. When no program reproduces the examples, the
server says so — it never fabricates code.

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
(supply more examples, fall back to the LLM tier).

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
protocol handshake, all four tools, the end-to-end "NL prompt → verified
Python function" path, the honest-refusal case, the no-examples guidance
path, the Python↔Rust fingerprint cross-check, and the prompt_parser
list-args regression.
