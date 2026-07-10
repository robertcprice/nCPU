# ncpu-synth MCP server — the never-confidently-wrong coding tool

`ncpu-synth` exposes the nsynth verified-synthesis engine to any
[Model Context Protocol](https://modelcontextprotocol.io) client (Claude
Code, Claude Desktop, Cursor, …) as a set of coding tools. Its
differentiator is a single hard property:

> **Every program it returns has been executed against every example you
> gave it and reproduced all of them — or the tool honestly refuses.**
> It never fabricates code that merely "looks right".

You give it input/output examples (or a prompt containing them); it
synthesizes a program, runs that program against all of your examples
*inside the engine*, and only then hands it back marked `verified: true`
with proof metadata (which method solved it, how many examples were
checked). If no program reproduces every example, you get
`verified: false` with a reason and a fallback protocol — never a guess.

The verified program is a small [Mog](./src/mog_transpile.rs) function,
transpiled on the way out to Python / Rust / TypeScript / Go / Java.

```
NL prompt ──parser──▶ I/O examples ──mog_synth──▶ verified Mog ──transpile──▶ your language
                            ▲                          │
       client drafts more examples / code             ▼
              (verify_candidate)         persistent solved / bias / rejected banks
```

The transport is a dependency-free stdio JSON-RPC 2.0 server
(`ncpu/mcp_server/server.py`, MCP revision `2024-11-05`). The heavy
lifting happens in a Rust subprocess — the release binary
`nsynth/target/release/mog_synth`.

> This is the deployment-focused quickstart. For the full protocol notes
> and design rationale see [`../ncpu/mcp_server/README.md`](../ncpu/mcp_server/README.md).

---

## 1. Requirements

- **Python 3.9+** — standard library only, no `pip install` needed.
- **The Rust release binary** `nsynth/target/release/mog_synth` (build
  below). If it is missing, the synthesis tools return an honest `503`
  refusal (`backend binary not found`) rather than crashing.
- **Node.js** (optional) — only needed if you pass `language: "javascript"`
  to `verify_candidate` / `run_program`. The Python sandbox always works
  without it.

## 2. Build the backend

```bash
# from the repo root
bash scripts/build_nsynth_mcp_release.sh
```

This builds exactly the binary the MCP shells out to and prints its path:

```
[nsynth-mcp] release backend built:
  <repo>/nsynth/target/release/mog_synth
```

Equivalent manual build:

```bash
cd nsynth && cargo build --release --bin mog_synth
```

The first build takes a minute or two; afterwards it is incremental. The
Python layer resolves the binary via
`ncpu.synthesis_api.server.default_backend_path()`
(`<repo>/nsynth/target/release/mog_synth`); override with
`python3 -m ncpu.mcp_server --backend /path/to/mog_synth` if you keep it
elsewhere.

## 3. Register it with an MCP client

The server is launched as `python3 -m ncpu.mcp_server`, so the client must
start it with the **repo root as the working directory** (or with the repo
root on `PYTHONPATH`) so the `ncpu` package is importable.

### Claude Code (one-liner)

```bash
claude mcp add ncpu-synth -- python3 -m ncpu.mcp_server
```

### `.mcp.json` (project-scoped, checked into this repo)

```json
{
  "mcpServers": {
    "ncpu-synth": {
      "command": "python3",
      "args": ["-m", "ncpu.mcp_server"]
    }
  }
}
```

Useful optional flags (append to `args`): `--backend <path>`,
`--timeout <seconds>` (default 120, hard max 300), and
`--solved-cache/--bias-bank/--rejected-cache <path>` to point the engine's
persistent memory banks at private paths (`''` disables a bank).

## 4. The six tools

Domain of the **synthesis** tools (1–3): inputs are `int`, `[int]`
(list of ints), or `str`; the expected output is an `int` (a `str`-output
lane also exists when every input is a string). The **sandbox** tools
(5–6) accept arbitrary JSON. Anything outside the synthesis domain is not
a dead end — the refusal tells the client to draft the function itself and
push it through `verify_candidate` (the out-of-domain tier).

Every example below is a real request/response captured from the running
server (the `arguments` object is what you send under
`tools/call → params → arguments`; the result is the JSON inside the
tool's `content[0].text`).

### 1. `synthesize_from_examples` — examples in, proof-carrying code out

Synthesize a program verified against the given examples, transpiled to
`language` (`python` default; also `rust`, `typescript`, `go`, `java`).

```json
{
  "name": "list_sum",
  "examples": [
    {"inputs": [[1, 2, 3]], "expected": 6},
    {"inputs": [[5]],        "expected": 5},
    {"inputs": [[4, 9]],     "expected": 13},
    {"inputs": [[2, 4, 6, 8]], "expected": 20}
  ],
  "language": "python"
}
```

→

```json
{
  "verified": true,
  "method": "combinator",
  "mog": "fn list_sum(xs: [i64]) -> i64 {\n    return array_sum(xs);\n}\n\nfn array_sum(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    for item in arr {\n        total = total + item;\n    }\n    return total;\n}\n",
  "code": "def list_sum(xs: list[int]):\n    return array_sum(xs)\n\ndef array_sum(arr: list[int]):\n    total = 0\n    for item in arr:\n        total = total + item\n    return total",
  "language": "python",
  "examples_checked": 4,
  "elapsed_ms": 36.31
}
```

A refusal (e.g. an unsupported `language`, or no program found) instead
returns `{"verified": false, "reason": "...", "guidance": "...",
"examples": [...]}` and carries **no** `code` field — refusals never ship
code.

### 2. `synthesize_from_prompt` — NL prompt → extracted examples → verified code

Extracts I/O pairs from a prompt (arrow notation `f(2,3) -> 5`, doctests
`>>> f(2,3)`, `assert f(2,3) == 5`, "returns" prose), echoes them back,
then runs tool 1.

```json
{
  "prompt": "write add(a,b): add(2,3) -> 5, add(10,20) -> 30, add(0,0) -> 0",
  "language": "python"
}
```

→

```json
{
  "verified": true,
  "method": "universal",
  "mog": "fn add(a0: i64, a1: i64) -> i64 {\n    return (a0 + a1);\n}\n",
  "code": "def add(a0: int, a1: int):\n    return (a0 + a1)",
  "language": "python",
  "examples_checked": 3,
  "function_name": "add",
  "extracted_examples": [
    {"args": [2, 3], "kwargs": {}, "expected": 5},
    {"args": [10, 20], "kwargs": {}, "expected": 30},
    {"args": [0, 0], "kwargs": {}, "expected": 0}
  ],
  "extraction_sources": {"arrow": 3}
}
```

No examples in the prompt → `{"verified": false, "reason": "no I/O
examples found", "guidance": "Provide concrete input/output examples like:
f(2,3) -> 5. …"}`. Supply examples, not a wish.

### 3. `consult_library` — instant answer from the persistent solved bank

Exact-fingerprint lookup against the on-disk solved cache. A hit returns
the already-verified program without running the solver; a miss returns
`hit: false` with the computed fingerprint.

```json
{ "examples": [{"inputs": [123456, 7], "expected": 999999}] }
```

→

```json
{ "hit": false, "fingerprint": "i:123456|i:7~999999" }
```

A hit adds `method`, `mog`, `success_count`, and `last_used_at`. (The bank
is populated across sessions as the engine solves problems; treat a hit as
a cache win and a miss as "run tool 1".)

### 4. `library_stats` — observable cross-session learning state

Sizes of the engine's three persistent memory banks. No arguments.

```json
{}
```

→

```json
{
  "solved_entries": 6470,
  "bias_entries": 190,
  "rejected_rows": 109,
  "rejected_hashes": 136329
}
```

### 5. `verify_candidate` — run client-drafted code through the same gate

The out-of-domain tier: when synthesis refuses, draft the function
yourself and submit it here with the **same examples**. The code is
executed against every example in a locked-down subprocess (`python3 -I`,
scrubbed env, temp dir, hard timeout with process-tree kill). Only
`verified: true` code should be shown to the user. Unlike tools 1–3, any
JSON values are allowed for `inputs`/`expected`.

Passing candidate:

```json
{
  "name": "list_sum",
  "code": "def list_sum(xs):\n    return sum(xs)\n",
  "examples": [
    {"inputs": [[1, 2, 3]], "expected": 6},
    {"inputs": [[9, 9]],    "expected": 18}
  ]
}
```

→ `{ "verified": true, "examples_checked": 2, "language": "python" }`

Failing candidate (`return xs[0]`) reports the first counterexample:

```json
{
  "verified": false,
  "first_failure": {"example_index": 0, "inputs": [[1, 2, 3]], "expected": 6, "got": 1},
  "failures": 2,
  "examples_checked": 2,
  "language": "python"
}
```

Set `language: "javascript"` to verify JS (requires Node on the host).

### 6. `run_program` — execute a verified program on new inputs

Closes the loop NL → code → actual output. Runs `name(*inputs)` once in the
same sandbox as tool 5. Single call:

```json
{ "name": "list_sum", "code": "def list_sum(xs):\n    return sum(xs)\n", "inputs": [[10, 20, 30]] }
```

→ `{ "ok": true, "output": 60, "language": "python" }`

Batch (`batch: true` → `inputs` is a list of argument-lists, one result per
call):

```json
{
  "name": "add",
  "code": "def add(a, b):\n    return a + b\n",
  "inputs": [[1, 2], [3, 4], [10, 20]],
  "batch": true
}
```

→

```json
{
  "ok": true,
  "outputs": [{"ok": true, "output": 3}, {"ok": true, "output": 7}, {"ok": true, "output": 30}],
  "language": "python"
}
```

A runtime error is reported honestly as `{"ok": false, "error": "..."}`.

---

## 5. The honest-refusal contract (why this tool is different)

Every answer this server gives is one of:

1. **`verified: true`** — a program that reproduced *every* example, either
   inside the synthesizer (tools 1–3) or inside the subprocess sandbox
   (tool 5). It carries proof metadata (`method`, `examples_checked`).
2. **`verified: false`** — an honest refusal with a `reason`, and (for
   synthesis) the `guidance` protocol plus your `examples` echoed back in
   `verify_candidate`-ready form, so the client can draft the code and push
   it through the same verification gate.

There is no third outcome. The server never emits unverified code labelled
as an answer — that is the "never confidently wrong" guarantee, and it is
what makes the tool safe to wire into an autonomous coding loop.

## 6. Verifying your install

Quick smoke test (from the repo root, after building the binary):

```bash
printf '%s\n%s\n%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"library_stats","arguments":{}}}' \
  | python3 -m ncpu.mcp_server
```

You should see the `initialize` result followed by a `library_stats`
payload. The full test suite lives at `tests/mcp_server/` and runs with
`python3 -m pytest tests/mcp_server/` (it skips automatically if the
release binary has not been built).
