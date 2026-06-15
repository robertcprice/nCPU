# Go client for the nCPU synthesis API

A minimal Go client that posts problems to the synthesis server and
prints the recovered Mog program transpiled to Go. With `--run`, it
also compiles and executes the emitted Go via `go run`.

## Quickstart

In one terminal, start the synthesis server:

```bash
python3 ncpu/synthesis_api/server.py
```

In another terminal, run the Go client:

```bash
go run examples/go_client/main.go
```

To execute the emitted Go on a fresh input after each solve:

```bash
go run examples/go_client/main.go --run
```

## What it shows

For each demo, the client:

1. POSTs a problem-json (signature + examples) to `/synthesize`
2. Prints the server's response (method, success, error if any)
3. Prints the **emitted Go** program from the server's transpiled dict
4. With `--run`: writes the emitted Go to a temp file, calls
   `go run` on it with a small wrapper main, and prints the runtime
   result

## Demos

The client runs four demos that exercise different search teachers:

| Demo | Teacher exercised |
|---|---|
| `strictly_increasing` | `search_strictly_increasing` |
| `first_index_of` | `search_first_index_of` |
| `count_distinct` | `search_count_distinct` |
| `is_anagram` | `search_is_anagram` |

## Wiring

The synthesis server's `transpiled.go` field carries the Go output.
The Python server (`ncpu/synthesis_api/server.py`) calls
`mog_synth.mog_transpile.to_go` to produce it. The Go client just
reads the JSON field — no Go-side transpiler needed.

## Requirements

- Go 1.21+ (for the `go run` invocation in `--run` mode)
- A running `ncpu/synthesis_api/server.py` (the Python server is the
  ground-truth transpiler; mog_synth itself ships the binary the
  server wraps)
