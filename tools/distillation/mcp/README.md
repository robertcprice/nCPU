# nsynth MCP Server

A stdio JSON-RPC MCP server that exposes execution-based verification
and a persistent verified-code cache as tools any MCP-aware LLM can
call during a coding session.

## What you get

Ten tools:

| tool | purpose |
|------|---------|
| `execute_python(code, timeout_s=5)` | Run code in a fresh namespace, return stdout/stderr/status/exceptions. Sanity-check generated code before committing. |
| `verify_against_tests(code, entry_point, test_cases)` | Run code and call `entry_point` against each `[args..., expected]` tuple. Returns per-case pass/fail with specific failing args — feed it back to the LLM for a targeted fix. |
| `fingerprint(examples)` | Compute a deterministic cache key for a list of I/O examples. |
| `cache_solution(fingerprint, code, model, examples?)` | Persist a verified solution. Optional `examples` list persists I/O pairs alongside the code — required for downstream semantic retrieval to work. |
| `lookup_solution(fingerprint)` | 0 ms answer to "have we solved this shape before?" Returns `status: hit` with code or `status: miss`. |
| `semantic_similar(examples, k, min_similarity)` | Retrieve cached solutions with similar (not identical) I/O shape. Returns raw matches with similarity scores. Caller should re-verify returned code. |
| `build_retrieval_prefix(examples, k, min_similarity)` | Return a ready-to-paste few-shot prefix from top-K similar cached solutions. Splice directly into your next prompt to condition generation on known-good solutions. |
| `evaluate_expression(expression)` | Evaluate a Python arithmetic expression in a restricted sandbox (math module + safe builtins). Used by reasoning agents to sanity-check arithmetic. |
| `check_numeric_answer(predicted, ground_truth, tolerance)` | Compare a predicted number to ground truth with tolerance. Returns match bool + abs/rel error. Verification primitive for math-reasoning loops. |
| `delegate_to_frontier(prompt, entry_point, test_cases, cheap_model, premium_model)` | Cheap→premium model cascade. LLM decides when to escalate a hard problem. |

All tools share one TSV at `~/.nsynth_llm_solutions.tsv` (override with
`NSYNTH_LLM_CACHE_PATH`). Multiple sessions + multiple LLMs see the same
cache — cross-session collective memory for verified code.

## Wiring into Claude Desktop / Cursor / Claude Code

Add to your MCP config (usually `~/Library/Application
Support/Claude/claude_desktop_config.json` on macOS, or
`~/.config/cursor/mcp.json` for Cursor, or the `mcpServers` block in
your `~/.claude/settings.json` for Claude Code):

```json
{
  "mcpServers": {
    "nsynth": {
      "command": "python3",
      "args": ["/ABSOLUTE/PATH/TO/nCPU/tools/mcp/nsynth_mcp_server.py"],
      "env": {
        "NSYNTH_LLM_CACHE_PATH": "/tmp/my_team_cache.tsv"
      }
    }
  }
}
```

Substitute the absolute path. Restart your IDE. The tools show up in
the LLM's tool list and it can call them during any coding
conversation.

## Verify the install

### Manually via stdio

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  | python3 tools/mcp/nsynth_mcp_server.py
# → {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05",...}}

echo '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  | python3 tools/mcp/nsynth_mcp_server.py
# → {"tools":[{"name":"execute_python",...},{"name":"verify_against_tests",...}]}
```

### End-to-end tool call

```bash
cat <<EOF | python3 tools/mcp/nsynth_mcp_server.py
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"execute_python","arguments":{"code":"print(2+2)"}}}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"verify_against_tests","arguments":{"code":"def double(x: int) -> int:\n    return x * 2","entry_point":"double","test_cases":[[3,6],[5,10],[0,0]]}}}
EOF
```

Expected: three JSON-RPC responses. The third's `result.content[0].text`
contains `{"status":"ok","passed":3,"total":3,...}`.

## Session walk — how an LLM actually uses these tools

In a real Claude Code session, the LLM typically uses these tools like
this (paraphrased from the tool-call trace):

1. **User**: "Write a Python function `abs_diff(a, b)` that returns
   the absolute difference."
2. **Claude**: thinks → drafts candidate → calls
   `verify_against_tests(code="def abs_diff(a, b): return abs(a - b)",
   entry_point="abs_diff", test_cases=[[5, 3, 2], [3, 5, 2], [0, 0, 0]])`.
3. **Tool response**: `{"status":"ok","passed":3,"total":3}`.
4. **Claude**: thinks → "verified. Let me also cache this." → calls
   `fingerprint` to get the key, then `cache_solution`.
5. **Claude**: replies to user with the verified code.

Later, in a different session, the user asks the same thing:
1. **Claude**: recognises the shape → calls `fingerprint`,
   then `lookup_solution`.
2. **Tool response**: `{"status":"hit","code":"def abs_diff(...)","model":"claude-haiku-..."}`.
3. **Claude**: returns the cached code immediately. No LLM inference,
   no API round-trip.

This is the "cross-session collective memory" pattern. Every verified
solve populates a shared store. Every subsequent ask can hit it.

## Scoped for production

- **Atomic writes**: cache uses temp-file-then-rename so concurrent
  Claude sessions don't corrupt each other's writes.
- **Deduplication**: same (fingerprint, model, code) triple increments a
  success counter instead of creating duplicates.
- **Timeout guards**: every `execute_python` / `verify_against_tests`
  call runs under a SIGALRM timeout so pathological code can't hang the
  server.
- **Zero external deps**: Python stdlib + the existing `llm_solution_cache`
  module. Drop in anywhere with Python 3.10+.

## Troubleshooting

- **"method not found"**: the LLM client sent an unsupported MCP
  method (e.g. `ping`). The server returns a proper JSON-RPC error and
  keeps running. Upgrade your client or ignore.
- **Tool calls produce `isError: true`**: the tool's implementation
  threw. Check the `text` field in `content[0]` for the traceback.
- **Cache seems empty**: verify `NSYNTH_LLM_CACHE_PATH` isn't pointed
  at a temp file that got cleaned up. Run
  `python3 tools/benchmarks/llm_solution_cache.py --list` with the
  same env to confirm.
