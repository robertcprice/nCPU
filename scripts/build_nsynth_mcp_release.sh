#!/usr/bin/env bash
# Build the release binary the ncpu-synth MCP server shells out to.
#
# The MCP server (`python3 -m ncpu.mcp_server`) and the HTTP synthesis API
# (`python3 -m ncpu.synthesis_api.server`) both resolve their backend via
# `ncpu.synthesis_api.server.default_backend_path()`, which points at
#   <repo>/nsynth/target/release/mog_synth
# (`mog_synth` is the default binary of the `mog_synth` crate — src/main.rs).
# If this binary is missing the tools return an honest 503 refusal
# ("backend binary not found"), so building it is mandatory before use.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/nsynth"
cargo build --release --bin mog_synth
echo "[nsynth-mcp] release backend built:"
echo "  $ROOT/nsynth/target/release/mog_synth"
echo "Register the server with:  claude mcp add ncpu-synth -- python3 -m ncpu.mcp_server"
