#!/usr/bin/env bash
# Build release binaries for nsynth-agent MCP (set NSYNTH_USE_RELEASE=1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/nsynth"
cargo build --release --bin coding_agent --bin build_backend_nl
echo "[nsynth-mcp] release binaries:"
echo "  $ROOT/nsynth/target/release/coding_agent"
echo "  $ROOT/nsynth/target/release/build_backend_nl"
echo "Set NSYNTH_USE_RELEASE=1 in MCP config for fast cold start."
