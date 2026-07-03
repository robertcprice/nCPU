#!/usr/bin/env bash
# Gate G5 nightly — full NL fixture repair corpus + synthesis proposer suite.
# Run from repo root. Expect several minutes (release build + 17 workflow fixtures).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/nsynth"

echo "[g5-nightly] backend intake suite"
cargo test --release --lib backend_ -- --test-threads=1

echo "[g5-nightly] G5 gate + coding intent + synthesis proposer"
cargo test --release --lib g5_gate -- --test-threads=1
cargo test --release --lib coding_intent -- --test-threads=1
cargo test --release --lib synthesis_proposer -- --test-threads=1

echo "[g5-nightly] full 17-fixture workflow corpus (ignored test)"
cargo test --release --lib workflow_runner_executes_nl_fixture_suite -- --test-threads=1 --ignored

echo "[g5-nightly] OK"
