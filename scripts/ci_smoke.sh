#!/usr/bin/env bash
# CI smoke for the Python side of the nCPU repo.
#
# Runs every targeted test suite that gates nCPU's robustness work:
#   - prompt parser + its list/dict regression tests
#   - LFS coverage guard (fails CI if a >50MB JSONL slips through)
#   - synthesis API + registry + MCP server end-to-end suites
#   - the new registry close-the-loop + mine-registry CLI tests
#   - the synthesis API refusal source
#
# Run from the repo root. Exits non-zero on any failure.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTEST=(python3 -m pytest)

${PYTEST[@]} -q \
    tests/autoresearch/test_prompt_parser.py \
    tests/autoresearch/test_registry_source.py \
    tests/autoresearch/test_synthesis_api_source.py \
    tests/autoresearch/test_cli_registry.py \
    tests/test_lfs_gitattributes.py \
    tests/registry/test_registry.py \
    tests/registry/test_registry_misses_loop.py \
    tests/synthesis_api/test_server.py \
    tests/synthesis_api/test_prompt_endpoint.py \
    tests/mcp_server/test_mcp_server.py

# nsynth new-language surface (bool, struct, float, Go/Java transpile,
# stateful benchmarks) — Rust unit tests. One positional filter per
# invocation; the script runs the relevant test groups separately.
(cd nsynth && cargo test --release --lib --quiet \
    runtime::tests::verifies_bool 2>&1 | tail -5)
(cd nsynth && cargo test --release --lib --quiet \
    runtime::tests::verifies_float 2>&1 | tail -5)
(cd nsynth && cargo test --release --lib --quiet \
    mog_transpile:: 2>&1 | tail -5)
(cd nsynth && cargo test --release --lib --quiet \
    routing_cases::new_teacher_preemption_cases:: 2>&1 | tail -5)

# nsynth backend NL intake + MCP release binaries (LOOP-12/13)
(cd nsynth && cargo test --release --lib --quiet backend_ 2>&1 | tail -8)
bash scripts/build_nsynth_mcp_release.sh 2>&1 | tail -5

# Gate G5 repair loop CI subset (add + triple + square holdouts)
(cd nsynth && cargo test --release --lib --quiet workflow_runner_executes_nl_fixture_ci_subset 2>&1 | tail -8)
