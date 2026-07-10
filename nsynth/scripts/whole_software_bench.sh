#!/usr/bin/env bash
# WP0 — whole-software product-path bench (scaffold + fill via coding_agent).
#
# Tiers:
#   A  schema-CRUD (model-free) — must RESOLVE via product path
#   B  must-refuse (out of reach without model / non-schema) — must NOT succeed
#   C  gated-spec (needs NSYNTH_LOCAL_LLM_URL) — SKIP if unset
#
# Usage:
#   ./scripts/whole_software_bench.sh
#   NSYNTH_LOCAL_LLM_URL=http://127.0.0.1:8080/v1/chat/completions ./scripts/whole_software_bench.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT}/target/release/coding_agent"
if [[ ! -x "$BIN" ]]; then
  BIN="${ROOT}/target/debug/coding_agent"
fi
if [[ ! -x "$BIN" ]]; then
  echo "building coding_agent…"
  (cd "$ROOT" && cargo build --bin coding_agent)
  BIN="${ROOT}/target/debug/coding_agent"
fi

WORKDIR="${TMPDIR:-/tmp}/nsynth_ws_bench_$$"
mkdir -p "$WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT

pass=0
fail=0
skip=0

json_success() {
  # coding_agent --json prints a single pretty-printed object; extract "success".
  python3 -c '
import json, sys
raw = sys.stdin.read()
# Prefer the last JSON object in the stream.
start = raw.rfind("{")
if start < 0:
    print("false"); raise SystemExit
try:
    d = json.loads(raw[start:])
except Exception:
    print("false"); raise SystemExit
print("true" if d.get("success") else "false")
'
}

run_case() {
  local tier="$1" id="$2" expect="$3" query="$4"
  local dir="$WORKDIR/$id"
  mkdir -p "$dir"
  local out rc=0
  set +e
  out="$("$BIN" --root "$dir" --json query "$query" 2>&1)"
  rc=$?
  set -e
  local success
  success="$(printf '%s' "$out" | json_success)"

  if [[ "$expect" == "SKIP" ]]; then
    echo "SKIP  [$tier] $id"
    skip=$((skip + 1))
    return
  fi
  if [[ "$expect" == "RESOLVE" && "$success" == "true" ]]; then
    echo "PASS  [$tier] $id"
    pass=$((pass + 1))
  elif [[ "$expect" == "REFUSE" && "$success" == "false" ]]; then
    echo "PASS  [$tier] $id (honest refuse)"
    pass=$((pass + 1))
  else
    echo "FAIL  [$tier] $id expect=$expect success=$success rc=$rc"
    echo "      $(printf '%s' "$out" | tail -c 400 | tr '\n' ' ')"
    fail=$((fail + 1))
  fi
}

echo "=== WP0 whole-software bench (product path) ==="
echo "binary: $BIN"
echo

# Tier A — schema CRUD (model-free)
run_case A inv_price RESOLVE "an inventory where each product has a price number"
run_case A shelf_pages RESOLVE "a shelf where each book has a title and a pages number"
run_case A todo_pri RESOLVE "a todo list where each task has a title and a priority number and a done flag"
run_case A build_cart RESOLVE "build a cart where each item has a name and a quantity number"

# Tier B — must refuse without inventing unverified code
run_case B snake REFUSE "build a snake game with keyboard controls"
run_case B gui REFUSE "make me a desktop GUI paint app"

# Tier C — gated spec (auth / non-decidable)
if [[ -n "${NSYNTH_LOCAL_LLM_URL:-}" ]]; then
  run_case C bank RESOLVE "build a bank account with deposit and withdraw"
  run_case C bank_pin RESOLVE "a bank account with PIN auth and no overdraft"
else
  run_case C bank SKIP "build a bank account with deposit and withdraw"
  run_case C bank_pin SKIP "a bank account with PIN auth and no overdraft"
fi

echo
echo "=== summary: pass=$pass fail=$fail skip=$skip ==="
[[ "$fail" -eq 0 ]]
