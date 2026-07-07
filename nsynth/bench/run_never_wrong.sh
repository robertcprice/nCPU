#!/bin/bash
# Never-wrong front-door eval: NL prompt (+ hidden examples) -> verified op OR refuse.
# Measures the differentiator: SOLVED / REFUSED / WRONG. WRONG must be 0 (structural).
# Usage: bash bench/run_never_wrong.sh
set -e
cd "$(dirname "$0")/.."
cargo build --release --bin nl_route 2>&1 | grep -E '^error' && exit 1 || true
BIN=./target/release/nl_route
IN=bench/never_wrong_eval.jsonl
: > /tmp/nw_results.txt
while read -r row; do printf '%s' "$row" | timeout 6 "$BIN" 2>/dev/null >> /tmp/nw_results.txt || echo "REFUSED ?" >> /tmp/nw_results.txt; done < "$IN"
echo "=== never-wrong eval ==="
awk '{print $1}' /tmp/nw_results.txt | sort | uniq -c
echo "WRONG (must be 0): $(grep -c '^WRONG' /tmp/nw_results.txt)"
