#!/usr/bin/env bash
# Always-on component crawler: continuously discover novel verified op
# compositions and PROMOTE them into the live component registry, forever
# (resumable — dedup by behavior signature survives restarts).
#
#   bash scripts/discover_daemon.sh [log.jsonl] [components.json] [sleep_s]
#
# To install as a launchd job (auto-start, survives reboot):
#   cp scripts/ncpu.discover.plist ~/Library/LaunchAgents/
#   launchctl load ~/Library/LaunchAgents/ncpu.discover.plist
set -u
cd "$(dirname "$0")/.."
LOG="${1:-$HOME/.ncpu_discoveries.jsonl}"
COMPS="${2:-$HOME/.ncpu_components.json}"
SLEEP="${3:-300}"
cargo build --release --bin discover 2>/dev/null
while true; do
  ./target/release/discover --log "$LOG" --rounds 3 --per-round 4 --promote "$COMPS"
  # When the space for the current leaf set is exhausted, idle before retrying —
  # newly promoted/registered leaves widen the space over time.
  sleep "$SLEEP"
done
