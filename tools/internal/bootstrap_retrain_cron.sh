#!/usr/bin/env bash
# Cron entry point: re-run bootstrap_train when the cache has grown past
# the configured threshold.
#
# The Rust side writes a marker file (~/.nsynth_bootstrap_needed or the
# path in NSYNTH_BOOTSTRAP_MARKER_PATH) whenever solved_cache::record
# observes enough growth. This script checks the marker via
# `bootstrap_train --if-due`, which exits 0 without training when the
# marker is absent. So the cron body is unconditional — it's always
# cheap when nothing has changed.
#
# Usage (local):
#   tools/bootstrap_retrain_cron.sh
#
# Usage (crontab):
#   */30 * * * * cd /path/to/repo && tools/bootstrap_retrain_cron.sh >> /var/log/nsynth_retrain.log 2>&1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT/nsynth"

if ! [[ -x target/release/bootstrap_train ]]; then
  cargo build --release --bin bootstrap_train > /dev/null 2>&1
fi

echo "[retrain-cron] $(date -u +%Y-%m-%dT%H:%M:%SZ)"

./target/release/bootstrap_train \
  --if-due \
  --epochs 50 \
  --negs-per-pos 4 \
  --margin 0.5 \
  --lr 0.02

# If training ran and succeeded, capture a fresh weight snapshot so the
# trajectory dataset records the retrain event.
if [[ -x target/release/weights_snapshot ]]; then
  ./target/release/weights_snapshot \
    --out "$REPO_ROOT/artifacts/meta_weights_history.tsv" \
    --label bootstrap_retrain \
    > /dev/null 2>&1 || true
fi
