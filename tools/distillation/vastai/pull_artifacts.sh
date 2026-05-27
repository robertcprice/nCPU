#!/usr/bin/env bash
# Pull benchmark artifacts back from a vast.ai instance.
#
# Usage:
#   tools/vastai/pull_artifacts.sh <instance_id>

set -euo pipefail

ID="${1:-}"
[[ -z "$ID" ]] && { echo "usage: $0 <instance_id>"; exit 2; }

if ! command -v vastai >/dev/null 2>&1; then
  echo "vastai CLI required"; exit 2
fi

# Resolve SSH connection details.
INFO=$(vastai show instance "$ID" --raw)
HOST=$(echo "$INFO" | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["ssh_host"])')
PORT=$(echo "$INFO" | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["ssh_port"])')

if [[ -z "$HOST" || -z "$PORT" ]]; then
  echo "instance $ID has no SSH endpoint. is it still running?"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "[pull] rsyncing artifacts from $HOST:$PORT..."
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"
rsync -av --progress \
    -e "ssh -p $PORT -i $SSH_KEY -o StrictHostKeyChecking=no" \
    "root@$HOST:/workspace/nsynth/artifacts/vastai/" \
    "$REPO_ROOT/artifacts/vastai/"

echo "[pull] artifacts at $REPO_ROOT/artifacts/vastai/"
ls -la "$REPO_ROOT/artifacts/vastai/"

echo
echo "[pull] run 'vastai destroy instance $ID' when you're done paying for it."
