#!/usr/bin/env bash
# Provision a vast.ai instance sized for an open-source model benchmark.
#
# Usage:
#   tools/vastai/launch.sh <target>
#
# Supported targets:
#   qwen3.5-2b   — RTX 4090, ~8GB VRAM, ~$0.15/hr
#   qwen3.5-4b   — RTX 4090, ~12GB VRAM, ~$0.30/hr
#   qwen3.5-9b   — A100 40GB, ~20GB VRAM, ~$0.70/hr
#   gemma-4-9b   — A100 40GB, ~20GB VRAM, ~$0.70/hr
#   distill-9b   — A100 80GB, ~60GB VRAM, ~$1.50/hr (LoRA fine-tune)
#
# Requires vast.ai CLI (`pip install vastai`) and $VAST_API_KEY.
# Does not require vast.ai-side GPUs on this machine.

set -euo pipefail

TARGET="${1:-}"
if [[ -z "$TARGET" ]]; then
  grep -E "^#   [a-z]" "$0" | head -6
  exit 2
fi

if ! command -v vastai >/dev/null 2>&1; then
  echo "[launch] vastai CLI not installed. pip install vastai" >&2
  exit 2
fi

# CLI auth can come from $VAST_API_KEY or prior `vastai set api-key`
# state. The explicit `show user` preflight has proven flaky and can
# hang; rely on the search/create calls below instead.

case "$TARGET" in
  qwen3.5-2b|qwen3.5-4b)
    # RTX 4090 is enough. Small pytorch runtime image for fast pull.
    GPU="RTX_4090"
    MIN_VRAM=16
    IMAGE="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"
    ;;
  qwen3.5-9b|gemma-4-9b)
    GPU="RTX_A6000"
    MIN_VRAM=40
    IMAGE="pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"
    ;;
  distill-9b)
    GPU="A100_SXM4"
    MIN_VRAM=80
    IMAGE="pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel"
    ;;
  *)
    echo "[launch] unknown target: $TARGET" >&2
    exit 2
    ;;
esac

echo "[launch] searching vast.ai offers for $GPU, ≥${MIN_VRAM}GB VRAM..."

# Find the cheapest matching offer. `reliability > 0.98` filters out
# flaky hosts. `dph_total` = dollars per hour total.
OFFER=$(vastai search offers \
    "gpu_name=$GPU gpu_ram>=${MIN_VRAM} reliability>0.995 dph_total<1.50 cuda_max_good>=12 direct_port_count>=3 inet_down>=500 inet_up>=200 verified=true" \
    --order "reliability2-,dph_total" \
    --limit 10 \
    --raw | python3 -c "
import json, sys
data = json.load(sys.stdin)
if not data:
    sys.exit(0)
# Pick highest reliability (ordered desc), ties broken by cheapest.
# On-demand (not interruptible) for more reliable boot.
print(json.dumps(data[0]))
")

if [[ -z "$OFFER" ]]; then
  echo "[launch] no matching offers; try a less constrained search or bump budget."
  exit 1
fi

OFFER_ID=$(echo "$OFFER" | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["id"])')
DPH=$(echo "$OFFER" | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["dph_total"])')

echo "[launch] picked offer $OFFER_ID at \$${DPH}/hr"

# Provision. Onstart script installs our deps + syncs the repo; we
# inject it via --onstart-cmd.
ONSTART="$(cat <<'EOF'
set -e
apt-get update && apt-get install -y rsync git
pip install --upgrade anthropic datasets transformers accelerate peft trl bitsandbytes
mkdir -p /workspace/nsynth
echo "READY — rsync the repo to /workspace/nsynth and run tools/vastai/setup_and_run.sh"
EOF
)"

INSTANCE=$(vastai create instance "$OFFER_ID" \
    --image "$IMAGE" \
    --disk 40 \
    --ssh \
    --onstart-cmd "$ONSTART" \
    --raw)

INSTANCE_ID=$(echo "$INSTANCE" | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["new_contract"])')

echo "[launch] created instance $INSTANCE_ID"
echo "[launch] waiting for SSH to come up (up to 2 min)..."

for _ in $(seq 1 24); do
  sleep 5
  SSH_URL=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null \
    | python3 -c '
import sys, json
d = json.loads(sys.stdin.read())
host = d.get("ssh_host")
port = d.get("ssh_port")
if host and port:
    print(f"{host}:{port}")
' 2>/dev/null || true)
  if [[ -n "$SSH_URL" ]]; then
    break
  fi
done

if [[ -z "$SSH_URL" ]]; then
  echo "[launch] instance not ready yet; check 'vastai show instance $INSTANCE_ID'"
  exit 1
fi

HOST=${SSH_URL%:*}
PORT=${SSH_URL#*:}

echo "[launch] ready."
echo "[launch] sync the repo:"
echo "    rsync -a --exclude='.git' --exclude='target' --exclude='artifacts' \\"
echo "        -e 'ssh -p $PORT' . root@$HOST:/workspace/nsynth/"
echo "[launch] SSH in:"
echo "    ssh -p $PORT root@$HOST"
echo "[launch] Once connected, run:"
echo "    cd /workspace/nsynth"
echo "    bash tools/vastai/setup_and_run.sh --model Qwen/Qwen3.5-4B-Instruct"
echo "[launch] Pull artifacts back:"
echo "    tools/vastai/pull_artifacts.sh $INSTANCE_ID"
echo "[launch] Destroy when done:"
echo "    vastai destroy instance $INSTANCE_ID"
