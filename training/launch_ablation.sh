#!/usr/bin/env bash
# =============================================================================
# launch_ablation.sh - Find a vast.ai instance, upload nCPU, run ablation
# =============================================================================
#
# Usage:
#   bash training/launch_ablation.sh             # Auto-find instance and launch
#   bash training/launch_ablation.sh --quick     # Quick test (200 steps)
#   bash training/launch_ablation.sh --dry-run   # Just search, don't create
#
# Requirements:
#   - vastai CLI installed (~/.local/bin/vastai)
#   - vast.ai API key configured (vastai set api-key <key>)
#   - rsync or scp available
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VASTAI="$HOME/.local/bin/vastai"

# Config
MAX_PRICE=0.30      # $/hr
MIN_GPU_RAM=22000   # MB (RTX 3090 = 24GB)
MIN_CUDA="12.4"
DISK_GB=40
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"

# Parse args
DRY_RUN=false
QUICK=""
for arg in "$@"; do
    case $arg in
        --dry-run)  DRY_RUN=true ;;
        --quick)    QUICK="--quick" ;;
    esac
done

echo "============================================================"
echo "nCPU Ablation - vast.ai Launcher"
echo "============================================================"
echo "Project: $PROJECT_ROOT"
echo "Max price: \$${MAX_PRICE}/hr"
echo "Min VRAM: ${MIN_GPU_RAM}MB"
echo "Image: $IMAGE"
echo ""

# Check vastai CLI
if [ ! -x "$VASTAI" ]; then
    echo "ERROR: vastai CLI not found at $VASTAI"
    echo "Install: pip install vastai"
    echo "Or: curl -s https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -o ~/.local/bin/vastai && chmod +x ~/.local/bin/vastai"
    exit 1
fi

# ── Step 1: Search for instances ─────────────────────────────────────────────

echo "[Step 1] Searching for GPU instances..."
echo "  Criteria: GPU RAM >= ${MIN_GPU_RAM}MB, CUDA >= ${MIN_CUDA}, < \$${MAX_PRICE}/hr"
echo ""

# Search for offers
# vast.ai query syntax for RTX 3090+ at good price
SEARCH_RESULTS=$($VASTAI search offers \
    "gpu_ram >= ${MIN_GPU_RAM} cuda_vers >= 12.4 dph <= ${MAX_PRICE} reliability > 0.95 num_gpus = 1 inet_down > 200 disk_space >= ${DISK_GB}" \
    --order "dph" \
    --limit 10 \
    --raw 2>/dev/null) || {
    echo "Search failed. Trying broader search..."
    SEARCH_RESULTS=$($VASTAI search offers \
        "gpu_ram >= ${MIN_GPU_RAM} dph <= ${MAX_PRICE} num_gpus = 1 disk_space >= ${DISK_GB}" \
        --order "dph" \
        --limit 10 \
        --raw 2>/dev/null) || {
        echo "ERROR: vast.ai search failed. Check API key with: $VASTAI show user"
        exit 1
    }
}

# Parse best offer
OFFER_ID=$(echo "$SEARCH_RESULTS" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if not data:
    print('NONE')
    sys.exit(0)
# Pick cheapest
best = data[0]
print(best['id'])
" 2>/dev/null)

if [ "$OFFER_ID" = "NONE" ] || [ -z "$OFFER_ID" ]; then
    echo "ERROR: No suitable instances found."
    echo "Try relaxing constraints (higher price, lower GPU RAM)."
    echo ""
    echo "Manual search:"
    echo "  $VASTAI search offers 'gpu_ram >= 22000 dph <= 0.50 num_gpus = 1' --order dph"
    exit 1
fi

# Show offer details
echo "Best offer found:"
echo "$SEARCH_RESULTS" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if data:
    o = data[0]
    print(f\"  ID:       {o['id']}\")
    print(f\"  GPU:      {o.get('gpu_name', 'unknown')}\")
    print(f\"  VRAM:     {o.get('gpu_ram', 0)/1024:.0f} GB\")
    print(f\"  CUDA:     {o.get('cuda_max_good', 'unknown')}\")
    print(f\"  Price:    \${o.get('dph_total', 0):.3f}/hr\")
    print(f\"  DL Speed: {o.get('inet_down', 0):.0f} Mbps\")
    print(f\"  Location: {o.get('geolocation', 'unknown')}\")
" 2>/dev/null
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would create instance from offer $OFFER_ID"
    echo "Top 5 offers:"
    echo "$SEARCH_RESULTS" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for o in data[:5]:
    print(f\"  {o['id']:>8}  {o.get('gpu_name','?'):<20}  \${o.get('dph_total',0):.3f}/hr  {o.get('gpu_ram',0)/1024:.0f}GB  CUDA {o.get('cuda_max_good','?')}\")
" 2>/dev/null
    exit 0
fi

# ── Step 2: Create instance ──────────────────────────────────────────────────

echo "[Step 2] Creating instance from offer $OFFER_ID..."

INSTANCE_ID=$($VASTAI create instance "$OFFER_ID" \
    --image "$IMAGE" \
    --disk "$DISK_GB" \
    --onstart-cmd "echo 'Instance ready'" \
    --raw 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
if 'new_contract' in data:
    print(data['new_contract'])
elif 'id' in data:
    print(data['id'])
else:
    print(data)
" 2>/dev/null)

echo "  Instance ID: $INSTANCE_ID"
echo ""

# ── Step 3: Wait for instance to start ───────────────────────────────────────

echo "[Step 3] Waiting for instance to start..."

MAX_WAIT=300
WAITED=0
SSH_CMD=""

while [ $WAITED -lt $MAX_WAIT ]; do
    INSTANCE_INFO=$($VASTAI show instance "$INSTANCE_ID" --raw 2>/dev/null) || true

    STATUS=$(echo "$INSTANCE_INFO" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('actual_status', data.get('status_msg', 'unknown')))
" 2>/dev/null) || STATUS="unknown"

    SSH_HOST=$(echo "$INSTANCE_INFO" | python3 -c "
import json, sys
data = json.load(sys.stdin)
host = data.get('ssh_host', '')
port = data.get('ssh_port', '')
if host and port:
    print(f'{host}:{port}')
" 2>/dev/null) || SSH_HOST=""

    echo "  Status: $STATUS (${WAITED}s elapsed)"

    if [ "$STATUS" = "running" ] && [ -n "$SSH_HOST" ]; then
        SSH_HOST_ONLY=$(echo "$SSH_HOST" | cut -d: -f1)
        SSH_PORT=$(echo "$SSH_HOST" | cut -d: -f2)
        SSH_CMD="ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST_ONLY"
        echo "  SSH: $SSH_CMD"
        break
    fi

    sleep 10
    WAITED=$((WAITED + 10))
done

if [ -z "$SSH_CMD" ]; then
    echo "ERROR: Instance did not start within ${MAX_WAIT}s"
    echo "Check status: $VASTAI show instance $INSTANCE_ID"
    exit 1
fi

echo ""

# Give it a moment for SSH to be fully ready
sleep 15

# ── Step 4: Upload project ───────────────────────────────────────────────────

echo "[Step 4] Uploading nCPU project..."

# Use rsync to upload, excluding large/unnecessary files
rsync -avz --progress \
    -e "ssh -o StrictHostKeyChecking=no -p $SSH_PORT" \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.venv' \
    --exclude 'venv' \
    --exclude 'node_modules' \
    --exclude 'models/os/*.pt' \
    --exclude 'training_results' \
    --exclude 'sweep_results' \
    --exclude '.mypy_cache' \
    --exclude '.pytest_cache' \
    --exclude '*.egg-info' \
    "$PROJECT_ROOT/" \
    "root@${SSH_HOST_ONLY}:/root/nCPU/"

echo ""
echo "  Upload complete."
echo ""

# ── Step 5: Launch ablation ──────────────────────────────────────────────────

echo "[Step 5] Launching ablation study..."
echo ""

# Run the ablation in a tmux session so it survives SSH disconnect
$SSH_CMD "
    # Start in tmux so it persists
    apt-get update -qq && apt-get install -y -qq tmux > /dev/null 2>&1 || true
    tmux new-session -d -s ablation 'cd /root/nCPU && bash training/run_exec_ablation_vast.sh $QUICK 2>&1 | tee /root/ablation_output.log'
    echo 'Ablation launched in tmux session: ablation'
"

echo ""
echo "============================================================"
echo "ABLATION STUDY LAUNCHED SUCCESSFULLY"
echo "============================================================"
echo ""
echo "Instance ID:  $INSTANCE_ID"
echo "SSH:          $SSH_CMD"
echo ""
echo "Monitor progress:"
echo "  $SSH_CMD 'tail -f /root/ablation_output.log'"
echo ""
echo "Check tmux session:"
echo "  $SSH_CMD -t 'tmux attach -t ablation'"
echo ""
echo "Download results when done:"
echo "  rsync -avz -e 'ssh -o StrictHostKeyChecking=no -p $SSH_PORT' root@${SSH_HOST_ONLY}:/root/nCPU/ablation_results/ ./ablation_results/"
echo ""
echo "Destroy instance when done:"
echo "  $VASTAI destroy instance $INSTANCE_ID"
echo ""
echo "Estimated cost: ~\$${MAX_PRICE}/hr x ~4hrs = ~\$1.20"
echo "============================================================"
