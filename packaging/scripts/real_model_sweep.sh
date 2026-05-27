#!/usr/bin/env bash
# Run real HumanEval + MBPP baselines across multiple open-source LLMs.
#
# Runs entirely on the remote vast.ai instance (called via ssh).
# Writes per-model JSON reports to /workspace/reports/ on the remote.
#
# Usage:
#   ./real_model_sweep.sh <ssh-host> <ssh-port> humaneval
#   ./real_model_sweep.sh <ssh-host> <ssh-port> mbpp
#   ./real_model_sweep.sh <ssh-host> <ssh-port> both

set -euo pipefail

SSH_HOST="${1:-}"
SSH_PORT="${2:-}"
MODE="${3:-both}"
if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ]; then
    echo "usage: $0 <ssh-host> <ssh-port> {humaneval|mbpp|both}" >&2
    exit 1
fi

MODELS=(
    "Qwen/Qwen3.5-0.8B"
    "Qwen/Qwen3.5-2B"
    "Qwen/Qwen3.5-4B"
    "Qwen/Qwen3.5-9B"
)

MAX_PROBLEMS="${MAX_PROBLEMS:-50}"   # HumanEval full is 164; MBPP test is ~500

ssh_cmd() {
    ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "root@${SSH_HOST}" "$@"
}

ssh_cmd 'mkdir -p /workspace/reports'

for MODEL in "${MODELS[@]}"; do
    SAFE="${MODEL//\//_}"
    echo
    echo "===== ${MODEL} ====="
    if [ "$MODE" = "humaneval" ] || [ "$MODE" = "both" ]; then
        echo "[sweep] humaneval on ${MODEL}"
        ssh_cmd "cd /workspace/nCPU && python3 -m ncpu.self_optimizing.humaneval_runner \
            --model '${MODEL}' --no-library --max-problems ${MAX_PROBLEMS} \
            --out /workspace/reports/humaneval_${SAFE}.json 2>&1 | tail -30"
    fi
    if [ "$MODE" = "mbpp" ] || [ "$MODE" = "both" ]; then
        echo "[sweep] mbpp on ${MODEL}"
        ssh_cmd "cd /workspace/nCPU && python3 -m ncpu.self_optimizing.mbpp_runner \
            --model '${MODEL}' --no-library --max-problems ${MAX_PROBLEMS} \
            --out /workspace/reports/mbpp_${SAFE}.json 2>&1 | tail -30"
    fi
done

echo
echo "[sweep] all done. reports in /workspace/reports/ on remote."
ssh_cmd 'ls -lh /workspace/reports/'
