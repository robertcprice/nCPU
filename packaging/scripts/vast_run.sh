#!/usr/bin/env bash
# vast.ai one-shot launcher for NPCoT benchmarks (DEPLOY-1).
#
# Provisions a vast.ai instance, uploads the repo, installs dependencies,
# runs the requested benchmark, pulls the results back, and destroys
# the instance. Total cost is minutes-of-GPU-time × $/hr.
#
# Usage:
#   VASTAI=/Users/bobbyprice/Library/Python/3.14/bin/vastai \
#     ./vast_run.sh tests                             # run test suite
#   ./vast_run.sh bench3                              # curated NPCoT bench
#   ./vast_run.sh humaneval Qwen/Qwen3.5-1.5B lib.json # full HumanEval
#   ./vast_run.sh mbpp Qwen/Qwen3.5-1.5B lib.json     # full MBPP
#
# Environment:
#   VASTAI         path to vastai CLI (default: vastai in PATH)
#   GPU_BUDGET     max $/hr to consider (default: 0.10)
#   GPU_RAM_MIN    min GB of VRAM (default: 16)
#   INSTANCE_IMAGE docker image (default: pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel)
#   DISK_GB        disk allocation (default: 40)
#
# This script is a thin wrapper around the exact commands we've validated
# against real vast.ai hardware on April 18, 2026 (see
# artifacts/vast_ai_run/run_report.md).

set -euo pipefail

VASTAI="${VASTAI:-vastai}"
GPU_BUDGET="${GPU_BUDGET:-0.10}"
GPU_RAM_MIN="${GPU_RAM_MIN:-16}"
INSTANCE_IMAGE="${INSTANCE_IMAGE:-pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel}"
DISK_GB="${DISK_GB:-40}"

COMMAND="${1:-help}"
shift || true

if [ "$COMMAND" = "help" ] || [ "$COMMAND" = "-h" ] || [ "$COMMAND" = "--help" ]; then
    head -25 "$0" | tail -22
    exit 0
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
ARTIFACT_DIR="${REPO_ROOT}/artifacts/vast_runs"
mkdir -p "$ARTIFACT_DIR"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${ARTIFACT_DIR}/${RUN_ID}-${COMMAND}"
mkdir -p "$RUN_DIR"

echo "[vast_run] command=${COMMAND}"
echo "[vast_run] run_dir=${RUN_DIR}"
echo "[vast_run] GPU budget ≤ \$${GPU_BUDGET}/hr, min ${GPU_RAM_MIN}GB VRAM"

# -----------------------------------------------------------------------
# 1. Find and rent a GPU
# -----------------------------------------------------------------------
echo "[vast_run] searching for offers…"
OFFERS="$("$VASTAI" search offers \
    "reliability > 0.99 num_gpus=1 gpu_ram>=${GPU_RAM_MIN} verified=true rentable=true cuda_max_good>=12.0 dph_total<=${GPU_BUDGET}" \
    -o 'dph_total' 2>&1 | head -10)"
echo "$OFFERS"
OFFER_ID="$(echo "$OFFERS" | awk 'NR==2 {print $1}')"
if [ -z "$OFFER_ID" ] || [ "$OFFER_ID" = "No" ]; then
    echo "[vast_run] no offers found under budget; raise GPU_BUDGET or widen filter" >&2
    exit 2
fi
echo "[vast_run] selected offer_id=${OFFER_ID}"

CREATE_OUT="$("$VASTAI" create instance "$OFFER_ID" \
    --image "$INSTANCE_IMAGE" --disk "$DISK_GB" --ssh --direct 2>&1)"
echo "$CREATE_OUT"
INSTANCE_ID="$(echo "$CREATE_OUT" | grep -o "'new_contract': [0-9]*" | awk '{print $2}')"
if [ -z "$INSTANCE_ID" ]; then
    echo "[vast_run] failed to parse instance id from output" >&2
    exit 3
fi
echo "[vast_run] instance_id=${INSTANCE_ID}"

# Teardown trap
cleanup() {
    echo "[vast_run] destroying instance ${INSTANCE_ID}"
    "$VASTAI" destroy instance "$INSTANCE_ID" || true
}
trap cleanup EXIT

# -----------------------------------------------------------------------
# 2. Wait for running state
# -----------------------------------------------------------------------
echo "[vast_run] waiting for running state…"
until "$VASTAI" show instance "$INSTANCE_ID" 2>&1 | grep -q running; do
    sleep 15
done

# Extract SSH details.
INFO="$("$VASTAI" show instance "$INSTANCE_ID" 2>&1)"
SSH_HOST="$(echo "$INFO" | awk 'NR==2 {print $11}')"
SSH_PORT="$(echo "$INFO" | awk 'NR==2 {print $12}')"
echo "[vast_run] ssh ${SSH_HOST}:${SSH_PORT}"

ssh_cmd() {
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -p "$SSH_PORT" "root@${SSH_HOST}" "$@"
}

# -----------------------------------------------------------------------
# 3. Upload repo
# -----------------------------------------------------------------------
echo "[vast_run] uploading repo…"
rsync -az \
    --exclude='.git/' \
    --exclude='*/target/' \
    --exclude='**/node_modules/' \
    --exclude='**/__pycache__/' \
    --exclude='**/.pytest_cache/' \
    --exclude='models/' \
    --exclude='training_results/' \
    --exclude='*.so' \
    --exclude='*.wasm' \
    --exclude='dist/' \
    --exclude='artifacts/' \
    -e "ssh -o StrictHostKeyChecking=no -p ${SSH_PORT}" \
    "${REPO_ROOT}/" "root@${SSH_HOST}:/workspace/nCPU/"

# -----------------------------------------------------------------------
# 4. Install deps + run benchmark
# -----------------------------------------------------------------------
echo "[vast_run] installing deps…"
ssh_cmd 'pip install -q pytest pytest-asyncio hypothesis transformers datasets 2>&1 | tail -3'

case "$COMMAND" in
    tests)
        echo "[vast_run] running NPCoT-focused test suite…"
        ssh_cmd 'cd /workspace/nCPU && python3 -m pytest \
            tests/self_optimizing/test_array_executable_thought_head.py \
            tests/self_optimizing/test_array_program_library.py \
            tests/self_optimizing/test_array_program_library_audit.py \
            tests/self_optimizing/test_array_library_transfer.py \
            tests/self_optimizing/test_array_library_device.py \
            tests/self_optimizing/test_array_thought_curriculum.py \
            tests/self_optimizing/test_array_transform_fidelity.py \
            tests/self_optimizing/test_npcot_native_backend.py \
            tests/self_optimizing/test_program_library_session.py \
            tests/self_optimizing/test_array_thought_coprocessor.py \
            tests/self_optimizing/test_executable_thought_head.py \
            tests/self_optimizing/test_executable_thought_training.py \
            tests/self_optimizing/test_array_log_product.py \
            tests/self_optimizing/test_library_snapshot_diff.py \
            tests/self_optimizing/test_program_verifier.py \
            tests/self_optimizing/test_compliance_report.py \
            tests/self_optimizing/test_verifier_offset.py \
            tests/self_optimizing/test_library_signing.py \
            tests/self_optimizing/test_library_fingerprint.py \
            tests/self_optimizing/test_library_merge.py \
            tests/self_optimizing/test_library_privacy.py \
            tests/self_optimizing/test_npcot_server.py \
            tests/self_optimizing/test_library_distillation.py \
            tests/self_optimizing/test_npcot_sweep_runner.py \
            tests/self_optimizing/test_compliance_cli_exit_codes.py \
            tests/self_optimizing/test_humaneval_runner.py \
            tests/self_optimizing/test_mbpp_runner.py \
            -q 2>&1 | tail -5' | tee "${RUN_DIR}/tests.log"
        ;;
    bench3)
        echo "[vast_run] running curated NPCoT bench…"
        ssh_cmd 'cd /workspace/nCPU && python3 -m benchmarks.benchmark_npcot_coding_bench \
            --n-problems 200 --json /tmp/bench3.json 2>&1 | tail -15' | tee "${RUN_DIR}/bench3.log"
        rsync -az -e "ssh -o StrictHostKeyChecking=no -p ${SSH_PORT}" \
            "root@${SSH_HOST}:/tmp/bench3.json" "${RUN_DIR}/bench3.json" || true
        ;;
    humaneval)
        MODEL="${1:-Qwen/Qwen3.5-1.5B}"
        LIBRARY="${2:-}"
        LIB_FLAG=""
        if [ -n "$LIBRARY" ]; then
            # Upload the library
            scp -o StrictHostKeyChecking=no -P "$SSH_PORT" "$LIBRARY" \
                "root@${SSH_HOST}:/tmp/library.json"
            LIB_FLAG="--library /tmp/library.json"
        else
            LIB_FLAG="--no-library"
        fi
        echo "[vast_run] running HumanEval on ${MODEL} with ${LIB_FLAG}"
        ssh_cmd "cd /workspace/nCPU && python3 -m ncpu.self_optimizing.humaneval_runner \
            --model '${MODEL}' ${LIB_FLAG} \
            --out /tmp/humaneval.json 2>&1 | tail -20" | tee "${RUN_DIR}/humaneval.log"
        rsync -az -e "ssh -o StrictHostKeyChecking=no -p ${SSH_PORT}" \
            "root@${SSH_HOST}:/tmp/humaneval.json" "${RUN_DIR}/humaneval.json" || true
        ;;
    mbpp)
        MODEL="${1:-Qwen/Qwen3.5-1.5B}"
        LIBRARY="${2:-}"
        LIB_FLAG=""
        if [ -n "$LIBRARY" ]; then
            scp -o StrictHostKeyChecking=no -P "$SSH_PORT" "$LIBRARY" \
                "root@${SSH_HOST}:/tmp/library.json"
            LIB_FLAG="--library /tmp/library.json"
        else
            LIB_FLAG="--no-library"
        fi
        echo "[vast_run] running MBPP on ${MODEL} with ${LIB_FLAG}"
        ssh_cmd "cd /workspace/nCPU && python3 -m ncpu.self_optimizing.mbpp_runner \
            --model '${MODEL}' ${LIB_FLAG} \
            --out /tmp/mbpp.json 2>&1 | tail -20" | tee "${RUN_DIR}/mbpp.log"
        rsync -az -e "ssh -o StrictHostKeyChecking=no -p ${SSH_PORT}" \
            "root@${SSH_HOST}:/tmp/mbpp.json" "${RUN_DIR}/mbpp.json" || true
        ;;
    livecodebench)
        MODEL="${1:-Qwen/Qwen3.5-4B}"
        LIBRARY="${2:-}"
        CKPT="${3:-}"
        LIB_FLAG=""
        CKPT_FLAG=""
        if [ -n "$LIBRARY" ] && [ -f "$LIBRARY" ]; then
            scp -o StrictHostKeyChecking=no -P "$SSH_PORT" "$LIBRARY" \
                "root@${SSH_HOST}:/tmp/library.json"
            LIB_FLAG="--library /tmp/library.json"
        fi
        if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
            scp -o StrictHostKeyChecking=no -P "$SSH_PORT" "$CKPT" \
                "root@${SSH_HOST}:/tmp/checkpoint.pt"
            CKPT_FLAG="--coprocessor-checkpoint /tmp/checkpoint.pt"
        fi
        echo "[vast_run] running LiveCodeBench code-gen on ${MODEL}"
        ssh_cmd "cd /workspace/nCPU && python3 -m ncpu.self_optimizing.run_livecodebench \
            --model '${MODEL}' ${LIB_FLAG} ${CKPT_FLAG} \
            --out /tmp/livecodebench_codegen.json 2>&1 | tail -30" \
            | tee "${RUN_DIR}/livecodebench_codegen.log"
        rsync -az -e "ssh -o StrictHostKeyChecking=no -p ${SSH_PORT}" \
            "root@${SSH_HOST}:/tmp/livecodebench_codegen.json" \
            "${RUN_DIR}/livecodebench_codegen.json" || true
        ;;
    livecodebench-repair)
        MODEL="${1:-Qwen/Qwen3.5-4B}"
        LIBRARY="${2:-}"
        CKPT="${3:-}"
        LIB_FLAG=""
        CKPT_FLAG=""
        if [ -n "$LIBRARY" ] && [ -f "$LIBRARY" ]; then
            scp -o StrictHostKeyChecking=no -P "$SSH_PORT" "$LIBRARY" \
                "root@${SSH_HOST}:/tmp/library.json"
            LIB_FLAG="--library /tmp/library.json"
        fi
        if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
            scp -o StrictHostKeyChecking=no -P "$SSH_PORT" "$CKPT" \
                "root@${SSH_HOST}:/tmp/checkpoint.pt"
            CKPT_FLAG="--coprocessor-checkpoint /tmp/checkpoint.pt"
        fi
        echo "[vast_run] running LiveCodeBench self-repair on ${MODEL}"
        ssh_cmd "cd /workspace/nCPU && python3 -m ncpu.self_optimizing.run_livecodebench \
            --scenario selfrepair \
            --model '${MODEL}' ${LIB_FLAG} ${CKPT_FLAG} \
            --out /tmp/livecodebench_repair.json 2>&1 | tail -30" \
            | tee "${RUN_DIR}/livecodebench_repair.log"
        rsync -az -e "ssh -o StrictHostKeyChecking=no -p ${SSH_PORT}" \
            "root@${SSH_HOST}:/tmp/livecodebench_repair.json" \
            "${RUN_DIR}/livecodebench_repair.json" || true
        ;;
    *)
        echo "[vast_run] unknown command: ${COMMAND}" >&2
        echo "  commands: tests | bench3 | humaneval | mbpp | livecodebench | livecodebench-repair" >&2
        exit 4
        ;;
esac

echo
echo "[vast_run] complete. artifacts in: ${RUN_DIR}"
"$VASTAI" show user 2>&1 | grep -E "^[0-9]+" | head -1
