#!/bin/bash
# Run HumanEval ablation for remaining models (9B, 27B, MoE)
# 4B is already complete. Resume-safe: skips completed conditions.

export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/opt/conda/lib:$LD_LIBRARY_PATH

MODELS=(
    "Qwen/Qwen3.5-9B:0.20:humaneval_ablation_9b_vllm"
    "Qwen/Qwen3.5-27B:0.45:humaneval_ablation_27b_vllm"
    "Qwen/Qwen3-30B-A3B:0.40:humaneval_ablation_moe_vllm"
)

VLLM_PORT=8000
BASE_URL="http://localhost:${VLLM_PORT}"

for entry in "${MODELS[@]}"; do
    IFS=':' read -r MODEL GPU_UTIL OUTPUT_DIR <<< "$entry"
    echo ""
    echo "========================================"
    echo "MODEL: $MODEL"
    echo "GPU UTIL: $GPU_UTIL"
    echo "OUTPUT: /root/training/$OUTPUT_DIR"
    echo "========================================"

    # Kill any existing vLLM
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 5

    # Start vLLM for this model
    echo "Starting vLLM server for $MODEL..."
    nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port $VLLM_PORT \
        --max-model-len 4096 \
        --gpu-memory-utilization "$GPU_UTIL" \
        --trust-remote-code \
        --dtype auto \
        > "/tmp/vllm_${OUTPUT_DIR}.log" 2>&1 &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"

    # Wait for vLLM to be ready (up to 10 min for larger models)
    echo "Waiting for vLLM to be ready..."
    READY=0
    for i in $(seq 1 120); do
        if curl -s "$BASE_URL/v1/models" >/dev/null 2>&1; then
            echo "vLLM ready after ${i}0 seconds"
            READY=1
            break
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "ERROR: vLLM died. Log:"
            tail -30 "/tmp/vllm_${OUTPUT_DIR}.log"
            READY=0
            break
        fi
        sleep 10
    done

    if [ "$READY" -ne 1 ]; then
        echo "SKIP: vLLM failed for $MODEL"
        continue
    fi

    echo "Running HumanEval ablation (parallel=8, resume-safe)..."
    python3 -u /root/ncpu/self_optimizing/run_ablation_study.py \
        --model "$MODEL" \
        --base-url "$BASE_URL" \
        --benchmark humaneval \
        --output-dir "/root/training/$OUTPUT_DIR" \
        --repeats 1 \
        --request-timeout 120 \
        --parallel 8 || echo "WARNING: ablation exited with error for $MODEL"

    echo "Done with $MODEL"
    echo ""
done

# Kill vLLM after last model
pkill -f "vllm.entrypoints" 2>/dev/null || true

echo ""
echo "========================================"
echo "ALL REMAINING MODELS COMPLETE"
echo "========================================"

# Print summary
python3 << 'PYEOF'
import json, os
models = {
    "4B": "/root/training/humaneval_ablation_4b_vllm",
    "9B": "/root/training/humaneval_ablation_9b_vllm",
    "27B": "/root/training/humaneval_ablation_27b_vllm",
    "MoE": "/root/training/humaneval_ablation_moe_vllm",
}
for label, d in models.items():
    f = os.path.join(d, "ablation_progress.json")
    if os.path.exists(f):
        data = json.load(open(f))
        conds = data.get("conditions", {})
        print("\n%s: %d conditions" % (label, len(conds)))
        for k in sorted(conds):
            sr = conds[k]["summary"]["success_rate"]
            print("  %s: %.1f%%" % (k, sr*100))
    else:
        print("\n%s: no results" % label)
PYEOF
