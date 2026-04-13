#!/bin/bash
# Run HumanEval ablation across all 4 models using vLLM
# Each model is loaded into vLLM, benchmarked, then swapped out

set -e

export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/opt/conda/lib:$LD_LIBRARY_PATH

MODELS=(
    "Qwen/Qwen3.5-4B:0.15:humaneval_ablation_4b_vllm"
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
    echo "Starting vLLM server..."
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

    # Wait for vLLM to be ready (up to 5 min)
    echo "Waiting for vLLM to be ready..."
    for i in $(seq 1 60); do
        if curl -s "$BASE_URL/v1/models" >/dev/null 2>&1; then
            echo "vLLM ready after ${i}0 seconds"
            break
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "ERROR: vLLM died. Check /tmp/vllm_${OUTPUT_DIR}.log"
            tail -20 "/tmp/vllm_${OUTPUT_DIR}.log"
            continue 2
        fi
        sleep 10
    done

    # Verify model is loaded
    if ! curl -s "$BASE_URL/v1/models" >/dev/null 2>&1; then
        echo "ERROR: vLLM failed to start for $MODEL"
        tail -20 "/tmp/vllm_${OUTPUT_DIR}.log"
        continue
    fi

    echo "Running HumanEval ablation with parallel=8..."
    python3 -u /root/ncpu/self_optimizing/run_ablation_study.py \
        --model "$MODEL" \
        --base-url "$BASE_URL" \
        --benchmark humaneval \
        --output-dir "/root/training/$OUTPUT_DIR" \
        --repeats 1 \
        --request-timeout 120 \
        --parallel 8

    echo "Done with $MODEL"
    echo ""
done

echo "ALL MODELS COMPLETE"
