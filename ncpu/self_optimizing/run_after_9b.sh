#!/bin/bash
# Run after 9B ablation completes: 27B then MoE
# 27B needs --enforce-eager (GDN), MoE (Qwen3) does not

export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/opt/conda/lib:$LD_LIBRARY_PATH

VLLM_PORT=8000
BASE_URL="http://localhost:${VLLM_PORT}"

# Wait for 9B to finish
echo "Waiting for 9B ablation to finish..."
while pgrep -f "run_ablation_study.*9B" > /dev/null 2>&1; do
    sleep 60
done
echo "9B ablation finished."

# ---- 27B ----
echo ""
echo "========================================"
echo "MODEL: Qwen/Qwen3.5-27B"
echo "========================================"

pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 5

nohup python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-27B \
    --port $VLLM_PORT \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.50 \
    --trust-remote-code \
    --dtype auto \
    --enforce-eager \
    > /tmp/vllm_27b.log 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

echo "Waiting for vLLM to be ready..."
READY=0
for i in $(seq 1 120); do
    if curl -s "$BASE_URL/v1/models" >/dev/null 2>&1; then
        echo "vLLM ready after ${i}0 seconds"
        READY=1
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM died."
        tail -30 /tmp/vllm_27b.log
        READY=0
        break
    fi
    sleep 10
done

if [ "$READY" -eq 1 ]; then
    python3 -u /root/ncpu/self_optimizing/run_ablation_study.py \
        --model "Qwen/Qwen3.5-27B" \
        --base-url "$BASE_URL" \
        --benchmark humaneval \
        --output-dir "/root/training/humaneval_ablation_27b_vllm" \
        --repeats 1 \
        --request-timeout 180 \
        --parallel 4 || echo "WARNING: 27B ablation error"
    echo "Done with 27B"
else
    echo "SKIP: 27B vLLM failed"
fi

# ---- MoE ----
echo ""
echo "========================================"
echo "MODEL: Qwen/Qwen3-30B-A3B (MoE)"
echo "========================================"

pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 5

# MoE is Qwen3 (not Qwen3.5), no GDN, should work without --enforce-eager
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B \
    --port $VLLM_PORT \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.45 \
    --trust-remote-code \
    --dtype auto \
    > /tmp/vllm_moe.log 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

echo "Waiting for vLLM to be ready..."
READY=0
for i in $(seq 1 120); do
    if curl -s "$BASE_URL/v1/models" >/dev/null 2>&1; then
        echo "vLLM ready after ${i}0 seconds"
        READY=1
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM died."
        tail -30 /tmp/vllm_moe.log
        READY=0
        break
    fi
    sleep 10
done

if [ "$READY" -eq 1 ]; then
    python3 -u /root/ncpu/self_optimizing/run_ablation_study.py \
        --model "Qwen/Qwen3-30B-A3B" \
        --base-url "$BASE_URL" \
        --benchmark humaneval \
        --output-dir "/root/training/humaneval_ablation_moe_vllm" \
        --repeats 1 \
        --request-timeout 120 \
        --parallel 8 || echo "WARNING: MoE ablation error"
    echo "Done with MoE"
else
    echo "SKIP: MoE vLLM failed"
fi

pkill -f "vllm.entrypoints" 2>/dev/null || true

echo ""
echo "========================================"
echo "ALL MODELS COMPLETE"
echo "========================================"

# Summary
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
        print("\n%s (%d conditions):" % (label, len(conds)))
        for k in sorted(conds):
            sr = conds[k]["summary"]["success_rate"]
            att = conds[k]["summary"].get("avg_attempts", 1.0)
            print("  %s: %.1f%% (att=%.1f)" % (k, sr*100, att))
    else:
        print("\n%s: no results" % label)
PYEOF
