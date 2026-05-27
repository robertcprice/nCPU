#!/usr/bin/env bash
# Runs on the vast.ai instance after the repo is rsynced to
# /workspace/nsynth. Installs model-specific deps, downloads the model
# weights, runs inference_enhanced.py against humaneval_lite, writes
# results to artifacts/.
#
# The companion pull_artifacts.sh fetches the results back.
#
# Usage on the instance:
#   bash tools/vastai/setup_and_run.sh --model Qwen/Qwen3.5-4B-Instruct
#   bash tools/vastai/setup_and_run.sh --model google/gemma-4-9b-it
#   bash tools/vastai/setup_and_run.sh --model Qwen/Qwen3.5-9B-Instruct \
#       --full-humaneval
#
# The model weights are cached under /workspace/hf_cache by default so
# they survive instance restarts (if the vast.ai host preserves the
# volume) but never touch your local machine.

set -euo pipefail

MODEL=""
FULL=0
K=3
MAX_RETRIES=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --full-humaneval) FULL=1; shift ;;
    --k) K="$2"; shift 2 ;;
    --max-retries) MAX_RETRIES="$2"; shift 2 ;;
    *) echo "unknown: $1"; exit 2 ;;
  esac
done

[[ -z "$MODEL" ]] && { echo "--model required"; exit 2; }

export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
mkdir -p "$HF_HOME"

cd /workspace/nsynth

# Pre-download the model (visible progress vs. silent failure mid-run).
python3 - <<EOF
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
print(f"[setup] downloading {'$MODEL'}...")
tok = AutoTokenizer.from_pretrained("$MODEL")
model = AutoModelForCausalLM.from_pretrained(
    "$MODEL", torch_dtype=torch.bfloat16, device_map="auto",
)
print(f"[setup] loaded. params = {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
EOF

mkdir -p artifacts/vastai

SAFE_NAME=$(echo "$MODEL" | sed 's|/|_|g;s|\.|-|g')

# Run 1: the 30-problem humaneval_lite with enhanced inference.
echo "[setup] running humaneval_lite on $MODEL (k=$K, retries=$MAX_RETRIES)..."
NSYNTH_LLM_CACHE_PATH=/workspace/cache_${SAFE_NAME}.tsv \
python3 tools/benchmarks/run_humaneval_agent.py \
    --problems tools/benchmarks/humaneval_lite.jsonl \
    --backend hf \
    --model "$MODEL" \
    --k "$K" --max-retries "$MAX_RETRIES" \
    --out "artifacts/vastai/${SAFE_NAME}_humaneval_lite.md" \
    --verbose

# Optional: full HumanEval. The --mode agent codepath uses the
# Anthropic client by default; for local HF models we extend
# run_humaneval_full.py similarly (or wrap via the LocalModelClient).
# For the first pass here we use the humaneval_lite agent path which
# already has a --backend flag.
if [[ "$FULL" -eq 1 ]]; then
  echo "[setup] full HumanEval on local backends is wired via"
  echo "        run_humaneval_full.py + LocalModelClient. Extending"
  echo "        that path is a follow-up; lite results above are the"
  echo "        primary signal for this vast.ai run."
fi

echo "[setup] done. artifacts at /workspace/nsynth/artifacts/vastai/"
ls -la /workspace/nsynth/artifacts/vastai/
