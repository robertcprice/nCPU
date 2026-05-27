#!/bin/bash
#
# Deploy full real-world coprocessor evaluation on vast.ai.
#
# PURPOSE
#   The existing instruct-sweep real-world benchmarks were run on 10-problem
#   samples, which is too small to draw transfer conclusions. Section 11.9 of
#   the paper flags a -40% regression on 3B-instruct; that number came from a
#   ~30-problem HumanEval run. This script runs full HumanEval-164 and a
#   500-problem GSM8K slice on Qwen3.5-4B and -9B with and without the
#   trained coprocessor, producing honest publication numbers.
#
# REQUIRED INPUTS
#   training_results/instruct_sweep/qwen3.5-4b/coprocessor_weights.pt
#   training_results/instruct_sweep/qwen3.5-9b/coprocessor_weights.pt
#   HumanEval.jsonl (~800KB, downloaded on the instance)
#
# COST
#   A100 80GB on vast.ai: ~$1.00 - $2.00 / hr depending on availability.
#   Estimated wall time:
#     Qwen3.5-4B HumanEval-164  ~1.5 hr
#     Qwen3.5-4B GSM8K-500      ~1.0 hr
#     Qwen3.5-9B HumanEval-164  ~3.0 hr
#     Qwen3.5-9B GSM8K-500      ~2.0 hr
#   Total: 7-8 hr × $1.50/hr = $11-$12 typical, $20-24 worst case.
#
# OPT-IN
#   This script will NOT run automatically. It prints the plan and requires
#   you to re-invoke with --confirm to proceed. Nothing contacts vast.ai until
#   --confirm is passed.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIRM=0
MODELS=("qwen3.5-4b" "qwen3.5-9b")
GPU_MODEL="A100_80GB"
MAX_PRICE_PER_HR="2.00"
HUMANEVAL_COUNT=164
GSM8K_COUNT=500

while [[ $# -gt 0 ]]; do
  case "$1" in
    --confirm) CONFIRM=1; shift ;;
    --only) shift; MODELS=("$1"); shift ;;
    --gpu) shift; GPU_MODEL="$1"; shift ;;
    --max-price) shift; MAX_PRICE_PER_HR="$1"; shift ;;
    --humaneval-count) shift; HUMANEVAL_COUNT="$1"; shift ;;
    --gsm8k-count) shift; GSM8K_COUNT="$1"; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

echo "========================================================"
echo "  Coprocessor real-world evaluation — vast.ai deploy plan"
echo "========================================================"
echo "Repo:             $REPO_ROOT"
echo "Models:           ${MODELS[*]}"
echo "GPU:              $GPU_MODEL"
echo "Price cap:        \$$MAX_PRICE_PER_HR / hr"
echo "HumanEval count:  $HUMANEVAL_COUNT"
echo "GSM8K count:      $GSM8K_COUNT"
echo ""
echo "This run will:"
for model in "${MODELS[@]}"; do
  weights="training_results/instruct_sweep/$model/coprocessor_weights.pt"
  if [[ -f "$weights" ]]; then
    echo "  [$model] use weights at $weights"
  else
    echo "  [$model] MISSING WEIGHTS at $weights"
  fi
done
echo ""
echo "Per model, it invokes:"
echo "  python benchmarks/benchmark_coprocessor_realworld.py \\"
echo "    --model <HF_ID> \\"
echo "    --coprocessor-weights <weights>.pt \\"
echo "    --humaneval-path ./HumanEval.jsonl \\"
echo "    --humaneval-count $HUMANEVAL_COUNT \\"
echo "    --gsm8k-count $GSM8K_COUNT"
echo ""
echo "Estimated wall time: 7-8 hours total"
echo "Estimated cost:      \$11-\$24 (depending on availability and run time)"
echo ""

if [[ $CONFIRM -ne 1 ]]; then
  echo "========================================================"
  echo "  DRY RUN — nothing contacted yet."
  echo "  Re-run with --confirm to actually launch on vast.ai."
  echo "========================================================"
  exit 0
fi

# Verify prerequisites before spending money.
command -v vastai >/dev/null || { echo "vastai CLI not found; install from https://vast.ai/cli"; exit 1; }

MISSING=0
for model in "${MODELS[@]}"; do
  if [[ ! -f "training_results/instruct_sweep/$model/coprocessor_weights.pt" ]]; then
    echo "ERROR: missing weights for $model"
    MISSING=1
  fi
done
[[ $MISSING -eq 1 ]] && exit 2

echo "[1/6] Packaging code + weights..."
DEPLOY_DIR=$(mktemp -d)
tar czf "$DEPLOY_DIR/coprocessor_deploy.tar.gz" \
    ncpu/coprocessor/ \
    benchmarks/benchmark_coprocessor_realworld.py \
    "${MODELS[@]/#/training_results/instruct_sweep/}" \
    requirements.txt \
    2>/dev/null
echo "  package: $(du -h "$DEPLOY_DIR/coprocessor_deploy.tar.gz" | cut -f1)"

echo "[2/6] Finding an instance..."
# vastai search format; returns a list of instance ids.
OFFER=$(vastai search offers \
    "gpu_name=$GPU_MODEL rentable=true dph<=$MAX_PRICE_PER_HR num_gpus=1" \
    -o 'dph' 2>/dev/null | head -2 | tail -1 | awk '{print $1}')
if [[ -z "$OFFER" ]]; then
  echo "ERROR: no offers at price cap \$$MAX_PRICE_PER_HR/hr for $GPU_MODEL"
  exit 3
fi
echo "  selected offer: $OFFER"

echo "[3/6] Creating instance..."
INSTANCE_JSON=$(vastai create instance "$OFFER" \
    --image 'pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime' \
    --disk 100 \
    --env '-e HF_HUB_ENABLE_HF_TRANSFER=1' \
    --onstart-cmd 'cd /root && apt-get update && apt-get install -y curl git' \
    --raw 2>/dev/null)
INSTANCE_ID=$(echo "$INSTANCE_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["new_contract"])')
echo "  instance: $INSTANCE_ID"

echo "[4/6] Waiting for instance to boot (this can take 30-90 s)..."
vastai wait-instance "$INSTANCE_ID" --ready >/dev/null

echo "[5/6] Uploading code + launching benchmark..."
vastai copy "$DEPLOY_DIR/coprocessor_deploy.tar.gz" "$INSTANCE_ID:/root/deploy.tar.gz"
vastai copy "scripts/gpu/run_coprocessor_realworld_remote.sh" "$INSTANCE_ID:/root/run.sh" 2>/dev/null || true
vastai ssh "$INSTANCE_ID" "bash -eux -c '
  cd /root && tar xzf deploy.tar.gz
  pip install -q transformers==5.3.0 accelerate datasets
  curl -sL https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz | gunzip > HumanEval.jsonl
  for model in '"${MODELS[*]}"'; do
    case \$model in
      qwen3.5-4b) HF_ID=Qwen/Qwen3.5-4B ;;
      qwen3.5-9b) HF_ID=Qwen/Qwen3.5-9B ;;
      *) echo skipping unknown \$model; continue ;;
    esac
    python benchmarks/benchmark_coprocessor_realworld.py \
      --model \$HF_ID \
      --coprocessor-weights training_results/instruct_sweep/\$model/coprocessor_weights.pt \
      --humaneval-path ./HumanEval.jsonl \
      --humaneval-count '"$HUMANEVAL_COUNT"' \
      --gsm8k-count '"$GSM8K_COUNT"' \
      --output training_results/instruct_sweep/\$model/realworld_vastai.json
  done
'"

echo "[6/6] Downloading results..."
for model in "${MODELS[@]}"; do
  mkdir -p "training_results/instruct_sweep/$model"
  vastai copy \
    "$INSTANCE_ID:/root/training_results/instruct_sweep/$model/realworld_vastai.json" \
    "training_results/instruct_sweep/$model/realworld_vastai.json" 2>/dev/null || \
    echo "  WARN: could not fetch $model result"
done

echo ""
echo "DONE. Instance $INSTANCE_ID is still running."
echo "Stop it manually with: vastai destroy instance $INSTANCE_ID"
