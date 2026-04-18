#!/usr/bin/env bash
# ============================================================
# train_vastai.sh — Run on vast.ai instance to train v5 model
#
# Usage on the instance:
#   bash train_vastai.sh
#
# Expects this dir to be the mog_synth project root with:
#   data/combined_v5.jsonl  (31,236 records)
#   scripts/train_metalearner.py
#   models/ (will write metalearner_1arg_v5.pt here)
# ============================================================
set -e

echo "=== mog_synth v5 meta-learner training ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"
echo ""

mkdir -p models

# Full 400-epoch run on combined_v5.jsonl
# On RTX 4090 / A100: ~5-10 minutes
python3 scripts/train_metalearner.py \
    --data  data/combined_v5.jsonl \
    --save  models/metalearner_1arg_v5.pt \
    --n-args 1 \
    --epochs 400 \
    --batch-size 256 \
    --lr 3e-4 \
    --d-model 128 \
    --n-heads 4 \
    --k 8

echo ""
echo "=== Evaluating v5 ==="
python3 scripts/eval_metalearner.py \
    --models models/metalearner_1arg_v5.pt \
    --bench  data/bench_known_v2.jsonl

echo ""
echo "=== Done. Download models/metalearner_1arg_v5.pt ==="
