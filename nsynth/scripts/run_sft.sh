#!/usr/bin/env bash
# SFT the model to WRITE Mog (the powerful path), then re-eval the code path.
# Prereq: scripts/rlvr_data_gen.py has produced the trace file (nsynth is the
# teacher). Runs on Apple-Silicon MLX.
#
#   bash scripts/run_sft.sh [traces.jsonl] [base_model] [iters]
#
# Base model: a NON-reasoning small model SFTs cleanest for direct code output —
# SmolLM3-3B (Apache) or Phi-4-mini. Gemma-4-E4B works but is a reasoning model
# (heavier; wants reasoning-format data). Model auto-downloads via mlx_lm.
set -e
TRACES="${1:-/tmp/sft_full.jsonl}"
MODEL="${2:-mlx-community/SmolLM3-3B-4bit}"
ITERS="${3:-400}"
ADAPTER="/tmp/mog-planner-adapter"
[ -s "$TRACES" ] || { echo "no traces at $TRACES — run scripts/rlvr_data_gen.py first"; exit 1; }
echo "[sft] $(wc -l < "$TRACES") traces | model=$MODEL | iters=$ITERS"

# mlx_lm lora wants a data DIR with train.jsonl / valid.jsonl (chat format).
D=$(mktemp -d); n=$(wc -l < "$TRACES"); v=$(( n / 10 + 1 ))
tail -n +$((v+1)) "$TRACES" > "$D/train.jsonl"
head -n "$v"        "$TRACES" > "$D/valid.jsonl"

python3 -m mlx_lm lora --model "$MODEL" --train --data "$D" \
    --iters "$ITERS" --batch-size 1 --num-layers 8 --adapter-path "$ADAPTER"

echo "[sft] adapter -> $ADAPTER. Re-eval the CODE path:"
echo "  python3 -m mlx_lm server --model $MODEL --adapter-path $ADAPTER --port 8766 &"
echo "  cargo build --release --bin nsynth_tool"
echo "  MODE=code NSYNTH_LOCAL_LLM_URL=http://localhost:8766/v1/chat/completions \\"
echo "    python3 scripts/bon_eval.py /tmp/he_bench.jsonl 4 154"
