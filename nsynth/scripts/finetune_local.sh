#!/usr/bin/env bash
# Last mile of the self-sufficiency loop: fine-tune the LOCAL model on nsynth's OWN
# verified harvest. Takes the JSONL from harvest_corpus.sh, splits train/valid, and
# runs mlx_lm.lora (LoRA — lightweight, Apple-Silicon-local) on the base model. The
# model learns to emit verified-shaped Mog for a task from nsynth's own proofs — no
# external teacher, no cloud. The verifier still gates everything at inference, so a
# weaker/self-trained model never lowers the guarantee.
#
# Usage: finetune_local.sh <harvest.jsonl> [adapter_out_dir] [iters]
#   NSYNTH_LOCAL_LLM_MODEL must name the base model (e.g. mlx-community/gemma-...-4bit)
set -u
HARVEST="${1:?usage: finetune_local.sh <harvest.jsonl> [adapter_dir] [iters]}"
ADAPTER="${2:-adapters/nsynth-selftrained}"
ITERS="${3:-300}"
BASE="${NSYNTH_LOCAL_LLM_MODEL:?set NSYNTH_LOCAL_LLM_MODEL to the base model name}"

command -v python3 >/dev/null || { echo "python3 required"; exit 1; }
python3 -c 'import mlx_lm' 2>/dev/null || { echo "mlx_lm required: pip install mlx-lm"; exit 1; }
[ -s "$HARVEST" ] || { echo "empty/missing harvest: $HARVEST"; exit 1; }

# mlx_lm.lora reads a data DIR containing train.jsonl + valid.jsonl (chat format —
# exactly what local_llm::training_record emits). Deterministic 90/10 split (no
# shuffle) so a run is reproducible.
DATA="$(mktemp -d)"
N=$(wc -l < "$HARVEST" | tr -d ' ')
VN=$(( N / 10 )); [ "$VN" -lt 1 ] && VN=1
tail -n "$VN" "$HARVEST" > "$DATA/valid.jsonl"
head -n $(( N - VN )) "$HARVEST" > "$DATA/train.jsonl"
echo "[finetune] base=$BASE train=$(( N - VN )) valid=$VN iters=$ITERS -> $ADAPTER"

mkdir -p "$ADAPTER"
python3 -m mlx_lm.lora --model "$BASE" --train --data "$DATA" \
  --adapter-path "$ADAPTER" --iters "$ITERS" --batch-size 1
rc=$?
rm -rf "$DATA"
[ "$rc" -eq 0 ] || { echo "[finetune] mlx_lm.lora failed (rc=$rc)"; exit "$rc"; }
echo "[finetune] done -> $ADAPTER"
echo "[finetune] serve: python3 -m mlx_lm server --model $BASE --adapter-path $ADAPTER --port 8765"
echo "[finetune] then: export NSYNTH_LOCAL_LLM_URL=http://localhost:8765/v1/chat/completions"
