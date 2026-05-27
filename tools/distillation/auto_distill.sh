#!/usr/bin/env bash
# Weekly auto-distillation pipeline:
#
#   1. Export the latest verified-solution cache as a training JSONL.
#   2. SSH to (or provision) a vast.ai A100 instance.
#   3. LoRA-fine-tune Qwen3.5-4B-Instruct on the export.
#   4. Pull the adapter back to artifacts/adapters/.
#   5. Run the agent benchmark against the fine-tuned model +
#      update artifacts/distilled_full.md.
#
# Intended to run via .github/workflows/auto_distill.yml on a cron.
# Locally, pass --dry-run to see the command sequence without
# invoking vast.ai.
#
# Requires:
#   - VAST_API_KEY env var (weekly cron secret)
#   - ANTHROPIC_API_KEY (for the post-distill comparison run)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=0
MODEL="Qwen/Qwen3.5-4B-Instruct"
CACHE_PATH="${NSYNTH_LLM_CACHE_PATH:-$HOME/.nsynth_llm_solutions.tsv}"
MIN_SUCCESS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    --cache) CACHE_PATH="$2"; shift 2 ;;
    --min-success) MIN_SUCCESS="$2"; shift 2 ;;
    *) echo "unknown: $1"; exit 2 ;;
  esac
done

# ── [1] Export dataset ──────────────────────────────────────────────────────

OUT_DATASET="$REPO_ROOT/artifacts/distillation_dataset.jsonl"
echo "[distill] exporting cache from $CACHE_PATH → $OUT_DATASET"
if [[ $DRY_RUN -eq 0 ]]; then
  NSYNTH_LLM_CACHE_PATH="$CACHE_PATH" \
    python3 tools/export_distillation_dataset.py \
      --format hf --min-success "$MIN_SUCCESS" \
      --out "$OUT_DATASET"
fi
SAMPLES=$(wc -l < "$OUT_DATASET" 2>/dev/null || echo 0)
echo "[distill] exported $SAMPLES training samples"

if [[ "$SAMPLES" -lt 50 ]]; then
  echo "[distill] cache too small ($SAMPLES < 50); skipping this run."
  exit 0
fi

# ── [2] Provision vast.ai ───────────────────────────────────────────────────

echo "[distill] provisioning vast.ai A100 for $MODEL LoRA…"
if [[ $DRY_RUN -eq 0 ]]; then
  tools/vastai/launch.sh distill-9b > /tmp/vast_launch.log
  INSTANCE_ID=$(grep -oE 'created instance [0-9]+' /tmp/vast_launch.log \
    | head -1 | grep -oE '[0-9]+')
  SSH_HOST=$(grep -oE 'root@[^ ]+' /tmp/vast_launch.log | head -1 | sed 's/root@//')
  SSH_PORT=$(grep -oE '-p [0-9]+' /tmp/vast_launch.log | head -1 | sed 's/-p //')
else
  INSTANCE_ID=0
  SSH_HOST=dry-run.invalid
  SSH_PORT=22
fi

trap 'if [[ $DRY_RUN -eq 0 && -n "$INSTANCE_ID" && "$INSTANCE_ID" != "0" ]]; then \
  echo "[distill] destroying vast instance $INSTANCE_ID"; \
  vastai destroy instance "$INSTANCE_ID" || true; \
fi' EXIT

# ── [3] Sync + train ────────────────────────────────────────────────────────

echo "[distill] rsync repo + dataset to vast ($SSH_HOST:$SSH_PORT)…"
if [[ $DRY_RUN -eq 0 ]]; then
  rsync -a --exclude='.git' --exclude='target' --exclude='artifacts/vastai' \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
    . "root@$SSH_HOST:/workspace/nsynth/"
fi

echo "[distill] launching LoRA training on the instance…"
TRAIN_CMD=$(cat <<EOF
set -e
cd /workspace/nsynth
pip install -q peft trl bitsandbytes accelerate
python3 - <<PYEOF
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
import torch

MODEL = "$MODEL"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="flash_attention_2",
)
ds = load_dataset(
    "json", data_files="artifacts/distillation_dataset.jsonl", split="train"
).map(lambda r: {"text": f"<|user|>\\n{r['prompt']}\\n<|assistant|>\\n{r['completion']}"})

trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds,
    peft_config=LoraConfig(r=32, lora_alpha=64,
                           target_modules=["q_proj","v_proj","o_proj"]),
    args=SFTConfig(
        output_dir="/workspace/adapter",
        max_seq_length=2048,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5, bf16=True,
        logging_steps=10, save_steps=200,
    ),
)
trainer.train()
trainer.save_model("/workspace/adapter/final")
print("[distill] adapter saved to /workspace/adapter/final")
PYEOF
EOF
)

if [[ $DRY_RUN -eq 0 ]]; then
  ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no "root@$SSH_HOST" "$TRAIN_CMD"
fi

# ── [4] Pull adapter + evaluate ─────────────────────────────────────────────

echo "[distill] rsync adapter back…"
mkdir -p "$REPO_ROOT/artifacts/adapters"
if [[ $DRY_RUN -eq 0 ]]; then
  rsync -a -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
    "root@$SSH_HOST:/workspace/adapter/final/" \
    "$REPO_ROOT/artifacts/adapters/$(date +%Y%m%d)_qwen35_4b/"
fi

echo "[distill] done. adapter at artifacts/adapters/$(date +%Y%m%d)_qwen35_4b/"

# Evaluation of the fine-tuned model happens on a second vast.ai
# instance (the training instance is destroyed by the exit trap).
# To avoid making this script unboundedly long, we emit a ready-to-run
# command and stop here.

cat <<EOF

Next:  eval the distilled adapter back on vast.ai:
  tools/vastai/launch.sh qwen3.5-4b    # small eval instance
  # then on the instance:
  bash tools/vastai/setup_and_run.sh \\
      --model Qwen/Qwen3.5-4B-Instruct \\
      --adapter /workspace/adapters/$(date +%Y%m%d)_qwen35_4b \\
      --full-humaneval
  # rsync the results into artifacts/distilled_full.md, commit, done.
EOF
