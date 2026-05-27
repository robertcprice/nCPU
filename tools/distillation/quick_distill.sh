#!/usr/bin/env bash
# Quick LoRA distillation on vast.ai RTX 4090 — designed for the
# coreutils harvest corpus (~200 rows) where a full auto_distill.sh
# pipeline is overkill.
#
# Differences from auto_distill.sh:
#   - Uses Qwen/Qwen3-4B-Instruct-2507.
#   - RTX 4090 interruptible (~$0.30/hr) instead of A100.
#   - Tarball transfer (no repo rsync) — sidesteps our observed
#     multi-GB rsync failures.
#   - 1 epoch instead of 3.
#   - Pulls adapter back inline via ssh cat.
#
# Usage:
#   tools/distillation/quick_distill.sh <dataset.jsonl>
#
# Cost: ~$0.25-0.40 total (~30-40 min wall).

set -euo pipefail

DATASET="${1:-}"
if [[ -z "$DATASET" || ! -f "$DATASET" ]]; then
  echo "usage: $0 <training_dataset.jsonl>" >&2
  exit 2
fi

command -v vastai >/dev/null 2>&1 || { echo "vastai CLI required" >&2; exit 2; }
[[ -f "$HOME/.ssh/id_rsa" ]] || { echo "~/.ssh/id_rsa missing" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "[quick-distill] launching RTX 4090 for Qwen3-4B LoRA..."
"$REPO_ROOT/tools/vastai/launch.sh" qwen3.5-4b 2>&1 | tee /tmp/quick_distill_launch.log
INSTANCE_ID=$(grep -oE 'instance [0-9]+' /tmp/quick_distill_launch.log | head -1 | awk '{print $2}')
[[ -z "$INSTANCE_ID" ]] && { echo "[quick-distill] provision failed" >&2; exit 1; }
echo "[quick-distill] instance $INSTANCE_ID provisioned"

cleanup() {
  local ec=$?
  echo "[quick-distill] destroying instance $INSTANCE_ID (exit $ec)..."
  vastai destroy instance "$INSTANCE_ID" 2>&1 || true
}
trap cleanup EXIT

INFO=$(vastai show instance "$INSTANCE_ID" --raw)
HOST=$(echo "$INFO" | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["ssh_host"])')
PORT=$(echo "$INFO" | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["ssh_port"])')

echo "[quick-distill] waiting for sshd on $HOST:$PORT (up to 20 min)..."
for attempt in $(seq 1 240); do
  if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes \
       -i ~/.ssh/id_rsa -p "$PORT" "root@$HOST" true 2>/dev/null; then
    echo "[quick-distill] sshd up after ${attempt} tries"
    break
  fi
  if (( attempt == 240 )); then
    echo "[quick-distill] sshd never came up" >&2; exit 1
  fi
  sleep 5
done

echo "[quick-distill] uploading dataset..."
scp -P "$PORT" -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no \
    "$DATASET" "root@$HOST:/tmp/train_data.jsonl"

echo "[quick-distill] launching training..."
REMOTE_LOG="/tmp/quick_distill_remote.log"
ssh -p "$PORT" -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no \
    "root@$HOST" bash <<'REMOTE' 2>&1 | tee "$REMOTE_LOG"
set -euxo pipefail

python3 -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

# Keep torch from the base image, but isolate the rest of the Python
# stack so broken global metadata cannot leak into the run.
python3 -m venv --clear --system-site-packages /workspace/distill-venv
. /workspace/distill-venv/bin/activate
python -m pip install -q --upgrade pip setuptools wheel typing_extensions
python -m pip install -q --upgrade \
    tokenizers==0.22.1 \
    transformers==5.5.4 \
    peft==0.18.1 \
    datasets==4.6.0 \
    accelerate==1.12.0 \
    safetensors \
    sentencepiece
python -c "import torch, transformers, peft, tokenizers, datasets, accelerate; print('torch:', torch.__version__, 'transformers:', transformers.__version__, 'peft:', peft.__version__, 'tokenizers:', tokenizers.__version__, 'datasets:', datasets.__version__, 'accelerate:', accelerate.__version__)"

mkdir -p /workspace/adapter
echo "=== BEGIN TRAINING ==="
python - <<'PYEOF'
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
print(f"[distill] loading tokenizer {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print(f"[distill] loading model {MODEL}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True,
)

print("[distill] wrapping with LoRA")
peft_cfg = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM", lora_dropout=0.05, bias="none",
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()
model.train()

# Load + tokenize dataset manually.
rows = [json.loads(l) for l in
        Path("/tmp/train_data.jsonl").read_text().splitlines() if l.strip()]
print(f"[distill] dataset: {len(rows)} rows")

def format_example(r):
    text = (f"<|im_start|>user\n{r['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n{r['completion']}<|im_end|>")
    enc = tok(text, truncation=True, max_length=1024,
              return_tensors="pt", padding=False)
    return enc.input_ids[0], enc.attention_mask[0]

examples = [format_example(r) for r in rows]

# Vanilla training loop with gradient accumulation.
import torch.optim as optim
optimizer = optim.AdamW(model.parameters(), lr=2e-4)
ACC_STEPS = 8
N_EPOCHS = 1

print("[distill] starting training...")
step = 0
running_loss = 0.0
for epoch in range(N_EPOCHS):
    for i, (ids, mask) in enumerate(examples):
        ids = ids.unsqueeze(0).to(model.device)
        mask = mask.unsqueeze(0).to(model.device)
        out = model(input_ids=ids, attention_mask=mask, labels=ids)
        loss = out.loss / ACC_STEPS
        loss.backward()
        running_loss += loss.item() * ACC_STEPS
        if (i + 1) % ACC_STEPS == 0:
            optimizer.step(); optimizer.zero_grad()
            step += 1
            if step % 3 == 0:
                avg = running_loss / ACC_STEPS / 3
                print(f"  step {step:>3}  loss={avg:.4f}")
                running_loss = 0.0
    # flush tail batch
    if (len(examples) % ACC_STEPS) != 0:
        optimizer.step(); optimizer.zero_grad()

print("[distill] saving adapter...")
model.save_pretrained("/workspace/adapter/final")
tok.save_pretrained("/workspace/adapter/final")
print("[distill] adapter saved")
PYEOF
echo "=== END TRAINING ==="

# Package the adapter for transfer back.
test -f /workspace/adapter/final/adapter_config.json
cd /workspace && tar czf /tmp/adapter.tar.gz adapter/final/
ls -la /tmp/adapter.tar.gz
REMOTE

echo "[quick-distill] pulling adapter..."
STAMP=$(date +%Y%m%d_%H%M)
mkdir -p "$REPO_ROOT/artifacts/adapters"
scp -P "$PORT" -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no \
    "root@$HOST:/tmp/adapter.tar.gz" \
    "$REPO_ROOT/artifacts/adapters/qwen3_4b_${STAMP}.tar.gz" || \
    echo "[quick-distill] adapter pull failed — training may not have completed"

echo "[quick-distill] done. Adapter at artifacts/adapters/qwen3_4b_${STAMP}.tar.gz"
ls -la "$REPO_ROOT/artifacts/adapters/"
