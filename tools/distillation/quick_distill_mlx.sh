#!/usr/bin/env bash
# Local MLX LoRA fallback for Apple Silicon.
#
# Usage:
#   tools/distillation/quick_distill_mlx.sh <dataset.jsonl|split_dir>

set -euo pipefail

DATASET_INPUT="${1:-}"
if [[ -z "$DATASET_INPUT" || (! -f "$DATASET_INPUT" && ! -d "$DATASET_INPUT") ]]; then
  echo "usage: $0 <training_dataset.jsonl|split_dir>" >&2
  exit 2
fi

command -v python3 >/dev/null 2>&1 || { echo "python3 required" >&2; exit 2; }
python3 - <<'PY' >/dev/null
import importlib.util, sys
mods = ("mlx_lm", "mlx")
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    sys.stderr.write("missing Python modules: " + ", ".join(missing) + "\n")
    raise SystemExit(2)
PY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STAMP=$(date +%Y%m%d_%H%M)
SPLIT_DIR="$REPO_ROOT/artifacts/mlx_distill_${STAMP}"
ADAPTER_DIR="$REPO_ROOT/artifacts/adapters/qwen3_4b_mlx_${STAMP}"
MODEL_ID="${MODEL_ID:-mlx-community/Qwen3-4B-Instruct-2507-4bit}"
ITERS="${ITERS:-200}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
NUM_LAYERS="${NUM_LAYERS:-16}"

mkdir -p "$REPO_ROOT/artifacts" "$REPO_ROOT/artifacts/adapters"

if [[ -d "$DATASET_INPUT" ]]; then
  for split in train valid test; do
    if [[ ! -f "$DATASET_INPUT/$split.jsonl" ]]; then
      echo "split dir missing $split.jsonl: $DATASET_INPUT" >&2
      exit 2
    fi
  done
  SPLIT_DIR="$DATASET_INPUT"
else
python3 - "$DATASET_INPUT" "$SPLIT_DIR" <<'PY'
import json
import random
import sys
from pathlib import Path

src = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
if len(rows) < 20:
    raise SystemExit("need at least 20 rows for a stable split")

rng = random.Random(0)
rng.shuffle(rows)
n = len(rows)
n_valid = max(8, round(n * 0.05))
n_test = max(8, round(n * 0.05))
n_train = n - n_valid - n_test
splits = {
    "train": rows[:n_train],
    "valid": rows[n_train:n_train + n_valid],
    "test": rows[n_train + n_valid:],
}
out.mkdir(parents=True, exist_ok=True)
for name, data in splits.items():
    with (out / f"{name}.jsonl").open("w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")
print({k: len(v) for k, v in splits.items()})
PY
fi

echo "[quick-distill-mlx] training on Apple Silicon with $MODEL_ID"
echo "[quick-distill-mlx] data at $SPLIT_DIR"

python3 -m mlx_lm lora \
    --model "$MODEL_ID" \
    --train \
    --data "$SPLIT_DIR" \
    --adapter-path "$ADAPTER_DIR" \
    --batch-size "$BATCH_SIZE" \
    --iters "$ITERS" \
    --num-layers "$NUM_LAYERS" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --mask-prompt \
    --save-every 50

echo "[quick-distill-mlx] done. Adapter at $ADAPTER_DIR"
