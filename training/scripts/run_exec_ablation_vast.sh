#!/usr/bin/env bash
# =============================================================================
# run_exec_ablation_vast.sh - Full execution training ablation on vast.ai
# =============================================================================
#
# Runs the complete ablation study for the nCPU differentiable execution
# training paper. Tests three conditions across two model sizes:
#
#   Models:  Qwen/Qwen3.5-0.8B, Qwen/Qwen3.5-2B
#   Conditions:
#     1. Baseline: frozen backbone, no coprocessor, no exec loss (baseline LM)
#     2. Coprocessor-only: coprocessor injected, aux loss, NO exec loss
#     3. Execution training: coprocessor + differentiable execution loss
#
# Designed for RTX 3090 (24GB VRAM). Adjust batch sizes for other GPUs.
#
# Usage:
#   bash run_exec_ablation_vast.sh          # Full ablation
#   bash run_exec_ablation_vast.sh --quick  # Quick test (200 steps)
#
# =============================================================================

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/root/nCPU/ablation_results/${TIMESTAMP}"
LOG_FILE="${RESULTS_DIR}/ablation.log"
PROJECT_DIR="/root/nCPU"

# Parse args
QUICK=false
STEPS=2000
EVAL_EVERY=500
DATA_SIZE=5000
for arg in "$@"; do
    case $arg in
        --quick) QUICK=true; STEPS=200; EVAL_EVERY=100; DATA_SIZE=1000 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

# Tee all output to log
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "nCPU EXECUTION TRAINING ABLATION STUDY"
echo "============================================================"
echo "Timestamp:   $TIMESTAMP"
echo "Results dir: $RESULTS_DIR"
echo "Steps:       $STEPS"
echo "Quick mode:  $QUICK"
echo ""

# ── Step 1: Environment Setup ────────────────────────────────────────────────

echo "[Step 1] Environment Setup"
echo "------------------------------------------------------------"

# System info
echo "Hostname: $(hostname)"
nvidia-smi 2>/dev/null || echo "WARNING: nvidia-smi not found"
python3 --version

# Disable xet/hf_transfer to avoid download issues
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Install dependencies
echo ""
echo "Installing dependencies..."
cd "$PROJECT_DIR"

pip3 install -q --upgrade pip 2>/dev/null || true

# Core deps - torch should already be on vast.ai pytorch images
pip3 install -q transformers accelerate huggingface_hub sentencepiece protobuf 2>/dev/null
pip3 install -q einops 2>/dev/null || true

# Install nCPU in development mode
pip3 install -q -e . 2>/dev/null || {
    echo "pip install -e . failed, trying direct PYTHONPATH..."
    export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
}

# Verify torch+CUDA
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    mem_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f'VRAM: {mem_gb:.1f} GB')
"

# Verify nCPU imports
python3 -c "
from ncpu.execution_training.train import ExecutionTrainingConfig, train_execution_grounded
from ncpu.execution_training.evaluate import ExecutionEvaluator
print('nCPU execution training modules loaded OK')
"

echo ""
echo "[Step 1] Environment setup complete."
echo ""

# ── Step 2: Download Models ──────────────────────────────────────────────────

echo "[Step 2] Downloading Models"
echo "------------------------------------------------------------"

download_model() {
    local hf_id="$1"
    local local_path="$2"

    if [ -d "$local_path" ] && [ -f "$local_path/config.json" ]; then
        echo "  [CACHED] $hf_id -> $local_path"
        return 0
    fi

    echo "  [DOWNLOADING] $hf_id -> $local_path ..."
    python3 -c "
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${hf_id}',
    local_dir='${local_path}',
    local_dir_use_symlinks=False,
    ignore_patterns=['*.safetensors.index.json', '*.bin.index.json', 'consolidated*'],
)
print('  Download complete: ${hf_id}')
"
}

MODEL_08B="/root/qwen35_08b"
MODEL_2B="/root/qwen35_2b"

download_model "Qwen/Qwen3.5-0.8B" "$MODEL_08B"
download_model "Qwen/Qwen3.5-2B" "$MODEL_2B"

echo ""
echo "[Step 2] Models downloaded."
echo ""

# ── Step 3: Run Ablation ─────────────────────────────────────────────────────

echo "[Step 3] Running Ablation Study"
echo "============================================================"

# Helper function to run one training condition
run_condition() {
    local model_path="$1"
    local model_label="$2"
    local condition="$3"
    local batch_size="$4"
    local grad_accum="$5"
    local exec_weight="$6"
    local aux_weight="$7"
    local extra_flags="${8:-}"

    local run_name="${model_label}_${condition}"
    local run_dir="${RESULTS_DIR}/${run_name}"
    mkdir -p "$run_dir"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  RUN: ${run_name}"
    echo "  Model:       ${model_path}"
    echo "  Condition:   ${condition}"
    echo "  Batch:       ${batch_size} x ${grad_accum} accum = effective $(( batch_size * grad_accum ))"
    echo "  Exec weight: ${exec_weight}"
    echo "  Aux weight:  ${aux_weight}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    local start_time=$(date +%s)

    python3 -m ncpu.execution_training.run_sweep \
        --models "$model_path" \
        --ablations "$condition" \
        --steps "$STEPS" \
        --data-size "$DATA_SIZE" \
        --batch-size "$batch_size" \
        --eval-every "$EVAL_EVERY" \
        --layers -1 \
        --output-dir "$run_dir" \
        2>&1 || {
        echo "WARNING: run_sweep failed for ${run_name}, trying direct train..."

        # Fallback: call train directly
        python3 -c "
import json, sys
from ncpu.execution_training.train import ExecutionTrainingConfig, train_execution_grounded

config = ExecutionTrainingConfig(
    model_name='${model_path}',
    layers=[-1],
    steps=${STEPS},
    data_size=${DATA_SIZE},
    batch_size=${batch_size},
    grad_accum_steps=${grad_accum},
    eval_every=${EVAL_EVERY},
    exec_loss_weight=${exec_weight},
    aux_loss_weight=${aux_weight},
    freeze_backbone=True,
    confidence_aware=True,
    max_gate=0.1,
    gate_warmup_steps=200,
    n_bits=8,
    lr=1e-3,
    warmup_steps=100,
    output_dir='${run_dir}',
)

result = train_execution_grounded(config)

# Save result summary
summary = {
    'model': '${model_path}',
    'condition': '${condition}',
    'final_loss': result.final_loss,
    'final_exec_loss': result.final_exec_loss,
    'final_lm_loss': result.final_lm_loss,
    'final_aux_loss': result.final_aux_loss,
    'parse_success_rate': result.parse_success_rate,
    'exec_accuracy': result.exec_accuracy,
    'trainable_params': result.trainable_params,
    'wall_time_seconds': result.wall_time_seconds,
}
if result.eval_result:
    summary['eval'] = result.eval_result

with open('${run_dir}/result_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(json.dumps(summary, indent=2, default=str))
" 2>&1
    }

    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    echo ""
    echo "  Completed ${run_name} in ${elapsed}s"
    echo ""
}

# ── 3a. Qwen3.5-0.8B Ablation ──

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  MODEL: Qwen3.5-0.8B                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Baseline: no exec loss, just coprocessor aux loss
run_condition "$MODEL_08B" "qwen08b" "baseline" 2 4 0.0 1.0

# Coprocessor-only: aux loss but no execution loss (tests coprocessor alone)
# Uses "exec_only" ablation inverted - we set exec=0, aux=1 via baseline
# Actually use the run_sweep ablation configs
run_condition "$MODEL_08B" "qwen08b" "exec_only" 2 4 1.0 0.0

# Full execution training: coprocessor + execution loss
run_condition "$MODEL_08B" "qwen08b" "exec_plus_copro" 2 4 1.0 1.0


# ── 3b. Qwen3.5-2B Ablation ──

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  MODEL: Qwen3.5-2B                                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Smaller batch for 2B to fit in 24GB VRAM
run_condition "$MODEL_2B" "qwen2b" "baseline" 1 8 0.0 1.0
run_condition "$MODEL_2B" "qwen2b" "exec_only" 1 8 1.0 0.0
run_condition "$MODEL_2B" "qwen2b" "exec_plus_copro" 1 8 1.0 1.0


# ── Step 4: Collect Results ──────────────────────────────────────────────────

echo ""
echo "[Step 4] Collecting Results"
echo "------------------------------------------------------------"

# Create a combined results JSON
python3 -c "
import json, os, glob

results_dir = '${RESULTS_DIR}'
combined = []

for run_dir in sorted(glob.glob(os.path.join(results_dir, '*/'))):
    # Check for sweep results
    sweep_file = os.path.join(run_dir, 'sweep_results.json')
    summary_file = os.path.join(run_dir, 'result_summary.json')

    if os.path.exists(sweep_file):
        with open(sweep_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                combined.extend(data)
            else:
                combined.append(data)
    elif os.path.exists(summary_file):
        with open(summary_file) as f:
            combined.append(json.load(f))
    else:
        # Look for any JSON results in subdirs
        for jf in glob.glob(os.path.join(run_dir, '**/*.json'), recursive=True):
            try:
                with open(jf) as f:
                    d = json.load(f)
                    if isinstance(d, dict) and ('final_loss' in d or 'exec_accuracy' in d):
                        combined.append(d)
            except:
                pass

output_file = os.path.join(results_dir, 'all_results.json')
with open(output_file, 'w') as f:
    json.dump(combined, f, indent=2, default=str)

print(f'Collected {len(combined)} results -> {output_file}')
"

echo ""

# ── Step 5: Print Summary ────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "ABLATION STUDY RESULTS SUMMARY"
echo "============================================================"

python3 -c "
import json, os

results_file = '${RESULTS_DIR}/all_results.json'
if not os.path.exists(results_file):
    print('No results file found!')
    exit(0)

with open(results_file) as f:
    results = json.load(f)

if not results:
    print('No results collected.')
    exit(0)

print(f'')
print(f'{\"Model\":<25} {\"Condition\":<18} {\"ExecLoss\":>10} {\"ParseRate\":>10} {\"ExecAcc\":>10} {\"EvalAcc\":>10} {\"Time\":>8}')
print('-' * 93)

for r in results:
    model = r.get('model', 'unknown')
    # Shorten model name
    if 'qwen35_08b' in model.lower() or '0.8b' in model.lower() or 'qwen08b' in model.lower():
        model_short = 'Qwen3.5-0.8B'
    elif 'qwen35_2b' in model.lower() or '2b' in model.lower() or 'qwen2b' in model.lower():
        model_short = 'Qwen3.5-2B'
    else:
        model_short = model[-25:]

    condition = r.get('ablation', r.get('condition', 'unknown'))
    exec_loss = r.get('final_exec_loss', float('nan'))
    parse_rate = r.get('parse_success_rate', r.get('eval_parse_rate', 0))
    exec_acc = r.get('exec_accuracy', r.get('eval_exec_accuracy', 0))
    eval_acc = r.get('eval_accuracy', 0)
    wall_time = r.get('wall_time_seconds', 0)
    error = r.get('error', None)

    if error:
        print(f'{model_short:<25} {condition:<18} {\"ERROR\":>10}  {error}')
    else:
        exec_loss_s = f'{exec_loss:.4f}' if isinstance(exec_loss, (int, float)) else str(exec_loss)
        parse_s = f'{parse_rate:.1%}' if isinstance(parse_rate, (int, float)) else str(parse_rate)
        exec_s = f'{exec_acc:.1%}' if isinstance(exec_acc, (int, float)) else str(exec_acc)
        eval_s = f'{eval_acc:.1%}' if isinstance(eval_acc, (int, float)) else str(eval_acc)
        time_s = f'{wall_time:.0f}s' if isinstance(wall_time, (int, float)) else str(wall_time)
        print(f'{model_short:<25} {condition:<18} {exec_loss_s:>10} {parse_s:>10} {exec_s:>10} {eval_s:>10} {time_s:>8}')

print('=' * 93)
print()
print('Key:')
print('  baseline       = frozen backbone, coprocessor aux loss only, no exec loss')
print('  exec_only      = execution loss only, no coprocessor aux loss')
print('  exec_plus_copro = execution loss + coprocessor aux loss (full system)')
print()
"

echo ""
echo "============================================================"
echo "Results saved to: $RESULTS_DIR"
echo "  - all_results.json     (combined results)"
echo "  - ablation.log         (full log)"
echo "  - Per-run subdirs with detailed metrics"
echo ""
echo "To download results:"
echo "  vastai copy <instance_id>:${RESULTS_DIR}/ ./ablation_results/"
echo "============================================================"
