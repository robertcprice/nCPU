#!/usr/bin/env bash
# =============================================================================
# vast_setup.sh - Setup and run nCPU training on a vast.ai GPU instance
# =============================================================================
#
# Usage:
#   1. Rent a GPU instance on vast.ai (RTX 3090/4090/A100 recommended)
#   2. SSH into the instance
#   3. Upload this script and the nCPU project (or clone from git)
#   4. Run: bash vast_setup.sh
#
# Or use the step-by-step commands below.
#
# Recommended vast.ai instance:
#   - GPU: RTX 3090 or better (24GB+ VRAM)
#   - Image: pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
#   - Disk: 20GB minimum
#   - RAM: 16GB minimum
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "nCPU Neural Training - vast.ai GPU Setup"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Script dir:   $SCRIPT_DIR"

# ─── Step 1: System info ──────────────────────────────────────────────────
echo ""
echo "[Step 1] System Information"
echo "----------------------------------------------"
echo "Hostname: $(hostname)"
echo "OS:       $(uname -s -r)"
echo "Python:   $(python3 --version 2>&1 || echo 'not found')"
nvidia-smi 2>/dev/null || echo "No NVIDIA GPU detected (will use CPU/MPS)"

# ─── Step 2: Install dependencies ──────────────────────────────────────────
echo ""
echo "[Step 2] Installing Dependencies"
echo "----------------------------------------------"

# Check if pip is available
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "Installing pip..."
    python3 -m ensurepip --upgrade 2>/dev/null || apt-get update && apt-get install -y python3-pip
fi

PIP_CMD="pip3"
command -v pip3 &> /dev/null || PIP_CMD="pip"

# Install PyTorch (CUDA 12.1 for vast.ai instances)
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "PyTorch with CUDA already installed"
else
    echo "Installing PyTorch..."
    $PIP_CMD install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null \
        || $PIP_CMD install torch torchvision
fi

# Verify PyTorch and CUDA
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version:    {torch.version.cuda}')
    print(f'GPU:             {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory:      {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS available:   True (Apple Silicon)')
else:
    print('WARNING: No GPU detected, training will be slow')
"

# ─── Step 3: Verify project structure ──────────────────────────────────────
echo ""
echo "[Step 3] Verifying Project Structure"
echo "----------------------------------------------"

cd "$PROJECT_ROOT"

# Check critical files exist
REQUIRED_FILES=(
    "ncpu/os/assembler.py"
    "ncpu/os/compiler.py"
    "ncpu/os/language.py"
    "ncpu/os/device.py"
    "ncpu/os/__init__.py"
    "training/train_neuros_gpu.py"
)

ALL_PRESENT=true
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "  [OK] $f"
    else
        echo "  [MISSING] $f"
        ALL_PRESENT=false
    fi
done

# Check programs directory
ASM_COUNT=$(ls programs/*.asm 2>/dev/null | wc -l)
echo "  [INFO] Found $ASM_COUNT .asm programs in programs/"

if [ "$ALL_PRESENT" = false ]; then
    echo ""
    echo "ERROR: Missing required files. Ensure the full nCPU project is present."
    echo "Clone with: git clone <your-repo-url> nCPU"
    exit 1
fi

# Ensure models/os directory exists
mkdir -p models/os
echo "  [OK] models/os/ directory ready"

# ─── Step 4: Run training ──────────────────────────────────────────────────
echo ""
echo "[Step 4] Starting Training"
echo "----------------------------------------------"
echo "Logs will be written to: training/training.log"
echo ""

# Parse command-line args to pass through to the training script
EXTRA_ARGS="${@}"
if [ -z "$EXTRA_ARGS" ]; then
    # Default: full training with GPU-appropriate settings
    EXTRA_ARGS="--epochs-codegen 3000 --epochs-tokenizer 800 --epochs-optimizer 500 --hidden-dim 128 --batch-size 1024 --num-immediates 128 --target-accuracy 0.95"
fi

echo "Training args: $EXTRA_ARGS"
echo ""

# Run training with output to both console and log file
python3 training/train_neuros_gpu.py $EXTRA_ARGS 2>&1 | tee training/training.log

TRAIN_EXIT=$?

# ─── Step 5: Results ──────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "Training Complete (exit code: $TRAIN_EXIT)"
echo "=============================================="

if [ $TRAIN_EXIT -eq 0 ]; then
    echo ""
    echo "Trained models saved to:"
    ls -lh models/os/*.pt 2>/dev/null || echo "  (no .pt files found)"
    ls -lh models/os/training_stats.json 2>/dev/null || echo "  (no stats file found)"

    echo ""
    echo "To download results to your local machine:"
    echo "  scp -r <instance>:$PROJECT_ROOT/models/os/ ./models/os/"
    echo ""
    echo "Or use vast.ai CLI:"
    echo "  vastai copy <instance_id>:$PROJECT_ROOT/models/os/ ./models/os/"
else
    echo "Training failed! Check training/training.log for details."
    echo "Last 20 lines of log:"
    tail -20 training/training.log
fi
