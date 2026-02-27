#!/bin/bash
# Deploy and train on H200 GPU server

set -e

# Configuration
SERVER="root@ssh7.vast.ai"
PORT="12673"
REMOTE_DIR="~/semantic_engine"
LOCAL_DIR="/Users/bobbyprice/projects/KVRM/kvrm-spnc/semantic_engine"

echo "============================================================"
echo "DEPLOYING TO H200 GPU SERVER"
echo "============================================================"

# Step 1: Create remote directory
echo "[1/4] Creating remote directory..."
ssh -p $PORT $SERVER "mkdir -p $REMOTE_DIR"

# Step 2: Copy all Python files
echo "[2/4] Copying files to server..."
scp -P $PORT $LOCAL_DIR/*.py $SERVER:$REMOTE_DIR/

# Step 3: Install dependencies on server
echo "[3/4] Installing dependencies..."
ssh -p $PORT $SERVER "cd $REMOTE_DIR && pip install torch sympy numpy --quiet"

# Step 4: Start training
echo "[4/4] Starting training..."
echo ""
echo "============================================================"
echo "TRAINING STARTED ON H200"
echo "============================================================"
echo ""
echo "To monitor training:"
echo "  ssh -p $PORT $SERVER"
echo "  cd $REMOTE_DIR"
echo "  tail -f training.log"
echo ""
echo "To check GPU usage:"
echo "  nvidia-smi"
echo ""

# Run training in background with logging
ssh -p $PORT $SERVER "cd $REMOTE_DIR && nohup python3 train_h200.py \
    --epochs 1000 \
    --batch_size 256 \
    --embedding_dim 512 \
    --hidden_dim 1024 \
    --num_layers 6 \
    --target_accuracy 1.0 \
    > training.log 2>&1 &"

echo "Training launched in background!"
echo "Check logs with: ssh -p $PORT $SERVER 'tail -f $REMOTE_DIR/training.log'"
