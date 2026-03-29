#!/bin/bash
# ONE-COMMAND GPU DEPLOYMENT
# Paste this into a vast.ai instance terminal after SSHing in
#
# From your Mac:
#   1. vastai create instance OFFER_ID --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel --disk 100
#   2. Wait for it to start, get SSH info from: vastai show instances
#   3. scp -P PORT /path/to/nCPU/egdc_deploy.tar.gz root@HOST:/root/
#   4. SSH in: ssh -p PORT root@HOST
#   5. Run: bash gpu_quickstart.sh

set -e
cd /root

echo "=== EGDC GPU Quick Start ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Unpack if tarball exists
if [ -f egdc_deploy.tar.gz ]; then
    echo "Unpacking code..."
    tar xzf egdc_deploy.tar.gz
fi

# Install deps
pip install datasets transformers 2>&1 | tail -1

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
python -c "from egdc.model import MaskedDiffusionTransformer; print('EGDC imports OK')"

# Start training
echo "Starting GPU training..."
nohup python -u gpu_train.py > train.log 2>&1 &
echo "Training started in background. Monitor with: tail -f train.log"
