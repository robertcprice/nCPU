#!/bin/bash
#
# EGDC GPU Training Deployment for vast.ai
# 
# Usage:
#   1. Run this script from the nCPU repo root
#   2. It will rent a GPU, upload code, and start training
#   3. Training logs are streamed to stdout
#
# Prerequisites: vastai CLI configured with API key
#
set -e

echo "=========================================="
echo "  EGDC GPU Deployment to vast.ai"
echo "=========================================="

# --- Config ---
GPU_MODEL="RTX_4090"
DISK_GB=100
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
MAX_PRICE=0.35  # $/hr

# --- Package code ---
echo "[1/5] Packaging code..."
DEPLOY_DIR=$(mktemp -d)
tar czf "$DEPLOY_DIR/egdc_deploy.tar.gz" \
    egdc/ \
    checkpoints/egdc/best.pt \
    checkpoints/egdc/proxy_best.pt \
    2>/dev/null || true
echo "  Package: $(du -h "$DEPLOY_DIR/egdc_deploy.tar.gz" | cut -f1)"

# --- Write remote training script ---
cat > "$DEPLOY_DIR/gpu_train.py" << 'TRAINSCRIPT'
#!/usr/bin/env python3
"""EGDC GPU Training - runs on vast.ai with CUDA."""
import sys, os, time, json
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

print("=" * 70)
print("EGDC GPU Training")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print("=" * 70, flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================================================================
# PHASE 1: Scale up nCPU ISA model (small config)
# ===================================================================
print("\n>>> Phase 1: Training larger nCPU ISA model...", flush=True)

from egdc.model import MaskedDiffusionTransformer, ModelConfig
from egdc.dataset import NCPUDataset
from egdc.train import train

# Use small config (6 layers, 384 hidden, 6 heads)
cfg = ModelConfig.small()
model = MaskedDiffusionTransformer(cfg)
params = sum(p.numel() for p in model.parameters())
print(f"  Model: {params:,} params (small)", flush=True)

# Generate larger dataset
ds = NCPUDataset(num_samples=50000)
loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
print(f"  Dataset: {len(ds)} programs, {len(loader)} batches/epoch", flush=True)

train(model, loader, epochs=30, lr=5e-4, warmup_steps=500,
      log_every=50, checkpoint_every=1000,
      checkpoint_dir="checkpoints/egdc_gpu", device=device)

# ===================================================================
# PHASE 2: GRPO RL fine-tuning on GPU
# ===================================================================
print("\n>>> Phase 2: GRPO RL fine-tuning...", flush=True)

from egdc.grpo import GRPOTrainer, GRPOConfig

grpo_cfg = GRPOConfig(
    lr=1e-5,
    kl_weight=0.05,
    num_samples_per_spec=8,
    num_specs_per_batch=20,
    mask_rate=0.5,
    temperature=0.5,
    num_sampling_steps=48,
    num_epochs=10,
    log_interval=1,
)
trainer = GRPOTrainer(model, grpo_cfg, device)
history = trainer.train(num_epochs=10)
torch.save(model.state_dict(), "checkpoints/egdc_gpu/grpo_small.pt")
print("  GRPO complete", flush=True)

# ===================================================================
# PHASE 3: Evaluate
# ===================================================================
print("\n>>> Phase 3: Evaluation...", flush=True)

from egdc.sampler import generate as unguided_generate
from egdc.guided_sampler import guided_generate
from egdc.evaluate import execute_program
from egdc.data_generator import NCPUDataGenerator
from egdc.dataset import NCPUDataset as DS2

ds2 = DS2(num_samples=1)
gen = NCPUDataGenerator()
data = gen.generate_dataset(50, balanced=True)

model.eval()
model.to(device)

K = 10
STEPS = 64
passed = 0; executed = 0

for spec, ref in data:
    st = ds2._encode_spec(spec)
    stensor = torch.tensor([st], dtype=torch.long)
    tc = spec['test_cases'][0]
    ir = {i: v for i, (_, v) in enumerate(tc['inputs'].items())}
    exp = tc['expected_output']
    tp = te = False
    for _ in range(K):
        tok = unguided_generate(model, stensor, seq_len=128, num_steps=STEPS,
                                temperature=0.3, constrained=True, device=device)
        r = execute_program(tok[0].tolist(), ir)
        if r:
            te = True
            if r.get(0) == exp: tp = True; break
    passed += int(tp); executed += int(te)

print(f"  GRPO small model: pass@{K}={passed}/50 ({100*passed/50:.0f}%)  "
      f"exec={executed}/50 ({100*executed/50:.0f}%)", flush=True)

# ===================================================================
# PHASE 4: Train Python model on large code corpus
# ===================================================================
print("\n>>> Phase 4: Python Code Diffusion...", flush=True)

try:
    from datasets import load_dataset
    print("  Downloading The Stack (Python subset)...", flush=True)
    code_ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/python",
                           split="train", streaming=True)

    from egdc.python_model import PythonMaskedDiffusion, PythonDiffusionConfig
    from egdc.python_tokenizer import PythonCodeTokenizer

    # Larger Python model
    pcfg = PythonDiffusionConfig(
        vocab_size=260, hidden_dim=512, num_layers=8, num_heads=8,
        ff_dim=2048, max_seq_len=1024, dropout=0.1, timestep_dim=256,
    )
    pmodel = PythonMaskedDiffusion(pcfg).to(device)
    pparams = sum(p.numel() for p in pmodel.parameters())
    print(f"  Python model: {pparams:,} params", flush=True)

    ptok = PythonCodeTokenizer()
    optimizer = torch.optim.AdamW(pmodel.parameters(), lr=3e-4, weight_decay=0.01)
    pmodel.train()

    # Stream training data
    batch_tokens = []
    step = 0
    total_loss = 0
    for example in code_ds:
        content = example.get("content", "")
        if len(content) < 50 or len(content) > 5000:
            continue

        tokens = ptok.encode(content)
        if len(tokens) > 1024:
            tokens = tokens[:1024]
        while len(tokens) < 1024:
            tokens.append(ptok.PAD)
        batch_tokens.append(tokens)

        if len(batch_tokens) >= 32:
            # Create masked batch
            batch = torch.tensor(batch_tokens, dtype=torch.long, device=device)
            t = torch.rand(batch.shape[0], device=device)
            mask = torch.rand_like(batch.float()) < (t.unsqueeze(1) * 0.5 + 0.25)
            is_pad = (batch == ptok.PAD)
            mask = mask & ~is_pad
            noisy = batch.clone()
            noisy[mask] = ptok.MASK

            optimizer.zero_grad()
            logits = pmodel(noisy, t)
            lf = logits.view(-1, logits.shape[-1])
            tf = batch.view(-1)
            mf = mask.view(-1)
            if mf.any():
                loss = F.cross_entropy(lf[mf], tf[mf])
                loss.backward()
                nn.utils.clip_grad_norm_(pmodel.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            step += 1
            batch_tokens = []

            if step % 100 == 0:
                avg = total_loss / step
                print(f"  Step {step}: loss={avg:.4f}", flush=True)

            if step >= 5000:
                break

    torch.save(pmodel.state_dict(), "checkpoints/egdc_gpu/python_model.pt")
    print(f"  Python model trained ({step} steps)", flush=True)

    # HumanEval evaluation
    print("  Evaluating on HumanEval...", flush=True)
    from egdc.humaneval import load_humaneval, evaluate_humaneval
    metrics = evaluate_humaneval(pmodel, ptok, num_problems=20, k=5,
                                 seq_len=1024, num_steps=128, temperature=0.5,
                                 device=device)
    print(f"  HumanEval pass@5: {metrics.get('pass_at_5', 0):.1f}%", flush=True)

except Exception as e:
    import traceback
    print(f"  Python training error: {e}", flush=True)
    traceback.print_exc()

print("\n" + "=" * 70)
print("ALL TRAINING COMPLETE")
print("=" * 70, flush=True)

# Save all results
results = {
    "ncpu_isa": {
        "model": "small",
        "pass_at_10": f"{passed}/50",
        "exec_rate": f"{executed}/50",
    },
}
with open("checkpoints/egdc_gpu/results.json", "w") as f:
    json.dump(results, f, indent=2)
TRAINSCRIPT

echo "  Training script ready"

# --- Find and rent GPU ---
echo "[2/5] Finding cheapest $GPU_MODEL..."
OFFER=$(vastai search offers "gpu_name=$GPU_MODEL num_gpus=1 inet_down>200 disk_space>$DISK_GB dph<=$MAX_PRICE reliability>98" --limit 1 --raw 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['id'] if d else '')" 2>/dev/null)

if [ -z "$OFFER" ]; then
    echo "  No offers found under \$$MAX_PRICE/hr. Try increasing MAX_PRICE."
    exit 1
fi
echo "  Found offer: $OFFER"

echo "[3/5] Renting instance..."
RESULT=$(vastai create instance "$OFFER" --image "$IMAGE" --disk "$DISK_GB" 2>&1)
INSTANCE_ID=$(echo "$RESULT" | python3 -c "import sys,json; d=json.loads(sys.stdin.read().split('{',1)[1].rsplit('}',1)[0].join(['[{','}]'])); print(d[0].get('new_contract',''))" 2>/dev/null || echo "")

if [ -z "$INSTANCE_ID" ]; then
    echo "  Instance creation output: $RESULT"
    echo "  Extract the instance ID from above and run:"
    echo "    scp -P PORT egdc_deploy.tar.gz root@HOST:/root/"
    echo "    scp -P PORT gpu_train.py root@HOST:/root/"
    echo "    ssh -p PORT root@HOST 'cd /root && tar xzf egdc_deploy.tar.gz && python gpu_train.py'"
    exit 0
fi

echo "  Instance: $INSTANCE_ID"

echo "[4/5] Waiting for instance to start..."
for i in $(seq 1 60); do
    STATUS=$(vastai show instances 2>/dev/null | grep "$INSTANCE_ID" | awk '{print $3}')
    if [ "$STATUS" = "running" ]; then
        SSH_INFO=$(vastai show instances 2>/dev/null | grep "$INSTANCE_ID" | awk '{print $10, $11}')
        echo "  Running! SSH: $SSH_INFO"
        break
    fi
    echo "  Status: $STATUS (attempt $i/60)"
    sleep 10
done

echo "[5/5] Upload and start training"
echo ""
echo "=== MANUAL STEPS ==="
echo "Run these commands to deploy:"
echo ""
SSH_HOST=$(vastai show instances 2>/dev/null | grep "$INSTANCE_ID" | awk '{print $10}')
SSH_PORT=$(vastai show instances 2>/dev/null | grep "$INSTANCE_ID" | awk '{print $11}')
echo "  scp -P $SSH_PORT $DEPLOY_DIR/egdc_deploy.tar.gz root@$SSH_HOST:/root/"
echo "  scp -P $SSH_PORT $DEPLOY_DIR/gpu_train.py root@$SSH_HOST:/root/"
echo "  ssh -p $SSH_PORT root@$SSH_HOST 'cd /root && tar xzf egdc_deploy.tar.gz && pip install datasets transformers && nohup python -u gpu_train.py > train.log 2>&1 &'"
echo ""
echo "Monitor with:"
echo "  ssh -p $SSH_PORT root@$SSH_HOST 'tail -f /root/train.log'"
echo ""
echo "When done, download results:"
echo "  scp -P $SSH_PORT root@$SSH_HOST:/root/checkpoints/egdc_gpu/* checkpoints/egdc_gpu/"
echo "  vastai destroy instance $INSTANCE_ID"
