# GPU Training Instructions for Semantic Synthesizer

## Quick Start

```bash
# 1. SSH to your Vast.ai server
ssh -p 12673 root@ssh7.vast.ai -L 8080:localhost:8080

# 2. Copy semantic_engine to server (run from local machine)
scp -P 12673 -r /Users/bobbyprice/projects/KVRM/kvrm-spnc/semantic_engine root@ssh7.vast.ai:~/

# 3. On the server, install dependencies
pip install torch

# 4. Run training
cd ~/semantic_engine
python3 train_mco.py --iterations 10000 --device cuda --epistemic
```

## Expected Results

| Iterations | Success Rate | Time (GPU) | Tactics |
|------------|--------------|------------|---------|
| 100 | ~40% | ~30s | 10-15 |
| 1,000 | ~60% | ~5min | 20-30 |
| 10,000 | ~80% | ~45min | 50+ |
| 100,000 | ~90%+ | ~8hr | 100+ |

## Training Parameters

```bash
# Full training run
python3 train_mco.py \
    --iterations 10000 \
    --device cuda \
    --checkpoint 500 \
    --epistemic

# Quick test
python3 train_mco.py --iterations 100 --device cuda
```

## What Gets Trained

1. **Program Encoder** (neural network)
   - Learns to embed programs into 64-dim vectors
   - Similar programs → similar embeddings

2. **Synthesis Policy** (RL policy network)
   - Learns which actions lead to successful synthesis
   - Actions: apply_identity, introduce_square, exploit_known, etc.

3. **Tactic Memory** (persistent)
   - Remembers successful patterns
   - Saved to `tactic_memory.json`

4. **Epistemic Frontier** (discovery)
   - Finds novel combinations through bisociation
   - Detects unknown unknowns

## Monitoring

```bash
# Watch training progress
tail -f training.log

# Check GPU usage
nvidia-smi

# Check memory
watch -n 1 nvidia-smi
```

## Checkpoints

Training saves checkpoints at intervals:
- `mco_checkpoint_epoch100.pt`
- `mco_checkpoint_epoch200.pt`
- etc.

Final model: `mco_final.pt`

## After Training

Copy trained model back to local:
```bash
scp -P 12673 root@ssh7.vast.ai:~/semantic_engine/mco_final.pt .
scp -P 12673 root@ssh7.vast.ai:~/semantic_engine/tactic_memory.json .
```

## The Goal

After sufficient training:
- System discovers `SQUARE(x)` from `MUL(x,x)` instantly
- Learns patterns for `DOUBLE`, `CUBE`, `TRIANGULAR`, etc.
- Generalizes to new functions it hasn't seen
- Self-improves through continued operation
