# SOME Complete Guide

## What SOME Is

SOME is the repository's self-optimizing code and reasoning stack. It started as an external generate -> verify -> retry loop and has been moving inward toward the actual goal:

- hidden internal deliberation before visible output
- task-local weight changes during inference
- segmented committed-history decode so longer hidden computation is affordable
- latent control heads that decide what hidden work to do and when to stop

The implementation is meant to be honest about where it is today:

- stronger than a plain wrapper loop
- not yet a literal full dense CPU inside a stock transformer

## Where It Fits In nCPU

SOME is not a separate project glued on top of nCPU. It sits beside the broader GPU-native stack:

- the GPU OS and microkernel provide execution substrates and verification surfaces
- the GPU-native debugging toolkit provides deep state visibility
- the self-optimizing stack uses benchmark loops, verifier outputs, and trajectory logging to train internal control and adaptation modules

That is why the project now has two complementary stories:

- the GPU-native machine itself
- the hidden self-improving controller that tries to turn part of that machine into an internal coprocessor for code and reasoning

## Current Architecture

### Hidden Controller

Main file: `ncpu/self_optimizing/internal_controller.py`

The controller runs a buffered hidden loop:

1. initialize workspace and latent state
2. think
3. write candidate
4. verify
5. patch if needed
6. commit or fail

Only the committed output is exposed by default.

### Latent Heads

Current learned or learnable hidden modules:

- latent action head
- latent halt head
- latent descriptor head
- latent memory head
- state patch head

The current runtime now conditions those heads on a richer recurrent-memory feature vector, not just a tiny pooled memory summary. That matters because the controller is starting to use learned hidden state more directly when choosing actions, halting, and weight-update behavior.

These are trained from JSONL trajectories captured during hidden SOME runs.

### Task-Local Fast Weights

Main files:

- `task_local_fast_weights.py`
- `ncpu_adaptation_backend.py`

These modules apply bounded per-task weight changes during inference. The base model stays mostly fixed; the temporary task-local adapters are the part that gets updated inside the hidden loop.

### Segmented Decode Path

Main files:

- `segmented_kv_cache.py`
- `descriptor_decode_runtime.py`

This is the cache/decoding side of the "computer inside the model" goal: the system keeps a recent exact token window and compresses older hidden history into committed descriptors so longer internal computation does not blow up attention cost.

### Training and Bundles

Main files:

- `trajectory_dataset.py`
- `controller_training.py`
- `train_internal_controller.py`
- `controller_bundle.py`
- `controller_runtime.py`

The training pipeline turns hidden trajectories into:

- response SFT data
- text action data
- latent action/halt data
- latent memory data
- latent descriptor data
- state-patch data

These are then packaged into controller bundles that benchmarks and the API server can load directly.

## What Has Actual Evidence

### Official Benchmarks

Finished benchmark results already show that verifier-guided hidden inference matters:

- HumanEval+
  - `qwen3.5:4b 147/164 -> 154/164`
  - `qwen3.5:9b 144/164 -> 156/164`
  - `qwen3.5:27b 153/164 -> 156/164`
- BigCodeBench-Hard
  - `qwen3.5:9b 33/148 -> 49/148`

That is enough to justify deeper internalization work, especially because the gains are strongest on the smaller or weaker models.

### Internal-Memory Proof

The first meaningful memory-head evidence uses a fresh synthetic corpus of hidden-controller traces generated after memory state existed in the runtime.

Held-out validation metrics:

- zero baseline MSE: `0.0006947`
- learned memory-head MSE: `0.0001163`
- relative MSE improvement: `83.26%`
- cosine similarity improvement: `0.0 -> 0.9028`

That is not yet the final proof on a large real-model corpus, but it is enough to show that the learned recurrent memory path is doing something real and worth funding/exploring further.

## What To Say Publicly

Good claims:

- SOME is an execution-grounded hidden-controller system for code and reasoning.
- It combines verification, task-local adaptation, segmented decode, and latent internal control.
- It already improves official coding-benchmark results and shows promising internal-memory evidence.

Bad claims:

- "the entire dense model is already a literal CPU"
- "this is the first self-improving system of its kind"
- "the old wrapper loop is the final architecture"

## Best Files To Read

- `docs/SOME_ARCHITECTURE.md`
- `docs/WEIGHT_CPU_ARCHITECTURE.md`
- `docs/SOME_RESULTS.md`
- `ncpu/self_optimizing/internal_controller.py`
- `ncpu/self_optimizing/task_local_fast_weights.py`
- `ncpu/self_optimizing/descriptor_decode_runtime.py`
- `ncpu/self_optimizing/train_internal_controller.py`
- `ncpu/self_optimizing/run_real_memory_proof.py`

## Quick Start

Prepare training data:

```bash
python3 ncpu/self_optimizing/prepare_internal_training_data.py \
  --trajectory-root benchmarks/internal_trajectories \
  --output benchmarks/internal_training/some_sft.jsonl
```

Build or train a controller bundle:

```bash
python3 ncpu/self_optimizing/train_internal_controller.py \
  --trajectory-root benchmarks/internal_trajectories \
  --output-dir training/internal_controller_bootstrap \
  --prepare-only
```

Run buffered hidden inference through the API:

```bash
uvicorn ncpu.self_optimizing.api_server:app --reload
```

Run the real latent-memory proof path:

```bash
python3 ncpu/self_optimizing/run_real_memory_proof.py \
  --provider local \
  --model qwen3.5:4b \
  --output-dir training/real_memory_proof_qwen35_4b
```
