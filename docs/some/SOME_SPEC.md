# Self-Optimizing Machine Engine (SOME) - Current Spec

## Status

This document describes the current SOME runtime in this repository. It replaces the older "tensor descriptors everywhere" description.

Current SOME is:

- a hidden-controller runtime for code and reasoning tasks
- execution-grounded through verification and benchmark loops
- partially internalized through latent heads, recurrent memory, task-local fast weights, and segmented decode

Current SOME is not:

- a claim that the full dense backbone is already a general-purpose CPU
- a claim that the old wrapper loop is the final architecture

## Runtime Contract

### High-Level Flow

1. Receive a task prompt.
2. Initialize a hidden workspace and latent controller state.
3. Run hidden internal steps:
   - `think`
   - `write`
   - `verify`
   - `patch`
   - `commit`
   - `fail`
4. Apply task-local descriptor-driven weight updates when hidden state indicates they are useful.
5. Use verifier results and hidden policies to decide whether to continue hidden computation or commit.
6. Emit only the committed final output.

## Core Components

### 1. Hidden Controller

File: `ncpu/self_optimizing/internal_controller.py`

Responsibilities:

- maintain the hidden deliberation loop
- log hidden workspace steps
- manage retries and verifier feedback
- coordinate latent action, halt, descriptor, patch, and memory heads
- control task-local fast-weight sessions

### 2. Hidden Workspace

Files:

- `ncpu/self_optimizing/hidden_workspace.py`
- `ncpu/self_optimizing/trajectory_logger.py`

Responsibilities:

- hold candidate text, verification history, error summaries, and commit state
- record hidden step traces
- write JSONL trajectories for training and evaluation

### 3. Latent Controller State

Files:

- `ncpu/self_optimizing/latent_controller_state.py`
- `ncpu/self_optimizing/latent_action_policy.py`
- `ncpu/self_optimizing/latent_halt_policy.py`
- `ncpu/self_optimizing/latent_memory_head.py`
- `ncpu/self_optimizing/latent_descriptor_head.py`
- `ncpu/self_optimizing/state_patch_head.py`

Responsibilities:

- maintain structured hidden state across internal steps
- choose hidden actions
- decide when to stop hidden computation
- produce descriptor and patch signals for weight updates
- maintain recurrent learned memory during inference

### 4. Task-Local Fast Weights

Files:

- `ncpu/self_optimizing/task_local_fast_weights.py`
- `ncpu/self_optimizing/ncpu_adaptation_backend.py`

Responsibilities:

- inject reversible low-rank task-local adapters
- update those adapters during inference
- translate gradient/state signals into compact nCPU descriptors
- reset the task-local state between tasks

### 5. Segmented Decode Path

Files:

- `ncpu/self_optimizing/segmented_kv_cache.py`
- `ncpu/self_optimizing/descriptor_decode_runtime.py`
- `ncpu/self_optimizing/recurrent_commit_policy.py`

Responsibilities:

- keep a recent exact token window
- compress older committed hidden history into descriptor memory
- bound the cost of longer hidden deliberation

### 6. Verifier Bridge

Files:

- `ncpu/self_optimizing/code_verifier.py`
- `ncpu/self_optimizing/sandbox_actions.py`
- `ncpu/self_optimizing/llm_benchmark.py`

Responsibilities:

- compile and execute candidate code
- summarize failures into structured feedback
- feed benchmark and verifier evidence back into the hidden controller

## Training Surface

### Trajectory Preparation

Files:

- `ncpu/self_optimizing/trajectory_dataset.py`
- `ncpu/self_optimizing/prepare_internal_training_data.py`

Outputs:

- response SFT examples
- text action-policy examples
- latent action datasets
- latent halt datasets
- latent descriptor datasets
- latent memory datasets
- state-patch datasets

### Controller Training

Files:

- `ncpu/self_optimizing/controller_training.py`
- `ncpu/self_optimizing/train_internal_controller.py`
- `ncpu/self_optimizing/latent_action_training.py`
- `ncpu/self_optimizing/latent_halt_training.py`
- `ncpu/self_optimizing/latent_descriptor_training.py`
- `ncpu/self_optimizing/latent_memory_training.py`
- `ncpu/self_optimizing/state_patch_training.py`

Artifacts:

- response adapter
- action adapter
- latent action head
- latent halt head
- latent descriptor head
- latent memory head
- state patch head
- controller bundle manifests

## API Surface

File: `ncpu/self_optimizing/api_server.py`

Important endpoints:

- `POST /internal/infer`
- `POST /internal/tasks`
- `GET /internal/tasks/{task_id}`
- `GET /internal/tasks/{task_id}/events`
- `GET /internal/tasks/{task_id}/stream`

Properties:

- buffered inference only returns committed output by default
- hidden traces can be requested explicitly
- async tasks expose sanitized hidden events without leaking raw hidden candidate text

## Benchmark Surface

Files:

- `ncpu/self_optimizing/run_qwen_benchmark.py`
- `ncpu/self_optimizing/run_evalplus_benchmark.py`
- `ncpu/self_optimizing/run_bigcodebench_benchmark.py`
- `ncpu/self_optimizing/run_real_memory_proof.py`

Supported paths:

- baseline vs SOME comparisons
- controller-bundle-backed evaluation
- fast-weight and segmented-cache providers
- latent-memory proof collection/training/evaluation

## Evidence Surface

Current evidence lives in two places:

- official benchmark summaries documented in `docs/SOME_RESULTS.md`
- generated local trajectories and proof runs under ignored `benchmarks/` and `training/` artifact trees

Generated artifacts are intentionally not versioned. The repo tracks the code and the summary documentation, not every local checkpoint and benchmark dump.
