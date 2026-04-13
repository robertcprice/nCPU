# SOME Architecture

## Current Shape

```
task prompt
    |
    v
buffered internal controller
    |
    +--> hidden workspace
    |      - candidate text
    |      - verifier history
    |      - repair summaries
    |      - trajectory logging
    |
    +--> latent controller state
    |      - action state
    |      - halt state
    |      - recurrent memory
    |      - descriptor state
    |
    +--> hidden policies / heads
    |      - latent action head
    |      - latent halt head
    |      - latent memory head
    |      - latent descriptor head
    |      - state patch head
    |
    +--> task-local fast weights
    |      - low-rank temporary adapters
    |      - nCPU descriptor-backed updates
    |
    +--> segmented decode path
    |      - recent exact KV window
    |      - compressed committed-history descriptors
    |
    +--> verifier bridge
           - compile / run / test
           - summarize failures
           - decide patch vs commit
    |
    v
committed final output
```

## Component Map

### Runtime

- `internal_controller.py`
- `hidden_workspace.py`
- `latent_controller_state.py`
- `sandbox_actions.py`
- `trajectory_logger.py`

### Latent Heads

- `latent_action_policy.py`
- `latent_halt_policy.py`
- `latent_memory_head.py`
- `latent_descriptor_head.py`
- `state_patch_head.py`

### Adaptation and Decode

- `task_local_fast_weights.py`
- `ncpu_adaptation_backend.py`
- `segmented_kv_cache.py`
- `descriptor_decode_runtime.py`
- `recurrent_commit_policy.py`

### Training

- `trajectory_dataset.py`
- `controller_training.py`
- `train_internal_controller.py`
- `latent_action_training.py`
- `latent_halt_training.py`
- `latent_memory_training.py`
- `latent_descriptor_training.py`
- `state_patch_training.py`

### Serving and Benchmarks

- `api_server.py`
- `llm_benchmark.py`
- `run_qwen_benchmark.py`
- `run_evalplus_benchmark.py`
- `run_bigcodebench_benchmark.py`
- `run_real_memory_proof.py`

## Design Principle

Move control inward in stages:

1. external verifier-guided retries
2. buffered hidden controller
3. latent control heads
4. task-local weight updates during inference
5. segmented cache / committed descriptor decode
6. learned recurrent memory

The repo is in stages 2 through 5 already, with stage 6 now partially trained and evaluated.

## Why This Matters

This architecture is the bridge between:

- ordinary wrapper-loop self-debugging

and

- the longer-term goal of a model that can perform bounded internal computation and adaptation before it emits visible output

The benchmark and proof work should be read in that context.
