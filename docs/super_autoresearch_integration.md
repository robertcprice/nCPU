# Super-AutoResearch + SOME

## Goal

The point of combining Super-AutoResearch with SOME is not just "more agent loops." The useful combination is:

- agent planning and experiment management
- verified hidden-controller execution for code/reasoning work
- reproducible training and benchmark loops for internal controller components

## Current Best Architecture

```
planner / agent runtime
    |
    +--> experiment proposal
    +--> bundle selection
    +--> benchmark selection
    |
    v
SOME buffered controller
    |
    +--> hidden workspace
    +--> latent control heads
    +--> recurrent memory
    +--> task-local fast weights
    +--> segmented decode path
    +--> verifier bridge
    |
    v
committed output + trajectories + metrics
```

## Practical Use Cases

### 1. Parallel Controller Search

The agent proposes multiple controller changes:

- latent-action head config changes
- latent-memory head retrains
- fast-weight config changes
- benchmark timeout / retry policies

Each candidate is evaluated with the benchmark runners and compared on the same metrics.

### 2. Trajectory-Driven Distillation

The agent can:

1. collect fresh trajectories
2. prepare datasets
3. train a new bundle
4. run evaluation
5. keep or discard the candidate

That loop is much more valuable than plain prompt-chaining.

### 3. Proof-Oriented Research

The agent can run proof tracks such as:

- benchmark-loop SOME gains
- latent-memory prediction improvement
- descriptor-head or patch-head evaluation

The output of each loop should be:

- code changes
- a reproducible command path
- updated summary documentation

## Why This Matters

Super-AutoResearch gives search and orchestration.

SOME gives:

- execution grounding
- verification
- trajectory logging
- an upgrade path from wrapper loops toward internalized computation

Together, that means the agent is not just editing files blindly. It has a mechanism to test whether a change actually improves execution-grounded behavior.

## Current Boundaries

This should still be described honestly:

- the controller is increasingly internalized, but not fully dense-CPU-in-weights
- the strongest finished evidence is still benchmark improvement plus the latent-memory proof
- the architecture is real and worth exploring, but not "done"

## Useful Entry Points

- `ncpu/self_optimizing/autoresearch_agent.py`
- `ncpu/self_optimizing/controller_bundle.py`
- `ncpu/self_optimizing/controller_runtime.py`
- `ncpu/self_optimizing/train_internal_controller.py`
- `ncpu/self_optimizing/run_real_memory_proof.py`
- `docs/SOME_RESULTS.md`
