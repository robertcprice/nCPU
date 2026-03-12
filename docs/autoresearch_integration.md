# AutoResearch Integration

## Purpose

This note describes how an autonomous research agent should use SOME inside this repository.

The old version of this document assumed a tensor-descriptor-centric SOME pipeline. The current repo is different: the useful integration point is the hidden-controller runtime plus the verifier-backed execution surface.

## Recommended Integration Model

AutoResearch should treat SOME as a **verified internal tool** for code and reasoning tasks.

High-level flow:

1. planner chooses an experiment or coding task
2. AutoResearch calls buffered SOME inference
3. SOME performs hidden `think/write/verify/patch/commit`
4. only committed output comes back to the agent
5. trajectories and verifier results are logged for future training

That is stronger than a plain "ask an LLM for code and hope it works" tool.

## Best Current Interfaces

### 1. Buffered Inference API

Use:

- `POST /internal/infer`
- `POST /internal/tasks`
- `GET /internal/tasks/{task_id}`
- `GET /internal/tasks/{task_id}/events`
- `GET /internal/tasks/{task_id}/stream`

Files:

- `ncpu/self_optimizing/api_server.py`
- `ncpu/self_optimizing/internal_controller.py`

Why this is the right surface:

- committed output is separated from hidden intermediate failures
- hidden traces can be exposed selectively for research
- async event streams give progress without leaking the whole hidden workspace by default

### 2. Controller Bundles

Use:

- `ncpu/self_optimizing/controller_bundle.py`
- `ncpu/self_optimizing/controller_runtime.py`

Why:

- lets the agent select a full response/action/memory/descriptor configuration as one unit
- makes experiments reproducible
- avoids bespoke per-run wiring

### 3. Benchmark and Proof Runners

Use:

- `run_qwen_benchmark.py`
- `run_evalplus_benchmark.py`
- `run_bigcodebench_benchmark.py`
- `run_real_memory_proof.py`

Why:

- they are the cleanest path to evaluate whether agent-generated controller changes are actually helping

## What AutoResearch Should Optimize For

Near-term:

- better benchmark pass rate
- fewer visible retries
- better verifier success on first committed answer
- improved latent-memory and latent-control metrics

Long-term:

- less dependence on text-side hidden prompts
- more control transferred into latent heads and weight-side adaptation
- cheaper long-horizon internal compute through segmented decode

## Good AutoResearch Loops

### Controller Improvement Loop

1. propose controller config or bundle change
2. run benchmark slice
3. compare against baseline
4. keep only changes that improve metrics

### Training Loop

1. collect fresh hidden trajectories
2. retrain one latent head or adapter
3. benchmark the new bundle
4. accept or reject the update

### Proof Loop

1. generate a fresh corpus
2. train latent-memory or descriptor head
3. evaluate against the zero baseline
4. record improvement in tracked docs

## Things To Avoid

- treating old wrapper-loop SOME as the final architecture
- using unverified plain LLM outputs when the buffered controller path is available
- publishing claims about "CPU in the dense weights" that go beyond the current evidence

## Related Repo Files

- `docs/SOME_COMPLETE_GUIDE.md`
- `docs/SOME_RESULTS.md`
- `docs/WEIGHT_CPU_ARCHITECTURE.md`
- `ncpu/self_optimizing/api_server.py`
- `ncpu/self_optimizing/internal_controller.py`
- `ncpu/self_optimizing/train_internal_controller.py`
