# Weight CPU Architecture

## Goal

Move SOME from:

- external `generate -> verify -> feedback -> retry`

to:

- hidden latent reasoning before visible output
- task-local weight updates during inference
- verifier-gated commit after internal compute

The target is not literal arbitrary CPU execution inside dense weights on day one. The practical target is a model whose hidden state and small task-local fast weights act like a bounded internal coprocessor.

## What Recent Work Suggests

The most relevant lines of work are:

1. Hidden reasoning before visible output
- Quiet-STaR (2024): hidden reasoning before speaking
- Fast Quiet-STaR (2025): similar direction with lower token overhead
- Coconut (2024): continuous latent thought rather than explicit thought tokens

2. Computation via recurrence / looping
- Looped Transformers as Programmable Computers (2023): repeated transformer application can implement iterative computation

3. Weight updates during inference
- Learning to (Learn at Test Time) (2024): test-time training style hidden adaptation
- Titans (2025): neural memory learned and updated at test time
- End-to-End Test-Time Training for Long Context (2025): compress context into task-local weights
- Verifier-driven sample selection for test-time training (2025): use verifier signal to decide what to adapt on

These do not prove that we can instantly get a “CPU in the weights,” but they do support a credible architecture direction:

- hidden recurrent compute
- fast task-local weight updates
- verifier-gated adaptation

## Translation To SOME

Current SOME already gives us the right supervision source:

- hidden think/write/patch/verify trajectories
- verifier outcomes
- benchmark evidence on HumanEval+, MBPP+, and BigCodeBench-Hard

That means the path forward is:

1. Distill the external controller into a hidden controller.
2. Replace textual hidden thoughts with latent thought states.
3. Add task-local low-rank weight updates during inference.
4. Train the model to decide when to update weights, when to keep reasoning in hidden state, and when to commit.

## Proposed Runtime

### Runtime Layers

1. Attention / hidden state
- short-horizon working memory
- local token-to-token computation

2. Latent recurrent controller
- several hidden compute steps before visible output
- chooses `think`, `write`, `patch`, `verify`, `commit`, `fail`
- maintains structured latent controller state across those steps
- should ultimately move action and halt selection off text prompts and into a latent policy head

3. Fast-weight adapter
- low-rank modules updated per task
- task-local only
- reset between tasks

4. External verifier bridge
- still used during early stages
- eventually invoked less often as verifier behavior internalizes

5. Segmented decode / cache path
- compressed committed-history descriptors instead of full long-horizon KV state
- recent live token window stays exact
- hidden compute can run for longer before attention cost explodes

### Inference Loop

1. Prompt enters model.
2. Model runs latent reasoning steps.
3. If uncertainty or likely failure is detected, small fast-weight adapter updates occur.
4. Candidate is checked by hidden verifier state and optionally external execution.
5. Only then does the commit head emit visible output.

## Why Low-Rank Fast Weights

We should not start with full-model online training during inference.

Low-rank task-local adapters are a better first target because they are:

- cheaper
- reversible
- easier to benchmark
- easier to isolate from the base model
- closer to a “coprocessor inside the model” than an external wrapper loop

So the first implementation target should be:

- frozen backbone
- small fast-weight LoRA-style modules
- learned update rule
- verifier-gated updates

## Current Prototype

The repo now has a first concrete implementation of this idea:

- `task_local_fast_weights.py`
  - injects low-rank residual adapters into selected linear layers
  - updates only those adapters during hidden SOME inference
  - now captures the actual fast-weight gradients and routes them through an nCPU-backed adaptation backend
  - uses those nCPU gradient descriptors to steer within-task learning-rate and step budgeting
  - resets them at task end
- `ncpu_adaptation_backend.py`
  - converts real fast-weight gradients into compressed nCPU gradient descriptors
  - uses the repo's `GradientAwareNetworkProtocol` compression stack instead of a pure wrapper-side heuristic
  - records per-task adaptation sessions for later training and analysis
- `internal_controller.py`
  - starts a task-local fast-weight session at task start
  - writes hidden plan text into the fast weights before visible generation
  - writes verifier failure state into the fast weights during repair
  - records fast-weight update events in the hidden workspace trace
  - now also supports a latent action-selection policy so `think/write/patch/fail` can come from structured state instead of prompt-only action selection
- `latent_action_policy.py`
  - encodes latent controller state plus workspace status into a dense feature vector
  - supports a learned action head with heuristic fallback
  - is the bridge toward an actual latent halt/commit controller rather than a text-side action prompt
- `latent_memory_head.py`
  - learns recurrent memory deltas from hidden-controller event streams
  - lets hidden inference update a learned memory vector instead of rebuilding everything from textual state
- `latent_descriptor_head.py`
  - learns latent-state-to-descriptor projections for weight updates
- `state_patch_head.py`
  - learns latent-state-to-patch behavior for fast-weight edits
- `llm_provider.py`
  - exposes this runtime as `hf_fast_weights`
- `train_internal_controller.py`
  - emits a fast-weight controller bundle that defaults to descriptor-first controller settings
  - disables the old text-target fast-weight loop in that runtime variant unless explicitly re-enabled

This is still not a literal dense CPU inside the full transformer. It is the first credible bridge:

- private latent-ish planning
- small reversible task-local weight changes
- descriptor-backed segmented decode for long hidden deliberation
- verifier-gated commit

## Current Proof Status

There are two evidence tracks in the repo right now:

1. Official benchmark evidence
- HumanEval+ finished runs showed:
  - `qwen3.5:4b 147/164 -> 154/164`
  - `qwen3.5:9b 144/164 -> 156/164`
  - `qwen3.5:27b 153/164 -> 156/164`
- BigCodeBench-Hard finished result:
  - `qwen3.5:9b 33/148 -> 49/148`

2. Internal-memory proof evidence
- The first meaningful latent-memory proof uses fresh synthetic memory-aware hidden trajectories generated by `generate_synthetic_internal_trajectories.py`
- On that held-out set, the learned memory head beat the zero-delta baseline by:
  - validation MSE `0.0006947 -> 0.0001163`
  - relative MSE improvement `83.26%`
  - cosine similarity `0.0 -> 0.9028`

Those numbers are strong enough to justify continuing the architecture work, but they are not yet proof that the final "CPU in the model" vision is solved. The next publishable step is collecting fresh real-model trajectories with the current memory-aware controller and rerunning the same evaluation path there.

## What This Is Not

- It is not a claim that the entire dense backbone has already become a literal general-purpose CPU.
- It is not a claim that prompt retries alone equal internal computation.
- It is not a claim that the wrapper-era SOME loops are the final architecture.

The current system should be described more narrowly and more honestly:

- hidden controller before visible output
- task-local weight adaptation during inference
- segmented committed-history decode
- verifier-gated commit
- a training path that is steadily moving more control from text prompts into latent heads and learned update rules

## Training Plan

### Stage 1

Distill the existing hidden SOME controller.

Train:

- response adapter
- action adapter

Data:

- `benchmarks/internal_trajectories/...`

Success metric:

- higher pass@1 with fewer visible retries

### Stage 2

Internalize hidden textual reasoning into latent recurrent state.

Train:

- latent compute head
- halt / commit head

Success metric:

- same or better accuracy with fewer thought tokens

### Stage 3

Introduce fast-weight adaptation during inference.

Train:

- low-rank task-local adapters
- update policy

Success metric:

- better hard-benchmark pass rate on first visible answer

### Stage 4

Meta-train the update rule itself.

Train:

- update rule
- commit policy
- latent verifier state

Success metric:

- fewer update steps, better generalization, less dependence on external retry

## Benchmark Implications

The best benchmarks for this direction are still:

- HumanEval+
- MBPP+
- BigCodeBench-Hard

because they measure:

- code correctness
- repair value
- verifier-grounded improvement

The right headline metric changes from:

- “Did wrapper-loop SOME help?”

to:

- “Did internalized hidden reasoning plus fast weights improve first committed output?”

## Repo Mapping

Current relevant modules:

- `internal_controller.py`
- `latent_controller_state.py`
- `controller_bundle.py`
- `controller_runtime.py`
- `task_local_fast_weights.py`
- `train_internal_controller.py`
- `api_server.py`
- `llm_benchmark.py`

New planning artifact:

- `weight_cpu_blueprint.py`

That blueprint is the transition contract from the current external SOME runtime to the future latent fast-weight runtime.

## Research References

- Quiet-STaR (2024): https://arxiv.org/abs/2403.09629
- Learning to (Learn at Test Time) (2024): https://arxiv.org/abs/2407.04620
- Coconut (2024): https://arxiv.org/abs/2412.06769
- Titans (2025): https://arxiv.org/abs/2501.00663
- Fast Quiet-STaR (2025): https://arxiv.org/abs/2505.17746
- Verifier-driven test-time training (2025): https://arxiv.org/abs/2505.19475
- End-to-End Test-Time Training for Long Context (2025): https://arxiv.org/abs/2512.23675
- Looped Transformers as Programmable Computers (2023): https://arxiv.org/abs/2301.13196
