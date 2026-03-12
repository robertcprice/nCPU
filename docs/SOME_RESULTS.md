# SOME Results

## Scope

This file tracks the current summary evidence for the Self-Optimizing Machine Engine (SOME).

The repository intentionally does **not** version every local benchmark dump, checkpoint, or trajectory file. Raw outputs live in ignored local artifact trees under `benchmarks/` and `training/`. This document records the headline results that are worth citing.

## Official Benchmark Evidence

### HumanEval+

Finished SOME vs baseline results:

- `qwen3.5:4b`: `147/164 -> 154/164` (`+7`)
- `qwen3.5:9b`: `144/164 -> 156/164` (`+12`)
- `qwen3.5:27b`: `153/164 -> 156/164` (`+3`)

Interpretation:

- the verifier-guided hidden loop is clearly helping
- the largest benefit showed up on the middle model (`9b`)
- stronger models still improved, but by less

### BigCodeBench-Hard

Finished result:

- `qwen3.5:9b`: `33/148 -> 49/148` (`+16`, `+10.8` points)

This is the strongest finished "hard benchmark" result in the repo right now.

Partial 4b/27b runs existed during the March 2026 H200 campaign, but those are not promoted here as final headline results because the long-running jobs were interrupted multiple times by instance churn and resume recovery.

## Internal-Memory Proof

### Synthetic Memory-Aware Trajectory Corpus

Fresh hidden-controller trajectories were generated specifically for the memory-head proof after the runtime started recording meaningful recurrent memory state.

Held-out validation metrics for the learned latent-memory head:

- baseline MSE: `0.0006946895737200975`
- model MSE: `0.00011629469372564927`
- relative MSE improvement: `83.26%`
- cosine similarity delta: `0.9028`

Event-level held-out relative MSE improvements:

- `verify`: `93.88%`
- `patch`: `89.06%`
- `commit`: `93.19%`
- `write`: `54.21%`
- `think`: `58.93%`

Interpretation:

- the learned recurrent memory path is doing nontrivial predictive work
- this is real evidence that further internalization is worth exploring
- this is still a proof step, not the final publishable claim on a large real-model corpus

## Current Caveats

- Current SOME is **not** a claim that the entire dense backbone has become a literal CPU.
- The strongest completed benchmark evidence is still benchmark-loop SOME, not the full latent-memory architecture.
- The memory proof is currently strongest on a fresh synthetic hidden-controller corpus. The real-model proof path exists and is being used to collect fresh trajectories, but the synthetic result is the cleanest finished evidence today.

## Relevant Code Paths

- `ncpu/self_optimizing/run_qwen_benchmark.py`
- `ncpu/self_optimizing/run_evalplus_benchmark.py`
- `ncpu/self_optimizing/run_bigcodebench_benchmark.py`
- `ncpu/self_optimizing/run_real_memory_proof.py`
- `ncpu/self_optimizing/latent_memory_evaluation.py`
- `ncpu/self_optimizing/generate_synthetic_internal_trajectories.py`
