# Benchmarks Directory

This directory contains two kinds of content:

- versioned benchmark scripts and helpers
- local generated benchmark artifacts

Only the scripts and helper code belong in git. Generated result dumps, trajectory captures, remote snapshots, and model outputs are intentionally ignored to keep the repository readable.

If a benchmark run produces a result worth citing, summarize it in `docs/some/results.md` or another tracked report rather than committing the raw artifact tree.

See also: `docs/REPO_HYGIENE.md` for the repo-wide rule of thumb on tracked source vs local experimental artifacts.

## Vast.ai Coprocessor Runbook

Use [`run_vast_coprocessor_benchmark.sh`](run_vast_coprocessor_benchmark.sh) to launch the real-world coprocessor benchmark on Vast.ai without manually babysitting instance startup.

Default behavior:

- searches reliable single-GPU 4090/3090/A100 offers
- creates the instance with `--ssh`
- rejects known boot failures such as CDI / OCI GPU injection errors
- uploads only the benchmark assets needed for `benchmark_coprocessor_realworld.py`
- installs the required CUDA PyTorch and Transformers stack remotely
- starts the benchmark under `nohup`

Minimal example:

```bash
benchmarks/run_vast_coprocessor_benchmark.sh --wait --destroy-on-success
```

That uses:

- model: `Qwen/Qwen3.5-2B`
- weights: `training_results/code_embedded/qwen35-2b/coprocessor_weights.pt`
- benchmarks: `coding,reasoning`
- output: `training_results/code_embedded/qwen35-2b/realworld_benchmark_vast.json`

Useful overrides:

```bash
benchmarks/run_vast_coprocessor_benchmark.sh \
  --query 'gpu_name=RTX_4090 reliability > 0.98 num_gpus=1 dph_total<0.45' \
  --benchmarks humaneval,coding,reasoning \
  --humaneval-path HumanEval.jsonl \
  --wait
```

Important note:

- Do not create the instance without `--ssh`. Vast will still return an SSH gateway URL, but the container will not have SSH injected, so key auth will keep failing even if the account key is correct.
