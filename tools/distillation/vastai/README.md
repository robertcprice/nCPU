# vast.ai deployment for open-source model benchmarks

The user's explicit rule: no local GPU downloads; route Qwen3.5 /
Gemma 4 workloads to vast.ai. This directory has the scripts to do
that end-to-end.

## Prerequisites (one-time)

1. **vast.ai CLI installed locally**:
   ```bash
   pip install vastai
   export VAST_API_KEY=<your key from vast.ai/console>
   ```

2. **An SSH key registered with vast.ai**:
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/vast_ed25519 -C "nsynth-bench"
   vastai create ssh-key --key-file ~/.ssh/vast_ed25519.pub
   ```

3. **Anthropic API key** for the baseline comparison jobs:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   ```

## Workflow

### Provision an instance

```bash
# Pick a cheap GPU. RTX 4090 is usually enough for Qwen3.5-9B at 4-bit.
tools/vastai/launch.sh qwen3.5-4b
# Or:
tools/vastai/launch.sh qwen3.5-9b
# Or:
tools/vastai/launch.sh gemma-4-9b
```

`launch.sh` queries `vastai search offers`, picks the cheapest matching
the VRAM requirement, creates the instance, prints the `ssh` command
for you to connect.

### Run the benchmark on the remote instance

Once connected, the setup script is already copied over:

```bash
# On the vast.ai instance:
cd ~/nsynth
bash tools/vastai/setup_and_run.sh --model Qwen/Qwen3.5-4B-Instruct
# Produces:
#   ~/nsynth/artifacts/qwen3.5-4b_humaneval_lite.md
#   ~/nsynth/artifacts/qwen3.5-4b_humaneval_full.md
```

### Pull results back

```bash
# From your local machine:
tools/vastai/pull_artifacts.sh <instance_id>
# Copies the remote artifacts/ back into the local artifacts/.
```

### Shutdown

```bash
vastai destroy instance <instance_id>
```

## Cost discipline

- **Pick spot/interruptible instances** for benchmarks — they're 2-4×
  cheaper and benchmarks tolerate restart. RTX 4090 interruptible is
  ~$0.30/hr.
- **Always pass --disk 40 or less** when creating. Default 64GB is
  wasted on benchmark workloads.
- **Destroy immediately after pull**. Even idle billing adds up.

## Estimated costs

| job | GPU | ~wall-clock | ~cost |
|---|---|---:|---:|
| Qwen3.5-2B HumanEval-lite (30 problems) | RTX 4090 | 10 min | $0.05 |
| Qwen3.5-4B HumanEval full (164 problems) | RTX 4090 | 25 min | $0.13 |
| Gemma-4-9B HumanEval full | A100 40GB | 20 min | $0.35 |
| Qwen3.5-9B LoRA fine-tune on 500 cache pairs | A100 80GB | 2 hours | $3.50 |

## What this gives you

- **Measured pass@1 for open-source models** under our full enhanced-
  inference stack (cache + best-of-N + retry).
- **Direct comparison to Claude-tier** using the same benchmark runner
  and the same cache file format.
- **Distilled adapters** that plug into `LocalModelClient` back on your
  local machine for zero-marginal-cost inference.

See `launch.sh`, `setup_and_run.sh`, `pull_artifacts.sh`.
