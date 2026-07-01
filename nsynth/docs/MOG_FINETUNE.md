# Fine-tuning the model to write Mog (the STaR / self-improvement loop)

**Goal:** teach the local model to write valid Mog first-shot and program in nsynth's
verification-friendly style, so the repair loop wastes no iterations on syntax and the
single-shot rate climbs. The ablation showed the raw 4B model at 43% (unverified) and
the full stack at 63% (verified) — much of the 43→X gap is *Mog syntax/idiom, not
reasoning*, which is exactly what fine-tuning fixes.

## Why this is sound (the key property)

Every training pair comes from a program that **passed the verifier** (reproduces all
tests). So the corpus is guaranteed-correct — a self-training loop that can't poison
itself. This is rejection-sampling fine-tuning / STaR, and nsynth's verifier is the
filter that makes it safe.

```
harvest verified (task -> Mog) pairs  →  LoRA fine-tune  →  better Mog writer
       ↑                                                            │
       └──────────  more solves = more verified pairs  ←────────────┘
```

## Step 1 — Harvest a corpus

Set `NSYNTH_HARVEST=<path>`; every SOLVE (engine or repair) appends a verified pair.

```bash
# Engine-only (NO server) — free seed corpus (~20% of tasks solve):
rm -f ~/.nsynth_solved_programs.json
NSYNTH_CACHE_PATH= NSYNTH_HARVEST=/tmp/mog_corpus.jsonl NSYNTH_SOLVE_BUDGET_MS=3000 \
  bash scripts/run_mbpp_bench.sh /tmp/mbpp_full_desc.jsonl 4

# RICH corpus (server up) — adds every repair-loop solve (whole-program Mog):
python3 -m mlx_lm server --model lmstudio-community/gemma-4-E4B-it-MLX-8bit --port 8765 &
export NSYNTH_LOCAL_LLM_URL=http://localhost:8765/v1/chat/completions \
       NSYNTH_LOCAL_LLM_MODEL=lmstudio-community/gemma-4-E4B-it-MLX-8bit \
       NSYNTH_LOCAL_LLM_REPAIR=1 NSYNTH_LOCAL_LLM_REPAIR_TRIES=3 \
       NSYNTH_HARVEST=/tmp/mog_corpus.jsonl NSYNTH_CACHE_PATH= NSYNTH_SOLVE_BUDGET_MS=4000
bash scripts/run_mbpp_bench.sh /tmp/mbpp_full_desc.jsonl 250    # hours; harvests as it goes
```

Corpus format = `mlx_lm.lora` chat: `{"messages":[{system},{user},{assistant}]}`, where
system = `MOG_SYSTEM_PROMPT`, user = the task, assistant = the verified Mog (fenced).

## Step 2 — Prepare train/valid split

```bash
python3 scripts/mog_finetune_prepare.py /tmp/mog_corpus.jsonl /tmp/mog_ft
# dedupes by (user,assistant), writes /tmp/mog_ft/{train,valid}.jsonl (90/10)
```

## Step 3 — LoRA fine-tune (local, Apple Silicon)

```bash
python3 -m mlx_lm.lora \
  --model lmstudio-community/gemma-4-E4B-it-MLX-8bit \
  --train --data /tmp/mog_ft \
  --iters 600 --batch-size 1 --num-layers 8 \
  --adapter-path /tmp/mog_ft/adapter
```

## Step 4 — Fuse the adapter into a servable model

```bash
python3 -m mlx_lm.fuse \
  --model lmstudio-community/gemma-4-E4B-it-MLX-8bit \
  --adapter-path /tmp/mog_ft/adapter \
  --save-path /tmp/mog_ft/mog-gemma
```

## Step 5 — Re-measure (the number that proves it)

Serve the fused model and rerun the ablation; compare single-shot A (raw) vs A′
(fine-tuned) — that delta is what fine-tuning bought:

```bash
python3 -m mlx_lm server --model /tmp/mog_ft/mog-gemma --port 8765 &
export NSYNTH_LOCAL_LLM_MODEL=/tmp/mog_ft/mog-gemma
# A' : fine-tuned model, single-shot
NSYNTH_LLM_ONLY=1 NSYNTH_LOCAL_LLM_REPAIR_TRIES=1 \
  bash scripts/run_mbpp_bench.sh /tmp/mbpp_full_desc.jsonl 150 30
```

## Step 6 — Loop (STaR)

Feed the fine-tuned model's new solves back into the corpus (Step 1 with the fused
model), retrain (Step 3). Each round the model writes better Mog → solves more →
generates more verified data. The verifier keeps every round honest.

## Honest expectations

- Fine-tuning fixes *syntax/idiom* (writes `x: i64 =` not `let mut`, knows the string
  methods, uses `arr.len`), so single-shot should rise materially (43% → plausibly
  60–75%) with modest data.
- It does NOT add reasoning the 4B model lacks — the hard-algorithm tail still needs a
  bigger model or the engine/library. Fine-tuning + engine + repair compound.
- Thin data (<100 pairs) underfits; harvest the full representable set (912 tasks) with
  the repair loop first for a few hundred–thousand pairs.
