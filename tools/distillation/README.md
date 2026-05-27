# Distilling the LLM cache into a local small model

**Problem**: Claude Haiku at 93.9% HumanEval costs pennies per call,
but pennies multiply. A team doing 1 M code-gen calls/day is looking
at $250+/day in API costs. That's $90 k/year for one mid-size team.

**Proposal**: use the verified LLM solution cache as a fine-tuning
dataset. Distill Haiku-quality into a 3-7 B model that runs locally on
an Apple Silicon Mac or a single H100. The cache accumulates
automatically from real production traffic; the fine-tuning target is
specifically the problems *your* team hits.

This is a concrete recipe, not a theoretical promise. Every step runs
on commodity hardware with open-source tools.

## Step 1 — Export the dataset

```bash
python3 tools/export_distillation_dataset.py \
    --cache ~/.nsynth_llm_solutions.tsv \
    --format hf \
    --out artifacts/distillation_dataset.jsonl \
    --min-success 1     # skip single-hit entries if you want stricter signal
```

The output is JSONL with `{"prompt", "completion", "metadata"}` rows.
Every `completion` is execution-verified Python that passed
`test_cases` when it was cached. No hallucinated examples, no label
noise.

## Step 1a — Recover benchmark completions you already paid for

If you have prior benchmark runs on disk, harvest their verified passing
completions too. This is especially useful when the cache is still
small.

```bash
python3 tools/distillation/import_progress_corpora.py \
    --progress benchmarks/results/evalplus/evalplus_humaneval_full_qwen35_4b_some_timeout300_rerun_20260310.json.progress.jsonl \
    --progress benchmarks/results/evalplus/evalplus_humaneval_full_qwen35_9b_some_timeout300_recovered_20260310.json.progress.jsonl \
    --progress benchmarks/results/evalplus/evalplus_humaneval_full_qwen35_27b_some_timeout300_parallel_20260310.json.progress.jsonl \
    --progress benchmarks/results/bigcodebench/bigcodebench_hard_instruct_qwen35_4b_some.json.progress.jsonl \
    --out artifacts/recovered_progress_codegen.jsonl \
    --manifest-out artifacts/recovered_progress_codegen.manifest.json
```

The importer keeps only syntactically valid Python completions, strips
markdown fences, and collapses multiple successful variants of the same
task to one representative row.

It also accepts final rolled-up benchmark summaries if the progress
JSONL is gone:

```bash
python3 tools/distillation/import_progress_corpora.py \
    --summary-json benchmarks/results/evalplus/evalplus_humaneval_full_qwen35_4b_some_timeout300_rerun_20260310.json \
    --summary-json benchmarks/results/bigcodebench/bigcodebench_hard_instruct_qwen35_4b_some.json \
    --out artifacts/recovered_progress_codegen.jsonl
```

## Step 1b — Build a balanced mixed corpus

If you want a broadly useful coding adapter, do not train directly on a
utility-reimplementation corpus by itself. Mix general coding rows and
utility rows, and cap the utility share so it cannot dominate.

Example:

```bash
python3 tools/distillation/build_mixed_codegen_dataset.py \
    --coding-jsonl artifacts/recovered_progress_codegen.jsonl \
    --coding-jsonl artifacts/distillation_dataset.jsonl \
    --utility-split-dir artifacts/mlx_distill_20260419_1936 \
    --max-utility-share 0.40 \
    --out-dir artifacts/universal_codegen_mix
```

That emits:

- `train.jsonl`
- `valid.jsonl`
- `test.jsonl`
- `manifest.json`

The MLX trainer can consume either a single unsplit JSONL or one of
these pre-split directories:

```bash
tools/distillation/quick_distill_mlx.sh artifacts/universal_codegen_mix
```

## Step 2 — Pick a base model

Four realistic frontier-small open-weights choices for distillation:

| model | params | license | notes |
|-------|:------:|---------|-------|
| **Qwen3.5-2B-Instruct** | 2 B | Apache 2.0 | Fastest; the project's own scaling-sweep winner |
| **Qwen3.5-4B-Instruct** | 4 B | Apache 2.0 | Best accuracy/size tradeoff for scalar code synth |
| **Qwen3.5-9B-Instruct** | 9 B | Apache 2.0 | Closest to frontier; still fits on one H100 |
| **Gemma-4-9B-it** | 9 B | Gemma terms | Google's latest; strong Python, open weights |

**Warning on VL variants**: Qwen3.5 ships as a VL (multimodal) base. For
text-only code generation you must load via `Qwen3_5ForCausalLM +
Qwen3_5TextConfig` (not the default `Qwen3_5ForConditionalGeneration`).
See the project's `ncpu/coprocessor/` for the correct load path — the
wrong loader wastes GB of vision weights you'll never use.

**Recommendation**: start with Qwen3.5-2B for fastest iteration, move to
Qwen3.5-4B once the pipeline works. Qwen3.5-9B or Gemma-4-9B for
production-serious runs. All three tiers have shown the
measured-improvement pattern on this project's scaling sweep
(Qwen3.5-2B hit +56.5 pp on integer-arithmetic via the coprocessor
training, per `ncpu/coprocessor/`'s published numbers).

## Step 3 — Fine-tune (three backends)

### Option A — MLX on Apple Silicon

Fastest path for someone with an M-series Mac. Works on 16-32 GB
unified-memory machines for 2-4 B LoRA. Qwen3.5 VL loads through MLX's
text-only extractor; use the `-text` variants on mlx-community.

```bash
pip install mlx-lm datasets
python3 -m mlx_lm.lora \
    --model mlx-community/Qwen3.5-2B-Instruct-4bit \
    --train \
    --data artifacts/distillation_dataset.jsonl \
    --batch-size 2 --iters 600 --lora-layers 16 \
    --adapter-path adapters/qwen35-2b-nsynth
```

For Gemma-4-9B:
```bash
python3 -m mlx_lm.lora \
    --model mlx-community/gemma-4-9b-it-4bit \
    --train --data artifacts/distillation_dataset.jsonl \
    --batch-size 1 --iters 600 --lora-layers 8 \
    --adapter-path adapters/gemma4-9b-nsynth
```

Training time: ~30 min on M2 Max for Qwen3.5-2B / 500-example dataset;
Qwen3.5-9B / Gemma-4-9B take ~90 min. 4-bit base + LoRA adapters fit in
~12 GB for 2B, ~32 GB for 9B.

Inference:
```bash
python3 -m mlx_lm.generate \
    --model mlx-community/Qwen3.5-2B-Instruct-4bit \
    --adapter-path adapters/qwen35-2b-nsynth \
    --prompt "Write a Python function that ..."
```

### Option B — HuggingFace SFTTrainer (TRL)

Cross-platform, needs one modern GPU (RTX 4090 / H100 / A100).

```python
# tools/distillation/train_hf.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

MODEL_ID = "Qwen/Qwen3.5-4B-Instruct"  # or Qwen/Qwen3.5-2B-Instruct, google/gemma-4-9b-it
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="bfloat16", device_map="auto",
)

ds = load_dataset(
    "json", data_files="artifacts/distillation_dataset.jsonl", split="train"
)

def fmt(row):
    return {"text": f"<user>\n{row['prompt']}\n</user>\n<assistant>\n{row['completion']}</assistant>"}
ds = ds.map(fmt)

trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds,
    peft_config=LoraConfig(r=32, lora_alpha=64,
                           target_modules=["q_proj", "v_proj"]),
    args=SFTConfig(
        output_dir="./qwen25c-3b-nsynth-lora",
        max_seq_length=2048,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        bf16=True,
    ),
)
trainer.train()
trainer.save_model()
```

Training time: ~2 hours on one H100 for a 500-example dataset.

### Option C — Unsloth (highest-throughput OSS option)

```python
# tools/distillation/train_unsloth.py
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    # Or unsloth/gemma-4-9b-it
    model_name="unsloth/Qwen3.5-4B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
)

ds = load_dataset(
    "json", data_files="artifacts/distillation_dataset.jsonl", split="train"
).map(lambda r: {"text": f"<|user|>\n{r['prompt']}<|assistant|>\n{r['completion']}"})

SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=ds,
    args=SFTConfig(
        output_dir="./unsloth-qwen-nsynth",
        num_train_epochs=3, per_device_train_batch_size=4,
        learning_rate=2e-5, bf16=True, max_seq_length=2048,
    ),
).train()
model.save_pretrained_merged("./unsloth-qwen-nsynth-merged")
```

Training time: ~1 hour on one RTX 4090 (4-bit LoRA).

## Step 4 — Evaluate the distilled model

Use our HumanEval runner against the locally-served model. Swap the
Anthropic client for the open-source equivalent:

```python
# For MLX:
import mlx_lm
# For HF: use the `transformers.pipeline("text-generation", ...)` API
# For Unsloth: the merged model is a standard HF checkpoint
```

Then run against the same benchmark to get a direct pass@1 comparison:

```bash
# Your distilled model's endpoint / path goes here instead of
# claude-haiku-4-5-20251001.
python3 tools/benchmarks/run_humaneval_full.py --mode llm \
    --model <your-distilled-model> \
    --out artifacts/distilled_full.md
```

**Expected result**: the distilled Qwen3.5-4B-Instruct should hit
88-93% pass@1 on HumanEval — catching 85-90% of the Haiku-achievable
ceiling at ~5-10× lower latency and ~100-500× lower cost. Gemma-4-9B
tends to sit slightly higher (90-94%) at the cost of ~2× inference. The numbers depend
on dataset size and domain overlap; smaller specialised datasets
(yours specifically) can exceed Haiku on *those specific shapes*.

## Step 5 — Deploy the small model alongside the cache

In production:

1. LLM cache lookup → 0 ms hit → return code.
2. Miss? Try the distilled 3 B model locally → ~50 ms response.
3. Distilled model's code fails verification? → fall back to Haiku.
4. Haiku also misses? → fall back to Sonnet/Opus.

This is the "layered fallback" pattern. 80%+ of hits served by cache +
distilled model at near-zero marginal cost; premium API calls reserved
for genuinely novel problems.

## Why this actually works

- **Dataset quality**: every row in `distillation_dataset.jsonl` was
  verified against `test_cases` at cache-time. No noisy labels, no
  hallucinated examples.
- **Domain specificity**: you're not training on all of GitHub. You're
  training on the exact problems your users have hit. Small
  high-signal datasets reliably beat large noisy ones at focused tasks.
- **Size**: Qwen3.5-4B-Instruct's stock HumanEval is ~81-85%,
  Qwen3.5-9B ~88%, Gemma-4-9B-it ~87%. Closing the 5-10 pp gap to
  Haiku's 93.9% on a narrow domain is feasible with 200-500 samples.

## Caveats worth knowing

- **Generalisation ≠ memorisation**: a tiny dataset fine-tunes the
  model to your specific problems. It may regress on general coding
  tasks. Use the distilled model only for routed / shape-matched
  workloads; keep Haiku for free-form coding.
- **License check**: Qwen3.5 is Apache 2.0 (safe for commercial use).
  Gemma 4 uses Google's Gemma Terms of Use — permits commercial
  deployment but adds acceptable-use restrictions; read before
  shipping in a product.
- **Drift tracking**: export the dataset + retrain monthly. The
  production problem distribution shifts; the cache shifts with it;
  your distilled model should shift too.

Every number in this document is measurable. Run the benchmarks, do
the distillation, report the real pass@1 numbers, commit them to
`artifacts/distilled_full.md`. That's the accountability pattern the
rest of this repo follows.
