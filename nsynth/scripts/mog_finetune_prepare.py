#!/usr/bin/env python3
"""Prepare a harvested Mog corpus for `mlx_lm.lora` LoRA fine-tuning.

Input: a JSONL of verified training records (from NSYNTH_HARVEST), each
    {"messages":[{system},{user},{assistant}]}  — mlx chat format.
Output: an mlx_lm.lora data dir with train.jsonl / valid.jsonl (90/10), DEDUPED by
    (user, assistant) so repeated tasks don't over-weight.

Usage: python3 scripts/mog_finetune_prepare.py <corpus.jsonl> <out_dir> [valid_frac]
Then LoRA fine-tune (local, on Apple Silicon):
    python3 -m mlx_lm.lora --model lmstudio-community/gemma-4-E4B-it-MLX-8bit \\
        --train --data <out_dir> --iters 600 --batch-size 1 --num-layers 8 \\
        --adapter-path <out_dir>/adapter
Merge the adapter into a servable model:
    python3 -m mlx_lm.fuse --model lmstudio-community/gemma-4-E4B-it-MLX-8bit \\
        --adapter-path <out_dir>/adapter --save-path <out_dir>/mog-gemma
Serve + re-measure (point NSYNTH_LOCAL_LLM_MODEL at <out_dir>/mog-gemma), then rerun
the ablation to see the lift over the base model.
"""
import json, sys, os, hashlib


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    corpus, out_dir = sys.argv[1], sys.argv[2]
    valid_frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    os.makedirs(out_dir, exist_ok=True)

    seen, records = set(), []
    for line in open(corpus):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            msgs = rec["messages"]
            u = next(m["content"] for m in msgs if m["role"] == "user")
            a = next(m["content"] for m in msgs if m["role"] == "assistant")
        except Exception:
            continue
        key = hashlib.md5((u + "\x00" + a).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        records.append(rec)

    # Deterministic split (no shuffle RNG -> reproducible): every 1/valid_frac-th
    # record to valid.
    n = len(records)
    step = max(int(1 / valid_frac), 2) if valid_frac > 0 else 10**9
    train = [r for i, r in enumerate(records) if i % step != 0]
    valid = [r for i, r in enumerate(records) if i % step == 0]

    def dump(name, rows):
        with open(os.path.join(out_dir, name), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    dump("train.jsonl", train)
    dump("valid.jsonl", valid)
    print(f"corpus: {n} unique records -> train {len(train)}, valid {len(valid)} in {out_dir}")
    if n < 100:
        print("NOTE: <100 examples is thin for LoRA; harvest more (run the repair loop "
              "over the full representable set with a served model) before training.")


if __name__ == "__main__":
    main()
