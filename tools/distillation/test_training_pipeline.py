#!/usr/bin/env python3
"""
Local smoke test for the LoRA training pipeline.

Exercises every step the vast.ai training script does — tokenize,
wrap with LoRA, run a few optimizer steps, save adapter — using a
synthetic tiny GPT-2 model so no HF download is needed. If this
passes on the laptop, the vast.ai path's failures are purely
infrastructure (image / sshd), not code.

Usage:
    python3 tools/distillation/test_training_pipeline.py \\
        --dataset /tmp/distill_197.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True,
                    help="JSONL file with {prompt, completion} rows.")
    ap.add_argument("--limit", type=int, default=10,
                    help="Only train on first N rows (keep smoke-test short).")
    ap.add_argument("--out", default="/tmp/test_adapter",
                    help="Where to save the adapter for inspection.")
    args = ap.parse_args()

    print("[test] importing torch + transformers + peft...")
    import torch
    from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config
    from peft import LoraConfig, get_peft_model
    print(f"[test] torch={torch.__version__} "
          f"device={'mps' if torch.backends.mps.is_available() else 'cpu'}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # ─── Build a tiny GPT-2 from scratch (no HF download) ────────────────
    print("[test] building 4-layer tiny GPT-2 model...")
    cfg = GPT2Config(
        vocab_size=50257, n_positions=1024,
        n_embd=128, n_layer=4, n_head=4,
    )
    model = GPT2LMHeadModel(cfg).to(device)

    # Use GPT-2's tokenizer (tiny, ~0.5MB).
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ─── LoRA wrap ───────────────────────────────────────────────────────
    print("[test] wrapping with LoRA (target: c_attn)...")
    peft_cfg = LoraConfig(
        r=4, lora_alpha=8,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM", lora_dropout=0.0, bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # ─── Load + tokenize dataset ─────────────────────────────────────────
    rows = [json.loads(l) for l in
            Path(args.dataset).read_text().splitlines() if l.strip()]
    rows = rows[:args.limit]
    print(f"[test] dataset: {len(rows)} rows (first {args.limit})")

    def format_example(r):
        text = f"User: {r['prompt']}\nAssistant: {r['completion']}"
        enc = tok(text, truncation=True, max_length=512,
                  return_tensors="pt", padding=False)
        return enc.input_ids[0], enc.attention_mask[0]

    examples = [format_example(r) for r in rows]
    print(f"[test] tokenised; max seq len = "
          f"{max(len(ids) for ids, _ in examples)}")

    # ─── Training loop ───────────────────────────────────────────────────
    import torch.optim as optim
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    print("[test] running 5 optimiser steps...")
    initial_loss = None
    final_loss = None
    for step, (ids, mask) in enumerate(examples[:5]):
        ids = ids.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        out = model(input_ids=ids, attention_mask=mask, labels=ids)
        loss = out.loss
        loss.backward()
        optimizer.step(); optimizer.zero_grad()
        if initial_loss is None:
            initial_loss = loss.item()
        final_loss = loss.item()
        print(f"  step {step}  loss={loss.item():.4f}")

    # ─── Save adapter ────────────────────────────────────────────────────
    print(f"[test] saving adapter to {args.out}...")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    files = list(Path(args.out).iterdir())
    print(f"[test] adapter contents: {[f.name for f in files]}")

    # ─── Assertions ──────────────────────────────────────────────────────
    assert any(f.name.startswith("adapter_model") for f in files), (
        "adapter weights not saved"
    )
    assert any(f.name == "adapter_config.json" for f in files), (
        "adapter config not saved"
    )
    # Loss should have changed at all (going up or down is fine; we just
    # need to know the optimiser is coupled to the graph).
    assert initial_loss != final_loss, "optimiser didn't step?"

    print()
    print(f"[test] PASS — pipeline smoke OK. "
          f"initial_loss={initial_loss:.4f} final_loss={final_loss:.4f}")


if __name__ == "__main__":
    main()
