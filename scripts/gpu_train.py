#!/usr/bin/env python3
"""EGDC GPU Training - runs on vast.ai with CUDA."""
import sys, os, time, json
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

print("=" * 70)
print("EGDC GPU Training")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 70, flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("checkpoints/egdc_gpu", exist_ok=True)

# ===================================================================
# PHASE 1: Scale up nCPU ISA model
# ===================================================================
print("\n>>> Phase 1: Training larger nCPU ISA model...", flush=True)

from egdc.core.model import MaskedDiffusionTransformer, ModelConfig
from egdc.core.dataset import NCPUDataset
from egdc.core.train import train

cfg = ModelConfig.small()
model = MaskedDiffusionTransformer(cfg)
params = sum(p.numel() for p in model.parameters())
print(f"  Model: {params:,} params (small: 6L/384H/6heads)", flush=True)

ds = NCPUDataset(num_samples=50000)
loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
print(f"  Dataset: {len(ds)} programs, {len(loader)} batches/epoch", flush=True)

train(model, loader, epochs=30, lr=5e-4, warmup_steps=500,
      log_every=50, checkpoint_every=1000,
      checkpoint_dir="checkpoints/egdc_gpu", device=device)

torch.save(model.state_dict(), "checkpoints/egdc_gpu/small_final.pt")
print("  Phase 1 complete!", flush=True)

# ===================================================================
# PHASE 2: GRPO RL fine-tuning
# ===================================================================
print("\n>>> Phase 2: GRPO RL fine-tuning...", flush=True)

from egdc.core.grpo import GRPOTrainer, GRPOConfig

grpo_cfg = GRPOConfig(
    lr=1e-5,
    kl_weight=0.05,
    num_samples_per_spec=8,
    num_specs_per_batch=20,
    mask_rate=0.5,
    temperature=0.5,
    num_sampling_steps=48,
    num_epochs=10,
    log_interval=1,
)
trainer = GRPOTrainer(model, grpo_cfg, device)
history = trainer.train(num_epochs=10)
torch.save(model.state_dict(), "checkpoints/egdc_gpu/grpo_small.pt")
print("  Phase 2 complete!", flush=True)

# ===================================================================
# PHASE 3: Evaluate
# ===================================================================
print("\n>>> Phase 3: Evaluation...", flush=True)

from egdc.core.sampler import generate as unguided_generate
from egdc.core.guided_sampler import guided_generate
from egdc.core.evaluate import execute_program
from egdc.core.data_generator import NCPUDataGenerator
from egdc.core.dataset import NCPUDataset as DS2

ds2 = DS2(num_samples=1)
gen = NCPUDataGenerator()
data = gen.generate_dataset(50, balanced=True)
model.eval()

K = 10; STEPS = 64
results = {}

for method_name, use_beam, bw in [("unguided", False, 1), ("beam4", True, 4), ("beam8", True, 8)]:
    passed = 0; executed = 0
    for spec, ref in data:
        st = ds2._encode_spec(spec)
        stensor = torch.tensor([st], dtype=torch.long)
        tc = spec['test_cases'][0]
        ir = {i: v for i, (_, v) in enumerate(tc['inputs'].items())}
        exp = tc['expected_output']
        tp = te = False
        for _ in range(K):
            if use_beam:
                tok, _ = guided_generate(model, stensor, spec['test_cases'],
                    seq_len=128, num_steps=STEPS, beam_width=bw, temperature=0.3, device=device)
            else:
                tok = unguided_generate(model, stensor, seq_len=128, num_steps=STEPS,
                    temperature=0.3, constrained=True, device=device)
            r = execute_program(tok[0].tolist(), ir)
            if r:
                te = True
                if r.get(0) == exp: tp = True; break
        passed += int(tp); executed += int(te)
    results[method_name] = {"pass": passed, "exec": executed}
    print(f"  {method_name:12s}: pass@{K}={passed}/50 ({100*passed/50:.0f}%)  "
          f"exec={executed}/50 ({100*executed/50:.0f}%)", flush=True)

# ===================================================================
# PHASE 4: Train Python code diffusion
# ===================================================================
print("\n>>> Phase 4: Python Code Diffusion on The Stack...", flush=True)

try:
    from datasets import load_dataset
    from egdc.python.model import PythonMaskedDiffusion, PythonDiffusionConfig
    from egdc.python.tokenizer import PythonCodeTokenizer

    pcfg = PythonDiffusionConfig(
        vocab_size=260, hidden_dim=512, num_layers=8, num_heads=8,
        ff_dim=2048, max_seq_len=1024, dropout=0.1, timestep_dim=256,
    )
    pmodel = PythonMaskedDiffusion(pcfg).to(device)
    pparams = sum(p.numel() for p in pmodel.parameters())
    print(f"  Python model: {pparams:,} params", flush=True)

    ptok = PythonCodeTokenizer()
    optimizer = torch.optim.AdamW(pmodel.parameters(), lr=3e-4, weight_decay=0.01)
    pmodel.train()

    print("  Loading The Stack (Python)...", flush=True)
    code_ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/python",
                           split="train", streaming=True)

    batch_tokens = []
    step = 0; total_loss = 0; t0 = time.time()

    for example in code_ds:
        content = example.get("content", "")
        if len(content) < 50 or len(content) > 5000:
            continue
        tokens = ptok.encode(content)
        if len(tokens) > 1024: tokens = tokens[:1024]
        while len(tokens) < 1024: tokens.append(ptok.PAD)
        batch_tokens.append(tokens)

        if len(batch_tokens) >= 32:
            batch = torch.tensor(batch_tokens, dtype=torch.long, device=device)
            t = torch.rand(batch.shape[0], device=device)
            mask = torch.rand_like(batch.float()) < (t.unsqueeze(1) * 0.5 + 0.25)
            is_pad = (batch == ptok.PAD)
            mask = mask & ~is_pad
            noisy = batch.clone()
            noisy[mask] = ptok.MASK

            optimizer.zero_grad()
            logits = pmodel(noisy, t)
            lf = logits.view(-1, logits.shape[-1])
            tf = batch.view(-1)
            mf = mask.view(-1)
            if mf.any():
                loss = F.cross_entropy(lf[mf], tf[mf])
                loss.backward()
                nn.utils.clip_grad_norm_(pmodel.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            step += 1
            batch_tokens = []

            if step % 100 == 0:
                avg = total_loss / step
                elapsed = time.time() - t0
                sps = step / elapsed
                print(f"  Step {step:5d}: loss={avg:.4f}  {sps:.1f} steps/s", flush=True)

            if step >= 10000:
                break

    torch.save(pmodel.state_dict(), "checkpoints/egdc_gpu/python_model.pt")
    print(f"  Python model trained ({step} steps, {time.time()-t0:.0f}s)", flush=True)

    # HumanEval eval
    print("  Evaluating on HumanEval...", flush=True)
    from egdc.core.humaneval import load_humaneval
    from egdc.python.train import generate_python
    problems = load_humaneval()

    he_passed = 0
    for prob in problems[:20]:
        prompt = prob['prompt']
        spec_ids = ptok.encode(prompt)[:128]
        while len(spec_ids) < 128: spec_ids.append(ptok.PAD)
        spec_t = torch.tensor([spec_ids], dtype=torch.long, device=device)

        for trial in range(5):
            toks = generate_python(pmodel, spec_t, seq_len=1024, num_steps=128,
                                   temperature=0.5, device=device)
            code = prompt + ptok.decode(toks[0].tolist())
            try:
                test_code = code + "\n" + prob.get("test", "")
                exec(compile(test_code, "<test>", "exec"), {})
                he_passed += 1
                break
            except:
                pass

    print(f"  HumanEval pass@5: {he_passed}/20 ({100*he_passed/20:.0f}%)", flush=True)

except Exception as e:
    import traceback
    print(f"  Python training error: {e}", flush=True)
    traceback.print_exc()

# ===================================================================
# Save results
# ===================================================================
print("\n" + "=" * 70)
print("ALL TRAINING COMPLETE")
print("=" * 70, flush=True)

all_results = {
    "ncpu_isa": results,
    "model_params": params,
    "device": str(device),
}
with open("checkpoints/egdc_gpu/results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Results saved to checkpoints/egdc_gpu/results.json", flush=True)
