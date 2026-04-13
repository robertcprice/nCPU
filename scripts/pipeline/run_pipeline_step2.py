#!/usr/bin/env python3
"""Pipeline from Step 2 onward (proxy already trained)."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import torch, time

print("=" * 70)
print("EGDC PIPELINE (Steps 2-5)")
print("=" * 70, flush=True)

# =====================================================================
# STEP 2: GRPO RL Fine-tuning
# =====================================================================
print("\n>>> STEP 2: GRPO RL Fine-tuning...", flush=True)
t0 = time.time()

from egdc.core.model import MaskedDiffusionTransformer, ModelConfig
from egdc.core.grpo import GRPOTrainer, GRPOConfig

model = MaskedDiffusionTransformer(ModelConfig.tiny())
model.load_state_dict(torch.load('checkpoints/egdc/best.pt', map_location='cpu', weights_only=True))

grpo_cfg = GRPOConfig(
    lr=5e-6,
    kl_weight=0.05,
    num_samples_per_spec=6,
    mask_rate=0.5,
    temperature=0.5,
    num_sampling_steps=32,
    seq_len=128,
)
trainer = GRPOTrainer(model, grpo_cfg, torch.device('cpu'))

grpo_cfg.num_epochs = 3
grpo_cfg.num_specs_per_batch = 10
grpo_cfg.log_interval = 1
history = trainer.train(num_epochs=3)
print(f"    GRPO done. Final rewards: {history.get('rewards', ['?'])[-3:]}", flush=True)

torch.save(model.state_dict(), "checkpoints/egdc/grpo_best.pt")
print(f"    GRPO model saved ({time.time()-t0:.0f}s)", flush=True)

# =====================================================================
# STEP 3: Evaluate all methods
# =====================================================================
print("\n>>> STEP 3: Full Evaluation...", flush=True)
t0 = time.time()

from egdc.core.sampler import generate as unguided_generate
from egdc.core.guided_sampler import guided_generate
from egdc.core.evaluate import execute_program
from egdc.core.data_generator import NCPUDataGenerator
from egdc.core.dataset import NCPUDataset
from egdc.core.neural_proxy import NeuralExecutionProxy, ProxyGuidedSampler

ds = NCPUDataset(num_samples=1)
gen = NCPUDataGenerator()

NUM_TASKS = 30
K = 5
STEPS = 48

data = gen.generate_dataset(NUM_TASKS, balanced=True)

# Load models
base_model = MaskedDiffusionTransformer(ModelConfig.tiny())
base_model.load_state_dict(torch.load('checkpoints/egdc/best.pt', map_location='cpu', weights_only=True))
base_model.eval()

grpo_model = MaskedDiffusionTransformer(ModelConfig.tiny())
grpo_model.load_state_dict(torch.load('checkpoints/egdc/grpo_best.pt', map_location='cpu', weights_only=True))
grpo_model.eval()

# Load proxy
proxy = NeuralExecutionProxy()
proxy.load_state_dict(torch.load('checkpoints/egdc/proxy_best.pt', map_location='cpu', weights_only=True))
proxy.eval()
proxy_sampler = ProxyGuidedSampler(base_model, proxy, guidance_scale=3.0)

def eval_method(name, model_fn):
    passed = 0; executed = 0; mt0 = time.time()
    for spec, ref_tokens in data:
        spec_toks = ds._encode_spec(spec)
        spec_tensor = torch.tensor([spec_toks], dtype=torch.long)
        tc = spec['test_cases'][0]
        input_regs = {idx: v for idx, (_, v) in enumerate(tc['inputs'].items())}
        expected = tc['expected_output']
        tp = False; te = False
        for trial in range(K):
            tokens = model_fn(spec_tensor, spec)
            result = execute_program(tokens[0].tolist(), input_regs)
            if result:
                te = True
                if result.get(0) == expected:
                    tp = True; break
        passed += int(tp); executed += int(te)
    elapsed = time.time() - mt0
    print(f"  {name:<25s} pass@{K}={passed:2d}/{NUM_TASKS} ({100*passed/NUM_TASKS:3.0f}%)  "
          f"exec={executed:2d}/{NUM_TASKS} ({100*executed/NUM_TASKS:3.0f}%)  {elapsed:4.0f}s", flush=True)

# Base unguided
eval_method("base_unguided", lambda s, sp:
    unguided_generate(base_model, s, seq_len=128, num_steps=STEPS, temperature=0.3, constrained=True))

# Base + beam reranking
eval_method("base_beam4", lambda s, sp:
    guided_generate(base_model, s, sp['test_cases'], seq_len=128, num_steps=STEPS, beam_width=4, temperature=0.3)[0])

# Base + proxy guidance
eval_method("base_proxy", lambda s, sp:
    proxy_sampler.generate(s, seq_len=128, num_steps=STEPS, temperature=0.3))

# GRPO unguided
eval_method("grpo_unguided", lambda s, sp:
    unguided_generate(grpo_model, s, seq_len=128, num_steps=STEPS, temperature=0.3, constrained=True))

# GRPO + beam
eval_method("grpo_beam4", lambda s, sp:
    guided_generate(grpo_model, s, sp['test_cases'], seq_len=128, num_steps=STEPS, beam_width=4, temperature=0.3)[0])

print(f"\nEval time: {time.time()-t0:.0f}s", flush=True)

# =====================================================================
# STEP 4: Train Python model on HumanEval
# =====================================================================
print("\n>>> STEP 4: Python Code Diffusion on HumanEval...", flush=True)
t0 = time.time()

from egdc.python.model import PythonMaskedDiffusion, PythonDiffusionConfig
from egdc.core.humaneval import HumanEvalDataset
from egdc.python.tokenizer import PythonCodeTokenizer
from torch.utils.data import DataLoader
import torch.nn.functional as F

pcfg = PythonDiffusionConfig.tiny()
pmodel = PythonMaskedDiffusion(pcfg)
print(f"    Python model: {sum(p.numel() for p in pmodel.parameters()):,} params", flush=True)

try:
    py_ds = HumanEvalDataset(seq_len=512, spec_len=128)
    py_loader = DataLoader(py_ds, batch_size=16, shuffle=True)
    print(f"    HumanEval dataset: {len(py_ds)} examples", flush=True)

    optimizer = torch.optim.AdamW(pmodel.parameters(), lr=3e-4, weight_decay=0.01)
    pmodel.train()

    for epoch in range(15):
        total_loss = 0; total_acc = 0; n = 0
        for batch in py_loader:
            masked, mask_pos, original, spec, timesteps = batch
            mask_pos = mask_pos.bool()
            optimizer.zero_grad()
            logits = pmodel(masked, timesteps, spec_tokens=spec)
            lf = logits.view(-1, logits.shape[-1])
            tf = original.view(-1)
            mf = mask_pos.view(-1)
            if mf.any():
                loss = F.cross_entropy(lf[mf], tf[mf])
                acc = (lf[mf].argmax(-1) == tf[mf]).float().mean().item()
            else:
                loss = torch.tensor(0.0, requires_grad=True); acc = 0.0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pmodel.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); total_acc += acc; n += 1

        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/15: loss={total_loss/n:.4f} acc={total_acc/n:.3f}", flush=True)

    torch.save(pmodel.state_dict(), "checkpoints/egdc/python_best.pt")
    print(f"    Saved ({time.time()-t0:.0f}s)", flush=True)

    # Generate samples
    print("\n    Sample generations:", flush=True)
    from egdc.python.train import generate_python
    from egdc.core.humaneval import load_humaneval
    ptok = PythonCodeTokenizer()
    pmodel.eval()
    problems = load_humaneval()
    for prob in problems[:3]:
        prompt = prob['prompt']
        spec_ids = ptok.encode(prompt)[:128]
        while len(spec_ids) < 128: spec_ids.append(ptok.PAD)
        spec_t = torch.tensor([spec_ids], dtype=torch.long)
        toks = generate_python(pmodel, spec_t, seq_len=512, num_steps=64, temperature=0.5)
        code = ptok.decode(toks[0].tolist())
        lines = [l for l in code.split('\n') if l.strip()][:2]
        print(f"      {prob['task_id']}: {' | '.join(lines)}", flush=True)

except Exception as e:
    import traceback
    print(f"    Error: {e}", flush=True)
    traceback.print_exc()

print(f"\n{'='*70}")
print("PIPELINE COMPLETE")
print(f"{'='*70}", flush=True)
