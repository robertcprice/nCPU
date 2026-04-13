#!/usr/bin/env python3
"""Full EGDC training pipeline: proxy -> GRPO -> eval."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import torch, time, os

print("=" * 70)
print("EGDC FULL PIPELINE")
print("=" * 70, flush=True)

# =====================================================================
# STEP 1: Train Neural Execution Proxy
# =====================================================================
print("\n>>> STEP 1: Training Neural Execution Proxy...", flush=True)
t0 = time.time()

from egdc.core.neural_proxy import NeuralExecutionProxy, train_proxy
proxy, proxy_metrics = train_proxy(
    num_samples=10000,
    num_epochs=15,
    batch_size=64,
    lr=3e-4,
    checkpoint_dir="checkpoints/egdc",
)
print(f"    Proxy trained in {time.time()-t0:.0f}s", flush=True)
print(f"    Final: loss={proxy_metrics.get('best_loss', '?'):.4f}, "
      f"acc={proxy_metrics.get('best_acc', '?'):.3f}", flush=True)

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

# Run GRPO for a few epochs
num_epochs = 5
specs_per_epoch = 30

for epoch in range(num_epochs):
    epoch_stats = trainer.train_epoch(num_specs=specs_per_epoch)
    print(f"    Epoch {epoch+1}/{num_epochs}: "
          f"reward={epoch_stats['mean_reward']:.3f}  "
          f"loss={epoch_stats['mean_loss']:.4f}  "
          f"kl={epoch_stats['mean_kl']:.4f}  "
          f"pass_rate={epoch_stats.get('mean_pass_rate', 0):.3f}", flush=True)

# Save GRPO model
grpo_path = "checkpoints/egdc/grpo_best.pt"
torch.save(model.state_dict(), grpo_path)
print(f"    GRPO model saved to {grpo_path} ({time.time()-t0:.0f}s)", flush=True)

# =====================================================================
# STEP 3: Evaluate all methods
# =====================================================================
print("\n>>> STEP 3: Evaluation...", flush=True)
t0 = time.time()

from egdc.core.sampler import generate as unguided_generate
from egdc.core.guided_sampler import guided_generate
from egdc.core.evaluate import execute_program
from egdc.core.data_generator import NCPUDataGenerator
from egdc.core.dataset import NCPUDataset

ds = NCPUDataset(num_samples=1)
gen = NCPUDataGenerator()

NUM_TASKS = 30
K = 5
STEPS = 48

data = gen.generate_dataset(NUM_TASKS, balanced=True)

# Load both models
base_model = MaskedDiffusionTransformer(ModelConfig.tiny())
base_model.load_state_dict(torch.load('checkpoints/egdc/best.pt', map_location='cpu', weights_only=True))
base_model.eval()

grpo_model = MaskedDiffusionTransformer(ModelConfig.tiny())
grpo_model.load_state_dict(torch.load(grpo_path, map_location='cpu', weights_only=True))
grpo_model.eval()

methods = {
    "base_unguided": (base_model, False, 1),
    "base_beam4": (base_model, True, 4),
    "grpo_unguided": (grpo_model, False, 1),
    "grpo_beam4": (grpo_model, True, 4),
}

print(f"\n{'Method':<20s} {'pass@5':>8s} {'exec':>8s} {'time':>6s}")
print("-" * 50, flush=True)

for method_name, (m, use_beam, bw) in methods.items():
    passed = 0
    executed = 0
    mt0 = time.time()

    for spec, ref_tokens in data:
        spec_toks = ds._encode_spec(spec)
        spec_tensor = torch.tensor([spec_toks], dtype=torch.long)
        tc = spec['test_cases'][0]
        input_regs = {idx: v for idx, (_, v) in enumerate(tc['inputs'].items())}
        expected = tc['expected_output']

        task_passed = False
        task_executed = False
        for trial in range(K):
            if use_beam:
                tokens, _ = guided_generate(m, spec_tensor, spec['test_cases'],
                    seq_len=128, num_steps=STEPS, beam_width=bw, temperature=0.3)
            else:
                tokens = unguided_generate(m, spec_tensor, seq_len=128,
                    num_steps=STEPS, temperature=0.3, constrained=True)
            result = execute_program(tokens[0].tolist(), input_regs)
            if result:
                task_executed = True
                if result.get(0) == expected:
                    task_passed = True
                    break

        passed += int(task_passed)
        executed += int(task_executed)

    elapsed = time.time() - mt0
    print(f"{method_name:<20s} {passed:>3d}/{NUM_TASKS} ({100*passed/NUM_TASKS:3.0f}%) "
          f"{executed:>3d}/{NUM_TASKS} ({100*executed/NUM_TASKS:3.0f}%) "
          f"{elapsed:5.0f}s", flush=True)

print(f"\nTotal eval time: {time.time()-t0:.0f}s", flush=True)

# =====================================================================
# STEP 4: Train Python model on HumanEval
# =====================================================================
print("\n>>> STEP 4: Training Python Code Diffusion on HumanEval...", flush=True)
t0 = time.time()

from egdc.python.model import PythonMaskedDiffusion, PythonDiffusionConfig
from egdc.core.humaneval import HumanEvalDataset, load_humaneval
from torch.utils.data import DataLoader

pcfg = PythonDiffusionConfig.tiny()
pmodel = PythonMaskedDiffusion(pcfg)
pparams = sum(p.numel() for p in pmodel.parameters())
print(f"    Python model: {pparams:,} params", flush=True)

# Create dataset from HumanEval solutions
try:
    py_ds = HumanEvalDataset(seq_len=512, spec_len=128)
    py_loader = DataLoader(py_ds, batch_size=16, shuffle=True)
    print(f"    Dataset: {len(py_ds)} examples, {len(py_loader)} batches", flush=True)

    import torch.nn.functional as F
    optimizer = torch.optim.AdamW(pmodel.parameters(), lr=3e-4, weight_decay=0.01)
    pmodel.train()

    for epoch in range(10):
        total_loss = 0
        total_acc = 0
        n_batches = 0
        for batch in py_loader:
            masked, mask_pos, original, spec, timesteps = batch
            mask_pos = mask_pos.bool()
            optimizer.zero_grad()
            logits = pmodel(masked, timesteps, spec_tokens=spec)
            logits_flat = logits.view(-1, logits.shape[-1])
            targets_flat = original.view(-1)
            mask_flat = mask_pos.view(-1)
            if mask_flat.any():
                loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat])
                acc = (logits_flat[mask_flat].argmax(-1) == targets_flat[mask_flat]).float().mean().item()
            else:
                loss = torch.tensor(0.0, requires_grad=True)
                acc = 0.0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pmodel.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_acc = total_acc / max(n_batches, 1)
        print(f"    Epoch {epoch+1}/10: loss={avg_loss:.4f} acc={avg_acc:.3f}", flush=True)

    torch.save(pmodel.state_dict(), "checkpoints/egdc/python_best.pt")
    print(f"    Python model saved ({time.time()-t0:.0f}s)", flush=True)
except Exception as e:
    print(f"    Python training failed: {e}", flush=True)

# =====================================================================
# STEP 5: Generate Python code samples
# =====================================================================
print("\n>>> STEP 5: Python Code Generation Samples...", flush=True)
try:
    from egdc.python.tokenizer import PythonCodeTokenizer
    from egdc.python.train import generate_python
    from egdc.core.humaneval import load_humaneval

    ptok = PythonCodeTokenizer()
    pmodel.eval()
    problems = load_humaneval()

    for prob in problems[:5]:
        prompt = prob['prompt']
        # Encode prompt as spec
        spec_ids = ptok.encode(prompt)[:128]
        while len(spec_ids) < 128:
            spec_ids.append(ptok.PAD)
        spec_tensor = torch.tensor([spec_ids], dtype=torch.long)

        tokens = generate_python(pmodel, spec_tensor, seq_len=512, num_steps=64, temperature=0.5)
        code = ptok.decode(tokens[0].tolist())
        # Show first 3 lines
        lines = [l for l in code.split('\n') if l.strip()][:3]
        print(f"    {prob['task_id']}: {' | '.join(lines)}", flush=True)
except Exception as e:
    print(f"    Python generation failed: {e}", flush=True)

print(f"\n{'='*70}")
print("PIPELINE COMPLETE")
print(f"{'='*70}", flush=True)
