#!/usr/bin/env python3
"""Ablation: guided vs unguided generation."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import torch
from egdc.core.model import MaskedDiffusionTransformer, ModelConfig, PAD_TOKEN
from egdc.core.tokenizer import NCPUTokenizer, IMM_OFFSET
from egdc.core.sampler import generate as unguided_generate
from egdc.core.evaluate import execute_program
from egdc.core.data_generator import NCPUDataGenerator
from egdc.core.dataset import NCPUDataset
from egdc.core.execution_guidance import (
    DifferentiableExecutionScorer, ExecutionGuidedSampler, ExecutionSpec,
)

# Load model
cfg = ModelConfig.tiny()
model = MaskedDiffusionTransformer(cfg)
state = torch.load('checkpoints/egdc/best.pt', map_location='cpu', weights_only=True)
model.load_state_dict(state)
model.eval()
print("Model loaded", flush=True)

tok = NCPUTokenizer()
ds = NCPUDataset(num_samples=1)
gen = NCPUDataGenerator()

# Create execution scorer and guided sampler
scorer = DifferentiableExecutionScorer(max_instructions=16, num_registers=8, max_exec_steps=24)
guided_sampler = ExecutionGuidedSampler(
    model, scorer, gamma=3.0, gamma_schedule='cosine_ramp', guidance_start=0.3,
)

NUM_TASKS = 30
K = 5  # samples per task
NUM_STEPS = 64

data = gen.generate_dataset(NUM_TASKS, balanced=True)

results = {"unguided": {"pass": 0, "exec": 0}, "guided": {"pass": 0, "exec": 0}}

print(f"\n{'='*70}")
print(f"ABLATION: Unguided vs Execution-Guided Diffusion")
print(f"Tasks={NUM_TASKS}, K={K}, Steps={NUM_STEPS}")
print(f"{'='*70}\n")

for i, (spec, ref_tokens) in enumerate(data):
    spec_toks = ds._encode_spec(spec)
    spec_tensor = torch.tensor([spec_toks], dtype=torch.long)
    exec_spec = ExecutionSpec.from_data_spec(spec)
    tc = spec['test_cases'][0]
    input_regs = {idx: v for idx, (_, v) in enumerate(tc['inputs'].items())}
    expected = tc['expected_output']

    # --- Unguided ---
    ug_passed = False
    ug_executed = False
    for trial in range(K):
        tokens = unguided_generate(model, spec_tensor, seq_len=128,
                                   num_steps=NUM_STEPS, temperature=0.3, constrained=True)
        gen_tokens = tokens[0].tolist()
        result = execute_program(gen_tokens, input_regs)
        if result:
            ug_executed = True
            if result.get(0) == expected:
                ug_passed = True
                break

    # --- Guided ---
    g_passed = False
    g_executed = False
    for trial in range(K):
        tokens, metrics = guided_sampler.generate(
            spec_tensor, exec_spec, seq_len=128,
            num_steps=NUM_STEPS, temperature=0.3, constrained=True,
        )
        gen_tokens = tokens[0].tolist()
        result = execute_program(gen_tokens, input_regs)
        if result:
            g_executed = True
            if result.get(0) == expected:
                g_passed = True
                break

    results["unguided"]["pass"] += int(ug_passed)
    results["unguided"]["exec"] += int(ug_executed)
    results["guided"]["pass"] += int(g_passed)
    results["guided"]["exec"] += int(g_executed)

    ug_mark = "OK" if ug_passed else ("EX" if ug_executed else "--")
    g_mark = "OK" if g_passed else ("EX" if g_executed else "--")
    print(f"{i+1:2d}. [{spec['name']:12s}]  unguided={ug_mark}  guided={g_mark}", flush=True)

# Summary
print(f"\n{'='*70}")
print(f"RESULTS (pass@{K} / {NUM_TASKS} tasks)")
print(f"{'='*70}")
for method in ["unguided", "guided"]:
    r = results[method]
    pass_rate = r['pass'] / NUM_TASKS * 100
    exec_rate = r['exec'] / NUM_TASKS * 100
    print(f"  {method:12s}: pass@{K}={r['pass']}/{NUM_TASKS} ({pass_rate:.0f}%)  "
          f"executes={r['exec']}/{NUM_TASKS} ({exec_rate:.0f}%)")

improvement = results["guided"]["pass"] - results["unguided"]["pass"]
print(f"\n  Guidance improvement: {improvement:+d} tasks ({improvement/NUM_TASKS*100:+.0f}%)")
print(f"{'='*70}")
