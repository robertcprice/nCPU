#!/usr/bin/env python3
"""Quick verification of instruct-trained coprocessor weights.

Tests arithmetic accuracy with and without coprocessor using the same
evaluation format as training.
"""

import argparse
import json
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ncpu.coprocessor.inject import inject_ncpu_coprocessor, load_coprocessor_weights
from ncpu.coprocessor.config import NCPUCoprocessorConfig

# Arithmetic operations
ARITHMETIC_OPS = ['ADD', 'SUB', 'MUL', 'DIV', 'MOD', 'AND', 'OR', 'XOR']


def load_model(model_id: str, device: str = "mps"):
    """Load model with VL config handling."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    if hasattr(config, 'text_config') and not hasattr(config, 'vocab_size'):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        text_config = Qwen3_5TextConfig(**config.text_config.to_dict())
        model = Qwen3_5ForCausalLM.from_pretrained(
            model_id,
            config=text_config,
            dtype=torch.bfloat16,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
        if device == "cpu":
            model = model.to("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            model = model.to("cpu")

    return model, tokenizer


def evaluate_arithmetic(model, tokenizer, device, n_samples=50):
    """Evaluate arithmetic accuracy like training does."""
    model.eval()
    correct_by_op = {op: 0 for op in ARITHMETIC_OPS}
    total_by_op = {op: 0 for op in ARITHMETIC_OPS}

    for op in ARITHMETIC_OPS:
        for _ in range(n_samples):
            a = torch.randint(10, 1000, (1,)).item()
            b = torch.randint(10, 1000, (1,)).item()
            if op == 'DIV' or op == 'MOD':
                b = max(1, b)

            if op == 'ADD':
                expected = a + b
                prompt = f"{a}+{b}="
            elif op == 'SUB':
                expected = a - b
                prompt = f"{a}-{b}="
            elif op == 'MUL':
                expected = a * b
                prompt = f"{a}*{b}="
            elif op == 'DIV':
                expected = a // b
                prompt = f"{a}/{b}="
            elif op == 'MOD':
                expected = a % b
                prompt = f"{a}%{b}="
            elif op == 'AND':
                expected = a & b
                prompt = f"{a}&{b}="
            elif op == 'OR':
                expected = a | b
                prompt = f"{a}|{b}="
            elif op == 'XOR':
                expected = a ^ b
                prompt = f"{a}^{b}="

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=16,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            output = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract result
            import re
            numbers = re.findall(r'-?\d+', output[len(prompt):])
            predicted = int(numbers[0]) if numbers else None

            if predicted == expected:
                correct_by_op[op] += 1
            total_by_op[op] += 1

    overall_correct = sum(correct_by_op.values())
    overall_total = sum(total_by_op.values())

    return {
        "overall_accuracy": overall_correct / overall_total,
        "correct": overall_correct,
        "total": overall_total,
        "per_operation": {op: correct_by_op[op] / total_by_op[op] for op in ARITHMETIC_OPS},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model ID")
    parser.add_argument("--weights", required=True, help="Path to coprocessor weights")
    parser.add_argument("--device", default="mps", help="Device")
    parser.add_argument("--samples", type=int, default=30, help="Samples per operation")
    args = parser.parse_args()

    device = args.device
    if not torch.backends.mps.is_available() and device == "mps":
        device = "cpu"

    # Load model
    model, tokenizer = load_model(args.model, device)

    # Baseline evaluation
    print("=== BASELINE ===")
    baseline = evaluate_arithmetic(model, tokenizer, device, n_samples=args.samples)
    print(f"Overall: {baseline['overall_accuracy']:.1%} ({baseline['correct']}/{baseline['total']})")
    for op, acc in sorted(baseline['per_operation'].items()):
        print(f"  {op}: {acc:.0%}")
    print()

    # Inject coprocessor
    print(f"=== INJECTING COPROCESSOR ===")
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    config_dict = checkpoint.get("_config", {})

    config = NCPUCoprocessorConfig(
        layer_indices=[-1],
        confidence_aware=config_dict.get("confidence_aware", True),
        max_gate=config_dict.get("max_gate", 0.1),
        target_load=config_dict.get("target_load", 0.01),
        deterministic_alu=config_dict.get("deterministic_alu", False),
        layer_gate_strategy=config_dict.get("layer_gate_strategy", "uniform"),
        models_dir=Path(__file__).parent.parent / "models",
    )

    inject_ncpu_coprocessor(model, config=config)
    load_coprocessor_weights(model, args.weights)
    model.eval()
    print("Coprocessor loaded!")

    # Coprocessor evaluation
    print("\n=== WITH COPROCESSOR ===")
    coproc = evaluate_arithmetic(model, tokenizer, device, n_samples=args.samples)
    print(f"Overall: {coproc['overall_accuracy']:.1%} ({coproc['correct']}/{coproc['total']})")
    for op, acc in sorted(coproc['per_operation'].items()):
        delta = acc - baseline['per_operation'].get(op, 0)
        print(f"  {op}: {acc:.0%} ({delta:+.0%})")

    # Summary
    print("\n=== SUMMARY ===")
    delta = coproc['overall_accuracy'] - baseline['overall_accuracy']
    print(f"Baseline: {baseline['overall_accuracy']:.1%}")
    print(f"Coprocessor: {coproc['overall_accuracy']:.1%}")
    print(f"Delta: {delta:+.1%}")


if __name__ == "__main__":
    main()
