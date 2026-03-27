#!/usr/bin/env python3
"""Verify instruct-trained weights work correctly."""

import argparse
import json
import torch
from pathlib import Path
from typing import Dict, Any

# Add ncpu to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ncpu.coprocessor.inject import inject_ncpu_coprocessor, load_coprocessor_weights
from ncpu.coprocessor.config import NCPUCoprocessorConfig


# Coding tasks
CODING_TASKS = [
    {
        "name": "fibonacci",
        "prompt": "Write a Python function that returns the nth Fibonacci number.\ndef fibonacci(n):\n",
        "test": "lambda f: f(10) == 55 and f(0) == 00 f(1) == 1",
    },
    {
        "name": "factorial",
        "prompt": "Write a Python function that computes factorial.\ndef factorial(n):\n",
        "test": "lambda f: f(5) == 120 and f(0) == 1",
    },
    {
        "name": "is_prime",
        "prompt": "Write a Python function that checks if a number is prime.\ndef is_prime(n):\n",
        "test": "lambda f: f(7) == True and f(4) == False and f(2) == True",
    },
    {
        "name": "binary_search",
        "prompt": "Write a Python function for binary search.\ndef binary_search(arr, target):\n",
        "test": "lambda f: f([1,2,3,4,5], 3) == 2 and f([1,2,3,4,5], 6) == -1",
    },
    {
        "name": "quick_sort",
        "prompt": "Write a Python function that implements quicksort.\ndef quick_sort(arr):\n",
        "test": "lambda f: f([3,1,4,1,5,9,2,6]) == [1,1,2,3,4,5,6,9]",
    },
]

# Reasoning tasks
REASONING_TASKS = [
    {
        "name": "arithmetic_word",
        "prompt": "A store has 23 apples. They sell 7 and receive a shipment of 15 more. How many apples do they have now?",
        "answer": "31",
    },
    {
        "name": "multi_step",
        "prompt": "John has 50 dollars. He buys 3 books at $12 each. How much money does he have left?",
        "answer": "14",
    },
    {
        "name": "percentage",
        "prompt": "A shirt costs $40. It's on sale for 25% off. What's the sale price?",
        "answer": "30",
    },
]


def load_model(model_id: str, device: str = "mps"):
    """Load model with proper handling for Qwen3.5 VL models."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # Check for VL/composite config
    if hasattr(config, 'text_config') and not hasattr(config, 'vocab_size'):
        print("Detected Qwen3.5 VL model, loading text-only CausalLM...")
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

        text_cfg_dict = (config.text_config if isinstance(config.text_config, dict)
                             else config.text_config.to_dict())
        text_config = Qwen3_5TextConfig(**text_cfg_dict)

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


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate text from model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def extract_code(text: str) -> str:
    """Extract code from model output."""
    import re
    # Find code block
    if "```" in text:
        match = re.search(r'```(?:python)?\s*(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
    # Take first function definition
    lines = text.split('\n')
    code_lines = []
    in_function = False
    for line in lines:
        if line.strip().startswith('def ') or line.strip().startswith('async def'):
            in_function = True
        if in_function:
            code_lines.append(line)
            if line.strip() and not line.startswith(' ') and not line.strip().startswith('def') and code_lines:
                break
    return '\n'.join(code_lines) if code_lines else text


    return text


def test_code(code: str, test_expr: str) -> bool:
    """Test generated code."""
    import re
    try:
        exec(code)
        match = re.search(r'def\s+(\w+)\s*\(', code)
        if match:
            func_name = match.group(1)
            test_fn = eval(test_expr)
            return test_fn(eval(func_name))
    except:
        return False


def benchmark_coding(model, tokenizer, tasks: list) -> dict:
    """Run coding benchmark."""
    import re
    correct = 0
    results = []

    for task in tasks:
        try:
            output = generate(model, tokenizer, task["prompt"], max_new_tokens=256)
            code = extract_code(output)
            passed = test_code(code, task["test"])
            if passed:
                correct += 1
            results.append({
                "name": task["name"],
                "passed": passed,
                "output": output[:200],
            })
        except Exception as e:
            results.append({
                "name": task["name"],
                "passed": False,
                "error": str(e),
            })

    return {
        "correct": correct,
        "total": len(tasks),
        "accuracy": correct / len(tasks),
        "details": results,
    }


def benchmark_reasoning(model, tokenizer, tasks: list) -> dict:
    """Run reasoning benchmark."""
    correct = 0
    results = []

    for task in tasks:
        try:
            output = generate(model, tokenizer, task["prompt"], max_new_tokens=50)
            # Extract number from output
            import re
            numbers = re.findall(r'\d+', output)
            predicted = numbers[-1] if numbers else None
            passed = predicted == task["answer"]
            if passed:
                correct += 1
            results.append({
                "name": task["name"],
                "passed": passed,
                "expected": task["answer"],
                "predicted": predicted,
            })
        except Exception as e:
            results.append({
                "name": task["name"],
                "passed": False,
                "error": str(e),
            })

    return {
        "correct": correct,
        "total": len(tasks),
        "accuracy": correct / len(tasks),
        "details": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model ID")
    parser.add_argument("--weights", required=True, help="Path to coprocessor weights")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--device", default="mps", help="Device (mps/cuda/cpu)")
    parser.add_argument("--benchmarks", default="coding,reasoning", help="Benchmarks to run")
    args = parser.parse_args()

    device = args.device
    if not torch.backends.mps.is_available() and device == "mps":
        device = "cpu"

    # Load model
    model, tokenizer = load_model(args.model, device)

    results = {
        "model": args.model,
        "weights": args.weights,
        "benchmarks": {},
    }

    benchmarks = args.benchmarks.split(",")

    # Baseline
    print("\n=== BASELINE (stock model) ===")
    if "coding" in benchmarks:
        print("Running coding benchmark...")
        results["baseline"] = {"coding": benchmark_coding(model, tokenizer, CODING_TASKS)}
        print(f"  Coding: {results['baseline']['coding']['accuracy']:.1%}")

    if "reasoning" in benchmarks:
        print("Running reasoning benchmark...")
        results["baseline"]["reasoning"] = benchmark_reasoning(model, tokenizer, REASONING_TASKS)
        print(f"  Reasoning: {results['baseline']['reasoning']['accuracy']:.1%}")

    # Inject coprocessor
    print(f"\n=== INJECTING COPROCESSOR ({args.weights}) ===")

    # Load config from weights file
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    config_from_weights = checkpoint.get("_config", {})
    if config_from_weights:
        config = NCPUCoprocessorConfig(**config_from_weights)
    else:
        config = NCPUCoprocessorConfig(
            layers=[-1],
            confidence_aware=True,
            max_gate=0.1,
            n_bits=8,
            target_load=0.01,
            models_dir=Path(__file__).parent.parent / "models",
            freeze_alu=True,
            residual_init_scale=0.01,
            gate_warmup_steps=200,
            deterministic_alu=False,
            layer_gate_strategy="uniform",
        )

    # Inject coprocessor
    inject_ncpu_coprocessor(model, config=config)

    # Load weights
    load_coprocessor_weights(model, args.weights)
    print("Coprocessor weights loaded!")
    model.eval()

    # With coprocessor
    print("\n=== WITH COPROCESSOR ===")
    if "coding" in benchmarks:
        print("Running coding benchmark...")
        results["coprocessor"] = {"coding": benchmark_coding(model, tokenizer, CODING_TASKS)}
        print(f"  Coding: {results['coprocessor']['coding']['accuracy']:.1%}")

    if "reasoning" in benchmarks:
        print("Running reasoning benchmark...")
        results["coprocessor"]["reasoning"] = benchmark_reasoning(model, tokenizer, REASONING_TASKS)
        print(f"  Reasoning: {results['coprocessor']['reasoning']['accuracy']:.1%}")

    # Compute deltas
    results["deltas"] = {}
    for bench in benchmarks:
        if bench in results.get("baseline", {}) and bench in results.get("coprocessor", {}):
            base = results["baseline"][bench]["accuracy"]
            cop = results["coprocessor"][bench]["accuracy"]
            delta = cop - base
            results["deltas"][bench] = delta

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for bench in benchmarks:
        if bench in results.get("deltas", {}):
            base = results["baseline"][bench]["accuracy"]
            cop = results["coprocessor"][bench]["accuracy"]
            delta = results["deltas"][bench]
            print(f"{bench.upper():>12}: {base:.1%} -> {cop:.1%}  ({delta:+.1%})")
        else:
            print(f"{bench.upper():>12}: baseline={base:.1%}, coprocessor={cop:.1%}, delta={delta:+.1%}")


if __name__ == "__main__":
    main()
