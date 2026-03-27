#!/usr/bin/env python3
"""Benchmark instruct-trained coprocessor weights on real-world tasks.

Compares:
- Baseline (stock model)
- Coprocessor injected with instruct-trained weights

Benchmarks:
- Coding: 10 problems (fibonacci, factorial, etc.)
- Reasoning: 10 problems (weighted_path, etc.)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch

# Add ncpu to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ncpu.coprocessor.inject import inject_ncpu_coprocessor, load_coprocessor_weights


# Coding tasks (same as realworld benchmark)
CODING_TASKS = [
    {
        "name": "fibonacci",
        "prompt": "Write a Python function that returns the nth Fibonacci number.\ndef fibonacci(n):\n",
        "test": "lambda f: f(10) == 55 and f(0) == 0 and f(1) == 1",
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

# Reasoning tasks (simplified)
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
    """Load base model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # Check for Qwen3.5 VL/composite config (has text_config but no vocab_size)
    if hasattr(config, 'text_config') and not hasattr(config, 'vocab_size'):
        print("Detected Qwen3.5 VL model, loading text-only CausalLM...")
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        # Extract text config dict properly
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


def test_code(code: str, test_expr: str) -> bool:
    """Test generated code."""
    try:
        import re
        # Extract function body
        exec(code)
        # Get function name
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
    parser.add_argument("--model", required=True, help="Model ID (e.g., Qwen/Qwen3.5-2B)")
    parser.add_argument("--weights", required=True, help="Path to coprocessor weights")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--device", default="mps", help="Device (mps/cuda/cpu)")
    parser.add_argument("--benchmarks", default="coding,reasoning", help="Benchmarks to run")
    args = parser.parse_args()
    
    device = args.device
    if not torch.backends.mps.is_available() and device == "mps":
        device = "cpu"
    
    print(f"Loading {args.model}...")
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
    model = inject_ncpu_coprocessor(
        model,
        layers=[-1],
        n_bits=8,
        max_gate=0.1,
        confidence_aware=True,
    )
    load_coprocessor_weights(model, args.weights)
    model.eval()
    
    print("\n=== WITH COPROCESSOR ===")
    if "coding" in benchmarks:
        print("Running coding benchmark...")
        results["coprocessor"] = {"coding": benchmark_coding(model, tokenizer, CODING_TASKS)}
        print(f"  Coding: {results['coprocessor']['coding']['accuracy']:.1%}")
    
    if "reasoning" in benchmarks:
        print("Running reasoning benchmark...")
        results["coprocessor"]["reasoning"] = benchmark_reasoning(model, tokenizer, REASONING_TASKS)
        print(f"  Reasoning: {results['coprocessor']['reasoning']['accuracy']:.1%}")
    
    # Deltas
    results["deltas"] = {}
    for bench in benchmarks:
        if bench in results.get("baseline", {}) and bench in results.get("coprocessor", {}):
            base = results["baseline"][bench]["accuracy"]
            cop = results["coprocessor"][bench]["accuracy"]
            results["deltas"][bench] = cop - base
    
    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for bench in benchmarks:
        if bench in results.get("deltas", {}):
            base = results["baseline"][bench]["accuracy"]
            cop = results["coprocessor"][bench]["accuracy"]
            delta = results["deltas"][bench]
            print(f"{bench.upper():>12}: {base:.1%} → {cop:.1%}  ({delta:+.1%})")


if __name__ == "__main__":
    main()
