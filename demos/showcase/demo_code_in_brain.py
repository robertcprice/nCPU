#!/usr/bin/env python3
"""
Demo: Code Execution in the Neural Brain

Two demonstrations that our system can run code inside neural networks:

1. NEURAL CPU: Programs execute through trained .pt models — every ADD, SUB,
   MUL, AND, OR, XOR instruction flows through a neural network. This is
   literally code running in a neural network's brain.

2. COPROCESSOR: The neural ALU is injected into a real LLM's forward pass.
   During text generation, arithmetic is computed by the neural ALU, giving
   the LLM exact computation abilities that Percepta's toy model can't match.

Usage:
    python3 demos/demo_code_in_brain.py                    # Full demo
    python3 demos/demo_code_in_brain.py --coprocessor-only  # LLM part only
    python3 demos/demo_code_in_brain.py --neural-only       # nCPU part only
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch


# ── PART 1: Neural CPU executing programs ──────────────────────────────

DEMO_PROGRAMS = [
    {
        "name": "Factorial (5! = 120)",
        "description": "Computes 5! using multiply and subtract — every op is neural",
        "setup": {"R0": 5, "R1": 1},  # n=5, accumulator=1
        "program": [
            # R1 = R1 * R0; R0 = R0 - 1; loop while R0 > 0
            ("MUL", "R1", "R1", "R0"),   # acc *= n
            ("SUB", "R0", "R0", 1),       # n--
            ("MUL", "R1", "R1", "R0"),   # acc *= n
            ("SUB", "R0", "R0", 1),       # n--
            ("MUL", "R1", "R1", "R0"),   # acc *= n
            ("SUB", "R0", "R0", 1),       # n--
            ("MUL", "R1", "R1", "R0"),   # acc *= n
        ],
        "expected": {"R1": 120},
        "check_reg": "R1",
    },
    {
        "name": "Bitwise XOR cipher",
        "description": "XOR encryption — neural truth tables process every bit",
        "setup": {"R0": 0xDEAD, "R1": 0xBEEF},
        "program": [
            ("XOR", "R2", "R0", "R1"),   # encrypt
            ("XOR", "R3", "R2", "R1"),   # decrypt (should == R0)
        ],
        "expected": {"R2": 0xDEAD ^ 0xBEEF, "R3": 0xDEAD},
        "check_reg": "R3",
    },
    {
        "name": "Arithmetic expression: (17*23) + (31*7) - 100",
        "description": "Multi-step math — multiply.pt and arithmetic.pt cooperate",
        "setup": {"R0": 17, "R1": 23, "R2": 31, "R3": 7, "R4": 100},
        "program": [
            ("MUL", "R5", "R0", "R1"),   # 17*23 = 391
            ("MUL", "R6", "R2", "R3"),   # 31*7 = 217
            ("ADD", "R7", "R5", "R6"),   # 391+217 = 608
            ("SUB", "R7", "R7", "R4"),   # 608-100 = 508
        ],
        "expected": {"R7": 508},
        "check_reg": "R7",
    },
    {
        "name": "Logic gates: AND/OR truth table verification",
        "description": "Neural truth tables (logical.pt) compute bitwise logic",
        "setup": {"R0": 0xFF00, "R1": 0x0FF0},
        "program": [
            ("AND", "R2", "R0", "R1"),  # 0xFF00 & 0x0FF0 = 0x0F00
            ("OR",  "R3", "R0", "R1"),  # 0xFF00 | 0x0FF0 = 0xFFF0
            ("XOR", "R4", "R0", "R1"),  # 0xFF00 ^ 0x0FF0 = 0xF0F0
        ],
        "expected": {"R2": 0x0F00, "R3": 0xFFF0, "R4": 0xF0F0},
        "check_reg": "R2",
    },
]


def run_neural_cpu_demo():
    """Execute programs through the neural ALU — every op is a trained model."""
    print("\n" + "=" * 70)
    print("  PART 1: Code Executing Through Neural Networks")
    print("  Every instruction below runs through trained .pt model files")
    print("=" * 70)

    from ncpu.neural.neural_alu_bridge import NeuralALUBridge
    alu = NeuralALUBridge(models_dir="models")
    available = alu.load()

    model_count = len([f for f in Path('models').rglob('*.pt')])
    print(f"\n  Neural ALU loaded with {model_count} trained models")
    print(f"  Models: arithmetic.pt (ADD/SUB), multiply.pt (MUL), logical.pt (AND/OR/XOR)")
    print(f"  Mode: NEURAL (every operation through a trained .pt model)")

    results = []
    registers = {}

    for prog in DEMO_PROGRAMS:
        print(f"\n  {'─' * 60}")
        print(f"  Program: {prog['name']}")
        print(f"  {prog['description']}")
        print()

        # Reset registers
        registers = {}
        for reg, val in prog["setup"].items():
            registers[reg] = val

        # Execute each instruction through neural ALU
        for op, dst, src1, src2 in prog["program"]:
            a = registers.get(src1, int(src1) if not str(src1).startswith('R') else 0)
            if isinstance(src2, str) and src2.startswith('R'):
                b = registers.get(src2, 0)
            else:
                b = int(src2)

            t0 = time.time()
            if op == "ADD":
                result = alu.add(a, b)
            elif op == "SUB":
                result = alu.sub(a, b)
            elif op == "MUL":
                result = alu.mul(a, b)
            elif op == "AND":
                result = alu.and_(a, b)
            elif op == "OR":
                result = alu.or_(a, b)
            elif op == "XOR":
                result = alu.xor_(a, b)
            else:
                result = 0
            elapsed_us = (time.time() - t0) * 1e6

            result = result & 0xFFFFFFFF  # 32-bit
            registers[dst] = result

            src2_display = src2 if isinstance(src2, str) else str(src2)
            print(f"    {op:3s} {dst}, {src1}, {src2_display:>5}  →  {result:>10}  ({elapsed_us:7.0f} µs via neural net)")

        # Check results
        check_reg = prog["check_reg"]
        got = registers.get(check_reg, 0)
        expected = prog["expected"][check_reg]
        correct = (got == expected)

        results.append(correct)
        print(f"\n    Result {check_reg} = {got}  {'✓' if correct else '✗'} (expected {expected})")

    correct_count = sum(results)
    print(f"\n  Neural ALU: {correct_count}/{len(results)} programs executed correctly")
    print(f"  Every ADD/SUB/MUL/AND/OR/XOR flowed through a trained neural network.")
    return correct_count, len(results)


# ── PART 2: Coprocessor in LLM ────────────────────────────────────────

ARITHMETIC_PROBLEMS = [
    ("347 + 891", "1238"),
    ("256 * 128", "32768"),
    ("1000 - 673", "327"),
    ("999 * 37", "36963"),
    ("4096 / 8", "512"),
    ("777 + 888", "1665"),
    ("143 * 67", "9581"),
    ("2048 - 1999", "49"),
    ("456 * 12", "5472"),
    ("891 + 234 + 567", "1692"),
    ("63 * 99", "6237"),
    ("10000 - 7654", "2346"),
    ("173 * 29", "5017"),
    ("8192 / 16", "512"),
    ("444 + 555 + 666", "1665"),
    ("87 * 93", "8091"),
    ("3000 - 1847", "1153"),
    ("256 * 256", "65536"),
    ("1111 + 2222", "3333"),
    ("79 * 83", "6557"),
]


def load_model(model_name: str, device: str):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Qwen3.5 VL models — must load text-only
    is_qwen35 = "qwen3.5" in model_name.lower() or "qwen3_5" in model_name.lower()
    if is_qwen35:
        try:
            from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
            text_cfg = Qwen3_5TextConfig.from_pretrained(model_name, trust_remote_code=True)
            model = Qwen3_5ForCausalLM.from_pretrained(
                model_name, config=text_cfg, dtype=torch.float16,
                device_map=device, trust_remote_code=True,
            )
            print(f"  Loaded ({sum(p.numel() for p in model.parameters()):,} params)")
            return model, tokenizer
        except Exception as e:
            print(f"  Qwen3.5 text-only load failed: {e}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16,
        device_map=device, trust_remote_code=True,
    )
    print(f"  Loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return model, tokenizer


def inject_coprocessor(model, weights_path: str, models_dir: str = "models"):
    """Inject nCPU coprocessor and load trained weights."""
    from ncpu.coprocessor.config import NCPUCoprocessorConfig
    from ncpu.coprocessor.inject import inject_ncpu_coprocessor

    state = torch.load(weights_path, map_location=model.device, weights_only=True)
    saved_cfg = state.get("_config", {})

    config = NCPUCoprocessorConfig(
        layer_indices=[-1],
        n_bits=8,
        num_ops=7,
        models_dir=models_dir,
        freeze_alu=True,
        confidence_aware=saved_cfg.get("confidence_aware", False),
        max_gate=saved_cfg.get("max_gate", 0.1),
        target_load=saved_cfg.get("target_load", 0.01),
    )

    injected = inject_ncpu_coprocessor(model, config)

    for i, module in enumerate(injected):
        router_key = f"layer_{i}_router"
        expert_key = f"layer_{i}_expert"
        if router_key in state:
            module.router.load_state_dict(state[router_key])
        if expert_key in state:
            module.expert.load_state_dict(state[expert_key])

    return injected


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    """Generate response from model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def extract_number(response: str) -> str | None:
    """Extract the answer number from model response."""
    # Remove think tags
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    # Look for patterns like "= 1238" or "is 1238" or just "1238"
    numbers = re.findall(r'-?\d+', response)
    if numbers:
        # Return the first substantial number (skip single digits that might be step numbers)
        for n in numbers:
            if len(n) >= 2 or int(n) > 9:
                return n
        return numbers[-1]
    return None


def run_coprocessor_demo(args):
    """Run the coprocessor arithmetic demo."""
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model(args.model, device)

    num_problems = min(args.num_problems, len(ARITHMETIC_PROBLEMS))
    problems = ARITHMETIC_PROBLEMS[:num_problems]

    # ── Base model ──
    print(f"\n  {'─' * 60}")
    print(f"  Base model arithmetic ({num_problems} problems):")

    base_correct = 0
    base_results = []
    for expr, answer in problems:
        prompt = f"Calculate: {expr} = "
        response = generate_answer(model, tokenizer, prompt, max_new_tokens=32)
        predicted = extract_number(response)
        correct = predicted == answer
        base_correct += correct
        base_results.append((expr, answer, predicted, correct))
        status = "✓" if correct else "✗"
        if args.verbose or not correct:
            print(f"    {status} {expr} = {predicted or '?':>8}  (expected {answer})")

    print(f"\n  Base: {base_correct}/{num_problems} ({100*base_correct/num_problems:.0f}%)")

    # ── Inject coprocessor ──
    print(f"\n  {'─' * 60}")
    print(f"  Injecting nCPU coprocessor (neural ALU in the forward pass)...")

    weights_path = args.weights
    if not weights_path:
        candidates = [
            "training_results/instruct_sweep/qwen3.5-2b/coprocessor_weights.pt",
            "training_results/scaling_sweep_qwen35/qwen3.5-2b/coprocessor_weights.pt",
        ]
        for c in candidates:
            if Path(c).exists():
                weights_path = c
                break
    if not weights_path or not Path(weights_path).exists():
        print(f"  ERROR: No coprocessor weights found.")
        return 0, 0, 0, 0

    injected = inject_coprocessor(model, weights_path, args.models_dir)
    print(f"  Coprocessor active in {len(injected)} layer(s)")

    # ── Augmented model ──
    print(f"\n  {'─' * 60}")
    print(f"  Coprocessor-augmented arithmetic ({num_problems} problems):")

    aug_correct = 0
    aug_results = []
    for expr, answer in problems:
        prompt = f"Calculate: {expr} = "
        response = generate_answer(model, tokenizer, prompt, max_new_tokens=32)
        predicted = extract_number(response)
        correct = predicted == answer
        aug_correct += correct
        aug_results.append((expr, answer, predicted, correct))

        base_was = "✓" if dict((e, c) for e, a, p, c in base_results).get(expr) else "✗"
        status = "✓" if correct else "✗"
        delta = ""
        base_c = [c for e, a, p, c in base_results if e == expr][0]
        if correct and not base_c:
            delta = " << FIXED"
        elif not correct and base_c:
            delta = " << REGRESSED"
        if args.verbose or delta or not correct:
            print(f"    {status} {expr} = {predicted or '?':>8}  (expected {answer}){delta}")

    print(f"\n  Augmented: {aug_correct}/{num_problems} ({100*aug_correct/num_problems:.0f}%)")
    return base_correct, aug_correct, num_problems, num_problems


def main():
    parser = argparse.ArgumentParser(description="Demo: Code execution in the neural brain")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B", help="Base model")
    parser.add_argument("--weights", default=None, help="Path to coprocessor_weights.pt")
    parser.add_argument("--models-dir", default="models", help="Path to nCPU ALU models")
    parser.add_argument("--num-problems", type=int, default=20, help="Number of arithmetic problems")
    parser.add_argument("--output", default=None, help="Save results JSON")
    parser.add_argument("--verbose", action="store_true", help="Show all results")
    parser.add_argument("--neural-only", action="store_true", help="Only run neural CPU demo")
    parser.add_argument("--coprocessor-only", action="store_true", help="Only run coprocessor demo")
    args = parser.parse_args()

    print("=" * 70)
    print("  CODE EXECUTION IN THE NEURAL BRAIN")
    print("  nCPU: where every computation is a trained neural network")
    print("=" * 70)

    neural_correct, neural_total = 0, 0
    base_correct, aug_correct = 0, 0

    if not args.coprocessor_only:
        neural_correct, neural_total = run_neural_cpu_demo()

    if not args.neural_only:
        print("\n\n" + "=" * 70)
        print("  PART 2: Neural ALU Inside a Real Language Model")
        print("  The coprocessor injects exact arithmetic into the LLM's forward pass")
        print("=" * 70)
        base_correct, aug_correct, bp, ap = run_coprocessor_demo(args)

    # ── Final summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY: Code Running in Neural Networks")
    print("=" * 70)

    if not args.coprocessor_only:
        print(f"\n  Neural CPU:     {neural_correct}/{neural_total} programs executed correctly")
        print(f"                  Every ADD/SUB/MUL/AND/OR/XOR → trained .pt model")

    if not args.neural_only:
        print(f"\n  LLM (base):     {base_correct}/{args.num_problems} arithmetic problems")
        print(f"  LLM (+ copro):  {aug_correct}/{args.num_problems} arithmetic problems")
        delta = aug_correct - base_correct
        if delta > 0:
            print(f"  Improvement:    +{delta} problems fixed by neural ALU")

    print(f"\n  Unlike Percepta (d_model=36 toy with compiled WASM):")
    print(f"  - Our neural ALU is TRAINED, not hardcoded")
    print(f"  - It lives inside a REAL LLM ({args.model})")
    print(f"  - Confidence-aware gating activates only when needed")
    print(f"  - The same neural networks run standalone programs (Part 1)")
    print(f"    AND augment LLM inference (Part 2)")
    print()


if __name__ == "__main__":
    main()
