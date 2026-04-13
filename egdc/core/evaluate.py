"""Evaluation script for EGDC.

Generates programs from specs, executes them on a simple nCPU ISA
interpreter, and reports pass@1 / pass@k metrics.
"""

from __future__ import annotations
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch

from .tokenizer import NCPUTokenizer, OPCODES, REG_OFFSET, IMM_OFFSET, BR_OFFSET, NUM_REGISTERS
from .model import MaskedDiffusionTransformer, ModelConfig, MASK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN
from .sampler import generate
from .data_generator import NCPUDataGenerator


# ---------------------------------------------------------------------------
# Simple nCPU ISA interpreter (non-differentiable, for evaluation)
# ---------------------------------------------------------------------------

@dataclass
class CPUState:
    registers: List[int]
    flags: Dict[str, bool]  # Z, N, GT
    pc: int
    halted: bool


def execute_program(tokens: List[int], input_registers: Dict[int, int],
                    max_steps: int = 200) -> Optional[Dict[int, int]]:
    """Execute an nCPU ISA program given as token IDs.

    Returns final register state, or None if execution fails/loops.
    """
    # Strip special tokens
    clean = [t for t in tokens if t not in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, MASK_TOKEN)]

    # Parse into instructions
    if len(clean) % 4 != 0:
        clean = clean[:len(clean) // 4 * 4]

    if len(clean) == 0:
        return None

    instructions = []
    for i in range(0, len(clean), 4):
        opcode_id, dst_id, src_id, imm_id = clean[i:i+4]
        instructions.append((opcode_id, dst_id, src_id, imm_id))

    # Init state
    regs = [0] * 8
    for reg_idx, val in input_registers.items():
        if 0 <= reg_idx < 8:
            regs[reg_idx] = val

    flags = {"Z": False, "N": False, "GT": False}
    pc = 0
    steps = 0

    while pc < len(instructions) and steps < max_steps:
        opcode_id, dst_id, src_id, imm_id = instructions[pc]
        steps += 1

        # Decode register indices
        dst = dst_id - REG_OFFSET if 0 <= dst_id - REG_OFFSET < 8 else 0
        src = src_id - REG_OFFSET if 0 <= src_id - REG_OFFSET < 8 else 0
        imm = imm_id - IMM_OFFSET if imm_id >= IMM_OFFSET and imm_id < BR_OFFSET else 0
        branch_target = imm_id - BR_OFFSET if imm_id >= BR_OFFSET else imm

        opcode_name = OPCODES[opcode_id] if 0 <= opcode_id < len(OPCODES) else "NOP"

        if opcode_name == "NOP":
            pc += 1
        elif opcode_name == "HALT":
            return {i: regs[i] for i in range(8)}
        elif opcode_name == "MOV_IMM":
            regs[dst] = imm
            pc += 1
        elif opcode_name == "MOV_REG":
            regs[dst] = regs[src]
            pc += 1
        elif opcode_name == "ADD":
            regs[dst] = regs[dst] + regs[src]
            pc += 1
        elif opcode_name == "SUB":
            regs[dst] = regs[dst] - regs[src]
            pc += 1
        elif opcode_name == "MUL":
            regs[dst] = regs[dst] * regs[src]
            pc += 1
        elif opcode_name == "AND":
            regs[dst] = regs[dst] & regs[src]
            pc += 1
        elif opcode_name == "OR":
            regs[dst] = regs[dst] | regs[src]
            pc += 1
        elif opcode_name == "XOR":
            regs[dst] = regs[dst] ^ regs[src]
            pc += 1
        elif opcode_name == "CMP":
            diff = regs[dst] - regs[src]
            flags["Z"] = (diff == 0)
            flags["N"] = (diff < 0)
            flags["GT"] = (diff > 0)
            pc += 1
        elif opcode_name == "BEQ":
            if flags["Z"]:
                pc = branch_target
            else:
                pc += 1
        elif opcode_name == "BNE":
            if not flags["Z"]:
                pc = branch_target
            else:
                pc += 1
        elif opcode_name == "BGT":
            if flags["GT"]:
                pc = branch_target
            else:
                pc += 1
        else:
            pc += 1

    # Didn't halt cleanly
    return None


def evaluate_program(tokens: List[int], spec: dict) -> Tuple[bool, str]:
    """Evaluate a generated program against its spec.

    Returns (passed, detail_string).
    """
    test_cases = spec.get("test_cases", [])
    if not test_cases:
        return False, "no test cases"

    passed = 0
    total = len(test_cases)

    for tc in test_cases:
        inputs = tc.get("inputs", {})
        expected = tc.get("expected_output")

        # Map input names to register indices
        # Convention: a -> R0, b -> R1, lo -> R2, hi -> R3, n -> R0
        reg_map = {}
        for i, (name, val) in enumerate(inputs.items()):
            reg_map[i] = val

        result = execute_program(tokens, reg_map)
        if result is None:
            continue

        # Check output register (R0 for most programs)
        actual = result.get(0, None)
        if actual == expected:
            passed += 1

    if passed == total:
        return True, f"passed {passed}/{total}"
    else:
        return False, f"passed {passed}/{total}"


def evaluate_model(
    model: MaskedDiffusionTransformer,
    num_tasks: int = 100,
    k: int = 5,
    num_steps: int = 64,
    temperature: float = 0.5,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> dict:
    """Evaluate the model on generated specs.

    Returns:
        dict with pass@1, pass@k, syntactic_validity, execution_rate
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    gen = NCPUDataGenerator()
    tok = NCPUTokenizer()

    results = {
        "pass_at_1": 0,
        "pass_at_k": 0,
        "valid_syntax": 0,
        "executes": 0,
        "total": 0,
    }

    for task_idx in range(num_tasks):
        # Generate a spec
        data = gen.generate_dataset(1)
        spec, ref_tokens = data[0]

        # Encode spec as conditioning tokens
        test_cases = spec["test_cases"]
        spec_token_list = []
        for tc in test_cases[:2]:  # use first 2 test cases as conditioning
            for name, val in tc["inputs"].items():
                spec_token_list.append(IMM_OFFSET + min(val % 256, 255))
            spec_token_list.append(IMM_OFFSET + min(tc["expected_output"] % 256, 255))
        # Pad to 32
        while len(spec_token_list) < 32:
            spec_token_list.append(PAD_TOKEN)
        spec_tensor = torch.tensor([spec_token_list[:32]], dtype=torch.long)

        # Generate k samples
        any_passed = False
        first_passed = False

        for sample_idx in range(k):
            generated = generate(
                model, spec_tensor, seq_len=128,
                num_steps=num_steps, temperature=temperature, device=device,
            )
            gen_tokens = generated[0].tolist()

            # Check syntax: should have valid opcodes
            clean = [t for t in gen_tokens if t not in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, MASK_TOKEN)]
            valid = all(0 <= t < 346 for t in clean)
            if valid:
                results["valid_syntax"] += 1

            # Check execution
            test_input = {}
            for i, (name, val) in enumerate(test_cases[0]["inputs"].items()):
                test_input[i] = val
            exec_result = execute_program(gen_tokens, test_input)
            if exec_result is not None:
                results["executes"] += 1

            # Check correctness
            passed, detail = evaluate_program(gen_tokens, spec)
            if passed:
                any_passed = True
                if sample_idx == 0:
                    first_passed = True

        if first_passed:
            results["pass_at_1"] += 1
        if any_passed:
            results["pass_at_k"] += 1
        results["total"] += 1

        if verbose and (task_idx + 1) % 10 == 0:
            p1 = results["pass_at_1"] / results["total"] * 100
            pk = results["pass_at_k"] / results["total"] * 100
            print(f"  [{task_idx+1}/{num_tasks}] pass@1={p1:.1f}%, pass@{k}={pk:.1f}%")

    # Compute final metrics
    total = results["total"]
    total_samples = total * k
    metrics = {
        "pass_at_1": results["pass_at_1"] / total * 100,
        f"pass_at_{k}": results["pass_at_k"] / total * 100,
        "syntactic_validity": results["valid_syntax"] / total_samples * 100,
        "execution_rate": results["executes"] / total_samples * 100,
        "num_tasks": total,
        "k": k,
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"EGDC Evaluation Results")
        print(f"{'='*50}")
        for key, val in metrics.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.1f}%")
            else:
                print(f"  {key}: {val}")
        print(f"{'='*50}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_size", choices=["tiny", "small", "medium"], default="tiny")
    parser.add_argument("--num_tasks", type=int, default=50)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.5)
    args = parser.parse_args()

    if args.model_size == "tiny":
        cfg = ModelConfig.tiny()
    elif args.model_size == "small":
        cfg = ModelConfig.small()
    else:
        cfg = ModelConfig.medium()

    model = MaskedDiffusionTransformer(cfg)
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {args.checkpoint}")

    evaluate_model(
        model, num_tasks=args.num_tasks, k=args.k,
        num_steps=args.num_steps, temperature=args.temperature,
        device=torch.device("cpu"),
    )
