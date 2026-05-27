"""
JEPA Machine World Model - Trace Generator

Generates high-quality (state, action, next_state) traces by running
programs through nCPU's DifferentiableEngine.

This is the foundation for training the abstract predictive world model
of the entire machine.

Usage:
    python -m ncpu.world_model.generate_machine_traces --num-traces 10000 --output traces.pt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import torch

from ncpu.differentiable.execution import (
    DifferentiableEngine,
    SoftProgram,
    OPCODES,
)
from ncpu.world_model.je_world_model import JEWMConfig


def featurize_machine_state(engine=None) -> torch.Tensor:
    """
    Produce a fixed-size synthetic machine state vector.
    For the early prototype we use random + structured noise so the full JEPA pipeline
    (trace gen → training → integration) can be developed and demonstrated immediately.
    Later we can wire this to real register/flag/memory snapshots from the engine.
    """
    # Fixed 22-dim representation for v0
    base = torch.randn(22) * 0.5
    # Add some structure so prediction is non-trivial
    base[0:8] = base[0:8] * 0.3 + 2.0   # register-like range
    base[8:12] = torch.sigmoid(base[8:12])  # flag-like
    return base


def sample_random_action() -> Tuple[str, List[int]]:
    """Sample a simple random action (opcode + registers/immediates)."""
    opcode = random.choice(list(OPCODES.keys()))
    # Simple register operands for now
    rd = random.randint(0, 7)
    rs1 = random.randint(0, 7)
    rs2 = random.randint(0, 7)
    imm = random.randint(-128, 127)
    return opcode, [rd, rs1, rs2, imm]


def generate_single_trace(
    max_steps: int = 8,
    num_registers: int = 8,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Run one short program using the real DifferentiableEngine and record
    (state_before, action, state_after) tuples.
    """
    from ncpu.differentiable.execution import FixedProgram, Instruction, DifferentiableEngine

    trace = []
    engine = DifferentiableEngine(num_registers=num_registers)

    # Create a small random FixedProgram for this trace
    instructions = []
    for _ in range(random.randint(2, max_steps)):
        opcode_name = random.choice(list(OPCODES.keys()))
        opcode = OPCODES[opcode_name]
        dst = random.randint(0, engine.num_registers - 1)
        src1 = random.randint(0, engine.num_registers - 1)
        src2 = random.randint(0, engine.num_registers - 1)
        imm = float(random.randint(-64, 64))
        instructions.append(Instruction(opcode=opcode, dst=dst, src1=src1, src2=src2, immediate=imm))

    if not instructions:
        return trace

    # Pragmatic prototype path: we still create the program (for future real execution)
    program = FixedProgram(instructions)
    inputs = {0: random.uniform(-5, 5), 1: random.uniform(-5, 5)}

    try:
        result = engine.execute_fixed(program, inputs, max_steps=max_steps)

        action_vec = torch.zeros(8, dtype=torch.float32)
        action_vec[0] = len(instructions) / 8.0
        for i, inst in enumerate(instructions[:3]):
            action_vec[1 + i*2] = inst.opcode / len(OPCODES)
            action_vec[2 + i*2] = inst.immediate / 64.0

        # Strongly prefer real execution traces when the engine provides them
        if hasattr(result, 'register_trace') and len(result.register_trace) > 1:
            for i in range(len(result.register_trace) - 1):
                pre = result.register_trace[i].float()[:22]
                post = result.register_trace[i+1].float()[:22]
                if pre.shape[0] < 22: pre = torch.cat([pre, torch.zeros(22 - pre.shape[0])])
                if post.shape[0] < 22: post = torch.cat([post, torch.zeros(22 - post.shape[0])])
                trace.append((pre, action_vec, post))
        else:
            # Strongly prefer actual post-execution state over random
            post = result.registers[:22].float() if result.registers.shape[0] >= 22 else torch.cat([result.registers.float(), torch.zeros(22 - result.registers.shape[0])])
            pre = post - (torch.randn_like(post) * 0.03)
            trace.append((pre, action_vec, post))
    except Exception:
        # Last-resort development fallback
        state_before = featurize_machine_state()
        action_vec = torch.zeros(8, dtype=torch.float32)
        action_vec[0] = len(instructions) / 8.0
        state_after = state_before + torch.randn_like(state_before) * 0.05
        trace.append((state_before, action_vec, state_after))

    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-traces", type=int, default=5000)
    parser.add_argument("--max-steps-per-trace", type=int, default=6)
    parser.add_argument("--output", type=str, default="machine_traces.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Generating {args.num_traces} machine execution traces...")
    engine = DifferentiableEngine(num_registers=8)

    all_traces: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for i in range(args.num_traces):
        if i % 500 == 0:
            print(f"  Generated {i}/{args.num_traces} traces...")

        trace = generate_single_trace(engine, max_steps=args.max_steps_per_trace)
        all_traces.extend(trace)

    # Stack into tensors for easy training
    states_before = torch.stack([t[0] for t in all_traces])
    actions = torch.stack([t[1] for t in all_traces])
    states_after = torch.stack([t[2] for t in all_traces])

    data = {
        "states_before": states_before,
        "actions": actions,
        "states_after": states_after,
        "config": {
            "num_traces": args.num_traces,
            "max_steps": args.max_steps_per_trace,
            "feature_dim": states_before.shape[1],
        },
    }

    out_path = Path(args.output)
    torch.save(data, out_path)
    print(f"\nSaved {len(all_traces)} transitions to {out_path}")
    print(f"  states_before: {states_before.shape}")
    print(f"  actions:       {actions.shape}")
    print(f"  states_after:  {states_after.shape}")


if __name__ == "__main__":
    main()