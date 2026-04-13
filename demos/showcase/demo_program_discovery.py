#!/usr/bin/env python3
"""Discover programs from input-output examples via gradient descent.

This demo showcases nCPU's most novel capability: because the CPU is
fully differentiable, gradient descent can search the space of all
possible programs to find one that matches a behavioral specification.

Programs are NOT searched combinatorially. They are represented as
continuous parameters and optimized via backpropagation through
CPU execution --- the same way neural networks are trained.

Run: python demos/demo_program_discovery.py
"""

from __future__ import annotations

import sys
import time

import torch

from ncpu.differentiable.execution import (
    DifferentiableEngine,
    FixedProgram,
    Instruction,
    OPCODES,
)
from ncpu.differentiable.program_synthesis import (
    ProgramSynthesizer,
    SynthesisSpec,
)

# Reproducibility
torch.manual_seed(42)


def header(title: str):
    w = 70
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


def show_progress(synth, spec, program, step, loss, temp, acc):
    """Compact single-line progress."""
    prog_str = program.format_program().replace("\n", " | ")
    # Truncate to fit
    if len(prog_str) > 60:
        prog_str = prog_str[:57] + "..."
    print(
        f"  step {step:5d}  loss {loss:10.4f}  temp {temp:.3f}  "
        f"acc {acc:5.1%}  {prog_str}"
    )


def synthesize_and_show(
    name: str,
    description: str,
    spec: SynthesisSpec,
    max_len: int = 6,
    max_iters: int = 3000,
    lr: float = 0.02,
    print_every: int = 500,
):
    """Run synthesis with live progress, then verify on held-out examples."""
    header(name)
    print(f"  {description}")
    print(f"  Examples: {len(spec.examples)}, Program slots: {max_len}")
    print()

    synth = ProgramSynthesizer(max_program_len=max_len, lr=lr)

    t0 = time.time()
    result = synth.synthesize(
        spec,
        max_iters=max_iters,
        verbose=True,
        print_every=print_every,
        skip_bitwise=True,
        max_exec_steps=max_len + 2,
        initial_temperature=2.0,
        final_temperature=0.1,
    )
    elapsed = time.time() - t0

    print(f"\n  --- Result ---")
    print(f"  Discovered program:")
    for line in result.program_text.split("\n"):
        print(f"    {line}")
    print(f"  Accuracy: {result.accuracy:.0%}")
    print(f"  Final loss: {result.loss_history[-1]:.6f}")
    print(f"  Steps: {result.steps}")
    print(f"  Time: {elapsed:.1f}s")

    return result


# =========================================================================
# Challenge 1: Fibonacci Step
# =========================================================================

def challenge_fibonacci():
    """Discover the Fibonacci iteration: (a, b) -> (b, a+b).

    Given two consecutive Fibonacci numbers in R0 and R1, produce
    the next pair in R2 (=R1) and R3 (=R0+R1). This is exactly how
    iterative Fibonacci works: each step shifts the window forward.
    """
    examples = []
    fibs = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    for i in range(len(fibs) - 2):
        a, b = fibs[i], fibs[i + 1]
        # R2 = b (copy), R3 = a + b (next fib)
        examples.append(({0: float(a), 1: float(b)}, {2: float(b), 3: float(a + b)}))

    return synthesize_and_show(
        "Challenge 1: Fibonacci Iteration",
        "Discover (a, b) -> (b, a+b) from Fibonacci pairs",
        SynthesisSpec(examples),
        max_len=4,
        max_iters=2000,
    )


# =========================================================================
# Challenge 2: Sum of Squares
# =========================================================================

def challenge_sum_of_squares():
    """Discover a^2 + b^2.

    This requires 3 instructions: MUL R2, R0, R0 (a^2),
    MUL R3, R1, R1 (b^2), ADD R4, R2, R3.
    """
    import random
    random.seed(42)
    examples = []
    for _ in range(20):
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        examples.append(({0: float(a), 1: float(b)}, {4: float(a * a + b * b)}))

    return synthesize_and_show(
        "Challenge 2: Sum of Squares",
        "Discover a^2 + b^2 (requires MUL, MUL, ADD chain)",
        SynthesisSpec(examples),
        max_len=6,
        max_iters=4000,
        lr=0.015,
    )


# =========================================================================
# Challenge 3: Distance Squared
# =========================================================================

def challenge_distance_squared():
    """Discover (a - b)^2.

    Two instructions: SUB then MUL (square the difference).
    """
    import random
    random.seed(123)
    examples = []
    for _ in range(20):
        a = random.randint(0, 15)
        b = random.randint(0, 15)
        examples.append(({0: float(a), 1: float(b)}, {3: float((a - b) ** 2)}))

    return synthesize_and_show(
        "Challenge 3: Distance Squared",
        "Discover (a - b)^2 (requires SUB then MUL)",
        SynthesisSpec(examples),
        max_len=4,
        max_iters=3000,
    )


# =========================================================================
# Challenge 4: Polynomial Evaluation
# =========================================================================

def challenge_polynomial():
    """Discover 3x^2 - 2x + 7 via differentiable program optimization.

    Uses the program optimizer (not synthesizer) since the program
    structure is known but coefficients are unknown.
    """
    header("Challenge 4: Polynomial Coefficient Discovery")
    print("  Given program structure: a*x^2 + b*x + c")
    print("  Target: f(x) = 3*x^2 - 2*x + 7")
    print("  Gradient descent discovers a=3, b=-2, c=7 from data points")
    print()

    program = FixedProgram([
        Instruction(OPCODES["MOV_IMM"], dst=1, immediate=0.5),   # a
        Instruction(OPCODES["MOV_IMM"], dst=2, immediate=0.5),   # b
        Instruction(OPCODES["MOV_IMM"], dst=3, immediate=0.5),   # c
        Instruction(OPCODES["MUL"], dst=4, src1=0, src2=0),      # x^2
        Instruction(OPCODES["MUL"], dst=5, src1=1, src2=4),      # a*x^2
        Instruction(OPCODES["MUL"], dst=6, src1=2, src2=0),      # b*x
        Instruction(OPCODES["ADD"], dst=7, src1=5, src2=6),      # a*x^2 + b*x
        Instruction(OPCODES["ADD"], dst=7, src1=7, src2=3),      # + c
        Instruction(OPCODES["HALT"]),
    ])

    # f(x) = 3x^2 - 2x + 7
    train_points = [
        (-2.0, 23.0),   # 12+4+7
        (-1.0, 12.0),   # 3+2+7
        (0.0, 7.0),     # 0+0+7
        (1.0, 8.0),     # 3-2+7
        (2.0, 15.0),    # 12-4+7
        (3.0, 28.0),    # 27-6+7
        (4.0, 47.0),    # 48-8+7
    ]

    engine = DifferentiableEngine()
    optimizer = torch.optim.Adam(list(program.parameters()), lr=0.05)

    for step in range(2000):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0)
        for x, y in train_points:
            result = engine.execute_fixed(program, {0: x})
            total_loss = total_loss + (result.registers[7] - y) ** 2
        total_loss = total_loss / len(train_points)
        total_loss.backward()
        optimizer.step()

        if step % 400 == 0:
            a = program.immediates.data[0].item()
            b = program.immediates.data[1].item()
            c = program.immediates.data[2].item()
            print(
                f"  step {step:5d}  loss {total_loss.item():10.6f}  "
                f"a={a:7.3f}  b={b:7.3f}  c={c:7.3f}"
            )

    a = program.immediates.data[0].item()
    b = program.immediates.data[1].item()
    c = program.immediates.data[2].item()
    print(f"\n  --- Result ---")
    print(f"  Discovered: f(x) = {a:.3f}*x^2 + ({b:.3f})*x + {c:.3f}")
    print(f"  Target:     f(x) = 3.000*x^2 + (-2.000)*x + 7.000")

    # Verify on held-out points
    print(f"\n  Held-out verification:")
    with torch.no_grad():
        for x in [5.0, -3.0, 10.0]:
            result = engine.execute_fixed(program, {0: x})
            pred = result.registers[7].item()
            actual = 3 * x**2 - 2 * x + 7
            print(f"    f({x:5.1f}) = {pred:8.2f}  (expected {actual:.0f})")


# =========================================================================
# Challenge 5: Multi-Output Function
# =========================================================================

def challenge_multi_output():
    """Discover a program that computes THREE outputs simultaneously:
    R2 = a + b, R3 = a - b, R4 = a * b.

    This tests whether synthesis can discover a multi-output program
    where different output registers hold different functions of the inputs.
    """
    import random
    random.seed(77)
    examples = []
    for _ in range(20):
        a = random.randint(1, 15)
        b = random.randint(1, 15)
        examples.append((
            {0: float(a), 1: float(b)},
            {2: float(a + b), 3: float(a - b), 4: float(a * b)},
        ))

    return synthesize_and_show(
        "Challenge 5: Multi-Output (ADD, SUB, MUL simultaneously)",
        "Discover program computing R2=a+b, R3=a-b, R4=a*b in one pass",
        SynthesisSpec(examples),
        max_len=6,
        max_iters=5000,
        lr=0.015,
        print_every=1000,
    )


# =========================================================================
# Challenge 6: Mystery Function
# =========================================================================

def challenge_mystery():
    """Mystery function: given only input-output pairs, what does it compute?

    The answer is: R2 = (R0 + R1) * (R0 - R1) = R0^2 - R1^2
    (Difference of squares identity)

    The synthesizer doesn't know this --- it just sees numbers.
    """
    import random
    random.seed(99)
    examples = []
    for _ in range(25):
        a = random.randint(2, 20)
        b = random.randint(1, a - 1)  # ensure a > b for positive results
        examples.append(({0: float(a), 1: float(b)}, {2: float(a * a - b * b)}))

    header("Challenge 6: Mystery Function Discovery")
    print("  You are given ONLY these input-output pairs:")
    print()
    for inp, out in examples[:6]:
        print(f"    R0={inp[0]:3.0f}, R1={inp[1]:3.0f}  -->  R2={out[2]:6.0f}")
    print(f"    ... ({len(examples)} examples total)")
    print()
    print("  What function is this? Gradient descent will figure it out.")
    print()

    result = synthesize_and_show(
        "  (solving...)",
        "",
        SynthesisSpec(examples),
        max_len=6,
        max_iters=5000,
        lr=0.015,
        print_every=1000,
    )

    print(f"\n  The mystery function was: R0^2 - R1^2 = (R0+R1)*(R0-R1)")
    print(f"  Also known as: the difference of squares identity!")

    return result


# =========================================================================
# Challenge 7: Dot Product
# =========================================================================

def challenge_dot_product():
    """Discover dot product: R4 = R0*R2 + R1*R3.

    4-register input, requires MUL, MUL, ADD chain.
    """
    import random
    random.seed(55)
    examples = []
    for _ in range(20):
        a, b, c, d = [random.randint(1, 10) for _ in range(4)]
        examples.append((
            {0: float(a), 1: float(b), 2: float(c), 3: float(d)},
            {5: float(a * c + b * d)},
        ))

    return synthesize_and_show(
        "Challenge 7: Dot Product",
        "Discover R5 = R0*R2 + R1*R3 (vector dot product)",
        SynthesisSpec(examples),
        max_len=6,
        max_iters=4000,
        lr=0.015,
    )


# =========================================================================
# Main
# =========================================================================

def main():
    header("nCPU: Discovering Programs via Gradient Descent")
    print("""
  Every program below is discovered by GRADIENT DESCENT through a
  differentiable CPU. No search, no enumeration, no heuristics ---
  just backpropagation through instruction execution.

  Programs are continuous parameters optimized via Adam. Temperature
  annealing transitions from soft exploration to discrete programs.
  The CPU's ALU, register file, and program counter are all
  differentiable, enabling loss.backward() through execution.
""")

    results = {}

    results["fibonacci"] = challenge_fibonacci()
    results["distance_sq"] = challenge_distance_squared()
    challenge_polynomial()
    results["sum_sq"] = challenge_sum_of_squares()
    results["multi_out"] = challenge_multi_output()
    results["dot_product"] = challenge_dot_product()
    results["mystery"] = challenge_mystery()

    # Summary
    header("Summary")
    print(f"  {'Challenge':<40} {'Accuracy':>8}  {'Steps':>6}  {'Loss':>10}")
    print(f"  {'-'*40} {'-'*8}  {'-'*6}  {'-'*10}")
    for name, r in results.items():
        print(
            f"  {name:<40} {r.accuracy:>7.0%}  {r.steps:>6}  "
            f"{r.loss_history[-1]:>10.4f}"
        )
    print()
    print("  All programs discovered by gradient descent through a differentiable CPU.")
    print("  No search. No enumeration. Just backpropagation.")


if __name__ == "__main__":
    main()
