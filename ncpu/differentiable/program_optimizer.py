"""Optimize program parameters by backpropagating through execution.

This module proves nCPU's central thesis: because the CPU is differentiable,
gradient descent can flow backward through an entire program execution trace
to optimize program constants, discover inputs, and fit program parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .execution import (
    DifferentiableEngine,
    FixedProgram,
    Instruction,
    OPCODES,
)


@dataclass
class OptimizationResult:
    """Result of optimizing a program via gradient descent."""

    final_registers: torch.Tensor
    loss_history: list[float]
    initial_immediates: torch.Tensor
    optimized_immediates: torch.Tensor
    converged: bool
    steps: int


class ProgramOptimizer:
    """Optimize program parameters by backpropagating through execution.

    Three optimization modes:

    1. optimize_immediates(): Given a fixed program structure, find the
       immediate values that produce a desired output.

    2. optimize_inputs(): Given a fixed program, find the input register
       values that produce a desired output.

    3. optimize_toward(): General optimization --- provide any loss function
       over the execution result and optimize program immediates.
    """

    def __init__(
        self,
        engine: Optional[DifferentiableEngine] = None,
        lr: float = 0.1,
    ):
        self.engine = engine or DifferentiableEngine()
        self.lr = lr

    def optimize_immediates(
        self,
        program: FixedProgram,
        target_registers: dict[int, float],
        inputs: Optional[dict[int, float]] = None,
        max_iters: int = 1000,
        tolerance: float = 1e-4,
        lr: Optional[float] = None,
    ) -> OptimizationResult:
        """Find immediate values that make the program produce target outputs.

        This is the key demo: gradient descent flows backward through the
        entire execution trace to optimize the program's constants.

        Example: program is "MOV R0, #X; MOV R1, #3; MUL R2, R0, R1; HALT"
        with target R2=42. Gradient descent discovers X=14.
        """
        initial_imm = program.immediates.data.clone()
        optimizer = torch.optim.Adam(list(program.parameters()), lr=lr or self.lr)
        loss_history: list[float] = []
        inputs = inputs or {}

        for step in range(max_iters):
            optimizer.zero_grad()

            result = self.engine.execute_fixed(program, inputs)

            # Loss: MSE between target and actual register values
            loss = torch.tensor(0.0)
            for reg_idx, target_val in target_registers.items():
                loss = loss + (result.registers[reg_idx] - target_val) ** 2

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            loss_history.append(loss_val)

            if loss_val < tolerance:
                result = self.engine.execute_fixed(program, inputs)
                return OptimizationResult(
                    final_registers=result.registers.detach(),
                    loss_history=loss_history,
                    initial_immediates=initial_imm,
                    optimized_immediates=program.immediates.data.clone(),
                    converged=True,
                    steps=step + 1,
                )

        result = self.engine.execute_fixed(program, inputs)
        return OptimizationResult(
            final_registers=result.registers.detach(),
            loss_history=loss_history,
            initial_immediates=initial_imm,
            optimized_immediates=program.immediates.data.clone(),
            converged=False,
            steps=max_iters,
        )

    def optimize_inputs(
        self,
        program: FixedProgram,
        target_registers: dict[int, float],
        input_registers: list[int],
        initial_values: Optional[dict[int, float]] = None,
        max_iters: int = 1000,
        tolerance: float = 1e-4,
        lr: Optional[float] = None,
    ) -> OptimizationResult:
        """Find input values that produce target outputs.

        Creates differentiable input parameters and optimizes them
        through program execution.

        Example: program is "ADD R2, R0, R1; MUL R3, R2, R2; HALT"
        with target R3=100. Finds R0, R1 such that (R0+R1)^2 = 100.
        """
        init_vals = initial_values or {}
        input_params: dict[int, nn.Parameter] = {}
        for reg in input_registers:
            val = init_vals.get(reg, 1.0)
            input_params[reg] = nn.Parameter(torch.tensor(float(val)))

        all_params = list(input_params.values())
        optimizer = torch.optim.Adam(all_params, lr=lr or self.lr)
        loss_history: list[float] = []
        initial_imm = torch.stack([p.data.clone() for p in all_params])

        for step in range(max_iters):
            optimizer.zero_grad()

            inputs = {reg: param for reg, param in input_params.items()}
            result = self.engine.execute_fixed(program, inputs)

            loss = torch.tensor(0.0)
            for reg_idx, target_val in target_registers.items():
                loss = loss + (result.registers[reg_idx] - target_val) ** 2

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            loss_history.append(loss_val)

            if loss_val < tolerance:
                final_imm = torch.stack([p.data.clone() for p in all_params])
                inputs_final = {reg: param for reg, param in input_params.items()}
                result = self.engine.execute_fixed(program, inputs_final)
                return OptimizationResult(
                    final_registers=result.registers.detach(),
                    loss_history=loss_history,
                    initial_immediates=initial_imm,
                    optimized_immediates=final_imm,
                    converged=True,
                    steps=step + 1,
                )

        final_imm = torch.stack([p.data.clone() for p in all_params])
        inputs_final = {reg: param for reg, param in input_params.items()}
        result = self.engine.execute_fixed(program, inputs_final)
        return OptimizationResult(
            final_registers=result.registers.detach(),
            loss_history=loss_history,
            initial_immediates=initial_imm,
            optimized_immediates=final_imm,
            converged=False,
            steps=max_iters,
        )

    def optimize_toward(
        self,
        program: FixedProgram,
        loss_fn: callable,
        inputs: Optional[dict[int, float]] = None,
        max_iters: int = 1000,
        tolerance: float = 1e-4,
        lr: Optional[float] = None,
    ) -> OptimizationResult:
        """General optimization: provide any loss function over execution result.

        loss_fn receives an ExecutionResult and returns a scalar loss tensor.
        Useful for custom objectives (minimize cycles, maximize output, etc).
        """
        initial_imm = program.immediates.data.clone()
        optimizer = torch.optim.Adam(list(program.parameters()), lr=lr or self.lr)
        loss_history: list[float] = []
        inputs = inputs or {}

        for step in range(max_iters):
            optimizer.zero_grad()
            result = self.engine.execute_fixed(program, inputs)
            loss = loss_fn(result)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            loss_history.append(loss_val)

            if loss_val < tolerance:
                result = self.engine.execute_fixed(program, inputs)
                return OptimizationResult(
                    final_registers=result.registers.detach(),
                    loss_history=loss_history,
                    initial_immediates=initial_imm,
                    optimized_immediates=program.immediates.data.clone(),
                    converged=True,
                    steps=step + 1,
                )

        result = self.engine.execute_fixed(program, inputs)
        return OptimizationResult(
            final_registers=result.registers.detach(),
            loss_history=loss_history,
            initial_immediates=initial_imm,
            optimized_immediates=program.immediates.data.clone(),
            converged=False,
            steps=max_iters,
        )


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------


def demo_find_constant():
    """Demo: gradient descent finds that 14 * 3 = 42.

    Program: MOV R0, #X; MOV R1, #3; MUL R2, R0, R1; HALT
    Target: R2 = 42
    Gradient descent discovers X = 14.
    """
    print("=" * 60)
    print("Demo: Find Constant via Gradient Descent")
    print("=" * 60)
    print("Program: MOV R0, #X; MOV R1, #3; MUL R2, R0, R1; HALT")
    print("Target: R2 = 42")
    print("Question: What is X?\n")

    # Use optimize_inputs on R0 with fixed program that multiplies R0 * 3
    program = FixedProgram([
        Instruction(OPCODES["MOV_IMM"], dst=1, immediate=3.0),   # R1 = 3 (fixed)
        Instruction(OPCODES["MUL"], dst=2, src1=0, src2=1),      # R2 = R0 * R1
        Instruction(OPCODES["HALT"]),
    ])

    opt = ProgramOptimizer(lr=0.5)
    result = opt.optimize_inputs(
        program,
        target_registers={2: 42.0},
        input_registers=[0],
        initial_values={0: 1.0},
        max_iters=200,
    )

    x_found = result.optimized_immediates[0].item()
    print(f"Gradient descent found X = {x_found:.4f} (expected 14.0)")
    print(f"R2 = X * 3 = {result.final_registers[2].item():.4f} (expected 42.0)")
    print(f"Converged: {result.converged} in {result.steps} steps")
    print(f"Loss: {result.loss_history[0]:.2f} -> {result.loss_history[-1]:.6f}")
    return result


def demo_find_inputs():
    """Demo: gradient descent finds inputs to a computation.

    Program: ADD R2, R0, R1; MUL R3, R2, R2; HALT
    Target: R3 = 100
    Finds R0, R1 such that (R0 + R1)^2 = 100, i.e. R0 + R1 = 10.
    """
    print("\n" + "=" * 60)
    print("Demo: Find Inputs via Gradient Descent")
    print("=" * 60)
    print("Program: ADD R2, R0, R1; MUL R3, R2, R2; HALT")
    print("Target: R3 = 100  (i.e., (R0 + R1)^2 = 100)")
    print("Question: What are R0 and R1?\n")

    program = FixedProgram([
        Instruction(OPCODES["ADD"], dst=2, src1=0, src2=1),      # R2 = R0 + R1
        Instruction(OPCODES["MUL"], dst=3, src1=2, src2=2),      # R3 = R2 * R2
        Instruction(OPCODES["HALT"]),
    ])

    opt = ProgramOptimizer(lr=0.1)
    result = opt.optimize_inputs(
        program,
        target_registers={3: 100.0},
        input_registers=[0, 1],
        initial_values={0: 1.0, 1: 1.0},
        max_iters=500,
    )

    r0 = result.optimized_immediates[0].item()
    r1 = result.optimized_immediates[1].item()
    print(f"Gradient descent found R0 = {r0:.4f}, R1 = {r1:.4f}")
    print(f"R0 + R1 = {r0 + r1:.4f} (should be ~10)")
    print(f"R3 = (R0+R1)^2 = {result.final_registers[3].item():.4f}")
    print(f"Converged: {result.converged} in {result.steps} steps")
    return result


def demo_optimize_polynomial():
    """Demo: fit polynomial coefficients via differentiable execution.

    Program computes a*x^2 + b*x + c.
    Gradient descent finds a, b, c that fit target points.

    Target: f(x) = 2*x^2 + 3*x + 5
    """
    print("\n" + "=" * 60)
    print("Demo: Fit Polynomial via Differentiable Execution")
    print("=" * 60)
    print("Target: f(x) = 2*x^2 + 3*x + 5")
    print("Program computes a*x^2 + b*x + c; gradient descent finds a, b, c\n")

    # Program: given x in R0, compute a*x^2 + b*x + c in R7
    #   MOV R1, #a        ; a (to be learned)
    #   MOV R2, #b        ; b (to be learned)
    #   MOV R3, #c        ; c (to be learned)
    #   MUL R4, R0, R0    ; x^2
    #   MUL R5, R1, R4    ; a*x^2
    #   MUL R6, R2, R0    ; b*x
    #   ADD R7, R5, R6    ; a*x^2 + b*x
    #   ADD R7, R7, R3    ; a*x^2 + b*x + c
    #   HALT
    program = FixedProgram([
        Instruction(OPCODES["MOV_IMM"], dst=1, immediate=0.5),   # a (guess)
        Instruction(OPCODES["MOV_IMM"], dst=2, immediate=0.5),   # b (guess)
        Instruction(OPCODES["MOV_IMM"], dst=3, immediate=0.5),   # c (guess)
        Instruction(OPCODES["MUL"], dst=4, src1=0, src2=0),      # x^2
        Instruction(OPCODES["MUL"], dst=5, src1=1, src2=4),      # a*x^2
        Instruction(OPCODES["MUL"], dst=6, src1=2, src2=0),      # b*x
        Instruction(OPCODES["ADD"], dst=7, src1=5, src2=6),      # a*x^2 + b*x
        Instruction(OPCODES["ADD"], dst=7, src1=7, src2=3),      # + c
        Instruction(OPCODES["HALT"]),
    ])

    # Training data: f(x) = 2x^2 + 3x + 5 at several points
    train_points = [
        (1.0, 10.0),   # 2+3+5=10
        (2.0, 19.0),   # 8+6+5=19
        (3.0, 32.0),   # 18+9+5=32
        (0.0, 5.0),    # 0+0+5=5
        (-1.0, 4.0),   # 2-3+5=4
    ]

    engine = DifferentiableEngine()
    optimizer = torch.optim.Adam(list(program.parameters()), lr=0.05)
    loss_history = []

    for step in range(2000):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0)

        for x_val, y_target in train_points:
            result = engine.execute_fixed(program, {0: x_val})
            total_loss = total_loss + (result.registers[7] - y_target) ** 2

        total_loss = total_loss / len(train_points)
        total_loss.backward()
        optimizer.step()
        loss_history.append(total_loss.item())

    a = program.immediates.data[0].item()
    b = program.immediates.data[1].item()
    c = program.immediates.data[2].item()
    print(f"Discovered: f(x) = {a:.3f}*x^2 + {b:.3f}*x + {c:.3f}")
    print(f"Target:     f(x) = 2.000*x^2 + 3.000*x + 5.000")
    print(f"Final loss: {loss_history[-1]:.6f}")

    # Verify on a held-out point
    with torch.no_grad():
        result = engine.execute_fixed(program, {0: 5.0})
    predicted = result.registers[7].item()
    actual = 2 * 25 + 3 * 5 + 5  # = 70
    print(f"Verification: f(5) = {predicted:.2f} (expected {actual})")
    return loss_history


if __name__ == "__main__":
    demo_find_constant()
    demo_find_inputs()
    demo_optimize_polynomial()
