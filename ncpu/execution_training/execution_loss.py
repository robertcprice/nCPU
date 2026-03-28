"""Differentiable execution loss: run parsed programs on the DifferentiableEngine
and return a scalar loss with full gradient flow.

The key insight: instead of checking "did the program produce the right output?"
with a binary pass/fail, we compute MSE between the engine's register state and
the expected output. Because the engine is fully differentiable, gradients flow
from this loss through every instruction execution, every ALU operation, and
every register write — back into whatever produced the program.

Three loss levels:
  1. Output loss: MSE on final register values vs expected output
  2. Trace loss: MSE on intermediate register states (per-step supervision)
  3. Structure loss: penalty for programs that don't halt or use too many steps

Usage:
    engine = DifferentiableEngine()
    loss_fn = ExecutionLoss(engine)

    # From a FixedProgram
    result = loss_fn.compute_fixed(program, inputs={0: 5.0, 1: 3.0},
                                    expected={0: 15.0})

    # From a SoftProgram (full gradient through program structure)
    result = loss_fn.compute_soft(soft_program, inputs={0: 5.0, 1: 3.0},
                                   expected={0: 15.0}, temperature=1.0)

    # Batched (efficient for training)
    result = loss_fn.compute_soft_batched(soft_program, batch_inputs, batch_expected)

    # Combined with LM loss
    total_loss = lm_loss + result.total_loss
    total_loss.backward()  # Gradients flow through execution!
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from ncpu.differentiable.execution import (
    DifferentiableEngine,
    FixedProgram,
    SoftProgram,
    ExecutionResult,
)

logger = logging.getLogger(__name__)


@dataclass
class ExecutionLossResult:
    """Result of computing execution loss."""

    total_loss: torch.Tensor  # Combined scalar loss (differentiable)
    output_loss: torch.Tensor  # MSE on final registers
    trace_loss: Optional[torch.Tensor] = None  # MSE on intermediate states
    structure_loss: Optional[torch.Tensor] = None  # Halt/length penalty
    per_register_loss: dict[int, float] = field(default_factory=dict)
    execution_result: Optional[ExecutionResult] = None
    num_correct: int = 0  # How many output registers are within tolerance
    num_total: int = 0  # Total expected registers

    @property
    def accuracy(self) -> float:
        """Fraction of output registers within tolerance."""
        return self.num_correct / max(self.num_total, 1)


class ExecutionLoss(nn.Module):
    """Differentiable execution loss computation.

    Wraps DifferentiableEngine to provide a clean loss interface for training.
    All loss components maintain gradient flow back through the execution graph.
    """

    def __init__(
        self,
        engine: Optional[DifferentiableEngine] = None,
        output_weight: float = 1.0,
        trace_weight: float = 0.1,
        structure_weight: float = 0.01,
        correctness_tolerance: float = 0.5,
        max_exec_steps: int = 64,
        device: str = "cpu",
    ):
        """
        Args:
            engine: DifferentiableEngine instance (created if None)
            output_weight: Weight for output MSE loss
            trace_weight: Weight for trace MSE loss (0 to disable)
            structure_weight: Weight for structure penalties (0 to disable)
            correctness_tolerance: Threshold for "correct" output
            max_exec_steps: Maximum execution steps
            device: Device for engine tensors
        """
        super().__init__()
        self.engine = engine or DifferentiableEngine(device=device)
        self.output_weight = output_weight
        self.trace_weight = trace_weight
        self.structure_weight = structure_weight
        self.correctness_tolerance = correctness_tolerance
        self.max_exec_steps = max_exec_steps

    def compute_fixed(
        self,
        program: FixedProgram,
        inputs: dict[int, float],
        expected: dict[int, float],
        expected_trace: Optional[list[tuple[int, dict[int, float]]]] = None,
    ) -> ExecutionLossResult:
        """Compute execution loss using a FixedProgram.

        Gradients flow through the program's differentiable immediates
        and the engine's arithmetic operations, but NOT through opcode
        or register selection (those are hard in FixedProgram).

        Args:
            program: FixedProgram to execute
            inputs: Initial register values {reg_idx: value}
            expected: Expected final register values {reg_idx: value}
            expected_trace: Optional per-step expected registers
                            [(step, {reg_idx: value}), ...]

        Returns:
            ExecutionLossResult with differentiable loss tensors
        """
        result = self.engine.execute_fixed(
            program, inputs=inputs, max_steps=self.max_exec_steps
        )
        return self._compute_loss(result, expected, expected_trace)

    def compute_soft(
        self,
        program: SoftProgram,
        inputs: dict[int, float],
        expected: dict[int, float],
        temperature: float = 1.0,
        expected_trace: Optional[list[tuple[int, dict[int, float]]]] = None,
        skip_bitwise: bool = True,
    ) -> ExecutionLossResult:
        """Compute execution loss using a SoftProgram.

        Full gradient flow through opcode selection (Gumbel-softmax),
        register addressing (softmax attention), and all ALU operations.
        This is the richest gradient signal.

        Args:
            program: SoftProgram to execute
            inputs: Initial register values
            expected: Expected final register values
            temperature: Gumbel-softmax temperature (lower = more discrete)
            expected_trace: Optional per-step expected registers
            skip_bitwise: Skip bitwise ops for speed (default True)

        Returns:
            ExecutionLossResult with full-gradient loss
        """
        result = self.engine.execute_soft(
            program,
            inputs=inputs,
            max_steps=self.max_exec_steps,
            temperature=temperature,
            skip_bitwise=skip_bitwise,
        )
        return self._compute_loss(result, expected, expected_trace)

    def compute_soft_batched(
        self,
        program: SoftProgram,
        batch_inputs: list[dict[int, float]],
        batch_expected: list[dict[int, float]],
        temperature: float = 1.0,
        batch_traces: Optional[list[Optional[list[tuple[int, dict[int, float]]]]]] = None,
        skip_bitwise: bool = True,
    ) -> ExecutionLossResult:
        """Compute execution loss over a batch of test cases.

        Uses execute_soft_batched for ~25x speedup over sequential execution.
        The program parameters are shared across all test cases; each test
        case gets its own register state.

        Args:
            program: SoftProgram (shared across batch)
            batch_inputs: List of input dicts, one per test case
            batch_expected: List of expected output dicts
            temperature: Gumbel-softmax temperature
            batch_traces: Optional per-example trace expectations
            skip_bitwise: Skip bitwise ops for speed

        Returns:
            Aggregated ExecutionLossResult (mean over batch)
        """
        results = self.engine.execute_soft_batched(
            program,
            batch_inputs=batch_inputs,
            max_steps=self.max_exec_steps,
            temperature=temperature,
            skip_bitwise=skip_bitwise,
        )

        # Aggregate losses across batch
        total_output_loss = torch.tensor(0.0, device=self.engine.device)
        total_trace_loss = torch.tensor(0.0, device=self.engine.device)
        total_structure_loss = torch.tensor(0.0, device=self.engine.device)
        total_correct = 0
        total_expected = 0
        all_per_reg = {}

        for i, (result, expected) in enumerate(zip(results, batch_expected)):
            trace = batch_traces[i] if batch_traces else None
            loss_result = self._compute_loss(result, expected, trace)

            total_output_loss = total_output_loss + loss_result.output_loss
            if loss_result.trace_loss is not None:
                total_trace_loss = total_trace_loss + loss_result.trace_loss
            if loss_result.structure_loss is not None:
                total_structure_loss = total_structure_loss + loss_result.structure_loss
            total_correct += loss_result.num_correct
            total_expected += loss_result.num_total

            for reg, loss_val in loss_result.per_register_loss.items():
                all_per_reg[reg] = all_per_reg.get(reg, 0.0) + loss_val

        n = len(results)
        mean_output = total_output_loss / n
        mean_trace = total_trace_loss / n if self.trace_weight > 0 else None
        mean_structure = total_structure_loss / n if self.structure_weight > 0 else None

        total_loss = self.output_weight * mean_output
        if mean_trace is not None:
            total_loss = total_loss + self.trace_weight * mean_trace
        if mean_structure is not None:
            total_loss = total_loss + self.structure_weight * mean_structure

        mean_per_reg = {k: v / n for k, v in all_per_reg.items()}

        return ExecutionLossResult(
            total_loss=total_loss,
            output_loss=mean_output,
            trace_loss=mean_trace,
            structure_loss=mean_structure,
            per_register_loss=mean_per_reg,
            num_correct=total_correct,
            num_total=total_expected,
        )

    def _compute_loss(
        self,
        result: ExecutionResult,
        expected: dict[int, float],
        expected_trace: Optional[list[tuple[int, dict[int, float]]]] = None,
    ) -> ExecutionLossResult:
        """Compute loss from a single execution result.

        All operations maintain gradient flow through result.registers.
        """
        device = result.registers.device

        # ── Output loss: MSE on final registers ──
        output_loss = torch.tensor(0.0, device=device)
        per_reg_loss = {}
        num_correct = 0

        for reg_idx, expected_val in expected.items():
            actual = result.registers[reg_idx]
            target = torch.tensor(expected_val, device=device, dtype=actual.dtype)
            reg_loss = (actual - target) ** 2
            # Clamp to prevent NaN/Inf from overflow in soft execution
            reg_loss = torch.clamp(reg_loss, max=1e6)
            if torch.isnan(reg_loss) or torch.isinf(reg_loss):
                reg_loss = torch.tensor(1e4, device=device)
            output_loss = output_loss + reg_loss
            per_reg_loss[reg_idx] = reg_loss.item()

            # Check correctness
            actual_val = actual.item()
            if not (torch.isnan(actual) or torch.isinf(actual)):
                if abs(actual_val - expected_val) < self.correctness_tolerance:
                    num_correct += 1

        if expected:
            output_loss = output_loss / len(expected)

        # ── Trace loss: MSE on intermediate states ──
        trace_loss = None
        if expected_trace and self.trace_weight > 0 and result.register_trace:
            trace_loss = torch.tensor(0.0, device=device)
            trace_count = 0

            for step, expected_regs in expected_trace:
                if step < len(result.register_trace):
                    step_regs = result.register_trace[step]
                    for reg_idx, val in expected_regs.items():
                        actual = step_regs[reg_idx]
                        target = torch.tensor(val, device=device, dtype=actual.dtype)
                        trace_loss = trace_loss + (actual - target) ** 2
                        trace_count += 1

            if trace_count > 0:
                trace_loss = trace_loss / trace_count

        # ── Structure loss: penalties for bad execution ──
        structure_loss = None
        if self.structure_weight > 0:
            structure_loss = torch.tensor(0.0, device=device)

            # Penalty if program didn't halt
            if not result.halted:
                structure_loss = structure_loss + torch.tensor(1.0, device=device)

            # Penalty proportional to steps used (encourage shorter programs)
            step_fraction = result.steps_executed / self.max_exec_steps
            structure_loss = structure_loss + torch.tensor(
                step_fraction * 0.1, device=device
            )

        # ── Combined loss ──
        total_loss = self.output_weight * output_loss
        if trace_loss is not None:
            total_loss = total_loss + self.trace_weight * trace_loss
        if structure_loss is not None:
            total_loss = total_loss + self.structure_weight * structure_loss

        return ExecutionLossResult(
            total_loss=total_loss,
            output_loss=output_loss,
            trace_loss=trace_loss,
            structure_loss=structure_loss,
            per_register_loss=per_reg_loss,
            execution_result=result,
            num_correct=num_correct,
            num_total=len(expected),
        )


class ExecutionLossWithParsing(nn.Module):
    """Higher-level loss that takes Python code strings, parses them,
    executes on the differentiable engine, and returns the loss.

    This is the main interface for the training loop. It handles
    parse failures gracefully by returning a fallback loss.

    Usage:
        loss_module = ExecutionLossWithParsing()
        result = loss_module(
            code="x = a + b * 2",
            arg_names=["a", "b"],
            test_cases=[
                {"inputs": {"a": 3, "b": 5}, "expected": {"x": 13}},
                {"inputs": {"a": 7, "b": 2}, "expected": {"x": 11}},
            ]
        )
        result.total_loss.backward()
    """

    def __init__(
        self,
        execution_loss: Optional[ExecutionLoss] = None,
        use_soft_programs: bool = True,
        temperature: float = 1.0,
        fallback_loss: float = 10.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.execution_loss = execution_loss or ExecutionLoss(device=device)
        self.use_soft_programs = use_soft_programs
        self.temperature = temperature
        self.fallback_loss_value = fallback_loss
        self.device = device

        # Lazy import to avoid circular dependency
        self._parser = None

    @property
    def parser(self):
        if self._parser is None:
            from .code_parser import CodeToISAParser
            self._parser = CodeToISAParser()
        return self._parser

    def forward(
        self,
        code: str,
        test_cases: list[dict],
        arg_names: Optional[list[str]] = None,
        output_var: Optional[str] = None,
        is_function: bool = False,
    ) -> ExecutionLossResult:
        """Parse and execute code, returning differentiable loss.

        Args:
            code: Python source code
            test_cases: List of {"inputs": {var: val}, "expected": {var: val}}
                       or {"inputs": {var: val}, "output": val} (output goes to R0)
            arg_names: Input variable names (inferred from test_cases if None)
            output_var: Output variable name (inferred if None)
            is_function: Whether code is a function definition

        Returns:
            ExecutionLossResult (with fallback loss if parsing fails)
        """
        # Infer arg_names from test_cases if not provided
        if arg_names is None and test_cases:
            arg_names = sorted(test_cases[0].get("inputs", {}).keys())

        # Parse
        try:
            if is_function:
                parse_result = self.parser.parse_function(code)
            else:
                parse_result = self.parser.parse_block(
                    code, arg_names=arg_names, output_var=output_var
                )
        except Exception as e:
            logger.debug(f"Parse failed: {e}")
            return self._fallback_result(str(e))

        # Build inputs and expected for engine
        if self.use_soft_programs:
            return self._execute_soft(parse_result, test_cases, arg_names)
        else:
            return self._execute_fixed(parse_result, test_cases, arg_names)

    def _execute_soft(
        self,
        parse_result,
        test_cases: list[dict],
        arg_names: list[str],
    ) -> ExecutionLossResult:
        """Execute with SoftProgram for full gradient flow."""
        from .code_parser import ParseResult

        soft_prog = parse_result.to_soft_program()

        batch_inputs = []
        batch_expected = []

        for tc in test_cases:
            inputs = {}
            for var_name, val in tc.get("inputs", {}).items():
                reg = parse_result.variable_map.get(var_name)
                if reg is not None:
                    inputs[reg] = float(val)

            expected = {}
            if "expected" in tc:
                for var_name, val in tc["expected"].items():
                    reg = parse_result.variable_map.get(var_name)
                    if reg is not None:
                        expected[reg] = float(val)
            elif "output" in tc:
                expected[parse_result.output_register] = float(tc["output"])

            batch_inputs.append(inputs)
            batch_expected.append(expected)

        if len(batch_inputs) == 1:
            return self.execution_loss.compute_soft(
                soft_prog,
                inputs=batch_inputs[0],
                expected=batch_expected[0],
                temperature=self.temperature,
            )
        else:
            return self.execution_loss.compute_soft_batched(
                soft_prog,
                batch_inputs=batch_inputs,
                batch_expected=batch_expected,
                temperature=self.temperature,
            )

    def _execute_fixed(
        self,
        parse_result,
        test_cases: list[dict],
        arg_names: list[str],
    ) -> ExecutionLossResult:
        """Execute with FixedProgram (gradient through immediates only)."""
        fixed_prog = parse_result.to_fixed_program()

        # For FixedProgram, run each test case sequentially
        # (no batched execution for FixedProgram)
        total_loss = torch.tensor(0.0, device=self.device)
        total_correct = 0
        total_expected = 0

        for tc in test_cases:
            inputs = {}
            for var_name, val in tc.get("inputs", {}).items():
                reg = parse_result.variable_map.get(var_name)
                if reg is not None:
                    inputs[reg] = float(val)

            expected = {}
            if "expected" in tc:
                for var_name, val in tc["expected"].items():
                    reg = parse_result.variable_map.get(var_name)
                    if reg is not None:
                        expected[reg] = float(val)
            elif "output" in tc:
                expected[parse_result.output_register] = float(tc["output"])

            result = self.execution_loss.compute_fixed(
                fixed_prog, inputs=inputs, expected=expected
            )
            total_loss = total_loss + result.total_loss
            total_correct += result.num_correct
            total_expected += result.num_total

        n = len(test_cases)
        return ExecutionLossResult(
            total_loss=total_loss / n,
            output_loss=total_loss / n,
            num_correct=total_correct,
            num_total=total_expected,
        )

    def _fallback_result(self, error_msg: str) -> ExecutionLossResult:
        """Return a non-gradient fallback loss when parsing fails."""
        device = self.device
        loss = torch.tensor(self.fallback_loss_value, device=device, requires_grad=False)
        return ExecutionLossResult(
            total_loss=loss,
            output_loss=loss,
            num_correct=0,
            num_total=0,
        )
