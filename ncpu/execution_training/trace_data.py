"""Program trace datasets with intermediate state supervision.

Instead of just checking "did the final output match?", trace-level data
provides expected register state at EVERY step of execution. This gives
the differentiable engine maximum gradient signal — every instruction
has a target, not just the last one.

The trace generator:
1. Takes Python code + inputs
2. Executes it in Python, recording variable state after each statement
3. Parses the code to nCPU ISA
4. Maps Python variable states to expected nCPU register states at each step
5. Produces (program, inputs, full_trace) tuples for trace-level training

Usage:
    from ncpu.execution_training.trace_data import TraceGenerator, TraceSample

    gen = TraceGenerator()
    sample = gen.trace_code(
        code="x = a + b\\ny = x * 2",
        inputs={"a": 3, "b": 5},
    )
    # sample.trace = [
    #   (step=0, expected={0: 3}),   # After MOV R0, a
    #   (step=1, expected={1: 5}),   # After MOV R1, b
    #   (step=2, expected={2: 8}),   # After ADD R2, R0, R1
    #   (step=3, expected={3: 16}),  # After MUL R3, R2, #2
    # ]
"""

from __future__ import annotations

import ast
import random
import logging
from dataclasses import dataclass, field
from typing import Optional

from .code_parser import CodeToISAParser, ParseResult, ParseError
from .data import ExecutionTrainingSample

logger = logging.getLogger(__name__)


@dataclass
class TraceStep:
    """Expected register state at one point during execution."""

    instruction_index: int  # Which instruction just completed
    expected_registers: dict[int, float]  # {reg_idx: expected_value}
    description: str = ""  # Human-readable description


@dataclass
class TraceSample:
    """A training sample with full execution trace."""

    code: str
    inputs: dict[str, float]
    parse_result: ParseResult
    trace: list[TraceStep]  # Per-instruction expected states
    final_expected: dict[int, float]  # Expected final register state
    output_var: str
    arg_names: list[str]

    @property
    def trace_as_tuples(self) -> list[tuple[int, dict[int, float]]]:
        """Convert to the format expected by ExecutionLoss."""
        return [
            (step.instruction_index, step.expected_registers)
            for step in self.trace
        ]

    @property
    def n_trace_points(self) -> int:
        return len(self.trace)


class TraceGenerator:
    """Generates execution traces by simulating Python code.

    For each line of code, records the variable state, maps variables
    to nCPU registers, and produces trace-level supervision targets.
    """

    def __init__(self, parser: Optional[CodeToISAParser] = None):
        self.parser = parser or CodeToISAParser()

    def trace_code(
        self,
        code: str,
        inputs: Optional[dict[str, float]] = None,
        arg_names: Optional[list[str]] = None,
        output_var: Optional[str] = None,
    ) -> TraceSample:
        """Generate a full execution trace for Python code.

        Args:
            code: Python source code
            inputs: Input variable values
            arg_names: Input variable names
            output_var: Which variable holds the output

        Returns:
            TraceSample with per-instruction trace points
        """
        inputs = inputs or {}
        arg_names = arg_names or sorted(inputs.keys())

        # Parse to nCPU ISA
        parse_result = self.parser.parse_block(
            code, arg_names=arg_names, output_var=output_var
        )

        # Simulate Python execution, recording state after each statement
        python_trace = self._simulate_python(code, inputs)

        # Map Python variable states to nCPU register states
        trace = self._map_trace_to_registers(
            python_trace, parse_result, inputs, arg_names
        )

        # Build final expected state
        final_state = python_trace[-1] if python_trace else {}
        final_expected = {}
        for var_name, val in final_state.items():
            reg = parse_result.variable_map.get(var_name)
            if reg is not None:
                final_expected[reg] = float(val)

        return TraceSample(
            code=code,
            inputs=inputs,
            parse_result=parse_result,
            trace=trace,
            final_expected=final_expected,
            output_var=output_var or "",
            arg_names=arg_names,
        )

    def trace_sample(self, sample: ExecutionTrainingSample) -> Optional[TraceSample]:
        """Generate a trace from an existing ExecutionTrainingSample."""
        try:
            inputs = {}
            if sample.test_cases:
                inputs = sample.test_cases[0].get("inputs", {})

            return self.trace_code(
                code=sample.reference_code,
                inputs=inputs,
                arg_names=sample.arg_names if sample.arg_names else None,
                output_var=sample.output_var,
            )
        except (ParseError, Exception) as e:
            logger.debug(f"Trace generation failed: {e}")
            return None

    def generate_traced_dataset(
        self,
        n_samples: int = 1000,
        seed: int = 42,
        max_value: int = 50,
    ) -> list[TraceSample]:
        """Generate a dataset of traced arithmetic programs."""
        rng = random.Random(seed)
        samples = []

        templates = [
            # (code_template, arg_names, output_var, description)
            ("result = a + b", ["a", "b"], "result", "simple add"),
            ("result = a - b", ["a", "b"], "result", "simple sub"),
            ("result = a * b", ["a", "b"], "result", "simple mul"),
            ("result = a + b + c", ["a", "b", "c"], "result", "chain add"),
            (
                "temp = a * b\nresult = temp + c",
                ["a", "b", "c"],
                "result",
                "mul then add",
            ),
            (
                "temp = a + b\nresult = temp * c",
                ["a", "b", "c"],
                "result",
                "add then mul",
            ),
            (
                "x = a\nx = x + b\nx = x * c",
                ["a", "b", "c"],
                "x",
                "sequential update",
            ),
            (
                "x = a * 2\ny = b * 3\nresult = x + y",
                ["a", "b"],
                "result",
                "parallel then combine",
            ),
            (
                "x = a + 1\ny = x + 1\nz = y + 1",
                ["a"],
                "z",
                "chain increment",
            ),
            (
                "diff = a - b\nresult = diff * diff",
                ["a", "b"],
                "result",
                "squared difference",
            ),
            # Multi-step variable tracking
            (
                "x = a\nx = x + 3\nx = x * 2",
                ["a"],
                "x",
                "augmented chain",
            ),
            (
                "total = 0\ntotal = total + a\ntotal = total + b\ntotal = total + c",
                ["a", "b", "c"],
                "total",
                "accumulator",
            ),
        ]

        for _ in range(n_samples):
            template = rng.choice(templates)
            code_template, arg_names, output_var, desc = template

            # Generate random inputs
            inputs = {name: rng.randint(1, max_value) for name in arg_names}

            try:
                trace_sample = self.trace_code(
                    code=code_template,
                    inputs=inputs,
                    arg_names=arg_names,
                    output_var=output_var,
                )

                # Verify final values are reasonable
                if trace_sample.final_expected:
                    max_val = max(abs(v) for v in trace_sample.final_expected.values())
                    if max_val > 1e6:
                        continue

                if trace_sample.n_trace_points > 0:
                    samples.append(trace_sample)

            except (ParseError, Exception) as e:
                logger.debug(f"Trace generation failed for '{desc}': {e}")
                continue

        return samples

    def _simulate_python(
        self, code: str, inputs: dict[str, float]
    ) -> list[dict[str, float]]:
        """Execute Python code line by line, recording variable state.

        Returns list of variable state dicts, one per statement.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        env = dict(inputs)
        states = []

        for stmt in tree.body:
            # Execute the statement
            try:
                stmt_code = ast.unparse(stmt)
                exec(stmt_code, {"__builtins__": {"range": range, "abs": abs}}, env)
            except Exception:
                continue

            # Record numeric variable state
            state = {}
            for name, value in env.items():
                if isinstance(value, (int, float)) and not name.startswith("_"):
                    state[name] = float(value)
            states.append(state)

        return states

    def _map_trace_to_registers(
        self,
        python_trace: list[dict[str, float]],
        parse_result: ParseResult,
        inputs: dict[str, float],
        arg_names: list[str],
    ) -> list[TraceStep]:
        """Map Python variable states to nCPU register states at each instruction.

        This is approximate: Python statements don't map 1:1 to nCPU instructions.
        We insert trace points after key instructions (MOV_IMM, ADD, SUB, MUL, etc.)
        where we can predict the register state.
        """
        trace = []
        var_map = parse_result.variable_map
        instructions = parse_result.instructions

        # Track expected register values as we walk through instructions
        expected_regs = {}

        # Initialize from inputs
        for var_name, val in inputs.items():
            reg = var_map.get(var_name)
            if reg is not None:
                expected_regs[reg] = float(val)

        # Walk through instructions and create trace points where we can
        # predict the state from the Python trace
        python_stmt_idx = 0
        inst_idx = 0

        from ncpu.differentiable.execution import OPCODES

        MOV_IMM = OPCODES["MOV_IMM"]
        ADD = OPCODES["ADD"]
        SUB = OPCODES["SUB"]
        MUL = OPCODES["MUL"]
        AND = OPCODES["AND"]
        OR = OPCODES["OR"]
        XOR = OPCODES["XOR"]
        HALT = OPCODES["HALT"]

        alu_ops = {ADD, SUB, MUL, AND, OR, XOR}

        for i, inst in enumerate(instructions):
            if inst.opcode == MOV_IMM:
                expected_regs[inst.dst] = inst.immediate
                trace.append(
                    TraceStep(
                        instruction_index=i,
                        expected_registers=dict(expected_regs),
                        description=f"MOV R{inst.dst}, #{inst.immediate}",
                    )
                )
            elif inst.opcode in alu_ops:
                # Try to compute expected result from known register values
                src1_val = expected_regs.get(inst.src1)
                src2_val = expected_regs.get(inst.src2)

                if src1_val is not None and src2_val is not None:
                    if inst.opcode == ADD:
                        result = src1_val + src2_val
                    elif inst.opcode == SUB:
                        result = src1_val - src2_val
                    elif inst.opcode == MUL:
                        result = src1_val * src2_val
                    elif inst.opcode == AND:
                        result = float(int(src1_val) & int(src2_val))
                    elif inst.opcode == OR:
                        result = float(int(src1_val) | int(src2_val))
                    elif inst.opcode == XOR:
                        result = float(int(src1_val) ^ int(src2_val))
                    else:
                        continue

                    expected_regs[inst.dst] = result
                    trace.append(
                        TraceStep(
                            instruction_index=i,
                            expected_registers=dict(expected_regs),
                            description=f"OP R{inst.dst}, R{inst.src1}, R{inst.src2} = {result}",
                        )
                    )
            elif inst.opcode == HALT:
                trace.append(
                    TraceStep(
                        instruction_index=i,
                        expected_registers=dict(expected_regs),
                        description="HALT",
                    )
                )

        return trace


class TraceLossDataset:
    """Dataset that provides trace-level supervision for training.

    Each item includes:
    - The program (as ParseResult)
    - Input register values
    - Expected trace: per-instruction register states
    - Final expected output
    """

    def __init__(
        self,
        n_samples: int = 1000,
        seed: int = 42,
        max_value: int = 50,
    ):
        gen = TraceGenerator()
        self.samples = gen.generate_traced_dataset(
            n_samples=n_samples, seed=seed, max_value=max_value
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TraceSample:
        return self.samples[idx]

    @property
    def mean_trace_length(self) -> float:
        if not self.samples:
            return 0
        return sum(s.n_trace_points for s in self.samples) / len(self.samples)

    def summary(self) -> str:
        cats = {}
        for s in self.samples:
            n_inst = len(s.parse_result.instructions)
            bucket = f"{n_inst} instructions"
            cats[bucket] = cats.get(bucket, 0) + 1

        lines = [
            f"TraceLossDataset: {len(self.samples)} samples",
            f"Mean trace length: {self.mean_trace_length:.1f} points",
            "Instruction count distribution:",
        ]
        for bucket, count in sorted(cats.items()):
            lines.append(f"  {bucket}: {count}")
        return "\n".join(lines)
