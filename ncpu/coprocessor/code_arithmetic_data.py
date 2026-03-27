"""Code-embedded arithmetic training data generator.

Generates arithmetic problems embedded in realistic code contexts,
helping the coprocessor learn WHEN to activate during code generation.

Key insight: The coprocessor currently trains on raw arithmetic ("42 + 17 = 59")
but coding involves contextual arithmetic embedded in loops, indices, and logic.

Training patterns:
- Array indexing: arr[i + offset], arr[i * stride]
- Loop bounds: range(0, n), range(start, end, step)
- Memory addressing: ptr + offset * size
- Boundary checks: if i < len(arr) - 1
- Bit operations: val & mask, 1 << n
- Accumulators: total += item, count *= factor
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class CodePattern(Enum):
    """Types of code-embedded arithmetic patterns."""
    ARRAY_INDEX = "array_index"          # arr[i + offset]
    LOOP_BOUND = "loop_bound"            # range(0, n)
    BOUNDARY_CHECK = "boundary_check"    # if i < len(arr) - 1
    MEMORY_OFFSET = "memory_offset"      # ptr + offset * size
    BIT_OPERATION = "bit_operation"      # val & mask, 1 << n
    ACCUMULATOR = "accumulator"          # total += item
    SLICE_OPERATION = "slice_operation"  # s[start:end]
    POINTER_ARITHMETIC = "pointer_arith" # base + index * stride


@dataclass
class CodeArithmeticSample:
    """A code-embedded arithmetic training example."""
    pattern: CodePattern
    code_snippet: str           # Full code context
    arithmetic_expr: str        # The embedded arithmetic: "i + offset"
    operands: Tuple[int, ...]   # Numeric values involved
    operation: str              # ADD, SUB, MUL, etc.
    result: int                 # Expected result
    explanation: str            # Why this arithmetic matters


class CodeArithmeticGenerator:
    """Generates code-embedded arithmetic training data."""

    # Templates for each pattern type
    TEMPLATES = {
        CodePattern.ARRAY_INDEX: [
            "arr[{a} + {b}] = {result}  # array index calculation",
            "data[{a} * {b}]  # 2D index flattened",
            "buffer[{a} + {b} * 4]  # interleaved data",
            "result = arr[{a} + {b}];  # index = {result}",
            "matrix[row * cols + {a}] = {b}  # row-major order",
        ],
        CodePattern.LOOP_BOUND: [
            "for i in range({a}, {b}):  # {result} iterations",
            "while i < {b}: i += {a}  # loop step",
            "for j in range({a}, {b}, {step}):  # stepped loop",
            "range(0, {a} * {b})  # {result} elements",
        ],
        CodePattern.BOUNDARY_CHECK: [
            "if i < {b} - {a}:  # check i < {result}",
            "assert idx >= {a} and idx < {b}  # valid range",
            "while pos < len(arr) - {a}:  # stop before end-{a}",
            "if start + {a} <= {b}:  # {a} + start <= {b}",
        ],
        CodePattern.MEMORY_OFFSET: [
            "ptr = base + {a} * {b}  # offset = {result}",
            "addr = &data[{a} + {b} * stride]  # byte offset",
            "offset = {a} * sizeof(int) + {b}  # {result} bytes",
            "p = (char*)base + {a} + {b} * 4;  # address calc",
        ],
        CodePattern.BIT_OPERATION: [
            "mask = {a} | {b}  # bits set: {result}",
            "flags = {a} & {b}  # common bits: {result}",
            "bit = 1 << {a}  # value: {result}",
            "xor = {a} ^ {b}  # diff bits: {result}",
            "flags |= (1 << {a})  # set bit {a} = add {result}",
        ],
        CodePattern.ACCUMULATOR: [
            "total += {a}  # total = total + {a}",
            "count *= {b}  # count = count * {b}",
            "sum += arr[{a}]  # add element at index {a}",
            "index = (index + {a}) % {b}  # circular buffer",
        ],
        CodePattern.SLICE_OPERATION: [
            "s[{a}:{b}]  # slice length = {result}",
            "data[start:start + {a}]  # {a} elements",
            "arr[{a}:{a} + {b}]  # slice of {b} items",
            "chunk = buf[i:i + {a}]  # chunk size {a}",
        ],
        CodePattern.POINTER_ARITHMETIC: [
            "ptr + {a}  # advance by {a} elements",
            "offset = {a} * stride + {b}  # 2D offset",
            "index = (row * cols) + col  # linearize 2D",
            "next = curr + {a} * sizeof(Node)  # struct offset",
        ],
    }

    # Operation mapping
    OPS = {
        "ADD": ("+", lambda a, b: a + b),
        "SUB": ("-", lambda a, b: a - b),
        "MUL": ("*", lambda a, b: a * b),
        "DIV": ("//", lambda a, b: a // b if b != 0 else 0),
        "MOD": ("%", lambda a, b: a % b if b != 0 else 0),
        "AND": ("&", lambda a, b: a & b),
        "OR": ("|", lambda a, b: a | b),
        "XOR": ("^", lambda a, b: a ^ b),
        "SHL": ("<<", lambda a, b: a << b),
        "SHR": (">>", lambda a, b: a >> b),
    }

    def __init__(self, seed: int = 42, max_value: int = 255):
        self.rng = random.Random(seed)
        self.max_value = max_value

    def _rand_int(self, min_val: int = 1, max_val: Optional[int] = None) -> int:
        """Generate random integer in range."""
        max_val = max_val or self.max_value
        return self.rng.randint(min_val, max_val)

    def _rand_small(self) -> int:
        """Generate small integer (1-16) for offsets/shifts."""
        return self.rng.randint(1, 16)

    def generate_array_index_sample(self) -> CodeArithmeticSample:
        """Generate array indexing pattern."""
        # Common patterns: i + offset, i * stride, base + offset
        pattern_type = self.rng.choice(["add", "mul", "compound"])

        if pattern_type == "add":
            a, b = self._rand_int(0, 50), self._rand_small()
            result = a + b
            expr = f"{a} + {b}"
            explanation = f"Access element at index {result}"
            code = self.rng.choice([
                f"arr[{a} + {b}]  # array index {result}",
                f"result = arr[{a} + {b}]  # read index {result}",
                f"buffer[{a} + {b}] = value  # write index {result}",
            ])
            operands = (a, b)
            operation = "ADD"
        elif pattern_type == "mul":
            a, b = self._rand_int(0, 20), self._rand_small()
            result = a * b
            expr = f"{a} * {b}"
            explanation = f"2D array: row {a}, col stride {b} → flat index {result}"
            code = self.rng.choice([
                f"flat_index = {a} * {b}  # row-major offset {result}",
                f"data[{a} * {b}]  # flattened index {result}",
                f"row_offset = {a} * {b}  # offset {result}",
            ])
            operands = (a, b)
            operation = "MUL"
        else:  # compound
            a, b = self._rand_int(0, 20), self._rand_int(1, 8)
            stride = self._rand_int(2, 8)
            result = a + b * stride
            expr = f"{a} + {b} * {stride}"
            explanation = f"Interleaved data: base {a} + channel {b} * stride {stride} = {result}"
            code = self.rng.choice([
                f"buffer[{a} + {b} * {stride}]  # interleaved index {result}",
                f"offset = {a} + {b} * {stride}  # byte offset {result}",
                f"pixel_index = {a} + {b} * {stride}  # flattened position {result}",
            ])
            operands = (a, b, stride)
            operation = "COMPOUND"

        return CodeArithmeticSample(
            pattern=CodePattern.ARRAY_INDEX,
            code_snippet=code,
            arithmetic_expr=expr,
            operands=operands,
            operation=operation,
            result=result,
            explanation=explanation,
        )

    def generate_loop_bound_sample(self) -> CodeArithmeticSample:
        """Generate loop bound pattern."""
        start = self._rand_int(0, 20)
        end = start + self._rand_int(5, 50)
        step = self.rng.choice([1, 1, 1, 2, 3, 4])  # bias toward 1

        iterations = (end - start) // step

        if step == 1:
            template = self.rng.choice([
                "for i in range({start}, {end}):  # {result} iterations",
                "range({start}, {end})  # {result} elements",
            ])
            expr = f"{end} - {start}"
            op = "SUB"
        else:
            template = "for i in range({start}, {end}, {step}):  # {result} iterations"
            expr = f"({end} - {start} + {step} - 1) // {step}"
            op = "DIV"
            iterations = (end - start + step - 1) // step

        code = template.format(start=start, end=end, step=step, result=iterations)

        return CodeArithmeticSample(
            pattern=CodePattern.LOOP_BOUND,
            code_snippet=code,
            arithmetic_expr=expr,
            operands=(end - start, step),
            operation=op,
            result=iterations,
            explanation=f"Loop from {start} to {end} (step {step}) runs {iterations} times",
        )

    def generate_boundary_check_sample(self) -> CodeArithmeticSample:
        """Generate boundary check pattern."""
        array_len = self._rand_int(10, 100)
        offset = self._rand_small()

        # Pattern: check if i < len(arr) - offset
        effective_bound = array_len - offset

        templates = [
            f"if i < {{bound}} - {{off}}:  # i < {effective_bound}",
            f"assert idx >= 0 and idx < {{bound}} - {{off}}  # max valid: {effective_bound - 1}",
            f"while pos < {{bound}} - {{off}}:  # pos < {effective_bound}",
        ]

        template = self.rng.choice(templates)
        code = template.format(bound=array_len, off=offset)

        return CodeArithmeticSample(
            pattern=CodePattern.BOUNDARY_CHECK,
            code_snippet=code,
            arithmetic_expr=f"{array_len} - {offset}",
            operands=(array_len, offset),
            operation="SUB",
            result=effective_bound,
            explanation=f"Valid index range is 0 to {effective_bound - 1}",
        )

    def generate_bit_operation_sample(self) -> CodeArithmeticSample:
        """Generate bit manipulation pattern."""
        op_name = self.rng.choice(["AND", "OR", "XOR", "SHL"])
        op_sym, func = self.OPS[op_name]

        if op_name == "SHL":
            b = self._rand_int(0, 7)
            result = 1 << b
            expr = f"1 << {b}"
            code = f"mask = 1 << {b}  # bit value {result}"
            explanation = f"Set bit {b} = value {result}"
            operands = (1, b)
        else:
            a = self._rand_int(0, 255)
            b = self._rand_int(0, 255)
            result = func(a, b)
            expr = f"{a} {op_sym} {b}"

            op_desc = {"AND": "common bits", "OR": "bits set", "XOR": "diff bits"}
            code = f"result = {a} {op_sym} {b}  # {op_desc[op_name]}: {result}"
            explanation = f"{a} {op_sym} {b} = {result} (binary: {bin(result)})"
            operands = (a, b)

        return CodeArithmeticSample(
            pattern=CodePattern.BIT_OPERATION,
            code_snippet=code,
            arithmetic_expr=expr,
            operands=operands,
            operation=op_name,
            result=result,
            explanation=explanation,
        )

    def generate_accumulator_sample(self) -> CodeArithmeticSample:
        """Generate accumulator pattern."""
        op_name = self.rng.choice(["ADD", "MUL", "MOD"])
        op_sym, func = self.OPS[op_name]

        if op_name == "ADD":
            current = self._rand_int(0, 100)
            addend = self._rand_int(1, 50)
            result = current + addend
            code = f"total = {current}\ntotal += {addend}  # total = {result}"
            explanation = f"Accumulate {addend} into sum"
            operands = (current, addend)
            expr = f"{current} + {addend}"
        elif op_name == "MUL":
            current = self._rand_int(1, 10)
            factor = self._rand_int(2, 5)
            result = current * factor
            code = f"count = {current}\ncount *= {factor}  # count = {result}"
            explanation = f"Multiply accumulator by {factor}"
            operands = (current, factor)
            expr = f"{current} * {factor}"
        else:  # MOD (circular buffer)
            current = self._rand_int(0, 100)
            modulus = self._rand_int(5, 20)
            result = (current + 1) % modulus
            code = f"index = ({current} + 1) % {modulus}  # next slot {result}"
            explanation = f"Circular buffer: wrap index to {result}"
            operands = (current + 1, modulus)
            expr = f"({current} + 1) % {modulus}"

        return CodeArithmeticSample(
            pattern=CodePattern.ACCUMULATOR,
            code_snippet=code,
            arithmetic_expr=expr,
            operands=operands,
            operation=op_name,
            result=result,
            explanation=explanation,
        )

    def generate_slice_sample(self) -> CodeArithmeticSample:
        """Generate slice operation pattern."""
        start = self._rand_int(0, 30)
        length = self._rand_int(3, 20)
        end = start + length

        templates = [
            f"s[{{start}}:{{end}}]  # end index {end}",
            f"data[{{start}}:{{start}} + {{length}}]  # end {end}",
            f"arr[{{start}}:{{end}}]  # {length} items",
        ]

        template = self.rng.choice(templates)
        code = template.format(start=start, end=end, length=length)

        return CodeArithmeticSample(
            pattern=CodePattern.SLICE_OPERATION,
            code_snippet=code,
            arithmetic_expr=f"{start} + {length}",
            operands=(start, length),
            operation="ADD",
            result=end,
            explanation=f"Slice from {start} to {end} has {length} elements",
        )

    def generate(self, n_samples: int = 1000) -> List[CodeArithmeticSample]:
        """Generate diverse code-embedded arithmetic samples."""
        samples = []
        generators = [
            self.generate_array_index_sample,
            self.generate_loop_bound_sample,
            self.generate_boundary_check_sample,
            self.generate_bit_operation_sample,
            self.generate_accumulator_sample,
            self.generate_slice_sample,
        ]

        # Weights based on frequency in real code
        weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]

        for _ in range(n_samples):
            gen = self.rng.choices(generators, weights=weights)[0]
            samples.append(gen())

        return samples

    def format_for_training(self, sample: CodeArithmeticSample) -> str:
        """Format sample as training string for causal LM."""
        # Format: code snippet → arithmetic result
        # Model learns to predict the result given the code context
        return f"{sample.code_snippet}\nResult: {sample.result}"

    def to_dataset_dict(self, sample: CodeArithmeticSample) -> dict:
        """Convert sample to dict for dataset."""
        return {
            "pattern": sample.pattern.value,
            "code_snippet": sample.code_snippet,
            "arithmetic_expr": sample.arithmetic_expr,
            "operands": sample.operands,
            "operation": sample.operation,
            "result": sample.result,
            "explanation": sample.explanation,
            "training_text": self.format_for_training(sample),
        }


def generate_code_arithmetic_dataset(
    n_samples: int = 10000,
    seed: int = 42,
    max_value: int = 255,
) -> List[dict]:
    """Generate code-embedded arithmetic training dataset.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed
        max_value: Maximum value for operands

    Returns:
        List of sample dicts ready for training
    """
    gen = CodeArithmeticGenerator(seed=seed, max_value=max_value)
    samples = gen.generate(n_samples)
    return [gen.to_dataset_dict(s) for s in samples]


# Example usage and demo
if __name__ == "__main__":
    print("="*70)
    print("CODE-EMBEDDED ARITHMETIC TRAINING DATA")
    print("="*70)

    gen = CodeArithmeticGenerator(seed=42)
    samples = gen.generate(20)

    for i, sample in enumerate(samples):
        print(f"\n[{i+1}] Pattern: {sample.pattern.value}")
        print(f"    Code: {sample.code_snippet}")
        print(f"    Arithmetic: {sample.arithmetic_expr} = {sample.result}")
        print(f"    Explanation: {sample.explanation}")

    print("\n" + "="*70)
    print("TRAINING FORMAT EXAMPLES")
    print("="*70)

    for sample in samples[:5]:
        print(f"\n{gen.format_for_training(sample)}")
