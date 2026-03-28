"""Training data generation for differentiable execution training.

Generates Python code + test cases suitable for:
1. Code-to-ISA parsing
2. Differentiable execution on nCPU engine
3. Execution loss computation

Three generators:
  ArithmeticFunctionGenerator: Simple arithmetic functions (x*y+z, etc.)
  VariableTrackingGenerator: Multi-step variable assignments
  LoopProblemGenerator: Bounded loops with accumulators

All generators produce ExecutionTrainingSample objects that contain
the code, test cases, and metadata needed for the training loop.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from torch.utils.data import Dataset


@dataclass
class ExecutionTrainingSample:
    """A single training example for execution-grounded training.

    Contains everything needed to:
    1. Prompt the LM with the task
    2. Parse the expected code into nCPU ISA
    3. Execute differentiably and compute loss
    """

    prompt: str  # Task description for the LM
    reference_code: str  # Ground-truth Python code
    test_cases: list[dict]  # [{"inputs": {...}, "expected": {...}}, ...]
    arg_names: list[str]  # Input variable names
    output_var: str  # Output variable name
    category: str  # "arithmetic", "variable_tracking", "loop", etc.
    difficulty: str  # "easy", "medium", "hard"
    is_function: bool = False  # Whether reference_code is a function def
    metadata: dict = field(default_factory=dict)


class ArithmeticFunctionGenerator:
    """Generates simple arithmetic function problems.

    Examples:
        "Write a function that computes x * y + z"
        "Write a function that returns (a + b) * (a - b)"
    """

    # Templates: (description, code_template, arg_names, output_expr)
    TEMPLATES = [
        # Single-op
        ("compute {a} + {b}", "result = {a} + {b}", ["a", "b"], lambda a, b: a + b),
        ("compute {a} - {b}", "result = {a} - {b}", ["a", "b"], lambda a, b: a - b),
        ("compute {a} * {b}", "result = {a} * {b}", ["a", "b"], lambda a, b: a * b),
        ("compute {a} & {b}", "result = {a} & {b}", ["a", "b"], lambda a, b: a & b),
        ("compute {a} | {b}", "result = {a} | {b}", ["a", "b"], lambda a, b: a | b),
        ("compute {a} ^ {b}", "result = {a} ^ {b}", ["a", "b"], lambda a, b: a ^ b),
        # Two-op
        (
            "compute {a} * {b} + {c}",
            "result = {a} * {b} + {c}",
            ["a", "b", "c"],
            lambda a, b, c: a * b + c,
        ),
        (
            "compute ({a} + {b}) * {c}",
            "result = ({a} + {b}) * {c}",
            ["a", "b", "c"],
            lambda a, b, c: (a + b) * c,
        ),
        (
            "compute {a} * {b} - {c}",
            "result = {a} * {b} - {c}",
            ["a", "b", "c"],
            lambda a, b, c: a * b - c,
        ),
        (
            "compute ({a} - {b}) * {c}",
            "result = ({a} - {b}) * {c}",
            ["a", "b", "c"],
            lambda a, b, c: (a - b) * c,
        ),
        (
            "compute {a} + {b} + {c}",
            "result = {a} + {b} + {c}",
            ["a", "b", "c"],
            lambda a, b, c: a + b + c,
        ),
        # Three-op
        (
            "compute {a} * {b} + {c} * {d}",
            "result = {a} * {b} + {c} * {d}",
            ["a", "b", "c", "d"],
            lambda a, b, c, d: a * b + c * d,
        ),
        (
            "compute ({a} + {b}) * ({c} - {d})",
            "result = ({a} + {b}) * ({c} - {d})",
            ["a", "b", "c", "d"],
            lambda a, b, c, d: (a + b) * (c - d),
        ),
        # Augmented assignment patterns
        (
            "start with {a}, add {b}, then multiply by {c}",
            "result = {a}\nresult = result + {b}\nresult = result * {c}",
            ["a", "b", "c"],
            lambda a, b, c: (a + b) * c,
        ),
        (
            "start with {a}, multiply by {b}, then subtract {c}",
            "result = {a}\nresult = result * {b}\nresult = result - {c}",
            ["a", "b", "c"],
            lambda a, b, c: a * b - c,
        ),
    ]

    def __init__(self, seed: int = 42, max_value: int = 100):
        self.rng = random.Random(seed)
        self.max_value = max_value

    def generate(self, n_samples: int = 1000) -> list[ExecutionTrainingSample]:
        """Generate n arithmetic function training samples."""
        samples = []
        for _ in range(n_samples):
            template = self.rng.choice(self.TEMPLATES)
            desc_template, code_template, arg_names, compute_fn = template
            n_args = len(arg_names)

            # Generate test cases
            test_cases = []
            for _ in range(5):  # 5 test cases per problem
                vals = {
                    name: self.rng.randint(1, self.max_value) for name in arg_names
                }
                try:
                    output = compute_fn(*[vals[n] for n in arg_names])
                    # Skip if output too large for differentiable engine
                    if abs(output) > 1e6:
                        continue
                    test_cases.append(
                        {
                            "inputs": vals,
                            "expected": {"result": output},
                        }
                    )
                except (OverflowError, ZeroDivisionError):
                    continue

            if len(test_cases) < 3:
                continue

            # Build the code with actual variable names
            code = code_template.format(**{n: n for n in arg_names})
            desc = desc_template.format(**{n: n for n in arg_names})

            difficulty = "easy" if n_args <= 2 else "medium" if n_args <= 3 else "hard"

            samples.append(
                ExecutionTrainingSample(
                    prompt=f"Write Python code to {desc}",
                    reference_code=code,
                    test_cases=test_cases,
                    arg_names=arg_names,
                    output_var="result",
                    category="arithmetic",
                    difficulty=difficulty,
                    metadata={"template": desc_template, "n_ops": n_args - 1},
                )
            )

        return samples


class VariableTrackingGenerator:
    """Generates multi-step variable tracking problems.

    These test whether the model can follow sequential state changes:
        x = 5
        x = x + 3     (now x = 8)
        y = x * 2     (now y = 16)
        z = y - x     (now z = 8)
    """

    # Operations that can be applied in sequence
    OPS = [
        ("add_const", lambda v, c: v + c, "{var} = {var} + {const}"),
        ("sub_const", lambda v, c: v - c, "{var} = {var} - {const}"),
        ("mul_const", lambda v, c: v * c, "{var} = {var} * {const}"),
        ("add_var", lambda v, o: v + o, "{var} = {var} + {other}"),
        ("sub_var", lambda v, o: v - o, "{var} = {var} - {other}"),
        ("mul_var", lambda v, o: v * o, "{var} = {var} * {other}"),
    ]

    def __init__(self, seed: int = 42, max_value: int = 50):
        self.rng = random.Random(seed)
        self.max_value = max_value

    def generate(self, n_samples: int = 1000) -> list[ExecutionTrainingSample]:
        """Generate variable tracking samples."""
        samples = []

        for _ in range(n_samples):
            n_steps = self.rng.randint(2, 6)
            n_vars = self.rng.randint(1, min(3, n_steps))

            var_names = [chr(ord("x") + i) for i in range(n_vars)]
            code_lines = []
            state = {}

            # Initialize variables
            for v in var_names:
                val = self.rng.randint(1, self.max_value)
                code_lines.append(f"{v} = {val}")
                state[v] = val

            # Apply random operations
            trace_points = []
            for step_i in range(n_steps):
                target_var = self.rng.choice(var_names)

                if self.rng.random() < 0.5 or n_vars == 1:
                    # Const operation
                    const = self.rng.randint(1, 10)
                    op_name, op_fn, op_template = self.rng.choice(self.OPS[:3])
                    try:
                        new_val = op_fn(state[target_var], const)
                        if abs(new_val) > 1e5:
                            continue
                        state[target_var] = new_val
                        code_lines.append(
                            op_template.format(var=target_var, const=const)
                        )
                    except (OverflowError, ZeroDivisionError):
                        continue
                else:
                    # Var-var operation
                    other_var = self.rng.choice(
                        [v for v in var_names if v != target_var] or var_names
                    )
                    op_name, op_fn, op_template = self.rng.choice(self.OPS[3:])
                    try:
                        new_val = op_fn(state[target_var], state[other_var])
                        if abs(new_val) > 1e5:
                            continue
                        state[target_var] = new_val
                        code_lines.append(
                            op_template.format(var=target_var, other=other_var)
                        )
                    except (OverflowError, ZeroDivisionError):
                        continue

                # Record trace point
                trace_points.append(dict(state))

            code = "\n".join(code_lines)
            last_var = var_names[-1]

            # Build test cases (variable tracking → one "test case" with no
            # inputs, just expected final state)
            test_cases = [
                {
                    "inputs": {},
                    "expected": {v: state[v] for v in var_names},
                }
            ]

            # Determine what question to ask
            question_var = self.rng.choice(var_names)
            prompt = (
                f"After running this code, what is the value of {question_var}?\n"
                f"```python\n{code}\n```"
            )

            difficulty = "easy" if n_steps <= 3 else "medium" if n_steps <= 5 else "hard"

            samples.append(
                ExecutionTrainingSample(
                    prompt=prompt,
                    reference_code=code,
                    test_cases=test_cases,
                    arg_names=[],
                    output_var=question_var,
                    category="variable_tracking",
                    difficulty=difficulty,
                    metadata={
                        "n_steps": n_steps,
                        "n_vars": n_vars,
                        "trace": trace_points,
                        "final_state": dict(state),
                    },
                )
            )

        return samples


class LoopProblemGenerator:
    """Generates bounded loop problems with accumulators.

    These produce for-loop code that nCPU unrolls:
        total = 0
        for i in range(5):
            total = total + i
    """

    PATTERNS = [
        # (description, code_template, compute_fn, difficulty)
        (
            "sum of integers from 0 to n-1",
            "total = 0\nfor i in range({n}):\n    total = total + i",
            lambda n: sum(range(n)),
            "easy",
        ),
        (
            "sum of squares from 0 to n-1",
            "total = 0\nfor i in range({n}):\n    total = total + i * i",
            lambda n: sum(i * i for i in range(n)),
            "medium",
        ),
        (
            "factorial of n",
            "result = 1\nfor i in range(1, {n_plus1}):\n    result = result * i",
            lambda n: 1 if n == 0 else eval("__import__('math').factorial(n)", {"n": n}),
            "medium",
        ),
        (
            "count even numbers from 0 to n-1",
            "count = 0\nfor i in range({n}):\n    count = count + 1",
            # Simplified: just counts iterations (even-check requires conditionals)
            lambda n: n,
            "easy",
        ),
        (
            "product of 1 to n",
            "result = 1\nfor i in range(1, {n_plus1}):\n    result = result * i",
            lambda n: 1 if n == 0 else eval("__import__('math').factorial(n)", {"n": n}),
            "medium",
        ),
        (
            "running sum: sum(1..n)",
            "total = 0\nfor i in range(1, {n_plus1}):\n    total = total + i",
            lambda n: n * (n + 1) // 2,
            "easy",
        ),
    ]

    def __init__(self, seed: int = 42, max_n: int = 10):
        self.rng = random.Random(seed)
        self.max_n = max_n

    def generate(self, n_samples: int = 500) -> list[ExecutionTrainingSample]:
        """Generate loop problem samples."""
        samples = []

        for _ in range(n_samples):
            pattern = self.rng.choice(self.PATTERNS)
            desc, code_template, compute_fn, difficulty = pattern

            # Pick a small n (loops get unrolled, so keep it manageable)
            n = self.rng.randint(2, min(self.max_n, 10))

            try:
                expected_output = compute_fn(n)
                if abs(expected_output) > 1e6:
                    continue
            except (OverflowError, ValueError):
                continue

            code = code_template.format(n=n, n_plus1=n + 1)

            # Determine output variable
            output_var = "total" if "total" in code else "result" if "result" in code else "count"

            test_cases = [
                {
                    "inputs": {},
                    "expected": {output_var: expected_output},
                }
            ]

            # Also generate a few more test cases with different n
            for _ in range(3):
                n2 = self.rng.randint(2, min(self.max_n, 10))
                try:
                    out2 = compute_fn(n2)
                    if abs(out2) > 1e6:
                        continue
                    code2 = code_template.format(n=n2, n_plus1=n2 + 1)
                    test_cases.append(
                        {
                            "inputs": {},
                            "expected": {output_var: out2},
                            "_code_variant": code2,  # Different code for this test
                        }
                    )
                except (OverflowError, ValueError):
                    continue

            prompt = f"Write Python code to compute the {desc} (n={n})"

            samples.append(
                ExecutionTrainingSample(
                    prompt=prompt,
                    reference_code=code,
                    test_cases=test_cases[:1],  # Only use the primary test case
                    arg_names=[],
                    output_var=output_var,
                    category="loop",
                    difficulty=difficulty,
                    metadata={"n": n, "pattern": desc},
                )
            )

        return samples


class ExecutionTrainingDataset(Dataset):
    """PyTorch Dataset combining all generators.

    Usage:
        dataset = ExecutionTrainingDataset(size=5000, seed=42)
        sample = dataset[0]
        # sample is an ExecutionTrainingSample

        # With tokenizer for LM training:
        dataset = ExecutionTrainingDataset(size=5000, tokenizer=tokenizer)
        item = dataset[0]
        # item has 'input_ids', 'labels', 'sample' keys
    """

    def __init__(
        self,
        size: int = 5000,
        seed: int = 42,
        max_value: int = 100,
        max_loop_n: int = 10,
        tokenizer=None,
        max_length: int = 256,
        category_weights: Optional[dict[str, float]] = None,
    ):
        """
        Args:
            size: Total number of samples
            seed: Random seed
            max_value: Maximum integer value in arithmetic problems
            max_loop_n: Maximum loop iteration count
            tokenizer: Optional HF tokenizer for encoding
            max_length: Max token length when tokenizer is provided
            category_weights: Relative weights for categories
                             Default: {"arithmetic": 0.5, "variable_tracking": 0.3, "loop": 0.2}
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        weights = category_weights or {
            "arithmetic": 0.5,
            "variable_tracking": 0.3,
            "loop": 0.2,
        }
        total_w = sum(weights.values())

        n_arith = int(size * weights.get("arithmetic", 0) / total_w)
        n_track = int(size * weights.get("variable_tracking", 0) / total_w)
        n_loop = size - n_arith - n_track

        arith_gen = ArithmeticFunctionGenerator(seed=seed, max_value=max_value)
        track_gen = VariableTrackingGenerator(seed=seed + 1, max_value=max_value)
        loop_gen = LoopProblemGenerator(seed=seed + 2, max_n=max_loop_n)

        self.samples = (
            arith_gen.generate(n_arith)
            + track_gen.generate(n_track)
            + loop_gen.generate(n_loop)
        )

        # Shuffle deterministically
        rng = random.Random(seed + 3)
        rng.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        if self.tokenizer is None:
            return sample

        # Format for LM training: prompt + response
        text = f"{sample.prompt}\n\n```python\n{sample.reference_code}\n```"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # For causal LM: labels = input_ids (shifted internally by the model)
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Mask prompt tokens in labels (only supervise the code part)
        prompt_text = f"{sample.prompt}\n\n```python\n"
        prompt_len = len(
            self.tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
        )

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # Mask prompt in loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sample": sample,
        }
