"""Self-growing benchmark for Mog synthesis.

When the system solves a problem, it auto-generates harder variants:
- more test cases (including edge cases)
- larger inputs
- boundary conditions
- negative inputs
- zero inputs

The benchmark grows from use instead of being fixed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from egdc.mog.solvers.program_search import _eval_code_on_examples


@dataclass
class HarderVariant:
    name: str
    examples: list[tuple[tuple[float, ...], float]]
    description: str


class GrowingBenchmark:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.solved: dict[str, dict[str, Any]] = {}

    def register_solved(self, name: str, arg_names: list[str], code: str,
                         examples: list[tuple[tuple[float, ...], float]]):
        self.solved[name] = {
            "arg_names": arg_names,
            "code": code,
            "examples": examples,
        }

    def generate_harder_variants(self, name: str, num_variants: int = 5) -> list[HarderVariant]:
        """Generate harder test variants for a solved problem."""
        if name not in self.solved:
            return []

        info = self.solved[name]
        code = info["code"]
        arg_names = info["arg_names"]
        original_examples = info["examples"]
        num_args = len(arg_names)

        variants = []

        # Variant 1: Edge cases (zeros, ones, negatives)
        edge_inputs = self._generate_edge_cases(num_args)
        edge_examples = self._evaluate_inputs(code, arg_names, edge_inputs)
        if edge_examples:
            variants.append(HarderVariant(
                f"{name}_edge_cases", edge_examples,
                f"Edge case variant of {name} with zeros, ones, and negatives"
            ))

        # Variant 2: Large inputs
        large_inputs = self._generate_large_inputs(num_args)
        large_examples = self._evaluate_inputs(code, arg_names, large_inputs)
        if large_examples:
            variants.append(HarderVariant(
                f"{name}_large", large_examples,
                f"Large input variant of {name}"
            ))

        # Variant 3: Random stress
        for i in range(min(num_variants - 2, 3)):
            rand_inputs = self._generate_random_inputs(num_args, 8)
            rand_examples = self._evaluate_inputs(code, arg_names, rand_inputs)
            if rand_examples:
                variants.append(HarderVariant(
                    f"{name}_stress_{i}", rand_examples,
                    f"Random stress variant {i} of {name}"
                ))

        return variants[:num_variants]

    def _generate_edge_cases(self, num_args: int) -> list[tuple[float, ...]]:
        edges = [0, 1, -1, 2, -2, 100, -100]
        cases = []
        for _ in range(6):
            case = tuple(float(self.rng.choice(edges)) for _ in range(num_args))
            cases.append(case)
        return cases

    def _generate_large_inputs(self, num_args: int) -> list[tuple[float, ...]]:
        cases = []
        for _ in range(4):
            case = tuple(float(self.rng.randint(50, 500)) for _ in range(num_args))
            cases.append(case)
        return cases

    def _generate_random_inputs(self, num_args: int, count: int) -> list[tuple[float, ...]]:
        cases = []
        for _ in range(count):
            case = tuple(float(self.rng.randint(-100, 100)) for _ in range(num_args))
            cases.append(case)
        return cases

    def _evaluate_inputs(self, code: str, arg_names: list[str],
                          inputs: list[tuple[float, ...]]) -> list[tuple[tuple[float, ...], float]]:
        """Run the solved program on new inputs to get expected outputs."""
        from egdc.mog.lang import interpret

        fn_name = code.split("fn ")[1].split("(")[0] if "fn " in code else "program"
        examples = []
        for args in inputs:
            arg_str = ", ".join(str(int(a)) for a in args)
            test = code + f"\nfn main() -> i64 {{ println_i64({fn_name}({arg_str})); return 0; }}"
            result = interpret(test)
            if result.success and result.output.strip():
                try:
                    out = float(result.output.strip().split("\n")[0])
                    examples.append((args, out))
                except ValueError:
                    pass
        return examples
