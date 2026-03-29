"""Arithmetic-focused code generation benchmark.

100 curated problems that test whether a model can write correct
arithmetic/logic code — the exact operations nCPU can execute
differentiably. This measures what execution training should improve.

Usage:
    from ncpu.execution_training.arithmetic_bench import ArithmeticBench
    bench = ArithmeticBench()
    result = bench.run(model, tokenizer)
    print(result.summary())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from .data import ExecutionTrainingSample
from .evaluate import ExecutionEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class BenchProblem:
    """A single benchmark problem."""
    id: str
    prompt: str
    reference_code: str
    test_cases: list[dict]
    arg_names: list[str]
    output_var: str
    difficulty: str  # easy, medium, hard
    category: str  # single_op, multi_op, variable_tracking, accumulation, bitwise, conditional

    def to_sample(self) -> ExecutionTrainingSample:
        return ExecutionTrainingSample(
            prompt=self.prompt,
            reference_code=self.reference_code,
            test_cases=self.test_cases,
            arg_names=self.arg_names,
            output_var=self.output_var,
            category=self.category,
            difficulty=self.difficulty,
        )


@dataclass
class BenchResult:
    """Benchmark results."""
    total: int = 0
    correct: int = 0
    parseable: int = 0
    exec_correct: int = 0
    by_difficulty: dict[str, dict] = field(default_factory=dict)
    by_category: dict[str, dict] = field(default_factory=dict)
    eval_result: Optional[EvaluationResult] = None
    wall_time: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 1)

    @property
    def parse_rate(self) -> float:
        return self.parseable / max(self.total, 1)

    @property
    def exec_accuracy(self) -> float:
        return self.exec_correct / max(self.parseable, 1)

    def summary(self) -> str:
        lines = [
            "=" * 65,
            "ARITHMETIC BENCHMARK RESULTS",
            "=" * 65,
            f"Total: {self.correct}/{self.total} correct ({self.accuracy:.1%})",
            f"Parse rate: {self.parseable}/{self.total} ({self.parse_rate:.1%})",
            f"Exec correct: {self.exec_correct}/{self.parseable} ({self.exec_accuracy:.1%})",
            f"Time: {self.wall_time:.1f}s",
            "",
            "By difficulty:",
        ]
        for diff in ["easy", "medium", "hard"]:
            d = self.by_difficulty.get(diff, {})
            n = d.get("total", 0)
            c = d.get("correct", 0)
            lines.append(f"  {diff:8s}: {c}/{n} ({c/max(n,1):.1%})")

        lines.append("\nBy category:")
        for cat, d in sorted(self.by_category.items()):
            n = d.get("total", 0)
            c = d.get("correct", 0)
            lines.append(f"  {cat:22s}: {c}/{n} ({c/max(n,1):.1%})")

        lines.append("=" * 65)
        return "\n".join(lines)


def _tc(inputs: dict, output: int, output_var: str = "result") -> dict:
    """Helper to build a test case."""
    return {"inputs": inputs, "expected": {output_var: output}}


# ══════════════════════════════════════════════════════════════════
# BENCHMARK PROBLEMS
# ══════════════════════════════════════════════════════════════════

PROBLEMS: list[BenchProblem] = []

# ── EASY: Single operations (40 problems) ──

_easy_single = [
    ("e01", "Write code that computes a + b", "result = a + b", ["a", "b"],
     [_tc({"a": 3, "b": 5}, 8), _tc({"a": 10, "b": 20}, 30), _tc({"a": 0, "b": 7}, 7),
      _tc({"a": 100, "b": 1}, 101), _tc({"a": 42, "b": 58}, 100)]),
    ("e02", "Write code that computes a - b", "result = a - b", ["a", "b"],
     [_tc({"a": 10, "b": 3}, 7), _tc({"a": 20, "b": 5}, 15), _tc({"a": 100, "b": 100}, 0),
      _tc({"a": 50, "b": 25}, 25), _tc({"a": 7, "b": 2}, 5)]),
    ("e03", "Write code that computes a * b", "result = a * b", ["a", "b"],
     [_tc({"a": 3, "b": 7}, 21), _tc({"a": 5, "b": 5}, 25), _tc({"a": 10, "b": 0}, 0),
      _tc({"a": 6, "b": 8}, 48), _tc({"a": 12, "b": 4}, 48)]),
    ("e04", "Write code that computes a & b (bitwise AND)", "result = a & b", ["a", "b"],
     [_tc({"a": 0xFF, "b": 0x0F}, 0x0F), _tc({"a": 12, "b": 10}, 8),
      _tc({"a": 7, "b": 3}, 3), _tc({"a": 15, "b": 15}, 15), _tc({"a": 8, "b": 7}, 0)]),
    ("e05", "Write code that computes a | b (bitwise OR)", "result = a | b", ["a", "b"],
     [_tc({"a": 0xF0, "b": 0x0F}, 0xFF), _tc({"a": 12, "b": 3}, 15),
      _tc({"a": 0, "b": 5}, 5), _tc({"a": 8, "b": 4}, 12), _tc({"a": 1, "b": 2}, 3)]),
    ("e06", "Write code that computes a ^ b (bitwise XOR)", "result = a ^ b", ["a", "b"],
     [_tc({"a": 0xFF, "b": 0x0F}, 0xF0), _tc({"a": 5, "b": 3}, 6),
      _tc({"a": 10, "b": 10}, 0), _tc({"a": 7, "b": 0}, 7), _tc({"a": 12, "b": 5}, 9)]),
    ("e07", "Set x to 5 then add 3 to it", "x = 5\nx = x + 3\nresult = x", [],
     [_tc({}, 8, "result")]),
    ("e08", "Set x to 10 then multiply by 2", "x = 10\nx = x * 2\nresult = x", [],
     [_tc({}, 20, "result")]),
    ("e09", "Set x to 100 then subtract 37", "x = 100\nx = x - 37\nresult = x", [],
     [_tc({}, 63, "result")]),
    ("e10", "What is 7 * 8 + 3?", "result = 7 * 8 + 3", [],
     [_tc({}, 59, "result")]),
    ("e11", "What is 15 - 6 * 2?", "result = 15 - 6 * 2", [],
     [_tc({}, 3, "result")]),
    ("e12", "What is 100 + 50 - 25?", "result = 100 + 50 - 25", [],
     [_tc({}, 125, "result")]),
    ("e13", "Double the value of a", "result = a * 2", ["a"],
     [_tc({"a": 5}, 10), _tc({"a": 0}, 0), _tc({"a": 50}, 100), _tc({"a": 13}, 26), _tc({"a": 1}, 2)]),
    ("e14", "Triple the value of a then subtract b", "result = a * 3 - b", ["a", "b"],
     [_tc({"a": 5, "b": 3}, 12), _tc({"a": 10, "b": 10}, 20), _tc({"a": 4, "b": 1}, 11),
      _tc({"a": 7, "b": 0}, 21), _tc({"a": 3, "b": 9}, 0)]),
    ("e15", "Compute a + b + c", "result = a + b + c", ["a", "b", "c"],
     [_tc({"a": 1, "b": 2, "c": 3}, 6), _tc({"a": 10, "b": 20, "c": 30}, 60),
      _tc({"a": 0, "b": 0, "c": 0}, 0), _tc({"a": 5, "b": 5, "c": 5}, 15), _tc({"a": 100, "b": 1, "c": 1}, 102)]),
    ("e16", "Compute a * b + 1", "result = a * b + 1", ["a", "b"],
     [_tc({"a": 3, "b": 4}, 13), _tc({"a": 0, "b": 5}, 1), _tc({"a": 7, "b": 7}, 50),
      _tc({"a": 1, "b": 1}, 2), _tc({"a": 10, "b": 3}, 31)]),
    ("e17", "Compute a - b + c", "result = a - b + c", ["a", "b", "c"],
     [_tc({"a": 10, "b": 3, "c": 5}, 12), _tc({"a": 20, "b": 10, "c": 10}, 20),
      _tc({"a": 5, "b": 5, "c": 5}, 5), _tc({"a": 100, "b": 50, "c": 25}, 75), _tc({"a": 1, "b": 1, "c": 1}, 1)]),
    ("e18", "Add 10 to a", "result = a + 10", ["a"],
     [_tc({"a": 0}, 10), _tc({"a": 5}, 15), _tc({"a": 90}, 100), _tc({"a": 25}, 35), _tc({"a": 1}, 11)]),
    ("e19", "Subtract 1 from a", "result = a - 1", ["a"],
     [_tc({"a": 10}, 9), _tc({"a": 1}, 0), _tc({"a": 100}, 99), _tc({"a": 50}, 49), _tc({"a": 2}, 1)]),
    ("e20", "Multiply a by itself (square)", "result = a * a", ["a"],
     [_tc({"a": 3}, 9), _tc({"a": 5}, 25), _tc({"a": 0}, 0), _tc({"a": 7}, 49), _tc({"a": 10}, 100)]),
]

for pid, prompt, code, args, tcs in _easy_single[:20]:
    PROBLEMS.append(BenchProblem(
        id=pid, prompt=prompt, reference_code=code,
        test_cases=tcs, arg_names=args, output_var="result",
        difficulty="easy", category="single_op",
    ))

_easy_const = [
    ("e21", "What is 2 + 2?", "result = 2 + 2", 4),
    ("e22", "What is 3 * 3?", "result = 3 * 3", 9),
    ("e23", "What is 50 - 17?", "result = 50 - 17", 33),
    ("e24", "What is 6 * 7?", "result = 6 * 7", 42),
    ("e25", "What is 8 + 9?", "result = 8 + 9", 17),
    ("e26", "What is 12 * 5?", "result = 12 * 5", 60),
    ("e27", "What is 100 - 1?", "result = 100 - 1", 99),
    ("e28", "What is 11 * 11?", "result = 11 * 11", 121),
    ("e29", "What is 25 + 75?", "result = 25 + 75", 100),
    ("e30", "What is 64 - 32?", "result = 64 - 32", 32),
    ("e31", "What is 2 * 3 * 4?", "result = 2 * 3 * 4", 24),
    ("e32", "What is 10 + 20 + 30?", "result = 10 + 20 + 30", 60),
    ("e33", "What is 5 * 5 + 1?", "result = 5 * 5 + 1", 26),
    ("e34", "What is 9 * 9 - 1?", "result = 9 * 9 - 1", 80),
    ("e35", "What is 7 * 6 + 2?", "result = 7 * 6 + 2", 44),
    ("e36", "What is 10 * 10 - 10?", "result = 10 * 10 - 10", 90),
    ("e37", "What is 4 * 8 + 3?", "result = 4 * 8 + 3", 35),
    ("e38", "What is 15 * 3 - 5?", "result = 15 * 3 - 5", 40),
    ("e39", "What is 20 - 7 + 3?", "result = 20 - 7 + 3", 16),
    ("e40", "What is 8 * 8?", "result = 8 * 8", 64),
]

for pid, prompt, code, expected in _easy_const:
    PROBLEMS.append(BenchProblem(
        id=pid, prompt=prompt, reference_code=code,
        test_cases=[_tc({}, expected, "result")],
        arg_names=[], output_var="result",
        difficulty="easy", category="constant_eval",
    ))

# ── MEDIUM: Multi-step and variable tracking (35 problems) ──

_medium = [
    ("m01", "Compute (a + b) * (a - b)", "result = (a + b) * (a - b)", ["a", "b"],
     [_tc({"a": 5, "b": 3}, 16), _tc({"a": 10, "b": 2}, 96), _tc({"a": 7, "b": 7}, 0),
      _tc({"a": 20, "b": 5}, 375), _tc({"a": 4, "b": 1}, 15)], "multi_op"),
    ("m02", "Compute a * b + c * d", "result = a * b + c * d", ["a", "b", "c", "d"],
     [_tc({"a": 2, "b": 3, "c": 4, "d": 5}, 26), _tc({"a": 1, "b": 1, "c": 1, "d": 1}, 2),
      _tc({"a": 5, "b": 5, "c": 5, "d": 5}, 50), _tc({"a": 3, "b": 7, "c": 2, "d": 4}, 29),
      _tc({"a": 10, "b": 2, "c": 3, "d": 4}, 32)], "multi_op"),
    ("m03", "Compute a * a + b * b", "result = a * a + b * b", ["a", "b"],
     [_tc({"a": 3, "b": 4}, 25), _tc({"a": 5, "b": 12}, 169), _tc({"a": 0, "b": 0}, 0),
      _tc({"a": 1, "b": 1}, 2), _tc({"a": 6, "b": 8}, 100)], "multi_op"),
    ("m04", "x = 10, y = x * 2, z = y - x. What is z?",
     "x = 10\ny = x * 2\nz = y - x\nresult = z", [],
     [_tc({}, 10, "result")], "variable_tracking"),
    ("m05", "x = a, x = x + 3, x = x * 2. What is x?",
     "x = a\nx = x + 3\nx = x * 2\nresult = x", ["a"],
     [_tc({"a": 5}, 16), _tc({"a": 0}, 6), _tc({"a": 10}, 26),
      _tc({"a": 1}, 8), _tc({"a": 7}, 20)], "variable_tracking"),
    ("m06", "x = a, y = b, temp = x, x = y, y = temp. What is x?",
     "x = a\ny = b\ntemp = x\nx = y\ny = temp\nresult = x", ["a", "b"],
     [_tc({"a": 3, "b": 7}, 7), _tc({"a": 10, "b": 20}, 20),
      _tc({"a": 1, "b": 2}, 2), _tc({"a": 50, "b": 25}, 25), _tc({"a": 0, "b": 99}, 99)], "variable_tracking"),
    ("m07", "Compute sum of 1 to 5",
     "total = 0\nfor i in range(1, 6):\n    total = total + i\nresult = total", [],
     [_tc({}, 15, "result")], "accumulation"),
    ("m08", "Compute sum of 1 to 10",
     "total = 0\nfor i in range(1, 11):\n    total = total + i\nresult = total", [],
     [_tc({}, 55, "result")], "accumulation"),
    ("m09", "Compute factorial of 5",
     "result = 1\nfor i in range(1, 6):\n    result = result * i", [],
     [_tc({}, 120, "result")], "accumulation"),
    ("m10", "Compute 2 raised to the power of 5 using a loop",
     "result = 1\nfor i in range(5):\n    result = result * 2", [],
     [_tc({}, 32, "result")], "accumulation"),
    ("m11", "Compute (a + b) * c - d", "result = (a + b) * c - d", ["a", "b", "c", "d"],
     [_tc({"a": 2, "b": 3, "c": 4, "d": 5}, 15), _tc({"a": 5, "b": 5, "c": 2, "d": 10}, 10),
      _tc({"a": 1, "b": 1, "c": 10, "d": 0}, 20), _tc({"a": 3, "b": 7, "c": 3, "d": 30}, 0),
      _tc({"a": 10, "b": 10, "c": 5, "d": 50}, 50)], "multi_op"),
    ("m12", "Compute a * (b + c)", "result = a * (b + c)", ["a", "b", "c"],
     [_tc({"a": 3, "b": 4, "c": 5}, 27), _tc({"a": 2, "b": 10, "c": 10}, 40),
      _tc({"a": 5, "b": 3, "c": 7}, 50), _tc({"a": 1, "b": 1, "c": 1}, 2),
      _tc({"a": 10, "b": 0, "c": 5}, 50)], "multi_op"),
    ("m13", "x = a + b, y = a - b, result = x * y",
     "x = a + b\ny = a - b\nresult = x * y", ["a", "b"],
     [_tc({"a": 5, "b": 3}, 16), _tc({"a": 10, "b": 2}, 96),
      _tc({"a": 7, "b": 4}, 33), _tc({"a": 20, "b": 10}, 300), _tc({"a": 8, "b": 1}, 63)], "variable_tracking"),
    ("m14", "x = a, x += b, x += c, x += d",
     "x = a\nx = x + b\nx = x + c\nx = x + d\nresult = x", ["a", "b", "c", "d"],
     [_tc({"a": 1, "b": 2, "c": 3, "d": 4}, 10), _tc({"a": 10, "b": 10, "c": 10, "d": 10}, 40),
      _tc({"a": 5, "b": 0, "c": 0, "d": 0}, 5), _tc({"a": 0, "b": 0, "c": 0, "d": 0}, 0),
      _tc({"a": 25, "b": 25, "c": 25, "d": 25}, 100)], "variable_tracking"),
    ("m15", "Compute a*a - 2*a*b + b*b (i.e., (a-b)^2)",
     "result = a * a - 2 * a * b + b * b", ["a", "b"],
     [_tc({"a": 5, "b": 3}, 4), _tc({"a": 10, "b": 7}, 9), _tc({"a": 3, "b": 3}, 0),
      _tc({"a": 8, "b": 2}, 36), _tc({"a": 1, "b": 0}, 1)], "multi_op"),
]

# Pad to 35 medium problems with more variable tracking
_medium_extra = [
    ("m16", "x=a, y=x+1, z=y+1, w=z+1. What is w?",
     "x = a\ny = x + 1\nz = y + 1\nw = z + 1\nresult = w", ["a"],
     [_tc({"a": 0}, 3), _tc({"a": 5}, 8), _tc({"a": 10}, 13), _tc({"a": 97}, 100), _tc({"a": 1}, 4)], "variable_tracking"),
    ("m17", "Compute a*b - c", "result = a * b - c", ["a", "b", "c"],
     [_tc({"a": 3, "b": 4, "c": 2}, 10), _tc({"a": 5, "b": 5, "c": 25}, 0),
      _tc({"a": 10, "b": 3, "c": 5}, 25), _tc({"a": 7, "b": 7, "c": 49}, 0),
      _tc({"a": 2, "b": 6, "c": 1}, 11)], "multi_op"),
    ("m18", "Compute (a - b) * (c - d)", "result = (a - b) * (c - d)", ["a", "b", "c", "d"],
     [_tc({"a": 10, "b": 3, "c": 8, "d": 2}, 42), _tc({"a": 5, "b": 5, "c": 5, "d": 5}, 0),
      _tc({"a": 20, "b": 10, "c": 15, "d": 5}, 100), _tc({"a": 7, "b": 1, "c": 4, "d": 1}, 18),
      _tc({"a": 3, "b": 1, "c": 3, "d": 1}, 4)], "multi_op"),
    ("m19", "Sum of first n even numbers (2+4+...+2n) for n=5",
     "total = 0\nfor i in range(1, 6):\n    total = total + i * 2\nresult = total", [],
     [_tc({}, 30, "result")], "accumulation"),
    ("m20", "x=a, x=x*x, x=x+1. What is x?",
     "x = a\nx = x * x\nx = x + 1\nresult = x", ["a"],
     [_tc({"a": 3}, 10), _tc({"a": 5}, 26), _tc({"a": 2}, 5), _tc({"a": 0}, 1), _tc({"a": 4}, 17)], "variable_tracking"),
    ("m21", "Compute a + 2*b + 3*c", "result = a + 2 * b + 3 * c", ["a", "b", "c"],
     [_tc({"a": 1, "b": 2, "c": 3}, 14), _tc({"a": 10, "b": 5, "c": 1}, 23),
      _tc({"a": 0, "b": 0, "c": 0}, 0), _tc({"a": 5, "b": 5, "c": 5}, 30),
      _tc({"a": 3, "b": 7, "c": 2}, 23)], "multi_op"),
    ("m22", "Compute 0xAA ^ 0x55", "result = 0xAA ^ 0x55", [],
     [_tc({}, 0xFF, "result")], "bitwise"),
    ("m23", "Compute (a | b) & c", "result = (a | b) & c", ["a", "b", "c"],
     [_tc({"a": 0xF0, "b": 0x0F, "c": 0xFF}, 0xFF), _tc({"a": 12, "b": 3, "c": 7}, 7),
      _tc({"a": 8, "b": 4, "c": 15}, 12), _tc({"a": 0, "b": 0, "c": 255}, 0),
      _tc({"a": 255, "b": 0, "c": 128}, 128)], "bitwise"),
    ("m24", "Compute a ^ b ^ c", "result = a ^ b ^ c", ["a", "b", "c"],
     [_tc({"a": 5, "b": 3, "c": 6}, 0), _tc({"a": 0xFF, "b": 0x0F, "c": 0xF0}, 0),
      _tc({"a": 1, "b": 2, "c": 3}, 0), _tc({"a": 10, "b": 10, "c": 10}, 10),
      _tc({"a": 7, "b": 0, "c": 0}, 7)], "bitwise"),
    ("m25", "x=a*b, y=c*d, result=x+y",
     "x = a * b\ny = c * d\nresult = x + y", ["a", "b", "c", "d"],
     [_tc({"a": 2, "b": 3, "c": 4, "d": 5}, 26), _tc({"a": 1, "b": 10, "c": 2, "d": 5}, 20),
      _tc({"a": 5, "b": 5, "c": 3, "d": 3}, 34), _tc({"a": 7, "b": 3, "c": 2, "d": 8}, 37),
      _tc({"a": 0, "b": 0, "c": 0, "d": 0}, 0)], "variable_tracking"),
    ("m26", "Product of 1 to 4",
     "result = 1\nfor i in range(1, 5):\n    result = result * i", [],
     [_tc({}, 24, "result")], "accumulation"),
    ("m27", "Sum of squares 1 to 4",
     "total = 0\nfor i in range(1, 5):\n    total = total + i * i\nresult = total", [],
     [_tc({}, 30, "result")], "accumulation"),
    ("m28", "Compute a*a + 2*a + 1 (i.e., (a+1)^2)",
     "result = a * a + 2 * a + 1", ["a"],
     [_tc({"a": 0}, 1), _tc({"a": 3}, 16), _tc({"a": 5}, 36), _tc({"a": 9}, 100), _tc({"a": 1}, 4)], "multi_op"),
    ("m29", "x=a+b, y=a*b, result=x+y",
     "x = a + b\ny = a * b\nresult = x + y", ["a", "b"],
     [_tc({"a": 2, "b": 3}, 11), _tc({"a": 5, "b": 5}, 35), _tc({"a": 1, "b": 1}, 3),
      _tc({"a": 10, "b": 2}, 32), _tc({"a": 4, "b": 3}, 19)], "variable_tracking"),
    ("m30", "Compute a*b*c", "result = a * b * c", ["a", "b", "c"],
     [_tc({"a": 2, "b": 3, "c": 4}, 24), _tc({"a": 1, "b": 1, "c": 1}, 1),
      _tc({"a": 5, "b": 5, "c": 5}, 125), _tc({"a": 10, "b": 2, "c": 3}, 60),
      _tc({"a": 3, "b": 3, "c": 3}, 27)], "multi_op"),
    ("m31", "Count down from 10 by 2 (sum: 10+8+6+4+2)",
     "total = 0\nfor i in range(5):\n    total = total + 10 - i * 2\nresult = total", [],
     [_tc({}, 30, "result")], "accumulation"),
    ("m32", "x=1, double x five times",
     "x = 1\nfor i in range(5):\n    x = x * 2\nresult = x", [],
     [_tc({}, 32, "result")], "accumulation"),
    ("m33", "Compute (a+1)*(b+1)-(a*b)", "result = (a + 1) * (b + 1) - a * b", ["a", "b"],
     [_tc({"a": 3, "b": 4}, 8), _tc({"a": 0, "b": 0}, 1), _tc({"a": 10, "b": 10}, 21),
      _tc({"a": 5, "b": 7}, 13), _tc({"a": 1, "b": 1}, 3)], "multi_op"),
    ("m34", "x=a, y=b, z=x+y, w=x*y, result=z+w",
     "x = a\ny = b\nz = x + y\nw = x * y\nresult = z + w", ["a", "b"],
     [_tc({"a": 2, "b": 3}, 11), _tc({"a": 5, "b": 5}, 35), _tc({"a": 1, "b": 10}, 21),
      _tc({"a": 4, "b": 3}, 19), _tc({"a": 0, "b": 7}, 7)], "variable_tracking"),
    ("m35", "Compute 3*a*a + 2*a + 1", "result = 3 * a * a + 2 * a + 1", ["a"],
     [_tc({"a": 0}, 1), _tc({"a": 1}, 6), _tc({"a": 2}, 17), _tc({"a": 3}, 34), _tc({"a": 5}, 86)], "multi_op"),
]

for pid, prompt, code, args, tcs, cat in _medium + _medium_extra:
    PROBLEMS.append(BenchProblem(
        id=pid, prompt=prompt, reference_code=code,
        test_cases=tcs, arg_names=args, output_var="result",
        difficulty="medium", category=cat,
    ))

# ── HARD: Complex multi-step (25 problems) ──

_hard = [
    ("h01", "Compute a*b + c*d - e*f", "result = a * b + c * d - e * f",
     ["a", "b", "c", "d", "e", "f"],
     [_tc({"a": 2, "b": 3, "c": 4, "d": 5, "e": 1, "f": 2}, 24),
      _tc({"a": 5, "b": 5, "c": 3, "d": 3, "e": 2, "f": 2}, 30),
      _tc({"a": 1, "b": 1, "c": 1, "d": 1, "e": 1, "f": 1}, 1),
      _tc({"a": 10, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}, 2),
      _tc({"a": 7, "b": 3, "c": 2, "d": 8, "e": 4, "f": 3}, 25)], "multi_op"),
    ("h02", "x=a, x=x+b, y=x*c, z=y-a, return z",
     "x = a\nx = x + b\ny = x * c\nz = y - a\nresult = z", ["a", "b", "c"],
     [_tc({"a": 2, "b": 3, "c": 4}, 18), _tc({"a": 5, "b": 5, "c": 2}, 15),
      _tc({"a": 1, "b": 1, "c": 10}, 19), _tc({"a": 10, "b": 0, "c": 3}, 20),
      _tc({"a": 3, "b": 7, "c": 5}, 47)], "variable_tracking"),
    ("h03", "Sum of squares from 1 to 5",
     "total = 0\nfor i in range(1, 6):\n    total = total + i * i\nresult = total", [],
     [_tc({}, 55, "result")], "accumulation"),
    ("h04", "Compute (a+b)*(c+d)*(e+f)", "x = a + b\ny = c + d\nz = e + f\nresult = x * y * z",
     ["a", "b", "c", "d", "e", "f"],
     [_tc({"a": 1, "b": 1, "c": 1, "d": 1, "e": 1, "f": 1}, 8),
      _tc({"a": 2, "b": 3, "c": 4, "d": 1, "e": 2, "f": 3}, 125),
      _tc({"a": 0, "b": 5, "c": 0, "d": 5, "e": 0, "f": 5}, 125),
      _tc({"a": 1, "b": 0, "c": 1, "d": 0, "e": 1, "f": 0}, 1),
      _tc({"a": 3, "b": 3, "c": 2, "d": 2, "e": 1, "f": 1}, 48)], "multi_op"),
    ("h05", "Fibonacci-like: a0=a, a1=b, a2=a0+a1, a3=a1+a2, a4=a2+a3. What is a4?",
     "a0 = a\na1 = b\na2 = a0 + a1\na3 = a1 + a2\na4 = a2 + a3\nresult = a4", ["a", "b"],
     [_tc({"a": 1, "b": 1}, 5), _tc({"a": 0, "b": 1}, 3), _tc({"a": 2, "b": 3}, 13),
      _tc({"a": 1, "b": 2}, 8), _tc({"a": 5, "b": 5}, 25)], "variable_tracking"),
    ("h06", "Compute a^3 (a cubed)",
     "result = a * a * a", ["a"],
     [_tc({"a": 2}, 8), _tc({"a": 3}, 27), _tc({"a": 4}, 64), _tc({"a": 5}, 125), _tc({"a": 1}, 1)], "multi_op"),
    ("h07", "Compute (a*a + b*b + c*c)",
     "result = a * a + b * b + c * c", ["a", "b", "c"],
     [_tc({"a": 1, "b": 2, "c": 3}, 14), _tc({"a": 3, "b": 4, "c": 0}, 25),
      _tc({"a": 5, "b": 5, "c": 5}, 75), _tc({"a": 0, "b": 0, "c": 0}, 0),
      _tc({"a": 2, "b": 3, "c": 6}, 49)], "multi_op"),
    ("h08", "Sum of cubes 1 to 4",
     "total = 0\nfor i in range(1, 5):\n    total = total + i * i * i\nresult = total", [],
     [_tc({}, 100, "result")], "accumulation"),
    ("h09", "x=a*b, y=c*d, z=x-y, w=z*z. What is w?",
     "x = a * b\ny = c * d\nz = x - y\nw = z * z\nresult = w", ["a", "b", "c", "d"],
     [_tc({"a": 3, "b": 4, "c": 2, "d": 5}, 4), _tc({"a": 5, "b": 5, "c": 5, "d": 5}, 0),
      _tc({"a": 10, "b": 2, "c": 3, "d": 4}, 64), _tc({"a": 7, "b": 3, "c": 2, "d": 8}, 25),
      _tc({"a": 1, "b": 1, "c": 1, "d": 1}, 0)], "variable_tracking"),
    ("h10", "Compute (a & 0xFF) ^ (b & 0xFF)", "result = (a & 255) ^ (b & 255)", ["a", "b"],
     [_tc({"a": 0x1FF, "b": 0x2FF}, 0), _tc({"a": 0xAA, "b": 0x55}, 0xFF),
      _tc({"a": 256, "b": 0}, 0), _tc({"a": 300, "b": 300}, 0),
      _tc({"a": 0xFF, "b": 0x00}, 0xFF)], "bitwise"),
    ("h11", "Polynomial a^2 + 2ab + b^2 (should equal (a+b)^2)",
     "result = a * a + 2 * a * b + b * b", ["a", "b"],
     [_tc({"a": 3, "b": 4}, 49), _tc({"a": 5, "b": 5}, 100), _tc({"a": 0, "b": 7}, 49),
      _tc({"a": 10, "b": 1}, 121), _tc({"a": 2, "b": 3}, 25)], "multi_op"),
    ("h12", "Alternating sum: a - b + c - d + e",
     "result = a - b + c - d + e", ["a", "b", "c", "d", "e"],
     [_tc({"a": 10, "b": 3, "c": 7, "d": 2, "e": 5}, 17),
      _tc({"a": 1, "b": 1, "c": 1, "d": 1, "e": 1}, 1),
      _tc({"a": 5, "b": 5, "c": 5, "d": 5, "e": 5}, 5),
      _tc({"a": 20, "b": 10, "c": 15, "d": 5, "e": 3}, 23),
      _tc({"a": 0, "b": 0, "c": 0, "d": 0, "e": 0}, 0)], "multi_op"),
    ("h13", "x=1, for i in 1..6: x = x + i*i. What is x?",
     "x = 1\nfor i in range(1, 7):\n    x = x + i * i\nresult = x", [],
     [_tc({}, 92, "result")], "accumulation"),
    ("h14", "Compute a*(a+1)*(a+2) for given a",
     "result = a * (a + 1) * (a + 2)", ["a"],
     [_tc({"a": 1}, 6), _tc({"a": 2}, 24), _tc({"a": 3}, 60), _tc({"a": 4}, 120), _tc({"a": 5}, 210)], "multi_op"),
    ("h15", "x=a, y=b, z=c, x=x+y, y=y+z, z=z+x. What is z?",
     "x = a\ny = b\nz = c\nx = x + y\ny = y + z\nz = z + x\nresult = z", ["a", "b", "c"],
     [_tc({"a": 1, "b": 2, "c": 3}, 6), _tc({"a": 5, "b": 5, "c": 5}, 15),
      _tc({"a": 0, "b": 0, "c": 0}, 0), _tc({"a": 10, "b": 20, "c": 30}, 60),
      _tc({"a": 3, "b": 7, "c": 2}, 12)], "variable_tracking"),
    ("h16", "Compute a*b + b*c + c*a", "result = a * b + b * c + c * a", ["a", "b", "c"],
     [_tc({"a": 1, "b": 2, "c": 3}, 11), _tc({"a": 5, "b": 5, "c": 5}, 75),
      _tc({"a": 0, "b": 0, "c": 0}, 0), _tc({"a": 3, "b": 4, "c": 5}, 47),
      _tc({"a": 2, "b": 3, "c": 4}, 26)], "multi_op"),
    ("h17", "Geometric-ish: x=a, x=x*2+1, x=x*2+1, x=x*2+1",
     "x = a\nx = x * 2 + 1\nx = x * 2 + 1\nx = x * 2 + 1\nresult = x", ["a"],
     [_tc({"a": 0}, 7), _tc({"a": 1}, 15), _tc({"a": 2}, 23), _tc({"a": 3}, 31), _tc({"a": 5}, 47)], "variable_tracking"),
    ("h18", "Compute (a^2 - b^2) which equals (a+b)(a-b)",
     "result = a * a - b * b", ["a", "b"],
     [_tc({"a": 5, "b": 3}, 16), _tc({"a": 10, "b": 6}, 64), _tc({"a": 7, "b": 7}, 0),
      _tc({"a": 20, "b": 1}, 399), _tc({"a": 4, "b": 2}, 12)], "multi_op"),
    ("h19", "x=a, y=b, x=x*y, y=x+y, result=x*y",
     "x = a\ny = b\nx = x * y\ny = x + y\nresult = x * y", ["a", "b"],
     [_tc({"a": 2, "b": 3}, 54), _tc({"a": 3, "b": 4}, 192),
      _tc({"a": 1, "b": 1}, 2), _tc({"a": 5, "b": 2}, 120), _tc({"a": 4, "b": 3}, 180)], "variable_tracking"),
    ("h20", "Sum of 1*2 + 2*3 + 3*4 + 4*5",
     "total = 0\nfor i in range(1, 5):\n    total = total + i * (i + 1)\nresult = total", [],
     [_tc({}, 40, "result")], "accumulation"),
    ("h21", "Compute (a & b) | (a ^ b)", "result = (a & b) | (a ^ b)", ["a", "b"],
     [_tc({"a": 0xFF, "b": 0x0F}, 0xFF), _tc({"a": 12, "b": 10}, 14),
      _tc({"a": 7, "b": 3}, 7), _tc({"a": 0, "b": 0}, 0), _tc({"a": 255, "b": 255}, 255)], "bitwise"),
    ("h22", "x=a+b, y=a*b, z=x*x-y. What is z?",
     "x = a + b\ny = a * b\nz = x * x - y\nresult = z", ["a", "b"],
     [_tc({"a": 3, "b": 4}, 37), _tc({"a": 2, "b": 5}, 39), _tc({"a": 1, "b": 1}, 3),
      _tc({"a": 5, "b": 5}, 75), _tc({"a": 0, "b": 10}, 100)], "variable_tracking"),
    ("h23", "Compute 2*a*a + 3*b*b + a*b",
     "result = 2 * a * a + 3 * b * b + a * b", ["a", "b"],
     [_tc({"a": 1, "b": 1}, 6), _tc({"a": 2, "b": 3}, 41), _tc({"a": 3, "b": 2}, 42),
      _tc({"a": 0, "b": 5}, 75), _tc({"a": 4, "b": 0}, 32)], "multi_op"),
    ("h24", "3^4 via loop",
     "result = 1\nfor i in range(4):\n    result = result * 3", [],
     [_tc({}, 81, "result")], "accumulation"),
    ("h25", "x=a, y=b, z=c. Rotate: x=y, y=z, z=x. Then result=x+y+z",
     "x = a\ny = b\nz = c\nold_x = x\nx = y\ny = z\nz = old_x\nresult = x + y + z", ["a", "b", "c"],
     [_tc({"a": 1, "b": 2, "c": 3}, 6), _tc({"a": 10, "b": 20, "c": 30}, 60),
      _tc({"a": 5, "b": 5, "c": 5}, 15), _tc({"a": 0, "b": 0, "c": 0}, 0),
      _tc({"a": 7, "b": 3, "c": 1}, 11)], "variable_tracking"),
]

for pid, prompt, code, args, tcs, cat in _hard:
    PROBLEMS.append(BenchProblem(
        id=pid, prompt=prompt, reference_code=code,
        test_cases=tcs, arg_names=args, output_var="result",
        difficulty="hard", category=cat,
    ))


class ArithmeticBench:
    """Arithmetic code generation benchmark.

    100 curated problems testing integer arithmetic, variable tracking,
    accumulation, and bitwise operations.
    """

    def __init__(self, problems: Optional[list[BenchProblem]] = None):
        self.problems = problems or PROBLEMS

    @property
    def n_problems(self) -> int:
        return len(self.problems)

    def get_samples(self) -> list[ExecutionTrainingSample]:
        return [p.to_sample() for p in self.problems]

    @torch.no_grad()
    def run(
        self,
        model,
        tokenizer,
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        device: str = "cpu",
    ) -> BenchResult:
        """Run the benchmark on a model."""
        start = time.time()
        evaluator = ExecutionEvaluator(device=device)
        samples = self.get_samples()

        eval_result = evaluator.evaluate(
            model, tokenizer, samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        result = BenchResult(
            total=eval_result.total_samples,
            correct=eval_result.total_correct,
            parseable=eval_result.total_parseable,
            exec_correct=eval_result.total_exec_correct,
            eval_result=eval_result,
            wall_time=time.time() - start,
        )

        # Aggregate by difficulty and category
        for i, sample_r in enumerate(eval_result.per_sample):
            prob = self.problems[i]

            for grouping, key in [(result.by_difficulty, prob.difficulty),
                                   (result.by_category, prob.category)]:
                if key not in grouping:
                    grouping[key] = {"total": 0, "correct": 0, "parseable": 0, "exec_correct": 0}
                grouping[key]["total"] += 1
                if sample_r.get("code_correct"):
                    grouping[key]["correct"] += 1
                if sample_r.get("parseable"):
                    grouping[key]["parseable"] += 1
                if sample_r.get("exec_correct"):
                    grouping[key]["exec_correct"] += 1

        return result

    def validate_problems(self) -> dict:
        """Validate all problems by executing reference code."""
        from .code_parser import CodeToISAParser, ParseError

        parser = CodeToISAParser()
        results = {"total": len(self.problems), "valid": 0, "parse_fail": 0,
                    "exec_fail": 0, "wrong_answer": 0, "errors": []}

        for prob in self.problems:
            # Check Python execution
            for tc in prob.test_cases:
                env = dict(tc.get("inputs", {}))
                try:
                    exec(prob.reference_code, {"__builtins__": {"range": range}}, env)
                    expected = tc["expected"]
                    for var, val in expected.items():
                        if var not in env or abs(env[var] - val) > 0.5:
                            results["wrong_answer"] += 1
                            results["errors"].append(f"{prob.id}: expected {var}={val}, got {env.get(var)}")
                            break
                except Exception as e:
                    results["exec_fail"] += 1
                    results["errors"].append(f"{prob.id}: exec error: {e}")
                    break
            else:
                # Check nCPU parse
                try:
                    parser.parse_block(
                        prob.reference_code,
                        arg_names=prob.arg_names if prob.arg_names else None,
                        output_var=prob.output_var,
                    )
                    results["valid"] += 1
                except ParseError as e:
                    results["parse_fail"] += 1
                    results["errors"].append(f"{prob.id}: parse error: {e}")

        return results
