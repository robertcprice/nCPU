"""Training data generators for the nCPU differentiable coprocessor.

Provides:
  - ArithmeticDataset: synthetic arithmetic problems (a op b = c) as
    tokenized sequences for causal LM training
  - GSM8KArithmeticDataset: extracts arithmetic subproblems from GSM8K
  - MATHArithmeticDataset: extracts arithmetic from MATH competition set

The key idea: we don't train on the full GSM8K reasoning chain. We extract
the ARITHMETIC SUBEXPRESSIONS and train the coprocessor to route those
tokens through the nCPU expert, while leaving reasoning tokens on the
original MLP path.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class ArithmeticSample:
    """A single arithmetic training example."""
    expression: str          # e.g., "42 + 17 = 59"
    operand_a: int
    operand_b: int
    operation: str           # ADD, SUB, MUL, AND, OR, XOR, CMP
    result: int
    label_tokens: Optional[torch.Tensor] = None  # tokenized result


# ---------------------------------------------------------------------------
# Synthetic arithmetic dataset
# ---------------------------------------------------------------------------

class ArithmeticDataset(Dataset):
    """Generates synthetic arithmetic problems for coprocessor training.

    Each sample is a text string like "42 + 17 = 59" that gets tokenized
    for causal LM training. The model learns to produce the correct result
    after "=" by routing through the nCPU expert.

    Supports: ADD, SUB, MUL, AND, OR, XOR, CMP, MOD, DIV.
    """

    OPS = {
        "ADD": ("+", lambda a, b: a + b),
        "SUB": ("-", lambda a, b: a - b),
        "MUL": ("*", lambda a, b: a * b),
        "AND": ("&", lambda a, b: a & b),
        "OR":  ("|", lambda a, b: a | b),
        "XOR": ("^", lambda a, b: a ^ b),
        "MOD": ("%", lambda a, b: a % b if b != 0 else 0),
        "DIV": ("//", lambda a, b: a // b if b != 0 else 0),
    }

    def __init__(
        self,
        size: int = 10000,
        max_value: int = 999,
        ops: Optional[List[str]] = None,
        tokenizer=None,
        max_length: int = 32,
        seed: int = 42,
        difficulty: str = "mixed",
    ):
        """
        Args:
            size: number of samples to generate
            max_value: maximum operand value
            ops: which operations to include (default: all)
            tokenizer: HF tokenizer for encoding (None = return raw strings)
            max_length: max token length for padding
            seed: random seed for reproducibility
            difficulty: "easy" (1-99), "medium" (1-999), "hard" (1-9999), "mixed"
        """
        self.size = size
        self.max_value = max_value
        self.ops = ops or list(self.OPS.keys())
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.difficulty = difficulty
        self.samples: List[ArithmeticSample] = []

        rng = random.Random(seed)
        self._generate(rng)

    def _get_range(self, rng: random.Random) -> Tuple[int, int]:
        """Get operand range based on difficulty."""
        if self.difficulty == "easy":
            return 1, min(99, self.max_value)
        elif self.difficulty == "medium":
            return 1, min(999, self.max_value)
        elif self.difficulty == "hard":
            return 1, min(9999, self.max_value)
        else:  # mixed
            tier = rng.choice(["easy", "medium", "hard"])
            if tier == "easy":
                return 1, min(99, self.max_value)
            elif tier == "medium":
                return 1, min(999, self.max_value)
            else:
                return 1, min(9999, self.max_value)

    def _generate(self, rng: random.Random) -> None:
        for _ in range(self.size):
            op_name = rng.choice(self.ops)
            symbol, fn = self.OPS[op_name]
            lo, hi = self._get_range(rng)

            a = rng.randint(lo, hi)
            b = rng.randint(lo, hi)

            # Avoid division by zero
            if op_name in ("MOD", "DIV") and b == 0:
                b = 1

            result = fn(a, b)
            expr = f"{a} {symbol} {b} = {result}"

            self.samples.append(ArithmeticSample(
                expression=expr,
                operand_a=a,
                operand_b=b,
                operation=op_name,
                result=result,
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        if self.tokenizer is None:
            return {
                "text": sample.expression,
                "operation": sample.operation,
                "code_snippet": raw.code_snippet,
                "training_text": f"{raw.code_snippet}\nResult: {sample.result}",
            }

        # Tokenize for causal LM: input = "42 + 17 =", target = " 59"
        prompt = sample.expression.split("=")[0] + "="
        full = sample.expression

        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", padding=False, add_special_tokens=False
        ).input_ids[0]

        full_ids = self.tokenizer(
            full, return_tensors="pt", padding="max_length",
            max_length=self.max_length, truncation=True, add_special_tokens=False
        ).input_ids[0]

        # Create labels: -100 for prompt tokens (don't compute loss on input)
        labels = full_ids.clone()
        labels[:len(prompt_ids)] = -100

        attention_mask = (full_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "operation": sample.operation,
        }


# ---------------------------------------------------------------------------
# GSM8K arithmetic extraction
# ---------------------------------------------------------------------------

# Regex for arithmetic expressions in GSM8K solutions
_ARITH_PATTERN = re.compile(
    r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*=\s*(\d+(?:,\d{3})*(?:\.\d+)?)'
)


def _parse_number(s: str) -> float:
    """Parse a number string, handling commas."""
    return float(s.replace(",", ""))


class GSM8KArithmeticDataset(Dataset):
    """Extracts arithmetic subexpressions from GSM8K solutions.

    GSM8K problems contain step-by-step solutions with expressions like:
        "She has 42 + 17 = 59 apples"

    We extract these as arithmetic training samples for the coprocessor.
    The model learns to route these computation tokens through nCPU.

    Requires: gsm8k dataset (download from HuggingFace or provide path)
    """

    OP_MAP = {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV"}

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: str = "train",
        tokenizer=None,
        max_length: int = 64,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[ArithmeticSample] = []

        data = self._load_data(data_path, split)
        self._extract_arithmetic(data)

    def _load_data(self, data_path: Optional[str], split: str) -> list:
        """Load GSM8K from local file or HuggingFace."""
        if data_path and Path(data_path).exists():
            with open(data_path) as f:
                return [json.loads(line) for line in f]

        # Try HuggingFace datasets
        try:
            from datasets import load_dataset
            ds = load_dataset("openai/gsm8k", "main", split=split)
            return list(ds)
        except (ImportError, Exception):
            return []

    def _extract_arithmetic(self, data: list) -> None:
        """Extract arithmetic expressions from GSM8K solution strings."""
        for item in data:
            answer_text = item.get("answer", "")
            for match in _ARITH_PATTERN.finditer(answer_text):
                a_str, op, b_str, result_str = match.groups()
                try:
                    a = _parse_number(a_str)
                    b = _parse_number(b_str)
                    result = _parse_number(result_str)

                    # Only use integer arithmetic for coprocessor training
                    if a == int(a) and b == int(b) and result == int(result):
                        op_name = self.OP_MAP.get(op, "ADD")
                        symbol = op
                        expr = f"{int(a)} {symbol} {int(b)} = {int(result)}"
                        self.samples.append(ArithmeticSample(
                            expression=expr,
                            operand_a=int(a),
                            operand_b=int(b),
                            operation=op_name,
                            result=int(result),
                        ))
                except (ValueError, ZeroDivisionError):
                    continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        if self.tokenizer is None:
            return {"text": sample.expression, "operation": sample.operation}

        prompt = sample.expression.split("=")[0] + "="
        full = sample.expression

        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", padding=False, add_special_tokens=False
        ).input_ids[0]

        full_ids = self.tokenizer(
            full, return_tensors="pt", padding="max_length",
            max_length=self.max_length, truncation=True, add_special_tokens=False
        ).input_ids[0]

        labels = full_ids.clone()
        labels[:len(prompt_ids)] = -100
        attention_mask = (full_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "operation": sample.operation,
        }


# ---------------------------------------------------------------------------
# MATH competition dataset extraction
# ---------------------------------------------------------------------------

class MATHArithmeticDataset(Dataset):
    """Extracts arithmetic from MATH competition problems.

    Filters for number theory / algebra problems that contain
    explicit arithmetic computations.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: str = "train",
        tokenizer=None,
        max_length: int = 64,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[ArithmeticSample] = []

        data = self._load_data(data_path, split)
        self._extract_arithmetic(data)

    def _load_data(self, data_path: Optional[str], split: str) -> list:
        if data_path and Path(data_path).exists():
            with open(data_path) as f:
                return [json.loads(line) for line in f]
        try:
            from datasets import load_dataset
            ds = load_dataset("hendrycks/competition_math", split=split)
            return list(ds)
        except (ImportError, Exception):
            return []

    def _extract_arithmetic(self, data: list) -> None:
        for item in data:
            solution = item.get("solution", "")
            for match in _ARITH_PATTERN.finditer(solution):
                a_str, op, b_str, result_str = match.groups()
                try:
                    a = _parse_number(a_str)
                    b = _parse_number(b_str)
                    result = _parse_number(result_str)
                    if a == int(a) and b == int(b) and result == int(result):
                        op_name = GSM8KArithmeticDataset.OP_MAP.get(op, "ADD")
                        expr = f"{int(a)} {op} {int(b)} = {int(result)}"
                        self.samples.append(ArithmeticSample(
                            expression=expr,
                            operand_a=int(a),
                            operand_b=int(b),
                            operation=op_name,
                            result=int(result),
                        ))
                except (ValueError, ZeroDivisionError):
                    continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        if self.tokenizer is None:
            return {"text": sample.expression, "operation": sample.operation}
        # Same tokenization as GSM8K
        prompt = sample.expression.split("=")[0] + "="
        full = sample.expression
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", padding=False, add_special_tokens=False
        ).input_ids[0]
        full_ids = self.tokenizer(
            full, return_tensors="pt", padding="max_length",
            max_length=self.max_length, truncation=True, add_special_tokens=False
        ).input_ids[0]
        labels = full_ids.clone()
        labels[:len(prompt_ids)] = -100
        attention_mask = (full_ids != self.tokenizer.pad_token_id).long()
        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "operation": sample.operation,
        }


# ---------------------------------------------------------------------------
# Combined dataset
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Code-embedded arithmetic dataset
# ---------------------------------------------------------------------------

class CodeArithmeticDataset(Dataset):
    """Arithmetic embedded in realistic code contexts.

    Training on patterns like:
    - arr[i + offset]
    - range(start, end)
    - if i < len(arr) - 1
    - total += value

    This helps the coprocessor learn WHEN to activate during code generation,
    not just on raw arithmetic expressions.
    """

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

    def __init__(
        self,
        size: int = 10000,
        tokenizer=None,
        max_length: int = 64,
        seed: int = 42,
        max_value: int = 255,
    ):
        from .code_arithmetic_data import CodeArithmeticGenerator
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.raw_samples = []

        gen = CodeArithmeticGenerator(seed=seed, max_value=max_value)
        self.raw_samples = gen.generate(size)

        for sample in self.raw_samples:
            expr = f"{sample.arithmetic_expr} = {sample.result}"
            self.samples.append(ArithmeticSample(
                expression=expr,
                operand_a=sample.operands[0] if sample.operands else 0,
                operand_b=sample.operands[1] if len(sample.operands) > 1 else 0,
                operation=sample.operation,
                result=sample.result,
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        raw = self.raw_samples[idx]

        if self.tokenizer is None:
            return {"text": sample.expression, "operation": sample.operation}

        # Include code context for training
        full_text = f"{raw.code_snippet}\nResult: {sample.result}"

        full_ids = self.tokenizer(
            full_text, return_tensors="pt", padding="max_length",
            max_length=self.max_length, truncation=True, add_special_tokens=False
        ).input_ids[0]

        # Find where "Result:" starts for label masking
        result_marker = f"\nResult: {sample.result}"
        prompt_text = raw.code_snippet + "\nResult:"
        prompt_ids = self.tokenizer(
            prompt_text, return_tensors="pt", padding=False, add_special_tokens=False
        ).input_ids[0]

        labels = full_ids.clone()
        labels[:len(prompt_ids)] = -100
        attention_mask = (full_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "operation": sample.operation,
        }


class CombinedArithmeticDataset(Dataset):
    """Combines synthetic + GSM8K + MATH arithmetic samples."""

    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self._lengths = [len(d) for d in datasets]
        self._cumulative = []
        total = 0
        for length in self._lengths:
            self._cumulative.append(total)
            total += length
        self._total = total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int):
        for i, cum in enumerate(self._cumulative):
            if i + 1 < len(self._cumulative):
                if idx < self._cumulative[i + 1]:
                    return self.datasets[i][idx - cum]
            else:
                return self.datasets[i][idx - cum]
        return self.datasets[-1][idx - self._cumulative[-1]]
