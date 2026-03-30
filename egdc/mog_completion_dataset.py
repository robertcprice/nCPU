"""Body-completion dataset for benchmark-conditioned Mog training.

Each example keeps the structural scaffold of the reference solution fixed and
masks the function bodies. The model is trained to reconstruct the masked body
bytes given:
- the visible scaffold tokens as input tokens
- the signature/description as spec tokens

This directly tests whether the model can do structured code completion, which
is easier and more realistic than generating an entire program from all masks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from egdc.mog_tokenizer import MogCodeTokenizer, MASK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN
from egdc.mog_benchmark import MogBenchmarkProblem, get_benchmark
from egdc.mog_completion import build_completion_tokens


@dataclass
class MogCompletionExample:
    problem_name: str
    category: str
    prompt: str
    code: str


class MogBenchmarkCompletionDataset(Dataset):
    def __init__(
        self,
        problems: Optional[List[MogBenchmarkProblem]] = None,
        num_problems: Optional[int] = None,
        variants_per_factory: int = 1,
        seq_len: int = 512,
        spec_len: int = 128,
        repeat: int = 64,
        seed: int = 42,
        timestep_value: float = 1.0,
        include_description: bool = True,
    ):
        super().__init__()
        self.tokenizer = MogCodeTokenizer()
        self.seq_len = seq_len
        self.spec_len = spec_len
        self.repeat = repeat
        self.rng = random.Random(seed)
        self.timestep_value = timestep_value
        self.include_description = include_description

        if problems is None:
            problems = get_benchmark(seed=seed, variants_per_factory=variants_per_factory)
        if num_problems is not None:
            problems = problems[:num_problems]

        self.examples: List[MogCompletionExample] = []
        for p in problems:
            if not p.reference_solution:
                continue
            prompt = f"{p.signature}\n// {p.description}" if include_description else p.signature
            self.examples.append(MogCompletionExample(
                problem_name=p.name,
                category=p.category,
                prompt=prompt,
                code=p.reference_solution.strip() + "\n",
            ))

        self.data = []
        for ex in self.examples:
            init_tokens, fixed_positions, original_tokens = build_completion_tokens(ex.code, self.tokenizer, self.seq_len)
            spec_tokens = self.tokenizer.pad(self.tokenizer.encode(ex.prompt, add_bos_eos=False), self.spec_len)
            spec_tensor = torch.tensor(spec_tokens, dtype=torch.long)
            mask_positions = (init_tokens == MASK_TOKEN)
            for _ in range(self.repeat):
                self.data.append((init_tokens.clone(), fixed_positions.clone(), original_tokens.clone(), spec_tensor.clone(), ex, mask_positions.clone()))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        init_tokens, _fixed_positions, original_tokens, spec_tensor, _ex, mask_positions = self.data[idx]
        timestep = torch.tensor(self.timestep_value, dtype=torch.float32)
        return init_tokens.clone(), mask_positions.clone(), original_tokens.clone(), spec_tensor.clone(), timestep

    def get_scaffold(self, idx: int):
        return self.data[idx][0]

    def get_fixed_positions(self, idx: int):
        return self.data[idx][1]

    def get_example(self, idx: int) -> MogCompletionExample:
        return self.data[idx][4]

    def unique_examples(self) -> List[MogCompletionExample]:
        return self.examples
