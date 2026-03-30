"""Benchmark-supervised dataset for Mog code generation.

Unlike the synthetic mog_dataset.py, this dataset trains directly on benchmark
reference solutions. It is intended for overfit and supervised fine-tuning
experiments where the model should learn to map:

    (signature + natural language description) -> reference solution code

The training objective remains masked discrete diffusion over the target code.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from egdc.mog_tokenizer import MogCodeTokenizer, MASK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN
from egdc.mog_benchmark import MogBenchmarkProblem, get_benchmark


@dataclass
class MogSupervisedExample:
    problem_name: str
    category: str
    prompt: str
    code: str


class MogBenchmarkSupervisedDataset(Dataset):
    """Masked-diffusion dataset built from Mog benchmark reference solutions."""

    def __init__(
        self,
        problems: Optional[List[MogBenchmarkProblem]] = None,
        num_problems: Optional[int] = None,
        variants_per_factory: int = 1,
        seq_len: int = 512,
        spec_len: int = 128,
        repeat: int = 64,
        seed: int = 42,
        num_diffusion_steps: int = 1000,
        include_description: bool = True,
    ):
        super().__init__()
        self.tokenizer = MogCodeTokenizer()
        self.seq_len = seq_len
        self.spec_len = spec_len
        self.repeat = repeat
        self.seed = seed
        self.rng = random.Random(seed)
        self.num_diffusion_steps = num_diffusion_steps
        self.include_description = include_description

        if problems is None:
            problems = get_benchmark(seed=seed, variants_per_factory=variants_per_factory)
        if num_problems is not None:
            problems = problems[:num_problems]

        self.examples: List[MogSupervisedExample] = []
        for p in problems:
            if not p.reference_solution:
                continue
            prompt = self._make_prompt(p)
            self.examples.append(
                MogSupervisedExample(
                    problem_name=p.name,
                    category=p.category,
                    prompt=prompt,
                    code=p.reference_solution.strip() + "\n",
                )
            )

        self.data: List[Tuple[List[int], List[int], MogSupervisedExample]] = []
        for ex in self.examples:
            code_tokens = self.tokenizer.encode(ex.code)
            spec_tokens = self.tokenizer.encode(ex.prompt, add_bos_eos=False)
            for _ in range(self.repeat):
                self.data.append((code_tokens, spec_tokens, ex))

    def _make_prompt(self, p: MogBenchmarkProblem) -> str:
        if self.include_description:
            return f"{p.signature}\n// {p.description}"
        return p.signature

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        code_tokens, spec_tokens, _ex = self.data[idx]
        prog = self.tokenizer.pad(list(code_tokens), self.seq_len)
        original_tokens = torch.tensor(prog, dtype=torch.long)

        spec = self.tokenizer.pad(list(spec_tokens), self.spec_len)
        spec_tensor = torch.tensor(spec, dtype=torch.long)

        t_int = self.rng.randint(0, self.num_diffusion_steps - 1)
        timestep = torch.tensor(t_int / self.num_diffusion_steps, dtype=torch.float32)
        mask_prob = t_int / self.num_diffusion_steps

        mask_positions = torch.zeros(self.seq_len, dtype=torch.bool)
        for i in range(self.seq_len):
            tok = prog[i]
            if tok in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
                continue
            if self.rng.random() < mask_prob:
                mask_positions[i] = True

        masked_tokens = original_tokens.clone()
        masked_tokens[mask_positions] = MASK_TOKEN
        return masked_tokens, mask_positions, original_tokens, spec_tensor, timestep

    def get_example(self, idx: int) -> MogSupervisedExample:
        return self.data[idx][2]

    def unique_examples(self) -> List[MogSupervisedExample]:
        return self.examples

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size
