"""PyTorch Dataset for EGDC diffusion training.

Provides (masked_tokens, mask_positions, original_tokens, spec_tokens, timestep)
tuples for training a discrete diffusion model on nCPU programs.

Specs are encoded as conditioning tokens by serializing test-case I/O values
into a fixed-length token sequence.
"""

from __future__ import annotations
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from egdc.tokenizer import (
    NCPUTokenizer, MASK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN,
    IMM_OFFSET, VOCAB_SIZE,
)
from egdc.data_generator import NCPUDataGenerator


# Spec encoding constants
MAX_TEST_CASES = 4       # Max test cases per spec
MAX_IO_VALUES = 6        # Max input/output values per test case (3 inputs + 1 output + 2 pad)
SPEC_SEQ_LEN = 32        # Fixed length for spec token sequence


class NCPUDataset(Dataset):
    """PyTorch Dataset for nCPU program diffusion training.

    Each item returns a tuple of tensors:
        masked_tokens:   (seq_len,) int64 - program with random positions masked
        mask_positions:  (seq_len,) bool  - True where tokens are masked
        original_tokens: (seq_len,) int64 - original unmasked program
        spec_tokens:     (spec_len,) int64 - encoded specification (test cases)
        timestep:        scalar float     - diffusion timestep in [0, 1]

    Programs are padded/truncated to seq_len=128.
    Specs are encoded to spec_len=32.
    """

    def __init__(
        self,
        num_samples: int = 100_000,
        seq_len: int = 128,
        spec_len: int = SPEC_SEQ_LEN,
        seed: int = 42,
        cache_path: Optional[str] = None,
        balanced: bool = True,
        num_diffusion_steps: int = 1000,
    ):
        """Initialize the dataset.

        Args:
            num_samples: Number of program samples to generate.
            seq_len: Fixed sequence length for program tokens (pad/truncate).
            spec_len: Fixed sequence length for spec tokens.
            seed: Random seed for reproducibility.
            cache_path: If set, cache generated data to/from this JSON file.
            balanced: Whether to balance across template families.
            num_diffusion_steps: Number of discrete diffusion timesteps.
        """
        super().__init__()
        self.tokenizer = NCPUTokenizer()
        self.seq_len = seq_len
        self.spec_len = spec_len
        self.num_diffusion_steps = num_diffusion_steps
        self.rng = random.Random(seed)

        # Load or generate data
        if cache_path and os.path.exists(cache_path):
            self.data = self._load_cache(cache_path)
        else:
            gen = NCPUDataGenerator(seed=seed)
            self.data = gen.generate_dataset(num_samples, balanced=balanced)
            if cache_path:
                self._save_cache(cache_path, self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        spec_dict, program_tokens = self.data[idx]

        # --- Program tokens: pad/truncate to seq_len ---
        prog = self.tokenizer.pad(list(program_tokens), self.seq_len)
        original_tokens = torch.tensor(prog, dtype=torch.long)

        # --- Spec tokens: encode test cases ---
        spec_tokens = torch.tensor(
            self._encode_spec(spec_dict), dtype=torch.long
        )

        # --- Diffusion masking ---
        # Sample a random timestep t in [0, 1]
        # Higher t = more masking (noisier)
        t_int = self.rng.randint(0, self.num_diffusion_steps - 1)
        timestep = torch.tensor(t_int / self.num_diffusion_steps, dtype=torch.float32)

        # Mask probability proportional to timestep
        mask_prob = t_int / self.num_diffusion_steps

        # Create mask: only mask actual program tokens (not PAD, BOS, EOS)
        mask_positions = torch.zeros(self.seq_len, dtype=torch.bool)
        for i in range(self.seq_len):
            tok = prog[i]
            if tok in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
                continue
            if self.rng.random() < mask_prob:
                mask_positions[i] = True

        # Apply mask
        masked_tokens = original_tokens.clone()
        masked_tokens[mask_positions] = MASK_TOKEN

        return masked_tokens, mask_positions, original_tokens, spec_tokens, timestep

    # ------------------------------------------------------------------
    # Spec encoding
    # ------------------------------------------------------------------

    def _encode_spec(self, spec_dict: Dict[str, Any]) -> List[int]:
        """Encode a spec dict into a fixed-length token sequence.

        Format: for each test case, encode input values and expected output
        as IMM_offset tokens. Pad to spec_len.

        The encoding uses immediate tokens (IMM_0..IMM_255) to represent
        values, with PAD_TOKEN for unused slots.
        """
        tokens: List[int] = []

        test_cases = spec_dict.get("test_cases", [])[:MAX_TEST_CASES]

        for tc in test_cases:
            inputs = tc.get("inputs", {})
            expected = tc.get("expected_output", 0)

            # Encode input values (sorted by key for determinism)
            input_vals = [v for _, v in sorted(inputs.items())]
            for v in input_vals[:MAX_IO_VALUES - 1]:
                val = max(0, min(255, int(v)))
                tokens.append(IMM_OFFSET + val)

            # Pad inputs to MAX_IO_VALUES - 1
            while len(tokens) % MAX_IO_VALUES != MAX_IO_VALUES - 1:
                tokens.append(PAD_TOKEN)

            # Encode expected output
            out_val = max(0, min(255, int(expected)))
            tokens.append(IMM_OFFSET + out_val)

        # Pad to spec_len
        while len(tokens) < self.spec_len:
            tokens.append(PAD_TOKEN)

        return tokens[:self.spec_len]

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _save_cache(path: str, data: List[Tuple[Dict, List[int]]]) -> None:
        """Save generated data to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        serializable = [
            {"spec": spec, "tokens": tokens}
            for spec, tokens in data
        ]
        with open(path, "w") as f:
            json.dump(serializable, f)

    @staticmethod
    def _load_cache(path: str) -> List[Tuple[Dict, List[int]]]:
        """Load cached data from a JSON file."""
        with open(path) as f:
            raw = json.load(f)
        return [(item["spec"], item["tokens"]) for item in raw]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def decode_program(self, token_ids: torch.Tensor) -> str:
        """Decode a token tensor back to assembly text."""
        return self.tokenizer.decode(token_ids.tolist())

    def get_dataloader(self, batch_size: int = 64, shuffle: bool = True,
                       num_workers: int = 0, **kwargs) -> "torch.utils.data.DataLoader":
        """Convenience method to create a DataLoader."""
        from torch.utils.data import DataLoader
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, **kwargs,
        )
