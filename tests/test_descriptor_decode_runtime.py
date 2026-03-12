"""Tests for segmented-cache descriptor decoding."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from ncpu.self_optimizing.descriptor_decode_runtime import (
    DescriptorDecodeConfig,
    DescriptorDecodeRuntime,
)
from ncpu.self_optimizing.recurrent_commit_policy import RecurrentCommitPolicyConfig
from ncpu.self_optimizing.segmented_kv_cache import SegmentedKVCacheConfig


class _FakeBatch(dict):
    def to(self, _device: str) -> "_FakeBatch":
        return self


class _FakeDecodeTokenizer:
    pad_token = None
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 9
    bos_token_id = 1

    def __call__(self, prompt: str, return_tensors: str = "pt", add_special_tokens: bool = False) -> _FakeBatch:
        del return_tensors, add_special_tokens
        token_ids = [index + 2 for index, _ in enumerate(prompt.split())] or [self.bos_token_id]
        return _FakeBatch({"input_ids": torch.tensor([token_ids], dtype=torch.long)})

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        values = [int(token_id) for token_id in token_ids]
        if skip_special_tokens:
            values = [value for value in values if value != self.eos_token_id]
        return " ".join(f"tok{value}" for value in values)


class _FakeDecodeModel(nn.Module):
    def __init__(self, *, vocab_size: int = 16, hidden_size: int = 8, eos_token_id: int = 9):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.eos_token_id = eos_token_id

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def to(self, device: str) -> "_FakeDecodeModel":
        super().to(device)
        return self

    def eval(self) -> "_FakeDecodeModel":
        super().eval()
        return self

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
        attention_mask=None,
    ):
        del use_cache, attention_mask
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embedding(input_ids)
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        past_length = 0
        if past_key_values is not None:
            past_length = int(past_key_values[0][0].shape[2])
        total_length = past_length + seq_len
        logits = torch.full(
            (batch_size, seq_len, self.embedding.num_embeddings),
            -1000.0,
            device=inputs_embeds.device,
        )
        next_token_id = self.eos_token_id if total_length >= 8 else ((total_length % 7) + 1)
        logits[:, -1, next_token_id] = 1000.0
        hidden = inputs_embeds + (0.01 * total_length)
        cache = (
            (
                torch.zeros(1, 1, total_length, hidden_size, device=inputs_embeds.device),
                torch.zeros(1, 1, total_length, hidden_size, device=inputs_embeds.device),
            ),
        )
        output = SimpleNamespace(logits=logits, past_key_values=cache)
        if output_hidden_states:
            output.hidden_states = (hidden,)
        return output


class TestDescriptorDecodeRuntime(unittest.TestCase):
    def test_runtime_commits_prompt_prefix_into_memory(self):
        runtime = DescriptorDecodeRuntime(
            model=_FakeDecodeModel().to("cpu").eval(),
            tokenizer=_FakeDecodeTokenizer(),
            device="cpu",
            config=DescriptorDecodeConfig(
                max_new_tokens=4,
                cache=SegmentedKVCacheConfig(
                    recent_window_tokens=3,
                    commit_segment_tokens=2,
                    descriptor_tokens_per_segment=2,
                    max_memory_segments=8,
                    min_prompt_tokens_for_compression=4,
                    min_tokens_to_commit=2,
                ),
                commit_policy=RecurrentCommitPolicyConfig(
                    recent_window_tokens=3,
                    commit_segment_tokens=2,
                    min_tokens_to_commit=2,
                ),
            ),
        )

        result = runtime.generate("one two three four five six")

        self.assertEqual(result["cache_backend"], "segmented_kv")
        self.assertEqual(result["prompt_eval_count"], 6)
        self.assertGreater(result["total_commit_events"], 0)
        self.assertGreater(result["memory_segment_count"], 0)
        self.assertIn("tok", result["text"])


if __name__ == "__main__":
    unittest.main()
