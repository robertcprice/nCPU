"""Tests for LLM provider extensions used by SOME benchmarks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from ncpu.self_optimizing.llm_provider import LLMProviderFactory


class _FakeBatch(dict):
    def to(self, _device: str) -> "_FakeBatch":
        return self


class _FakeTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"
        self.pad_token_id = 0

    def __call__(self, _prompt: str, return_tensors: str = "pt") -> _FakeBatch:
        return _FakeBatch({"input_ids": torch.tensor([[1, 2, 3]])})

    def decode(self, _ids, skip_special_tokens: bool = True) -> str:
        return "```python\nprint(1)\n```"


class _FakeDecodeTokenizer(_FakeTokenizer):
    eos_token_id = 9
    bos_token_id = 1

    def __call__(self, prompt: str, return_tensors: str = "pt", add_special_tokens: bool = False) -> _FakeBatch:
        del return_tensors, add_special_tokens
        token_ids = [index + 2 for index, _ in enumerate(prompt.split())] or [self.bos_token_id]
        return _FakeBatch({"input_ids": torch.tensor([token_ids], dtype=torch.long)})


class _FakeModel:
    def to(self, _device: str) -> "_FakeModel":
        return self

    def eval(self) -> "_FakeModel":
        return self

    def generate(self, **_kwargs):
        return torch.tensor([[1, 2, 3, 4, 5]])


class _FakeDecodeModel(_FakeModel):
    def __init__(self):
        self.embedding = torch.nn.Embedding(16, 8)

    def get_input_embeddings(self):
        return self.embedding

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
            inputs_embeds = self.embedding(input_ids)
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        past_length = 0
        if past_key_values is not None:
            past_length = int(past_key_values[0][0].shape[2])
        total_length = past_length + seq_len
        logits = torch.full((batch_size, seq_len, self.embedding.num_embeddings), -1000.0)
        next_token = 9 if total_length >= 8 else ((total_length % 7) + 1)
        logits[:, -1, next_token] = 1000.0
        hidden = inputs_embeds + (0.01 * total_length)
        cache = (
            (
                torch.zeros(1, 1, total_length, hidden_size),
                torch.zeros(1, 1, total_length, hidden_size),
            ),
        )
        payload = mock.Mock()
        payload.logits = logits
        payload.past_key_values = cache
        if output_hidden_states:
            payload.hidden_states = (hidden,)
        return payload

    __call__ = forward


class TestLLMProviderFactory(unittest.TestCase):
    def setUp(self) -> None:
        LLMProviderFactory._hf_local_cache.clear()

    def test_hf_local_provider_decodes_direct_model(self):
        fake_model = _FakeModel()
        with (
            mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=_FakeTokenizer()) as tokenizer_load,
            mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=fake_model) as model_load,
        ):
            provider = LLMProviderFactory.create_provider("hf_local", model="stub-model", device="cpu", max_tokens=16)
            response = provider("Write a function.")

        tokenizer_load.assert_called_once_with("stub-model", trust_remote_code=False)
        model_load.assert_called_once()
        self.assertEqual(response["text"], "print(1)")
        self.assertEqual(response["eval_count"], 2)
        self.assertEqual(response["prompt_eval_count"], 3)
        self.assertEqual(response["device"], "cpu")

    def test_hf_local_provider_loads_lora_adapter(self):
        fake_model = _FakeModel()
        tokenizer = _FakeTokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir)
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "base-model"}),
                encoding="utf-8",
            )

            with (
                mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as tokenizer_load,
                mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=_FakeModel()) as model_load,
                mock.patch("peft.PeftModel.from_pretrained", return_value=fake_model) as peft_load,
            ):
                provider = LLMProviderFactory.create_provider("hf_local", model=str(adapter_dir), device="cpu")
                response = provider("Repair this function.")

        tokenizer_load.assert_called_once_with(str(adapter_dir), trust_remote_code=False)
        model_load.assert_called_once_with(
            "base-model",
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=False,
        )
        peft_load.assert_called_once()
        self.assertEqual(response["text"], "print(1)")

    def test_hf_segmented_cache_provider_uses_descriptor_runtime(self):
        tokenizer = _FakeDecodeTokenizer()
        fake_model = _FakeDecodeModel()
        with (
            mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
            mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=fake_model),
        ):
            provider = LLMProviderFactory.create_provider(
                "hf_segmented_cache",
                model="stub-model",
                device="cpu",
                max_tokens=4,
                segmented_cache_recent_window_tokens=3,
                segmented_cache_commit_segment_tokens=2,
                segmented_cache_min_prompt_tokens_for_compression=4,
                segmented_cache_min_tokens_to_commit=2,
            )
            response = provider("one two three four five six")

        self.assertEqual(response["cache_backend"], "segmented_kv")
        self.assertGreater(response["total_commit_events"], 0)
        self.assertEqual(response["device"], "cpu")

    def test_hf_fast_weights_provider_can_enable_state_patch_head(self):
        with mock.patch(
            "ncpu.self_optimizing.task_local_fast_weights.HFTaskLocalFastWeightsProvider",
            autospec=True,
        ) as provider_cls:
            instance = provider_cls.return_value
            provider = LLMProviderFactory.create_provider(
                "hf_fast_weights",
                model="stub-model",
                state_patch_head_enabled=True,
                state_patch_head_input_dim=8,
                state_patch_head_hidden_dim=16,
                state_patch_head_output_dim=8,
            )

        self.assertIs(provider, instance)
        _, kwargs = provider_cls.call_args
        self.assertIsNotNone(kwargs["state_patch_head"])

    def test_hf_fast_weights_provider_can_enable_segmented_decode_backend(self):
        with (
            mock.patch(
                "ncpu.self_optimizing.task_local_fast_weights.HFTaskLocalFastWeightsProvider",
                autospec=True,
            ) as provider_cls,
            mock.patch.object(
                LLMProviderFactory,
                "_build_segmented_decode_runtime",
                return_value="decode-runtime",
            ) as runtime_builder,
        ):
            instance = provider_cls.return_value
            instance.model = object()
            instance.tokenizer = object()
            instance.device = "cpu"

            provider = LLMProviderFactory.create_provider(
                "hf_fast_weights",
                model="stub-model",
                decode_backend="segmented_kv",
            )

        self.assertIs(provider, instance)
        runtime_builder.assert_called_once()
        self.assertEqual(instance.decode_runtime, "decode-runtime")


if __name__ == "__main__":
    unittest.main()
