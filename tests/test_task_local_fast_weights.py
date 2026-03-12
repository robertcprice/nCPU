"""Tests for task-local fast-weight adapters and providers."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn
import torch.nn.functional as F

from ncpu.self_optimizing.llm_provider import LLMProviderFactory
from ncpu.self_optimizing.ncpu_adaptation_backend import NCPUAdaptationBackend, NCPUAdaptationConfig
from ncpu.self_optimizing.state_patch_head import StatePatchHead, StatePatchHeadConfig
from ncpu.self_optimizing.task_local_fast_weights import (
    HFTaskLocalFastWeightsProvider,
    TaskLocalFastWeightConfig,
    find_target_linear_modules,
)


class _ToyBatch(dict):
    def to(self, device: str) -> "_ToyBatch":
        for key, value in list(self.items()):
            self[key] = value.to(device)
        return self


class _ToyTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        add_special_tokens: bool = False,
    ) -> _ToyBatch:
        del return_tensors, add_special_tokens
        token_ids = [((ord(ch) - 31) % 32) + 1 for ch in text] or [1]
        return _ToyBatch({"input_ids": torch.tensor([token_ids], dtype=torch.long)})

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(int(token) % 26 + 97) for token in token_ids.tolist())


class _ToyCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 8)
        self.q_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.lm_head = nn.Linear(8, 64)

    def forward(self, input_ids, attention_mask=None, labels=None):
        del attention_mask
        hidden = self.embed(input_ids)
        hidden = torch.tanh(self.q_proj(hidden))
        hidden = torch.tanh(self.v_proj(hidden))
        logits = self.lm_head(hidden)
        if labels is None:
            return SimpleNamespace(logits=logits)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(logits=logits, loss=loss)

    def generate(self, input_ids, **kwargs):
        max_new_tokens = int(kwargs.get("max_new_tokens", 1))
        generated = input_ids
        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids=generated)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
        return generated


class TestTaskLocalFastWeights(unittest.TestCase):
    def setUp(self) -> None:
        LLMProviderFactory._hf_local_cache.clear()

    def test_find_target_linear_modules_matches_named_projections(self):
        model = _ToyCausalModel()
        matches = find_target_linear_modules(
            model,
            target_fragments=("q_proj", "v_proj"),
        )

        self.assertEqual([name for name, _module in matches], ["q_proj", "v_proj"])

    def test_hf_fast_weights_provider_updates_and_resets_task_local_adapters(self):
        toy_model = _ToyCausalModel()
        tokenizer = _ToyTokenizer()
        with mock.patch.object(
            LLMProviderFactory,
            "_load_hf_local_model",
            return_value=(toy_model, tokenizer, "cpu"),
        ):
            provider = HFTaskLocalFastWeightsProvider(
                model="stub-model",
                config=TaskLocalFastWeightConfig(
                    rank=2,
                    gradient_steps=1,
                    learning_rate=0.1,
                    target_modules=("q_proj", "v_proj"),
                    max_target_tokens=8,
                ),
                device="cpu",
                max_tokens=2,
            )

            begin_metadata = provider.begin_task("demo", "Write code.")
            self.assertEqual(begin_metadata["updated_modules"], ["q_proj", "v_proj"])

            initial_b = provider._adapters[0].fast_b.weight.detach().clone()
            update = provider.apply_self_update(
                prompt="Plan:\n",
                target_text="Use recurrence",
                update_kind="plan",
                task_name="demo",
            )
            self.assertTrue(update.success)
            self.assertEqual(update.kind, "plan")
            self.assertGreaterEqual(update.steps, 1)
            self.assertIsNotNone(update.adaptation_descriptor)
            self.assertEqual(update.adaptation_descriptor["implementation"], "ncpu_gradient_protocol")
            self.assertFalse(torch.equal(initial_b, provider._adapters[0].fast_b.weight.detach()))

            response = provider("Write code.")
            self.assertIn("text", response)
            self.assertEqual(response["fast_weight_updates"], 1)
            self.assertEqual(response["active_task_name"], "demo")

            end_metadata = provider.end_task()
            self.assertEqual(end_metadata["update_count"], 1)
            self.assertEqual(end_metadata["ncpu_adaptation"]["successful_updates"], 1)
            self.assertTrue(torch.count_nonzero(provider._adapters[0].fast_b.weight).item() == 0)

    def test_ncpu_adaptation_backend_builds_descriptor_from_gradients(self):
        backend = NCPUAdaptationBackend(
            NCPUAdaptationConfig(
                compression_type="top_k",
                top_k_ratio=0.25,
                gradient_clip=0.5,
                max_gradient_steps=3,
            )
        )
        backend.begin_task("demo", "Write code.")

        descriptor = backend.build_update_descriptor(
            task_name="demo",
            update_kind="verify_failure",
            gradients=torch.tensor([1.5, -0.5, 0.0, 0.25], dtype=torch.float32).numpy(),
        )

        self.assertEqual(descriptor.task_name, "demo")
        self.assertEqual(descriptor.update_kind, "verify_failure")
        self.assertEqual(descriptor.descriptor["compression"], "top_k")
        self.assertGreaterEqual(descriptor.suggested_gradient_steps, 2)
        self.assertGreaterEqual(descriptor.learning_rate_scale, 0.75)
        backend.record_update_result(descriptor, success=True)
        summary = backend.end_task()
        self.assertEqual(summary["successful_updates"], 1)

    def test_provider_can_apply_latent_state_descriptor_update_without_text_target(self):
        toy_model = _ToyCausalModel()
        tokenizer = _ToyTokenizer()
        with mock.patch.object(
            LLMProviderFactory,
            "_load_hf_local_model",
            return_value=(toy_model, tokenizer, "cpu"),
        ):
            provider = HFTaskLocalFastWeightsProvider(
                model="stub-model",
                config=TaskLocalFastWeightConfig(
                    rank=2,
                    gradient_steps=1,
                    learning_rate=0.1,
                    target_modules=("q_proj", "v_proj"),
                    max_target_tokens=8,
                ),
                device="cpu",
                max_tokens=2,
            )
            provider.begin_task("demo", "Write code.")
            initial_a = provider._adapters[0].fast_a.weight.detach().clone()

            result = provider.apply_state_descriptor_update(
                task_name="demo",
                update_kind="verify_failure_descriptor",
                latent_state={
                    "verification_passes": 0,
                    "verification_failures": 1,
                    "fast_weight_updates_used": 0,
                    "descriptor_updates_used": 0,
                    "confidence": 0.1,
                    "failure_patterns": ["expected fibonacci recurrence"],
                    "verified_constraints": ["output_format=python"],
                    "recent_actions": ["write", "verify"],
                },
                error_text="expected 8, got 0",
                candidate_text="def fib(n): return 0",
            )

            self.assertTrue(result.success)
            self.assertEqual(result.kind, "verify_failure_descriptor")
            self.assertEqual(result.adaptation_descriptor["source"], "latent_state")
            self.assertEqual(result.implementation, "task_local_low_rank_residual+descriptor_update")
            self.assertFalse(torch.equal(initial_a, provider._adapters[0].fast_a.weight.detach()))

    def test_factory_can_build_hf_fast_weights_provider(self):
        with mock.patch(
            "ncpu.self_optimizing.task_local_fast_weights.HFTaskLocalFastWeightsProvider",
            autospec=True,
        ) as provider_cls:
            instance = provider_cls.return_value
            provider = LLMProviderFactory.create_provider(
                "hf_fast_weights",
                model="stub-model",
                fast_weights_rank=4,
                fast_weights_gradient_steps=2,
                fast_weights_target_modules="q_proj,v_proj",
            )

        self.assertIs(provider, instance)
        _, kwargs = provider_cls.call_args
        self.assertEqual(kwargs["model"], "stub-model")
        self.assertEqual(kwargs["config"].rank, 4)
        self.assertEqual(kwargs["config"].gradient_steps, 2)
        self.assertEqual(kwargs["config"].target_modules, ("q_proj", "v_proj"))

    def test_provider_uses_state_patch_head_when_present(self):
        toy_model = _ToyCausalModel()
        tokenizer = _ToyTokenizer()
        with mock.patch.object(
            LLMProviderFactory,
            "_load_hf_local_model",
            return_value=(toy_model, tokenizer, "cpu"),
        ):
            head = StatePatchHead(StatePatchHeadConfig(input_dim=16, hidden_dim=8, output_dim=16))
            provider = HFTaskLocalFastWeightsProvider(
                model="stub-model",
                config=TaskLocalFastWeightConfig(
                    rank=2,
                    gradient_steps=1,
                    learning_rate=0.1,
                    target_modules=("q_proj", "v_proj"),
                ),
                state_patch_head=head,
                device="cpu",
            )
            provider.begin_task("demo", "Write code.")
            result = provider.apply_state_descriptor_update(
                task_name="demo",
                update_kind="plan_descriptor",
                latent_state={"verification_passes": 1, "confidence": 0.5},
            )

            self.assertTrue(result.success)
            self.assertEqual(result.kind, "plan_descriptor")

    def test_provider_uses_learned_latent_descriptor_projection_when_present(self):
        class StubLatentDescriptorHead:
            def build_projection(self, **_kwargs):
                return [0.3, -0.1, 0.2, 0.05], {"source": "stub"}

        toy_model = _ToyCausalModel()
        tokenizer = _ToyTokenizer()
        with mock.patch.object(
            LLMProviderFactory,
            "_load_hf_local_model",
            return_value=(toy_model, tokenizer, "cpu"),
        ):
            provider = HFTaskLocalFastWeightsProvider(
                model="stub-model",
                config=TaskLocalFastWeightConfig(
                    rank=2,
                    gradient_steps=1,
                    learning_rate=0.1,
                    target_modules=("q_proj", "v_proj"),
                ),
                latent_descriptor_head=StubLatentDescriptorHead(),
                device="cpu",
            )
            provider.begin_task("demo", "Write code.")
            result = provider.apply_state_descriptor_update(
                task_name="demo",
                update_kind="verify_failure_descriptor",
                latent_state={
                    "verification_passes": 0,
                    "verification_failures": 1,
                    "fast_weight_updates_used": 0,
                    "descriptor_updates_used": 0,
                    "confidence": 0.1,
                    "failure_patterns": ["expected fibonacci recurrence"],
                    "verified_constraints": ["output_format=python"],
                    "recent_actions": ["write", "verify"],
                },
                error_text="expected 8, got 0",
                candidate_text="def fib(n): return 0",
            )

        self.assertTrue(result.success)
        self.assertEqual(result.adaptation_descriptor["source"], "latent_state+learned_descriptor_head")
        self.assertEqual(result.adaptation_descriptor["signal_projection"][:4], [0.3, -0.1, 0.2, 0.05])


if __name__ == "__main__":
    unittest.main()
