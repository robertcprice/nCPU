"""Descriptor-backed segmented-cache decoding for long hidden SOME inference."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
from typing import Any, Optional

import torch

from ncpu.self_optimizing.recurrent_commit_policy import (
    RecurrentCommitDecision,
    RecurrentCommitPolicy,
    RecurrentCommitPolicyConfig,
)
from ncpu.self_optimizing.segmented_kv_cache import (
    SegmentedKVCacheConfig,
    SegmentedKVCacheState,
)


@dataclass
class DescriptorDecodeConfig:
    """Configuration for descriptor-backed segmented decoding."""

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    cache: SegmentedKVCacheConfig = field(default_factory=SegmentedKVCacheConfig)
    commit_policy: RecurrentCommitPolicyConfig = field(default_factory=RecurrentCommitPolicyConfig)


class DescriptorDecodeRuntime:
    """Custom decode runtime with compressed committed history and a recent live window."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        device: str,
        config: Optional[DescriptorDecodeConfig] = None,
        commit_policy: Optional[RecurrentCommitPolicy] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or DescriptorDecodeConfig()
        self.commit_policy = commit_policy or RecurrentCommitPolicy(self.config.commit_policy)

    def generate(self, prompt: str) -> dict[str, Any]:
        started = time.perf_counter()
        prompt_ids = self._tokenize(prompt)
        cache_state = self._build_initial_cache_state(prompt_ids)
        past_key_values, logits = self._prefill_from_cache(cache_state)

        generated_ids: list[int] = []
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        while len(generated_ids) < self.config.max_new_tokens:
            next_token = self._sample_next_token(logits, generated_ids=generated_ids)
            token_id = int(next_token.item())
            generated_ids.append(token_id)
            cache_state.append_generated_token(token_id)
            if eos_token_id is not None and token_id == eos_token_id:
                break

            decision = self.commit_policy.decide(cache_state)
            if decision.should_commit:
                past_key_values, logits = self._commit_and_rebuild(
                    cache_state=cache_state,
                    decision=decision,
                )
                continue

            outputs = self._forward(
                input_ids=next_token.view(1, 1),
                past_key_values=past_key_values,
                output_hidden_states=False,
            )
            past_key_values = getattr(outputs, "past_key_values", None)
            logits = getattr(outputs, "logits")

        elapsed = time.perf_counter() - started
        text = self._strip_code_fences(
            self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        )
        metadata = cache_state.snapshot()
        metadata.update(
            {
                "text": text,
                "eval_count": len(generated_ids),
                "prompt_eval_count": len(prompt_ids),
                "total_duration": int(elapsed * 1_000_000_000),
                "device": self.device,
                "cache_backend": "segmented_kv",
            }
        )
        return metadata

    def _build_initial_cache_state(self, prompt_ids: list[int]) -> SegmentedKVCacheState:
        cache_state = SegmentedKVCacheState(self.config.cache)
        cache_state.total_prompt_tokens = len(prompt_ids)
        if (
            len(prompt_ids) < self.config.cache.min_prompt_tokens_for_compression
            or len(prompt_ids) <= self.config.cache.recent_window_tokens
        ):
            cache_state.append_live_tokens(prompt_ids)
            return cache_state

        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        outputs = self._forward(input_ids=prompt_tensor, output_hidden_states=True, use_cache=False)
        hidden_states = self._last_hidden_states(outputs)
        prefix_len = max(0, len(prompt_ids) - self.config.cache.recent_window_tokens)
        cursor = 0
        while cursor < prefix_len:
            chunk_end = min(cursor + self.config.cache.commit_segment_tokens, prefix_len)
            cache_state.commit_prefix(
                token_ids=prompt_ids[cursor:chunk_end],
                hidden_states=hidden_states[cursor:chunk_end],
                reason="prompt_prefill",
            )
            cursor = chunk_end
        cache_state.append_live_tokens(prompt_ids[prefix_len:])
        return cache_state

    def _prefill_from_cache(self, cache_state: SegmentedKVCacheState) -> tuple[Any, torch.Tensor]:
        context_embeds = self._build_context_embeddings(cache_state)
        outputs = self._forward(inputs_embeds=context_embeds, output_hidden_states=True, use_cache=True)
        return getattr(outputs, "past_key_values", None), getattr(outputs, "logits")

    def _commit_and_rebuild(
        self,
        *,
        cache_state: SegmentedKVCacheState,
        decision: RecurrentCommitDecision,
    ) -> tuple[Any, torch.Tensor]:
        context_embeds = self._build_context_embeddings(cache_state)
        outputs = self._forward(inputs_embeds=context_embeds, output_hidden_states=True, use_cache=False)
        hidden_states = self._last_hidden_states(outputs)
        memory_token_count = cache_state.memory_token_count
        live_hidden_states = hidden_states[memory_token_count:]

        commit_count = min(decision.commit_token_count, len(cache_state.live_token_ids))
        committed_token_ids = cache_state.drop_live_prefix(commit_count)
        cache_state.commit_prefix(
            token_ids=committed_token_ids,
            hidden_states=live_hidden_states[:commit_count],
            reason=decision.reason,
            metadata={"commit_count": commit_count},
        )
        cache_state.record_rebuild()
        return self._prefill_from_cache(cache_state)

    def _build_context_embeddings(self, cache_state: SegmentedKVCacheState) -> torch.Tensor:
        embedding_layer = self.model.get_input_embeddings()
        dtype = embedding_layer.weight.dtype
        recent_ids = cache_state.live_token_ids
        if not recent_ids:
            fallback_id = getattr(self.tokenizer, "bos_token_id", None)
            if fallback_id is None:
                fallback_id = getattr(self.tokenizer, "eos_token_id", None)
            if fallback_id is None:
                fallback_id = 0
            recent_ids = [int(fallback_id)]

        recent_tensor = torch.tensor([recent_ids], dtype=torch.long, device=self.device)
        recent_embeds = embedding_layer(recent_tensor)
        memory_embeds = cache_state.build_memory_embeddings(device=self.device, dtype=dtype)
        if memory_embeds is None:
            return recent_embeds
        return torch.cat([memory_embeds.unsqueeze(0), recent_embeds], dim=1)

    def _tokenize(self, prompt: str) -> list[int]:
        try:
            encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        except TypeError:
            encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        if hasattr(input_ids, "tolist"):
            values = input_ids[0].tolist()
        else:
            values = list(input_ids[0])
        return [int(value) for value in values]

    def _forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Any = None,
        output_hidden_states: bool,
        use_cache: bool = True,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "use_cache": use_cache,
            "output_hidden_states": output_hidden_states,
        }
        if input_ids is not None:
            kwargs["input_ids"] = input_ids
        if inputs_embeds is not None:
            kwargs["inputs_embeds"] = inputs_embeds
            kwargs["attention_mask"] = torch.ones(
                inputs_embeds.shape[:2],
                dtype=torch.long,
                device=inputs_embeds.device,
            )
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        with torch.no_grad():
            return self.model(**kwargs)

    def _last_hidden_states(self, outputs: Any) -> torch.Tensor:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            last_hidden = hidden_states[-1]
        else:
            last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            raise ValueError("Model outputs did not include hidden states for descriptor compression")
        if last_hidden.ndim != 3:
            raise ValueError("Expected hidden states with shape [batch, seq, hidden]")
        return last_hidden[0]

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        *,
        generated_ids: list[int],
    ) -> torch.Tensor:
        token_logits = logits[:, -1, :].clone()
        if self.config.repetition_penalty != 1.0 and generated_ids:
            for token_id in set(generated_ids):
                token_logits[:, token_id] /= self.config.repetition_penalty
        if self.config.temperature <= 0:
            return token_logits.argmax(dim=-1)

        token_logits = token_logits / max(self.config.temperature, 1e-5)
        probabilities = torch.softmax(token_logits, dim=-1)
        if self.config.top_p < 1.0:
            probabilities = self._apply_top_p(probabilities, self.config.top_p)
        return torch.multinomial(probabilities, num_samples=1).squeeze(-1)

    def _apply_top_p(self, probabilities: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
        renormalized = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        filtered = torch.zeros_like(probabilities)
        filtered.scatter_(dim=-1, index=sorted_indices, src=renormalized)
        return filtered

    def _strip_code_fences(self, text: str) -> str:
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        return text.strip()
