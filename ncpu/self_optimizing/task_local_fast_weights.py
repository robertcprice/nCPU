"""Task-local fast-weight adapters for hidden SOME inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time
from typing import Any, Optional

import torch
from torch import nn

from ncpu.self_optimizing.ncpu_adaptation_backend import (
    NCPUAdaptationBackend,
    NCPUAdaptationDescriptor,
)
from ncpu.self_optimizing.latent_descriptor_head import LatentDescriptorGenerator
from ncpu.self_optimizing.state_patch_head import StatePatchHead


@dataclass
class TaskLocalFastWeightConfig:
    """Configuration for small task-local weight updates during inference."""

    rank: int = 8
    learning_rate: float = 5e-3
    gradient_steps: int = 1
    adapter_scale: float = 1.0
    max_target_tokens: int = 256
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    weight_decay: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FastWeightUpdateResult:
    """Normalized result of one task-local fast-weight update."""

    success: bool
    kind: str
    updated_modules: list[str] = field(default_factory=list)
    task_name: Optional[str] = None
    loss: Optional[float] = None
    steps: int = 0
    target_tokens: int = 0
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    adaptation_descriptor: Optional[dict[str, Any]] = None
    implementation: str = "task_local_low_rank_residual"

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "kind": self.kind,
            "updated_modules": list(self.updated_modules),
            "task_name": self.task_name,
            "loss": self.loss,
            "steps": self.steps,
            "target_tokens": self.target_tokens,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
            "adaptation_descriptor": dict(self.adaptation_descriptor or {}) or None,
            "implementation": self.implementation,
        }


class FastWeightLinear(nn.Module):
    """Frozen base linear layer plus a task-local low-rank residual."""

    def __init__(self, base_linear: nn.Linear, *, rank: int, adapter_scale: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("Fast-weight rank must be positive")

        self.base_linear = base_linear
        for parameter in self.base_linear.parameters():
            parameter.requires_grad = False

        self.fast_a = nn.Linear(base_linear.in_features, rank, bias=False)
        self.fast_b = nn.Linear(rank, base_linear.out_features, bias=False)
        self.adapter_scale = float(adapter_scale) / float(rank)
        self.reset_fast_weights()

    def reset_fast_weights(self) -> None:
        nn.init.kaiming_uniform_(self.fast_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fast_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base_linear(inputs)
        residual = self.fast_b(self.fast_a(inputs)) * self.adapter_scale
        return base_output + residual

    def fast_parameters(self) -> list[nn.Parameter]:
        return [self.fast_a.weight, self.fast_b.weight]


def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parts = module_name.split(".")
    parent: Any = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    final = parts[-1]
    if final.isdigit():
        parent[int(final)] = new_module
    else:
        setattr(parent, final, new_module)


def find_target_linear_modules(
    model: nn.Module,
    *,
    target_fragments: tuple[str, ...],
) -> list[tuple[str, nn.Linear]]:
    """Return linear submodules whose names match one of the configured fragments."""
    matches: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        if any(fragment in name for fragment in target_fragments):
            matches.append((name, module))
    return matches


class HFTaskLocalFastWeightsProvider:
    """
    Hugging Face local provider with task-local low-rank test-time updates.

    The wrapped base model stays frozen. Only the injected low-rank residuals are
    updated during hidden SOME deliberation, then reset at task end.
    """

    def __init__(
        self,
        *,
        model: str,
        config: Optional[TaskLocalFastWeightConfig] = None,
        adaptation_backend: Optional[NCPUAdaptationBackend] = None,
        latent_descriptor_head: Optional[LatentDescriptorGenerator] = None,
        state_patch_head: Optional[StatePatchHead] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ):
        from ncpu.self_optimizing.llm_provider import LLMProviderFactory

        self.model_name = model
        self.config = config or TaskLocalFastWeightConfig()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.adaptation_backend = adaptation_backend or NCPUAdaptationBackend()
        self.latent_descriptor_head = latent_descriptor_head
        self.state_patch_head = state_patch_head
        self.decode_runtime: Optional[Any] = None

        loaded_model, tokenizer, resolved_device = LLMProviderFactory._load_hf_local_model(
            model,
            device=device,
            trust_remote_code=trust_remote_code,
            use_cache=False,
        )
        self.model = loaded_model
        self.tokenizer = tokenizer
        self.device = resolved_device
        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self._adapter_names: list[str] = []
        self._adapters: list[FastWeightLinear] = []
        self._install_fast_weight_adapters()
        if not self._adapters:
            target_list = ", ".join(self.config.target_modules)
            raise ValueError(
                f"No target linear modules matched fast-weight fragments: {target_list}"
            )

        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._active_task_name: Optional[str] = None
        self._update_count = 0

    def _install_fast_weight_adapters(self) -> None:
        matches = find_target_linear_modules(
            self.model,
            target_fragments=self.config.target_modules,
        )
        for module_name, module in matches:
            wrapped = FastWeightLinear(
                module,
                rank=self.config.rank,
                adapter_scale=self.config.adapter_scale,
            ).to(self.device)
            _replace_module(self.model, module_name, wrapped)
            self._adapter_names.append(module_name)
            self._adapters.append(wrapped)

    def _reset_fast_weights(self) -> None:
        for adapter in self._adapters:
            adapter.reset_fast_weights()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        parameters = [
            parameter
            for adapter in self._adapters
            for parameter in adapter.fast_parameters()
        ]
        return torch.optim.SGD(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _set_optimizer_learning_rate(self, learning_rate: float) -> None:
        if self._optimizer is None:
            return
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = float(learning_rate)

    def _tokenize(self, text: str, *, add_special_tokens: bool = False) -> dict[str, torch.Tensor]:
        try:
            batch = self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=add_special_tokens,
            )
        except TypeError:
            batch = self.tokenizer(text, return_tensors="pt")
        if hasattr(batch, "to"):
            batch = batch.to(self.device)
        return dict(batch)

    def _collect_fast_gradients(self) -> torch.Tensor:
        gradients: list[torch.Tensor] = []
        for adapter in self._adapters:
            for parameter in adapter.fast_parameters():
                if parameter.grad is not None:
                    gradients.append(parameter.grad.detach().reshape(-1))
        if not gradients:
            return torch.zeros(0, device=self.device)
        return torch.cat(gradients)

    def _apply_projection_to_adapters(
        self,
        projection: list[float],
        *,
        learning_rate_scale: float,
        update_kind: str,
    ) -> None:
        if not projection:
            return
        transformed_projection = projection
        if self.state_patch_head is not None:
            transformed_projection = self.state_patch_head.transform(
                projection,
                device=self.device,
            )
        signal = torch.tensor(transformed_projection, dtype=torch.float32, device=self.device)
        norm = torch.linalg.norm(signal)
        if float(norm.item()) > 0:
            signal = signal / norm
        kind_scale = {
            "plan_descriptor": 0.75,
            "verify_failure_descriptor": 1.0,
            "repair_plan_descriptor": 0.9,
        }.get(update_kind, 0.8)
        scale = self.config.learning_rate * learning_rate_scale * kind_scale
        for adapter_index, adapter in enumerate(self._adapters):
            fast_a = adapter.fast_a.weight.data
            fast_b = adapter.fast_b.weight.data

            repeated_a = signal.repeat(math.ceil(fast_a.numel() / signal.numel()))[: fast_a.numel()]
            repeated_b = torch.flip(signal, dims=[0]).repeat(math.ceil(fast_b.numel() / signal.numel()))[: fast_b.numel()]

            fast_a.add_(repeated_a.view_as(fast_a) * scale)
            fast_b.add_(repeated_b.view_as(fast_b) * scale * (1.0 + 0.05 * adapter_index))

    def begin_task(self, task_name: str, task_prompt: str = "") -> dict[str, Any]:
        self._active_task_name = task_name
        self._update_count = 0
        self._reset_fast_weights()
        self._optimizer = self._build_optimizer()
        metadata = {
            "enabled": True,
            "task_name": task_name,
            "task_prompt_length": len(task_prompt),
            "update_count": self._update_count,
            "config": self.config.to_dict(),
            "updated_modules": list(self._adapter_names),
            "implementation": "task_local_low_rank_residual",
        }
        if self.adaptation_backend is not None:
            metadata["ncpu_adaptation"] = self.adaptation_backend.begin_task(task_name, task_prompt)
        return metadata

    def apply_self_update(
        self,
        *,
        prompt: str,
        target_text: str,
        update_kind: str = "plan",
        task_name: Optional[str] = None,
    ) -> FastWeightUpdateResult:
        if not target_text.strip():
            return FastWeightUpdateResult(
                success=False,
                kind=update_kind,
                updated_modules=list(self._adapter_names),
                task_name=task_name or self._active_task_name,
                error="empty target_text",
            )

        if self._optimizer is None:
            self.begin_task(task_name or self._active_task_name or "implicit")

        started = time.perf_counter()
        adaptation_descriptor: Optional[NCPUAdaptationDescriptor] = None
        try:
            prompt_inputs = self._tokenize(prompt, add_special_tokens=False)
            target_inputs = self._tokenize(target_text, add_special_tokens=False)
            prompt_ids = prompt_inputs["input_ids"]
            target_ids = target_inputs["input_ids"]
            if target_ids.shape[-1] > self.config.max_target_tokens:
                target_ids = target_ids[:, : self.config.max_target_tokens]

            input_ids = torch.cat([prompt_ids, target_ids], dim=-1)
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
            labels = input_ids.clone()
            labels[:, : prompt_ids.shape[-1]] = -100

            self.model.train()
            last_loss: Optional[float] = None
            assert self._optimizer is not None
            self._set_optimizer_learning_rate(self.config.learning_rate)
            effective_steps = self.config.gradient_steps
            step_index = 0
            while step_index < effective_steps:
                self._optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                if step_index == 0 and self.adaptation_backend is not None:
                    gradient_vector = self._collect_fast_gradients().detach().cpu().numpy()
                    adaptation_descriptor = self.adaptation_backend.build_update_descriptor(
                        task_name=task_name or self._active_task_name or "implicit",
                        update_kind=update_kind,
                        gradients=gradient_vector,
                    )
                    effective_steps = max(effective_steps, adaptation_descriptor.suggested_gradient_steps)
                    self._set_optimizer_learning_rate(
                        self.config.learning_rate * adaptation_descriptor.learning_rate_scale
                    )
                self._optimizer.step()
                last_loss = float(loss.detach().cpu().item())
                step_index += 1
            self.model.eval()
            self._set_optimizer_learning_rate(self.config.learning_rate)
            self._update_count += 1
            elapsed = time.perf_counter() - started
            if adaptation_descriptor is not None and self.adaptation_backend is not None:
                self.adaptation_backend.record_update_result(adaptation_descriptor, success=True)
            return FastWeightUpdateResult(
                success=True,
                kind=update_kind,
                updated_modules=list(self._adapter_names),
                task_name=task_name or self._active_task_name,
                loss=last_loss,
                steps=effective_steps,
                target_tokens=int(target_ids.shape[-1]),
                elapsed_seconds=elapsed,
                adaptation_descriptor=(
                    adaptation_descriptor.to_dict() if adaptation_descriptor is not None else None
                ),
            )
        except Exception as exc:
            self.model.eval()
            self._set_optimizer_learning_rate(self.config.learning_rate)
            elapsed = time.perf_counter() - started
            if adaptation_descriptor is not None and self.adaptation_backend is not None:
                self.adaptation_backend.record_update_result(adaptation_descriptor, success=False)
            return FastWeightUpdateResult(
                success=False,
                kind=update_kind,
                updated_modules=list(self._adapter_names),
                task_name=task_name or self._active_task_name,
                elapsed_seconds=elapsed,
                error=f"{type(exc).__name__}: {exc}",
                adaptation_descriptor=(
                    adaptation_descriptor.to_dict() if adaptation_descriptor is not None else None
                ),
            )

    def apply_state_descriptor_update(
        self,
        *,
        task_name: str,
        update_kind: str,
        latent_state: Any,
        error_text: str = "",
        candidate_text: str = "",
    ) -> FastWeightUpdateResult:
        if self._optimizer is None:
            self.begin_task(task_name, candidate_text or error_text)

        started = time.perf_counter()
        descriptor = self.adaptation_backend.build_state_descriptor(
            task_name=task_name,
            update_kind=update_kind,
            latent_state=latent_state,
            error_text=error_text,
            candidate_text=candidate_text,
        )
        if self.latent_descriptor_head is not None:
            projection, feature_summary = self.latent_descriptor_head.build_projection(
                latent_state=latent_state,
                update_kind=update_kind,
                error_text=error_text,
                candidate_text=candidate_text,
            )
            descriptor.signal_projection = projection
            descriptor.source = "latent_state+learned_descriptor_head"
            descriptor.implementation = "ncpu_gradient_protocol+learned_descriptor_head"
            descriptor.descriptor["feature_summary"] = feature_summary
        try:
            self._apply_projection_to_adapters(
                descriptor.signal_projection,
                learning_rate_scale=descriptor.learning_rate_scale,
                update_kind=update_kind,
            )
            self._update_count += 1
            elapsed = time.perf_counter() - started
            self.adaptation_backend.record_update_result(descriptor, success=True)
            return FastWeightUpdateResult(
                success=True,
                kind=update_kind,
                updated_modules=list(self._adapter_names),
                task_name=task_name or self._active_task_name,
                steps=1,
                elapsed_seconds=elapsed,
                adaptation_descriptor=descriptor.to_dict(),
                implementation="task_local_low_rank_residual+descriptor_update",
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            self.adaptation_backend.record_update_result(descriptor, success=False)
            return FastWeightUpdateResult(
                success=False,
                kind=update_kind,
                updated_modules=list(self._adapter_names),
                task_name=task_name or self._active_task_name,
                elapsed_seconds=elapsed,
                error=f"{type(exc).__name__}: {exc}",
                adaptation_descriptor=descriptor.to_dict(),
                implementation="task_local_low_rank_residual+descriptor_update",
            )

    def end_task(self) -> dict[str, Any]:
        self._reset_fast_weights()
        self._optimizer = None
        summary = {
            "task_name": self._active_task_name,
            "update_count": self._update_count,
            "updated_modules": list(self._adapter_names),
        }
        if self.adaptation_backend is not None:
            summary["ncpu_adaptation"] = self.adaptation_backend.end_task()
        self._active_task_name = None
        self._update_count = 0
        return summary

    def __call__(self, prompt: str) -> dict[str, Any]:
        started = time.perf_counter()
        if self.decode_runtime is not None:
            result = dict(self.decode_runtime.generate(prompt))
            result.update(
                {
                    "fast_weight_updates": self._update_count,
                    "fast_weight_modules": list(self._adapter_names),
                    "active_task_name": self._active_task_name,
                }
            )
            return result
        inputs = self._tokenize(prompt)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_tokens,
            "do_sample": self.temperature > 0,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        prompt_tokens = int(inputs["input_ids"].shape[-1])
        generated_ids = outputs[0][prompt_tokens:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        elapsed = time.perf_counter() - started
        return {
            "text": text,
            "eval_count": int(generated_ids.shape[-1]),
            "prompt_eval_count": prompt_tokens,
            "total_duration": int(elapsed * 1_000_000_000),
            "device": self.device,
            "fast_weight_updates": self._update_count,
            "fast_weight_modules": list(self._adapter_names),
            "active_task_name": self._active_task_name,
        }
