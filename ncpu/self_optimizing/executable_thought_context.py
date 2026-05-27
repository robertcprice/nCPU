"""Shared prompt and hidden-state helpers for executable-thought training/runtime."""

from __future__ import annotations

import re
from typing import Any, Optional

import torch


def normalize_latent_state_payload(latent_state: Any) -> dict[str, Any]:
    if hasattr(latent_state, "to_dict"):
        return dict(latent_state.to_dict())
    if isinstance(latent_state, dict):
        return dict(latent_state)
    return dict(latent_state or {})


def latent_state_summary(latent_state: Any) -> str:
    if hasattr(latent_state, "to_prompt_summary"):
        return str(latent_state.to_prompt_summary())
    state = normalize_latent_state_payload(latent_state)
    lines = [
        f"confidence={float(state.get('confidence', 0.0) or 0.0):.3f}",
        "verification="
        f"{int(state.get('verification_passes', 0) or 0)} pass / "
        f"{int(state.get('verification_failures', 0) or 0)} fail",
    ]
    failure_patterns = list(state.get("failure_patterns") or [])
    verified_constraints = list(state.get("verified_constraints") or [])
    recent_actions = list(state.get("recent_actions") or [])
    if failure_patterns:
        lines.append("failure_patterns: " + "; ".join(str(item) for item in failure_patterns[-4:]))
    if verified_constraints:
        lines.append("verified_constraints: " + "; ".join(str(item) for item in verified_constraints[-4:]))
    if recent_actions:
        lines.append("recent_actions: " + ", ".join(str(item) for item in recent_actions[-4:]))
    return "\n".join(lines)


def build_executable_thought_prompt(
    *,
    task_name: str,
    update_kind: str,
    latent_state: Any,
    error_text: str,
    candidate_text: str,
) -> str:
    return (
        f"Task: {task_name}\n"
        f"Update kind: {update_kind}\n"
        "Latent state summary:\n"
        f"{latent_state_summary(latent_state)}\n\n"
        f"Error text:\n{str(error_text or '')[:512] or '(none)'}\n\n"
        f"Candidate text:\n{str(candidate_text or '')[:512] or '(none)'}\n"
    )


def extract_numeric_hints(text: str, *, limit: int) -> list[float]:
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", str(text or ""))
    hints: list[float] = []
    for token in matches:
        try:
            hints.append(float(token))
        except ValueError:
            continue
        if len(hints) >= limit:
            break
    return hints


def build_executable_register_values(
    *,
    latent_state: Any,
    error_text: str,
    candidate_text: str,
    num_registers: int,
) -> list[float]:
    state = normalize_latent_state_payload(latent_state)
    values: list[float] = []
    values.extend(extract_numeric_hints(error_text, limit=2))
    values.extend(extract_numeric_hints(candidate_text, limit=1))
    values.extend(
        [
            float(state.get("confidence", 0.0) or 0.0),
            float(state.get("verification_failures", 0) or 0),
            float(state.get("verification_passes", 0) or 0),
            float(state.get("descriptor_updates_used", 0) or 0),
            float(state.get("fast_weight_updates_used", 0) or 0),
        ]
    )
    memory_vector = list(state.get("memory_vector") or [])
    if memory_vector:
        values.extend(float(value) for value in memory_vector[:4])
    if len(values) < num_registers:
        values.extend([0.0] * (num_registers - len(values)))
    return values[:num_registers]


def build_executable_register_inputs(
    *,
    latent_state: Any,
    error_text: str,
    candidate_text: str,
    num_registers: int,
    device: Optional[str | torch.device] = None,
) -> torch.Tensor:
    values = build_executable_register_values(
        latent_state=latent_state,
        error_text=error_text,
        candidate_text=candidate_text,
        num_registers=num_registers,
    )
    return torch.tensor(
        [values],
        dtype=torch.float32,
        device=device,
    )


def tokenize_executable_thought_prompt(
    *,
    tokenizer: Any,
    text: str,
    device: str | torch.device,
    add_special_tokens: bool = False,
) -> dict[str, Any]:
    try:
        batch = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
    except TypeError:
        batch = tokenizer(text, return_tensors="pt")
    if hasattr(batch, "to"):
        batch = batch.to(device)
    return dict(batch)


def extract_hidden_state_from_prompt(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: str | torch.device,
    max_tokens: int,
    add_special_tokens: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    tokenized = tokenize_executable_thought_prompt(
        tokenizer=tokenizer,
        text=prompt,
        device=device,
        add_special_tokens=add_special_tokens,
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized.get("attention_mask")
    if input_ids.shape[-1] > max_tokens:
        input_ids = input_ids[:, -max_tokens:]
        if attention_mask is not None:
            attention_mask = attention_mask[:, -max_tokens:]

    forward_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "output_hidden_states": True,
        "use_cache": False,
    }
    if attention_mask is not None:
        forward_kwargs["attention_mask"] = attention_mask

    with torch.no_grad():
        try:
            outputs = model(**forward_kwargs)
        except TypeError:
            fallback_kwargs = {"input_ids": input_ids}
            if attention_mask is not None:
                fallback_kwargs["attention_mask"] = attention_mask
            outputs = model(**fallback_kwargs)

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states:
        last_hidden = hidden_states[-1]
        hidden_source = "model_hidden_states"
    else:
        last_hidden = getattr(outputs, "last_hidden_state", None)
        hidden_source = "model_last_hidden_state"
    if last_hidden is None:
        embedding_layer = model.get_input_embeddings()
        last_hidden = embedding_layer(input_ids)
        hidden_source = "input_embeddings"

    hidden_state = last_hidden[:, -1, :]
    metadata = {
        "hidden_state_source": hidden_source,
        "prompt_token_count": int(input_ids.shape[-1]),
    }
    return hidden_state, metadata


__all__ = [
    "normalize_latent_state_payload",
    "latent_state_summary",
    "build_executable_thought_prompt",
    "extract_numeric_hints",
    "build_executable_register_values",
    "build_executable_register_inputs",
    "tokenize_executable_thought_prompt",
    "extract_hidden_state_from_prompt",
]
