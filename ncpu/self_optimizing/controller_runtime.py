"""Shared runtime resolution for controller response/action model bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ncpu.self_optimizing.controller_bundle import load_controller_bundle


@dataclass
class ResolvedControllerRuntime:
    """Resolved runtime settings for a response model and optional action model."""

    controller_bundle_path: Optional[str]
    provider_name: str
    model: str
    temperature: float
    base_url: str
    request_timeout: Optional[float]
    max_tokens: Optional[int]
    provider_kwargs: dict[str, object]
    action_provider_name: Optional[str]
    action_model: Optional[str]
    action_temperature: Optional[float]
    action_base_url: Optional[str]
    action_request_timeout: Optional[float]
    action_max_tokens: Optional[int]
    action_provider_kwargs: dict[str, object]
    latent_action_head_path: Optional[str]
    latent_action_head_config: dict[str, object]
    latent_memory_head_path: Optional[str]
    latent_memory_head_config: dict[str, object]
    latent_halt_head_path: Optional[str]
    latent_halt_head_config: dict[str, object]


def _require_model(value: Optional[str]) -> str:
    if not value:
        raise ValueError("No response model was provided and no controller bundle response model is available")
    return value


def resolve_controller_runtime(
    *,
    controller_bundle_path: Optional[str] = None,
    provider_name: Optional[str],
    model: Optional[str],
    temperature: Optional[float],
    base_url: Optional[str],
    request_timeout: Optional[float] = None,
    max_tokens: Optional[int] = None,
    provider_kwargs: Optional[dict[str, object]] = None,
    action_provider_name: Optional[str] = None,
    action_model: Optional[str] = None,
    action_temperature: Optional[float] = None,
    action_base_url: Optional[str] = None,
    action_request_timeout: Optional[float] = None,
    action_max_tokens: Optional[int] = None,
    action_provider_kwargs: Optional[dict[str, object]] = None,
    default_provider: str,
    default_temperature: float,
    default_base_url: str,
    default_request_timeout: Optional[float] = None,
) -> ResolvedControllerRuntime:
    """Resolve runtime config from explicit overrides plus an optional bundle."""
    bundle = load_controller_bundle(controller_bundle_path) if controller_bundle_path else None
    response_component = bundle.response if bundle is not None else None
    action_component = bundle.action if bundle is not None else None

    resolved_provider_name = (
        provider_name
        if provider_name is not None
        else (response_component.provider if response_component is not None else default_provider)
    )
    resolved_model = _require_model(
        model if model is not None else (response_component.model if response_component is not None else None)
    )
    resolved_temperature = (
        temperature
        if temperature is not None
        else (
            response_component.temperature
            if response_component is not None and response_component.temperature is not None
            else default_temperature
        )
    )
    resolved_base_url = (
        base_url
        if base_url is not None
        else (
            response_component.base_url
            if response_component is not None and response_component.base_url is not None
            else default_base_url
        )
    )
    resolved_request_timeout = (
        request_timeout
        if request_timeout is not None
        else (
            response_component.request_timeout
            if response_component is not None and response_component.request_timeout is not None
            else default_request_timeout
        )
    )
    resolved_max_tokens = (
        max_tokens
        if max_tokens is not None
        else (
            response_component.max_tokens
            if response_component is not None and response_component.max_tokens is not None
            else None
        )
    )
    resolved_provider_kwargs = dict(response_component.provider_kwargs if response_component is not None else {})
    if provider_kwargs:
        resolved_provider_kwargs.update(provider_kwargs)

    explicit_action_requested = any(
        value is not None
        for value in (
            action_provider_name,
            action_model,
            action_temperature,
            action_base_url,
            action_request_timeout,
            action_max_tokens,
            action_provider_kwargs,
        )
    )
    has_action = explicit_action_requested or action_component is not None

    if has_action:
        resolved_action_provider_name = (
            action_provider_name
            if action_provider_name is not None
            else (
                action_component.provider
                if action_component is not None
                else resolved_provider_name
            )
        )
        resolved_action_model = (
            action_model
            if action_model is not None
            else (
                action_component.model
                if action_component is not None
                else resolved_model
            )
        )
        resolved_action_temperature = (
            action_temperature
            if action_temperature is not None
            else (
                action_component.temperature
                if action_component is not None and action_component.temperature is not None
                else resolved_temperature
            )
        )
        resolved_action_base_url = (
            action_base_url
            if action_base_url is not None
            else (
                action_component.base_url
                if action_component is not None and action_component.base_url is not None
                else resolved_base_url
            )
        )
        resolved_action_request_timeout = (
            action_request_timeout
            if action_request_timeout is not None
            else (
                action_component.request_timeout
                if action_component is not None and action_component.request_timeout is not None
                else resolved_request_timeout
            )
        )
        resolved_action_max_tokens = (
            action_max_tokens
            if action_max_tokens is not None
            else (
                action_component.max_tokens
                if action_component is not None and action_component.max_tokens is not None
                else resolved_max_tokens
            )
        )
        resolved_action_provider_kwargs = dict(action_component.provider_kwargs if action_component is not None else {})
        if action_provider_kwargs:
            resolved_action_provider_kwargs.update(action_provider_kwargs)
    else:
        resolved_action_provider_name = None
        resolved_action_model = None
        resolved_action_temperature = None
        resolved_action_base_url = None
        resolved_action_request_timeout = None
        resolved_action_max_tokens = None
        resolved_action_provider_kwargs = {}

    resolved_latent_action_head_path = None
    resolved_latent_action_head_config: dict[str, object] = {}
    resolved_latent_memory_head_path = None
    resolved_latent_memory_head_config: dict[str, object] = {}
    resolved_latent_halt_head_path = None
    resolved_latent_halt_head_config: dict[str, object] = {}
    if bundle is not None and bundle.latent_action_head_path:
        bundle_path = Path(controller_bundle_path).expanduser() if controller_bundle_path else None
        latent_path = Path(bundle.latent_action_head_path).expanduser()
        if not latent_path.is_absolute() and bundle_path is not None:
            latent_path = (bundle_path.parent / latent_path).resolve()
        resolved_latent_action_head_path = str(latent_path)
        resolved_latent_action_head_config = dict(bundle.latent_action_head_config or {})
    if bundle is not None and bundle.latent_memory_head_path:
        bundle_path = Path(controller_bundle_path).expanduser() if controller_bundle_path else None
        latent_path = Path(bundle.latent_memory_head_path).expanduser()
        if not latent_path.is_absolute() and bundle_path is not None:
            latent_path = (bundle_path.parent / latent_path).resolve()
        resolved_latent_memory_head_path = str(latent_path)
        resolved_latent_memory_head_config = dict(bundle.latent_memory_head_config or {})
    if bundle is not None and bundle.latent_halt_head_path:
        bundle_path = Path(controller_bundle_path).expanduser() if controller_bundle_path else None
        latent_path = Path(bundle.latent_halt_head_path).expanduser()
        if not latent_path.is_absolute() and bundle_path is not None:
            latent_path = (bundle_path.parent / latent_path).resolve()
        resolved_latent_halt_head_path = str(latent_path)
        resolved_latent_halt_head_config = dict(bundle.latent_halt_head_config or {})

    return ResolvedControllerRuntime(
        controller_bundle_path=controller_bundle_path,
        provider_name=resolved_provider_name,
        model=resolved_model,
        temperature=float(resolved_temperature),
        base_url=resolved_base_url,
        request_timeout=resolved_request_timeout,
        max_tokens=resolved_max_tokens,
        provider_kwargs=resolved_provider_kwargs,
        action_provider_name=resolved_action_provider_name,
        action_model=resolved_action_model,
        action_temperature=resolved_action_temperature,
        action_base_url=resolved_action_base_url,
        action_request_timeout=resolved_action_request_timeout,
        action_max_tokens=resolved_action_max_tokens,
        action_provider_kwargs=resolved_action_provider_kwargs,
        latent_action_head_path=resolved_latent_action_head_path,
        latent_action_head_config=resolved_latent_action_head_config,
        latent_memory_head_path=resolved_latent_memory_head_path,
        latent_memory_head_config=resolved_latent_memory_head_config,
        latent_halt_head_path=resolved_latent_halt_head_path,
        latent_halt_head_config=resolved_latent_halt_head_config,
    )


def load_bundle_latent_action_policy(
    *,
    controller_bundle_path: Optional[str],
    device: str = "cpu",
):
    """Load a latent action policy from a controller bundle, if configured."""
    if not controller_bundle_path:
        return None

    runtime = resolve_controller_runtime(
        controller_bundle_path=controller_bundle_path,
        provider_name=None,
        model=None,
        temperature=None,
        base_url=None,
        default_provider="local",
        default_temperature=0.0,
        default_base_url="http://localhost:11434",
        default_request_timeout=None,
    )
    if not runtime.latent_action_head_path:
        return None

    from ncpu.self_optimizing.latent_action_policy import (
        LatentActionHeadConfig,
        LatentActionPolicy,
        load_latent_action_head,
    )

    config = None
    if runtime.latent_action_head_config:
        config_payload = dict(runtime.latent_action_head_config)
        if "action_labels" in config_payload:
            config_payload["action_labels"] = tuple(config_payload["action_labels"])
        config = LatentActionHeadConfig(**config_payload)
    head = load_latent_action_head(
        path=runtime.latent_action_head_path,
        device=device,
        config=config,
    )
    return LatentActionPolicy(head=head, device=device, config=head.config)


def load_bundle_latent_halt_policy(
    *,
    controller_bundle_path: Optional[str],
    device: str = "cpu",
):
    """Load a latent halt policy from a controller bundle, if configured."""
    if not controller_bundle_path:
        return None

    runtime = resolve_controller_runtime(
        controller_bundle_path=controller_bundle_path,
        provider_name=None,
        model=None,
        temperature=None,
        base_url=None,
        default_provider="local",
        default_temperature=0.0,
        default_base_url="http://localhost:11434",
        default_request_timeout=None,
    )
    if not runtime.latent_halt_head_path:
        return None

    from ncpu.self_optimizing.latent_halt_policy import (
        LatentHaltHeadConfig,
        LatentHaltPolicy,
        load_latent_halt_head,
    )

    config = None
    if runtime.latent_halt_head_config:
        config_payload = dict(runtime.latent_halt_head_config)
        if "action_labels" in config_payload:
            config_payload["action_labels"] = tuple(config_payload["action_labels"])
        config = LatentHaltHeadConfig(**config_payload)
    head = load_latent_halt_head(
        path=runtime.latent_halt_head_path,
        device=device,
        config=config,
    )
    return LatentHaltPolicy(head=head, device=device, config=head.config)


def load_bundle_latent_memory_updater(
    *,
    controller_bundle_path: Optional[str],
    device: str = "cpu",
):
    """Load a latent memory updater from a controller bundle, if configured."""
    if not controller_bundle_path:
        return None

    runtime = resolve_controller_runtime(
        controller_bundle_path=controller_bundle_path,
        provider_name=None,
        model=None,
        temperature=None,
        base_url=None,
        default_provider="local",
        default_temperature=0.0,
        default_base_url="http://localhost:11434",
        default_request_timeout=None,
    )
    if not runtime.latent_memory_head_path:
        return None

    from ncpu.self_optimizing.latent_memory_head import (
        LatentMemoryHeadConfig,
        LatentMemoryUpdater,
        load_latent_memory_head,
    )

    config = None
    if runtime.latent_memory_head_config:
        config = LatentMemoryHeadConfig(**dict(runtime.latent_memory_head_config))
    head = load_latent_memory_head(
        path=runtime.latent_memory_head_path,
        device=device,
        config=config,
    )
    return LatentMemoryUpdater(head=head, device=device, config=head.config)
