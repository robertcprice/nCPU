"""Saved controller deployment bundles for SOME internal-runtime experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Optional


@dataclass
class ControllerComponentConfig:
    """Runtime configuration for one controller component."""

    provider: str
    model: str
    temperature: Optional[float] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    request_timeout: Optional[float] = None
    max_tokens: Optional[int] = None
    provider_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ControllerComponentConfig":
        return cls(
            provider=str(payload["provider"]),
            model=str(payload["model"]),
            temperature=payload.get("temperature"),
            base_url=payload.get("base_url"),
            api_key=payload.get("api_key"),
            request_timeout=payload.get("request_timeout"),
            max_tokens=payload.get("max_tokens"),
            provider_kwargs=dict(payload.get("provider_kwargs") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }
        if not self.provider_kwargs:
            payload.pop("provider_kwargs", None)
        return payload


@dataclass
class ControllerBundle:
    """Portable deployment bundle for the internal controller."""

    name: str
    response: ControllerComponentConfig
    action: Optional[ControllerComponentConfig] = None
    version: int = 1
    base_model: Optional[str] = None
    prepared_dataset_dir: Optional[str] = None
    training_run_path: Optional[str] = None
    latent_action_head_path: Optional[str] = None
    latent_action_head_config: Optional[dict[str, Any]] = None
    latent_memory_head_path: Optional[str] = None
    latent_memory_head_config: Optional[dict[str, Any]] = None
    latent_halt_head_path: Optional[str] = None
    latent_halt_head_config: Optional[dict[str, Any]] = None
    controller_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ControllerBundle":
        return cls(
            name=str(payload["name"]),
            version=int(payload.get("version", 1)),
            base_model=payload.get("base_model"),
            response=ControllerComponentConfig.from_dict(payload["response"]),
            action=(
                ControllerComponentConfig.from_dict(payload["action"])
                if payload.get("action")
                else None
            ),
            prepared_dataset_dir=payload.get("prepared_dataset_dir"),
            training_run_path=payload.get("training_run_path"),
            latent_action_head_path=payload.get("latent_action_head_path"),
            latent_action_head_config=dict(payload.get("latent_action_head_config") or {}) or None,
            latent_memory_head_path=payload.get("latent_memory_head_path"),
            latent_memory_head_config=dict(payload.get("latent_memory_head_config") or {}) or None,
            latent_halt_head_path=payload.get("latent_halt_head_path"),
            latent_halt_head_config=dict(payload.get("latent_halt_head_config") or {}) or None,
            controller_config=dict(payload.get("controller_config") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "version": self.version,
            "response": self.response.to_dict(),
            "metadata": self.metadata,
        }
        if self.base_model is not None:
            payload["base_model"] = self.base_model
        if self.action is not None:
            payload["action"] = self.action.to_dict()
        if self.prepared_dataset_dir is not None:
            payload["prepared_dataset_dir"] = self.prepared_dataset_dir
        if self.training_run_path is not None:
            payload["training_run_path"] = self.training_run_path
        if self.latent_action_head_path is not None:
            payload["latent_action_head_path"] = self.latent_action_head_path
        if self.latent_action_head_config:
            payload["latent_action_head_config"] = dict(self.latent_action_head_config)
        if self.latent_memory_head_path is not None:
            payload["latent_memory_head_path"] = self.latent_memory_head_path
        if self.latent_memory_head_config:
            payload["latent_memory_head_config"] = dict(self.latent_memory_head_config)
        if self.latent_halt_head_path is not None:
            payload["latent_halt_head_path"] = self.latent_halt_head_path
        if self.latent_halt_head_config:
            payload["latent_halt_head_config"] = dict(self.latent_halt_head_config)
        if self.controller_config:
            payload["controller_config"] = dict(self.controller_config)
        return payload


def load_controller_bundle(path: str | Path) -> ControllerBundle:
    """Load a controller bundle manifest from disk."""
    bundle_path = Path(path).expanduser()
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    return ControllerBundle.from_dict(payload)


def save_controller_bundle(bundle: ControllerBundle, path: str | Path) -> Path:
    """Write a controller bundle manifest to disk."""
    bundle_path = Path(path).expanduser()
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(
        json.dumps(bundle.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle_path
