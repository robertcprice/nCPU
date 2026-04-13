"""nCPU-backed gradient adaptation for inference-time fast-weight updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
from typing import Any, Optional

import numpy as np

try:
    from ncpu.os.gpu.protocols.gradient_aware_network import CompressionType, GradientCompressor
except (ImportError, ModuleNotFoundError):
    # Fallback when deployed without the full ncpu/os tree (e.g. remote training servers)
    from enum import Enum

    class CompressionType(Enum):
        NONE = "none"
        QUANTIZATION = "quantization"
        TOP_K = "top_k"
        RANDOM_K = "random_k"
        DWAL = "dwal"
        POWER_SGD = "power_sgd"
        AUTO = "auto"

    class _GradientDescriptor:
        __slots__ = (
            "name", "shape", "dtype", "size_bytes", "compression",
            "compression_ratio", "nonzero_indices", "nonzero_values",
            "scale", "zero_point",
        )
        def __init__(self, **kw: Any):
            for k in self.__slots__:
                setattr(self, k, kw.get(k))

    class GradientCompressor:
        def __init__(self, compression_type: CompressionType = CompressionType.AUTO):
            self.compression_type = compression_type

        def compress(self, gradient: np.ndarray, spec: dict) -> "_GradientDescriptor":
            if self.compression_type == CompressionType.TOP_K:
                flat = gradient.flatten()
                k = max(1, int(len(flat) * spec.get("k_ratio", 0.01)))
                indices = np.argpartition(np.abs(flat), -k)[-k:]
                values = flat[indices]
                return _GradientDescriptor(
                    name=spec.get("name", "grad"), shape=gradient.shape,
                    dtype=0, size_bytes=indices.nbytes + values.nbytes,
                    compression=CompressionType.TOP_K,
                    compression_ratio=gradient.nbytes / (indices.nbytes + values.nbytes),
                    nonzero_indices=indices, nonzero_values=values,
                )
            if self.compression_type == CompressionType.QUANTIZATION:
                bits = spec.get("bits", 8)
                mx = gradient.max()
                scale = mx / (2 ** (bits - 1)) if mx else 1.0
                quantized = np.round(gradient / scale).astype(np.int8)
                return _GradientDescriptor(
                    name=spec.get("name", "grad"), shape=gradient.shape,
                    dtype=5, size_bytes=quantized.nbytes,
                    compression=CompressionType.QUANTIZATION,
                    compression_ratio=gradient.nbytes / quantized.nbytes,
                    scale=float(scale), zero_point=0,
                )
            return _GradientDescriptor(
                name=spec.get("name", "grad"), shape=gradient.shape,
                dtype=0, size_bytes=gradient.nbytes,
                compression=CompressionType.NONE, compression_ratio=1.0,
            )


@dataclass
class NCPUAdaptationConfig:
    """Configuration for routing fast-weight updates through nCPU gradient protocols."""

    compression_type: str = CompressionType.TOP_K.value
    top_k_ratio: float = 0.1
    quantization_bits: int = 8
    gradient_clip: float = 1.0
    min_learning_rate_scale: float = 0.75
    max_learning_rate_scale: float = 1.5
    max_gradient_steps: int = 3
    verify_failure_boost: float = 1.2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolved_compression_type(self) -> CompressionType:
        value = str(self.compression_type).lower()
        for candidate in CompressionType:
            if candidate.value == value:
                return candidate
        raise ValueError(f"Unsupported nCPU adaptation compression type: {self.compression_type}")


@dataclass
class NCPUAdaptationDescriptor:
    """Serializable summary of one nCPU-mediated update decision."""

    task_name: str
    update_kind: str
    vector_size: int
    gradient_norm: float
    clipped_gradient_norm: float
    learning_rate_scale: float
    suggested_gradient_steps: int
    descriptor: dict[str, Any]
    signal_projection: list[float] = field(default_factory=list)
    source: str = "gradient"
    implementation: str = "ncpu_gradient_protocol"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NCPUAdaptationSession:
    """Task-local summary of nCPU-mediated adaptation activity."""

    task_name: str
    update_count: int = 0
    successful_updates: int = 0
    descriptors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NCPUAdaptationBackend:
    """
    Use the repo's gradient-aware nCPU protocol to summarize and steer fast-weight updates.

    This is not full arbitrary CPU execution inside dense weights. It is the concrete next
    step: actual fast-weight gradients are converted into nCPU-native gradient descriptors,
    and those descriptors steer within-task learning-rate and step budgeting.
    """

    def __init__(self, config: Optional[NCPUAdaptationConfig] = None):
        self.config = config or NCPUAdaptationConfig()
        self.compressor = GradientCompressor(self.config.resolved_compression_type())
        self._session: Optional[NCPUAdaptationSession] = None

    def begin_task(self, task_name: str, task_prompt: str = "") -> dict[str, Any]:
        self._session = NCPUAdaptationSession(task_name=task_name)
        return {
            "enabled": True,
            "task_name": task_name,
            "task_prompt_length": len(task_prompt),
            "config": self.config.to_dict(),
            "implementation": "ncpu_gradient_protocol",
        }

    def _clip_vector(self, vector: np.ndarray) -> tuple[np.ndarray, float, float]:
        raw_norm = float(np.linalg.norm(vector))
        if self.config.gradient_clip <= 0 or raw_norm <= self.config.gradient_clip:
            return vector, raw_norm, raw_norm
        scale = self.config.gradient_clip / max(raw_norm, 1e-8)
        clipped = vector * scale
        return clipped, raw_norm, float(np.linalg.norm(clipped))

    def _descriptor_to_dict(self, descriptor: Any) -> dict[str, Any]:
        nonzero_entries = 0
        if getattr(descriptor, "nonzero_indices", None) is not None:
            nonzero_entries = int(len(descriptor.nonzero_indices))
        elif getattr(descriptor, "nonzero_values", None) is not None:
            nonzero_entries = int(len(descriptor.nonzero_values))

        return {
            "name": str(getattr(descriptor, "name", "")),
            "shape": list(getattr(descriptor, "shape", ()) or ()),
            "dtype": int(getattr(descriptor, "dtype", 0)),
            "size_bytes": int(getattr(descriptor, "size_bytes", 0) or 0),
            "compression": getattr(getattr(descriptor, "compression", None), "value", str(getattr(descriptor, "compression", ""))),
            "compression_ratio": float(getattr(descriptor, "compression_ratio", 1.0) or 1.0),
            "nonzero_entries": nonzero_entries,
            "scale": getattr(descriptor, "scale", None),
            "zero_point": getattr(descriptor, "zero_point", None),
        }

    def _suggest_learning_rate_scale(self, update_kind: str, *, vector_size: int, clipped_norm: float) -> float:
        denom = max(1.0, math.sqrt(float(vector_size)))
        norm_factor = min(clipped_norm / denom, 1.0)
        kind_scale = {
            "plan": 0.85,
            "repair_plan": 1.0,
            "verify_failure": self.config.verify_failure_boost,
            "verified_candidate": 0.8,
        }.get(update_kind, 1.0)
        lr_scale = kind_scale * (0.85 + 0.65 * norm_factor)
        return float(
            min(
                self.config.max_learning_rate_scale,
                max(self.config.min_learning_rate_scale, lr_scale),
            )
        )

    def _suggest_gradient_steps(self, update_kind: str, *, clipped_norm: float) -> int:
        steps = 1
        if update_kind in {"repair_plan", "verify_failure"}:
            steps += 1
        if self.config.gradient_clip > 0 and clipped_norm >= self.config.gradient_clip * 0.75:
            steps += 1
        return int(max(1, min(self.config.max_gradient_steps, steps)))

    def _signal_projection(self, vector: np.ndarray, *, projection_dim: int = 16) -> list[float]:
        if vector.size == 0:
            return []
        projection_dim = max(1, min(projection_dim, int(vector.size)))
        chunks = np.array_split(vector.astype(np.float32), projection_dim)
        projection = [float(chunk.mean()) if chunk.size else 0.0 for chunk in chunks]
        return projection

    def _text_features(self, text: Any, *, dim: int = 16) -> np.ndarray:
        vector = np.zeros(dim, dtype=np.float32)
        normalized = " ".join(str(text or "").strip().split()).lower()
        if not normalized:
            return vector
        for token in normalized.split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % dim
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            magnitude = 0.5 + (digest[3] / 255.0)
            vector[index] += sign * magnitude
        return vector

    def _state_vector(
        self,
        *,
        latent_state: Any,
        error_text: str = "",
        candidate_text: str = "",
    ) -> np.ndarray:
        state = latent_state.to_dict() if hasattr(latent_state, "to_dict") else dict(latent_state or {})
        numeric = np.array(
            [
                float(state.get("verification_passes", 0)),
                float(state.get("verification_failures", 0)),
                float(state.get("fast_weight_updates_used", 0)),
                float(state.get("confidence", 0.0)),
                float(len(state.get("failure_patterns") or [])),
                float(len(state.get("verified_constraints") or [])),
            ],
            dtype=np.float32,
        )
        text_sections = [
            state.get("active_strategy", ""),
            state.get("hidden_plan", ""),
            state.get("repair_plan", ""),
            state.get("last_candidate_digest", ""),
            state.get("last_failure_summary", ""),
            " ".join(state.get("failure_patterns") or []),
            " ".join(state.get("verified_constraints") or []),
            " ".join(state.get("recent_actions") or []),
            error_text,
            candidate_text,
        ]
        text_vector = np.concatenate([self._text_features(section, dim=8) for section in text_sections])
        return np.concatenate([numeric, text_vector]).astype(np.float32)

    def build_update_descriptor(
        self,
        *,
        task_name: str,
        update_kind: str,
        gradients: np.ndarray,
        source: str = "gradient",
    ) -> NCPUAdaptationDescriptor:
        flat = np.asarray(gradients, dtype=np.float32).reshape(-1)
        if flat.size == 0:
            return NCPUAdaptationDescriptor(
                task_name=task_name,
                update_kind=update_kind,
                vector_size=0,
                gradient_norm=0.0,
                clipped_gradient_norm=0.0,
                learning_rate_scale=1.0,
                suggested_gradient_steps=1,
                descriptor={
                    "name": f"{task_name}:{update_kind}",
                    "shape": [0],
                    "dtype": 0,
                    "size_bytes": 0,
                    "compression": self.config.compression_type,
                    "compression_ratio": 1.0,
                    "nonzero_entries": 0,
                    "scale": None,
                    "zero_point": None,
                },
                signal_projection=[],
                source=source,
            )

        clipped, raw_norm, clipped_norm = self._clip_vector(flat)
        spec: dict[str, Any] = {"name": f"{task_name}:{update_kind}"}
        compression = self.config.resolved_compression_type()
        if compression == CompressionType.TOP_K:
            spec["k_ratio"] = self.config.top_k_ratio
        elif compression == CompressionType.QUANTIZATION:
            spec["bits"] = self.config.quantization_bits

        gradient_descriptor = self.compressor.compress(clipped.astype(np.float32), spec)
        descriptor_dict = self._descriptor_to_dict(gradient_descriptor)
        adaptation = NCPUAdaptationDescriptor(
            task_name=task_name,
            update_kind=update_kind,
            vector_size=int(flat.size),
            gradient_norm=raw_norm,
            clipped_gradient_norm=clipped_norm,
            learning_rate_scale=self._suggest_learning_rate_scale(
                update_kind,
                vector_size=int(flat.size),
                clipped_norm=clipped_norm,
            ),
            suggested_gradient_steps=self._suggest_gradient_steps(
                update_kind,
                clipped_norm=clipped_norm,
            ),
            descriptor=descriptor_dict,
            signal_projection=self._signal_projection(clipped),
            source=source,
        )
        if self._session is not None:
            self._session.update_count += 1
            self._session.descriptors.append(adaptation.to_dict())
        return adaptation

    def build_state_descriptor(
        self,
        *,
        task_name: str,
        update_kind: str,
        latent_state: Any,
        error_text: str = "",
        candidate_text: str = "",
    ) -> NCPUAdaptationDescriptor:
        return self.build_update_descriptor(
            task_name=task_name,
            update_kind=update_kind,
            gradients=self._state_vector(
                latent_state=latent_state,
                error_text=error_text,
                candidate_text=candidate_text,
            ),
            source="latent_state",
        )

    def record_update_result(self, descriptor: NCPUAdaptationDescriptor, *, success: bool) -> None:
        if self._session is None:
            return
        if success:
            self._session.successful_updates += 1
        if self._session.descriptors:
            self._session.descriptors[-1]["applied_successfully"] = bool(success)
            self._session.descriptors[-1]["final_learning_rate_scale"] = descriptor.learning_rate_scale
            self._session.descriptors[-1]["final_suggested_gradient_steps"] = descriptor.suggested_gradient_steps

    def end_task(self) -> dict[str, Any]:
        session = self._session or NCPUAdaptationSession(task_name="unknown")
        summary = session.to_dict()
        summary["implementation"] = "ncpu_gradient_protocol"
        self._session = None
        return summary
