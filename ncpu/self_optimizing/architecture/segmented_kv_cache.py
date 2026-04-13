"""Segmented cache state for descriptor-backed long-horizon decoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class SegmentDescriptor:
    """Compressed representation of one committed token segment."""

    segment_id: int
    token_ids: list[int]
    token_count: int
    descriptor_token_count: int
    embeddings: torch.Tensor
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def moved(self, *, device: torch.device | str, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        target = self.embeddings.to(device=device)
        if dtype is not None:
            target = target.to(dtype=dtype)
        return target

    def to_summary(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "token_count": self.token_count,
            "descriptor_token_count": self.descriptor_token_count,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


@dataclass
class SegmentedKVCacheConfig:
    """Configuration for segmented-cache decoding."""

    recent_window_tokens: int = 256
    commit_segment_tokens: int = 128
    descriptor_tokens_per_segment: int = 4
    max_memory_segments: int = 16
    min_prompt_tokens_for_compression: int = 384
    min_tokens_to_commit: int = 64


def compress_hidden_segment(
    hidden_states: torch.Tensor,
    *,
    descriptor_tokens: int,
) -> torch.Tensor:
    """Average-pool one hidden-state segment into a compact descriptor sequence."""
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [seq, hidden]")
    if hidden_states.shape[0] == 0:
        raise ValueError("hidden_states must contain at least one token")
    descriptor_tokens = max(1, min(descriptor_tokens, int(hidden_states.shape[0])))
    chunks = torch.chunk(hidden_states, descriptor_tokens, dim=0)
    pooled = [chunk.mean(dim=0, keepdim=True) for chunk in chunks if chunk.shape[0] > 0]
    return torch.cat(pooled, dim=0)


class SegmentedKVCacheState:
    """Mutable segmented cache state for descriptor-backed decoding."""

    def __init__(self, config: Optional[SegmentedKVCacheConfig] = None):
        self.config = config or SegmentedKVCacheConfig()
        self.memory_segments: list[SegmentDescriptor] = []
        self.live_token_ids: list[int] = []
        self.total_prompt_tokens = 0
        self.total_generated_tokens = 0
        self.total_committed_tokens = 0
        self.total_commit_events = 0
        self.total_rebuilds = 0
        self.total_evicted_segments = 0
        self._next_segment_id = 1

    @property
    def memory_token_count(self) -> int:
        return sum(segment.descriptor_token_count for segment in self.memory_segments)

    def append_live_tokens(self, token_ids: list[int]) -> None:
        if token_ids:
            self.live_token_ids.extend(int(token_id) for token_id in token_ids)

    def append_generated_token(self, token_id: int) -> None:
        self.live_token_ids.append(int(token_id))
        self.total_generated_tokens += 1

    def commit_prefix(
        self,
        *,
        token_ids: list[int],
        hidden_states: torch.Tensor,
        reason: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SegmentDescriptor:
        if not token_ids:
            raise ValueError("Cannot commit an empty token segment")
        descriptor = SegmentDescriptor(
            segment_id=self._next_segment_id,
            token_ids=list(token_ids),
            token_count=len(token_ids),
            descriptor_token_count=min(
                self.config.descriptor_tokens_per_segment,
                max(1, len(token_ids)),
            ),
            embeddings=compress_hidden_segment(
                hidden_states.detach().cpu(),
                descriptor_tokens=self.config.descriptor_tokens_per_segment,
            ),
            reason=reason,
            metadata=dict(metadata or {}),
        )
        self._next_segment_id += 1
        self.memory_segments.append(descriptor)
        self.total_commit_events += 1
        self.total_committed_tokens += len(token_ids)
        while len(self.memory_segments) > self.config.max_memory_segments:
            self.memory_segments.pop(0)
            self.total_evicted_segments += 1
        return descriptor

    def drop_live_prefix(self, token_count: int) -> list[int]:
        if token_count <= 0:
            return []
        prefix = self.live_token_ids[:token_count]
        self.live_token_ids = self.live_token_ids[token_count:]
        return prefix

    def build_memory_embeddings(
        self,
        *,
        device: torch.device | str,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[torch.Tensor]:
        if not self.memory_segments:
            return None
        return torch.cat(
            [segment.moved(device=device, dtype=dtype) for segment in self.memory_segments],
            dim=0,
        )

    def record_rebuild(self) -> None:
        self.total_rebuilds += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "live_token_count": len(self.live_token_ids),
            "memory_segment_count": len(self.memory_segments),
            "memory_token_count": self.memory_token_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_generated_tokens": self.total_generated_tokens,
            "total_committed_tokens": self.total_committed_tokens,
            "total_commit_events": self.total_commit_events,
            "total_rebuilds": self.total_rebuilds,
            "total_evicted_segments": self.total_evicted_segments,
            "segments": [segment.to_summary() for segment in self.memory_segments],
        }
