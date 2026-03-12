"""Commit policy for segmented-cache hidden decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ncpu.self_optimizing.segmented_kv_cache import SegmentedKVCacheState


@dataclass
class RecurrentCommitPolicyConfig:
    """Heuristics for deciding when to compress live tokens into descriptors."""

    recent_window_tokens: int = 256
    commit_segment_tokens: int = 128
    min_tokens_to_commit: int = 64


@dataclass
class RecurrentCommitDecision:
    """One commit decision for the segmented cache."""

    should_commit: bool
    commit_token_count: int = 0
    reason: str = ""


class RecurrentCommitPolicy:
    """Keep only a recent token window live and commit older tokens into memory."""

    def __init__(self, config: Optional[RecurrentCommitPolicyConfig] = None):
        self.config = config or RecurrentCommitPolicyConfig()

    def decide(self, cache_state: SegmentedKVCacheState) -> RecurrentCommitDecision:
        overflow = len(cache_state.live_token_ids) - self.config.recent_window_tokens
        if overflow < self.config.min_tokens_to_commit:
            return RecurrentCommitDecision(False)

        segment = self.config.commit_segment_tokens
        commit_count = (overflow // segment) * segment if segment > 0 else overflow
        if commit_count <= 0:
            commit_count = overflow
        if commit_count < self.config.min_tokens_to_commit:
            return RecurrentCommitDecision(False)

        return RecurrentCommitDecision(
            should_commit=True,
            commit_token_count=commit_count,
            reason="live_window_overflow",
        )
