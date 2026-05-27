"""NPCoT array-thought coprocessor layer (N4).

Adds the M2 `ArrayExecutableThoughtHead` and the M3 `ArrayProgramLibrary` as
an optional second expert alongside the existing soft-ALU `NCPUExpert` inside
a `NCPUCoprocessorMLP`-wrapped transformer layer.

At every token, the array-thought coprocessor:

1. Projects `hidden_state[token]` into a fixed-length synthetic array and a
   scalar effective length.
2. Runs the array-thought head on that `(hidden, array, length)` triple. If
   a `program_library` is attached, `consult_library` is used: library hits
   short-circuit to the discrete program, misses fall through to the soft
   forward and auto-cache on convergence.
3. Lifts the scalar output back into `hidden_dim` via a learned projection.
4. Gates the contribution by `sigmoid(hidden_proj) * max_gate`, matching the
   existing coprocessor's safety discipline: a freshly-initialized array
   coprocessor can never move the transformer's output by more than
   `max_gate` worth of perturbation, so it is safe to ship the code path
   even when the array head has not been trained.

The existing `NCPUCoprocessorMLP` is untouched; this module provides a
higher-order wrapper that composes both experts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

from ncpu.coprocessor.config import NCPUCoprocessorConfig
from ncpu.coprocessor.coprocessor_layer import NCPUCoprocessorMLP
from ncpu.self_optimizing.array_executable_thought_head import (
    ArrayExecutableThoughtHead,
    ArrayExecutableThoughtHeadConfig,
)
from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    ArrayThoughtLibraryResult,
)


@dataclass
class ArrayThoughtCoprocessorConfig:
    """Configuration for the NPCoT array-thought coprocessor expert."""

    array_max_len: int = 8
    max_gate: float = 0.05
    convergence_gap_threshold: float = 0.15
    auto_cache: bool = True
    temperature: float = 0.5
    head_config_overrides: dict[str, Any] = field(default_factory=dict)
    # Confidence-gating: if True, the per-token gate is proportional to
    # library-hit indicator (hits contribute, misses are zeroed). This
    # ensures the wrapper never perturbs hidden states when the library
    # doesn't confidently know this problem shape — "first do no harm".
    #
    # When False (legacy), the gate is a learned nn.Linear projection from
    # the hidden state, which can fire even on miss tokens and pollute
    # generation. Empirically the learned-gate path hurts out-of-domain
    # benchmarks (HumanEval vs MBPP-trained library).
    confidence_gate: bool = True
    # Weighting for the confidence gate: 1.0 means gate = hit_indicator
    # (fully on/off); in between uses a blend with the learned gate.
    confidence_weight: float = 1.0


class ArrayThoughtCoprocessor(nn.Module):
    """Optional NPCoT expert: hidden → array program → scalar contribution."""

    def __init__(
        self,
        hidden_dim: int,
        config: Optional[ArrayThoughtCoprocessorConfig] = None,
    ):
        super().__init__()
        self.config = config or ArrayThoughtCoprocessorConfig()
        self.hidden_dim = hidden_dim
        head_config = ArrayExecutableThoughtHeadConfig(
            hidden_dim=hidden_dim,
            array_max_len=self.config.array_max_len,
            **self.config.head_config_overrides,
        )
        self.array_head = ArrayExecutableThoughtHead(head_config)
        self.array_proj = nn.Linear(hidden_dim, self.config.array_max_len)
        self.length_proj = nn.Linear(hidden_dim, 1)
        self.output_proj = nn.Linear(1, hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, 1)

        # Cached lightweight stats for post-hoc inspection.
        self._last_library_hits: Optional[list[bool]] = None
        self._last_newly_cached: Optional[list[bool]] = None
        self._last_convergence_gaps: Optional[list[float]] = None

        # Externally-managed library handle. Setting it enables the fast
        # path. Left as `None` by default so the layer is safe to run in
        # vanilla (non-NPCoT) environments.
        self.program_library: Optional[ArrayProgramLibrary] = None
        self.task_name: Optional[str] = None

    def attach_library(
        self, library: ArrayProgramLibrary, *, task_name: Optional[str] = None
    ) -> None:
        self.program_library = library
        self.task_name = task_name

    def detach_library(self) -> None:
        self.program_library = None
        self.task_name = None

    def _derive_array_inputs(
        self, flat_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        arrays = self.array_proj(flat_hidden)
        length_logits = self.length_proj(flat_hidden).squeeze(-1)
        length_frac = torch.sigmoid(length_logits) * self.config.array_max_len
        lengths = torch.clamp(length_frac, min=1.0)
        return arrays, lengths

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden_states must be rank-3 [B, S, H], got {tuple(hidden_states.shape)}"
            )
        batch, seq_len, _ = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_dim)
        arrays, lengths = self._derive_array_inputs(flat)

        if self.program_library is not None:
            result: ArrayThoughtLibraryResult = self.array_head.consult_library(
                flat,
                arrays,
                self.program_library,
                lengths=lengths,
                temperature=self.config.temperature,
                auto_cache=self.config.auto_cache,
                convergence_gap_threshold=self.config.convergence_gap_threshold,
                task_name=self.task_name,
            )
            scalars = result.predicted_output.unsqueeze(-1)
            self._last_library_hits = list(result.library_hits)
            self._last_newly_cached = list(result.newly_cached)
            self._last_convergence_gaps = list(result.convergence_gaps)
        else:
            soft = self.array_head(
                flat,
                arrays,
                lengths=lengths,
                temperature=self.config.temperature,
            )
            scalars = soft.predicted_output.unsqueeze(-1)
            self._last_library_hits = [False] * scalars.shape[0]
            self._last_newly_cached = [False] * scalars.shape[0]
            self._last_convergence_gaps = [0.0] * scalars.shape[0]

        contribution = self.output_proj(scalars)

        # Confidence gate only applies when a library is attached — without
        # a library, there are no hits to gate on, so fall back to the
        # learned gate. This keeps standalone training / testing working.
        use_confidence = (
            self.config.confidence_gate
            and self.program_library is not None
            and self._last_library_hits is not None
        )
        if use_confidence:
            hit_mask = torch.tensor(
                [1.0 if h else 0.0 for h in self._last_library_hits],
                dtype=contribution.dtype,
                device=contribution.device,
            ).unsqueeze(-1)  # shape (batch*seq, 1)
            if self.config.confidence_weight >= 1.0:
                confidence = hit_mask
            else:
                learned = torch.sigmoid(self.gate_proj(flat))
                w = self.config.confidence_weight
                confidence = w * hit_mask + (1.0 - w) * learned
            gate = confidence * self.config.max_gate
        else:
            gate = torch.sigmoid(self.gate_proj(flat)) * self.config.max_gate

        gated = contribution * gate
        return gated.reshape(batch, seq_len, self.hidden_dim)

    @property
    def last_library_hit_rate(self) -> float:
        if not self._last_library_hits:
            return 0.0
        return sum(self._last_library_hits) / len(self._last_library_hits)

    @property
    def last_newly_cached_count(self) -> int:
        return int(sum(self._last_newly_cached or []))


class NCPUCoprocessorMLPWithArrayThought(nn.Module):
    """Augments `NCPUCoprocessorMLP` with the NPCoT array-thought expert.

    The base layer's behavior is preserved exactly — when
    `array_thought_config is None`, this class is a bit-for-bit passthrough.
    When a configuration is supplied, the array-thought coprocessor is added
    to the MLP output with its own learned gate capped at `max_gate`, so the
    freshly-initialized module never moves the transformer's output by more
    than the gate's cap.
    """

    def __init__(
        self,
        original_mlp: nn.Module,
        hidden_dim: int,
        config: NCPUCoprocessorConfig,
        *,
        array_thought_config: Optional[ArrayThoughtCoprocessorConfig] = None,
    ):
        super().__init__()
        self.base = NCPUCoprocessorMLP(original_mlp, hidden_dim, config)
        self.array_thought: Optional[ArrayThoughtCoprocessor]
        if array_thought_config is None:
            self.array_thought = None
        else:
            self.array_thought = ArrayThoughtCoprocessor(
                hidden_dim, array_thought_config
            )

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        base_output = self.base(hidden_states, **kwargs)
        if self.array_thought is None:
            return base_output
        array_contribution = self.array_thought(hidden_states)
        return base_output + array_contribution

    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        return self.base.aux_loss

    def attach_library(
        self,
        library: ArrayProgramLibrary,
        *,
        task_name: Optional[str] = None,
    ) -> None:
        if self.array_thought is None:
            raise RuntimeError(
                "cannot attach library: array-thought coprocessor not configured"
            )
        self.array_thought.attach_library(library, task_name=task_name)

    def detach_library(self) -> None:
        if self.array_thought is not None:
            self.array_thought.detach_library()


__all__ = [
    "ArrayThoughtCoprocessorConfig",
    "ArrayThoughtCoprocessor",
    "NCPUCoprocessorMLPWithArrayThought",
]
