"""Array-valued executable thought head (NPCoT milestone M2).

Extends the scalar `ExecutableThoughtHead` to array-valued inputs. The hidden
state still drives every program choice, but the program is now a
differentiable reduction over an input array rather than a short scalar
register sequence:

    acc = init
    for i in range(max_len):
        active = sigmoid((length - i - 0.5) / 0.3)
        f_i = sum(transform_w * [x_i, x_i^2, |x_i|, 1, 1{x_i>0}])
        new_acc = soft_reduce(acc, f_i, reduce_w)       # +, *, max, min
        acc = acc + active * (new_acc - acc)
    result = post_scale_w . [acc, acc/max(length,1)] + post_offset

All choices (init, element transform, reduction op, post-scaling, offset) are
predicted from the hidden state by a single linear projection, then softmaxed
at the caller-controlled temperature. This keeps the structure compact while
giving the transformer direct gradient-level control over the program.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ncpu.self_optimizing.latent_heads.state_patch_head import (
    StatePatchHead,
    StatePatchHeadConfig,
)


# Init choices are APPEND-ONLY: indices 0-2 are frozen so existing library
# JSONs and trained checkpoints keep their meaning. Index 3 (`+large`) was
# added to unlock the min/argmin reduction family — a min-reduce needs an
# accumulator that starts *above* every realistic element, which `0`/`1`/
# `-large` could never provide.
_INIT_CHOICES = ("0", "1", "-large", "+large")
_ELEM_TRANSFORMS = ("x", "x*x", "|x|", "1", "1{x>0}", "log|x|")
_REDUCE_OPS = ("+", "*", "max", "min")
_POST_SCALES = ("acc", "acc/len", "exp(acc)")
# Epsilon guard for log(|x| + eps) so x=0 gives a finite (negative) value.
_LOG_EPS = 1e-6
# Init sentinel for `max` reductions. Must be below any realistic array
# element, but small enough that partial softmax weight doesn't destabilize
# the initial loss. -20 covers value ranges commonly seen in smoke training.
_NEG_LARGE = -20.0
# Init sentinel for `min` reductions — the positive-infinity proxy mirroring
# `_NEG_LARGE`. Must sit *above* every realistic element so a min-fold starts
# high enough to be pulled down to the true minimum. Same magnitude as
# `_NEG_LARGE` (positive) so the two sentinels are symmetric.
_POS_LARGE = 20.0


@dataclass
class ArrayExecutableThoughtHeadConfig:
    """Configuration for the M2 array-reduction executable thought head."""

    hidden_dim: int
    array_max_len: int = 8
    trace_projection_dim: int = 16
    trace_hidden_dim: int = 64
    state_patch_dim: int = 16
    temperature: float = 1.0
    hidden_update_scale: float = 1.0
    # bias strengths: apply soft priors on sensible defaults so the head
    # isn't stuck exactly at uniform when hidden_state=0.
    init_prior_zero: float = 2.0
    transform_prior_x: float = 2.0
    reduce_prior_sum: float = 2.0
    # The `+large` init (idx 3) is a min-reduce sentinel. Like `-large`, mixing
    # it into the soft accumulator at uniform weight destabilizes the soft
    # forward (a +20 starting acc explodes the `*` / `exp(acc)` paths). A
    # negative prior keeps it logit-suppressed by default — it only earns
    # weight when the hidden state actively drives a min reduction — which also
    # makes the new init "start un-preferred" for backward compatibility. -6.0
    # was tuned so the standard sum/max/count curriculum is unperturbed (the
    # +20 sentinel stays out of the soft accumulator), while a hidden state
    # that wants a min reduction can still drive `+large` to win — the prior is
    # only an additive bias a sufficiently strong logit overrides.
    init_prior_pos_large: float = -6.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArrayExecutableThoughtResult:
    """Output of one array-thought forward pass."""

    predicted_output: torch.Tensor
    next_hidden_state: torch.Tensor
    trace_projection: torch.Tensor
    patch_signal: torch.Tensor
    program_texts: list[str] = field(default_factory=list)
    init_probs: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    transform_probs: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    reduce_probs: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    post_scale_probs: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    post_offsets: torch.Tensor = field(default_factory=lambda: torch.empty(0))


@dataclass
class ArrayExecutableThoughtSmokeMetrics:
    """Summary from a tiny convergence run."""

    initial_loss: float
    final_loss: float
    final_mae: float
    loss_history: list[float] = field(default_factory=list)
    final_program_texts: list[str] = field(default_factory=list)


_PARAM_SIZES = {
    "init": len(_INIT_CHOICES),
    "transform": len(_ELEM_TRANSFORMS),
    "reduce": len(_REDUCE_OPS),
    "post_scale": len(_POST_SCALES),
    "post_offset": 1,
}
_TOTAL_PARAMS = sum(_PARAM_SIZES.values())


def _param_slice(name: str) -> tuple[int, int]:
    start = 0
    for key, width in _PARAM_SIZES.items():
        if key == name:
            return start, start + width
        start += width
    raise KeyError(name)


# Number of init choices a *legacy* (pre-`+large`) checkpoint was trained with.
# Index ordering inside `_TOTAL_PARAMS` is init, transform, reduce, post_scale,
# post_offset, so a 3-init checkpoint packs 3 + 6 + 4 + 3 + 1 = 17 rows into
# `param_projector.{weight,bias}` and `_param_prior`.
_LEGACY_N_INIT = 3
_LEGACY_TOTAL_PARAMS = (
    _LEGACY_N_INIT
    + len(_ELEM_TRANSFORMS)
    + len(_REDUCE_OPS)
    + len(_POST_SCALES)
    + 1  # post_offset
)


def upgrade_state_dict_for_init_expansion(
    state_dict: dict[str, torch.Tensor],
    *,
    prior_pos_large: float = -6.0,
) -> tuple[dict[str, torch.Tensor], bool]:
    """Upgrade a legacy (3-init) array-thought state dict to the current space.

    The init enumeration grew from 3 (`0`, `1`, `-large`) to 4 by appending
    `+large`. Because `init` is the *first* slot in `_TOTAL_PARAMS`, that change
    shifts every downstream slice (transform/reduce/post_scale/post_offset) by
    one row inside ``param_projector.weight`` (shape ``(_TOTAL_PARAMS,
    hidden_dim)``), ``param_projector.bias`` and ``_param_prior`` (both shape
    ``(_TOTAL_PARAMS,)``), and lengthens ``_init_values`` from 3 to 4.

    A plain ``load_state_dict`` of a legacy checkpoint into a 4-init head fails
    on those shape mismatches. This helper produces an upgraded copy that:

    * preserves the 3 existing init rows (indices 0-2) byte-for-byte,
    * inserts a fresh `+large` init row at index 3 whose **projector weight and
      bias are zero** — the new init contributes nothing of its own to the
      hidden-state projection, so the legacy model's behavior is unchanged,
    * inserts the **negative `init_prior_pos_large` prior** for the new init in
      ``_param_prior`` (default -6.0) so `+large` is logit-suppressed by
      default and only earns softmax mass once the hidden state actively
      drives a min reduction — i.e. it "starts un-preferred",
    * leaves transform/reduce/post_scale/post_offset rows identical (just
      re-indexed),
    * extends ``_init_values`` to ``[0, 1, _NEG_LARGE, _POS_LARGE]``.

    Returns ``(upgraded_state_dict, upgraded)`` where ``upgraded`` is True iff a
    legacy layout was detected and rewritten. A state dict already at the
    current dimension is returned unchanged with ``upgraded=False``.
    """
    weight_key = "param_projector.weight"
    if weight_key not in state_dict:
        return dict(state_dict), False

    legacy_rows = int(state_dict[weight_key].shape[0])
    if legacy_rows == _TOTAL_PARAMS:
        return dict(state_dict), False
    if legacy_rows != _LEGACY_TOTAL_PARAMS:
        raise ValueError(
            "cannot upgrade array-thought state dict: param_projector has "
            f"{legacy_rows} rows, expected {_LEGACY_TOTAL_PARAMS} (legacy) or "
            f"{_TOTAL_PARAMS} (current)"
        )

    # Insert position = right after the legacy init rows (0.._LEGACY_N_INIT-1).
    insert_at = _LEGACY_N_INIT
    upgraded: dict[str, torch.Tensor] = dict(state_dict)

    def _insert_row(tensor: torch.Tensor, fill: float) -> torch.Tensor:
        # tensor: (_LEGACY_TOTAL_PARAMS, ...) → (_TOTAL_PARAMS, ...) with a row
        # spliced in at `insert_at`, filled with `fill`.
        row_shape = (1,) + tuple(tensor.shape[1:])
        row = torch.full(row_shape, fill, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor[:insert_at], row, tensor[insert_at:]], dim=0)

    # Learned projection rows are zero — the new init contributes nothing of
    # its own to the hidden-state projection (logit-neutral learned part).
    upgraded[weight_key] = _insert_row(state_dict[weight_key], 0.0)
    bias_key = "param_projector.bias"
    if bias_key in state_dict:
        upgraded[bias_key] = _insert_row(state_dict[bias_key], 0.0)
    # The prior row is the negative `init_prior_pos_large` (default -6.0) so the
    # new init starts logit-suppressed — same value a fresh 4-init head sets.
    prior_key = "_param_prior"
    if prior_key in state_dict:
        upgraded[prior_key] = _insert_row(
            state_dict[prior_key], float(prior_pos_large)
        )
    init_values_key = "_init_values"
    if init_values_key in state_dict:
        legacy_vals = state_dict[init_values_key]
        upgraded[init_values_key] = torch.tensor(
            [0.0, 1.0, _NEG_LARGE, _POS_LARGE],
            dtype=legacy_vals.dtype,
            device=legacy_vals.device,
        )
    return upgraded, True


class ArrayExecutableThoughtHead(nn.Module):
    """Hidden state -> soft array reduction -> hidden-state patch."""

    def __init__(
        self,
        config: ArrayExecutableThoughtHeadConfig,
        *,
        state_patch_head: Optional[StatePatchHead] = None,
    ):
        super().__init__()
        self.config = config
        self.param_projector = nn.Linear(config.hidden_dim, _TOTAL_PARAMS)
        with torch.no_grad():
            nn.init.normal_(self.param_projector.weight, std=1.0)
            nn.init.zeros_(self.param_projector.bias)

        trace_feature_dim = 5  # result, length_ratio, acc_abs, init_entropy, reduce_entropy
        self.trace_encoder = nn.Sequential(
            nn.Linear(trace_feature_dim, config.trace_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.trace_hidden_dim, config.trace_projection_dim),
        )
        self.state_patch_head = state_patch_head or StatePatchHead(
            StatePatchHeadConfig(
                input_dim=config.trace_projection_dim,
                hidden_dim=config.trace_hidden_dim,
                output_dim=config.state_patch_dim,
            )
        )
        self.hidden_patch_projector = nn.Linear(
            self.state_patch_head.config.output_dim, config.hidden_dim
        )
        self.hidden_update_gate = nn.Linear(config.hidden_dim, 1)

        prior = torch.zeros(_TOTAL_PARAMS)
        init_start, _ = _param_slice("init")
        transform_start, _ = _param_slice("transform")
        reduce_start, _ = _param_slice("reduce")
        prior[init_start + 0] = config.init_prior_zero
        # `+large` init (idx 3): negative prior so it starts logit-suppressed
        # and only earns mass when the hidden state drives a min reduction.
        if len(_INIT_CHOICES) > 3:
            prior[init_start + 3] = config.init_prior_pos_large
        prior[transform_start + 0] = config.transform_prior_x
        prior[reduce_start + 0] = config.reduce_prior_sum
        self.register_buffer("_param_prior", prior)

        self._init_values = torch.tensor(
            [0.0, 1.0, _NEG_LARGE, _POS_LARGE], dtype=torch.float32
        )

    def _coerce(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if hidden_state.ndim == 1:
            return hidden_state.unsqueeze(0), True
        if hidden_state.ndim != 2:
            raise ValueError(
                f"hidden_state must be rank-1 or rank-2, got shape {tuple(hidden_state.shape)}"
            )
        return hidden_state, False

    def _resolve_array_inputs(
        self,
        array_inputs: torch.Tensor,
        lengths: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if array_inputs.ndim != 2:
            raise ValueError(
                f"array_inputs must be rank-2 (batch, max_len), got shape {tuple(array_inputs.shape)}"
            )
        if array_inputs.shape[0] != batch_size:
            raise ValueError(
                f"array_inputs batch mismatch: expected {batch_size}, got {array_inputs.shape[0]}"
            )
        if array_inputs.shape[1] != self.config.array_max_len:
            raise ValueError(
                "array_inputs width mismatch: "
                f"expected {self.config.array_max_len}, got {array_inputs.shape[1]}"
            )
        # Preserve caller dtype for bf16 / fp16 compatibility when embedded
        # in a transformer forward pass. Fall back to float32 only for
        # integer-typed inputs (e.g. raw array indices).
        target_dtype = (
            array_inputs.dtype if array_inputs.is_floating_point() else torch.float32
        )
        array_inputs = array_inputs.to(device=device, dtype=target_dtype)
        if lengths is None:
            lengths_tensor = torch.full(
                (batch_size,),
                float(self.config.array_max_len),
                dtype=target_dtype,
                device=device,
            )
        else:
            lengths_tensor = lengths.to(device=device, dtype=target_dtype)
            if lengths_tensor.shape != (batch_size,):
                raise ValueError(
                    f"lengths must be shape ({batch_size},), got {tuple(lengths_tensor.shape)}"
                )
        return array_inputs, lengths_tensor

    def _program_distributions(
        self,
        hidden_state: torch.Tensor,
        temperature: float,
    ) -> dict[str, torch.Tensor]:
        raw = self.param_projector(hidden_state) + self._param_prior
        tau = max(float(temperature), 1e-3)
        out: dict[str, torch.Tensor] = {}
        for name in ("init", "transform", "reduce", "post_scale"):
            start, stop = _param_slice(name)
            out[name] = F.softmax(raw[:, start:stop] / tau, dim=-1)
        offset_start, offset_stop = _param_slice("post_offset")
        out["post_offset"] = raw[:, offset_start:offset_stop].squeeze(-1)
        return out

    @staticmethod
    def _soft_reduce(acc: torch.Tensor, f_i: torch.Tensor, reduce_w: torch.Tensor) -> torch.Tensor:
        plus = acc + f_i
        mult = acc * f_i
        diff = (acc - f_i).abs()
        soft_max = 0.5 * (acc + f_i + diff)
        soft_min = 0.5 * (acc + f_i - diff)
        candidates = torch.stack([plus, mult, soft_max, soft_min], dim=-1)
        return (reduce_w * candidates).sum(dim=-1)

    def _execute_batched(
        self,
        array_inputs: torch.Tensor,
        lengths: torch.Tensor,
        distributions: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        batch_size = array_inputs.shape[0]
        device = array_inputs.device
        init_values = self._init_values.to(device=device, dtype=array_inputs.dtype)
        acc = (distributions["init"] * init_values.unsqueeze(0)).sum(dim=-1)

        transform_w = distributions["transform"]
        reduce_w = distributions["reduce"]

        for i in range(self.config.array_max_len):
            x_i = array_inputs[:, i]
            exprs = torch.stack(
                [
                    x_i,
                    x_i * x_i,
                    x_i.abs(),
                    torch.ones_like(x_i),
                    torch.sigmoid(x_i / 0.25),
                    torch.log(x_i.abs() + _LOG_EPS),
                ],
                dim=-1,
            )
            f_i = (transform_w * exprs).sum(dim=-1)
            new_acc = self._soft_reduce(acc, f_i, reduce_w)
            remaining = lengths - float(i) - 0.5
            active = torch.sigmoid(remaining / 0.3)
            acc = acc + active * (new_acc - acc)

        post_scale_w = distributions["post_scale"]
        denom = torch.clamp(lengths, min=1.0)
        # `exp(acc)` is the numerically stable product-recovery path: combine
        # with transform=log|x| + reduce=+ to realize abs(product) without
        # overflow on large |values| * large L. The exp input is clamped to
        # keep gradients tractable during training.
        exp_acc = torch.exp(torch.clamp(acc, min=-30.0, max=30.0))
        post_candidates = torch.stack([acc, acc / denom, exp_acc], dim=-1)
        post_value = (post_scale_w * post_candidates).sum(dim=-1)
        return post_value + distributions["post_offset"]

    def _trace_features(
        self,
        result: torch.Tensor,
        lengths: torch.Tensor,
        distributions: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        max_len = float(max(self.config.array_max_len, 1))
        length_ratio = lengths / max_len
        acc_abs = result.abs()
        eps = 1e-8
        init_entropy = -(distributions["init"] * torch.log(distributions["init"] + eps)).sum(dim=-1)
        reduce_entropy = -(distributions["reduce"] * torch.log(distributions["reduce"] + eps)).sum(dim=-1)
        return torch.stack([result, length_ratio, acc_abs, init_entropy, reduce_entropy], dim=-1)

    @staticmethod
    def _render_program(distributions: dict[str, torch.Tensor], batch_index: int) -> str:
        init_idx = int(torch.argmax(distributions["init"][batch_index]).item())
        trans_idx = int(torch.argmax(distributions["transform"][batch_index]).item())
        red_idx = int(torch.argmax(distributions["reduce"][batch_index]).item())
        post_idx = int(torch.argmax(distributions["post_scale"][batch_index]).item())
        offset = float(distributions["post_offset"][batch_index].item())

        init_label = _INIT_CHOICES[init_idx]
        trans_label = _ELEM_TRANSFORMS[trans_idx].replace("x", "arr[i]")
        reduce_label = _REDUCE_OPS[red_idx]
        post_label = _POST_SCALES[post_idx]

        body_line = {
            "+": f"acc += {trans_label}",
            "*": f"acc *= {trans_label}",
            "max": f"acc = max(acc, {trans_label})",
            "min": f"acc = min(acc, {trans_label})",
        }[reduce_label]

        offset_str = f" + {offset:.3f}" if abs(offset) > 1e-3 else ""
        return_expr = "acc" if post_label == "acc" else "acc / max(len(arr), 1)"

        return (
            "fn array_thought(arr: &[i64]) -> f64 {\n"
            f"    let mut acc: f64 = {init_label};\n"
            "    for i in 0..arr.len() {\n"
            f"        {body_line};\n"
            "    }\n"
            f"    return {return_expr}{offset_str};\n"
            "}"
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        array_inputs: torch.Tensor,
        *,
        lengths: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
    ) -> ArrayExecutableThoughtResult:
        hidden_batch, squeezed = self._coerce(hidden_state)
        batch_size = hidden_batch.shape[0]
        device = hidden_batch.device
        resolved_temperature = float(
            self.config.temperature if temperature is None else temperature
        )
        array_batch, length_batch = self._resolve_array_inputs(
            array_inputs,
            lengths,
            batch_size=batch_size,
            device=device,
        )

        distributions = self._program_distributions(hidden_batch, resolved_temperature)
        result = self._execute_batched(array_batch, length_batch, distributions)

        trace_features = self._trace_features(result, length_batch, distributions)
        trace_projection = self.trace_encoder(trace_features)
        patch_signal = self.state_patch_head(trace_projection)
        update_gate = torch.sigmoid(self.hidden_update_gate(hidden_batch))
        hidden_delta = (
            self.hidden_patch_projector(patch_signal)
            * update_gate
            * self.config.hidden_update_scale
        )
        next_hidden = hidden_batch + hidden_delta

        program_texts = [self._render_program(distributions, idx) for idx in range(batch_size)]

        def _maybe_squeeze(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.squeeze(0) if squeezed else tensor

        return ArrayExecutableThoughtResult(
            predicted_output=_maybe_squeeze(result),
            next_hidden_state=_maybe_squeeze(next_hidden),
            trace_projection=_maybe_squeeze(trace_projection),
            patch_signal=_maybe_squeeze(patch_signal),
            program_texts=program_texts,
            init_probs=_maybe_squeeze(distributions["init"]),
            transform_probs=_maybe_squeeze(distributions["transform"]),
            reduce_probs=_maybe_squeeze(distributions["reduce"]),
            post_scale_probs=_maybe_squeeze(distributions["post_scale"]),
            post_offsets=_maybe_squeeze(distributions["post_offset"]),
        )

    def consult_library(
        self,
        hidden_state: torch.Tensor,
        array_inputs: torch.Tensor,
        library: Any,
        *,
        lengths: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        auto_cache: bool = True,
        convergence_gap_threshold: float = 0.15,
        task_name: Optional[str] = None,
    ) -> Any:
        """Library-aware forward pass (NPCoT milestone M3).

        For every sample, first ask `library.lookup(hidden_state_i)` for a
        cached discrete program. If every sample is a library hit, skip the
        soft forward entirely — the library becomes a pure fast path. If any
        sample is a miss, run the normal soft forward. For miss samples, take
        the argmax (discrete) program, measure its output gap vs the soft
        output, and — when `auto_cache` is true and the gap is below
        `convergence_gap_threshold` — cache the discrete program for next time.

        Returns an `ArrayThoughtLibraryResult` carrying the predicted output,
        the discrete programs in force, per-sample library-hit flags, and the
        convergence gaps that drove the caching decision.
        """
        from ncpu.self_optimizing.array_program_library import (
            ArrayThoughtLibraryResult,
            DiscreteArrayProgram,
        )

        hidden_batch, squeezed = self._coerce(hidden_state)
        batch_size = hidden_batch.shape[0]
        device = hidden_batch.device
        resolved_temperature = float(
            self.config.temperature if temperature is None else temperature
        )
        array_batch, length_batch = self._resolve_array_inputs(
            array_inputs,
            lengths,
            batch_size=batch_size,
            device=device,
        )

        hits: list[Any] = [None] * batch_size
        for idx in range(batch_size):
            entry = library.lookup(hidden_batch[idx].detach())
            hits[idx] = entry
        all_hits = all(entry is not None for entry in hits)

        if all_hits and batch_size > 0:
            # Group samples by identical program. When the library has
            # clustered N identical skills (the common case after warmup),
            # this reduces N python-loop execute calls to one vectorized call.
            group_lookup: dict[tuple[int, int, int, int, float], list[int]] = {}
            for idx, entry in enumerate(hits):
                assert entry is not None
                program = entry.program
                group_key = (
                    program.init_idx,
                    program.transform_idx,
                    program.reduce_idx,
                    program.post_scale_idx,
                    float(program.offset),
                )
                group_lookup.setdefault(group_key, []).append(idx)

            predicted = torch.empty(
                (batch_size,),
                device=device,
                dtype=array_batch.dtype
                if array_batch.is_floating_point()
                else torch.float32,
            )
            programs: list[DiscreteArrayProgram] = [None] * batch_size  # type: ignore[list-item]
            program_texts: list[str] = [""] * batch_size
            with torch.no_grad():
                for group_key, indices in group_lookup.items():
                    program = hits[indices[0]].program  # type: ignore[union-attr]
                    idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
                    out = program.execute(
                        array_batch.index_select(0, idx_tensor),
                        length_batch.index_select(0, idx_tensor),
                    )
                    predicted.index_copy_(0, idx_tensor, out)
                    rendered = program.render()
                    for sample_idx in indices:
                        programs[sample_idx] = program
                        program_texts[sample_idx] = rendered

            predicted_output = predicted.squeeze(0) if squeezed else predicted
            next_hidden = hidden_batch.squeeze(0) if squeezed else hidden_batch
            return ArrayThoughtLibraryResult(
                predicted_output=predicted_output,
                next_hidden_state=next_hidden,
                programs=programs,
                program_texts=program_texts,
                library_hits=[True] * batch_size,
                newly_cached=[False] * batch_size,
                convergence_gaps=[0.0] * batch_size,
            )

        soft_result = self.forward(
            hidden_batch,
            array_batch,
            lengths=length_batch,
            temperature=resolved_temperature,
        )

        init_probs = soft_result.init_probs
        if init_probs.ndim == 1:
            init_probs = init_probs.unsqueeze(0)
            transform_probs = soft_result.transform_probs.unsqueeze(0)
            reduce_probs = soft_result.reduce_probs.unsqueeze(0)
            post_scale_probs = soft_result.post_scale_probs.unsqueeze(0)
            post_offsets = soft_result.post_offsets.unsqueeze(0)
        else:
            transform_probs = soft_result.transform_probs
            reduce_probs = soft_result.reduce_probs
            post_scale_probs = soft_result.post_scale_probs
            post_offsets = soft_result.post_offsets
        distributions = {
            "init": init_probs,
            "transform": transform_probs,
            "reduce": reduce_probs,
            "post_scale": post_scale_probs,
            "post_offset": post_offsets,
        }

        soft_outputs = soft_result.predicted_output
        if soft_outputs.ndim == 0:
            soft_outputs = soft_outputs.unsqueeze(0)

        predicted = soft_outputs.detach().clone()
        programs: list[DiscreteArrayProgram] = []
        program_texts = list(soft_result.program_texts)
        newly_cached = [False] * batch_size
        convergence_gaps = [0.0] * batch_size

        for idx in range(batch_size):
            discrete = DiscreteArrayProgram.from_soft_distributions(distributions, idx)
            programs.append(discrete)

            row_arr = array_batch[idx : idx + 1]
            row_len = length_batch[idx : idx + 1]
            with torch.no_grad():
                discrete_value = discrete.execute(row_arr, row_len).squeeze(0)
            gap = float((discrete_value - soft_outputs[idx]).abs().item())
            convergence_gaps[idx] = gap

            if hits[idx] is not None:
                entry = hits[idx]
                with torch.no_grad():
                    predicted[idx] = entry.program.execute(row_arr, row_len).squeeze(0)
                programs[idx] = entry.program
                program_texts[idx] = entry.program.render()
            elif auto_cache and gap <= convergence_gap_threshold:
                library.record(
                    hidden_batch[idx].detach(),
                    discrete,
                    task_name=task_name,
                    convergence_gap=gap,
                )
                newly_cached[idx] = True
                program_texts[idx] = discrete.render()

        predicted_output = predicted.squeeze(0) if squeezed else predicted
        next_hidden = soft_result.next_hidden_state

        return ArrayThoughtLibraryResult(
            predicted_output=predicted_output,
            next_hidden_state=next_hidden,
            programs=programs,
            program_texts=program_texts,
            library_hits=[entry is not None for entry in hits],
            newly_cached=newly_cached,
            convergence_gaps=convergence_gaps,
        )


def _operation_hidden_prototypes(hidden_dim: int, num_ops: int) -> torch.Tensor:
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if num_ops <= 0:
        raise ValueError("num_ops must be positive")
    prototypes = torch.zeros(num_ops, hidden_dim, dtype=torch.float32)
    for op_index in range(num_ops):
        slot = op_index % hidden_dim
        prototypes[op_index, slot] = 1.0
        if op_index + hidden_dim // 2 < hidden_dim:
            prototypes[op_index, (slot + hidden_dim // 2) % hidden_dim] = -0.5
    return prototypes


_DEFAULT_OPERATIONS: tuple[str, ...] = ("sum", "max", "count_positive")
_EXTENDED_OPERATIONS: tuple[str, ...] = (
    "sum",
    "max",
    "min",
    "count_positive",
    "count_negative",
    "mean",
    "product",
)


def _compute_operation_target(op_name: str, values: torch.Tensor) -> float:
    """Compute the supervision target for a given operation on a 1-D slice."""
    if values.numel() == 0:
        raise ValueError("cannot compute operation target on empty slice")
    if op_name == "sum":
        return float(values.sum().item())
    if op_name == "max":
        return float(values.max().item())
    if op_name == "min":
        return float(values.min().item())
    if op_name == "count_positive":
        return float((values > 0).to(torch.float32).sum().item())
    if op_name == "count_negative":
        return float((values < 0).to(torch.float32).sum().item())
    if op_name == "mean":
        return float(values.to(torch.float32).mean().item())
    if op_name == "product":
        return float(values.to(torch.float32).prod().item())
    raise ValueError(f"unknown operation: {op_name}")


def build_array_thought_smoke_batch(
    *,
    hidden_dim: int,
    array_max_len: int = 6,
    samples_per_op: int = 8,
    min_length: int = 2,
    seed: int = 0,
    device: str | torch.device = "cpu",
    value_low: int = -3,
    value_high: int = 3,
    noise_scale: float = 0.02,
    operations: Optional[tuple[str, ...]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Build a curriculum of array-reduction samples.

    The default curriculum is `("sum", "max", "count_positive")` — the minimal
    M2 suite. Pass `operations=_EXTENDED_OPERATIONS` (or any subset of
    `{"sum", "max", "min", "count_positive", "count_negative", "mean",
    "product"}`) to exercise the transform↔reduce slot separation more
    aggressively.
    """
    if samples_per_op <= 0:
        raise ValueError("samples_per_op must be positive")
    if min_length < 1 or min_length > array_max_len:
        raise ValueError("invalid min_length")
    resolved_operations = tuple(operations) if operations is not None else _DEFAULT_OPERATIONS
    if not resolved_operations:
        raise ValueError("operations must be non-empty")
    for op_name in resolved_operations:
        if op_name not in _EXTENDED_OPERATIONS:
            raise ValueError(
                f"unknown operation {op_name!r}. Valid: {_EXTENDED_OPERATIONS}"
            )
    prototypes = _operation_hidden_prototypes(hidden_dim, len(resolved_operations))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    total = len(resolved_operations) * samples_per_op
    hidden_states = torch.zeros(total, hidden_dim, dtype=torch.float32)
    arrays = torch.zeros(total, array_max_len, dtype=torch.float32)
    lengths = torch.zeros(total, dtype=torch.float32)
    targets = torch.zeros(total, dtype=torch.float32)
    labels: list[str] = []

    row = 0
    for op_index, op_name in enumerate(resolved_operations):
        for _ in range(samples_per_op):
            length = int(
                torch.randint(min_length, array_max_len + 1, (1,), generator=generator).item()
            )
            values = torch.randint(
                value_low, value_high + 1, (length,), generator=generator
            ).to(torch.float32)
            # Product targets can explode if zero is in the alphabet and the
            # array becomes very long — rejection-sample a non-zero slice to
            # keep the curriculum numerically stable.
            if op_name == "product":
                attempts = 0
                while bool(torch.any(values == 0).item()) and attempts < 8:
                    values = torch.randint(
                        value_low, value_high + 1, (length,), generator=generator
                    ).to(torch.float32)
                    attempts += 1
                # Fall back: if we still have a zero, force non-zero entries.
                if bool(torch.any(values == 0).item()):
                    values = torch.where(
                        values == 0,
                        torch.ones_like(values),
                        values,
                    )
            arrays[row, :length] = values
            lengths[row] = float(length)
            hidden_states[row] = prototypes[op_index]
            if noise_scale > 0.0:
                hidden_states[row] += noise_scale * torch.randn(
                    hidden_dim, generator=generator
                )
            targets[row] = _compute_operation_target(op_name, values)
            labels.append(op_name)
            row += 1

    return (
        hidden_states.to(device),
        arrays.to(device),
        lengths.to(device),
        targets.to(device),
        labels,
    )


def run_array_thought_smoke_train(
    head: ArrayExecutableThoughtHead,
    *,
    hidden_state: torch.Tensor,
    array_inputs: torch.Tensor,
    lengths: torch.Tensor,
    targets: torch.Tensor,
    steps: int = 100,
    learning_rate: float = 5e-2,
    start_temperature: float = 1.5,
    end_temperature: float = 0.35,
) -> ArrayExecutableThoughtSmokeMetrics:
    """Anneal temperature from soft to near-discrete while minimizing MSE."""
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    targets = targets.to(hidden_state.device)

    with torch.no_grad():
        initial = head(
            hidden_state,
            array_inputs,
            lengths=lengths,
            temperature=start_temperature,
        )
        initial_loss = float(F.mse_loss(initial.predicted_output, targets).item())

    history: list[float] = []
    total_steps = max(steps, 1)
    for step in range(total_steps):
        progress = step / max(total_steps - 1, 1)
        temperature = start_temperature + (end_temperature - start_temperature) * progress
        optimizer.zero_grad()
        result = head(
            hidden_state,
            array_inputs,
            lengths=lengths,
            temperature=temperature,
        )
        loss = F.mse_loss(result.predicted_output, targets)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().item()))

    with torch.no_grad():
        final = head(
            hidden_state,
            array_inputs,
            lengths=lengths,
            temperature=end_temperature,
        )
        final_loss = float(F.mse_loss(final.predicted_output, targets).item())
        final_mae = float((final.predicted_output - targets).abs().mean().item())

    return ArrayExecutableThoughtSmokeMetrics(
        initial_loss=initial_loss,
        final_loss=final_loss,
        final_mae=final_mae,
        loss_history=history,
        final_program_texts=list(final.program_texts),
    )


def train_array_thought_head(
    *,
    output_path: str | Path,
    config: Optional[ArrayExecutableThoughtHeadConfig] = None,
    hidden_dim: int = 8,
    steps: int = 120,
    learning_rate: float = 5e-2,
    samples_per_op: int = 8,
    start_temperature: float = 1.5,
    end_temperature: float = 0.35,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Train and save a smoke-test array-thought checkpoint."""
    resolved_config = config or ArrayExecutableThoughtHeadConfig(hidden_dim=hidden_dim)
    torch.manual_seed(seed)
    head = ArrayExecutableThoughtHead(resolved_config).to(device)
    hidden_state, arrays, lengths, targets, labels = build_array_thought_smoke_batch(
        hidden_dim=resolved_config.hidden_dim,
        array_max_len=resolved_config.array_max_len,
        samples_per_op=samples_per_op,
        seed=seed,
        device=device,
    )
    metrics = run_array_thought_smoke_train(
        head,
        hidden_state=hidden_state,
        array_inputs=arrays,
        lengths=lengths,
        targets=targets,
        steps=steps,
        learning_rate=learning_rate,
        start_temperature=start_temperature,
        end_temperature=end_temperature,
    )
    head.eval()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "config": resolved_config.to_dict(),
            "metrics": asdict(metrics),
            "train_examples": len(labels),
            "samples_per_op": samples_per_op,
            "task_labels": labels,
        },
        destination,
    )
    return {
        "output_path": str(destination),
        "config": resolved_config.to_dict(),
        "train_examples": len(labels),
        "samples_per_op": samples_per_op,
        "initial_loss": metrics.initial_loss,
        "final_loss": metrics.final_loss,
        "final_mae": metrics.final_mae,
        "trained": True,
    }


def load_array_thought_head(
    *,
    path: str | Path,
    device: str | torch.device,
    config: Optional[ArrayExecutableThoughtHeadConfig] = None,
    state_patch_head: Optional[StatePatchHead] = None,
) -> ArrayExecutableThoughtHead:
    """Load a saved checkpoint."""
    checkpoint_path = Path(path).expanduser()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    resolved_config = config
    if resolved_config is None and isinstance(payload, dict) and "config" in payload:
        resolved_config = ArrayExecutableThoughtHeadConfig(**payload["config"])
    if resolved_config is None:
        raise ValueError("Array thought head checkpoint missing config")
    head = ArrayExecutableThoughtHead(resolved_config, state_patch_head=state_patch_head)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    # Tolerant load: upgrade legacy (3-init) checkpoints to the current
    # (4-init, `+large`) projector layout. New init's learned rows are zero and
    # its prior is the configured negative `init_prior_pos_large`, so it starts
    # un-preferred and the legacy model's behavior is preserved. No-op for the
    # current dim.
    upgraded_state_dict, _ = upgrade_state_dict_for_init_expansion(
        state_dict, prior_pos_large=resolved_config.init_prior_pos_large
    )
    head.load_state_dict(upgraded_state_dict)
    head = head.to(device)
    head.eval()
    return head


__all__ = [
    "ArrayExecutableThoughtHeadConfig",
    "ArrayExecutableThoughtResult",
    "ArrayExecutableThoughtSmokeMetrics",
    "ArrayExecutableThoughtHead",
    "build_array_thought_smoke_batch",
    "run_array_thought_smoke_train",
    "train_array_thought_head",
    "load_array_thought_head",
    "upgrade_state_dict_for_init_expansion",
    "_DEFAULT_OPERATIONS",
    "_EXTENDED_OPERATIONS",
    "_compute_operation_target",
]
