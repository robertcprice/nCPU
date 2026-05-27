"""Hotloop value model utilities for GPU-only auto handoff policy.

The model predicts a bounded handoff score from the raw policy features used by
``gpu_only.py``:

    [body_word_count, estimated_iterations, estimated_work,
     pre_sync_bytes, post_sync_bytes, remaining_instructions,
     tail_word_count, synthetic_stop, region_blocks,
     nested_branch_count, branch_kind_b, branch_kind_bcond,
     branch_kind_cbz, branch_kind_cbnz, tail_max_imm16,
     tail_large_imm16_count, segment, reused_state,
     previous_pre_sync_bytes, previous_post_sync_bytes,
     previous_region_blocks, previous_tail_word_count,
     previous_tail_max_imm16, previous_tail_large_imm16_count]

Training data comes from exported ``gpu_only_matrix.json`` artifacts. Each
hotloop trace segment inherits the workload's observed hotloop-vs-torch
speedup when available, falling back to the older Rust-vs-Neural comparison,
and learns a value score in ``(0, 1)``:

    score = speedup / (1 + speedup)

That keeps the existing threshold semantics intuitive:
    - ``0.50`` == break-even
    - ``0.55`` ~= 1.22x speedup
    - ``0.67`` == 2.0x speedup
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

HOTLOOP_VALUE_FEATURE_NAMES = (
    "body_word_count",
    "estimated_iterations",
    "estimated_work",
    "pre_sync_bytes",
    "post_sync_bytes",
    "remaining_instructions",
    "tail_word_count",
    "synthetic_stop",
    "region_blocks",
    "nested_branch_count",
    "branch_kind_b",
    "branch_kind_bcond",
    "branch_kind_cbz",
    "branch_kind_cbnz",
    "tail_max_imm16",
    "tail_large_imm16_count",
    "segment",
    "reused_state",
    "previous_pre_sync_bytes",
    "previous_post_sync_bytes",
    "previous_region_blocks",
    "previous_tail_word_count",
    "previous_tail_max_imm16",
    "previous_tail_large_imm16_count",
)

DEFAULT_HOTLOOP_VALUE_MODEL_PATH = Path(__file__).with_name("hotloop_value_model.pt")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        return float(value)
    except Exception:
        return default


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _with_sequential_hotloop_context(
    sample: Mapping[str, Any],
    *,
    previous_segment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(sample)
    previous = previous_segment
    if previous is None:
        candidate = merged.get("previous_segment")
        if isinstance(candidate, Mapping):
            previous = candidate
    segment = max(int(_as_float(merged.get("segment"), 1.0)), 1)
    merged.setdefault("segment", segment)
    merged.setdefault("reused_state", bool(segment > 1 and previous is not None))
    if previous is None:
        previous = {}
    merged.setdefault("previous_pre_sync_bytes", _as_float(previous.get("pre_sync_bytes")))
    merged.setdefault("previous_post_sync_bytes", _as_float(previous.get("post_sync_bytes")))
    merged.setdefault("previous_region_blocks", _as_float(previous.get("region_blocks"), 1.0 if segment > 1 else 0.0))
    merged.setdefault("previous_tail_word_count", _as_float(previous.get("tail_word_count")))
    merged.setdefault("previous_tail_max_imm16", _as_float(previous.get("tail_max_imm16")))
    merged.setdefault(
        "previous_tail_large_imm16_count",
        _as_float(previous.get("tail_large_imm16_count")),
    )
    return merged


def build_hotloop_value_feature_tensor(
    sample: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build the raw feature tensor expected by the value model."""
    contextual_sample = _with_sequential_hotloop_context(sample)
    branch_kind = str(contextual_sample.get("branch_kind", "")).strip().lower()
    nested_branch_count = contextual_sample.get("nested_branch_count")
    if nested_branch_count is None:
        nested_branches = contextual_sample.get("nested_branches")
        if isinstance(nested_branches, Sequence):
            nested_branch_count = len(nested_branches)
        else:
            nested_branch_count = 1
    values = [
        _as_float(contextual_sample.get("body_word_count")),
        _as_float(contextual_sample.get("estimated_iterations")),
        _as_float(contextual_sample.get("estimated_work")),
        _as_float(contextual_sample.get("pre_sync_bytes")),
        _as_float(contextual_sample.get("post_sync_bytes")),
        _as_float(contextual_sample.get("remaining_instructions")),
        _as_float(contextual_sample.get("tail_word_count")),
        1.0 if bool(contextual_sample.get("synthetic_stop")) else 0.0,
        _as_float(contextual_sample.get("region_blocks"), 1.0),
        _as_float(nested_branch_count, 1.0),
        1.0 if branch_kind == "b" else _as_float(contextual_sample.get("branch_kind_b")),
        1.0 if branch_kind == "bcond" else _as_float(contextual_sample.get("branch_kind_bcond")),
        1.0 if branch_kind == "cbz" else _as_float(contextual_sample.get("branch_kind_cbz")),
        1.0 if branch_kind == "cbnz" else _as_float(contextual_sample.get("branch_kind_cbnz")),
        _as_float(contextual_sample.get("tail_max_imm16")),
        _as_float(contextual_sample.get("tail_large_imm16_count")),
        _as_float(contextual_sample.get("segment"), 1.0),
        1.0 if bool(contextual_sample.get("reused_state")) else 0.0,
        _as_float(contextual_sample.get("previous_pre_sync_bytes")),
        _as_float(contextual_sample.get("previous_post_sync_bytes")),
        _as_float(contextual_sample.get("previous_region_blocks")),
        _as_float(contextual_sample.get("previous_tail_word_count")),
        _as_float(contextual_sample.get("previous_tail_max_imm16")),
        _as_float(contextual_sample.get("previous_tail_large_imm16_count")),
    ]
    return torch.tensor(values, dtype=torch.float32, device=device)


def predict_hotloop_value_score(
    model: Any,
    sample_or_features: Mapping[str, Any] | torch.Tensor,
    *,
    device: str | torch.device = "cpu",
) -> float:
    """Run a hotloop value model on either a feature mapping or raw feature tensor."""
    if torch.is_tensor(sample_or_features):
        features = sample_or_features.to(device=device, dtype=torch.float32)
    else:
        features = build_hotloop_value_feature_tensor(sample_or_features, device=device)
    if features.dim() == 1:
        features = features.unsqueeze(0)
    output = model(features)
    if isinstance(output, tuple):
        output = output[0]
    if torch.is_tensor(output):
        return float(output.reshape(-1)[0].item())
    return float(output)


def encode_hotloop_value_embedding(
    model: Any,
    sample_or_features: Mapping[str, Any] | torch.Tensor,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor | None:
    """Extract a latent hotloop embedding when the model exposes one."""
    encoder = getattr(model, "encode_features", None)
    if not callable(encoder):
        return None
    if torch.is_tensor(sample_or_features):
        features = sample_or_features.to(device=device, dtype=torch.float32)
    else:
        features = build_hotloop_value_feature_tensor(sample_or_features, device=device)
    embedding = encoder(features)
    if not torch.is_tensor(embedding):
        return None
    if embedding.dim() == 1:
        return embedding.detach()
    return embedding.reshape(embedding.shape[0], -1)[0].detach()


def derive_hotloop_value_target(sample: Mapping[str, Any]) -> float:
    """Bootstrap a bounded training target from observed dispatch data."""
    explicit = sample.get("value_target")
    if explicit is not None:
        return _clamp01(_as_float(explicit))

    if sample.get("approved") is False:
        return 0.0

    estimated_work = max(
        _as_float(sample.get("estimated_work")),
        _as_float(sample.get("executed_count")),
        _as_float(sample.get("body_word_count")),
    )
    total_sync = max(_as_float(sample.get("pre_sync_bytes")) + _as_float(sample.get("post_sync_bytes")), 0.0)
    observed_ips = _as_float(sample.get("observed_ips"))
    if observed_ips <= 0.0:
        executed = _as_float(sample.get("executed_count"))
        elapsed = _as_float(sample.get("elapsed_seconds"))
        if executed > 0.0 and elapsed > 0.0:
            observed_ips = executed / elapsed

    work_term = _clamp01(torch.log1p(torch.tensor(max(estimated_work, 0.0))).item() / 10.0)
    sync_penalty = _clamp01(torch.log1p(torch.tensor(max(total_sync, 0.0))).item() / 16.0)
    ips_term = _clamp01(torch.log1p(torch.tensor(max(observed_ips, 0.0))).item() / 16.0) if observed_ips > 0.0 else 0.65
    target = (0.45 * work_term) + (0.40 * ips_term) + (0.15 * (1.0 - sync_penalty))
    return _clamp01(target)


def iter_hotloop_samples(payload: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield flat hotloop samples from benchmark matrices or telemetry exports."""
    gpu_only = payload.get("gpu_only_hotloop")
    if isinstance(gpu_only, Mapping):
        for sample in gpu_only.get("recent_samples", []):
            if isinstance(sample, Mapping):
                yield dict(sample)

    for row in payload.get("results", []):
        if not isinstance(row, Mapping):
            continue
        for key in ("hotloop_samples", "hotloop_trace"):
            for sample in row.get(key, []):
                if not isinstance(sample, Mapping):
                    continue
                merged = dict(sample)
                merged.setdefault("workload", row.get("workload"))
                merged.setdefault("backend", row.get("backend"))
                yield merged


def load_hotloop_samples(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        samples.extend(iter_hotloop_samples(payload))
    return samples


def build_bootstrap_hotloop_dataset(
    samples: Sequence[Mapping[str, Any]],
    *,
    device: str | torch.device = "cpu",
    negative_augmentations: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    feature_rows: list[torch.Tensor] = []
    targets: list[float] = []

    for sample in samples:
        target = derive_hotloop_value_target(sample)
        feature_rows.append(build_hotloop_value_feature_tensor(sample, device=device))
        targets.append(target)

        if target < 0.55:
            continue

        base_work = max(int(_as_float(sample.get("estimated_work"), 1.0)), 1)
        base_body = max(int(_as_float(sample.get("body_word_count"), 1.0)), 1)
        base_pre = max(int(_as_float(sample.get("pre_sync_bytes"), 0.0)), 0)
        base_post = max(int(_as_float(sample.get("post_sync_bytes"), 0.0)), 0)
        base_remaining = max(int(_as_float(sample.get("remaining_instructions"), 1.0)), 1)

        for aug_idx in range(negative_augmentations):
            synthetic = dict(sample)
            synthetic["approved"] = False
            if aug_idx % 2 == 0:
                synthetic["estimated_iterations"] = 1
                synthetic["estimated_work"] = min(base_body, base_work)
                synthetic["remaining_instructions"] = min(base_remaining, base_body)
            else:
                synthetic["pre_sync_bytes"] = max(base_pre * 8, 4096)
                synthetic["post_sync_bytes"] = max(base_post * 8, 4096)
            synthetic["value_target"] = 0.0
            feature_rows.append(build_hotloop_value_feature_tensor(synthetic, device=device))
            targets.append(0.0)

    if not feature_rows:
        raise ValueError("expected at least one hotloop sample")

    features = torch.stack(feature_rows)
    labels = torch.tensor(targets, dtype=torch.float32, device=device)
    metadata = {
        "num_samples": float(len(samples)),
        "num_training_rows": float(len(labels)),
        "positive_rate": float((labels >= 0.5).float().mean().item()),
    }
    return features, labels, metadata


def hotloop_value_score_from_speedup(speedup_ratio: float) -> float:
    """Map a positive speedup ratio to a bounded handoff score."""
    speedup_ratio = float(speedup_ratio)
    if speedup_ratio <= 0:
        raise ValueError("speedup_ratio must be positive")
    return speedup_ratio / (1.0 + speedup_ratio)


def speedup_ratio_from_value_score(score: float) -> float:
    """Invert ``hotloop_value_score_from_speedup`` for reporting thresholds."""
    score = float(score)
    if not 0.0 < score < 1.0:
        raise ValueError("score must be in the open interval (0, 1)")
    return score / (1.0 - score)


def recommended_hotloop_value_threshold(min_speedup_ratio: float = 1.2) -> float:
    """Return the threshold that corresponds to a desired minimum speedup."""
    return hotloop_value_score_from_speedup(min_speedup_ratio)


def extract_hotloop_value_examples(payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten exported benchmark matrix payloads into training examples."""
    if isinstance(payload, Mapping):
        rows = payload.get("results", [])
    else:
        rows = payload

    examples: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("result_ok", True) or not row.get("insts_ok", True):
            continue

        speedup_ratio = float(
            row.get("hotloop_speedup_vs_torch")
            or row.get("best_rust_speedup_vs_neural")
            or 0.0
        )
        if speedup_ratio <= 0.0:
            continue

        target_score = hotloop_value_score_from_speedup(speedup_ratio)
        sample_rows = row.get("hotloop_samples") or row.get("hotloop_trace", [])
        previous_trace_entry = None
        for trace_entry in sample_rows:
            merged = _with_sequential_hotloop_context(
                trace_entry,
                previous_segment=previous_trace_entry,
            )
            merged.setdefault("workload", row.get("workload"))
            merged.setdefault("backend", row.get("backend"))
            if "remaining_instructions" not in merged and "remaining_after" in merged:
                merged["remaining_instructions"] = (
                    _as_float(merged.get("remaining_after"))
                    + _as_float(merged.get("executed_count"))
                )
            features = build_hotloop_value_feature_tensor(merged).detach().cpu().tolist()

            sample_weight = float(
                merged.get("executed_count")
                or merged.get("estimated_work")
                or 1.0
            )
            examples.append(
                {
                    "workload": merged.get("workload"),
                    "backend": merged.get("backend"),
                    "segment": int(merged.get("segment", len(examples) + 1)),
                    "approved": bool(merged.get("approved", False)),
                    "features": features,
                    "target_score": target_score,
                    "speedup_ratio": speedup_ratio,
                    "sample_weight": max(sample_weight, 1.0),
                    "trace_reason": merged.get("reason") or merged.get("policy_reason"),
                }
            )
            previous_trace_entry = merged
    return examples


def build_hotloop_value_tensors(
    examples: Sequence[Mapping[str, Any]],
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert extracted examples into model tensors."""
    if not examples:
        raise ValueError("expected at least one hotloop value example")

    features = torch.tensor(
        [example["features"] for example in examples],
        dtype=torch.float32,
        device=device,
    )
    targets = torch.tensor(
        [float(example["target_score"]) for example in examples],
        dtype=torch.float32,
        device=device,
    )
    weights = torch.tensor(
        [float(example.get("sample_weight", 1.0)) for example in examples],
        dtype=torch.float32,
        device=device,
    )
    weights = weights / weights.mean().clamp(min=1e-6)
    return features, targets, weights


class NeuralHotloopValueModel(nn.Module):
    """Tiny MLP that predicts a bounded hotloop handoff score."""

    def __init__(self, input_dim: int = len(HOTLOOP_VALUE_FEATURE_NAMES)):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def normalize_features(features: torch.Tensor) -> torch.Tensor:
        features = features.to(torch.float32)
        return torch.log1p(torch.clamp(features, min=0.0))

    def encode_features(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_features(features)
        if normalized.dim() == 1:
            normalized = normalized.unsqueeze(0)
            encoded = self.predictor[:4](normalized)
            return encoded.squeeze(0)
        return self.predictor[:4](normalized)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_features(features)
        if encoded.dim() == 1:
            return self.predictor[4:](encoded.unsqueeze(0)).squeeze(0).squeeze(-1)
        return self.predictor[4:](encoded).squeeze(-1)


def train_hotloop_value_model(
    examples: Sequence[Mapping[str, Any]],
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[NeuralHotloopValueModel, dict[str, float]]:
    """Train a hotloop value model from flattened matrix examples."""
    if len(examples) < 2:
        raise ValueError("need at least two examples to train a hotloop value model")

    features, targets, weights = build_hotloop_value_tensors(examples, device=device)
    total_examples = int(features.shape[0])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    operating_threshold = recommended_hotloop_value_threshold(1.2)
    positive_mask = targets >= operating_threshold
    positive_idx = torch.nonzero(positive_mask, as_tuple=False).reshape(-1)
    negative_idx = torch.nonzero(~positive_mask, as_tuple=False).reshape(-1)
    validation_mode = "holdout"

    if total_examples < 24 or positive_idx.numel() < 2 or negative_idx.numel() < 2:
        train_idx = torch.arange(total_examples, device=features.device)
        val_idx = train_idx.clone()
        validation_mode = "in-sample"
    else:
        positive_idx = positive_idx[
            torch.randperm(positive_idx.numel(), generator=generator, device="cpu").to(positive_idx.device)
        ]
        negative_idx = negative_idx[
            torch.randperm(negative_idx.numel(), generator=generator, device="cpu").to(negative_idx.device)
        ]
        val_positive_count = min(max(1, int(positive_idx.numel() * 0.2)), positive_idx.numel() - 1)
        val_negative_count = min(max(1, int(negative_idx.numel() * 0.2)), negative_idx.numel() - 1)
        val_idx = torch.cat(
            [
                positive_idx[:val_positive_count],
                negative_idx[:val_negative_count],
            ]
        )
        train_idx = torch.cat(
            [
                positive_idx[val_positive_count:],
                negative_idx[val_negative_count:],
            ]
        )
        if train_idx.numel() == 0 or val_idx.numel() == 0:
            train_idx = torch.arange(total_examples, device=features.device)
            val_idx = train_idx.clone()
            validation_mode = "in-sample"

    model = NeuralHotloopValueModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val_mae = float("inf")
    effective_epochs = max(int(epochs), 400 if validation_mode == "in-sample" else 1)

    for _epoch in range(effective_epochs):
        model.train()
        train_perm = train_idx[
            torch.randperm(train_idx.numel(), generator=generator, device="cpu").to(train_idx.device)
        ]
        for start in range(0, train_perm.numel(), max(1, int(batch_size))):
            batch = train_perm[start:start + max(1, int(batch_size))]
            predictions = model(features[batch])
            loss = ((predictions - targets[batch]) ** 2 * weights[batch]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(features[val_idx])
            val_mae = (
                (val_predictions - targets[val_idx]).abs() * weights[val_idx]
            ).mean().item()
        if val_mae < best_val_mae:
            best_val_mae = float(val_mae)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        train_predictions = model(features[train_idx])
        val_predictions = model(features[val_idx])
        train_mae = (
            (train_predictions - targets[train_idx]).abs() * weights[train_idx]
        ).mean().item()
        val_mae = (
            (val_predictions - targets[val_idx]).abs() * weights[val_idx]
        ).mean().item()
        train_mse = (
            ((train_predictions - targets[train_idx]) ** 2) * weights[train_idx]
        ).mean().item()
        val_mse = (
            ((val_predictions - targets[val_idx]) ** 2) * weights[val_idx]
        ).mean().item()

    metrics = {
        "example_count": float(total_examples),
        "train_count": float(train_idx.numel()),
        "val_count": float(val_idx.numel()),
        "train_mae": float(train_mae),
        "val_mae": float(val_mae),
        "train_mse": float(train_mse),
        "val_mse": float(val_mse),
        "target_mean": float(targets.mean().item()),
        "validation_mode": validation_mode,
        "effective_epochs": float(effective_epochs),
    }
    return model, metrics


def save_hotloop_value_model(
    model: NeuralHotloopValueModel,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save a trained hotloop value model checkpoint with metadata."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feature_names": list(HOTLOOP_VALUE_FEATURE_NAMES),
    }
    if metadata is not None:
        checkpoint["metadata"] = dict(metadata)
    torch.save(checkpoint, output_path)
    return output_path


def load_hotloop_value_model(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> NeuralHotloopValueModel:
    """Load a hotloop value model checkpoint from disk."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model = NeuralHotloopValueModel().to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
