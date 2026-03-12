"""Evaluate learned latent-memory heads against simple baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Optional

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ncpu.self_optimizing.latent_memory_head import (
    LatentMemoryHeadConfig,
    load_latent_memory_head,
)


@dataclass
class LatentMemoryEvalMetrics:
    """Aggregated regression metrics for latent-memory prediction."""

    num_examples: int
    mse: float
    mae: float
    mean_cosine_similarity: float
    mean_target_norm: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _tensorize(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    if not rows:
        return (
            torch.zeros((0, 0), dtype=torch.float32),
            torch.zeros((0, 0), dtype=torch.float32),
            [],
        )
    features = torch.tensor([row["feature_vector"] for row in rows], dtype=torch.float32)
    targets = torch.tensor([row["target_vector"] for row in rows], dtype=torch.float32)
    event_kinds = [str(row.get("event_kind") or "unknown") for row in rows]
    return features, targets, event_kinds


def _compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> LatentMemoryEvalMetrics:
    if predictions.numel() == 0 or targets.numel() == 0:
        return LatentMemoryEvalMetrics(
            num_examples=0,
            mse=0.0,
            mae=0.0,
            mean_cosine_similarity=0.0,
            mean_target_norm=0.0,
        )

    diff = predictions - targets
    mse = float(torch.mean(diff.pow(2)).item())
    mae = float(torch.mean(torch.abs(diff)).item())
    target_norms = torch.linalg.norm(targets, dim=1)
    prediction_norms = torch.linalg.norm(predictions, dim=1)
    dot = torch.sum(predictions * targets, dim=1)
    cosine = dot / torch.clamp(prediction_norms * target_norms, min=1e-8)
    cosine = torch.where(target_norms > 1e-8, cosine, torch.zeros_like(cosine))
    return LatentMemoryEvalMetrics(
        num_examples=int(targets.shape[0]),
        mse=mse,
        mae=mae,
        mean_cosine_similarity=float(torch.mean(cosine).item()),
        mean_target_norm=float(torch.mean(target_norms).item()),
    )


def _group_metrics_by_event(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    event_kinds: list[str],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[int]] = {}
    for index, event_kind in enumerate(event_kinds):
        grouped.setdefault(event_kind, []).append(index)

    output: dict[str, dict[str, Any]] = {}
    for event_kind, indices in grouped.items():
        index_tensor = torch.tensor(indices, dtype=torch.long)
        metrics = _compute_metrics(predictions[index_tensor], targets[index_tensor])
        output[event_kind] = metrics.to_dict()
    return output


def evaluate_latent_memory_head(
    *,
    train_path: str | Path,
    val_path: str | Path,
    checkpoint_path: str | Path,
    output_path: Optional[str | Path] = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate a trained latent-memory head against a zero-delta baseline."""
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    config_payload = dict(payload.get("config") or {})
    config = LatentMemoryHeadConfig(**config_payload) if config_payload else LatentMemoryHeadConfig()
    head = load_latent_memory_head(
        path=checkpoint_path,
        device=device,
        config=config,
    )

    report: dict[str, Any] = {
        "checkpoint_path": str(Path(checkpoint_path)),
        "device": device,
        "config": config.to_dict(),
        "checkpoint_metrics": dict(payload.get("metrics") or {}),
        "splits": {},
    }

    for split_name, split_path in (("train", train_path), ("val", val_path)):
        rows = _load_jsonl_rows(split_path)
        features, targets, event_kinds = _tensorize(rows)
        if features.numel() == 0:
            split_report = {
                "num_examples": 0,
                "baseline_zero": LatentMemoryEvalMetrics(0, 0.0, 0.0, 0.0, 0.0).to_dict(),
                "model": LatentMemoryEvalMetrics(0, 0.0, 0.0, 0.0, 0.0).to_dict(),
                "event_breakdown": {},
            }
            report["splits"][split_name] = split_report
            continue

        with torch.no_grad():
            predictions = head(features.to(device)).detach().cpu()
        zero_baseline = torch.zeros_like(targets)
        baseline_metrics = _compute_metrics(zero_baseline, targets)
        model_metrics = _compute_metrics(predictions, targets)
        baseline_by_event = _group_metrics_by_event(zero_baseline, targets, event_kinds)
        model_by_event = _group_metrics_by_event(predictions, targets, event_kinds)
        improvement = {
            "mse_delta": baseline_metrics.mse - model_metrics.mse,
            "mae_delta": baseline_metrics.mae - model_metrics.mae,
            "cosine_delta": model_metrics.mean_cosine_similarity - baseline_metrics.mean_cosine_similarity,
            "relative_mse_improvement": (
                0.0 if baseline_metrics.mse <= 1e-12 else (baseline_metrics.mse - model_metrics.mse) / baseline_metrics.mse
            ),
        }
        event_breakdown = {}
        for event_kind in sorted(set(event_kinds)):
            baseline_event = baseline_by_event.get(event_kind, {})
            model_event = model_by_event.get(event_kind, {})
            event_breakdown[event_kind] = {
                "baseline_zero": baseline_event,
                "model": model_event,
                "relative_mse_improvement": (
                    0.0
                    if baseline_event.get("mse", 0.0) <= 1e-12
                    else (baseline_event["mse"] - model_event.get("mse", 0.0)) / baseline_event["mse"]
                ),
            }

        report["splits"][split_name] = {
            "num_examples": len(rows),
            "baseline_zero": baseline_metrics.to_dict(),
            "model": model_metrics.to_dict(),
            "improvement": improvement,
            "event_breakdown": event_breakdown,
        }

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a latent memory head against a zero baseline")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-path")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    result = evaluate_latent_memory_head(
        train_path=args.train_path,
        val_path=args.val_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
