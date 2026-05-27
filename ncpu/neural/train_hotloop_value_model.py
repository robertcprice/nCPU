#!/usr/bin/env python3
"""Train a hotloop value model from benchmark matrices or telemetry JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.neural.hotloop_value_model import (  # noqa: E402
    HOTLOOP_VALUE_FEATURE_NAMES,
    NeuralHotloopValueModel,
    build_bootstrap_hotloop_dataset,
    extract_hotloop_value_examples,
    load_hotloop_samples,
    recommended_hotloop_value_threshold,
    save_hotloop_value_model,
    train_hotloop_value_model as train_supervised_hotloop_value_model,
)


def _select_operating_threshold(
    predictions: torch.Tensor,
    positive_labels: torch.Tensor,
    *,
    default: float,
) -> tuple[float, float]:
    candidates = torch.unique(predictions.detach().cpu()).tolist()
    candidates.extend([0.0, float(default), 1.0])
    best_threshold = float(default)
    best_accuracy = -1.0
    best_balanced_accuracy = -1.0
    positive_total = int(positive_labels.sum().item())
    negative_total = int((~positive_labels).sum().item())
    for threshold in sorted(set(float(item) for item in candidates)):
        predicted_positive = predictions >= threshold
        accuracy = float((predicted_positive == positive_labels).float().mean().item())
        if positive_total > 0 and negative_total > 0:
            true_positive = int((predicted_positive & positive_labels).sum().item())
            true_negative = int(((~predicted_positive) & (~positive_labels)).sum().item())
            balanced_accuracy = 0.5 * (
                (true_positive / positive_total) +
                (true_negative / negative_total)
            )
        else:
            balanced_accuracy = accuracy
        if (
            balanced_accuracy > best_balanced_accuracy or
            (
                balanced_accuracy == best_balanced_accuracy and (
                    accuracy > best_accuracy or
                    (
                        accuracy == best_accuracy and
                        abs(threshold - default) < abs(best_threshold - default)
                    )
                )
            )
        ):
            best_balanced_accuracy = balanced_accuracy
            best_accuracy = accuracy
            best_threshold = threshold
    return best_threshold, best_accuracy


def train_hotloop_value_model(
    input_paths: list[Path],
    *,
    epochs: int = 120,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 7,
    negative_augmentations: int = 2,
    label_source: str = "speedup",
) -> tuple[NeuralHotloopValueModel, dict]:
    torch.manual_seed(seed)
    if label_source == "speedup":
        examples = []
        for path in input_paths:
            examples.extend(extract_hotloop_value_examples(json.loads(path.read_text())))
        model, stats = train_supervised_hotloop_value_model(
            examples,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
        )
        all_features = torch.tensor([example["features"] for example in examples], dtype=torch.float32)
        speedup_labels = torch.tensor(
            [float(example["speedup_ratio"] >= 1.2) for example in examples],
            dtype=torch.bool,
        )
        with torch.no_grad():
            predictions = model(all_features)
        analytical_threshold = recommended_hotloop_value_threshold(1.2)
        operating_threshold, operating_accuracy = _select_operating_threshold(
            predictions,
            speedup_labels,
            default=analytical_threshold,
        )
        stats = {
            **stats,
            "num_training_rows": float(len(examples)),
            "positive_rate": float(sum(example["target_score"] >= 0.5 for example in examples) / max(len(examples), 1)),
            "feature_names": list(HOTLOOP_VALUE_FEATURE_NAMES),
            "label_source": "speedup",
            "recommended_threshold_1_2x": operating_threshold,
            "analytical_threshold_1_2x": analytical_threshold,
            "operating_accuracy_1_2x": operating_accuracy,
        }
        return model, stats

    samples = load_hotloop_samples(input_paths)
    features, targets, metadata = build_bootstrap_hotloop_dataset(
        samples,
        negative_augmentations=negative_augmentations,
    )

    num_rows = len(targets)
    if num_rows < 4:
        raise ValueError("need at least four training rows")

    indices = torch.randperm(num_rows)
    split = max(1, int(num_rows * 0.8))
    split = min(split, num_rows - 1)
    train_idx = indices[:split]
    eval_idx = indices[split:]

    x_train = features[train_idx]
    y_train = targets[train_idx]
    x_eval = features[eval_idx]
    y_eval = targets[eval_idx]

    model = NeuralHotloopValueModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_train))
        for start in range(0, len(x_train), batch_size):
            batch = perm[start:start + batch_size]
            preds = model(x_train[batch])
            loss = criterion(preds, y_train[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_preds = model(x_train)
        eval_preds = model(x_eval)
        train_acc = float(((train_preds >= 0.5) == (y_train >= 0.5)).float().mean().item())
        eval_acc = float(((eval_preds >= 0.5) == (y_eval >= 0.5)).float().mean().item())

    stats = {
        **metadata,
        "train_rows": int(len(x_train)),
        "eval_rows": int(len(x_eval)),
        "train_accuracy": train_acc,
        "eval_accuracy": eval_acc,
        "feature_names": list(HOTLOOP_VALUE_FEATURE_NAMES),
        "label_source": "bootstrap",
    }
    return model, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="Benchmark matrix or telemetry JSON file. Repeat for multiple inputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "ncpu" / "neural" / "hotloop_value_model.pt",
        help="Output checkpoint path (default: ncpu/neural/hotloop_value_model.pt)",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--negative-augmentations", type=int, default=2)
    parser.add_argument(
        "--label-source",
        choices=("speedup", "bootstrap"),
        default="speedup",
        help="Training target source (default: speedup)",
    )
    args = parser.parse_args()

    model, stats = train_hotloop_value_model(
        [Path(item) for item in args.inputs],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        negative_augmentations=args.negative_augmentations,
        label_source=args.label_source,
    )

    save_hotloop_value_model(model, args.output, metadata=stats)

    print(f"Saved {args.output}")
    print(f"Rows: {stats['num_training_rows']:.0f}")
    print(f"Positive rate: {stats['positive_rate']:.3f}")
    if args.label_source == "speedup":
        print(f"Train MAE: {stats['train_mae']:.4f}")
        print(f"Eval MAE: {stats['val_mae']:.4f}")
        print(f"Recommended threshold (1.2x): {stats['recommended_threshold_1_2x']:.4f}")
        print(f"Analytical threshold (1.2x): {stats['analytical_threshold_1_2x']:.4f}")
        print(f"Operating accuracy (1.2x): {stats['operating_accuracy_1_2x']:.3f}")
    else:
        print(f"Train accuracy: {stats['train_accuracy']:.3f}")
        print(f"Eval accuracy: {stats['eval_accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
