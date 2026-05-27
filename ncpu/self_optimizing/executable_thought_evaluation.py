"""Evaluate executable-thought heads on real prompt-derived hidden states."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Optional

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ncpu.self_optimizing.executable_thought_context import extract_hidden_state_from_prompt
from ncpu.self_optimizing.executable_thought_head import (
    ExecutableThoughtHeadConfig,
    load_executable_thought_head,
)


@dataclass
class ExecutableThoughtEvalMetrics:
    """Aggregated regression metrics for executable-thought patch prediction."""

    num_examples: int
    mse: float
    mae: float
    mean_cosine_similarity: float
    mean_target_norm: float
    mean_prediction_norm: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_model_and_tokenizer(
    *,
    model_name_or_path: Optional[str | Path],
    model: Optional[Any],
    tokenizer: Optional[Any],
    device: str,
    trust_remote_code: bool,
) -> tuple[Any, Any]:
    if model is not None and tokenizer is not None:
        return model, tokenizer
    if model is None and tokenizer is None and model_name_or_path is not None:
        from ncpu.self_optimizing.core.llm_provider import LLMProviderFactory

        loaded_model, loaded_tokenizer, _resolved_device = LLMProviderFactory._load_hf_local_model(
            str(model_name_or_path),
            device=device,
            trust_remote_code=trust_remote_code,
            use_cache=False,
        )
        del _resolved_device
        return loaded_model, loaded_tokenizer
    raise ValueError("model_name_or_path or both model and tokenizer are required for executable-thought evaluation")


def _tensorize_rows(
    *,
    rows: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    device: str,
    max_prompt_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    if not rows:
        return (
            torch.zeros((0, 0), dtype=torch.float32),
            torch.zeros((0, 0), dtype=torch.float32),
            torch.zeros((0, 0), dtype=torch.float32),
            [],
        )

    hidden_states: list[torch.Tensor] = []
    register_inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    for row in rows:
        hidden_state, prompt_metadata = extract_hidden_state_from_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=str(row.get("prompt_text") or ""),
            device=device,
            max_tokens=max_prompt_tokens,
            add_special_tokens=False,
        )
        hidden_states.append(hidden_state.squeeze(0).detach().cpu())
        register_inputs.append(torch.tensor(row["register_inputs"], dtype=torch.float32))
        targets.append(torch.tensor(row["target_vector"], dtype=torch.float32))
        metadata.append(
            {
                "example_id": str(row.get("example_id") or ""),
                "task_name": str(row.get("task_name") or ""),
                "update_kind": str(row.get("update_kind") or "unknown"),
                "prompt_text": str(row.get("prompt_text") or ""),
                "hidden_state_source": str(prompt_metadata.get("hidden_state_source") or "unknown"),
                "prompt_token_count": int(prompt_metadata.get("prompt_token_count") or 0),
            }
        )
    return (
        torch.stack(hidden_states),
        torch.stack(register_inputs),
        torch.stack(targets),
        metadata,
    )


def _compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> ExecutableThoughtEvalMetrics:
    if predictions.numel() == 0 or targets.numel() == 0:
        return ExecutableThoughtEvalMetrics(
            num_examples=0,
            mse=0.0,
            mae=0.0,
            mean_cosine_similarity=0.0,
            mean_target_norm=0.0,
            mean_prediction_norm=0.0,
        )

    diff = predictions - targets
    mse = float(torch.mean(diff.pow(2)).item())
    mae = float(torch.mean(torch.abs(diff)).item())
    target_norms = torch.linalg.norm(targets, dim=1)
    prediction_norms = torch.linalg.norm(predictions, dim=1)
    dot = torch.sum(predictions * targets, dim=1)
    cosine = dot / torch.clamp(prediction_norms * target_norms, min=1e-8)
    cosine = torch.where(target_norms > 1e-8, cosine, torch.zeros_like(cosine))
    return ExecutableThoughtEvalMetrics(
        num_examples=int(targets.shape[0]),
        mse=mse,
        mae=mae,
        mean_cosine_similarity=float(torch.mean(cosine).item()),
        mean_target_norm=float(torch.mean(target_norms).item()),
        mean_prediction_norm=float(torch.mean(prediction_norms).item()),
    )


def _normalize_program_key(text: str) -> str:
    normalized = "\n".join(line.rstrip() for line in str(text or "").strip().splitlines())
    return normalized or "(empty)"


def _summarize_programs(
    *,
    program_texts: list[str],
    mog_previews: list[str],
    steps_executed: list[int],
    halted: list[bool],
    metadata: list[dict[str, Any]],
    top_k_programs: int,
    sample_limit: int,
) -> dict[str, Any]:
    num_examples = len(program_texts)
    if num_examples == 0:
        return {
            "num_examples": 0,
            "unique_programs": 0,
            "unique_mog_previews": 0,
            "dominant_program_fraction": 0.0,
            "mean_steps_executed": 0.0,
            "halt_rate": 0.0,
            "mean_prompt_token_count": 0.0,
            "hidden_state_source_counts": {},
            "top_programs": [],
            "task_update_consistency": {
                "num_groups": 0,
                "mean_majority_program_fraction": 0.0,
                "groups": [],
            },
            "samples": [],
        }

    program_keys = [_normalize_program_key(text) for text in program_texts]
    counts = Counter(program_keys)
    first_index_by_key: dict[str, int] = {}
    for index, key in enumerate(program_keys):
        first_index_by_key.setdefault(key, index)

    top_programs: list[dict[str, Any]] = []
    for key, count in counts.most_common(max(1, top_k_programs)):
        index = first_index_by_key[key]
        top_programs.append(
            {
                "count": int(count),
                "fraction": float(count / num_examples),
                "program_text": program_texts[index],
                "mog_preview": mog_previews[index],
                "sample_task_name": metadata[index]["task_name"],
                "sample_update_kind": metadata[index]["update_kind"],
            }
        )

    grouped: dict[tuple[str, str], list[int]] = {}
    for index, item in enumerate(metadata):
        grouped.setdefault((item["task_name"], item["update_kind"]), []).append(index)

    consistency_groups: list[dict[str, Any]] = []
    for (task_name, update_kind), indices in grouped.items():
        subset_counts = Counter(program_keys[index] for index in indices)
        majority_key, majority_count = subset_counts.most_common(1)[0]
        exemplar_index = first_index_by_key[majority_key]
        consistency_groups.append(
            {
                "task_name": task_name,
                "update_kind": update_kind,
                "num_examples": len(indices),
                "majority_program_fraction": float(majority_count / len(indices)),
                "majority_program_text": program_texts[exemplar_index],
            }
        )
    consistency_groups.sort(key=lambda item: (-item["num_examples"], -item["majority_program_fraction"], item["task_name"]))

    samples: list[dict[str, Any]] = []
    for index in range(min(num_examples, max(1, sample_limit))):
        samples.append(
            {
                "example_id": metadata[index]["example_id"],
                "task_name": metadata[index]["task_name"],
                "update_kind": metadata[index]["update_kind"],
                "hidden_state_source": metadata[index]["hidden_state_source"],
                "prompt_token_count": metadata[index]["prompt_token_count"],
                "prompt_excerpt": metadata[index]["prompt_text"][:160],
                "program_text": program_texts[index],
                "mog_preview": mog_previews[index],
                "steps_executed": int(steps_executed[index]),
                "halted": bool(halted[index]),
            }
        )

    hidden_state_source_counts = Counter(item["hidden_state_source"] for item in metadata)
    dominant_count = counts.most_common(1)[0][1]
    return {
        "num_examples": num_examples,
        "unique_programs": len(counts),
        "unique_mog_previews": len({_normalize_program_key(text) for text in mog_previews}),
        "dominant_program_fraction": float(dominant_count / num_examples),
        "mean_steps_executed": float(sum(int(value) for value in steps_executed) / num_examples),
        "halt_rate": float(sum(1 for value in halted if value) / num_examples),
        "mean_prompt_token_count": float(sum(item["prompt_token_count"] for item in metadata) / num_examples),
        "hidden_state_source_counts": dict(hidden_state_source_counts),
        "top_programs": top_programs,
        "task_update_consistency": {
            "num_groups": len(consistency_groups),
            "mean_majority_program_fraction": (
                0.0
                if not consistency_groups
                else float(sum(item["majority_program_fraction"] for item in consistency_groups) / len(consistency_groups))
            ),
            "groups": consistency_groups[: max(1, sample_limit)],
        },
        "samples": samples,
    }


def _subset_tensor(values: torch.Tensor, indices: list[int]) -> torch.Tensor:
    if not indices:
        return torch.zeros((0, values.shape[1]), dtype=values.dtype)
    index_tensor = torch.tensor(indices, dtype=torch.long)
    return values[index_tensor]


def _subset_list(values: list[Any], indices: list[int]) -> list[Any]:
    return [values[index] for index in indices]


def evaluate_executable_thought_head(
    *,
    train_path: str | Path,
    val_path: str | Path,
    checkpoint_path: str | Path,
    model_name_or_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    device: str = "cpu",
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    max_prompt_tokens: int = 2048,
    temperature: float = 0.35,
    top_k_programs: int = 8,
    sample_limit: int = 6,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Evaluate executable-thought patch prediction and decoded-program consistency."""
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    config_payload = dict(payload.get("config") or {})
    if "allowed_opcodes" in config_payload:
        config_payload["allowed_opcodes"] = tuple(config_payload["allowed_opcodes"])
    config = ExecutableThoughtHeadConfig(**config_payload) if config_payload else None
    head = load_executable_thought_head(
        path=checkpoint_path,
        device=device,
        config=config,
    )

    report: dict[str, Any] = {
        "checkpoint_path": str(Path(checkpoint_path)),
        "device": device,
        "model_name_or_path": (str(model_name_or_path) if model_name_or_path is not None else None),
        "config": config.to_dict() if config is not None else {},
        "checkpoint_metrics": dict(payload.get("metrics") or {}),
        "splits": {},
    }

    cached_model = model
    cached_tokenizer = tokenizer
    for split_name, split_path in (("train", train_path), ("val", val_path)):
        rows = _load_jsonl_rows(split_path)
        if not rows:
            zero_metrics = ExecutableThoughtEvalMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0).to_dict()
            report["splits"][split_name] = {
                "num_examples": 0,
                "baseline_zero": zero_metrics,
                "model": zero_metrics,
                "improvement": {
                    "mse_delta": 0.0,
                    "mae_delta": 0.0,
                    "cosine_delta": 0.0,
                    "relative_mse_improvement": 0.0,
                },
                "update_kind_breakdown": {},
                "program_summary": _summarize_programs(
                    program_texts=[],
                    mog_previews=[],
                    steps_executed=[],
                    halted=[],
                    metadata=[],
                    top_k_programs=top_k_programs,
                    sample_limit=sample_limit,
                ),
            }
            continue

        if cached_model is None or cached_tokenizer is None:
            cached_model, cached_tokenizer = _load_model_and_tokenizer(
                model_name_or_path=model_name_or_path,
                model=cached_model,
                tokenizer=cached_tokenizer,
                device=device,
                trust_remote_code=trust_remote_code,
            )

        hidden_states, register_inputs, targets, metadata = _tensorize_rows(
            rows=rows,
            model=cached_model,
            tokenizer=cached_tokenizer,
            device=device,
            max_prompt_tokens=max_prompt_tokens,
        )

        with torch.no_grad():
            result = head(
                hidden_states.to(device),
                register_inputs.to(device),
                temperature=temperature,
            )
            predictions = result.patch_signal.detach().cpu()

        zero_baseline = torch.zeros_like(targets)
        baseline_metrics = _compute_metrics(zero_baseline, targets)
        model_metrics = _compute_metrics(predictions, targets)
        improvement = {
            "mse_delta": baseline_metrics.mse - model_metrics.mse,
            "mae_delta": baseline_metrics.mae - model_metrics.mae,
            "cosine_delta": model_metrics.mean_cosine_similarity - baseline_metrics.mean_cosine_similarity,
            "relative_mse_improvement": (
                0.0 if baseline_metrics.mse <= 1e-12 else (baseline_metrics.mse - model_metrics.mse) / baseline_metrics.mse
            ),
        }

        update_kind_groups: dict[str, list[int]] = {}
        for index, item in enumerate(metadata):
            update_kind_groups.setdefault(item["update_kind"], []).append(index)

        update_kind_breakdown: dict[str, dict[str, Any]] = {}
        for update_kind, indices in sorted(update_kind_groups.items()):
            baseline_subset = _subset_tensor(zero_baseline, indices)
            target_subset = _subset_tensor(targets, indices)
            prediction_subset = _subset_tensor(predictions, indices)
            baseline_subset_metrics = _compute_metrics(baseline_subset, target_subset)
            model_subset_metrics = _compute_metrics(prediction_subset, target_subset)
            update_kind_breakdown[update_kind] = {
                "num_examples": len(indices),
                "baseline_zero": baseline_subset_metrics.to_dict(),
                "model": model_subset_metrics.to_dict(),
                "relative_mse_improvement": (
                    0.0
                    if baseline_subset_metrics.mse <= 1e-12
                    else (baseline_subset_metrics.mse - model_subset_metrics.mse) / baseline_subset_metrics.mse
                ),
                "program_summary": _summarize_programs(
                    program_texts=_subset_list(result.program_texts, indices),
                    mog_previews=_subset_list(result.mog_previews, indices),
                    steps_executed=_subset_list(result.steps_executed, indices),
                    halted=_subset_list(result.halted, indices),
                    metadata=_subset_list(metadata, indices),
                    top_k_programs=top_k_programs,
                    sample_limit=sample_limit,
                ),
            }

        report["splits"][split_name] = {
            "num_examples": len(rows),
            "baseline_zero": baseline_metrics.to_dict(),
            "model": model_metrics.to_dict(),
            "improvement": improvement,
            "update_kind_breakdown": update_kind_breakdown,
            "program_summary": _summarize_programs(
                program_texts=result.program_texts,
                mog_previews=result.mog_previews,
                steps_executed=result.steps_executed,
                halted=result.halted,
                metadata=metadata,
                top_k_programs=top_k_programs,
                sample_limit=sample_limit,
            ),
        }

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate an executable-thought head on real prompt-derived hidden states")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--output-path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--top-k-programs", type=int, default=8)
    parser.add_argument("--sample-limit", type=int, default=6)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    result = evaluate_executable_thought_head(
        train_path=args.train_path,
        val_path=args.val_path,
        checkpoint_path=args.checkpoint_path,
        model_name_or_path=args.model_name_or_path,
        output_path=args.output_path,
        device=args.device,
        max_prompt_tokens=args.max_prompt_tokens,
        temperature=args.temperature,
        top_k_programs=args.top_k_programs,
        sample_limit=args.sample_limit,
        trust_remote_code=args.trust_remote_code,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
