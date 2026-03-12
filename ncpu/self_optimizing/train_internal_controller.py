"""Train internal controller adapters from hidden SOME trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Optional


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ncpu.self_optimizing.controller_training import build_controller_training_bundle
from ncpu.self_optimizing.controller_bundle import (
    ControllerBundle,
    ControllerComponentConfig,
    save_controller_bundle,
)
from ncpu.self_optimizing.latent_action_policy import LatentActionHeadConfig
from ncpu.self_optimizing.latent_action_training import train_latent_action_head
from ncpu.self_optimizing.latent_descriptor_head import LatentDescriptorHeadConfig
from ncpu.self_optimizing.latent_descriptor_training import train_latent_descriptor_head
from ncpu.self_optimizing.latent_halt_policy import LatentHaltHeadConfig
from ncpu.self_optimizing.latent_halt_training import train_latent_halt_head
from ncpu.self_optimizing.latent_memory_head import LatentMemoryHeadConfig
from ncpu.self_optimizing.latent_memory_training import train_latent_memory_head
from ncpu.self_optimizing.state_patch_head import StatePatchHeadConfig
from ncpu.self_optimizing.state_patch_training import train_state_patch_head
from ncpu.self_optimizing.weight_cpu_blueprint import (
    build_default_weight_cpu_blueprint,
    save_weight_cpu_blueprint,
)


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def write_controller_bundle(
    *,
    output_root: Path,
    base_model: str,
    dataset_dir: Path,
    response_dir: Path,
    action_dir: Optional[Path],
    training_run_path: Optional[Path],
    prepared_only: bool,
    response_provider: str = "hf_local",
    action_provider: str = "hf_local",
    response_provider_kwargs: Optional[dict[str, Any]] = None,
    action_provider_kwargs: Optional[dict[str, Any]] = None,
    latent_action_head_path: Optional[Path] = None,
    latent_action_head_config: Optional[dict[str, Any]] = None,
    latent_descriptor_head_path: Optional[Path] = None,
    latent_descriptor_head_config: Optional[dict[str, Any]] = None,
    latent_memory_head_path: Optional[Path] = None,
    latent_memory_head_config: Optional[dict[str, Any]] = None,
    latent_halt_head_path: Optional[Path] = None,
    latent_halt_head_config: Optional[dict[str, Any]] = None,
    state_patch_head_path: Optional[Path] = None,
    state_patch_head_config: Optional[dict[str, Any]] = None,
) -> Path:
    """Write a portable controller bundle manifest for inference/evaluation."""
    bundle = ControllerBundle(
        name=output_root.name,
        base_model=base_model,
        response=ControllerComponentConfig(
            provider=response_provider,
            model=str(response_dir),
            temperature=0.0,
            provider_kwargs=dict(response_provider_kwargs or {}),
        ),
        action=(
            ControllerComponentConfig(
                provider=action_provider,
                model=str(action_dir),
                temperature=0.0,
                provider_kwargs=dict(action_provider_kwargs or {}),
            )
            if action_dir is not None
            else None
        ),
        prepared_dataset_dir=str(dataset_dir),
        training_run_path=str(training_run_path) if training_run_path is not None else None,
        latent_action_head_path=str(latent_action_head_path) if latent_action_head_path is not None else None,
        latent_action_head_config=dict(latent_action_head_config or {}) or None,
        latent_memory_head_path=str(latent_memory_head_path) if latent_memory_head_path is not None else None,
        latent_memory_head_config=dict(latent_memory_head_config or {}) or None,
        latent_halt_head_path=str(latent_halt_head_path) if latent_halt_head_path is not None else None,
        latent_halt_head_config=dict(latent_halt_head_config or {}) or None,
        metadata={
            "prepared_only": prepared_only,
            "response_adapter_exists": response_dir.exists(),
            "action_adapter_exists": action_dir.exists() if action_dir is not None else False,
            "latent_action_head_exists": latent_action_head_path.exists() if latent_action_head_path is not None else False,
            "latent_memory_head_exists": latent_memory_head_path.exists() if latent_memory_head_path is not None else False,
            "latent_halt_head_exists": latent_halt_head_path.exists() if latent_halt_head_path is not None else False,
        },
    )
    filename = "controller_bundle.template.json" if prepared_only else "controller_bundle.json"
    return save_controller_bundle(bundle, output_root / filename)


def write_fast_weight_controller_bundle(
    *,
    output_root: Path,
    base_model: str,
    dataset_dir: Path,
    response_dir: Path,
    action_dir: Optional[Path],
    training_run_path: Optional[Path],
    prepared_only: bool,
    fast_weights_rank: int,
    fast_weights_learning_rate: float,
    fast_weights_gradient_steps: int,
    fast_weights_adapter_scale: float,
    fast_weights_max_target_tokens: int,
    fast_weights_target_modules: list[str],
    latent_action_head_path: Optional[Path] = None,
    latent_action_head_config: Optional[dict[str, Any]] = None,
    latent_descriptor_head_path: Optional[Path] = None,
    latent_descriptor_head_config: Optional[dict[str, Any]] = None,
    latent_memory_head_path: Optional[Path] = None,
    latent_memory_head_config: Optional[dict[str, Any]] = None,
    latent_halt_head_path: Optional[Path] = None,
    latent_halt_head_config: Optional[dict[str, Any]] = None,
    state_patch_head_path: Optional[Path] = None,
    state_patch_head_config: Optional[dict[str, Any]] = None,
) -> Path:
    """Write a bundle variant that runs the response model with task-local fast weights."""
    filename = (
        "controller_bundle.fast_weights.template.json"
        if prepared_only
        else "controller_bundle.fast_weights.json"
    )
    bundle = ControllerBundle(
        name=f"{output_root.name}-fast-weights",
        base_model=base_model,
        response=ControllerComponentConfig(
            provider="hf_fast_weights",
            model=str(response_dir),
            temperature=0.0,
            provider_kwargs={
                "fast_weights_rank": fast_weights_rank,
                "fast_weights_learning_rate": fast_weights_learning_rate,
                "fast_weights_gradient_steps": fast_weights_gradient_steps,
                "fast_weights_adapter_scale": fast_weights_adapter_scale,
                "fast_weights_max_target_tokens": fast_weights_max_target_tokens,
                "fast_weights_target_modules": list(fast_weights_target_modules),
                "fast_weights_use_ncpu_adaptation": True,
                "fast_weights_ncpu_compression_type": "top_k",
                "fast_weights_ncpu_top_k_ratio": 0.1,
                "fast_weights_ncpu_gradient_clip": 1.0,
                "fast_weights_ncpu_max_gradient_steps": 3,
                "latent_descriptor_head_path": str(latent_descriptor_head_path) if latent_descriptor_head_path is not None else None,
                "latent_descriptor_head_config": dict(latent_descriptor_head_config or {}) or None,
                "state_patch_head_enabled": state_patch_head_path is None,
                "state_patch_head_path": str(state_patch_head_path) if state_patch_head_path is not None else None,
                "state_patch_head_config": dict(state_patch_head_config or {}) or None,
                "state_patch_head_input_dim": (state_patch_head_config or {}).get("input_dim", 16),
                "state_patch_head_hidden_dim": (state_patch_head_config or {}).get("hidden_dim", 64),
                "state_patch_head_output_dim": (state_patch_head_config or {}).get("output_dim", 16),
                "decode_backend": "segmented_kv",
                "segmented_cache_recent_window_tokens": 256,
                "segmented_cache_commit_segment_tokens": 128,
                "segmented_cache_descriptor_tokens_per_segment": 4,
                "segmented_cache_max_memory_segments": 16,
                "segmented_cache_min_prompt_tokens_for_compression": 384,
                "segmented_cache_min_tokens_to_commit": 64,
            },
        ),
        action=(
            ControllerComponentConfig(
                provider="hf_local",
                model=str(action_dir),
                temperature=0.0,
            )
            if action_dir is not None
            else None
        ),
        prepared_dataset_dir=str(dataset_dir),
        training_run_path=str(training_run_path) if training_run_path is not None else None,
        latent_action_head_path=str(latent_action_head_path) if latent_action_head_path is not None else None,
        latent_action_head_config=dict(latent_action_head_config or {}) or None,
        latent_memory_head_path=str(latent_memory_head_path) if latent_memory_head_path is not None else None,
        latent_memory_head_config=dict(latent_memory_head_config or {}) or None,
        latent_halt_head_path=str(latent_halt_head_path) if latent_halt_head_path is not None else None,
        latent_halt_head_config=dict(latent_halt_head_config or {}) or None,
        controller_config={
            "plan_before_generate": True,
            "allow_unverified_commit": False,
            "commit_on_first_success": True,
            "fast_weight_updates_on_plan": False,
            "fast_weight_updates_on_repair_plan": False,
            "fast_weight_updates_on_verify_failure": False,
            "fast_weight_updates_on_verify_success": False,
            "descriptor_updates_on_plan": True,
            "descriptor_updates_on_verify_failure": True,
            "max_descriptor_updates_per_task": 8,
            "prefer_latent_memory_updater": True,
        },
        metadata={
            "prepared_only": prepared_only,
            "response_adapter_exists": response_dir.exists(),
            "action_adapter_exists": action_dir.exists() if action_dir is not None else False,
            "latent_action_head_exists": latent_action_head_path.exists() if latent_action_head_path is not None else False,
            "latent_descriptor_head_exists": (
                latent_descriptor_head_path.exists() if latent_descriptor_head_path is not None else False
            ),
            "latent_memory_head_exists": latent_memory_head_path.exists() if latent_memory_head_path is not None else False,
            "latent_halt_head_exists": latent_halt_head_path.exists() if latent_halt_head_path is not None else False,
            "state_patch_head_exists": state_patch_head_path.exists() if state_patch_head_path is not None else False,
            "runtime_variant": "response_fast_weights_descriptor_first+segmented_kv_decode",
        },
    )
    return save_controller_bundle(bundle, output_root / filename)


def write_weight_cpu_blueprint(
    *,
    output_root: Path,
    controller_bundle_path: Path,
    prepared_only: bool,
) -> Path:
    """Write a research-backed blueprint for the next architecture stage."""
    blueprint = build_default_weight_cpu_blueprint(
        name=output_root.name,
        controller_bundle_path=str(controller_bundle_path),
    )
    filename = "weight_cpu_blueprint.template.json" if prepared_only else "weight_cpu_blueprint.json"
    return save_weight_cpu_blueprint(blueprint, output_root / filename)


def format_messages(messages: list[dict[str, str]], tokenizer: Optional[Any] = None) -> str:
    """Format chat messages for causal LM SFT."""
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass

    parts: list[str] = []
    for message in messages:
        role = message["role"].upper()
        parts.append(f"{role}:\n{message['content']}")
    return "\n\n".join(parts)


def determine_device() -> str:
    """Select the most capable available training device."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_and_prepare_model(
    *,
    base_model: str,
    device: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
) -> tuple[Any, Any]:
    """Load a base model and apply LoRA adapters."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        device_map=None,
    )
    model = model.to(device)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def prepare_dataset(
    *,
    train_path: str,
    val_path: str,
    tokenizer: Any,
    max_seq_length: int,
) -> tuple[Any, Any]:
    """Load and tokenize train/validation datasets."""
    from datasets import load_dataset

    dataset = load_dataset("json", data_files={"train": train_path, "validation": val_path})

    def tokenize_function(batch: dict[str, list]) -> dict[str, Any]:
        texts = [format_messages(messages, tokenizer=tokenizer) for messages in batch["messages"]]
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

    tokenized_train = dataset["train"].map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    tokenized_val = dataset["validation"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )
    return tokenized_train, tokenized_val


def train_adapter(
    *,
    base_model: str,
    train_path: str,
    val_path: str,
    output_dir: Path,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
    resume: bool,
) -> dict[str, Any]:
    """Train a single LoRA adapter on one controller objective."""
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    model, tokenizer = load_and_prepare_model(
        base_model=base_model,
        device=device,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    train_dataset, val_dataset = prepare_dataset(
        train_path=train_path,
        val_path=val_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=False,
        bf16=False,
        report_to="none",
        dataloader_num_workers=0,
        use_cpu=(device == "cpu"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    start = time.time()
    if resume:
        checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[1]))
        if checkpoints:
            trainer.train(resume_from_checkpoint=str(checkpoints[-1]))
        else:
            trainer.train()
    else:
        trainer.train()
    elapsed = time.time() - start

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metrics = {
        "elapsed_seconds": elapsed,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "output_dir": str(output_dir),
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train internal controller adapters from hidden SOME trajectories")
    parser.add_argument("--trajectory-root", required=True, help="Root directory of hidden trajectory JSONL files")
    parser.add_argument("--output-dir", required=True, help="Directory for prepared datasets and trained adapters")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base causal LM to fine-tune")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare datasets and manifest, skip training")
    parser.add_argument(
        "--train-target",
        choices=["response", "action", "both"],
        default="response",
        help="Which controller objective to train",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--latent-action-epochs", type=int, default=12)
    parser.add_argument("--latent-action-batch-size", type=int, default=16)
    parser.add_argument("--latent-action-learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-action-hidden-dim", type=int, default=64)
    parser.add_argument("--latent-action-dropout", type=float, default=0.0)
    parser.add_argument("--latent-descriptor-epochs", type=int, default=12)
    parser.add_argument("--latent-descriptor-batch-size", type=int, default=16)
    parser.add_argument("--latent-descriptor-learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-descriptor-hidden-dim", type=int, default=64)
    parser.add_argument("--latent-descriptor-output-dim", type=int, default=16)
    parser.add_argument("--latent-descriptor-dropout", type=float, default=0.0)
    parser.add_argument("--latent-memory-epochs", type=int, default=12)
    parser.add_argument("--latent-memory-batch-size", type=int, default=16)
    parser.add_argument("--latent-memory-learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-memory-hidden-dim", type=int, default=64)
    parser.add_argument("--latent-memory-output-dim", type=int, default=16)
    parser.add_argument("--latent-memory-dropout", type=float, default=0.0)
    parser.add_argument("--latent-halt-epochs", type=int, default=12)
    parser.add_argument("--latent-halt-batch-size", type=int, default=16)
    parser.add_argument("--latent-halt-learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-halt-hidden-dim", type=int, default=48)
    parser.add_argument("--latent-halt-dropout", type=float, default=0.0)
    parser.add_argument("--state-patch-epochs", type=int, default=12)
    parser.add_argument("--state-patch-batch-size", type=int, default=16)
    parser.add_argument("--state-patch-learning-rate", type=float, default=1e-3)
    parser.add_argument("--state-patch-input-dim", type=int, default=16)
    parser.add_argument("--state-patch-hidden-dim", type=int, default=64)
    parser.add_argument("--state-patch-output-dim", type=int, default=16)
    parser.add_argument("--state-patch-dropout", type=float, default=0.0)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-failed-response-steps", action="store_true")
    parser.add_argument("--exclude-think-steps", action="store_true")
    parser.add_argument("--allow-unverified-trajectories", action="store_true")
    parser.add_argument("--skip-action-policy-export", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default=",".join(DEFAULT_TARGET_MODULES),
        help="Comma-separated LoRA target modules",
    )
    parser.add_argument("--fast-weights-rank", type=int, default=8)
    parser.add_argument("--fast-weights-learning-rate", type=float, default=5e-3)
    parser.add_argument("--fast-weights-gradient-steps", type=int, default=1)
    parser.add_argument("--fast-weights-adapter-scale", type=float, default=1.0)
    parser.add_argument("--fast-weights-max-target-tokens", type=int, default=256)
    parser.add_argument(
        "--fast-weights-target-modules",
        default=",".join(DEFAULT_TARGET_MODULES),
        help="Comma-separated target modules for task-local fast-weight updates",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_dir)
    dataset_dir = output_root / "prepared_datasets"
    adapters_dir = output_root / "adapters"
    latent_action_head_path = output_root / "latent_action_head.pt"
    latent_descriptor_head_path = output_root / "latent_descriptor_head.pt"
    latent_memory_head_path = output_root / "latent_memory_head.pt"
    latent_halt_head_path = output_root / "latent_halt_head.pt"
    state_patch_head_path = output_root / "state_patch_head.pt"
    latent_action_head_config = LatentActionHeadConfig(
        hidden_dim=args.latent_action_hidden_dim,
        dropout=args.latent_action_dropout,
    ).to_dict()
    latent_descriptor_head_config = LatentDescriptorHeadConfig(
        hidden_dim=args.latent_descriptor_hidden_dim,
        output_dim=args.latent_descriptor_output_dim,
        dropout=args.latent_descriptor_dropout,
    ).to_dict()
    latent_memory_head_config = LatentMemoryHeadConfig(
        hidden_dim=args.latent_memory_hidden_dim,
        output_dim=args.latent_memory_output_dim,
        dropout=args.latent_memory_dropout,
    ).to_dict()
    latent_halt_head_config = LatentHaltHeadConfig(
        hidden_dim=args.latent_halt_hidden_dim,
        dropout=args.latent_halt_dropout,
    ).to_dict()
    state_patch_head_config = StatePatchHeadConfig(
        input_dim=args.state_patch_input_dim,
        hidden_dim=args.state_patch_hidden_dim,
        output_dim=args.state_patch_output_dim,
        dropout=args.state_patch_dropout,
    ).to_dict()
    fast_weight_target_modules = [
        item.strip() for item in args.fast_weights_target_modules.split(",") if item.strip()
    ]

    bundle = build_controller_training_bundle(
        args.trajectory_root,
        dataset_dir,
        include_failed_response_steps=args.include_failed_response_steps,
        include_think_steps=not args.exclude_think_steps,
        allow_unverified_trajectories=args.allow_unverified_trajectories,
        include_action_policy=not args.skip_action_policy_export,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    if args.prepare_only:
        bundle_path = write_controller_bundle(
            output_root=output_root,
            base_model=args.base_model,
            dataset_dir=dataset_dir,
            response_dir=adapters_dir / "response",
            action_dir=None if args.skip_action_policy_export else adapters_dir / "action",
            training_run_path=None,
            prepared_only=True,
            latent_action_head_path=latent_action_head_path,
            latent_action_head_config=latent_action_head_config,
            latent_descriptor_head_path=latent_descriptor_head_path,
            latent_descriptor_head_config=latent_descriptor_head_config,
            latent_memory_head_path=latent_memory_head_path,
            latent_memory_head_config=latent_memory_head_config,
            latent_halt_head_path=latent_halt_head_path,
            latent_halt_head_config=latent_halt_head_config,
        )
        fast_weight_bundle_path = write_fast_weight_controller_bundle(
            output_root=output_root,
            base_model=args.base_model,
            dataset_dir=dataset_dir,
            response_dir=adapters_dir / "response",
            action_dir=None if args.skip_action_policy_export else adapters_dir / "action",
            training_run_path=None,
            prepared_only=True,
            fast_weights_rank=args.fast_weights_rank,
            fast_weights_learning_rate=args.fast_weights_learning_rate,
            fast_weights_gradient_steps=args.fast_weights_gradient_steps,
            fast_weights_adapter_scale=args.fast_weights_adapter_scale,
            fast_weights_max_target_tokens=args.fast_weights_max_target_tokens,
            fast_weights_target_modules=fast_weight_target_modules,
            latent_action_head_path=latent_action_head_path,
            latent_action_head_config=latent_action_head_config,
            latent_descriptor_head_path=latent_descriptor_head_path,
            latent_descriptor_head_config=latent_descriptor_head_config,
            latent_memory_head_path=latent_memory_head_path,
            latent_memory_head_config=latent_memory_head_config,
            latent_halt_head_path=latent_halt_head_path,
            latent_halt_head_config=latent_halt_head_config,
            state_patch_head_path=state_patch_head_path,
            state_patch_head_config=state_patch_head_config,
        )
        blueprint_path = write_weight_cpu_blueprint(
            output_root=output_root,
            controller_bundle_path=bundle_path,
            prepared_only=True,
        )
        print(f"Prepared controller training bundle at {dataset_dir}")
        print(f"Controller bundle template written to {bundle_path}")
        print(f"Fast-weight controller bundle template written to {fast_weight_bundle_path}")
        print(f"Weight CPU blueprint template written to {blueprint_path}")
        return 0

    device = determine_device()
    print(f"Using device: {device}")
    targets = ["response", "action"] if args.train_target == "both" else [args.train_target]
    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]

    metrics: dict[str, Any] = {
        "base_model": args.base_model,
        "device": device,
        "targets": {},
    }

    for target in targets:
        if target == "response":
            train_path = bundle.response_train_path
            val_path = bundle.response_val_path
        else:
            if not bundle.action_train_path or not bundle.action_val_path:
                raise RuntimeError("Action-policy dataset was not generated; remove --skip-action-policy-export")
            train_path = bundle.action_train_path
            val_path = bundle.action_val_path

        metrics["targets"][target] = train_adapter(
            base_model=args.base_model,
            train_path=train_path,
            val_path=val_path,
            output_dir=adapters_dir / target,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_length,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            resume=args.resume,
        )

    if bundle.latent_action_train_path and bundle.latent_action_val_path:
        metrics["latent_action_head"] = train_latent_action_head(
            train_path=bundle.latent_action_train_path,
            val_path=bundle.latent_action_val_path,
            output_path=latent_action_head_path,
            config=LatentActionHeadConfig(
                hidden_dim=args.latent_action_hidden_dim,
                dropout=args.latent_action_dropout,
            ),
            epochs=args.latent_action_epochs,
            batch_size=args.latent_action_batch_size,
            learning_rate=args.latent_action_learning_rate,
            device=device,
        )
    if bundle.latent_descriptor_train_path and bundle.latent_descriptor_val_path:
        metrics["latent_descriptor_head"] = train_latent_descriptor_head(
            train_path=bundle.latent_descriptor_train_path,
            val_path=bundle.latent_descriptor_val_path,
            output_path=latent_descriptor_head_path,
            config=LatentDescriptorHeadConfig(
                hidden_dim=args.latent_descriptor_hidden_dim,
                output_dim=args.latent_descriptor_output_dim,
                dropout=args.latent_descriptor_dropout,
            ),
            epochs=args.latent_descriptor_epochs,
            batch_size=args.latent_descriptor_batch_size,
            learning_rate=args.latent_descriptor_learning_rate,
            device=device,
        )
    if bundle.latent_memory_train_path and bundle.latent_memory_val_path:
        metrics["latent_memory_head"] = train_latent_memory_head(
            train_path=bundle.latent_memory_train_path,
            val_path=bundle.latent_memory_val_path,
            output_path=latent_memory_head_path,
            config=LatentMemoryHeadConfig(
                hidden_dim=args.latent_memory_hidden_dim,
                output_dim=args.latent_memory_output_dim,
                dropout=args.latent_memory_dropout,
            ),
            epochs=args.latent_memory_epochs,
            batch_size=args.latent_memory_batch_size,
            learning_rate=args.latent_memory_learning_rate,
            device=device,
        )
    if bundle.latent_halt_train_path and bundle.latent_halt_val_path:
        metrics["latent_halt_head"] = train_latent_halt_head(
            train_path=bundle.latent_halt_train_path,
            val_path=bundle.latent_halt_val_path,
            output_path=latent_halt_head_path,
            config=LatentHaltHeadConfig(
                hidden_dim=args.latent_halt_hidden_dim,
                dropout=args.latent_halt_dropout,
            ),
            epochs=args.latent_halt_epochs,
            batch_size=args.latent_halt_batch_size,
            learning_rate=args.latent_halt_learning_rate,
            device=device,
        )
    if bundle.state_patch_train_path and bundle.state_patch_val_path:
        metrics["state_patch_head"] = train_state_patch_head(
            train_path=bundle.state_patch_train_path,
            val_path=bundle.state_patch_val_path,
            output_path=state_patch_head_path,
            config=StatePatchHeadConfig(
                input_dim=args.state_patch_input_dim,
                hidden_dim=args.state_patch_hidden_dim,
                output_dim=args.state_patch_output_dim,
                dropout=args.state_patch_dropout,
            ),
            epochs=args.state_patch_epochs,
            batch_size=args.state_patch_batch_size,
            learning_rate=args.state_patch_learning_rate,
            device=device,
        )

    metrics_path = output_root / "training_run.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bundle_path = write_controller_bundle(
        output_root=output_root,
        base_model=args.base_model,
        dataset_dir=dataset_dir,
        response_dir=adapters_dir / "response",
        action_dir=(adapters_dir / "action") if "action" in metrics["targets"] else None,
        training_run_path=metrics_path,
        prepared_only=False,
        latent_action_head_path=latent_action_head_path if latent_action_head_path.exists() else None,
        latent_action_head_config=latent_action_head_config,
        latent_descriptor_head_path=latent_descriptor_head_path if latent_descriptor_head_path.exists() else None,
        latent_descriptor_head_config=latent_descriptor_head_config,
        latent_memory_head_path=latent_memory_head_path if latent_memory_head_path.exists() else None,
        latent_memory_head_config=latent_memory_head_config,
        latent_halt_head_path=latent_halt_head_path if latent_halt_head_path.exists() else None,
        latent_halt_head_config=latent_halt_head_config,
    )
    fast_weight_bundle_path = write_fast_weight_controller_bundle(
        output_root=output_root,
        base_model=args.base_model,
        dataset_dir=dataset_dir,
        response_dir=adapters_dir / "response",
        action_dir=(adapters_dir / "action") if "action" in metrics["targets"] else None,
        training_run_path=metrics_path,
        prepared_only=False,
        fast_weights_rank=args.fast_weights_rank,
        fast_weights_learning_rate=args.fast_weights_learning_rate,
        fast_weights_gradient_steps=args.fast_weights_gradient_steps,
        fast_weights_adapter_scale=args.fast_weights_adapter_scale,
        fast_weights_max_target_tokens=args.fast_weights_max_target_tokens,
        fast_weights_target_modules=fast_weight_target_modules,
        latent_action_head_path=latent_action_head_path if latent_action_head_path.exists() else None,
        latent_action_head_config=latent_action_head_config,
        latent_descriptor_head_path=latent_descriptor_head_path if latent_descriptor_head_path.exists() else None,
        latent_descriptor_head_config=latent_descriptor_head_config,
        latent_memory_head_path=latent_memory_head_path if latent_memory_head_path.exists() else None,
        latent_memory_head_config=latent_memory_head_config,
        latent_halt_head_path=latent_halt_head_path if latent_halt_head_path.exists() else None,
        latent_halt_head_config=latent_halt_head_config,
        state_patch_head_path=state_patch_head_path if state_patch_head_path.exists() else None,
        state_patch_head_config=state_patch_head_config,
    )
    blueprint_path = write_weight_cpu_blueprint(
        output_root=output_root,
        controller_bundle_path=bundle_path,
        prepared_only=False,
    )
    print(f"Training metrics written to {metrics_path}")
    print(f"Controller bundle written to {bundle_path}")
    print(f"Fast-weight controller bundle written to {fast_weight_bundle_path}")
    print(f"Weight CPU blueprint written to {blueprint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
