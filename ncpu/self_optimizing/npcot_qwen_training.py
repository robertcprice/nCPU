"""Train NCPUCoprocessorMLPWithArrayThought on a real Qwen3.5 model.

This is the "NPCoT built into the model" training pipeline. It:

1. Loads Qwen3.5-4B (or another size) with VL-aware text-only extraction.
2. Wraps the last N MLP sublayers with NCPUCoprocessorMLPWithArrayThought.
3. Freezes everything except the coprocessor + array-thought parameters
   (so we train ~1-2% of total params).
4. Attaches a `ProgramLibrarySession` to the array-thought modules. The
   library crystallizes during training as hidden-state patterns converge.
5. Trains on a coding dataset (MBPP train split by default — short code
   completions whose reasoning shape is dense with reductions).
6. Saves: (a) coprocessor state-dict + (b) library JSON.

At inference (separate pass — use `humaneval_runner --library ...`), the
saved library attaches to the same wrapped Qwen model and the library
fast path fires on per-token hidden states that match the trained
signatures.

Usage:

    python3 -m ncpu.self_optimizing.npcot_qwen_training \\
        --model Qwen/Qwen3.5-4B \\
        --target-layers -2,-1 \\
        --max-steps 500 \\
        --batch-size 2 \\
        --out-checkpoint /workspace/checkpoints/npcot_qwen3.5-4B.pt \\
        --out-library /workspace/checkpoints/npcot_qwen3.5-4B_library.json

After training:

    python3 -m ncpu.self_optimizing.humaneval_runner \\
        --model Qwen/Qwen3.5-4B \\
        --library /workspace/checkpoints/npcot_qwen3.5-4B_library.json \\
        --target-layers -2,-1 \\
        --out /workspace/reports/humaneval_qwen3.5-4B_npcot.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class QwenNPCoTTrainingConfig:
    model: str = "Qwen/Qwen3.5-4B"
    target_layers: list[int] = field(default_factory=lambda: [-2, -1])
    max_steps: int = 500
    batch_size: int = 2
    max_seq_len: int = 512
    learning_rate: float = 5e-4
    warmup_steps: int = 20
    dataset: str = "mbpp"            # mbpp | gsm8k | synthetic
    dataset_split: str = "train"
    array_max_len: int = 8
    array_thought_max_gate: float = 0.05
    convergence_gap_threshold: float = 0.5
    out_checkpoint: Path = Path("/workspace/checkpoints/npcot_qwen.pt")
    out_library: Path = Path("/workspace/checkpoints/npcot_qwen_library.json")
    log_every: int = 10
    trust_remote_code: bool = False
    device: str = "auto"


def parse_cli(argv: list[str] | None = None) -> QwenNPCoTTrainingConfig:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--target-layers", default="-2,-1")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--dataset", choices=["mbpp", "gsm8k"], default="mbpp")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--array-max-len", type=int, default=8)
    p.add_argument("--array-thought-max-gate", type=float, default=0.05)
    p.add_argument("--convergence-gap-threshold", type=float, default=0.5)
    p.add_argument("--out-checkpoint", type=Path, default=Path("/workspace/checkpoints/npcot_qwen.pt"))
    p.add_argument("--out-library", type=Path, default=Path("/workspace/checkpoints/npcot_qwen_library.json"))
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--device", default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args(argv)
    return QwenNPCoTTrainingConfig(
        model=args.model,
        target_layers=[int(x) for x in args.target_layers.split(",") if x.strip()],
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        dataset=args.dataset,
        dataset_split=args.dataset_split,
        array_max_len=args.array_max_len,
        array_thought_max_gate=args.array_thought_max_gate,
        convergence_gap_threshold=args.convergence_gap_threshold,
        out_checkpoint=args.out_checkpoint,
        out_library=args.out_library,
        log_every=args.log_every,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )


# ---------------------------------------------------------------------------
# Dataset loaders — short-sequence coding examples
# ---------------------------------------------------------------------------


def load_mbpp_training_texts(split: str = "train") -> list[str]:
    """Load MBPP short-form text+code pairs as flat training strings."""
    from datasets import load_dataset

    ds = load_dataset("mbpp", split=split)
    texts: list[str] = []
    for row in ds:
        text = row.get("text") or ""
        code = row.get("code") or ""
        tests = "\n".join(row.get("test_list") or [])
        joined = f"# {text}\n{code}\n{tests}\n"
        texts.append(joined)
    return texts


def load_gsm8k_training_texts(split: str = "train") -> list[str]:
    """GSM8K training split — math word problems with reductions."""
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split=split)
    return [f"{row['question']}\n{row['answer']}\n" for row in ds]


def load_dataset_texts(dataset: str, split: str) -> list[str]:
    if dataset == "mbpp":
        return load_mbpp_training_texts(split)
    if dataset == "gsm8k":
        return load_gsm8k_training_texts(split)
    raise ValueError(f"unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def build_wrapped_model(
    cfg: QwenNPCoTTrainingConfig,
) -> tuple[Any, Any, str, list[Any]]:
    """Load Qwen, wrap target MLP layers, freeze base, return handles."""
    import torch
    from transformers import AutoTokenizer

    from ncpu.coprocessor.array_thought_coprocessor import (
        ArrayThoughtCoprocessorConfig,
        NCPUCoprocessorMLPWithArrayThought,
    )
    from ncpu.coprocessor.config import NCPUCoprocessorConfig
    from ncpu.self_optimizing.humaneval_runner import _load_hf_model_vl_aware

    device = cfg.device if cfg.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    # Use the model's native bf16. The NPCoT wrapper correctly propagates
    # bf16 end-to-end since the `_resolve_array_inputs` dtype preservation
    # fix (see `array_executable_thought_head.py::_resolve_array_inputs`).
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model, trust_remote_code=cfg.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_hf_model_vl_aware(
        cfg.model, dtype=dtype, trust_remote_code=cfg.trust_remote_code
    ).to(device)

    # Freeze all base params.
    for param in model.parameters():
        param.requires_grad = False

    # Wrap target MLP layers with NPCoT coprocessor.
    hidden_dim = int(getattr(model.config, "hidden_size", 0) or 0)
    if hidden_dim <= 0:
        raise ValueError("could not infer hidden_size from model config")
    layers = model.model.layers
    n_layers = len(layers)

    coproc_cfg = NCPUCoprocessorConfig(
        n_bits=8, num_ops=7, max_gate=0.1, residual_init_scale=0.0,
    )
    array_cfg = ArrayThoughtCoprocessorConfig(
        array_max_len=cfg.array_max_len,
        max_gate=cfg.array_thought_max_gate,
    )

    wrapped_handles: list[Any] = []
    for raw_idx in cfg.target_layers:
        idx = raw_idx if raw_idx >= 0 else n_layers + raw_idx
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"target layer {raw_idx} out of range [0, {n_layers})")
        original_mlp = layers[idx].mlp
        wrapper = NCPUCoprocessorMLPWithArrayThought(
            original_mlp=original_mlp,
            hidden_dim=hidden_dim,
            config=coproc_cfg,
            array_thought_config=array_cfg,
        ).to(device=device, dtype=dtype)
        # Unfreeze only the wrapper's learnable params.
        for name, p in wrapper.named_parameters():
            # Keep original_mlp frozen; train everything else.
            if not name.startswith("base.original_mlp."):
                p.requires_grad = True
            else:
                p.requires_grad = False
        layers[idx].mlp = wrapper
        wrapped_handles.append(wrapper)

    # Dtype audit — catch any float32 leakage in the base model or
    # wrapped layers. If any parameter isn't our target dtype, list it.
    mismatches: list[tuple[str, str]] = []
    for name, p in model.named_parameters():
        if p.is_floating_point() and p.dtype != dtype:
            mismatches.append((name, str(p.dtype)))
    if mismatches:
        print(f"[npcot-train] DTYPE MISMATCH — {len(mismatches)} parameters not {dtype}:", flush=True)
        for name, d in mismatches[:20]:
            print(f"   {name}: {d}", flush=True)
        # Force-cast everything. If this is still required we have a real
        # bug but training should at least proceed.
        model.to(dtype=dtype)
        # Also convert buffers.
        for module in model.modules():
            for bname, buf in list(module._buffers.items()):
                if buf is not None and buf.is_floating_point() and buf.dtype != dtype:
                    module._buffers[bname] = buf.to(dtype=dtype)
        print("[npcot-train] force-cast complete", flush=True)

    return model, tokenizer, device, wrapped_handles


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def run_training(cfg: QwenNPCoTTrainingConfig) -> dict:
    import torch
    from torch import nn

    from ncpu.self_optimizing.array_program_library import (
        ArrayProgramLibrary,
    )
    from ncpu.self_optimizing.program_library_session import (
        ProgramLibrarySession,
        ProgramLibrarySessionConfig,
    )

    print(f"[npcot-train] loading {cfg.model}", flush=True)
    model, tokenizer, device, wrappers = build_wrapped_model(cfg)
    total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"[npcot-train] loaded on {device}; wrapped {len(wrappers)} layers; "
        f"trainable {total_train_params / 1e6:.2f}M / {total_params / 1e9:.2f}B = "
        f"{100 * total_train_params / total_params:.3f}% of total",
        flush=True,
    )

    print(f"[npcot-train] loading dataset {cfg.dataset}/{cfg.dataset_split}", flush=True)
    texts = load_dataset_texts(cfg.dataset, cfg.dataset_split)
    print(f"[npcot-train] {len(texts)} texts loaded", flush=True)

    cfg.out_library.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    session = ProgramLibrarySession(
        ProgramLibrarySessionConfig(
            library_path=cfg.out_library,
            convergence_gap_threshold=cfg.convergence_gap_threshold,
            auto_cache=True,
        )
    )
    session.begin_task(f"qwen-npcot-{cfg.model}")

    # Attach the library to every wrapped layer so its array-thought head
    # consults during training.
    for wrapper in wrappers:
        wrapper.attach_library(session.library, task_name=cfg.dataset)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)

    def tokenize(batch_texts: list[str]) -> dict:
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_seq_len,
        )
        return {k: v.to(device) for k, v in enc.items()}

    import random
    rng = random.Random(0)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    model.train()
    # Belt-and-braces: ensure every param+buffer matches the target dtype
    # before training begins. The Qwen3.5 VL extraction path can leave
    # some tensors in unexpected precision despite our earlier casts.
    model = model.to(dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    for m in model.modules():
        for bname, buf in list(m._buffers.items()):
            if buf is not None and buf.is_floating_point() and buf.dtype != torch.bfloat16:
                m._buffers[bname] = buf.to(dtype=torch.bfloat16)

    # Final dtype census — print the first 5 params and their dtypes.
    sample_params = list(model.named_parameters())[:5]
    print("[npcot-train] dtype census (first 5 params):", flush=True)
    for name, p in sample_params:
        print(f"   {name}: {p.dtype}", flush=True)
    # And check target layers explicitly.
    for idx in cfg.target_layers:
        resolved = idx if idx >= 0 else len(model.model.layers) + idx
        mlp = model.model.layers[resolved].mlp
        # Walk to the underlying Qwen MLP's gate_proj inside our wrapper.
        try:
            w = mlp.base.original_mlp.gate_proj.weight
            print(f"   layer[{idx}].mlp.base.original_mlp.gate_proj.weight: {w.dtype}", flush=True)
        except AttributeError:
            pass

    t_start = time.perf_counter()
    step_losses: list[float] = []
    library_sizes: list[int] = []

    for step in range(cfg.max_steps):
        # Simple fixed-LR warmup.
        if step < cfg.warmup_steps:
            lr = cfg.learning_rate * (step + 1) / max(cfg.warmup_steps, 1)
            for g in optimizer.param_groups:
                g["lr"] = lr

        batch_texts = rng.sample(texts, min(cfg.batch_size, len(texts)))
        inputs = tokenize(batch_texts)
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = tokenizer.pad_token_id

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        step_losses.append(float(loss.item()))
        library_sizes.append(len(session.library))

        if (step + 1) % cfg.log_every == 0:
            elapsed = time.perf_counter() - t_start
            sps = (step + 1) / elapsed
            print(
                f"[npcot-train] step {step + 1}/{cfg.max_steps} "
                f"loss={loss.item():.4f} lib={len(session.library)} "
                f"step/s={sps:.3f}",
                flush=True,
            )

    summary = session.end_task()
    print(
        f"[npcot-train] training complete. Final loss={step_losses[-1]:.4f}, "
        f"library={summary.entry_count} entries, {summary.newly_cached_count} new, "
        f"total hits={summary.total_hits}",
        flush=True,
    )

    # Save coprocessor state-dict.
    coproc_state = {}
    for idx, wrapper in enumerate(wrappers):
        for name, p in wrapper.named_parameters():
            if p.requires_grad:
                coproc_state[f"wrapper_{idx}.{name}"] = p.detach().cpu()
    torch.save({
        "model": cfg.model,
        "target_layers": cfg.target_layers,
        "coprocessor_state_dict": coproc_state,
    }, cfg.out_checkpoint)
    print(f"[npcot-train] checkpoint saved to {cfg.out_checkpoint}", flush=True)
    print(f"[npcot-train] library saved to {cfg.out_library}", flush=True)

    return {
        "mode": "qwen_npcot_training",
        "model": cfg.model,
        "target_layers": cfg.target_layers,
        "steps": len(step_losses),
        "final_loss": step_losses[-1],
        "library_entries": summary.entry_count,
        "newly_cached_count": summary.newly_cached_count,
        "trainable_params": int(total_train_params),
        "total_params": int(total_params),
        "checkpoint": str(cfg.out_checkpoint),
        "library": str(cfg.out_library),
        "wall_seconds": time.perf_counter() - t_start,
        "loss_curve": step_losses,
        "library_size_curve": library_sizes,
    }


def main(argv: list[str] | None = None) -> int:
    cfg = parse_cli(argv)
    try:
        report = run_training(cfg)
    except ImportError as exc:
        print(f"error: missing dependency ({exc}).", file=sys.stderr)
        return 2
    out_json = cfg.out_checkpoint.with_suffix(".json")
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[npcot-train] summary: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
