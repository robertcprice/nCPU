"""Training script for Mog code masked diffusion transformer.

Trains on synthetically generated Mog programs (35 template families).
MDLM objective: cross-entropy loss on masked positions only.
Periodic evaluation using mog_eval static analysis on generated samples.

Usage:
    python -m egdc.mog_train --epochs 100 --model_size tiny
    python -m egdc.mog_train --epochs 50 --batch_size 16 --num_samples 50000 --eval_every 5
"""

import argparse
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from egdc.mog_model import MogMaskedDiffusion, MogDiffusionConfig
from egdc.mog_tokenizer import MogCodeTokenizer, MASK_TOKEN, PAD_TOKEN
from egdc.mog_dataset import MogDataset
from egdc.mog_eval import evaluate_mog_program, evaluate_batch


def get_device() -> torch.device:
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_step(
    model: MogMaskedDiffusion,
    masked_tokens: torch.Tensor,
    mask_positions: torch.Tensor,
    original_tokens: torch.Tensor,
    spec_tokens: torch.Tensor,
    timesteps: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Single training step with MDLM loss."""
    masked_tokens = masked_tokens.to(device)
    mask_positions = mask_positions.bool().to(device)
    original_tokens = original_tokens.to(device)
    spec_tokens = spec_tokens.to(device)
    timesteps = timesteps.to(device)

    # Forward pass
    logits = model(masked_tokens, timesteps, spec_tokens=spec_tokens)

    # Loss: cross-entropy only on masked positions
    logits_flat = logits.view(-1, logits.shape[-1])
    targets_flat = original_tokens.view(-1)
    mask_flat = mask_positions.view(-1)

    if mask_flat.any():
        loss = F.cross_entropy(
            logits_flat[mask_flat],
            targets_flat[mask_flat],
        )
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    # Compute accuracy on masked positions
    with torch.no_grad():
        if mask_flat.any():
            preds = logits_flat[mask_flat].argmax(dim=-1)
            acc = (preds == targets_flat[mask_flat]).float().mean().item()
        else:
            acc = 0.0

    metrics = {
        "loss": loss.item(),
        "accuracy": acc,
        "num_masked": mask_flat.sum().item(),
        "avg_t": timesteps.mean().item(),
    }
    return loss, metrics


@torch.no_grad()
def generate_mog(
    model: MogMaskedDiffusion,
    spec_tokens: Optional[torch.Tensor],
    seq_len: int,
    num_steps: int = 64,
    temperature: float = 0.8,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate Mog code via iterative unmasking.

    No ISA slot constraints — Mog is free-form byte sequences.

    Args:
        model: trained Mog diffusion model
        spec_tokens: (1, S) conditioning spec tokens, or None
        seq_len: length of the sequence to generate
        num_steps: number of denoising steps
        temperature: sampling temperature
        device: device to run on

    Returns:
        (1, seq_len) generated token IDs
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Start fully masked
    tokens = torch.full((1, seq_len), MASK_TOKEN, dtype=torch.long, device=device)

    if spec_tokens is not None:
        spec_tokens = spec_tokens.to(device)

    for step in range(num_steps):
        # Compute effective timestep: goes from ~1 (fully masked) to ~0 (clean)
        t = 1.0 - (step + 1) / num_steps
        t_tensor = torch.tensor([max(t, 0.01)], device=device)

        # Get model predictions
        logits = model(tokens, t_tensor, spec_tokens=spec_tokens)

        # Apply temperature and compute probabilities
        probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)

        # Sample from distribution
        flat_probs = probs.view(-1, probs.shape[-1])
        flat_probs = flat_probs.clamp(min=1e-10)
        flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(flat_probs, num_samples=1).view(1, seq_len)

        # Compute confidence: max probability at each position
        confidence = probs.max(dim=-1).values

        # Find currently masked positions
        is_masked = (tokens == MASK_TOKEN)
        num_masked = is_masked.sum().item()

        if num_masked == 0:
            break

        # How many to unmask this step
        num_to_reveal = max(1, min(
            int(math.ceil(num_masked / max(num_steps - step, 1))),
            num_masked,
        ))

        # Among masked positions, pick the most confident to unmask
        masked_confidence = confidence.clone()
        masked_confidence[~is_masked] = -1.0

        _, top_indices = masked_confidence.topk(num_to_reveal, dim=-1)

        # Reveal those positions
        for idx in top_indices[0]:
            tokens[0, idx] = sampled[0, idx]

    # Final cleanup: replace any remaining masks
    if (tokens == MASK_TOKEN).any():
        t_tensor = torch.tensor([0.01], device=device)
        logits = model(tokens, t_tensor, spec_tokens=spec_tokens)
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            flat_probs = probs.view(-1, probs.shape[-1]).clamp(min=1e-10)
            flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True)
            final = torch.multinomial(flat_probs, num_samples=1).view(1, seq_len)
        else:
            final = logits.argmax(dim=-1)
        mask = (tokens == MASK_TOKEN)
        tokens = torch.where(mask, final, tokens)

    return tokens


def evaluate_generated_samples(
    model: MogMaskedDiffusion,
    tokenizer: MogCodeTokenizer,
    num_samples: int = 16,
    seq_len: int = 512,
    num_steps: int = 64,
    temperature: float = 0.8,
    device: Optional[torch.device] = None,
) -> dict:
    """Generate samples and evaluate with mog_eval static analysis.

    Returns:
        Dict with aggregate metrics and one sample program.
    """
    model.eval()
    programs = []

    for _ in range(num_samples):
        token_ids = generate_mog(
            model, spec_tokens=None, seq_len=seq_len,
            num_steps=num_steps, temperature=temperature, device=device,
        )
        code = tokenizer.decode(token_ids[0].tolist())
        programs.append(code)

    metrics = evaluate_batch(programs)
    metrics["sample_program"] = programs[0] if programs else ""
    return metrics


def train(
    model: MogMaskedDiffusion,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    warmup_steps: int = 100,
    checkpoint_dir: str = "checkpoints/mog",
    checkpoint_every: int = 200,
    max_grad_norm: float = 1.0,
    log_every: int = 10,
    eval_every: Optional[int] = None,
    eval_samples: int = 16,
    device: Optional[torch.device] = None,
) -> None:
    """Main training loop."""
    if device is None:
        device = get_device()

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_loss = float("inf")
    tokenizer = MogCodeTokenizer()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_steps = 0
        t0 = time.time()

        for batch in train_loader:
            masked_tokens, mask_positions, original_tokens, spec_tokens, timesteps = batch

            optimizer.zero_grad()

            loss, metrics = train_step(
                model, masked_tokens, mask_positions, original_tokens,
                spec_tokens, timesteps, device
            )
            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += metrics["loss"]
            epoch_acc += metrics["accuracy"]
            epoch_steps += 1

            if global_step % log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                steps_per_sec = epoch_steps / elapsed
                avg_loss = epoch_loss / epoch_steps
                avg_acc = epoch_acc / epoch_steps
                print(
                    f"[step {global_step:>6d}] "
                    f"loss={metrics['loss']:.4f} (avg={avg_loss:.4f})  "
                    f"acc={metrics['accuracy']:.3f} (avg={avg_acc:.3f})  "
                    f"gnorm={grad_norm:.3f}  "
                    f"lr={current_lr:.2e}  "
                    f"t={metrics['avg_t']:.2f}  "
                    f"{steps_per_sec:.1f} steps/s"
                )

            if global_step % checkpoint_every == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": model.config,
                    "global_step": global_step,
                    "epoch": epoch,
                }
                ckpt_file = ckpt_path / f"step_{global_step}.pt"
                torch.save(ckpt, ckpt_file)
                print(f"  >> Saved checkpoint: {ckpt_file}")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        avg_acc = epoch_acc / max(epoch_steps, 1)
        elapsed = time.time() - t0
        print(
            f"=== Epoch {epoch+1}/{epochs} done in {elapsed:.1f}s. "
            f"Avg loss: {avg_loss:.4f}, Avg acc: {avg_acc:.3f} ==="
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_path / "best.pt")
            print(f"  >> New best model (loss={best_loss:.4f})")

        # Periodic evaluation: generate samples and run mog_eval
        if eval_every and (epoch + 1) % eval_every == 0:
            print("\n--- Mog eval (static analysis on generated samples) ---")
            eval_metrics = evaluate_generated_samples(
                model, tokenizer, num_samples=eval_samples,
                seq_len=512, num_steps=32, temperature=0.8, device=device,
            )
            print(
                f"  validity={eval_metrics.get('validity_rate', 0):.1f}%  "
                f"overall={eval_metrics.get('overall_score', 0):.3f}  "
                f"syntax={eval_metrics.get('syntactic_validity', 0):.3f}  "
                f"types={eval_metrics.get('type_completeness', 0):.3f}  "
                f"structure={eval_metrics.get('structural_correctness', 0):.3f}"
            )
            sample = eval_metrics.get("sample_program", "")
            if sample:
                print(f"  Sample:\n{sample[:300]}")
            print("---\n")
            model.train()

    torch.save(model.state_dict(), ckpt_path / "final.pt")
    print(f"Training complete. Final model: {ckpt_path / 'final.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Mog code masked diffusion model"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--model_size", choices=["tiny", "small", "medium"],
                        default="tiny")
    parser.add_argument("--num_samples", type=int, default=50_000,
                        help="Number of synthetic Mog programs to generate")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Max sequence length for Mog programs")
    parser.add_argument("--spec_len", type=int, default=64,
                        help="Max sequence length for specs (fn signatures)")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="checkpoints/mog_egdc")
    parser.add_argument("--checkpoint_every", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_every", type=int, default=10,
                        help="Evaluate generated samples every N epochs")
    parser.add_argument("--eval_samples", type=int, default=16,
                        help="Number of samples to generate for eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Model config
    if args.model_size == "tiny":
        config = MogDiffusionConfig.tiny()
    elif args.model_size == "small":
        config = MogDiffusionConfig.small()
    else:
        config = MogDiffusionConfig.medium()

    # Ensure max_seq_len covers spec + code
    config.max_seq_len = max(args.seq_len + args.spec_len + 64, config.max_seq_len)

    print(f"Model: {args.model_size}")
    print(f"Config: {config}")
    print(f"Synthetic dataset: {args.num_samples} programs")

    model = MogMaskedDiffusion(config)
    device = get_device()

    # Resume from checkpoint
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)

    # Dataset: procedurally generated Mog programs
    dataset = MogDataset(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        spec_len=args.spec_len,
        seed=args.seed,
        balanced=True,
    )
    print(f"Dataset: {len(dataset)} samples")

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Batches/epoch: {len(train_loader)}")

    train(
        model=model,
        train_loader=train_loader,
        epochs=args.epochs,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        max_grad_norm=args.max_grad_norm,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_samples=args.eval_samples,
        device=device,
    )


if __name__ == "__main__":
    main()
