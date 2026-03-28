"""Training script for the EGDC masked diffusion transformer.

MDLM objective: cross-entropy loss on masked positions only.
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

from .model import MaskedDiffusionTransformer, ModelConfig, MASK_TOKEN, PAD_TOKEN


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
    model: MaskedDiffusionTransformer,
    masked_tokens: torch.Tensor,
    mask_positions: torch.Tensor,
    original_tokens: torch.Tensor,
    spec_tokens: torch.Tensor,
    timesteps: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Single training step.

    Args:
        masked_tokens: (B, L) input with MASK tokens
        mask_positions: (B, L) boolean mask of masked positions
        original_tokens: (B, L) ground truth tokens
        spec_tokens: (B, S) specification tokens
        timesteps: (B,) diffusion timesteps
        device: torch device

    Returns:
        (loss, metrics_dict)
    """
    masked_tokens = masked_tokens.to(device)
    mask_positions = mask_positions.bool().to(device)
    original_tokens = original_tokens.to(device)
    spec_tokens = spec_tokens.to(device)
    timesteps = timesteps.to(device)

    # Forward pass
    logits = model(masked_tokens, timesteps, spec_tokens=spec_tokens)  # (B, L, V)

    # Loss: cross-entropy only on masked positions
    logits_flat = logits.view(-1, logits.shape[-1])  # (B*L, V)
    targets_flat = original_tokens.view(-1)  # (B*L,)
    mask_flat = mask_positions.view(-1)  # (B*L,)

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


def train(
    model: MaskedDiffusionTransformer,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    warmup_steps: int = 500,
    checkpoint_dir: str = "checkpoints",
    checkpoint_every: int = 500,
    max_grad_norm: float = 1.0,
    log_every: int = 25,
    device: Optional[torch.device] = None,
) -> None:
    """Main training loop."""
    if device is None:
        device = get_device()

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Checkpoint directory
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_steps = 0
        t0 = time.time()

        for batch in train_loader:
            # NCPUDataset returns: (masked, mask_pos, original, spec, timestep)
            masked_tokens, mask_positions, original_tokens, spec_tokens, timesteps = batch

            optimizer.zero_grad()

            loss, metrics = train_step(
                model, masked_tokens, mask_positions, original_tokens,
                spec_tokens, timesteps, device
            )
            loss.backward()

            # Gradient clipping
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += metrics["loss"]
            epoch_acc += metrics["accuracy"]
            epoch_steps += 1

            # Logging
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

            # Checkpointing
            if global_step % checkpoint_every == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
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

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_path / "best.pt")
            print(f"  >> New best model (loss={best_loss:.4f})")

    # Final save
    torch.save(model.state_dict(), ckpt_path / "final.pt")
    print(f"Training complete. Final model: {ckpt_path / 'final.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EGDC masked diffusion model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--model_size", choices=["small", "medium"], default="small")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/egdc")
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=50000,
                        help="Number of training programs to generate")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    # Model config
    if args.model_size == "small":
        config = ModelConfig.small()
    else:
        config = ModelConfig.medium()

    print(f"Model: {args.model_size}")
    print(f"Config: {config}")
    print(f"Generating {args.num_samples} training programs...")

    model = MaskedDiffusionTransformer(config)
    device = get_device()

    # Dataset
    from .dataset import NCPUDataset
    dataset = NCPUDataset(num_samples=args.num_samples)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Dataset: {len(dataset)} programs, {len(train_loader)} batches/epoch")

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
        device=device,
    )


if __name__ == "__main__":
    main()
