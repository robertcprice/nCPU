#!/usr/bin/env python3
"""EGDC training with resume support."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import os, time, math, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from egdc.core.dataset import NCPUDataset
from egdc.core.model import MaskedDiffusionTransformer, ModelConfig, MASK_TOKEN, PAD_TOKEN

NUM_SAMPLES = 5000
BATCH_SIZE = 32
EPOCHS = 50
LR = 5e-4
WARMUP = 200
CKPT_DIR = "checkpoints/egdc"

print("=" * 60)
print("EGDC Training")
print("=" * 60, flush=True)

ds = NCPUDataset(num_samples=NUM_SAMPLES)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
print(f"Dataset: {len(ds)}, {len(loader)} batches/epoch", flush=True)

cfg = ModelConfig.tiny()
model = MaskedDiffusionTransformer(cfg)
device = torch.device("cpu")
model = model.to(device)
params = sum(p.numel() for p in model.parameters())
print(f"Model: {params:,} params on {device}", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=0.01)
total_steps = len(loader) * EPOCHS

def lr_lambda(step):
    if step < WARMUP:
        return step / max(1, WARMUP)
    progress = (step - WARMUP) / max(1, total_steps - WARMUP)
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Resume from latest checkpoint
ckpt_path = Path(CKPT_DIR)
ckpt_path.mkdir(parents=True, exist_ok=True)
start_epoch = 0
global_step = 0
best_loss = float("inf")

ckpt_files = sorted(ckpt_path.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
if ckpt_files:
    latest = ckpt_files[-1]
    print(f"Resuming from {latest}...", flush=True)
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    global_step = ckpt["global_step"]
    start_epoch = ckpt["epoch"] + 1
    print(f"  Resumed at step {global_step}, epoch {start_epoch}", flush=True)

for epoch in range(start_epoch, EPOCHS):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    epoch_steps = 0
    t0 = time.time()

    for batch in loader:
        masked, mask_pos, original, spec, timesteps = batch
        masked, original, spec = masked.to(device), original.to(device), spec.to(device)
        mask_pos = mask_pos.bool().to(device)
        timesteps = timesteps.to(device)

        optimizer.zero_grad()
        logits = model(masked, timesteps, spec_tokens=spec)
        logits_flat = logits.view(-1, logits.shape[-1])
        targets_flat = original.view(-1)
        mask_flat = mask_pos.view(-1)

        if mask_flat.any():
            loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat])
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        global_step += 1
        epoch_steps += 1
        epoch_loss += loss.item()

        with torch.no_grad():
            if mask_flat.any():
                acc = (logits_flat[mask_flat].argmax(-1) == targets_flat[mask_flat]).float().mean().item()
            else:
                acc = 0.0
        epoch_acc += acc

        if global_step % 50 == 0:
            avg_l = epoch_loss / epoch_steps
            avg_a = epoch_acc / epoch_steps
            lr = scheduler.get_last_lr()[0]
            sps = epoch_steps / (time.time() - t0)
            print(f"[step {global_step:>5d}] loss={loss.item():.4f} avg={avg_l:.4f} "
                  f"acc={acc:.3f} avg={avg_a:.3f} gnorm={grad_norm:.2f} "
                  f"lr={lr:.2e} {sps:.1f}it/s", flush=True)

        if global_step % 500 == 0:
            ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "global_step": global_step, "epoch": epoch}
            torch.save(ckpt, ckpt_path / f"step_{global_step}.pt")
            print(f"  >> Checkpoint saved: step_{global_step}.pt", flush=True)

    avg_loss = epoch_loss / max(epoch_steps, 1)
    avg_acc = epoch_acc / max(epoch_steps, 1)
    elapsed = time.time() - t0
    print(f"=== Epoch {epoch+1}/{EPOCHS} in {elapsed:.0f}s | loss={avg_loss:.4f} acc={avg_acc:.3f} ===", flush=True)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), ckpt_path / "best.pt")
        print(f"  >> New best (loss={best_loss:.4f})", flush=True)

torch.save(model.state_dict(), ckpt_path / "final.pt")
print(f"\nDone. Best loss: {best_loss:.4f}")
