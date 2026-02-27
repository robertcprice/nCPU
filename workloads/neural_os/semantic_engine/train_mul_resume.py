#!/usr/bin/env python3
"""
Resume MUL training from 32-bit checkpoint with extended epochs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import os
import sys
import argparse

# Enable line buffering for real-time output
try:
    sys.stdout.reconfigure(line_buffering=True)
except:
    pass  # Python < 3.7 compatibility

# Import the model class from the main script
from train_mul_add_hybrid import PureParallelMulWithADDv2, generate_mul_data


def resume_train(model, device, save_dir, start_bits=32, max_bits=64, batch_size=4096, model_name="add_hybrid_v3.5"):
    print("\n" + "=" * 70)
    print(f"RESUMING MUL TRAINING FROM {start_bits}-bit")
    print("=" * 70)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}")

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')

    # Extended curriculum with MORE epochs for convergence
    # DOUBLED epochs for 48+ bit since they need more time to hit 100% x3
    full_curriculum = [
        (8, 3000), (16, 3000), (24, 6000), (32, 8000),
        (40, 10000), (48, 20000), (56, 24000), (64, 30000)
    ]

    # Filter to only train from start_bits onwards
    curriculum = [(b, e) for (b, e) in full_curriculum if b >= start_bits and b <= max_bits]

    print(f"Curriculum: {curriculum}")

    for bits, max_epochs in curriculum:
        print(f"\n{'=' * 70}")
        print(f"Training: {bits}-bit MUL (RESUMED)")
        print(f"{'=' * 70}")

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
        best_acc = 0
        consecutive_100 = 0

        for epoch in range(max_epochs):
            model.train()

            a_bits, b_bits, target = generate_mul_data(batch_size, bits, device)

            if bits < max_bits:
                a_bits = F.pad(a_bits, (0, max_bits - bits))
                b_bits = F.pad(b_bits, (0, max_bits - bits))
                target = F.pad(target, (0, max_bits - bits))

            optimizer.zero_grad()

            with autocast('cuda', dtype=torch.bfloat16):
                output = model(a_bits, b_bits)
                loss = criterion(output[:, :bits], target[:, :bits])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                pred = (torch.sigmoid(output[:, :bits]) > 0.5).float()
                correct = (pred == target[:, :bits]).all(dim=1).float().mean().item() * 100

            if epoch % 100 == 0 or correct > best_acc:
                print(f"    Epoch {epoch:4d}: loss={loss.item():.4f}, acc={correct:.2f}%")

            if correct > best_acc:
                best_acc = correct
                torch.save(model.state_dict(), f"{save_dir}/MUL_{bits}bit_{model_name}_ckpt.pt")

            if correct >= 100.0:
                consecutive_100 += 1
                if consecutive_100 >= 3:
                    print(f"    ✅ {bits}-bit: 100% x3 - ADVANCING!")
                    torch.save(model.state_dict(), f"{save_dir}/MUL_{bits}bit_{model_name}_100pct.pt")
                    break
            else:
                consecutive_100 = 0

        print(f"    Level complete: best={best_acc:.2f}%")

    print("\n" + "=" * 70)
    print("RESUMED MUL TRAINING COMPLETE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--bits', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--start-bits', type=int, default=32, help='Resume from this bit width')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint to resume from')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # V3.5 architecture (12.7M params)
    model = PureParallelMulWithADDv2(max_bits=args.bits, d_model=384, nhead=16, num_layers=7)
    model = model.to(args.device)
    model_name = "add_hybrid_v3.5"

    # Find and load checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint_path = args.checkpoint
    else:
        # Auto-find best checkpoint for start_bits
        checkpoint_path = f"{args.save_dir}/MUL_{args.start_bits}bit_{model_name}_ckpt.pt"

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
        print("Checkpoint loaded successfully!")
    else:
        print(f"WARNING: No checkpoint found at {checkpoint_path}")
        print("Starting fresh training...")

    resume_train(model, args.device, args.save_dir,
                 start_bits=args.start_bits, max_bits=args.bits,
                 batch_size=args.batch_size, model_name=model_name)


if __name__ == '__main__':
    main()
