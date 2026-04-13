"""Supervised benchmark training for Mog diffusion models.

Primary use case: overfit experiments on a small number of benchmark problems.
If the model cannot overfit 10 reference solutions, scaling synthetic training
or reward tuning will not rescue it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from egdc.mog.model import MogMaskedDiffusion, MogDiffusionConfig
from egdc.mog.train import train, get_device
from egdc.mog.training.supervised_dataset import MogBenchmarkSupervisedDataset
from egdc.mog.benchmark import get_benchmark


def main() -> None:
    ap = argparse.ArgumentParser(description="Supervised benchmark training for Mog")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--model_size", choices=["tiny", "small", "medium"], default="tiny")
    ap.add_argument("--num_problems", type=int, default=10)
    ap.add_argument("--variants_per_factory", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=128)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--spec_len", type=int, default=128)
    ap.add_argument("--checkpoint_dir", type=str, default="checkpoints/mog_supervised")
    ap.add_argument("--checkpoint_every", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--eval_samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    args = ap.parse_args()

    if args.model_size == "tiny":
        config = MogDiffusionConfig.tiny()
    elif args.model_size == "small":
        config = MogDiffusionConfig.small()
    else:
        config = MogDiffusionConfig.medium()
    config.max_seq_len = max(config.max_seq_len, args.seq_len + args.spec_len + 64)

    problems = get_benchmark(seed=args.seed, variants_per_factory=args.variants_per_factory)[: args.num_problems]
    dataset = MogBenchmarkSupervisedDataset(
        problems=problems,
        seq_len=args.seq_len,
        spec_len=args.spec_len,
        repeat=args.repeat,
        seed=args.seed,
    )

    print(f"Model: {args.model_size}")
    print(f"Problems: {len(dataset.unique_examples())}")
    print(f"Training examples (with repeat): {len(dataset)}")
    print(f"Seq len: {args.seq_len}, spec len: {args.spec_len}")

    model = MogMaskedDiffusion(config)
    device = get_device(args.device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    train(
        model=model,
        train_loader=loader,
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
