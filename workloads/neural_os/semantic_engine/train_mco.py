#!/usr/bin/env python3
"""
TRAINING SCRIPT: Train the Meta-Cognitive Orchestrator on GPU

This script trains the neural components of the Semantic Synthesizer:
1. Program Encoder - learns embeddings of programs
2. Synthesis Policy - learns RL policy for synthesis actions
3. Tactic Memory - accumulates successful patterns

Run on GPU server:
  python3 train_mco.py --iterations 10000 --device cuda
"""

import argparse
import time
import torch
from datetime import datetime

from meta_cognitive_orchestrator import MetaCognitiveOrchestrator
from epistemic_frontier import EpistemicFrontier
from trace_analyzer import trace_from_io
from rewrite_engine import var, const, add, mul, square, double


def generate_training_functions():
    """Generate diverse training functions."""
    functions = [
        # Basic
        (lambda x: x, "identity"),
        (lambda x: x + 1, "increment"),
        (lambda x: x - 1, "decrement"),
        (lambda x: x * 2, "double"),
        (lambda x: x * 3, "triple"),
        (lambda x: x * x, "square"),

        # Linear
        (lambda x: 2 * x + 1, "2x+1"),
        (lambda x: 3 * x + 2, "3x+2"),
        (lambda x: 5 * x, "5x"),
        (lambda x: x + 10, "x+10"),

        # Polynomial
        (lambda x: x * x + 1, "x^2+1"),
        (lambda x: x * x + x, "x^2+x"),
        (lambda x: x * x * x, "cube"),
        (lambda x: 2 * x * x, "2x^2"),

        # Bitwise (as arithmetic)
        (lambda x: x * 4, "x*4"),
        (lambda x: x * 8, "x*8"),
        (lambda x: x * 16, "x*16"),
        (lambda x: x // 2, "x/2"),

        # Complex
        (lambda x: (x + 1) * 2, "(x+1)*2"),
        (lambda x: x * (x + 1), "x*(x+1)"),
        (lambda x: (x * (x + 1)) // 2, "triangular"),
    ]
    return functions


def train(iterations: int = 1000, device: str = 'cpu',
          checkpoint_interval: int = 100, verbose: bool = True):
    """
    Train the MCO neural networks.

    Args:
        iterations: Number of training iterations
        device: 'cpu' or 'cuda'
        checkpoint_interval: Save checkpoint every N iterations
        verbose: Print progress
    """
    print("=" * 70)
    print("META-COGNITIVE ORCHESTRATOR TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Iterations: {iterations}")
    print(f"PyTorch version: {torch.__version__}")

    if device == 'cuda' and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif device == 'cuda':
        print("⚠️ CUDA requested but not available, falling back to CPU")
        device = 'cpu'

    # Initialize MCO
    mco = MetaCognitiveOrchestrator(device=device)

    # Get training functions
    functions = generate_training_functions()
    print(f"\nTraining on {len(functions)} function types")

    # Training loop
    start_time = time.time()
    successes = 0
    total = 0

    for epoch in range(iterations):
        # Shuffle functions each epoch
        import random
        random.shuffle(functions)

        epoch_successes = 0
        epoch_total = 0

        for func, name in functions:
            # Generate trace
            inputs = list(range(0, 10))
            try:
                trace = trace_from_io([(x, func(x)) for x in inputs])
            except:
                continue

            # Synthesize with policy
            result = mco.synthesize_with_policy(trace, verbose=False)

            total += 1
            epoch_total += 1

            if result is not None:
                successes += 1
                epoch_successes += 1

        # Train policy
        if len(mco.experience_buffer) >= 32:
            mco.train_step(batch_size=32)

        # Progress report
        if verbose and (epoch + 1) % 10 == 0:
            success_rate = successes / total * 100 if total > 0 else 0
            epoch_rate = epoch_successes / epoch_total * 100 if epoch_total > 0 else 0
            elapsed = time.time() - start_time

            print(f"Epoch {epoch+1:4d}/{iterations}: "
                  f"success={success_rate:.1f}% (epoch: {epoch_rate:.1f}%) "
                  f"time={elapsed:.1f}s "
                  f"tactics={len(mco.tactic_memory.tactics)}")

        # Checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = f"mco_checkpoint_epoch{epoch+1}.pt"
            mco.save(checkpoint_path)
            if verbose:
                print(f"  → Saved checkpoint: {checkpoint_path}")

    # Final save
    mco.save("mco_final.pt")
    mco.tactic_memory.save()

    # Report
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Final success rate: {successes/total*100:.1f}%")
    print(f"Tactics learned: {len(mco.tactic_memory.tactics)}")
    print(f"Training steps: {mco.training_step}")
    print(f"\nSaved: mco_final.pt, tactic_memory.json")

    return mco


def train_with_epistemic(iterations: int = 500, device: str = 'cpu'):
    """
    Train with epistemic frontier exploration.

    This adds bisociation-based discovery to find novel patterns.
    """
    print("\n" + "=" * 70)
    print("EPISTEMIC FRONTIER TRAINING")
    print("=" * 70)

    ef = EpistemicFrontier()

    # Explore frontier
    discoveries = ef.explore_frontier(iterations=iterations, verbose=True)

    print(f"\nDiscoveries: {len(discoveries)}")

    # Reflect
    reflection = ef.self_reflect()
    print(f"\nReflection:")
    print(f"  Avg novelty: {reflection['avg_novelty']:.2f}")
    print(f"  Domains: {reflection['domains_explored']}")
    print(f"  Next goals: {reflection['next_goals'][:3]}")

    return ef


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MCO neural networks")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to train on")
    parser.add_argument("--checkpoint", type=int, default=50,
                        help="Checkpoint interval")
    parser.add_argument("--epistemic", action="store_true",
                        help="Also train epistemic frontier")

    args = parser.parse_args()

    # Train MCO
    mco = train(
        iterations=args.iterations,
        device=args.device,
        checkpoint_interval=args.checkpoint,
        verbose=True
    )

    # Train epistemic if requested
    if args.epistemic:
        ef = train_with_epistemic(
            iterations=args.iterations // 2,
            device=args.device
        )

    print("\n✅ Training complete!")
