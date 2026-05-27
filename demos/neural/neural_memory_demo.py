#!/usr/bin/env python3
"""Neural Memory Architectures Demo.

Trains and evaluates three novel neural memory architectures:
  1. Neural ECC Memory — fault-tolerant storage with learned error-correcting codes
  2. Neural CAM — content-addressable memory with neural value embeddings
  3. Differentiable Storage — registers and RAM with gradient flow for program synthesis

Each architecture solves a real problem that conventional memory cannot:
  - ECC: graceful degradation under corruption (vs hard failure boundary)
  - CAM: reverse lookup by value, not just address (vs address-only access)
  - Diff Storage: gradient-based register/address learning (vs discrete, non-differentiable indexing)

Run:
    python demos/neural/neural_memory_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on path
_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))

import torch


def section(title: str) -> None:
    """Print a section header."""
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()


def subsection(title: str) -> None:
    """Print a subsection header."""
    print(f"\n--- {title} ---\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Neural ECC Memory
# ─────────────────────────────────────────────────────────────────────────────


def demo_ecc_memory() -> dict:
    """Train and evaluate Neural ECC Memory."""
    section("1. Neural ECC Memory")
    print("Training a fault-tolerant memory with learned error-correcting codes.")
    print("16-bit values encoded into 128-dim embeddings (8x redundancy).")
    print("The decoder recovers original values even when embeddings are corrupted.\n")

    from ncpu.neural.neural_ecc_memory import (
        NeuralECCMemory,
        train_ecc_memory,
        evaluate_corruption_sweep,
    )

    t0 = time.time()
    mem = train_ecc_memory(
        n_bits=16,
        embed_dim=128,
        epochs=2000,
        batch_size=2048,
        lr=2e-3,
        device="cpu",
        verbose=True,
    )
    train_time = time.time() - t0
    print(f"\n  Training time: {train_time:.1f}s")

    subsection("Corruption Sweep (1000 values)")
    results = evaluate_corruption_sweep(
        mem,
        n_values=1000,
        corruption_levels=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
        verbose=True,
    )

    subsection("Memory Stats")
    stats = mem.stats
    print(f"  Parameters:    {stats['param_count']:,}")
    print(f"  Bit width:     {stats['n_bits']}-bit")
    print(f"  Embed dim:     {stats['embed_dim']}")
    print(f"  Redundancy:    {stats['redundancy']}")
    print(f"  Memory size:   {stats['size']} cells")
    print(f"  Storage per cell: {stats['embed_dim'] * 4} bytes "
          f"(vs {stats['n_bits'] // 8} bytes raw)")

    return {
        "corruption_sweep": results,
        "stats": stats,
        "train_time": train_time,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Neural CAM (Content-Addressable Memory)
# ─────────────────────────────────────────────────────────────────────────────


def demo_cam() -> dict:
    """Train and evaluate Neural CAM."""
    section("2. Neural Content-Addressable Memory")
    print("Training a memory that can be queried by VALUE, not just address.")
    print("The value encoder learns to map numerically close values to similar")
    print("embeddings via contrastive learning.\n")

    from ncpu.neural.neural_cam import NeuralCAM, train_cam, evaluate_cam

    t0 = time.time()
    cam = train_cam(
        embed_dim=64,
        epochs=1500,
        batch_size=512,
        lr=1e-3,
        device="cpu",
        verbose=True,
    )
    train_time = time.time() - t0
    print(f"\n  Training time: {train_time:.1f}s")

    subsection("Evaluation")
    metrics = evaluate_cam(cam, verbose=True)

    subsection("CAM Stats")
    stats = cam.stats
    print(f"  Parameters:    {stats['param_count']:,}")
    print(f"  Embed dim:     {stats['embed_dim']}")
    print(f"  Memory size:   {stats['size']} cells")

    return {
        "metrics": metrics,
        "stats": stats,
        "train_time": train_time,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Differentiable Storage
# ─────────────────────────────────────────────────────────────────────────────


def demo_differentiable_storage() -> dict:
    """Train and evaluate Differentiable Storage."""
    section("3. Differentiable Storage (Registers + RAM)")
    print("Demonstrating memory that supports gradient flow for program synthesis.")
    print("The optimizer learns WHICH register/address to read and write.\n")

    from ncpu.neural.differentiable_storage import (
        train_register_copy,
        train_memory_copy,
    )

    subsection("Register Copy: Learn R0 -> R5")
    print("Goal: discover via gradient descent that we should copy R0 to R5.\n")

    t0 = time.time()
    reg_results = train_register_copy(
        n_regs=8,
        word_size=16,
        src_reg=0,
        dst_reg=5,
        n_steps=300,
        lr=0.1,
        verbose=True,
    )
    reg_time = time.time() - t0

    subsection("Memory Copy: Learn Content-Based Retrieval")
    print("Goal: learn a key that retrieves a value stored at address 7.\n")

    t0 = time.time()
    mem_results = train_memory_copy(
        mem_size=16,
        word_size=8,
        n_steps=300,
        lr=0.02,
        verbose=True,
    )
    mem_time = time.time() - t0

    # Additional: demonstrate the full storage system
    subsection("Full Storage System (Registers + RAM)")
    from ncpu.neural.differentiable_storage import DifferentiableStorageSystem

    system = DifferentiableStorageSystem(
        n_regs=8, mem_size=32, word_size=16, controller_hidden=64
    )
    total_params = sum(p.numel() for p in system.parameters())
    print(f"  System parameters: {total_params:,}")
    print(f"  Registers: {system.n_regs} x {system.word_size}-bit")
    print(f"  RAM: {system.mem_size} x {system.word_size}-bit")

    # Run a few controller steps to show it works
    print("\n  Running 3 controller steps:")
    for i in range(3):
        reg_val, ram_val = system.step()
        print(
            f"    Step {i}: reg_read norm={reg_val.norm():.4f}, "
            f"ram_read norm={ram_val.norm():.4f}"
        )

    return {
        "register_copy": reg_results,
        "memory_copy": mem_results,
        "system_params": total_params,
        "reg_time": reg_time,
        "mem_time": mem_time,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────


def print_summary(ecc: dict, cam: dict, diff: dict) -> None:
    """Print a consolidated summary table."""
    section("Summary: Neural Memory Architectures")

    # ECC
    s = ecc["stats"]
    print(f"Neural ECC Memory ({s['n_bits']}-bit, {s['embed_dim']}-dim, "
          f"{s['redundancy']} redundancy):")
    print(f"  {'Corruption':>12s}    {'Recovery Rate':>14s}")
    print(f"  {'=' * 12}    {'=' * 14}")
    for frac, rate in ecc["corruption_sweep"].items():
        print(f"  {frac*100:10.0f}% dims    {rate*100:12.1f}%")
    print(f"  Parameters:    {s['param_count']:,}")
    print(f"  Train time:    {ecc['train_time']:.1f}s")

    # CAM
    print("\nNeural CAM:")
    m = cam["metrics"]
    print(f"  Exact match rate:    {m['exact_match_rate']:.1%}")
    print(f"  Approx match rate:   {m['approx_match_rate']:.1%}")
    print(f"  Ordering accuracy:   {m['ordering_accuracy']:.1%}")
    print(f"  Duplicates found:    {m['duplicates_found']}")
    print(f"  Parameters:          {cam['stats']['param_count']:,}")
    print(f"  Train time:          {cam['train_time']:.1f}s")

    # Differentiable Storage
    print("\nDifferentiable Registers:")
    rc = diff["register_copy"]
    mc = diff["memory_copy"]
    status_reg = "CORRECT" if rc["correct"] else "FAILED"
    status_mem = "CORRECT" if mc["correct"] else "FAILED"
    print(
        f"  Register copy R{rc['expected_src']}->R{rc['expected_dst']}: "
        f"learned R{rc['learned_src']}->R{rc['learned_dst']} "
        f"[{status_reg}] in {rc['steps']} steps"
    )
    print(f"  Final loss: {rc['final_loss']:.6f}")
    print(
        f"  Memory copy addr[{mc['target_addr']}]: "
        f"learned addr[{mc['learned_addr']}] "
        f"[{status_mem}] in {mc['steps']} steps"
    )
    print(f"  Final loss: {mc['final_loss']:.6f}")
    print(f"  Full system params:  {diff['system_params']:,}")

    print(
        f"\nTotal training time: "
        f"{ecc['train_time'] + cam['train_time'] + diff['reg_time'] + diff['mem_time']:.1f}s"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print("Neural Memory Architectures for nCPU")
    print("Three novel architectures solving problems conventional memory cannot.")
    print(f"PyTorch {torch.__version__}, device: cpu")

    ecc_results = demo_ecc_memory()
    cam_results = demo_cam()
    diff_results = demo_differentiable_storage()

    print_summary(ecc_results, cam_results, diff_results)

    print("\nModels saved to:")
    print(f"  {_root / 'models' / 'neural_ecc_memory.pt'}")
    print(f"  {_root / 'models' / 'neural_cam.pt'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
