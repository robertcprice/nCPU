#!/usr/bin/env python3
"""Differentiable OS Demo: learn optimal OS policies via gradient descent.

Demonstrates the DifferentiableOS learning to outperform heuristic baselines
(round-robin scheduling, LRU caching, first-fit allocation) by training on
synthetic workload traces.

Usage:
    python demos/neural/differentiable_os_demo.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ncpu.differentiable.differentiable_os import (
    DifferentiableOS,
    generate_workload,
    evaluate_baseline,
)


def main():
    print("=" * 60)
    print("  Differentiable OS: Learning Optimal Policies")
    print("=" * 60)

    n_processes = 4
    cache_size = 8
    n_events = 300
    n_epochs = 300

    # Generate workload
    print(f"\n1. Generating synthetic workload ({n_events} events, "
          f"{n_processes} processes)...")
    workload = generate_workload(n_events=n_events, n_processes=n_processes,
                                  locality=0.7)
    print(f"   Generated {len(workload)} events with temporal locality")

    # Evaluate baselines
    print("\n2. Evaluating baseline policies...")
    baseline = evaluate_baseline(workload, n_processes, cache_size)
    print(f"   Round-Robin scheduling accuracy: {baseline['schedule_accuracy']:.1%}")
    print(f"   LRU cache hit rate:              {baseline['cache_hit_rate']:.1%}")
    print(f"   First-Fit fragmentation:         {baseline['fragmentation']:.3f}")

    # Train differentiable OS
    print(f"\n3. Training DifferentiableOS ({n_epochs} epochs)...")
    diff_os = DifferentiableOS(
        n_processes=n_processes,
        cache_size=cache_size,
        memory_size=256,
    )

    param_count = sum(p.numel() for p in diff_os.parameters())
    print(f"   Total parameters: {param_count:,}")

    metrics = diff_os.optimize(
        workload,
        n_epochs=n_epochs,
        lr=1e-3,
        temperature_start=2.0,
        temperature_end=0.1,
    )

    # Final evaluation
    print("\n4. Final evaluation on the same workload...")
    diff_os.reset()
    diff_os.eval()

    schedule_correct = 0
    total = 0

    with torch.no_grad():
        for event in workload:
            schedule_w, eviction_w, placement_w = diff_os.step(event, temperature=0.1)
            chosen = int(schedule_w.argmax().item())
            if event.optimal_process is not None and chosen == event.optimal_process:
                schedule_correct += 1
            total += 1

    learned_schedule_acc = schedule_correct / max(total, 1)
    learned_cache_hr = diff_os.cache.hit_rate
    learned_frag = diff_os.allocator.fragmentation

    # Compare
    print("\n" + "=" * 60)
    print("  RESULTS: Learned vs Heuristic Policies")
    print("=" * 60)

    print(f"\n  {'Metric':<30s} {'Heuristic':>10s} {'Learned':>10s} {'Delta':>10s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

    sched_delta = learned_schedule_acc - baseline["schedule_accuracy"]
    cache_delta = learned_cache_hr - baseline["cache_hit_rate"]
    frag_delta = baseline["fragmentation"] - learned_frag  # Lower frag is better

    print(f"  {'Scheduling accuracy':<30s} "
          f"{baseline['schedule_accuracy']:>9.1%} "
          f"{learned_schedule_acc:>9.1%} "
          f"{'+'if sched_delta>=0 else ''}{sched_delta:>8.1%}")

    print(f"  {'Cache hit rate':<30s} "
          f"{baseline['cache_hit_rate']:>9.1%} "
          f"{learned_cache_hr:>9.1%} "
          f"{'+'if cache_delta>=0 else ''}{cache_delta:>8.1%}")

    print(f"  {'Fragmentation (lower=better)':<30s} "
          f"{baseline['fragmentation']:>9.3f} "
          f"{learned_frag:>9.3f} "
          f"{'+'if frag_delta>=0 else ''}{frag_delta:>8.3f}")

    # Training curve summary
    print(f"\n  Training curve:")
    print(f"    Initial loss: {metrics['loss'][0]:.4f}")
    print(f"    Final loss:   {metrics['loss'][-1]:.4f}")
    print(f"    Initial cache HR: {metrics['cache_hit_rate'][0]:.1%}")
    print(f"    Final cache HR:   {metrics['cache_hit_rate'][-1]:.1%}")

    # Verdict
    improvements = 0
    if sched_delta > 0:
        improvements += 1
    if cache_delta > 0:
        improvements += 1
    if frag_delta > 0:
        improvements += 1

    print(f"\n  Improved on {improvements}/3 metrics vs heuristic baselines.")

    if improvements >= 2:
        print("  The differentiable OS successfully learned better policies.")
    elif improvements >= 1:
        print("  The differentiable OS improved on some metrics.")
    else:
        print("  Heuristics were competitive on this workload.")

    print()
    return improvements


if __name__ == "__main__":
    improvements = main()
    sys.exit(0 if improvements >= 1 else 1)
