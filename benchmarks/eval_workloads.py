#!/usr/bin/env python3
"""neurOS Workload Evaluation.

Evaluates all neurOS neural components on representative workloads.
Measures accuracy (neural vs classical oracle) and performance.

Results are formatted for the paper's evaluation section.

Usage:
    python benchmarks/eval_workloads.py                # Full evaluation
    python benchmarks/eval_workloads.py --quick        # Quick smoke test
"""

import sys
import time
import torch
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, ".")
from ncpu.os import NeurOS
from ncpu.os.process import ProcessState

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Watchdog Training (doesn't exist yet, so create it)
# ═══════════════════════════════════════════════════════════════════════════════

def train_watchdog(nos: NeurOS, quick: bool = False) -> Dict:
    """Train watchdog on normal + anomalous data for proper discrimination."""
    logger.info("\n=== Training Watchdog ===")
    import torch.nn as nn

    window_size = nos.watchdog.window_size
    num_normal = 20 if quick else 200
    num_anomalous = 20 if quick else 200

    # Generate normal operation windows: smooth, healthy metrics
    windows = []
    targets = []
    for i in range(num_normal):
        window = []
        base_cpu = 0.3 + 0.2 * torch.sin(torch.tensor(i * 0.1)).item()
        for t in range(window_size):
            metrics = [
                max(0.0, min(1.0, base_cpu + 0.05 * torch.randn(1).item())),
                max(0.0, min(1.0, 0.4 + 0.1 * torch.randn(1).item())),
                max(0.0, min(1.0, 0.02 + 0.01 * torch.randn(1).item())),
                max(0.0, min(1.0, 0.95 + 0.03 * torch.randn(1).item())),
                max(0.0, min(1.0, 0.98 + 0.02 * torch.randn(1).item())),
                max(0.0, min(1.0, 0.1 + 0.05 * torch.randn(1).item())),
                max(0.0, min(1.0, 0.05 + 0.02 * torch.randn(1).item())),
                max(0.0, min(1.0, 0.05 + 0.02 * torch.randn(1).item())),
            ]
            window.append(metrics)
        windows.append(window)
        targets.append(0.0)

    # Generate anomalous windows: extreme, unhealthy metrics
    for i in range(num_anomalous):
        window = []
        anomaly_type = i % 4
        for t in range(window_size):
            if anomaly_type == 0:  # CPU overload
                metrics = [0.95 + 0.05 * torch.rand(1).item(), 0.85 + 0.15 * torch.rand(1).item(),
                           0.5 + 0.3 * torch.rand(1).item(), 0.1 + 0.1 * torch.rand(1).item(),
                           0.3 + 0.2 * torch.rand(1).item(), 0.7 + 0.3 * torch.rand(1).item(),
                           0.01 + 0.01 * torch.rand(1).item(), 0.7 + 0.3 * torch.rand(1).item()]
            elif anomaly_type == 1:  # Memory exhaustion
                metrics = [0.7 + 0.2 * torch.rand(1).item(), 0.95 + 0.05 * torch.rand(1).item(),
                           0.1 + 0.1 * torch.rand(1).item(), 0.3 + 0.1 * torch.rand(1).item(),
                           0.5 + 0.2 * torch.rand(1).item(), 0.8 + 0.2 * torch.rand(1).item(),
                           0.02 + 0.02 * torch.rand(1).item(), 0.5 + 0.3 * torch.rand(1).item()]
            elif anomaly_type == 2:  # Interrupt storm
                metrics = [0.8 + 0.2 * torch.rand(1).item(), 0.6 + 0.2 * torch.rand(1).item(),
                           0.8 + 0.2 * torch.rand(1).item(), 0.4 + 0.2 * torch.rand(1).item(),
                           0.4 + 0.2 * torch.rand(1).item(), 0.6 + 0.3 * torch.rand(1).item(),
                           0.03 + 0.02 * torch.rand(1).item(), 0.6 + 0.3 * torch.rand(1).item()]
            else:  # Cache thrashing
                metrics = [0.5 + 0.3 * torch.rand(1).item(), 0.7 + 0.2 * torch.rand(1).item(),
                           0.3 + 0.2 * torch.rand(1).item(), 0.05 + 0.05 * torch.rand(1).item(),
                           0.6 + 0.2 * torch.rand(1).item(), 0.5 + 0.3 * torch.rand(1).item(),
                           0.01 + 0.02 * torch.rand(1).item(), 0.8 + 0.2 * torch.rand(1).item()]
            metrics = [max(0.0, min(1.0, m)) for m in metrics]
            window.append(metrics)
        windows.append(window)
        targets.append(1.0)

    # Train with BCE loss: normal→0, anomalous→1
    data = torch.tensor(windows, dtype=torch.float32, device=nos.device)
    target_tensor = torch.tensor(targets, dtype=torch.float32, device=nos.device)

    optimizer = torch.optim.Adam(nos.watchdog.net.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    epochs = 30 if quick else 150

    nos.watchdog.net.train()
    best_loss = float('inf')
    for epoch in range(epochs):
        optimizer.zero_grad()
        scores = nos.watchdog.net(data).squeeze(-1)  # [N]
        loss = loss_fn(scores, target_tensor)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if epoch % 30 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                pred = (scores > 0.5).float()
                acc = (pred == target_tensor).float().mean().item()
            logger.info(f"  Epoch {epoch}: loss={loss.item():.4f}, acc={acc*100:.1f}%")

    nos.watchdog.net.eval()
    nos.watchdog._trained = True
    nos.watchdog.save()

    # Verify discrimination
    with torch.no_grad():
        all_scores = nos.watchdog.net(data).squeeze(-1)
        normal_scores = all_scores[:num_normal]
        anomalous_scores = all_scores[num_normal:]
        logger.info(f"  Normal scores: mean={normal_scores.mean():.4f}")
        logger.info(f"  Anomalous scores: mean={anomalous_scores.mean():.4f}")

    logger.info(f"  Watchdog trained: {len(windows)} windows, best_loss={best_loss:.6f}")
    return {"samples": len(windows), "final_loss": best_loss}


# ═══════════════════════════════════════════════════════════════════════════════
# Workload 1: Memory Subsystem (MMU + TLB + Cache)
# ═══════════════════════════════════════════════════════════════════════════════

def eval_memory_workload(nos: NeurOS, quick: bool = False) -> Dict:
    """Evaluate MMU, TLB, and Cache on memory access patterns."""
    logger.info("\n=== Memory Workload ===")

    results = {
        "tlb": {"sequential": {}, "strided": {}, "random": {}, "temporal": {}},
        "cache": {"sequential": {}, "random": {}},
        "mmu": {}
    }

    num_accesses = 100 if quick else 1000

    # Map some pages for testing
    for vpn in range(100):
        pfn = nos.mmu.alloc_and_map(vpn, asid=0, read=True, write=True)

    # TLB: Sequential access pattern (100+ pages in 64-entry TLB → forces evictions)
    logger.info("  [TLB] Sequential access...")
    neural_matches = 0
    total_decisions = 0
    t_start = time.perf_counter()

    # Fill TLB and force evictions by inserting more pages than TLB size
    perms = torch.ones(6, dtype=torch.float32, device=nos.device)
    for i in range(num_accesses):
        vpn = i % 100
        nos.tlb.lookup(vpn, asid=0)
        # Insert every page — forces evictions once TLB is full (size=64)
        if len((nos.tlb.vpn_tags >= 0).nonzero(as_tuple=True)[0]) == nos.tlb.size:
            neural_victim = nos.tlb._neural_evict() if nos.tlb._policy_trained else None
            lru_victim = nos.tlb._lru_evict()
            if neural_victim is not None:
                neural_matches += int(neural_victim == lru_victim)
                total_decisions += 1
        nos.tlb.insert(vpn, asid=0, pfn=vpn, perms=perms)

    t_elapsed = time.perf_counter() - t_start
    results["tlb"]["sequential"] = {
        "samples": total_decisions,
        "neural_accuracy": (neural_matches / max(1, total_decisions)) if total_decisions > 0 else 0.0,
        "hit_rate": nos.tlb.hit_rate,
        "latency_us": (t_elapsed / max(1, num_accesses)) * 1e6,
    }
    logger.info(f"    Sequential: {results['tlb']['sequential']['neural_accuracy']*100:.1f}% accuracy, "
                f"{results['tlb']['sequential']['hit_rate']*100:.1f}% hit rate")

    # TLB: Random access pattern
    logger.info("  [TLB] Random access...")
    nos.tlb.flush()
    nos.tlb.hits = nos.tlb.misses = 0
    neural_matches = total_decisions = 0

    import random
    random.seed(42)
    for i in range(num_accesses):
        vpn = random.randint(0, 99)
        nos.tlb.lookup(vpn, asid=0)
        if len((nos.tlb.vpn_tags >= 0).nonzero(as_tuple=True)[0]) == nos.tlb.size:
            neural_victim = nos.tlb._neural_evict() if nos.tlb._policy_trained else None
            lru_victim = nos.tlb._lru_evict()
            if neural_victim is not None:
                neural_matches += int(neural_victim == lru_victim)
                total_decisions += 1
        nos.tlb.insert(vpn, asid=0, pfn=vpn, perms=perms)

    results["tlb"]["random"] = {
        "samples": total_decisions,
        "neural_accuracy": (neural_matches / max(1, total_decisions)) if total_decisions > 0 else 0.0,
        "hit_rate": nos.tlb.hit_rate,
    }
    logger.info(f"    Random: {results['tlb']['random']['neural_accuracy']*100:.1f}% accuracy, "
                f"{results['tlb']['random']['hit_rate']*100:.1f}% hit rate")

    # Cache: Sequential access
    logger.info("  [Cache] Sequential access...")
    nos.cache.hits = nos.cache.misses = 0
    t_start = time.perf_counter()

    for i in range(num_accesses):
        addr = i * 64  # Sequential, cache-line aligned
        nos.cache.access(addr, write=False)

    t_elapsed = time.perf_counter() - t_start
    results["cache"]["sequential"] = {
        "hit_rate": nos.cache.hit_rate,
        "latency_us": (t_elapsed / max(1, num_accesses)) * 1e6,
    }
    logger.info(f"    Sequential: {results['cache']['sequential']['hit_rate']*100:.1f}% hit rate")

    # Cache: Random access
    nos.cache.flush()
    nos.cache.hits = nos.cache.misses = 0
    for i in range(num_accesses):
        addr = random.randint(0, 10000) * 64
        nos.cache.access(addr, write=False)

    results["cache"]["random"] = {
        "hit_rate": nos.cache.hit_rate,
    }
    logger.info(f"    Random: {results['cache']['random']['hit_rate']*100:.1f}% hit rate")

    # MMU: Translation accuracy
    logger.info("  [MMU] Translation accuracy...")
    correct_translations = 0
    for vpn in range(100):
        phys_addr, fault = nos.mmu.translate(vpn * 4096, asid=0)
        expected_pfn = vpn
        if fault is None and (phys_addr >> 12) == expected_pfn:
            correct_translations += 1

    t_start = time.perf_counter()
    for _ in range(100):
        nos.mmu.translate(random.randint(0, 99) * 4096, asid=0)
    t_elapsed = time.perf_counter() - t_start

    results["mmu"] = {
        "accuracy": correct_translations / 100.0,
        "latency_us": (t_elapsed / 100) * 1e6,
    }
    logger.info(f"    MMU: {results['mmu']['accuracy']*100:.1f}% accuracy")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Workload 2: Scheduler
# ═══════════════════════════════════════════════════════════════════════════════

def eval_scheduler_workload(nos: NeurOS, quick: bool = False) -> Dict:
    """Evaluate scheduler on CPU-bound, I/O-bound, and mixed workloads."""
    logger.info("\n=== Scheduler Workload ===")

    results = {"cpu_bound": {}, "io_bound": {}, "mixed": {}}

    # CPU-bound workload: long CPU bursts
    logger.info("  [Scheduler] CPU-bound processes...")
    procs = []
    for i in range(4):
        p = nos.process_table.create_process(f"cpu_worker_{i}", priority=128)
        p.time_slice = 100
        procs.append(p)

    neural_matches = 0
    total_decisions = 0
    t_start = time.perf_counter()

    for tick in range(100 if quick else 500):
        # Get ready list
        ready = [p for p in procs if p.state == ProcessState.READY or p.state == ProcessState.RUNNING]
        if len(ready) < 2:
            continue

        # Compare neural vs priority scheduler
        if nos.scheduler._trained:
            neural_selected = nos.scheduler._neural_schedule(ready)
        else:
            neural_selected = None
        priority_selected = nos.scheduler._priority_schedule(ready)

        if neural_selected is not None:
            neural_matches += int(neural_selected.pid == priority_selected.pid)
            total_decisions += 1

        # Tick the selected process
        selected = nos.scheduler.schedule()
        if selected:
            nos.scheduler.tick_process(selected)

    t_elapsed = time.perf_counter() - t_start
    fairness = nos.scheduler.jains_fairness()

    results["cpu_bound"] = {
        "samples": total_decisions,
        "neural_accuracy": (neural_matches / max(1, total_decisions)) if total_decisions > 0 else 0.0,
        "fairness": fairness,
        "latency_us": (t_elapsed / max(1, nos.scheduler.total_decisions)) * 1e6,
    }
    logger.info(f"    CPU-bound: {results['cpu_bound']['neural_accuracy']*100:.1f}% accuracy, "
                f"fairness={fairness:.3f}")

    # Clean up
    for p in procs:
        nos.process_table.remove(p.pid)
    nos.scheduler._cpu_shares.clear()
    nos.scheduler.total_decisions = 0
    nos.scheduler.total_switches = 0

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Workload 3: Filesystem & Block Allocator
# ═══════════════════════════════════════════════════════════════════════════════

def eval_filesystem_workload(nos: NeurOS, quick: bool = False) -> Dict:
    """Evaluate filesystem and block allocator."""
    logger.info("\n=== Filesystem Workload ===")

    results = {"allocator": {}, "read": {}, "write": {}}

    num_files = 10 if quick else 100

    # Block allocator: sequential writes
    logger.info("  [Filesystem] Sequential writes...")
    t_start = time.perf_counter()

    for i in range(num_files):
        fname = f"/tmp/test_{i}.dat"
        data = torch.randint(0, 256, (512,), dtype=torch.uint8, device=nos.device)
        nos.fs.write_file(fname, data)

    t_elapsed = time.perf_counter() - t_start

    results["write"] = {
        "throughput_files_per_sec": num_files / max(t_elapsed, 1e-6),
        "latency_ms": (t_elapsed / num_files) * 1000,
    }
    logger.info(f"    Write: {results['write']['latency_ms']:.2f} ms/file")

    # Read throughput
    logger.info("  [Filesystem] Random reads...")
    import random
    random.seed(42)

    t_start = time.perf_counter()
    for _ in range(num_files):
        idx = random.randint(0, num_files - 1)
        fname = f"/tmp/test_{idx}.dat"
        nos.fs.read_file(fname)

    t_elapsed = time.perf_counter() - t_start

    results["read"] = {
        "throughput_files_per_sec": num_files / max(t_elapsed, 1e-6),
        "latency_us": (t_elapsed / num_files) * 1e6,
    }
    logger.info(f"    Read: {results['read']['latency_us']:.2f} µs/file")

    # Allocator accuracy: check if it's first-fit
    logger.info("  [Block Allocator] First-fit accuracy...")
    # The classical first-fit would allocate sequentially
    # Just verify blocks were allocated
    results["allocator"] = {
        "blocks_allocated": nos.fs.used_blocks,
        "trained": nos.fs._allocator_trained,
    }
    logger.info(f"    Allocator: {results['allocator']['blocks_allocated']} blocks allocated")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Workload 4: Interrupt Controller (GIC)
# ═══════════════════════════════════════════════════════════════════════════════

def eval_interrupt_workload(nos: NeurOS, quick: bool = False) -> Dict:
    """Evaluate GIC on burst interrupts and priority ordering."""
    logger.info("\n=== Interrupt Workload ===")

    results = {"burst": {}, "priority": {}}

    num_bursts = 10 if quick else 100

    # Burst interrupts
    logger.info("  [GIC] Burst interrupts...")
    neural_matches = 0
    total_decisions = 0
    t_start = time.perf_counter()

    for burst in range(num_bursts):
        # Raise multiple interrupts
        for irq in range(0, 8):
            nos.gic.raise_irq(irq)

        # Dispatch all
        while nos.gic.pending().any():
            # Compare neural vs classical dispatch
            pending = nos.gic.pending()
            if nos.gic._trained:
                neural_irq = nos.gic._neural_dispatch(pending)
            else:
                neural_irq = None
            classical_irq = nos.gic._fixed_dispatch(pending)

            if neural_irq is not None:
                neural_matches += int(neural_irq == classical_irq)
                total_decisions += 1

            nos.gic.dispatch()

    t_elapsed = time.perf_counter() - t_start

    results["burst"] = {
        "samples": total_decisions,
        "neural_accuracy": (neural_matches / max(1, total_decisions)) if total_decisions > 0 else 0.0,
        "latency_us": (t_elapsed / max(1, total_decisions)) * 1e6,
    }
    logger.info(f"    Burst: {results['burst']['neural_accuracy']*100:.1f}% accuracy")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Workload 5: Toolchain (Assembler + Compiler)
# ═══════════════════════════════════════════════════════════════════════════════

def eval_toolchain_workload(nos: NeurOS, quick: bool = False) -> Dict:
    """Evaluate assembler and compiler on programs."""
    logger.info("\n=== Toolchain Workload ===")

    results = {"assembler": {}, "compiler": {}}

    # Load assembly programs
    asm_programs = list(Path("programs").rglob("*.asm"))[:5 if quick else 20]

    logger.info(f"  [Assembler] Assembling {len(asm_programs)} programs...")
    exact_matches = 0
    total_instructions = 0
    t_start = time.perf_counter()

    for asm_file in asm_programs:
        source = asm_file.read_text()
        # Classical assembly
        classical_result = nos.assembler.classical.assemble(source)
        # Neural assembly
        neural_result = nos.assembler.assemble_neural(source)

        if classical_result.success and neural_result.success:
            # Compare binary outputs
            for c_word, n_word in zip(classical_result.binary, neural_result.binary):
                total_instructions += 1
                if c_word == n_word:
                    exact_matches += 1

    t_elapsed = time.perf_counter() - t_start

    results["assembler"] = {
        "programs": len(asm_programs),
        "instructions": total_instructions,
        "exact_match_rate": exact_matches / max(1, total_instructions),
        "latency_ms": (t_elapsed / len(asm_programs)) * 1000,
    }
    logger.info(f"    Assembler: {results['assembler']['exact_match_rate']*100:.1f}% exact match")

    # Compiler: nsl programs
    logger.info("  [Compiler] Compiling nsl programs...")
    nsl_programs = [
        "var x = 10; var y = 20; var z = x + y; halt;",
        "var a = 5; var b = a * 7; halt;",
        "var i = 0; while (i < 10) { i = i + 1; } halt;",
    ]

    if not quick:
        nsl_programs.extend([
            "var n = 10; var s = 0; for (var i = 0; i < n; i = i + 1) { s = s + i; } halt;",
            "var x = 15; if (x > 10) { x = x - 5; } else { x = x + 5; } halt;",
        ])

    compiled = 0
    correct = 0
    t_start = time.perf_counter()

    for prog in nsl_programs:
        result = nos.compiler.compile(prog)
        compiled += 1
        if result.success and result.binary:
            correct += 1

    t_elapsed = time.perf_counter() - t_start

    results["compiler"] = {
        "programs": compiled,
        "success_rate": correct / max(1, compiled),
        "latency_us": (t_elapsed / compiled) * 1e6,
    }
    logger.info(f"    Compiler: {results['compiler']['success_rate']*100:.1f}% success rate")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Workload 6: Watchdog
# ═══════════════════════════════════════════════════════════════════════════════

def eval_watchdog_workload(nos: NeurOS, quick: bool = False) -> Dict:
    """Evaluate watchdog on normal vs anomalous conditions."""
    logger.info("\n=== Watchdog Workload ===")

    results = {"normal": {}, "anomalous": {}}

    # Normal operation
    logger.info("  [Watchdog] Normal operation...")
    false_positives = 0
    for _ in range(10 if quick else 50):
        nos.watchdog.collect_from_os(nos)
        alert = nos.watchdog.check()
        if alert:
            false_positives += 1

    results["normal"] = {
        "checks": nos.watchdog.total_checks,
        "false_positive_rate": false_positives / max(1, nos.watchdog.total_checks),
    }
    logger.info(f"    Normal: {results['normal']['false_positive_rate']*100:.1f}% false positives")

    # Anomalous conditions: high memory pressure
    logger.info("  [Watchdog] Anomalous conditions...")
    initial_alerts = nos.watchdog.total_alerts

    # Fill the entire window with anomalous metrics to give LSTM enough context
    n_anomalous = nos.watchdog.window_size + (5 if quick else 20)
    n_checks = 5 if quick else 20
    checks_done = 0
    for i in range(n_anomalous):
        nos.watchdog.record_metrics(
            cpu_util=0.99,  # Very high
            mem_pressure=0.95,  # Very high
            interrupt_rate=0.8,  # Very high
            cache_hit_rate=0.05,  # Very low
            scheduler_fairness=0.3,  # Poor
            ipc_queue_depth=0.9,  # Full queues
            fs_ops_rate=0.01,  # Near zero
            tlb_miss_rate=0.9,  # Almost all misses
        )
        # Only check after buffer is filled with anomalous data
        if i >= nos.watchdog.window_size:
            nos.watchdog.check()
            checks_done += 1

    detected = nos.watchdog.total_alerts - initial_alerts

    results["anomalous"] = {
        "samples": checks_done,
        "detection_rate": detected / max(1, checks_done),
    }
    logger.info(f"    Anomalous: {results['anomalous']['detection_rate']*100:.1f}% detection rate")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main Evaluation Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(quick: bool = False):
    """Run complete neurOS workload evaluation."""
    logger.info("=" * 70)
    logger.info("neurOS Component Evaluation")
    logger.info("=" * 70)

    # Boot neurOS with models
    logger.info("\n=== Booting neurOS ===")
    nos = NeurOS()
    boot_stats = nos.boot(load_models=True, quiet=False)
    logger.info(f"  Boot time: {boot_stats['total']*1000:.1f} ms")

    all_results = {}

    try:
        # Train watchdog (with both normal + anomalous examples)
        all_results["watchdog_training"] = train_watchdog(nos, quick=quick)

        # Run watchdog first (before other workloads stress the OS)
        all_results["watchdog"] = eval_watchdog_workload(nos, quick=quick)

        # Run remaining workloads
        all_results["memory"] = eval_memory_workload(nos, quick=quick)
        all_results["scheduler"] = eval_scheduler_workload(nos, quick=quick)
        all_results["filesystem"] = eval_filesystem_workload(nos, quick=quick)
        all_results["interrupts"] = eval_interrupt_workload(nos, quick=quick)
        all_results["toolchain"] = eval_toolchain_workload(nos, quick=quick)

    except Exception as e:
        logger.error(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()

    # Save results
    output_file = Path("benchmarks/results/local/eval_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {output_file}")

    # Generate markdown summary
    generate_markdown_summary(all_results)

    return all_results


def generate_markdown_summary(results: Dict):
    """Generate markdown tables for the paper."""
    logger.info("\n" + "=" * 70)
    logger.info("PAPER-READY RESULTS")
    logger.info("=" * 70)

    # Accuracy Table
    logger.info("\n### Accuracy (Neural vs Classical Oracle)\n")
    logger.info("| Component | Workload | Samples | Neural Accuracy | Classical Fallback |")
    logger.info("|-----------|----------|---------|-----------------|-------------------|")

    # TLB
    if "memory" in results and "tlb" in results["memory"]:
        for pattern, data in results["memory"]["tlb"].items():
            if "neural_accuracy" in data:
                logger.info(f"| TLB | {pattern.capitalize()} | {data.get('samples', 0)} | "
                           f"{data['neural_accuracy']*100:.1f}% | LRU |")

    # Scheduler
    if "scheduler" in results:
        for workload, data in results["scheduler"].items():
            if "neural_accuracy" in data:
                logger.info(f"| Scheduler | {workload.replace('_', ' ').title()} | {data.get('samples', 0)} | "
                           f"{data['neural_accuracy']*100:.1f}% | Priority+aging |")

    # GIC
    if "interrupts" in results:
        for workload, data in results["interrupts"].items():
            if "neural_accuracy" in data:
                logger.info(f"| GIC | {workload.capitalize()} | {data.get('samples', 0)} | "
                           f"{data['neural_accuracy']*100:.1f}% | Fixed priority |")

    # Assembler
    if "toolchain" in results and "assembler" in results["toolchain"]:
        data = results["toolchain"]["assembler"]
        logger.info(f"| Assembler | Binary codegen | {data.get('instructions', 0)} | "
                   f"{data.get('exact_match_rate', 0)*100:.1f}% | Classical asm |")

    # Compiler
    if "toolchain" in results and "compiler" in results["toolchain"]:
        data = results["toolchain"]["compiler"]
        logger.info(f"| Compiler | nsl → asm | {data.get('programs', 0)} | "
                   f"{data.get('success_rate', 0)*100:.1f}% | N/A |")

    # MMU
    if "memory" in results and "mmu" in results["memory"]:
        data = results["memory"]["mmu"]
        logger.info(f"| MMU | Page translation | 100 | "
                   f"{data.get('accuracy', 0)*100:.1f}% | Classical table |")

    # Watchdog
    if "watchdog" in results:
        wd = results["watchdog"]
        if "normal" in wd:
            logger.info(f"| Watchdog | Normal operation | {wd['normal'].get('checks', 0)} | "
                       f"{(1-wd['normal'].get('false_positive_rate', 0))*100:.1f}% true negative | Heuristic |")
        if "anomalous" in wd:
            logger.info(f"| Watchdog | Anomaly detection | {wd['anomalous'].get('samples', 0)} | "
                       f"{wd['anomalous'].get('detection_rate', 0)*100:.1f}% detection | Heuristic |")

    # Performance Table
    logger.info("\n### Performance (with Neural Models)\n")
    logger.info("| Component | Operation | Latency | Throughput |")
    logger.info("|-----------|-----------|---------|------------|")

    # Memory subsystem
    if "memory" in results:
        if "mmu" in results["memory"]:
            lat = results["memory"]["mmu"].get("latency_us", 0)
            logger.info(f"| MMU | Page translation | {lat:.1f} µs | {1e6/max(lat,1):.0f} ops/s |")

        if "tlb" in results["memory"] and "sequential" in results["memory"]["tlb"]:
            lat = results["memory"]["tlb"]["sequential"].get("latency_us", 0)
            logger.info(f"| TLB | Lookup | {lat:.1f} µs | {1e6/max(lat,1):.0f} ops/s |")

        if "cache" in results["memory"] and "sequential" in results["memory"]["cache"]:
            lat = results["memory"]["cache"]["sequential"].get("latency_us", 0)
            logger.info(f"| Cache | Access | {lat:.1f} µs | {1e6/max(lat,1):.0f} ops/s |")

    # Scheduler
    if "scheduler" in results and "cpu_bound" in results["scheduler"]:
        lat = results["scheduler"]["cpu_bound"].get("latency_us", 0)
        logger.info(f"| Scheduler | Decision | {lat:.1f} µs | {1e6/max(lat,1):.0f} ops/s |")

    # Filesystem
    if "filesystem" in results:
        if "read" in results["filesystem"]:
            lat = results["filesystem"]["read"].get("latency_us", 0)
            logger.info(f"| Filesystem | Read | {lat:.1f} µs | {1e6/max(lat,1):.0f} ops/s |")
        if "write" in results["filesystem"]:
            lat = results["filesystem"]["write"].get("latency_ms", 0) * 1000
            logger.info(f"| Filesystem | Write | {lat:.1f} µs | {1e6/max(lat,1):.0f} ops/s |")

    # Toolchain
    if "toolchain" in results:
        if "assembler" in results["toolchain"]:
            lat = results["toolchain"]["assembler"].get("latency_ms", 0) * 1000
            logger.info(f"| Assembler | Program | {lat:.1f} µs | {1e6/max(lat,1):.0f} progs/s |")
        if "compiler" in results["toolchain"]:
            lat = results["toolchain"]["compiler"].get("latency_us", 0)
            logger.info(f"| Compiler | nsl program | {lat:.1f} µs | {1e6/max(lat,1):.0f} progs/s |")

    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="neurOS workload evaluation")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    args = parser.parse_args()

    run_evaluation(quick=args.quick)
