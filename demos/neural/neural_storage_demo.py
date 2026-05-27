#!/usr/bin/env python3
"""Neural Storage Demo -- register file and SSD memory with neural prefetch.

Demonstrates the two novel neural storage components for nCPU:

  1. Neural Register File: registers stored as learned embeddings in a weight
     matrix, with encoder/decoder MLPs for int64 <-> embedding conversion.
     Trained online to 100% lossless reconstruction.

  2. SSD-Backed Neural Memory: memory-mapped file storage with a neural page
     cache. The trained prefetch.pt LSTM predicts upcoming pages from the
     address stream. The trained mmu.pt MLP provides address translation.

Usage:
    python demos/neural/neural_storage_demo.py
    python demos/neural/neural_storage_demo.py --skip-training  # use saved model
    python demos/neural/neural_storage_demo.py --device mps     # GPU training
"""

from __future__ import annotations

import argparse
import random
import struct
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


def section(title: str) -> None:
    """Print a section header."""
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def subsection(title: str) -> None:
    """Print a subsection header."""
    print(f"\n  --- {title} ---")


# ═══════════════════════════════════════════════════════════════════════════
# Part 1: Neural Register File
# ═══════════════════════════════════════════════════════════════════════════

def demo_register_file(device: str = "cpu", skip_training: bool = False) -> None:
    """Train, verify, and demonstrate the Neural Register File."""
    from ncpu.neural.neural_registers import (
        NeuralRegisterFile,
        train_register_file,
        verify_register_file,
    )

    section("NEURAL REGISTER FILE")
    print("  Registers live as learned embeddings in a weight matrix.")
    print("  Every read/write passes through trained encoder/decoder MLPs.")

    # ── Train or load ────────────────────────────────────────────────
    model_path = PROJECT_ROOT / "models" / "neural_registers.pt"

    if skip_training and model_path.exists():
        subsection("Loading pre-trained register file")
        rf = NeuralRegisterFile.load(path=model_path, device=device)
        print(f"  Loaded from {model_path}")
        print(f"  Parameters: {rf.stats['param_count']:,}")
    else:
        subsection("Training register file (self-supervised autoencoder)")
        print(f"  Device: {device}")
        print(f"  Target: 100% lossless int64 round-trip")
        print()
        rf = train_register_file(
            epochs=1500,
            batch_size=2048,
            lr=2e-3,
            device=device,
            save_path=model_path,
            verbose=True,
        )

    # ── Lossless verification ────────────────────────────────────────
    subsection("Lossless verification (1000 random int64 values)")
    correct, total = verify_register_file(rf, n_tests=1000, verbose=True)

    # ── Demonstrate read/write ───────────────────────────────────────
    subsection("Register read/write demonstration")

    test_values = {
        "x0":  42,
        "x1":  -1,
        "x2":  0,
        "x3":  (1 << 63) - 1,    # max int64
        "x4":  -(1 << 63),       # min int64
        "x5":  0xDEADBEEF,
        "x6":  -12345678,
        "x7":  255,
        "x8":  (1 << 32),        # 2^32 (needs >32 bits)
        "x9":  0x7FFFFFFFFFFFFFFF,
    }

    all_ok = True
    for name, value in test_values.items():
        reg_idx = int(name[1:])
        rf.write(reg_idx, value)
        result = rf.read(reg_idx)
        status = "OK" if result == value else "FAIL"
        if result != value:
            all_ok = False
        # Show compact hex for large values
        if abs(value) > 0xFFFF:
            print(f"    {name} = {value:#18x}  ->  {result:#18x}  [{status}]")
        else:
            print(f"    {name} = {value:18d}  ->  {result:18d}  [{status}]")

    # ── XZR convention ───────────────────────────────────────────────
    subsection("XZR (zero register) convention")
    rf.write(31, 9999)
    xzr_val = rf.read(31)
    print(f"    Write 9999 to x31 (XZR), read back: {xzr_val}")
    print(f"    XZR always returns 0: {'OK' if xzr_val == 0 else 'FAIL'}")

    # ── Batch operations ─────────────────────────────────────────────
    subsection("Batch operations")
    batch_values = [(i, random.randint(-1000, 1000)) for i in range(16)]
    rf.write_batch(batch_values)
    batch_results = rf.read_batch([i for i, _ in batch_values])
    batch_ok = sum(1 for (_, v), r in zip(batch_values, batch_results) if v == r)
    print(f"    Wrote 16 registers in batch, read back: {batch_ok}/16 correct")

    # ── Metal export ─────────────────────────────────────────────────
    subsection("Metal shader weight export")
    flat = rf.export_flat_weights()
    total_floats = sum(len(v) for v in flat.values())
    print(f"    Exported {len(flat)} weight buffers ({total_floats:,} floats)")
    for name, values in flat.items():
        print(f"      {name}: {len(values)} floats")

    metal_path = rf.export_metal_binary()
    print(f"    Binary export: {metal_path} ({metal_path.stat().st_size:,} bytes)")

    # ── Stats ────────────────────────────────────────────────────────
    subsection("Register file statistics")
    stats = rf.stats
    for k, v in stats.items():
        print(f"    {k}: {v}")

    # ── Throughput ───────────────────────────────────────────────────
    subsection("Throughput measurement")
    n_ops = 10000
    t0 = time.perf_counter()
    for i in range(n_ops):
        rf.write(i % 31, i * 7 - 3)
    t1 = time.perf_counter()
    write_ops = n_ops / (t1 - t0)

    t0 = time.perf_counter()
    for i in range(n_ops):
        _ = rf.read(i % 31)
    t1 = time.perf_counter()
    read_ops = n_ops / (t1 - t0)

    print(f"    Write throughput: {write_ops:,.0f} ops/sec")
    print(f"    Read throughput:  {read_ops:,.0f} ops/sec")

    return correct == total and all_ok


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: SSD-Backed Neural Memory
# ═══════════════════════════════════════════════════════════════════════════

def demo_ssd_memory(device: str = "cpu") -> None:
    """Demonstrate SSD-backed neural memory with prefetch prediction."""
    from ncpu.neural.neural_memory import NeuralSSDMemory, benchmark_ssd_memory

    section("SSD-BACKED NEURAL MEMORY")
    print("  Memory-mapped file storage with neural prefetch prediction.")
    print("  Uses trained prefetch.pt (LSTM) and mmu.pt (MLP) models.")

    mem_size = 8 * 1024 * 1024  # 8 MB for demo

    with NeuralSSDMemory(size=mem_size, page_size=4096,
                         max_cached_pages=256, device=device) as mem:

        # ── Model status ─────────────────────────────────────────────
        subsection("Neural model status")
        print(f"    Prefetch LSTM: {'loaded' if mem._prefetch_loaded else 'not found'}")
        print(f"    MMU MLP:       {'loaded' if mem._mmu_loaded else 'not found'}")
        print(f"    Memory size:   {mem.size // (1024*1024)} MB "
              f"({mem.total_pages} pages)")
        print(f"    Page size:     {mem.page_size} bytes")
        print(f"    Cache capacity: {mem.max_cached_pages} pages "
              f"({mem.max_cached_pages * mem.page_size // 1024} KB)")

        # ── Load a "program" into memory ─────────────────────────────
        subsection("Loading program into SSD memory")

        # Simulate ARM64 instructions (4 bytes each)
        program = bytearray()
        instructions = [
            0xD2800140,  # MOV X0, #10
            0xD2800001,  # MOV X1, #0
            0x8B010000,  # ADD X0, X0, X1
            0xD1000400,  # SUB X0, X0, #1
            0xB5FFFFE0,  # CBNZ X0, -8
            0xD4000681,  # SVC #0x34 (exit)
        ]
        for inst in instructions:
            program.extend(struct.pack("<I", inst))

        base_addr = 0x10000
        mem.load_program(base_addr, bytes(program))
        print(f"    Loaded {len(instructions)} instructions at {base_addr:#x}")
        print(f"    Program size: {len(program)} bytes")

        # Read back and verify
        readback = mem.read(base_addr, len(program))
        match = readback == bytes(program)
        print(f"    Read-back verification: {'OK' if match else 'FAIL'}")

        # Hex dump
        print(f"\n    Memory dump:")
        print(mem.hexdump(base_addr, len(program)))

        # ── Sequential access pattern ────────────────────────────────
        subsection("Sequential access pattern (program fetch simulation)")
        # Simulate instruction fetch: read 4 bytes at a time, sequentially
        n_fetches = 200
        data_base = 0x50000
        # Write sequential data
        for i in range(n_fetches):
            mem.write_u32(data_base + i * 4, i * 0x1111)

        # Reset stats
        mem._stats = {k: 0 for k in mem._stats}
        mem._prefetched_pages.clear()

        t0 = time.perf_counter()
        for i in range(n_fetches):
            _ = mem.read_u32(data_base + i * 4)
        t1 = time.perf_counter()

        stats = mem.stats
        print(f"    Fetched {n_fetches} words in {(t1-t0)*1000:.1f} ms")
        print(f"    Cache hits: {stats['cache_hits']}, "
              f"misses: {stats['cache_misses']}")
        print(f"    Hit rate: {stats['hit_rate']:.3f}")
        if stats['prefetch_runs'] > 0:
            print(f"    Prefetch runs: {stats['prefetch_runs']}, "
                  f"loads: {stats['prefetch_loads']}, "
                  f"hits: {stats['prefetch_hits']}")

        # ── Random access pattern ────────────────────────────────────
        subsection("Random access pattern (data structure traversal)")
        # Simulate hash table / pointer chasing
        random.seed(42)
        n_random = 500
        random_addrs = [random.randint(0, mem_size - 8) & ~3
                        for _ in range(n_random)]

        # Pre-fill
        for addr in set(random_addrs):
            mem.write_u32(addr, addr ^ 0xDEADBEEF)

        mem._stats = {k: 0 for k in mem._stats}
        mem._prefetched_pages.clear()
        mem._cache.clear()
        mem._dirty.clear()

        t0 = time.perf_counter()
        for addr in random_addrs:
            _ = mem.read_u32(addr)
        t1 = time.perf_counter()

        stats = mem.stats
        print(f"    {n_random} random reads in {(t1-t0)*1000:.1f} ms")
        print(f"    Cache hits: {stats['cache_hits']}, "
              f"misses: {stats['cache_misses']}")
        print(f"    Hit rate: {stats['hit_rate']:.3f}")
        if stats['prefetch_runs'] > 0:
            print(f"    Prefetch runs: {stats['prefetch_runs']}, "
                  f"loads: {stats['prefetch_loads']}, "
                  f"hits: {stats['prefetch_hits']}")

        # ── Neural MMU demonstration ─────────────────────────────────
        subsection("Neural MMU lookup")
        test_addrs = [0x10000, 0x50000, 0x80000, 0xFF000]
        for addr in test_addrs:
            result = mem.neural_mmu_lookup(addr, asid=0)
            if result["available"]:
                print(f"    addr={addr:#08x}  vpn={result['vpn']:4d}  "
                      f"output_norm={result['output_norm']:.2f}")
            else:
                print(f"    addr={addr:#08x}  MMU model not available")

        # ── Dirty page writeback ─────────────────────────────────────
        subsection("Dirty page writeback")
        for i in range(10):
            mem.write_u32(i * 4096, i)
        flushed = mem.flush()
        print(f"    Flushed {flushed} dirty pages to mmap")

        # Verify persistence: re-read from mmap directly
        mem._cache.clear()
        for i in range(10):
            val = mem.read_u32(i * 4096)
            assert val == i, f"Persistence check failed at page {i}"
        print(f"    Persistence verification: OK (10 pages)")

    # ── Full benchmark ───────────────────────────────────────────────
    subsection("Full benchmark")
    results = benchmark_ssd_memory(size=8 * 1024 * 1024, verbose=True)

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Part 3: Integration -- Register File + SSD Memory Together
# ═══════════════════════════════════════════════════════════════════════════

def demo_integration(device: str = "cpu") -> None:
    """Show register file and SSD memory working together."""
    from ncpu.neural.neural_registers import NeuralRegisterFile
    from ncpu.neural.neural_memory import NeuralSSDMemory

    section("INTEGRATION: REGISTER FILE + SSD MEMORY")
    print("  Simulating a simple load/store loop with both neural components.")

    model_path = PROJECT_ROOT / "models" / "neural_registers.pt"
    if not model_path.exists():
        print("  (Skipping -- register file not trained yet)")
        return

    rf = NeuralRegisterFile.load(path=model_path, device=device)

    with NeuralSSDMemory(size=1024 * 1024, page_size=4096,
                         max_cached_pages=64, device=device) as mem:

        # Store values at addresses via registers
        subsection("Store loop: register -> memory")
        test_data = [(i, i * 100 + 7) for i in range(20)]
        for reg_idx, value in test_data:
            # Write value to register via neural encoder
            actual_reg = reg_idx % 31
            rf.write(actual_reg, value)
            # Read it back via neural decoder
            reg_value = rf.read(actual_reg)
            # Store to SSD memory
            addr = 0x1000 + reg_idx * 8
            mem.write_u64(addr, reg_value & 0xFFFFFFFFFFFFFFFF)

        subsection("Load loop: memory -> register")
        all_ok = True
        for reg_idx, expected_value in test_data:
            addr = 0x1000 + reg_idx * 8
            # Load from SSD memory
            mem_value = mem.read_u64(addr)
            # Convert back to signed
            if mem_value >= (1 << 63):
                mem_value -= (1 << 64)
            # Write to register
            actual_reg = reg_idx % 31
            rf.write(actual_reg, mem_value)
            # Read back
            final = rf.read(actual_reg)
            ok = final == expected_value
            if not ok:
                all_ok = False
            if reg_idx < 5 or not ok:
                print(f"    x{actual_reg}: store {expected_value} -> "
                      f"mem[{addr:#x}] -> read {final}  "
                      f"{'OK' if ok else 'FAIL'}")

        if all_ok:
            print(f"    ... all 20 round-trips OK")

        # Final stats
        subsection("Combined statistics")
        rf_stats = rf.stats
        mem_stats = mem.stats
        print(f"    Register file: {rf_stats['reads']} reads, "
              f"{rf_stats['writes']} writes")
        print(f"    SSD memory:    {mem_stats['reads']} reads, "
              f"{mem_stats['writes']} writes, "
              f"hit_rate={mem_stats['hit_rate']:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Neural Storage Demo -- register file + SSD memory"
    )
    parser.add_argument("--device", default="cpu",
                        help="Torch device (cpu, mps, cuda)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Load pre-trained register file instead of training")
    args = parser.parse_args()

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("  MPS not available, falling back to CPU")
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("  CUDA not available, falling back to CPU")
        device = "cpu"

    print()
    print("  nCPU Neural Storage Demo")
    print("  " + "=" * 50)
    print(f"  Device: {device}")
    print(f"  PyTorch: {torch.__version__}")

    t_start = time.perf_counter()

    # Part 1: Neural Register File
    reg_ok = demo_register_file(device=device, skip_training=args.skip_training)

    # Part 2: SSD-Backed Neural Memory
    mem_ok = demo_ssd_memory(device=device)

    # Part 3: Integration
    demo_integration(device=device)

    t_total = time.perf_counter() - t_start

    # ── Summary ──────────────────────────────────────────────────────
    section("SUMMARY")
    print(f"  Total demo time: {t_total:.1f}s")
    print(f"  Register file:   {'PASS' if reg_ok else 'FAIL'}")
    print(f"  SSD memory:      {'PASS' if mem_ok else 'FAIL'}")
    print()
    print("  Neural storage closes another gap in the nCPU architecture:")
    print("  instruction -> decode -> ALU -> REGISTERS -> MEMORY -> OS -> display")
    print("  All neural, all the way down.")
    print()


if __name__ == "__main__":
    main()
