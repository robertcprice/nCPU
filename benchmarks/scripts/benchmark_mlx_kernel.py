#!/usr/bin/env python3
"""
Benchmark: MLX Metal Kernel vs PyTorch Tensor-Native CPU Execution.

This benchmark compares the performance of:
1. MLX Metal Kernel (new custom GPU kernel)
2. PyTorch Tensor-Native Batch Execution (existing implementation)
3. PyTorch Single-Step Execution (baseline)

The goal is to demonstrate the performance improvement from eliminating
GPU-CPU synchronization overhead by using a custom Metal kernel.

EXPECTED RESULTS:
=================

| Mode                    | Expected IPS      | vs Baseline |
|-------------------------|-------------------|-------------|
| PyTorch Single-Step     | ~50 IPS           | 1x          |
| PyTorch Batch-512       | ~120,000 IPS      | ~2,500x     |
| MLX Metal Kernel        | 1,000,000+ IPS    | ~20,000x+   |

The MLX kernel eliminates the per-batch `.item()` sync call that limits
PyTorch batch execution to ~120K IPS.

Author: KVRM Project
Date: 2024
"""

import time
import sys
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Test configurations
SMALL_TEST = 10_000        # Quick test
MEDIUM_TEST = 100_000      # Standard benchmark
LARGE_TEST = 1_000_000     # Performance test
STRESS_TEST = 10_000_000   # Stress test

# Default test size
DEFAULT_TEST_SIZE = MEDIUM_TEST


# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION ENCODERS
# ═══════════════════════════════════════════════════════════════════════════════

def encode_movz(rd: int, imm16: int, hw: int = 0) -> int:
    """Encode MOVZ instruction."""
    return (1 << 31) | (0b10100101 << 23) | (hw << 21) | ((imm16 & 0xFFFF) << 5) | rd


def encode_add_imm(rd: int, rn: int, imm12: int) -> int:
    """Encode ADD immediate instruction."""
    return (1 << 31) | (0b10001 << 24) | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd


def encode_sub_imm(rd: int, rn: int, imm12: int) -> int:
    """Encode SUB immediate instruction."""
    return (1 << 31) | (0b11010001 << 24) | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd


def encode_subs_imm(rd: int, rn: int, imm12: int) -> int:
    """Encode SUBS immediate instruction (sets flags)."""
    return (1 << 31) | (0b11110001 << 24) | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd


def encode_cbnz(rt: int, offset: int) -> int:
    """Encode CBNZ instruction."""
    imm19 = (offset // 4) & 0x7FFFF
    return (1 << 31) | 0x35000000 | (imm19 << 5) | rt


def encode_hlt() -> int:
    """Encode HLT instruction."""
    return 0xD4400000


# ═══════════════════════════════════════════════════════════════════════════════
# TEST PROGRAMS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_straight_line_program(num_instructions: int) -> list:
    """Generate straight-line ADD instructions (no branches)."""
    program = [encode_movz(0, 0)]  # X0 = 0

    for _ in range(num_instructions - 2):
        program.append(encode_add_imm(0, 0, 1))  # X0++

    program.append(encode_hlt())
    return program


def generate_loop_program(iterations: int) -> list:
    """Generate a counting loop program."""
    # Calculate high and low parts of iteration count
    low = iterations & 0xFFFF
    high = (iterations >> 16) & 0xFFFF

    program = [
        encode_movz(0, 0),              # X0 = 0 (counter)
        encode_movz(1, low),            # X1 = low part
    ]

    if high > 0:
        program.append((1 << 31) | (0b11100101 << 23) | (1 << 21) | ((high & 0xFFFF) << 5) | 1)  # MOVK X1, #high, LSL #16

    program.extend([
        # loop:
        encode_add_imm(0, 0, 1),         # X0++
        encode_subs_imm(2, 0, 0),        # Compare (dummy for flags)
        encode_sub_imm(2, 1, 0),         # X2 = X1 - X0
    ])

    # Add conditional branch back
    # CBNZ X2, -12 (back 3 instructions = -12 bytes)
    program.append(encode_cbnz(2, -12))

    program.append(encode_hlt())
    return program


# ═══════════════════════════════════════════════════════════════════════════════
# MLX KERNEL BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_mlx_kernel(program: list, name: str, expected_x0: int = None) -> dict:
    """Benchmark MLX Metal kernel execution."""
    try:
        from mlx_kernel import MLXKernelCPU, StopReason
    except ImportError as e:
        print(f"  [SKIP] MLX kernel not available: {e}")
        return None

    cpu = MLXKernelCPU()
    cpu.load_program(program, address=0)
    cpu.set_pc(0)

    # Warmup
    cpu.reset()
    cpu.load_program(program, address=0)
    cpu.set_pc(0)
    cpu.execute(max_cycles=1000)

    # Actual benchmark
    cpu.reset()
    cpu.load_program(program, address=0)
    cpu.set_pc(0)

    start = time.perf_counter()
    result = cpu.execute(max_cycles=20_000_000)
    elapsed = time.perf_counter() - start

    ips = result.cycles / elapsed if elapsed > 0 else 0

    # Verify correctness
    x0 = cpu.get_register(0)
    correct = expected_x0 is None or x0 == expected_x0

    return {
        'name': f"MLX Kernel ({name})",
        'cycles': result.cycles,
        'elapsed': elapsed,
        'ips': ips,
        'stop_reason': result.stop_reason.name_str,
        'x0': x0,
        'correct': correct,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PYTORCH BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_pytorch_batch(program: list, name: str, batch_size: int = 512, expected_x0: int = None) -> dict:
    """Benchmark PyTorch tensor-native batch execution."""
    try:
        sys.path.insert(0, '.')
        from neural_cpu_tensor_native import TensorNativeCPU
    except ImportError as e:
        print(f"  [SKIP] PyTorch tensor-native not available: {e}")
        return None

    cpu = TensorNativeCPU()

    # Load program
    program_bytes = b''
    for inst in program:
        program_bytes += inst.to_bytes(4, 'little')

    cpu.memory[:len(program_bytes)] = __import__('torch').tensor(
        list(program_bytes), dtype=__import__('torch').uint8, device=cpu.device
    )

    # Warmup
    cpu.pc = __import__('torch').tensor(0, dtype=__import__('torch').int64, device=cpu.device)
    cpu.regs = __import__('torch').zeros(32, dtype=__import__('torch').int64, device=cpu.device)
    cpu.run_zero_sync(max_instructions=1000, batch_size=batch_size)

    # Actual benchmark
    cpu.pc = __import__('torch').tensor(0, dtype=__import__('torch').int64, device=cpu.device)
    cpu.regs = __import__('torch').zeros(32, dtype=__import__('torch').int64, device=cpu.device)
    cpu.halted = False

    start = time.perf_counter()
    result = cpu.run_zero_sync(max_instructions=20_000_000, batch_size=batch_size)
    elapsed = time.perf_counter() - start

    x0 = int(cpu.regs[0].item())
    correct = expected_x0 is None or x0 == expected_x0

    return {
        'name': f"PyTorch Batch-{batch_size} ({name})",
        'cycles': result.instructions_executed,
        'elapsed': elapsed,
        'ips': result.ips,
        'stop_reason': 'HALT' if cpu.halted else 'MAX_CYCLES',
        'x0': x0,
        'correct': correct,
    }


def benchmark_pytorch_single(program: list, name: str, max_cycles: int = 10000, expected_x0: int = None) -> dict:
    """Benchmark PyTorch single-step execution (baseline)."""
    try:
        sys.path.insert(0, '.')
        from neural_cpu_tensor_native import TensorNativeCPU
    except ImportError as e:
        print(f"  [SKIP] PyTorch single-step not available: {e}")
        return None

    cpu = TensorNativeCPU()

    # Load program
    program_bytes = b''
    for inst in program:
        program_bytes += inst.to_bytes(4, 'little')

    cpu.memory[:len(program_bytes)] = __import__('torch').tensor(
        list(program_bytes), dtype=__import__('torch').uint8, device=cpu.device
    )

    cpu.pc = __import__('torch').tensor(0, dtype=__import__('torch').int64, device=cpu.device)
    cpu.regs = __import__('torch').zeros(32, dtype=__import__('torch').int64, device=cpu.device)

    start = time.perf_counter()
    result = cpu.run(max_instructions=max_cycles)
    elapsed = time.perf_counter() - start

    x0 = int(cpu.regs[0].item())
    correct = expected_x0 is None or x0 == expected_x0

    return {
        'name': f"PyTorch Single-Step ({name})",
        'cycles': result.instructions_executed,
        'elapsed': elapsed,
        'ips': result.ips,
        'stop_reason': 'HALT' if cpu.halted else 'MAX_CYCLES',
        'x0': x0,
        'correct': correct,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def format_result(result: dict) -> str:
    """Format a benchmark result for printing."""
    if result is None:
        return "  [SKIPPED]"

    status = "PASS" if result['correct'] else "FAIL"
    return (
        f"  {result['name']:40} | "
        f"{result['cycles']:>12,} cycles | "
        f"{result['elapsed']*1000:>8.2f}ms | "
        f"{result['ips']:>15,.0f} IPS | "
        f"X0={result['x0']:>10} | "
        f"[{status}]"
    )


def run_benchmark_suite():
    """Run complete benchmark suite."""
    print("=" * 100)
    print("MLX METAL KERNEL vs PYTORCH CPU BENCHMARK")
    print("=" * 100)

    results = []

    # ───────────────────────────────────────────────────────────────────────────
    # Test 1: Straight-line code (ideal for batching)
    # ───────────────────────────────────────────────────────────────────────────
    print("\n[1] STRAIGHT-LINE CODE (10,000 ADDs)")
    print("-" * 100)

    program = generate_straight_line_program(10000)
    expected = 9998  # 10000 - movz - hlt

    result = benchmark_mlx_kernel(program, "straight-line", expected)
    if result:
        print(format_result(result))
        results.append(result)

    result = benchmark_pytorch_batch(program, "straight-line", batch_size=512, expected_x0=expected)
    if result:
        print(format_result(result))
        results.append(result)

    # ───────────────────────────────────────────────────────────────────────────
    # Test 2: Loop with 10,000 iterations
    # ───────────────────────────────────────────────────────────────────────────
    print("\n[2] LOOP (10,000 iterations)")
    print("-" * 100)

    program = generate_loop_program(10000)
    expected = 10000

    result = benchmark_mlx_kernel(program, "loop-10K", expected)
    if result:
        print(format_result(result))
        results.append(result)

    result = benchmark_pytorch_batch(program, "loop-10K", batch_size=512, expected_x0=expected)
    if result:
        print(format_result(result))
        results.append(result)

    # ───────────────────────────────────────────────────────────────────────────
    # Test 3: Loop with 100,000 iterations
    # ───────────────────────────────────────────────────────────────────────────
    print("\n[3] LOOP (100,000 iterations)")
    print("-" * 100)

    program = generate_loop_program(100000)
    expected = 100000

    result = benchmark_mlx_kernel(program, "loop-100K", expected)
    if result:
        print(format_result(result))
        results.append(result)

    result = benchmark_pytorch_batch(program, "loop-100K", batch_size=512, expected_x0=expected)
    if result:
        print(format_result(result))
        results.append(result)

    # ───────────────────────────────────────────────────────────────────────────
    # Test 4: Loop with 1,000,000 iterations (stress test)
    # ───────────────────────────────────────────────────────────────────────────
    print("\n[4] LOOP (1,000,000 iterations - stress test)")
    print("-" * 100)

    program = generate_loop_program(1000000)
    expected = 1000000

    result = benchmark_mlx_kernel(program, "loop-1M", expected)
    if result:
        print(format_result(result))
        results.append(result)

    result = benchmark_pytorch_batch(program, "loop-1M", batch_size=512, expected_x0=expected)
    if result:
        print(format_result(result))
        results.append(result)

    # ───────────────────────────────────────────────────────────────────────────
    # Test 5: Single-step baseline (limited iterations)
    # ───────────────────────────────────────────────────────────────────────────
    print("\n[5] SINGLE-STEP BASELINE (1,000 iterations)")
    print("-" * 100)

    program = generate_loop_program(1000)
    expected = 1000

    result = benchmark_pytorch_single(program, "loop-1K", max_cycles=50000, expected_x0=expected)
    if result:
        print(format_result(result))
        results.append(result)

    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("PERFORMANCE SUMMARY")
    print("=" * 100)

    # Group by test type
    mlx_results = [r for r in results if 'MLX' in r['name']]
    pytorch_batch_results = [r for r in results if 'Batch' in r['name']]
    pytorch_single_results = [r for r in results if 'Single' in r['name']]

    if mlx_results:
        mlx_avg_ips = sum(r['ips'] for r in mlx_results) / len(mlx_results)
        print(f"\nMLX Metal Kernel:")
        print(f"  Average IPS: {mlx_avg_ips:,.0f}")

    if pytorch_batch_results:
        pytorch_avg_ips = sum(r['ips'] for r in pytorch_batch_results) / len(pytorch_batch_results)
        print(f"\nPyTorch Batch-512:")
        print(f"  Average IPS: {pytorch_avg_ips:,.0f}")

        if mlx_results:
            speedup = mlx_avg_ips / pytorch_avg_ips
            print(f"\n>>> MLX is {speedup:.1f}x faster than PyTorch Batch <<<")

    if pytorch_single_results:
        single_ips = pytorch_single_results[0]['ips']
        print(f"\nPyTorch Single-Step:")
        print(f"  IPS: {single_ips:,.0f}")

        if mlx_results:
            speedup_vs_single = mlx_avg_ips / single_ips
            print(f"\n>>> MLX is {speedup_vs_single:,.0f}x faster than Single-Step <<<")

    # Final verdict
    print("\n" + "=" * 100)
    all_correct = all(r['correct'] for r in results)
    if all_correct:
        print("ALL TESTS PASSED - Results verified correct")
    else:
        print("SOME TESTS FAILED - Check X0 values")
    print("=" * 100)

    return all_correct


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    success = run_benchmark_suite()
    sys.exit(0 if success else 1)
