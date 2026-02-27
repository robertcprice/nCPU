#!/usr/bin/env python3
"""Benchmark Out-of-Order Execution vs BBCache and TraceJIT"""

import kvrm_metal as metal

def assemble_simple_loop(iterations: int) -> bytes:
    """Simple tight loop: decrement counter until zero"""
    # X0 = counter (set by caller)
    # Loop: SUB X0, X0, #1; CMP X0, #0; B.NE loop; HLT
    code = bytearray()

    # SUB X0, X0, #1 (0xD1000400)
    code.extend((0x00, 0x04, 0x00, 0xD1))
    # CMP X0, #0 (SUBS XZR, X0, #0 = 0xF100001F)
    code.extend((0x1F, 0x00, 0x00, 0xF1))
    # B.NE -8 (back to SUB): imm19=-2, cond=1 -> 0x54FFFFC1
    code.extend((0xC1, 0xFF, 0xFF, 0x54))
    # HLT #0
    code.extend((0x00, 0x00, 0x40, 0xD4))

    return bytes(code)

def assemble_nested_loop(outer: int, inner: int) -> bytes:
    """Nested loop with more instructions for OoO opportunity"""
    code = bytearray()

    # X0 = outer counter, X1 = inner counter (reset each outer iteration)
    # X2 = accumulator, X3 = temp

    # outer_loop: (offset 0)
    #   MOV X1, #inner (MOVZ X1, #inner)
    movz = 0xD2800001 | ((inner & 0xFFFF) << 5)
    code.extend(movz.to_bytes(4, 'little'))  # offset 0

    # inner_loop: (offset 4)
    #   ADD X2, X2, X1        (independent - can run in parallel)
    code.extend((0x42, 0x00, 0x01, 0x8B))  # offset 4
    #   ADD X3, X3, #1        (independent - can run in parallel)
    code.extend((0x63, 0x04, 0x00, 0x91))  # offset 8
    #   SUB X1, X1, #1
    code.extend((0x21, 0x04, 0x00, 0xD1))  # offset 12
    #   CMP X1, #0
    code.extend((0x3F, 0x00, 0x00, 0xF1))  # offset 16
    #   B.NE inner_loop: target=4, current=20, offset=(4-20)/4=-4 -> imm19=0x7FFFC -> 0x54FFFF81
    code.extend((0x81, 0xFF, 0xFF, 0x54))  # offset 20

    # SUB X0, X0, #1
    code.extend((0x00, 0x04, 0x00, 0xD1))  # offset 24
    # CMP X0, #0
    code.extend((0x1F, 0x00, 0x00, 0xF1))  # offset 28
    # B.NE outer_loop: target=0, current=32, offset=(0-32)/4=-8 -> imm19=0x7FFF8 -> 0x54FFFF01
    code.extend((0x01, 0xFF, 0xFF, 0x54))  # offset 32

    # HLT
    code.extend((0x00, 0x00, 0x40, 0xD4))  # offset 36

    return bytes(code)

def assemble_parallel_adds() -> bytes:
    """Multiple independent ADDs to test OoO parallelism"""
    code = bytearray()

    # X0 = loop counter
    # Loop body has 4 independent ADDs that can execute in parallel

    # loop: (offset 0)
    #   ADD X1, X1, #1   (independent)
    code.extend((0x21, 0x04, 0x00, 0x91))  # offset 0
    #   ADD X2, X2, #2   (independent)
    code.extend((0x42, 0x08, 0x00, 0x91))  # offset 4
    #   ADD X3, X3, #3   (independent)
    code.extend((0x63, 0x0C, 0x00, 0x91))  # offset 8
    #   ADD X4, X4, #4   (independent)
    code.extend((0x84, 0x10, 0x00, 0x91))  # offset 12
    #   SUB X0, X0, #1
    code.extend((0x00, 0x04, 0x00, 0xD1))  # offset 16
    #   CMP X0, #0
    code.extend((0x1F, 0x00, 0x00, 0xF1))  # offset 20
    #   B.NE loop: target=0, current=24, offset=(0-24)/4=-6 -> imm19=0x7FFFA -> 0x54FFFF41
    code.extend((0x41, 0xFF, 0xFF, 0x54))  # offset 24
    #   HLT
    code.extend((0x00, 0x00, 0x40, 0xD4))  # offset 28

    return bytes(code)

def run_benchmark(name: str, cpu, program: bytes, iterations: int):
    """Run a benchmark and return results"""
    cpu.reset()
    cpu.load_program(list(program), 0)
    cpu.set_pc(0)
    cpu.set_register(0, iterations)

    result = cpu.execute(max_batches=100, timeout_seconds=30.0)
    return result

def main():
    print("=" * 70)
    print("Out-of-Order Execution Benchmark")
    print("=" * 70)

    # Initialize CPUs
    print("\nInitializing execution engines...")

    try:
        bb_cpu = metal.BBCacheMetalCPU(memory_size=4*1024*1024, cycles_per_batch=10_000_000)
    except Exception as e:
        print(f"BBCache init failed: {e}")
        bb_cpu = None

    try:
        trace_cpu = metal.TraceJITMetalCPU(memory_size=4*1024*1024, cycles_per_batch=10_000_000)
    except Exception as e:
        print(f"TraceJIT init failed: {e}")
        trace_cpu = None

    try:
        ooo_cpu = metal.OoOMetalCPU(memory_size=4*1024*1024, cycles_per_batch=10_000_000)
    except Exception as e:
        print(f"OoO init failed: {e}")
        ooo_cpu = None

    if not any([bb_cpu, trace_cpu, ooo_cpu]):
        print("No CPUs initialized!")
        return

    # Test 1: Simple tight loop (100K iterations)
    print("\n" + "-" * 70)
    print("TEST 1: Simple Tight Loop (100K iterations)")
    print("-" * 70)

    simple_prog = assemble_simple_loop(100_000)

    if bb_cpu:
        bb_result = run_benchmark("BBCache", bb_cpu, simple_prog, 100_000)
        bb_ips = bb_result.ips
        print(f"  BBCache   : {bb_result.total_cycles:>10,} cyc | {bb_result.ips:>12,.0f} IPS | Cache: {bb_result.cache_hit_rate:.1f}%")
    else:
        bb_ips = 0

    if trace_cpu:
        trace_result = run_benchmark("TraceJIT", trace_cpu, simple_prog, 100_000)
        trace_ips = trace_result.ips
        print(f"  TraceJIT  : {trace_result.total_cycles:>10,} cyc | {trace_result.ips:>12,.0f} IPS | Cache: {trace_result.cache_hit_rate:.1f}% | Hot: {trace_result.hot_executions}")
    else:
        trace_ips = 0

    if ooo_cpu:
        ooo_result = run_benchmark("OoO", ooo_cpu, simple_prog, 100_000)
        ooo_ips = ooo_result.ips
        parallel_ratio = ooo_result.parallelism_ratio * 100
        print(f"  OoO       : {ooo_result.total_cycles:>10,} cyc | {ooo_result.ips:>12,.0f} IPS | Parallel: {parallel_ratio:.1f}% | P:{ooo_result.parallel_executions} S:{ooo_result.serial_executions}")
    else:
        ooo_ips = 0

    # Test 2: Nested loop (outer=1000, inner=100)
    print("\n" + "-" * 70)
    print("TEST 2: Nested Loop (1000 x 100 iterations)")
    print("-" * 70)

    nested_prog = assemble_nested_loop(1000, 100)

    if bb_cpu:
        bb_result = run_benchmark("BBCache", bb_cpu, nested_prog, 1000)
        print(f"  BBCache   : {bb_result.total_cycles:>10,} cyc | {bb_result.ips:>12,.0f} IPS | Cache: {bb_result.cache_hit_rate:.1f}%")

    if trace_cpu:
        trace_result = run_benchmark("TraceJIT", trace_cpu, nested_prog, 1000)
        print(f"  TraceJIT  : {trace_result.total_cycles:>10,} cyc | {trace_result.ips:>12,.0f} IPS | Cache: {trace_result.cache_hit_rate:.1f}% | Hot: {trace_result.hot_executions}")

    if ooo_cpu:
        ooo_result = run_benchmark("OoO", ooo_cpu, nested_prog, 1000)
        parallel_ratio = ooo_result.parallelism_ratio * 100
        print(f"  OoO       : {ooo_result.total_cycles:>10,} cyc | {ooo_result.ips:>12,.0f} IPS | Parallel: {parallel_ratio:.1f}% | P:{ooo_result.parallel_executions} S:{ooo_result.serial_executions}")

    # Test 3: Parallel ADDs (50K iterations) - OoO should shine here
    print("\n" + "-" * 70)
    print("TEST 3: Parallel Independent ADDs (50K iterations)")
    print("       (4 independent ADDs per iteration - OoO opportunity)")
    print("-" * 70)

    parallel_prog = assemble_parallel_adds()

    if bb_cpu:
        bb_result = run_benchmark("BBCache", bb_cpu, parallel_prog, 50_000)
        print(f"  BBCache   : {bb_result.total_cycles:>10,} cyc | {bb_result.ips:>12,.0f} IPS | Cache: {bb_result.cache_hit_rate:.1f}%")

    if trace_cpu:
        trace_result = run_benchmark("TraceJIT", trace_cpu, parallel_prog, 50_000)
        print(f"  TraceJIT  : {trace_result.total_cycles:>10,} cyc | {trace_result.ips:>12,.0f} IPS | Cache: {trace_result.cache_hit_rate:.1f}% | Hot: {trace_result.hot_executions}")

    if ooo_cpu:
        ooo_result = run_benchmark("OoO", ooo_cpu, parallel_prog, 50_000)
        parallel_ratio = ooo_result.parallelism_ratio * 100
        print(f"  OoO       : {ooo_result.total_cycles:>10,} cyc | {ooo_result.ips:>12,.0f} IPS | Parallel: {parallel_ratio:.1f}% | P:{ooo_result.parallel_executions} S:{ooo_result.serial_executions}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if ooo_cpu and bb_cpu:
        print(f"OoO parallelism achieved in tests above")
        print(f"Higher Parallel% = more independent instructions executed together")

    print("\nDone!")

if __name__ == "__main__":
    main()
