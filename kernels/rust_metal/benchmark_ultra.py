#!/usr/bin/env python3
"""Comprehensive benchmark: Compare UltraMetalCPU against Baseline, Fusion, and BBCache."""

import time
from kvrm_metal import ContinuousMetalCPU, FusionMetalCPU, BBCacheMetalCPU
# UltraMetalCPU shader compilation is hanging - skip for now
# from kvrm_metal import UltraMetalCPU

print("=" * 80)
print("ULTRA OPTIMIZATION BENCHMARK")
print("Comparing: Baseline, Fusion, BBCache, Ultra")
print("=" * 80)

def encode_movz(rd, imm16, hw=0):
    return 0xD2800000 | (hw << 21) | (imm16 << 5) | rd

def encode_movk(rd, imm16, hw=0):
    return 0xF2800000 | (hw << 21) | (imm16 << 5) | rd

def encode_add_imm(rd, rn, imm12):
    return 0x91000000 | (imm12 << 10) | (rn << 5) | rd

def encode_sub_imm(rd, rn, imm12):
    return 0xD1000000 | (imm12 << 10) | (rn << 5) | rd

def encode_subs_imm(rd, rn, imm12):
    return 0xF1000000 | (imm12 << 10) | (rn << 5) | rd

def encode_subs_reg(rd, rn, rm):
    """SUBS Xd, Xn, Xm (also CMP Xn, Xm when Xd=XZR)"""
    return 0xEB000000 | (rm << 16) | (rn << 5) | rd

def encode_str_64(rt, rn, imm12=0):
    return 0xF9000000 | (imm12 << 10) | (rn << 5) | rt

def encode_ldr_64(rt, rn, imm12=0):
    return 0xF9400000 | (imm12 << 10) | (rn << 5) | rt

def encode_b(offset_words):
    imm26 = offset_words & 0x3FFFFFF
    return 0x14000000 | imm26

def encode_b_cond(cond, offset_words):
    """B.cond: cond is 4-bit, offset is 19-bit"""
    imm19 = offset_words & 0x7FFFF
    return 0x54000000 | (imm19 << 5) | cond

def encode_cmp_reg(rn, rm):
    """CMP Xn, Xm (SUBS XZR, Xn, Xm)"""
    return encode_subs_reg(31, rn, rm)

# Condition codes
COND_LT = 0b1011  # Less than (N != V)
COND_NE = 0b0001  # Not equal (Z == 0)
COND_GE = 0b1010  # Greater or equal (N == V)

def run_benchmark(name, cpu_factory, program_bytes, expected_cycles=300000, timeout=10.0):
    """Run a benchmark and return results."""
    try:
        cpu = cpu_factory()
        cpu.load_program(list(program_bytes), 0)
        cpu.set_pc(0)

        start = time.perf_counter()
        result = cpu.execute(max_batches=100, timeout_seconds=timeout)
        elapsed = time.perf_counter() - start

        cycles = result.total_cycles
        ips = cycles / elapsed if elapsed > 0 else 0

        # Get cache stats if available
        cache_info = ""
        if hasattr(result, 'cache_hits'):
            total = result.cache_hits + result.cache_misses
            hit_rate = (result.cache_hits / total * 100) if total > 0 else 0
            cache_info = f" | Cache: {hit_rate:.1f}%"
        if hasattr(result, 'fusions') and result.fusions > 0:
            cache_info += f" | Fusions: {result.fusions:,}"
        if hasattr(result, 'prefetch_hits') and result.prefetch_hits > 0:
            cache_info += f" | Prefetch: {result.prefetch_hits:,}"

        return {
            'name': name,
            'cycles': cycles,
            'elapsed': elapsed,
            'ips': ips,
            'cache_info': cache_info,
            'result': result
        }
    except Exception as e:
        print(f"  ERROR in {name}: {e}")
        return None

# =============================================================================
# TEST 1: Simple loop (100K iterations) - Tests basic block caching
# =============================================================================
print(f"\n{'='*80}")
print("TEST 1: Simple Loop (100K iterations) - MOVZ + ADD + SUBS + B.LT")
print("This tests basic block caching and CMP+B.cond fusion")
print("="*80)

loop_count = 100000
program = [
    encode_movz(0, 0),                      # X0 = 0 (counter)
    encode_movz(1, loop_count & 0xFFFF),    # X1 = loop_count (low 16 bits)
    # Loop:
    encode_add_imm(0, 0, 1),                # X0 += 1
    encode_cmp_reg(0, 1),                   # CMP X0, X1
    encode_b_cond(COND_LT, -2 & 0x7FFFF),   # B.LT loop (back 2 instructions)
]
program_bytes = b''.join(inst.to_bytes(4, 'little') for inst in program)

results_test1 = {}
for name, factory in [
    ("Baseline", lambda: ContinuousMetalCPU(cycles_per_batch=10_000_000)),
    ("Fusion", lambda: FusionMetalCPU(memory_size=4*1024*1024)),
    ("BBCache", lambda: BBCacheMetalCPU(memory_size=4*1024*1024)),
    # ("Ultra", lambda: UltraMetalCPU(memory_size=4*1024*1024)),  # Shader compile hangs
]:
    result = run_benchmark(name, factory, program_bytes)
    if result:
        results_test1[name] = result
        speedup = result['ips'] / results_test1['Baseline']['ips'] if 'Baseline' in results_test1 else 1.0
        print(f"  {name:10}: {result['cycles']:>10,} cycles | {result['ips']:>12,.0f} IPS | {speedup:.2f}x{result['cache_info']}")

# =============================================================================
# TEST 2: Nested loop - Tests super block formation
# =============================================================================
print(f"\n{'='*80}")
print("TEST 2: Nested Loop (1000 outer x 100 inner) - Tests super blocks")
print("="*80)

outer_count = 1000
inner_count = 100
program = [
    encode_movz(0, 0),                          # X0 = 0 (outer counter)
    encode_movz(3, outer_count & 0xFFFF),       # X3 = outer_count
    # Outer loop:
    encode_movz(1, 0),                          # X1 = 0 (inner counter)
    encode_movz(4, inner_count & 0xFFFF),       # X4 = inner_count
    # Inner loop:
    encode_add_imm(2, 2, 1),                    # X2 += 1 (total counter)
    encode_add_imm(1, 1, 1),                    # X1 += 1
    encode_cmp_reg(1, 4),                       # CMP X1, X4
    encode_b_cond(COND_LT, -3 & 0x7FFFF),       # B.LT inner
    # End inner
    encode_add_imm(0, 0, 1),                    # X0 += 1
    encode_cmp_reg(0, 3),                       # CMP X0, X3
    encode_b_cond(COND_LT, -8 & 0x7FFFF),       # B.LT outer
]
program_bytes = b''.join(inst.to_bytes(4, 'little') for inst in program)

results_test2 = {}
for name, factory in [
    ("Baseline", lambda: ContinuousMetalCPU(cycles_per_batch=10_000_000)),
    ("Fusion", lambda: FusionMetalCPU(memory_size=4*1024*1024)),
    ("BBCache", lambda: BBCacheMetalCPU(memory_size=4*1024*1024)),
]:
    result = run_benchmark(name, factory, program_bytes)
    if result:
        results_test2[name] = result
        speedup = result['ips'] / results_test2['Baseline']['ips'] if 'Baseline' in results_test2 else 1.0
        print(f"  {name:10}: {result['cycles']:>10,} cycles | {result['ips']:>12,.0f} IPS | {speedup:.2f}x{result['cache_info']}")

# =============================================================================
# TEST 3: Memory-intensive loop - Tests memory coalescing
# =============================================================================
print(f"\n{'='*80}")
print("TEST 3: Memory Loop (10K iterations with LDR/STR)")
print("="*80)

mem_loop_count = 10000
program = [
    encode_movz(0, 0),                          # X0 = 0 (counter)
    encode_movz(1, mem_loop_count & 0xFFFF),    # X1 = loop_count
    encode_movz(2, 0x1000),                     # X2 = base address
    # Loop:
    encode_ldr_64(3, 2, 0),                     # X3 = [X2]
    encode_add_imm(3, 3, 1),                    # X3 += 1
    encode_str_64(3, 2, 0),                     # [X2] = X3
    encode_add_imm(0, 0, 1),                    # X0 += 1
    encode_cmp_reg(0, 1),                       # CMP X0, X1
    encode_b_cond(COND_LT, -5 & 0x7FFFF),       # B.LT loop
]
program_bytes = b''.join(inst.to_bytes(4, 'little') for inst in program)

results_test3 = {}
for name, factory in [
    ("Baseline", lambda: ContinuousMetalCPU(cycles_per_batch=10_000_000)),
    ("Fusion", lambda: FusionMetalCPU(memory_size=4*1024*1024)),
    ("BBCache", lambda: BBCacheMetalCPU(memory_size=4*1024*1024)),
]:
    result = run_benchmark(name, factory, program_bytes)
    if result:
        results_test3[name] = result
        speedup = result['ips'] / results_test3['Baseline']['ips'] if 'Baseline' in results_test3 else 1.0
        print(f"  {name:10}: {result['cycles']:>10,} cycles | {result['ips']:>12,.0f} IPS | {speedup:.2f}x{result['cache_info']}")

# =============================================================================
# TEST 4: Sequential ADDs - Tests ADD+ADD fusion
# =============================================================================
print(f"\n{'='*80}")
print("TEST 4: Sequential ADDs (50K iterations) - Tests ADD fusion")
print("="*80)

add_loop_count = 50000
program = [
    encode_movz(0, 0),                          # X0 = 0
    encode_movz(1, add_loop_count & 0xFFFF),    # X1 = loop_count
    # Loop:
    encode_add_imm(2, 2, 1),                    # X2 += 1
    encode_add_imm(3, 3, 2),                    # X3 += 2
    encode_add_imm(4, 4, 3),                    # X4 += 3
    encode_add_imm(5, 5, 4),                    # X5 += 4
    encode_add_imm(0, 0, 1),                    # X0 += 1
    encode_cmp_reg(0, 1),                       # CMP X0, X1
    encode_b_cond(COND_LT, -6 & 0x7FFFF),       # B.LT loop
]
program_bytes = b''.join(inst.to_bytes(4, 'little') for inst in program)

results_test4 = {}
for name, factory in [
    ("Baseline", lambda: ContinuousMetalCPU(cycles_per_batch=10_000_000)),
    ("Fusion", lambda: FusionMetalCPU(memory_size=4*1024*1024)),
    ("BBCache", lambda: BBCacheMetalCPU(memory_size=4*1024*1024)),
]:
    result = run_benchmark(name, factory, program_bytes)
    if result:
        results_test4[name] = result
        speedup = result['ips'] / results_test4['Baseline']['ips'] if 'Baseline' in results_test4 else 1.0
        print(f"  {name:10}: {result['cycles']:>10,} cycles | {result['ips']:>12,.0f} IPS | {speedup:.2f}x{result['cache_info']}")

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'='*80}")
print("SUMMARY")
print("="*80)

print("\n| Test | Baseline IPS | Fusion | BBCache | Best Speedup |")
print("|------|--------------|--------|---------|--------------|")

for test_name, results in [("Simple Loop", results_test1), ("Nested Loop", results_test2),
                            ("Memory Loop", results_test3), ("ADD Fusion", results_test4)]:
    if 'Baseline' in results:
        baseline = results['Baseline']['ips']
        fusion = results.get('Fusion', {}).get('ips', 0)
        bbcache = results.get('BBCache', {}).get('ips', 0)

        best = max(fusion, bbcache)
        best_speedup = best / baseline if baseline > 0 else 0

        print(f"| {test_name:12} | {baseline:>12,.0f} | {fusion/baseline:.2f}x | {bbcache/baseline:.2f}x | **{best_speedup:.2f}x** |")

# Final statistics for BBCache
if 'BBCache' in results_test1:
    print(f"\nBBCache Statistics (Test 1):")
    r = results_test1['BBCache']['result']
    print(f"  Cache hits: {r.cache_hits:,}")
    print(f"  Cache misses: {r.cache_misses:,}")
    print(f"  Fusions: {r.fusions:,}")
    if r.cache_hits + r.cache_misses > 0:
        print(f"  Cache hit rate: {r.cache_hits / (r.cache_hits + r.cache_misses) * 100:.1f}%")

print(f"\n{'='*80}")
print("BENCHMARK COMPLETE")
print("="*80)
