#!/usr/bin/env python3
"""Comprehensive benchmark: Compare Baseline, Fusion, and BBCache implementations."""

import time
from kvrm_metal import ContinuousMetalCPU, FusionMetalCPU, BBCacheMetalCPU

print("=" * 80)
print("OPTIMIZATION BENCHMARK")
print("Comparing: Baseline (Continuous), Fusion, BBCache")
print("=" * 80)

def encode_movz(rd, imm16, hw=0):
    return 0xD2800000 | (hw << 21) | (imm16 << 5) | rd

def encode_add_imm(rd, rn, imm12):
    return 0x91000000 | (imm12 << 10) | (rn << 5) | rd

def encode_sub_imm(rd, rn, imm12):
    return 0xD1000000 | (imm12 << 10) | (rn << 5) | rd

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
    imm19 = offset_words & 0x7FFFF
    return 0x54000000 | (imm19 << 5) | cond

def encode_cmp_reg(rn, rm):
    return encode_subs_reg(31, rn, rm)

COND_LT = 0b1011
COND_NE = 0b0001

def run_continuous(program_bytes, timeout=10.0):
    """Run with ContinuousMetalCPU (baseline)."""
    cpu = ContinuousMetalCPU(cycles_per_batch=10_000_000)
    cpu.load_program(list(program_bytes), 0)
    cpu.set_pc(0)
    start = time.perf_counter()
    result = cpu.execute_continuous(max_batches=100, timeout_seconds=timeout)
    elapsed = time.perf_counter() - start
    return result.total_cycles, elapsed, result.total_cycles / elapsed if elapsed > 0 else 0

def run_fusion(program_bytes, timeout=10.0):
    """Run with FusionMetalCPU."""
    cpu = FusionMetalCPU(memory_size=4*1024*1024)
    cpu.load_program(list(program_bytes), 0)
    cpu.set_pc(0)
    start = time.perf_counter()
    result = cpu.execute_fusion(max_batches=100, timeout_seconds=timeout)
    elapsed = time.perf_counter() - start
    return result.total_cycles, elapsed, result.total_cycles / elapsed if elapsed > 0 else 0

def run_bbcache(program_bytes, timeout=10.0):
    """Run with BBCacheMetalCPU."""
    cpu = BBCacheMetalCPU(memory_size=4*1024*1024)
    cpu.load_program(list(program_bytes), 0)
    cpu.set_pc(0)
    start = time.perf_counter()
    result = cpu.execute(max_batches=100, timeout_seconds=timeout)
    elapsed = time.perf_counter() - start
    cache_hit_rate = (result.cache_hits / (result.cache_hits + result.cache_misses) * 100) if (result.cache_hits + result.cache_misses) > 0 else 0
    return result.total_cycles, elapsed, result.total_cycles / elapsed if elapsed > 0 else 0, cache_hit_rate, result

# =============================================================================
# TEST 1: Simple loop (100K iterations)
# =============================================================================
print(f"\n{'='*80}")
print("TEST 1: Simple Loop (100K iterations) - MOVZ + ADD + CMP + B.LT")
print("="*80)

loop_count = 100000
program = [
    encode_movz(0, 0),                      # X0 = 0 (counter)
    encode_movz(1, loop_count & 0xFFFF),    # X1 = loop_count
    # Loop:
    encode_add_imm(0, 0, 1),                # X0 += 1
    encode_cmp_reg(0, 1),                   # CMP X0, X1
    encode_b_cond(COND_LT, -2 & 0x7FFFF),   # B.LT loop
]
program_bytes = b''.join(inst.to_bytes(4, 'little') for inst in program)

print("Running Baseline (Continuous)...")
baseline_cycles, baseline_elapsed, baseline_ips = run_continuous(program_bytes)
print(f"  Baseline:  {baseline_cycles:>10,} cycles | {baseline_ips:>12,.0f} IPS")

print("Running Fusion...")
fusion_cycles, fusion_elapsed, fusion_ips = run_fusion(program_bytes)
print(f"  Fusion:    {fusion_cycles:>10,} cycles | {fusion_ips:>12,.0f} IPS | {fusion_ips/baseline_ips:.2f}x")

print("Running BBCache...")
bbcache_cycles, bbcache_elapsed, bbcache_ips, bbcache_hit_rate, bbcache_result = run_bbcache(program_bytes)
print(f"  BBCache:   {bbcache_cycles:>10,} cycles | {bbcache_ips:>12,.0f} IPS | {bbcache_ips/baseline_ips:.2f}x | Cache: {bbcache_hit_rate:.1f}%")

results_test1 = {'Baseline': baseline_ips, 'Fusion': fusion_ips, 'BBCache': bbcache_ips}

# =============================================================================
# TEST 2: Nested loop
# =============================================================================
print(f"\n{'='*80}")
print("TEST 2: Nested Loop (1000 outer x 100 inner)")
print("="*80)

outer_count = 1000
inner_count = 100
program = [
    encode_movz(0, 0),                          # X0 = outer counter
    encode_movz(3, outer_count & 0xFFFF),       # X3 = outer_count
    # Outer loop:
    encode_movz(1, 0),                          # X1 = inner counter
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

baseline_cycles, baseline_elapsed, baseline_ips = run_continuous(program_bytes)
print(f"  Baseline:  {baseline_cycles:>10,} cycles | {baseline_ips:>12,.0f} IPS")

fusion_cycles, fusion_elapsed, fusion_ips = run_fusion(program_bytes)
print(f"  Fusion:    {fusion_cycles:>10,} cycles | {fusion_ips:>12,.0f} IPS | {fusion_ips/baseline_ips:.2f}x")

bbcache_cycles, bbcache_elapsed, bbcache_ips, bbcache_hit_rate, _ = run_bbcache(program_bytes)
print(f"  BBCache:   {bbcache_cycles:>10,} cycles | {bbcache_ips:>12,.0f} IPS | {bbcache_ips/baseline_ips:.2f}x | Cache: {bbcache_hit_rate:.1f}%")

results_test2 = {'Baseline': baseline_ips, 'Fusion': fusion_ips, 'BBCache': bbcache_ips}

# =============================================================================
# TEST 3: Memory-intensive loop
# =============================================================================
print(f"\n{'='*80}")
print("TEST 3: Memory Loop (10K iterations with LDR/STR)")
print("="*80)

mem_loop_count = 10000
program = [
    encode_movz(0, 0),                          # X0 = counter
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

baseline_cycles, baseline_elapsed, baseline_ips = run_continuous(program_bytes)
print(f"  Baseline:  {baseline_cycles:>10,} cycles | {baseline_ips:>12,.0f} IPS")

fusion_cycles, fusion_elapsed, fusion_ips = run_fusion(program_bytes)
print(f"  Fusion:    {fusion_cycles:>10,} cycles | {fusion_ips:>12,.0f} IPS | {fusion_ips/baseline_ips:.2f}x")

bbcache_cycles, bbcache_elapsed, bbcache_ips, bbcache_hit_rate, _ = run_bbcache(program_bytes)
print(f"  BBCache:   {bbcache_cycles:>10,} cycles | {bbcache_ips:>12,.0f} IPS | {bbcache_ips/baseline_ips:.2f}x | Cache: {bbcache_hit_rate:.1f}%")

results_test3 = {'Baseline': baseline_ips, 'Fusion': fusion_ips, 'BBCache': bbcache_ips}

# =============================================================================
# TEST 4: Sequential ADDs
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

baseline_cycles, baseline_elapsed, baseline_ips = run_continuous(program_bytes)
print(f"  Baseline:  {baseline_cycles:>10,} cycles | {baseline_ips:>12,.0f} IPS")

fusion_cycles, fusion_elapsed, fusion_ips = run_fusion(program_bytes)
print(f"  Fusion:    {fusion_cycles:>10,} cycles | {fusion_ips:>12,.0f} IPS | {fusion_ips/baseline_ips:.2f}x")

bbcache_cycles, bbcache_elapsed, bbcache_ips, bbcache_hit_rate, _ = run_bbcache(program_bytes)
print(f"  BBCache:   {bbcache_cycles:>10,} cycles | {bbcache_ips:>12,.0f} IPS | {bbcache_ips/baseline_ips:.2f}x | Cache: {bbcache_hit_rate:.1f}%")

results_test4 = {'Baseline': baseline_ips, 'Fusion': fusion_ips, 'BBCache': bbcache_ips}

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'='*80}")
print("SUMMARY")
print("="*80)

print("\n| Test | Baseline IPS | Fusion Speedup | BBCache Speedup | Best |")
print("|------|--------------|----------------|-----------------|------|")

all_results = [
    ("Simple Loop", results_test1),
    ("Nested Loop", results_test2),
    ("Memory Loop", results_test3),
    ("ADD Fusion", results_test4),
]

for test_name, results in all_results:
    baseline = results['Baseline']
    fusion_speedup = results['Fusion'] / baseline if baseline > 0 else 0
    bbcache_speedup = results['BBCache'] / baseline if baseline > 0 else 0
    best = "BBCache" if bbcache_speedup > fusion_speedup else "Fusion"
    print(f"| {test_name:12} | {baseline:>12,.0f} | {fusion_speedup:>14.2f}x | {bbcache_speedup:>15.2f}x | {best:6} |")

print(f"\n{'='*80}")
print("BENCHMARK COMPLETE")
print("="*80)

# Calculate averages
avg_fusion = sum(r['Fusion']/r['Baseline'] for _, r in all_results) / len(all_results)
avg_bbcache = sum(r['BBCache']/r['Baseline'] for _, r in all_results) / len(all_results)
print(f"\nAverage Speedup:")
print(f"  Fusion:  {avg_fusion:.2f}x")
print(f"  BBCache: {avg_bbcache:.2f}x")
