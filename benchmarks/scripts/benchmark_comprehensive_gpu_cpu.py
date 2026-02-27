#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║              COMPREHENSIVE GPU vs CPU BENCHMARK SUITE                            ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  Measures ACTUAL performance of different execution paths:                       ║
║                                                                                  ║
║  1. Pure Python/NumPy CPU path (_run_fast)                                      ║
║  2. GPU Tensor path (run_parallel_gpu)                                          ║
║  3. GPU Micro-batch path (run_gpu_microbatch)                                   ║
║  4. Adaptive path (run_elf_adaptive)                                            ║
║                                                                                  ║
║  Also measures individual component overhead:                                    ║
║  - Instruction fetch latency                                                     ║
║  - GPU↔CPU sync overhead                                                         ║
║  - Neural model inference time                                                   ║
║  - Memory access patterns                                                        ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import time
import sys
import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import statistics

sys.path.insert(0, str(Path(__file__).parent))

# Suppress init output
class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
    def __exit__(self, *args):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr

print("Loading Neural CPU...")
with SuppressOutput():
    from neural_cpu import NeuralCPU, device

print(f"Device: {device}")


@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    total_time_ms: float
    ops_per_second: float
    avg_latency_us: float
    min_latency_us: float
    max_latency_us: float
    std_dev_us: float


def benchmark_gpu_cpu_sync():
    """Measure GPU↔CPU synchronization overhead."""
    print("\n" + "="*70)
    print("BENCHMARK 1: GPU↔CPU Synchronization Overhead")
    print("="*70)

    cpu = NeuralCPU()
    results = {}

    # Test 1: .item() call overhead
    print("\n[1.1] Tensor.item() overhead (GPU→CPU sync)")
    tensor = torch.tensor(42, dtype=torch.int64, device=device)

    latencies = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        _ = tensor.item()
        latencies.append(time.perf_counter_ns() - start)

    avg_ns = statistics.mean(latencies)
    results['item_call'] = BenchmarkResult(
        name="tensor.item()",
        iterations=1000,
        total_time_ms=sum(latencies) / 1e6,
        ops_per_second=1e9 / avg_ns,
        avg_latency_us=avg_ns / 1000,
        min_latency_us=min(latencies) / 1000,
        max_latency_us=max(latencies) / 1000,
        std_dev_us=statistics.stdev(latencies) / 1000
    )
    print(f"  Average: {avg_ns/1000:.2f} µs per .item() call")
    print(f"  Max throughput: {1e9/avg_ns:,.0f} syncs/sec")

    # Test 2: Batch sync vs individual
    print("\n[1.2] Batch sync vs individual .item() calls")
    regs = torch.zeros(32, dtype=torch.int64, device=device)

    # Individual
    latencies_individual = []
    for _ in range(100):
        start = time.perf_counter_ns()
        for i in range(32):
            _ = regs[i].item()
        latencies_individual.append(time.perf_counter_ns() - start)

    # Batch
    latencies_batch = []
    for _ in range(100):
        start = time.perf_counter_ns()
        _ = regs.cpu().numpy()
        latencies_batch.append(time.perf_counter_ns() - start)

    avg_individual = statistics.mean(latencies_individual)
    avg_batch = statistics.mean(latencies_batch)

    results['individual_32_items'] = BenchmarkResult(
        name="32 individual .item()",
        iterations=100,
        total_time_ms=sum(latencies_individual) / 1e6,
        ops_per_second=1e9 / avg_individual,
        avg_latency_us=avg_individual / 1000,
        min_latency_us=min(latencies_individual) / 1000,
        max_latency_us=max(latencies_individual) / 1000,
        std_dev_us=statistics.stdev(latencies_individual) / 1000
    )

    results['batch_32_sync'] = BenchmarkResult(
        name="batch .cpu().numpy()",
        iterations=100,
        total_time_ms=sum(latencies_batch) / 1e6,
        ops_per_second=1e9 / avg_batch,
        avg_latency_us=avg_batch / 1000,
        min_latency_us=min(latencies_batch) / 1000,
        max_latency_us=max(latencies_batch) / 1000,
        std_dev_us=statistics.stdev(latencies_batch) / 1000
    )

    print(f"  Individual 32x .item(): {avg_individual/1000:.2f} µs")
    print(f"  Batch .cpu().numpy():   {avg_batch/1000:.2f} µs")
    print(f"  Speedup: {avg_individual/avg_batch:.1f}x")

    return results


def benchmark_instruction_fetch():
    """Measure instruction fetch overhead."""
    print("\n" + "="*70)
    print("BENCHMARK 2: Instruction Fetch Patterns")
    print("="*70)

    cpu = NeuralCPU()
    results = {}

    # Fill memory with NOPs (0xD503201F)
    nop = 0xD503201F
    for i in range(0, 10000, 4):
        cpu.memory[i] = nop & 0xFF
        cpu.memory[i+1] = (nop >> 8) & 0xFF
        cpu.memory[i+2] = (nop >> 16) & 0xFF
        cpu.memory[i+3] = (nop >> 24) & 0xFF

    # Test 1: Single instruction fetch (with .item())
    print("\n[2.1] Single instruction fetch with .item()")
    latencies = []
    for pc in range(0, 4000, 4):
        cpu.pc = torch.tensor(pc, dtype=torch.int64, device=device)
        start = time.perf_counter_ns()
        pc_int = int(cpu.pc.item())
        inst = (int(cpu.memory[pc_int].item()) |
                (int(cpu.memory[pc_int+1].item()) << 8) |
                (int(cpu.memory[pc_int+2].item()) << 16) |
                (int(cpu.memory[pc_int+3].item()) << 24))
        latencies.append(time.perf_counter_ns() - start)

    avg_ns = statistics.mean(latencies)
    results['single_fetch_with_item'] = BenchmarkResult(
        name="single fetch (5x .item())",
        iterations=1000,
        total_time_ms=sum(latencies) / 1e6,
        ops_per_second=1e9 / avg_ns,
        avg_latency_us=avg_ns / 1000,
        min_latency_us=min(latencies) / 1000,
        max_latency_us=max(latencies) / 1000,
        std_dev_us=statistics.stdev(latencies) / 1000
    )
    print(f"  Average: {avg_ns/1000:.2f} µs per fetch")
    print(f"  Max IPS (fetch only): {1e9/avg_ns:,.0f}")

    # Test 2: Batch instruction fetch (pure tensor)
    print("\n[2.2] Batch instruction fetch (tensor ops only)")
    latencies = []
    batch_size = 64

    for _ in range(100):
        pc = torch.tensor(0, dtype=torch.int64, device=device)
        start = time.perf_counter_ns()

        # Fetch batch_size instructions using tensor ops
        byte_offsets = torch.arange(batch_size * 4, device=device, dtype=torch.int64)
        byte_indices = (pc + byte_offsets).clamp(0, cpu.mem_size - 1)
        byte_range = cpu.memory.gather(0, byte_indices).view(batch_size, 4).long()
        insts = (byte_range[:, 0] |
                (byte_range[:, 1] << 8) |
                (byte_range[:, 2] << 16) |
                (byte_range[:, 3] << 24))

        # Force sync
        _ = insts[0].item()
        latencies.append(time.perf_counter_ns() - start)

    avg_ns = statistics.mean(latencies)
    results['batch_fetch_64'] = BenchmarkResult(
        name=f"batch fetch ({batch_size} insts)",
        iterations=100,
        total_time_ms=sum(latencies) / 1e6,
        ops_per_second=batch_size * 1e9 / avg_ns,
        avg_latency_us=avg_ns / 1000,
        min_latency_us=min(latencies) / 1000,
        max_latency_us=max(latencies) / 1000,
        std_dev_us=statistics.stdev(latencies) / 1000
    )
    print(f"  Average: {avg_ns/1000:.2f} µs for {batch_size} instructions")
    print(f"  Per-instruction: {avg_ns/1000/batch_size:.2f} µs")
    print(f"  Max IPS (batch fetch): {batch_size * 1e9/avg_ns:,.0f}")

    return results


def benchmark_alu_operations():
    """Measure ALU operation performance."""
    print("\n" + "="*70)
    print("BENCHMARK 3: ALU Operations")
    print("="*70)

    cpu = NeuralCPU()
    results = {}

    # Test 1: Single ADD (tensor)
    print("\n[3.1] Single ADD operation (tensor)")
    a = torch.tensor(100, dtype=torch.int64, device=device)
    b = torch.tensor(200, dtype=torch.int64, device=device)

    latencies = []
    for _ in range(10000):
        start = time.perf_counter_ns()
        c = a + b
        _ = c.item()  # Force sync
        latencies.append(time.perf_counter_ns() - start)

    avg_ns = statistics.mean(latencies)
    results['single_add'] = BenchmarkResult(
        name="single ADD",
        iterations=10000,
        total_time_ms=sum(latencies) / 1e6,
        ops_per_second=1e9 / avg_ns,
        avg_latency_us=avg_ns / 1000,
        min_latency_us=min(latencies) / 1000,
        max_latency_us=max(latencies) / 1000,
        std_dev_us=statistics.stdev(latencies) / 1000
    )
    print(f"  Average: {avg_ns/1000:.2f} µs")
    print(f"  Max ops/sec: {1e9/avg_ns:,.0f}")

    # Test 2: Batch ADD (1024 parallel)
    print("\n[3.2] Batch ADD (1024 parallel operations)")
    a_batch = torch.randint(0, 1000, (1024,), dtype=torch.int64, device=device)
    b_batch = torch.randint(0, 1000, (1024,), dtype=torch.int64, device=device)

    latencies = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        c_batch = a_batch + b_batch
        _ = c_batch[0].item()  # Force sync
        latencies.append(time.perf_counter_ns() - start)

    avg_ns = statistics.mean(latencies)
    results['batch_add_1024'] = BenchmarkResult(
        name="batch ADD (1024)",
        iterations=1000,
        total_time_ms=sum(latencies) / 1e6,
        ops_per_second=1024 * 1e9 / avg_ns,
        avg_latency_us=avg_ns / 1000,
        min_latency_us=min(latencies) / 1000,
        max_latency_us=max(latencies) / 1000,
        std_dev_us=statistics.stdev(latencies) / 1000
    )
    print(f"  Average: {avg_ns/1000:.2f} µs for 1024 ops")
    print(f"  Per-op: {avg_ns/1024:.2f} ns")
    print(f"  Throughput: {1024 * 1e9/avg_ns:,.0f} ops/sec")

    # Test 3: Complex ALU chain (no sync until end)
    print("\n[3.3] ALU chain (ADD→SUB→MUL→AND→OR, no intermediate sync)")

    latencies = []
    for _ in range(1000):
        a = torch.randint(1, 100, (1024,), dtype=torch.int64, device=device)
        b = torch.randint(1, 100, (1024,), dtype=torch.int64, device=device)

        start = time.perf_counter_ns()
        c = a + b
        d = c - a
        e = d * b
        f = e & 0xFF
        g = f | 0x100
        _ = g[0].item()  # Single sync at end
        latencies.append(time.perf_counter_ns() - start)

    avg_ns = statistics.mean(latencies)
    results['alu_chain_5ops'] = BenchmarkResult(
        name="ALU chain (5 ops, 1024 wide)",
        iterations=1000,
        total_time_ms=sum(latencies) / 1e6,
        ops_per_second=5 * 1024 * 1e9 / avg_ns,
        avg_latency_us=avg_ns / 1000,
        min_latency_us=min(latencies) / 1000,
        max_latency_us=max(latencies) / 1000,
        std_dev_us=statistics.stdev(latencies) / 1000
    )
    print(f"  Average: {avg_ns/1000:.2f} µs for 5*1024 ops")
    print(f"  Throughput: {5 * 1024 * 1e9/avg_ns:,.0f} ops/sec")

    return results


def benchmark_memory_operations():
    """Measure memory operation performance."""
    print("\n" + "="*70)
    print("BENCHMARK 4: Memory Operations")
    print("="*70)

    cpu = NeuralCPU()
    results = {}

    # Test 1: Single byte read
    print("\n[4.1] Single byte read with .item()")
    latencies = []
    for addr in range(10000):
        start = time.perf_counter_ns()
        val = cpu.memory[addr].item()
        latencies.append(time.perf_counter_ns() - start)

    avg_ns = statistics.mean(latencies)
    results['single_byte_read'] = BenchmarkResult(
        name="single byte read",
        iterations=10000,
        total_time_ms=sum(latencies) / 1e6,
        ops_per_second=1e9 / avg_ns,
        avg_latency_us=avg_ns / 1000,
        min_latency_us=min(latencies) / 1000,
        max_latency_us=max(latencies) / 1000,
        std_dev_us=statistics.stdev(latencies) / 1000
    )
    print(f"  Average: {avg_ns/1000:.2f} µs")
    print(f"  Bandwidth: {1e9/avg_ns:,.0f} bytes/sec")

    # Test 2: Batch read (1KB)
    print("\n[4.2] Batch read (1KB slice)")
    latencies = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        data = cpu.memory[0:1024].cpu().numpy()
        latencies.append(time.perf_counter_ns() - start)

    avg_ns = statistics.mean(latencies)
    results['batch_read_1kb'] = BenchmarkResult(
        name="batch read (1KB)",
        iterations=1000,
        total_time_ms=sum(latencies) / 1e6,
        ops_per_second=1024 * 1e9 / avg_ns,
        avg_latency_us=avg_ns / 1000,
        min_latency_us=min(latencies) / 1000,
        max_latency_us=max(latencies) / 1000,
        std_dev_us=statistics.stdev(latencies) / 1000
    )
    print(f"  Average: {avg_ns/1000:.2f} µs for 1KB")
    print(f"  Bandwidth: {1024 * 1e9/avg_ns / 1e6:.1f} MB/sec")

    # Test 3: Gather operation (random access)
    print("\n[4.3] Gather operation (64 random addresses)")
    indices = torch.randint(0, cpu.mem_size, (64,), dtype=torch.int64, device=device)

    latencies = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        data = cpu.memory.gather(0, indices)
        _ = data[0].item()
        latencies.append(time.perf_counter_ns() - start)

    avg_ns = statistics.mean(latencies)
    results['gather_64'] = BenchmarkResult(
        name="gather (64 random)",
        iterations=1000,
        total_time_ms=sum(latencies) / 1e6,
        ops_per_second=64 * 1e9 / avg_ns,
        avg_latency_us=avg_ns / 1000,
        min_latency_us=min(latencies) / 1000,
        max_latency_us=max(latencies) / 1000,
        std_dev_us=statistics.stdev(latencies) / 1000
    )
    print(f"  Average: {avg_ns/1000:.2f} µs for 64 gathers")
    print(f"  Throughput: {64 * 1e9/avg_ns:,.0f} random reads/sec")

    return results


def benchmark_execution_paths():
    """Measure actual execution path performance."""
    print("\n" + "="*70)
    print("BENCHMARK 5: Execution Path Comparison")
    print("="*70)

    results = {}

    # Create a simple test program: countdown loop
    # MOV X0, #1000      ; 0xD2807D00
    # loop:
    # SUB X0, X0, #1     ; 0xD1000400
    # CBNZ X0, loop      ; 0xB5FFFFC0
    # MOV X8, #93        ; 0xD2800BA8 (exit syscall)
    # SVC #0             ; 0xD4000001

    test_program = bytes([
        0x00, 0x7D, 0x80, 0xD2,  # MOV X0, #1000
        0x00, 0x04, 0x00, 0xD1,  # SUB X0, X0, #1
        0xC0, 0xFF, 0xFF, 0xB5,  # CBNZ X0, -4
        0xA8, 0x0B, 0x80, 0xD2,  # MOV X8, #93
        0x01, 0x00, 0x00, 0xD4,  # SVC #0
    ])

    iterations_to_test = [100, 1000, 10000]

    for loop_count in iterations_to_test:
        # Adjust the MOV instruction for different loop counts
        mov_inst = 0xD2800000 | (loop_count << 5)
        adjusted_program = bytes([
            mov_inst & 0xFF, (mov_inst >> 8) & 0xFF,
            (mov_inst >> 16) & 0xFF, (mov_inst >> 24) & 0xFF,
        ]) + test_program[4:]

        print(f"\n[5.{iterations_to_test.index(loop_count)+1}] Loop count: {loop_count}")

        # Test CPU fast path
        cpu = NeuralCPU()
        cpu.memory[:len(adjusted_program)] = torch.tensor(list(adjusted_program), dtype=torch.uint8, device=device)
        cpu.pc = torch.tensor(0, dtype=torch.int64, device=device)
        cpu.regs.zero_()
        cpu.halted = False

        start = time.perf_counter()
        executed, _ = cpu._run_fast(max_instructions=loop_count * 3 + 10)
        elapsed = time.perf_counter() - start

        ips_fast = executed / elapsed if elapsed > 0 else 0
        results[f'cpu_fast_{loop_count}'] = {
            'loop_count': loop_count,
            'executed': executed,
            'time_ms': elapsed * 1000,
            'ips': ips_fast
        }
        print(f"  CPU Fast:       {executed:,} inst in {elapsed*1000:.2f}ms = {ips_fast:,.0f} IPS")

        # Test GPU parallel
        cpu = NeuralCPU()
        cpu.memory[:len(adjusted_program)] = torch.tensor(list(adjusted_program), dtype=torch.uint8, device=device)
        cpu.pc = torch.tensor(0, dtype=torch.int64, device=device)
        cpu.regs.zero_()
        cpu.halted = False

        start = time.perf_counter()
        executed_t, _ = cpu.run_parallel_gpu(max_instructions=loop_count * 3 + 10, batch_size=64)
        executed = int(executed_t.item())
        elapsed = time.perf_counter() - start

        ips_gpu = executed / elapsed if elapsed > 0 else 0
        results[f'gpu_parallel_{loop_count}'] = {
            'loop_count': loop_count,
            'executed': executed,
            'time_ms': elapsed * 1000,
            'ips': ips_gpu
        }
        print(f"  GPU Parallel:   {executed:,} inst in {elapsed*1000:.2f}ms = {ips_gpu:,.0f} IPS")

        # Test GPU microbatch
        cpu = NeuralCPU()
        cpu.memory[:len(adjusted_program)] = torch.tensor(list(adjusted_program), dtype=torch.uint8, device=device)
        cpu.pc = torch.tensor(0, dtype=torch.int64, device=device)
        cpu.regs.zero_()
        cpu.halted = False

        start = time.perf_counter()
        executed_t, _ = cpu.run_gpu_microbatch(max_instructions=loop_count * 3 + 10, microbatch_size=32)
        executed = int(executed_t.item())
        elapsed = time.perf_counter() - start

        ips_micro = executed / elapsed if elapsed > 0 else 0
        results[f'gpu_microbatch_{loop_count}'] = {
            'loop_count': loop_count,
            'executed': executed,
            'time_ms': elapsed * 1000,
            'ips': ips_micro
        }
        print(f"  GPU Microbatch: {executed:,} inst in {elapsed*1000:.2f}ms = {ips_micro:,.0f} IPS")

        # Compare
        if ips_fast > 0:
            print(f"  GPU Parallel vs CPU:   {ips_gpu/ips_fast:.2f}x")
            print(f"  GPU Microbatch vs CPU: {ips_micro/ips_fast:.2f}x")

    return results


def benchmark_neural_models():
    """Measure neural model inference overhead."""
    print("\n" + "="*70)
    print("BENCHMARK 6: Neural Model Inference")
    print("="*70)

    cpu = NeuralCPU()
    results = {}

    # Test 1: Loop detector inference
    print("\n[6.1] Neural Loop Detector")
    if hasattr(cpu, 'loop_detector') and cpu.loop_detector is not None:
        # Create fake input
        body_bits = torch.randint(0, 2, (8, 32), dtype=torch.float32, device=device)
        reg_values = torch.randint(0, 1000, (32,), dtype=torch.int64, device=device)

        latencies = []
        for _ in range(100):
            start = time.perf_counter_ns()
            with torch.no_grad():
                type_logits, counter_probs, iterations = cpu.loop_detector(body_bits, reg_values)
            _ = type_logits[0].item()  # Force sync
            latencies.append(time.perf_counter_ns() - start)

        avg_ns = statistics.mean(latencies)
        results['loop_detector'] = BenchmarkResult(
            name="Loop Detector",
            iterations=100,
            total_time_ms=sum(latencies) / 1e6,
            ops_per_second=1e9 / avg_ns,
            avg_latency_us=avg_ns / 1000,
            min_latency_us=min(latencies) / 1000,
            max_latency_us=max(latencies) / 1000,
            std_dev_us=statistics.stdev(latencies) / 1000
        )
        print(f"  Average: {avg_ns/1000:.2f} µs per inference")
    else:
        print("  Loop detector not available")

    # Test 2: Memory Oracle inference
    print("\n[6.2] Memory Oracle LSTM")
    if hasattr(cpu, 'memory_oracle') and cpu.memory_oracle.trained_model_loaded:
        latencies = []
        for _ in range(100):
            start = time.perf_counter_ns()
            cpu.memory_oracle.predict_and_prefetch(0x1000)
            latencies.append(time.perf_counter_ns() - start)

        avg_ns = statistics.mean(latencies)
        results['memory_oracle'] = BenchmarkResult(
            name="Memory Oracle",
            iterations=100,
            total_time_ms=sum(latencies) / 1e6,
            ops_per_second=1e9 / avg_ns,
            avg_latency_us=avg_ns / 1000,
            min_latency_us=min(latencies) / 1000,
            max_latency_us=max(latencies) / 1000,
            std_dev_us=statistics.stdev(latencies) / 1000
        )
        print(f"  Average: {avg_ns/1000:.2f} µs per prediction")
    else:
        print("  Memory Oracle not loaded or not trained")

    return results


def benchmark_bottleneck_analysis():
    """Detailed analysis of where time is spent in execution loop."""
    print("\n" + "="*70)
    print("BENCHMARK 7: Bottleneck Analysis (per-instruction breakdown)")
    print("="*70)

    cpu = NeuralCPU()

    # Simple program
    test_program = bytes([
        0x00, 0x00, 0x80, 0xD2,  # MOV X0, #0
        0x00, 0x04, 0x00, 0x91,  # ADD X0, X0, #1
        0x00, 0x04, 0x00, 0x91,  # ADD X0, X0, #1
        0x00, 0x04, 0x00, 0x91,  # ADD X0, X0, #1
        0xA8, 0x0B, 0x80, 0xD2,  # MOV X8, #93
        0x01, 0x00, 0x00, 0xD4,  # SVC #0
    ])

    cpu.memory[:len(test_program)] = torch.tensor(list(test_program), dtype=torch.uint8, device=device)

    # Measure individual phases of one instruction execution
    print("\n[7.1] Time breakdown for single instruction execution:")

    cpu.pc = torch.tensor(4, dtype=torch.int64, device=device)  # Point to ADD

    # Phase 1: PC read
    times_pc_read = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        pc_val = int(cpu.pc.item())
        times_pc_read.append(time.perf_counter_ns() - start)

    # Phase 2: Instruction fetch
    times_fetch = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        inst = (int(cpu.memory[4].item()) |
                (int(cpu.memory[5].item()) << 8) |
                (int(cpu.memory[6].item()) << 16) |
                (int(cpu.memory[7].item()) << 24))
        times_fetch.append(time.perf_counter_ns() - start)

    # Phase 3: Decode (pure Python)
    times_decode = []
    inst = 0x91000400  # ADD X0, X0, #1
    for _ in range(1000):
        start = time.perf_counter_ns()
        op_byte = (inst >> 24) & 0xFF
        rd = inst & 0x1F
        rn = (inst >> 5) & 0x1F
        imm12 = (inst >> 10) & 0xFFF
        is_add = (op_byte == 0x91)
        times_decode.append(time.perf_counter_ns() - start)

    # Phase 4: Execute (tensor op)
    times_execute = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        result = cpu.regs[0] + 1
        cpu.regs[0] = result
        times_execute.append(time.perf_counter_ns() - start)

    # Phase 5: PC update
    times_pc_update = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        cpu.pc = cpu.pc + 4
        times_pc_update.append(time.perf_counter_ns() - start)

    results = {
        'pc_read_us': statistics.mean(times_pc_read) / 1000,
        'fetch_us': statistics.mean(times_fetch) / 1000,
        'decode_us': statistics.mean(times_decode) / 1000,
        'execute_us': statistics.mean(times_execute) / 1000,
        'pc_update_us': statistics.mean(times_pc_update) / 1000,
    }

    total = sum(results.values())

    print(f"  PC Read:     {results['pc_read_us']:6.2f} µs ({results['pc_read_us']/total*100:5.1f}%)")
    print(f"  Fetch:       {results['fetch_us']:6.2f} µs ({results['fetch_us']/total*100:5.1f}%)")
    print(f"  Decode:      {results['decode_us']:6.2f} µs ({results['decode_us']/total*100:5.1f}%)")
    print(f"  Execute:     {results['execute_us']:6.2f} µs ({results['execute_us']/total*100:5.1f}%)")
    print(f"  PC Update:   {results['pc_update_us']:6.2f} µs ({results['pc_update_us']/total*100:5.1f}%)")
    print(f"  ─────────────────────────")
    print(f"  TOTAL:       {total:6.2f} µs")
    print(f"  Max IPS:     {1e6/total:,.0f}")

    return results


def main():
    """Run all benchmarks and save results."""
    print("="*70)
    print("     COMPREHENSIVE GPU vs CPU BENCHMARK SUITE")
    print("="*70)
    print(f"\nDevice: {device}")
    print(f"PyTorch: {torch.__version__}")

    all_results = {
        'device': str(device),
        'pytorch_version': torch.__version__,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Run all benchmarks
    all_results['sync_overhead'] = {k: asdict(v) for k, v in benchmark_gpu_cpu_sync().items()}
    all_results['instruction_fetch'] = {k: asdict(v) for k, v in benchmark_instruction_fetch().items()}
    all_results['alu_operations'] = {k: asdict(v) for k, v in benchmark_alu_operations().items()}
    all_results['memory_operations'] = {k: asdict(v) for k, v in benchmark_memory_operations().items()}
    all_results['execution_paths'] = benchmark_execution_paths()
    all_results['neural_models'] = {k: asdict(v) for k, v in benchmark_neural_models().items()}
    all_results['bottleneck'] = benchmark_bottleneck_analysis()

    # Save results
    output_dir = Path(__file__).parent / "docs" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "gpu_cpu_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: KEY FINDINGS")
    print("="*70)

    bottleneck = all_results['bottleneck']
    total_per_inst = sum(bottleneck.values())

    print(f"""
Per-Instruction Overhead Breakdown:
  • GPU↔CPU Sync (PC + Fetch): {bottleneck['pc_read_us'] + bottleneck['fetch_us']:.1f} µs ({(bottleneck['pc_read_us'] + bottleneck['fetch_us'])/total_per_inst*100:.0f}%)
  • Decode (Python):           {bottleneck['decode_us']:.1f} µs ({bottleneck['decode_us']/total_per_inst*100:.0f}%)
  • Execute (Tensor):          {bottleneck['execute_us']:.1f} µs ({bottleneck['execute_us']/total_per_inst*100:.0f}%)
  • PC Update:                 {bottleneck['pc_update_us']:.1f} µs ({bottleneck['pc_update_us']/total_per_inst*100:.0f}%)

Theoretical Max IPS (per-instruction): {1e6/total_per_inst:,.0f}

Key Insight: GPU↔CPU sync dominates! Each .item() call costs ~{all_results['sync_overhead']['item_call']['avg_latency_us']:.0f} µs

To achieve true GPU parallelism, we need:
  1. Eliminate per-instruction .item() calls
  2. Keep PC and control flow on GPU tensors
  3. Batch instructions into tensor operations
  4. Only sync to CPU for syscalls
""")

    return all_results


if __name__ == "__main__":
    results = main()
