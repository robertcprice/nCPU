#!/usr/bin/env python3
"""
Collect Memory Access Traces for LSTM Training

Runs real programs (busybox echo, uname, etc.) and collects memory access
patterns to train the Memory Oracle LSTM predictor.

Output: JSON file with sequences of (address, size, type) tuples suitable
for training stride prediction and pattern detection models.
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np


class MemoryTraceCollector:
    """Collect memory access traces from program execution."""

    def __init__(self, max_traces: int = 100000):
        self.traces: List[Dict] = []
        self.max_traces = max_traces
        self.current_pc = 0

    def record(self, addr: int, size: int, is_load: bool, pc: int = 0):
        """Record a memory access."""
        if len(self.traces) < self.max_traces:
            self.traces.append({
                'addr': addr,
                'size': size,
                'type': 'load' if is_load else 'store',
                'pc': pc
            })

    def analyze_patterns(self) -> Dict:
        """Analyze collected traces for patterns."""
        if len(self.traces) < 2:
            return {'patterns': [], 'stride_distribution': {}}

        # Compute strides between consecutive accesses
        strides = []
        for i in range(1, len(self.traces)):
            stride = self.traces[i]['addr'] - self.traces[i-1]['addr']
            strides.append(stride)

        # Count stride frequency
        stride_counts = {}
        for s in strides:
            stride_counts[s] = stride_counts.get(s, 0) + 1

        # Sort by frequency
        sorted_strides = sorted(stride_counts.items(), key=lambda x: -x[1])[:20]

        # Detect patterns
        patterns = []

        # Check for sequential (stride = 1, 2, 4, 8)
        sequential_count = sum(stride_counts.get(s, 0) for s in [1, 2, 4, 8])
        if sequential_count > len(strides) * 0.3:
            patterns.append('sequential')

        # Check for consistent stride
        if sorted_strides and sorted_strides[0][1] > len(strides) * 0.5:
            patterns.append(f'strided-{sorted_strides[0][0]}')

        # Check for pointer chasing (large variable strides)
        large_strides = [s for s in strides if abs(s) > 64]
        if len(large_strides) > len(strides) * 0.3:
            patterns.append('pointer-chase')

        return {
            'total_accesses': len(self.traces),
            'unique_addresses': len(set(t['addr'] for t in self.traces)),
            'patterns': patterns,
            'top_strides': sorted_strides,
            'load_count': sum(1 for t in self.traces if t['type'] == 'load'),
            'store_count': sum(1 for t in self.traces if t['type'] == 'store')
        }

    def export_training_data(self, filename: str):
        """Export traces as training data for LSTM."""
        # Format: sequences of (addr, size, type_id)
        # type_id: 0 = load, 1 = store
        sequences = []

        # Create overlapping sequences of length 64
        seq_len = 64
        for i in range(0, len(self.traces) - seq_len, seq_len // 2):
            seq = []
            for j in range(seq_len):
                t = self.traces[i + j]
                seq.append({
                    'addr': t['addr'],
                    'size': t['size'],
                    'type': 0 if t['type'] == 'load' else 1
                })
            sequences.append(seq)

        data = {
            'sequences': sequences,
            'metadata': self.analyze_patterns()
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Exported {len(sequences)} training sequences to {filename}")
        return len(sequences)


def patch_cpu_for_tracing(cpu, collector: MemoryTraceCollector):
    """Monkey-patch CPU to collect traces instead of normal Oracle tracking."""

    # Store original methods
    original_record_load = cpu.memory_oracle.record_load
    original_record_store = cpu.memory_oracle.record_store

    def traced_record_load(addr: int, size: int = 8):
        collector.record(addr, size, is_load=True, pc=int(cpu.pc.item()))
        original_record_load(addr, size)

    def traced_record_store(addr: int, size: int = 8):
        collector.record(addr, size, is_load=False, pc=int(cpu.pc.item()))
        original_record_store(addr, size)

    cpu.memory_oracle.record_load = traced_record_load
    cpu.memory_oracle.record_store = traced_record_store


def run_simple_memory_test(collector: MemoryTraceCollector):
    """Run simple memory access patterns for baseline traces."""
    from neural_cpu import NeuralCPU

    print("\n" + "="*60)
    print("  COLLECTING MEMORY TRACES - Simple Patterns")
    print("="*60)

    cpu = NeuralCPU()
    patch_cpu_for_tracing(cpu, collector)

    # Test 1: Sequential memory scan (memset-like)
    print("\n[Test 1] Sequential memory scan...")
    code = bytearray()
    # MOVZ X0, #0x2000 (base)
    code += bytes([0x00, 0x00, 0x84, 0xD2])
    # MOVZ X1, #100 (count)
    code += bytes([0x81, 0x0C, 0x80, 0xD2])
    # Loop: STR XZR, [X0], #8
    code += bytes([0x1F, 0x84, 0x00, 0xF8])
    # SUB X1, X1, #1
    code += bytes([0x21, 0x04, 0x00, 0xD1])
    # CBNZ X1, -8
    code += bytes([0xC1, 0xFF, 0xFF, 0xB5])
    # HLT
    code += bytes([0x00, 0x00, 0x40, 0xD4])

    cpu.load_binary(bytes(code), addr=0x1000)
    cpu.pc = torch.tensor(0x1000, dtype=torch.int64, device=cpu.device)
    cpu.regs[31] = torch.tensor(0xFFF00, dtype=torch.int64, device=cpu.device)

    executed, elapsed = cpu.run_gpu_microbatch(max_instructions=10000)
    print(f"  Executed: {int(executed.item())} instructions in {elapsed:.3f}s")
    print(f"  Traces collected: {len(collector.traces)}")

    # Test 2: Strided access (struct-like)
    print("\n[Test 2] Strided memory access...")
    base_traces = len(collector.traces)

    code2 = bytearray()
    # MOVZ X0, #0x3000 (base)
    code2 += bytes([0x00, 0x00, 0x86, 0xD2])
    # MOVZ X1, #50 (count)
    code2 += bytes([0x41, 0x06, 0x80, 0xD2])
    # Loop: LDR X2, [X0], #24 (struct stride)
    code2 += bytes([0x02, 0x84, 0x41, 0xF8])
    # SUB X1, X1, #1
    code2 += bytes([0x21, 0x04, 0x00, 0xD1])
    # CBNZ X1, -8
    code2 += bytes([0xC1, 0xFF, 0xFF, 0xB5])
    # HLT
    code2 += bytes([0x00, 0x00, 0x40, 0xD4])

    cpu2 = NeuralCPU()
    patch_cpu_for_tracing(cpu2, collector)
    cpu2.load_binary(bytes(code2), addr=0x1000)
    cpu2.pc = torch.tensor(0x1000, dtype=torch.int64, device=cpu2.device)
    cpu2.regs[31] = torch.tensor(0xFFF00, dtype=torch.int64, device=cpu2.device)

    executed, elapsed = cpu2.run_gpu_microbatch(max_instructions=10000)
    print(f"  Executed: {int(executed.item())} instructions in {elapsed:.3f}s")
    print(f"  New traces: {len(collector.traces) - base_traces}")

    return len(collector.traces)


def run_linked_list_test(collector: MemoryTraceCollector):
    """Run linked list traversal for pointer-chasing patterns."""
    from neural_cpu import NeuralCPU

    print("\n" + "="*60)
    print("  COLLECTING MEMORY TRACES - Linked List Traversal")
    print("="*60)

    cpu = NeuralCPU()
    patch_cpu_for_tracing(cpu, collector)

    # Setup linked list at 0x4000
    base = 0x4000
    stride = 48  # Variable stride for more realistic pattern
    num_nodes = 50

    for i in range(num_nodes):
        addr = base + i * stride
        next_addr = base + (i + 1) * stride if i < num_nodes - 1 else 0
        for j in range(8):
            cpu.memory[addr + j] = ((next_addr) >> (j * 8)) & 0xFF

    print(f"  Setup: {num_nodes} nodes, stride={stride} bytes")

    base_traces = len(collector.traces)

    # Program: traverse list
    code = bytearray()
    # MOVZ X0, #0x4000
    code += bytes([0x00, 0x00, 0x88, 0xD2])
    # MOVZ X1, #0 (counter)
    code += bytes([0x01, 0x00, 0x80, 0xD2])
    # Loop: ADD X1, X1, #1
    code += bytes([0x21, 0x04, 0x00, 0x91])
    # LDR X0, [X0] (follow next)
    code += bytes([0x00, 0x00, 0x40, 0xF9])
    # CBNZ X0, -8
    code += bytes([0xC0, 0xFF, 0xFF, 0xB5])
    # HLT
    code += bytes([0x00, 0x00, 0x40, 0xD4])

    cpu.load_binary(bytes(code), addr=0x1000)
    cpu.pc = torch.tensor(0x1000, dtype=torch.int64, device=cpu.device)
    cpu.regs[31] = torch.tensor(0xFFF00, dtype=torch.int64, device=cpu.device)

    executed, elapsed = cpu.run_gpu_microbatch(max_instructions=10000)
    print(f"  Executed: {int(executed.item())} instructions in {elapsed:.3f}s")
    print(f"  New traces: {len(collector.traces) - base_traces}")

    return len(collector.traces) - base_traces


def main():
    print("="*70)
    print("  MEMORY TRACE COLLECTION FOR LSTM TRAINING")
    print("  Phase 1: Intelligent Dispatcher Architecture")
    print("="*70)

    collector = MemoryTraceCollector(max_traces=50000)

    # Collect traces from different patterns
    run_simple_memory_test(collector)
    run_linked_list_test(collector)

    # Analyze patterns
    print("\n" + "="*60)
    print("  PATTERN ANALYSIS")
    print("="*60)
    analysis = collector.analyze_patterns()

    print(f"\n  Total accesses: {analysis['total_accesses']:,}")
    print(f"  Unique addresses: {analysis['unique_addresses']:,}")
    print(f"  Loads: {analysis['load_count']:,}")
    print(f"  Stores: {analysis['store_count']:,}")
    print(f"  Patterns detected: {analysis['patterns']}")
    print(f"\n  Top 10 strides:")
    for stride, count in analysis['top_strides'][:10]:
        pct = 100 * count / (analysis['total_accesses'] - 1)
        print(f"    {stride:>8}: {count:>6} ({pct:5.1f}%)")

    # Export training data
    print("\n" + "="*60)
    print("  EXPORTING TRAINING DATA")
    print("="*60)

    output_file = Path(__file__).parent / "memory_traces_training.json"
    num_seqs = collector.export_training_data(str(output_file))

    print(f"\n  Output: {output_file}")
    print(f"  Sequences: {num_seqs}")

    print("\n" + "="*70)
    print("  TRACE COLLECTION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
