#!/usr/bin/env python3
"""
Comprehensive Memory Trace Collector for LSTM Training

Generates synthetic memory access patterns that represent real workloads:
1. Sequential scans (memset, memcpy)
2. Strided access (struct iteration, matrix operations)
3. Pointer chasing (linked lists, tree traversal)
4. Random access (hash tables)
5. Stack operations (function calls)

This bypasses loop vectorization by generating traces directly, which gives
us the training data the Memory Oracle needs to learn prediction.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import math

@dataclass
class MemoryAccess:
    addr: int
    size: int
    is_load: bool
    pc: int = 0

    def to_dict(self):
        return {
            'addr': self.addr,
            'size': self.size,
            'type': 'load' if self.is_load else 'store',
            'pc': self.pc
        }


class SyntheticTraceGenerator:
    """Generate synthetic memory traces representing real workload patterns."""

    def __init__(self, base_addr: int = 0x10000):
        self.base_addr = base_addr
        self.traces: List[Dict] = []
        self.pattern_labels: List[str] = []

    def generate_sequential(self, count: int, size: int = 8, start: int = None) -> List[Dict]:
        """Generate sequential memory access pattern (memset, buffer scan)."""
        traces = []
        addr = start or self.base_addr + random.randint(0, 0x10000)

        for i in range(count):
            is_load = random.random() > 0.3  # 70% loads
            traces.append({
                'addr': addr,
                'size': size,
                'type': 'load' if is_load else 'store',
                'pc': 0x1000 + (i % 16) * 4,
                'pattern': 'sequential'
            })
            addr += size

        return traces

    def generate_strided(self, count: int, stride: int, element_size: int = 8, start: int = None) -> List[Dict]:
        """Generate strided access pattern (struct fields, matrix columns)."""
        traces = []
        addr = start or self.base_addr + random.randint(0, 0x10000)

        for i in range(count):
            is_load = random.random() > 0.2  # 80% loads
            traces.append({
                'addr': addr,
                'size': element_size,
                'type': 'load' if is_load else 'store',
                'pc': 0x2000 + (i % 8) * 4,
                'pattern': f'strided-{stride}'
            })
            addr += stride

        return traces

    def generate_linked_list(self, count: int, node_size: int = 48, start: int = None) -> List[Dict]:
        """Generate pointer chasing pattern (linked list traversal)."""
        traces = []

        # Create scattered node addresses (realistic allocation)
        nodes = []
        addr = start or self.base_addr + random.randint(0, 0x10000)
        for i in range(count):
            nodes.append(addr)
            # Variable spacing between nodes (realistic heap allocation)
            addr += node_size + random.randint(0, 128)

        # Traverse the list
        for i, node_addr in enumerate(nodes):
            # Load next pointer (at offset 0 typically)
            traces.append({
                'addr': node_addr,
                'size': 8,
                'type': 'load',
                'pc': 0x3000,
                'pattern': 'pointer-chase'
            })

            # Sometimes access data field too
            if random.random() > 0.5:
                data_offset = random.choice([8, 16, 24])
                traces.append({
                    'addr': node_addr + data_offset,
                    'size': random.choice([4, 8]),
                    'type': 'load',
                    'pc': 0x3004,
                    'pattern': 'pointer-chase'
                })

        return traces

    def generate_binary_tree(self, depth: int, start: int = None) -> List[Dict]:
        """Generate tree traversal pattern (binary search tree)."""
        traces = []
        node_size = 32  # left, right, key, value

        # Simulate tree layout (BFS order for simplicity, but random access)
        num_nodes = 2 ** depth - 1
        base = start or self.base_addr + random.randint(0, 0x10000)

        # Random search path down the tree
        current_idx = 0
        path_len = depth + random.randint(0, 3)

        for _ in range(path_len):
            if current_idx >= num_nodes:
                current_idx = 0  # Restart

            node_addr = base + current_idx * node_size

            # Load key for comparison
            traces.append({
                'addr': node_addr + 16,  # key offset
                'size': 8,
                'type': 'load',
                'pc': 0x4000,
                'pattern': 'tree-traversal'
            })

            # Follow left or right pointer
            child_offset = 0 if random.random() > 0.5 else 8
            traces.append({
                'addr': node_addr + child_offset,
                'size': 8,
                'type': 'load',
                'pc': 0x4004,
                'pattern': 'tree-traversal'
            })

            # Move to child
            current_idx = 2 * current_idx + (1 if child_offset == 0 else 2)

        return traces

    def generate_hash_table(self, lookups: int, buckets: int = 256, start: int = None) -> List[Dict]:
        """Generate hash table lookup pattern (random buckets + chains)."""
        traces = []
        base = start or self.base_addr + random.randint(0, 0x10000)
        bucket_size = 8  # Pointer to chain
        entry_size = 32  # key, value, next

        for _ in range(lookups):
            # Hash computation result -> bucket index
            bucket = random.randint(0, buckets - 1)
            bucket_addr = base + bucket * bucket_size

            # Load bucket pointer
            traces.append({
                'addr': bucket_addr,
                'size': 8,
                'type': 'load',
                'pc': 0x5000,
                'pattern': 'hash-lookup'
            })

            # Chain traversal (random length 1-5)
            chain_len = random.randint(1, 5)
            entry_addr = base + buckets * bucket_size + random.randint(0, 0x1000)

            for c in range(chain_len):
                # Load key for comparison
                traces.append({
                    'addr': entry_addr,
                    'size': 8,
                    'type': 'load',
                    'pc': 0x5004,
                    'pattern': 'hash-lookup'
                })

                # Load value or next pointer
                traces.append({
                    'addr': entry_addr + (8 if c == chain_len - 1 else 24),
                    'size': 8,
                    'type': 'load',
                    'pc': 0x5008,
                    'pattern': 'hash-lookup'
                })

                entry_addr += entry_size + random.randint(0, 64)

        return traces

    def generate_stack_ops(self, call_depth: int, local_vars: int = 4, start: int = None) -> List[Dict]:
        """Generate stack operation pattern (function calls, local variables)."""
        traces = []
        sp = start or 0xFFFF0000
        frame_size = local_vars * 8 + 16  # locals + saved fp/lr

        # Nested function calls
        for depth in range(call_depth):
            # Push frame (store saved registers)
            for offset in [0, 8]:  # FP, LR
                traces.append({
                    'addr': sp - frame_size + offset,
                    'size': 8,
                    'type': 'store',
                    'pc': 0x6000 + depth * 0x100,
                    'pattern': 'stack'
                })

            # Access local variables
            for var in range(local_vars):
                var_addr = sp - frame_size + 16 + var * 8
                # Mix of loads and stores
                traces.append({
                    'addr': var_addr,
                    'size': 8,
                    'type': 'store' if random.random() > 0.6 else 'load',
                    'pc': 0x6010 + depth * 0x100 + var * 4,
                    'pattern': 'stack'
                })

            sp -= frame_size

        # Unwind
        for depth in range(call_depth - 1, -1, -1):
            sp += frame_size
            # Pop frame (load saved registers)
            for offset in [8, 0]:  # LR, FP
                traces.append({
                    'addr': sp - frame_size + offset,
                    'size': 8,
                    'type': 'load',
                    'pc': 0x6080 + depth * 0x100,
                    'pattern': 'stack'
                })

        return traces

    def generate_memcpy(self, size: int, src: int = None, dst: int = None) -> List[Dict]:
        """Generate memcpy pattern (alternating load-store)."""
        traces = []
        src = src or self.base_addr + random.randint(0, 0x10000)
        dst = dst or src + size + random.randint(0x1000, 0x10000)

        chunk_size = random.choice([8, 16, 32])  # Different copy strategies

        for offset in range(0, size, chunk_size):
            # Load from source
            traces.append({
                'addr': src + offset,
                'size': min(chunk_size, size - offset),
                'type': 'load',
                'pc': 0x7000,
                'pattern': 'memcpy'
            })
            # Store to destination
            traces.append({
                'addr': dst + offset,
                'size': min(chunk_size, size - offset),
                'type': 'store',
                'pc': 0x7004,
                'pattern': 'memcpy'
            })

        return traces

    def generate_mixed_workload(self, total_traces: int) -> List[Dict]:
        """Generate a realistic mixed workload."""
        all_traces = []

        # Distribution of patterns (roughly matches real programs)
        patterns = [
            (0.25, lambda: self.generate_sequential(random.randint(50, 200))),
            (0.20, lambda: self.generate_strided(random.randint(30, 100),
                                                 random.choice([16, 24, 32, 48, 64]))),
            (0.15, lambda: self.generate_linked_list(random.randint(20, 100))),
            (0.10, lambda: self.generate_binary_tree(random.randint(4, 8))),
            (0.15, lambda: self.generate_hash_table(random.randint(10, 50))),
            (0.10, lambda: self.generate_stack_ops(random.randint(3, 10))),
            (0.05, lambda: self.generate_memcpy(random.randint(64, 512))),
        ]

        while len(all_traces) < total_traces:
            # Pick pattern based on distribution
            r = random.random()
            cumulative = 0
            for prob, generator in patterns:
                cumulative += prob
                if r < cumulative:
                    traces = generator()
                    all_traces.extend(traces)
                    break

        return all_traces[:total_traces]


def create_training_sequences(traces: List[Dict], seq_len: int = 64, overlap: int = 32) -> List[List[Dict]]:
    """Convert trace list into overlapping training sequences."""
    sequences = []

    for i in range(0, len(traces) - seq_len, seq_len - overlap):
        seq = traces[i:i + seq_len]
        sequences.append(seq)

    return sequences


def compute_delta_targets(sequence: List[Dict]) -> List[int]:
    """Compute delta targets for a sequence (what the LSTM should predict)."""
    deltas = []
    for i in range(1, len(sequence)):
        delta = sequence[i]['addr'] - sequence[i-1]['addr']
        deltas.append(delta)
    return deltas


def analyze_patterns(traces: List[Dict]) -> Dict:
    """Analyze pattern distribution in traces."""
    patterns = {}
    for t in traces:
        p = t.get('pattern', 'unknown')
        patterns[p] = patterns.get(p, 0) + 1

    # Compute delta statistics
    deltas = []
    for i in range(1, len(traces)):
        deltas.append(traces[i]['addr'] - traces[i-1]['addr'])

    if deltas:
        delta_counts = {}
        for d in deltas:
            delta_counts[d] = delta_counts.get(d, 0) + 1
        top_deltas = sorted(delta_counts.items(), key=lambda x: -x[1])[:10]
    else:
        top_deltas = []

    return {
        'total_traces': len(traces),
        'pattern_distribution': patterns,
        'top_deltas': top_deltas,
        'unique_addresses': len(set(t['addr'] for t in traces)),
        'load_ratio': sum(1 for t in traces if t['type'] == 'load') / len(traces) if traces else 0
    }


def main():
    print("=" * 70)
    print("  COMPREHENSIVE MEMORY TRACE COLLECTION FOR LSTM TRAINING")
    print("=" * 70)

    generator = SyntheticTraceGenerator()

    # Generate lots of traces
    total_traces = 100000
    print(f"\n[1] Generating {total_traces:,} synthetic memory traces...")

    traces = generator.generate_mixed_workload(total_traces)
    print(f"    Generated: {len(traces):,} traces")

    # Analyze patterns
    print("\n[2] Analyzing pattern distribution...")
    analysis = analyze_patterns(traces)

    print(f"\n    Total traces: {analysis['total_traces']:,}")
    print(f"    Unique addresses: {analysis['unique_addresses']:,}")
    print(f"    Load ratio: {analysis['load_ratio']:.1%}")
    print(f"\n    Pattern distribution:")
    for pattern, count in sorted(analysis['pattern_distribution'].items(), key=lambda x: -x[1]):
        pct = 100 * count / analysis['total_traces']
        print(f"      {pattern:20s}: {count:6,} ({pct:5.1f}%)")

    print(f"\n    Top 10 deltas:")
    for delta, count in analysis['top_deltas']:
        pct = 100 * count / (analysis['total_traces'] - 1)
        print(f"      {delta:>10}: {count:>6,} ({pct:5.1f}%)")

    # Create training sequences
    print("\n[3] Creating training sequences...")
    seq_len = 64
    overlap = 32
    sequences = create_training_sequences(traces, seq_len=seq_len, overlap=overlap)
    print(f"    Created {len(sequences):,} sequences of length {seq_len}")

    # Split into train/val/test
    random.shuffle(sequences)
    train_split = int(0.8 * len(sequences))
    val_split = int(0.9 * len(sequences))

    train_seqs = sequences[:train_split]
    val_seqs = sequences[train_split:val_split]
    test_seqs = sequences[val_split:]

    print(f"    Train: {len(train_seqs):,} sequences")
    print(f"    Val:   {len(val_seqs):,} sequences")
    print(f"    Test:  {len(test_seqs):,} sequences")

    # Export
    print("\n[4] Exporting training data...")

    output_dir = Path(__file__).parent

    # Format for training
    def format_for_export(seqs):
        return [{
            'sequence': seq,
            'deltas': compute_delta_targets(seq)
        } for seq in seqs]

    train_data = {
        'sequences': format_for_export(train_seqs),
        'metadata': analysis,
        'config': {
            'seq_len': seq_len,
            'overlap': overlap,
            'total_traces': total_traces
        }
    }

    val_data = {
        'sequences': format_for_export(val_seqs),
        'metadata': analysis
    }

    test_data = {
        'sequences': format_for_export(test_seqs),
        'metadata': analysis
    }

    # Save files
    train_file = output_dir / "memory_traces_train.json"
    val_file = output_dir / "memory_traces_val.json"
    test_file = output_dir / "memory_traces_test.json"

    with open(train_file, 'w') as f:
        json.dump(train_data, f)
    print(f"    Saved: {train_file} ({train_file.stat().st_size / 1024 / 1024:.1f} MB)")

    with open(val_file, 'w') as f:
        json.dump(val_data, f)
    print(f"    Saved: {val_file} ({val_file.stat().st_size / 1024 / 1024:.1f} MB)")

    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    print(f"    Saved: {test_file} ({test_file.stat().st_size / 1024 / 1024:.1f} MB)")

    print("\n" + "=" * 70)
    print("  TRACE COLLECTION COMPLETE")
    print("=" * 70)
    print(f"\n  Next: Run train_memory_oracle.py to train the LSTM predictor")


if __name__ == "__main__":
    main()
