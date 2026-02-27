#!/usr/bin/env python3
"""
🧠 FAST NEURAL MEMORY SYSTEM
=============================

High-performance neural memory implementations:

1. TensorMemory - External tensor on GPU/MPS (fast, large)
2. SparseNeuralMemory - Sparse tensors (huge address space)
3. CachedNeuralMemory - Hot path caching (very fast)
4. ContentAddressableMemory - Neural CAM (associative)

All maintain neural computation characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from collections import OrderedDict
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# =============================================================================
# 1. EXTERNAL TENSOR MEMORY (Fast, GPU-backed)
# =============================================================================

class TensorMemory(nn.Module):
    """
    Fast neural memory using external GPU/MPS tensors.

    - Memory stored as torch tensor on GPU/MPS
    - Vectorized read/write operations
    - Supports batched access
    - Up to GPU memory size (8-48GB typically)
    """

    def __init__(self, size_mb: int = 64, dtype=torch.uint8):
        super().__init__()
        self.size = size_mb * 1024 * 1024
        # Store on device as non-parameter tensor (external)
        self.register_buffer('data', torch.zeros(self.size, dtype=dtype, device=device))
        self.dtype = dtype

        # Stats
        self.reads = 0
        self.writes = 0

    def read_byte(self, addr: int) -> int:
        """Read single byte."""
        if 0 <= addr < self.size:
            self.reads += 1
            return self.data[addr].item()
        return 0

    def write_byte(self, addr: int, value: int):
        """Write single byte."""
        if 0 <= addr < self.size:
            self.writes += 1
            self.data[addr] = value & 0xFF

    def read_batch(self, addresses: torch.Tensor) -> torch.Tensor:
        """
        Vectorized batch read - MUCH faster than individual reads.

        Args:
            addresses: [N] tensor of addresses

        Returns:
            [N] tensor of values
        """
        self.reads += len(addresses)
        # Clamp addresses to valid range
        addresses = torch.clamp(addresses, 0, self.size - 1)
        return self.data[addresses]

    def write_batch(self, addresses: torch.Tensor, values: torch.Tensor):
        """
        Vectorized batch write.

        Args:
            addresses: [N] tensor of addresses
            values: [N] tensor of values
        """
        self.writes += len(addresses)
        addresses = torch.clamp(addresses, 0, self.size - 1)
        self.data[addresses] = values.to(self.dtype)

    def read_word(self, addr: int) -> int:
        """Read 32-bit word (little endian)."""
        if addr + 3 >= self.size:
            return 0
        bytes_tensor = self.data[addr:addr+4]
        val = int(bytes_tensor[0]) | (int(bytes_tensor[1]) << 8) | \
              (int(bytes_tensor[2]) << 16) | (int(bytes_tensor[3]) << 24)
        self.reads += 1
        return val

    def write_word(self, addr: int, value: int):
        """Write 32-bit word (little endian)."""
        if addr + 3 >= self.size:
            return
        self.data[addr] = value & 0xFF
        self.data[addr+1] = (value >> 8) & 0xFF
        self.data[addr+2] = (value >> 16) & 0xFF
        self.data[addr+3] = (value >> 24) & 0xFF
        self.writes += 1

    def read_dword(self, addr: int) -> int:
        """Read 64-bit dword (little endian)."""
        low = self.read_word(addr)
        high = self.read_word(addr + 4)
        return low | (high << 32)

    def write_dword(self, addr: int, value: int):
        """Write 64-bit dword (little endian)."""
        self.write_word(addr, value & 0xFFFFFFFF)
        self.write_word(addr + 4, (value >> 32) & 0xFFFFFFFF)

    def load_binary(self, data: bytes, addr: int):
        """Load binary data at address."""
        data_tensor = torch.tensor(list(data), dtype=self.dtype, device=device)
        end = min(addr + len(data), self.size)
        self.data[addr:end] = data_tensor[:end-addr]

    def get_stats(self) -> Dict:
        return {'reads': self.reads, 'writes': self.writes, 'size_mb': self.size // (1024*1024)}


# =============================================================================
# 2. SPARSE NEURAL MEMORY (Huge address space)
# =============================================================================

class SparseNeuralMemory(nn.Module):
    """
    Sparse neural memory for huge address spaces.

    - Only stores non-zero values
    - Supports address spaces up to 2^64
    - Uses hash table for O(1) access
    - Memory efficient for sparse access patterns
    """

    def __init__(self, default_value: int = 0):
        super().__init__()
        self.storage: Dict[int, int] = {}
        self.default = default_value
        self.reads = 0
        self.writes = 0

    def read_byte(self, addr: int) -> int:
        self.reads += 1
        return self.storage.get(addr, self.default)

    def write_byte(self, addr: int, value: int):
        self.writes += 1
        if value == self.default:
            self.storage.pop(addr, None)  # Don't store default values
        else:
            self.storage[addr] = value & 0xFF

    def read_word(self, addr: int) -> int:
        val = 0
        for i in range(4):
            val |= self.read_byte(addr + i) << (i * 8)
        return val

    def write_word(self, addr: int, value: int):
        for i in range(4):
            self.write_byte(addr + i, (value >> (i * 8)) & 0xFF)

    def read_dword(self, addr: int) -> int:
        low = self.read_word(addr)
        high = self.read_word(addr + 4)
        return low | (high << 32)

    def write_dword(self, addr: int, value: int):
        self.write_word(addr, value & 0xFFFFFFFF)
        self.write_word(addr + 4, (value >> 32) & 0xFFFFFFFF)

    def load_binary(self, data: bytes, addr: int):
        for i, b in enumerate(data):
            if b != self.default:
                self.storage[addr + i] = b

    def get_stats(self) -> Dict:
        return {
            'reads': self.reads,
            'writes': self.writes,
            'stored_bytes': len(self.storage),
            'memory_mb': len(self.storage) * 16 / (1024*1024)  # ~16 bytes per entry
        }


# =============================================================================
# 3. CACHED NEURAL MEMORY (Hot path optimization)
# =============================================================================

class CachedNeuralMemory(nn.Module):
    """
    Neural memory with hot-path caching.

    - L1 cache: Small, fast (tensor-based)
    - L2 cache: Larger, medium speed
    - Backing store: Full memory

    Optimized for temporal and spatial locality.
    """

    def __init__(self, size_mb: int = 64, l1_size: int = 4096, l2_size: int = 65536):
        super().__init__()

        # L1 Cache (on device, tensor)
        self.l1_size = l1_size
        self.l1_cache = torch.zeros(l1_size, dtype=torch.uint8, device=device)
        self.l1_tags = torch.full((l1_size,), -1, dtype=torch.long, device=device)
        self.l1_valid = torch.zeros(l1_size, dtype=torch.bool, device=device)

        # L2 Cache (Python dict, larger)
        self.l2_cache: OrderedDict = OrderedDict()
        self.l2_size = l2_size

        # Backing store (external tensor)
        self.backing = TensorMemory(size_mb=size_mb)

        # Stats
        self.l1_hits = 0
        self.l2_hits = 0
        self.misses = 0

    def _l1_index(self, addr: int) -> int:
        """Direct-mapped L1 index."""
        return addr % self.l1_size

    def read_byte(self, addr: int) -> int:
        """Read with cache hierarchy."""
        # L1 lookup
        idx = self._l1_index(addr)
        if self.l1_valid[idx] and self.l1_tags[idx].item() == addr:
            self.l1_hits += 1
            return self.l1_cache[idx].item()

        # L2 lookup
        if addr in self.l2_cache:
            self.l2_hits += 1
            value = self.l2_cache[addr]
            self.l2_cache.move_to_end(addr)  # LRU
            # Promote to L1
            self._l1_write(addr, value)
            return value

        # Miss - fetch from backing store
        self.misses += 1
        value = self.backing.read_byte(addr)

        # Add to L2
        if len(self.l2_cache) >= self.l2_size:
            self.l2_cache.popitem(last=False)
        self.l2_cache[addr] = value

        # Add to L1
        self._l1_write(addr, value)

        return value

    def _l1_write(self, addr: int, value: int):
        """Write to L1 cache."""
        idx = self._l1_index(addr)
        self.l1_cache[idx] = value
        self.l1_tags[idx] = addr
        self.l1_valid[idx] = True

    def write_byte(self, addr: int, value: int):
        """Write-through cache policy."""
        # Update L1
        self._l1_write(addr, value)

        # Update L2
        self.l2_cache[addr] = value

        # Write to backing store
        self.backing.write_byte(addr, value)

    def read_word(self, addr: int) -> int:
        val = 0
        for i in range(4):
            val |= self.read_byte(addr + i) << (i * 8)
        return val

    def write_word(self, addr: int, value: int):
        for i in range(4):
            self.write_byte(addr + i, (value >> (i * 8)) & 0xFF)

    def read_dword(self, addr: int) -> int:
        return self.read_word(addr) | (self.read_word(addr + 4) << 32)

    def write_dword(self, addr: int, value: int):
        self.write_word(addr, value & 0xFFFFFFFF)
        self.write_word(addr + 4, (value >> 32) & 0xFFFFFFFF)

    def load_binary(self, data: bytes, addr: int):
        self.backing.load_binary(data, addr)

    def get_stats(self) -> Dict:
        total = self.l1_hits + self.l2_hits + self.misses
        return {
            'l1_hits': self.l1_hits,
            'l2_hits': self.l2_hits,
            'misses': self.misses,
            'l1_hit_rate': self.l1_hits / total if total > 0 else 0,
            'l2_hit_rate': self.l2_hits / total if total > 0 else 0,
            'total_hit_rate': (self.l1_hits + self.l2_hits) / total if total > 0 else 0,
        }


# =============================================================================
# 4. NEURAL CONTENT-ADDRESSABLE MEMORY (Associative)
# =============================================================================

class NeuralCAM(nn.Module):
    """
    Neural Content-Addressable Memory.

    - Associative lookup by content similarity
    - Learned address/value embeddings
    - Soft attention over all locations
    - Good for pattern matching, not random access
    """

    def __init__(self, num_slots: int = 1024, key_dim: int = 64, value_dim: int = 64):
        super().__init__()
        self.num_slots = num_slots
        self.key_dim = key_dim
        self.value_dim = value_dim

        # Memory slots
        self.keys = nn.Parameter(torch.randn(num_slots, key_dim) * 0.02)
        self.values = nn.Parameter(torch.zeros(num_slots, value_dim))

        # Query network
        self.query_net = nn.Sequential(
            nn.Linear(64, key_dim * 2),
            nn.ReLU(),
            nn.Linear(key_dim * 2, key_dim)
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(64, value_dim * 2),
            nn.ReLU(),
            nn.Linear(value_dim * 2, value_dim)
        )

        self.temperature = 1.0

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """
        Associative read using attention.

        Args:
            query: [batch, 64] query tensor

        Returns:
            [batch, value_dim] retrieved values
        """
        # Generate query embedding
        q = self.query_net(query)  # [batch, key_dim]

        # Compute attention
        attention = F.softmax(
            torch.matmul(q, self.keys.T) / self.temperature,
            dim=-1
        )  # [batch, num_slots]

        # Weighted sum of values
        return torch.matmul(attention, self.values)  # [batch, value_dim]

    def write(self, key: torch.Tensor, value: torch.Tensor, slot: Optional[int] = None):
        """
        Write to memory.

        Args:
            key: [key_dim] key tensor
            value: [value_dim] value tensor
            slot: Optional specific slot, otherwise find best match
        """
        if slot is None:
            # Find best matching slot
            attention = F.softmax(
                torch.matmul(key.unsqueeze(0), self.keys.T),
                dim=-1
            )
            slot = attention.argmax().item()

        with torch.no_grad():
            self.keys[slot] = key
            self.values[slot] = value


# =============================================================================
# 5. DIGITAL RAM (Hybrid approach)
# =============================================================================

class DigitalRAM(nn.Module):
    """
    Digital RAM - Hybrid neural/traditional memory.

    - Fast tensor storage for bulk data
    - Neural addressing for learned patterns
    - Efficient for both random and sequential access
    """

    def __init__(self, size_mb: int = 64):
        super().__init__()
        self.size = size_mb * 1024 * 1024

        # Primary storage (fast tensor)
        self.register_buffer('ram', torch.zeros(self.size, dtype=torch.uint8, device=device))

        # Neural address translation (for virtual memory)
        self.address_net = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )

        # Stats
        self.ops = 0

    def read(self, addr: int) -> int:
        """Direct read."""
        self.ops += 1
        if 0 <= addr < self.size:
            return self.ram[addr].item()
        return 0

    def write(self, addr: int, value: int):
        """Direct write."""
        self.ops += 1
        if 0 <= addr < self.size:
            self.ram[addr] = value & 0xFF

    def read_region(self, addr: int, size: int) -> torch.Tensor:
        """Read contiguous region (fast)."""
        self.ops += 1
        end = min(addr + size, self.size)
        return self.ram[addr:end].clone()

    def write_region(self, addr: int, data: torch.Tensor):
        """Write contiguous region (fast)."""
        self.ops += 1
        size = min(len(data), self.size - addr)
        self.ram[addr:addr+size] = data[:size]

    def read_word(self, addr: int) -> int:
        if addr + 3 >= self.size:
            return 0
        b = self.ram[addr:addr+4]
        return int(b[0]) | (int(b[1]) << 8) | (int(b[2]) << 16) | (int(b[3]) << 24)

    def write_word(self, addr: int, value: int):
        if addr + 3 >= self.size:
            return
        self.ram[addr:addr+4] = torch.tensor([
            value & 0xFF,
            (value >> 8) & 0xFF,
            (value >> 16) & 0xFF,
            (value >> 24) & 0xFF
        ], dtype=torch.uint8, device=device)

    def load_binary(self, data: bytes, addr: int):
        """Load binary data."""
        t = torch.tensor(list(data), dtype=torch.uint8, device=device)
        end = min(addr + len(t), self.size)
        self.ram[addr:end] = t[:end-addr]


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark_memory_systems():
    """Benchmark different memory implementations."""
    import time

    print("=" * 60)
    print("🧠 NEURAL MEMORY BENCHMARK")
    print("=" * 60)

    n_ops = 100000

    # Test TensorMemory
    print("\n1. TensorMemory (64MB GPU tensor):")
    mem = TensorMemory(size_mb=64)
    start = time.time()
    for i in range(n_ops):
        mem.write_byte(i % (64*1024*1024), i & 0xFF)
        _ = mem.read_byte(i % (64*1024*1024))
    elapsed = time.time() - start
    print(f"   {n_ops*2} ops in {elapsed:.2f}s = {n_ops*2/elapsed:.0f} ops/sec")

    # Test CachedNeuralMemory
    print("\n2. CachedNeuralMemory (L1+L2+Backing):")
    mem = CachedNeuralMemory(size_mb=64)
    start = time.time()
    for i in range(n_ops):
        mem.write_byte(i % (64*1024*1024), i & 0xFF)
        _ = mem.read_byte(i % (64*1024*1024))
    elapsed = time.time() - start
    stats = mem.get_stats()
    print(f"   {n_ops*2} ops in {elapsed:.2f}s = {n_ops*2/elapsed:.0f} ops/sec")
    print(f"   L1 hit rate: {stats['l1_hit_rate']*100:.1f}%")
    print(f"   Total hit rate: {stats['total_hit_rate']*100:.1f}%")

    # Test SparseNeuralMemory
    print("\n3. SparseNeuralMemory (unlimited address space):")
    mem = SparseNeuralMemory()
    start = time.time()
    for i in range(n_ops):
        mem.write_byte(i * 1000, i & 0xFF)  # Sparse addresses
        _ = mem.read_byte(i * 1000)
    elapsed = time.time() - start
    stats = mem.get_stats()
    print(f"   {n_ops*2} ops in {elapsed:.2f}s = {n_ops*2/elapsed:.0f} ops/sec")
    print(f"   Stored bytes: {stats['stored_bytes']}")

    # Test DigitalRAM
    print("\n4. DigitalRAM (hybrid):")
    mem = DigitalRAM(size_mb=64)
    start = time.time()
    for i in range(n_ops):
        mem.write(i % (64*1024*1024), i & 0xFF)
        _ = mem.read(i % (64*1024*1024))
    elapsed = time.time() - start
    print(f"   {n_ops*2} ops in {elapsed:.2f}s = {n_ops*2/elapsed:.0f} ops/sec")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    benchmark_memory_systems()
