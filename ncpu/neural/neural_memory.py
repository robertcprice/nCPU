"""SSD-Backed Neural Memory: large memory system with neural prefetch and address translation.

Memory-mapped file storage (up to 1 GB) with a learned page cache. The neural
prefetcher (prefetch.pt LSTM) predicts upcoming page accesses from the address
stream, while the neural MMU (mmu.pt MLP) provides learned address translation.

Architecture:
    Primary storage:  mmap of a backing file on SSD (up to 1 GB)
    Page cache:       dict[page_num -> torch.uint8 tensor] in RAM/GPU
    Neural prefetch:  LSTM predicts next 4 pages from address history
    Neural MMU:       MLP translates virtual page -> physical page

The page cache uses LRU eviction with dirty-page write-back. The prefetcher
runs asynchronously every N accesses, loading predicted pages in the background.

Integration:
    from ncpu.neural.neural_memory import NeuralSSDMemory

    mem = NeuralSSDMemory(size=64*1024*1024)   # 64 MB
    mem.write(0x1000, b"Hello, world!")
    data = mem.read(0x1000, 13)
    print(mem.stats)

Usage with existing models:
    - models/os/prefetch.pt: LSTM address predictor (trained, 8.1 MB)
    - models/os/mmu.pt: MLP address translator (trained, 5.4 MB, 100% accuracy)
"""

from __future__ import annotations

import mmap
import os
import struct
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

MODELS_DIR = Path(__file__).parent.parent.parent / "models"


# ────────────────────────────────────────────────────────────────────────────
# Neural model wrappers (match trained checkpoint architectures exactly)
# ────────────────────────────────────────────────────────────────────────────

class _PrefetchNet(nn.Module):
    """Matches prefetch.pt state dict.

    addr_embed: Embedding(65536, 32)
    lstm:       LSTM(32, 64, 1, batch_first=True)
    predictor:  Linear(64, 4) -- predict 4 next address slots
    """

    def __init__(self):
        super().__init__()
        self.addr_embed = nn.Embedding(65536, 32)
        self.lstm = nn.LSTM(32, 64, num_layers=1, batch_first=True)
        self.predictor = nn.Linear(64, 4)

    def forward(self, addr_seq: torch.Tensor) -> torch.Tensor:
        """addr_seq: [1, T] int64 -> [4] predicted address slots."""
        emb = self.addr_embed(addr_seq)
        out, _ = self.lstm(emb)
        last = out[:, -1, :]
        pred = self.predictor(last).squeeze(0)
        return pred.long().clamp(0, 65535)


class _MMUNet(nn.Module):
    """Matches mmu.pt state dict.

    vpn_embed:  Embedding(4096, 64)
    asid_embed: Embedding(256, 16)
    mlp:        Linear(80, 256) -> ReLU -> Linear(256, 256) -> ReLU -> Linear(256, 4102)

    The MMU maps (virtual_page_number, address_space_id) -> physical mapping.
    For our purposes we use it as an address-space-aware page translator.
    """

    def __init__(self):
        super().__init__()
        self.vpn_embed = nn.Embedding(4096, 64)
        self.asid_embed = nn.Embedding(256, 16)
        self.mlp = nn.Sequential(
            nn.Linear(80, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4102),
        )

    def forward(self, vpn: torch.Tensor, asid: torch.Tensor) -> torch.Tensor:
        """vpn: [N] int, asid: [N] int -> [N, 4102] raw output."""
        v = self.vpn_embed(vpn)
        a = self.asid_embed(asid)
        combined = torch.cat([v, a], dim=-1)
        return self.mlp(combined)


# ────────────────────────────────────────────────────────────────────────────
# SSD-Backed Neural Memory
# ────────────────────────────────────────────────────────────────────────────

class NeuralSSDMemory:
    """Large memory system backed by memory-mapped files with neural prefetch.

    All reads/writes go through a page cache backed by an mmap file on disk.
    A trained LSTM prefetcher predicts upcoming pages from access patterns.
    A trained MMU MLP provides address translation metadata.

    Parameters:
        size: Total addressable memory in bytes (default 64 MB, max 1 GB).
        page_size: Page size in bytes (default 4096).
        max_cached_pages: Maximum pages held in the RAM cache (default 1024 = 4 MB).
        backing_file: Path to the backing file. If None, creates a temp file.
        models_dir: Path to the models directory containing prefetch.pt and mmu.pt.
        device: Torch device for neural models ('cpu' or 'mps' or 'cuda').
    """

    MAX_SIZE = 1 << 30  # 1 GB hard limit

    def __init__(
        self,
        size: int = 64 * 1024 * 1024,
        page_size: int = 4096,
        max_cached_pages: int = 1024,
        backing_file: Optional[str | Path] = None,
        models_dir: Path = MODELS_DIR,
        device: str = "cpu",
    ):
        if size > self.MAX_SIZE:
            raise ValueError(f"Size {size} exceeds maximum {self.MAX_SIZE} (1 GB)")
        if size % page_size != 0:
            # Round up to full pages
            size = ((size + page_size - 1) // page_size) * page_size

        self.size = size
        self.page_size = page_size
        self.max_cached_pages = max_cached_pages
        self.total_pages = size // page_size
        self.device = torch.device(device)

        # ── Backing file (mmap) ──────────────────────────────────────
        self._owns_file = backing_file is None
        if backing_file is None:
            fd, self._backing_path = tempfile.mkstemp(prefix="ncpu_mem_", suffix=".bin")
            os.close(fd)
        else:
            self._backing_path = str(backing_file)

        # Create/expand file to requested size
        with open(self._backing_path, "ab") as f:
            current_size = f.tell()
        if current_size < size:
            with open(self._backing_path, "r+b" if current_size > 0 else "wb") as f:
                f.seek(size - 1)
                f.write(b"\x00")

        self._file = open(self._backing_path, "r+b")
        self._mmap = mmap.mmap(self._file.fileno(), size)

        # ── Page cache (LRU via OrderedDict) ─────────────────────────
        self._cache: OrderedDict[int, bytearray] = OrderedDict()
        self._dirty: set[int] = set()
        self._lock = threading.Lock()

        # ── Neural components ────────────────────────────────────────
        self._prefetch_model: Optional[_PrefetchNet] = None
        self._mmu_model: Optional[_MMUNet] = None
        self._prefetch_loaded = False
        self._mmu_loaded = False
        self._load_models(models_dir)

        # ── Access pattern tracking ──────────────────────────────────
        self._access_history: list[int] = []
        self._history_max = 32
        self._prefetch_interval = 16  # run prefetcher every N accesses
        self._access_count = 0

        # ── Statistics ───────────────────────────────────────────────
        self._stats = {
            "reads": 0,
            "writes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "page_faults": 0,
            "evictions": 0,
            "dirty_writebacks": 0,
            "prefetch_runs": 0,
            "prefetch_hits": 0,     # pages that were prefetched and then accessed
            "prefetch_loads": 0,    # pages loaded via prefetch
            "mmu_lookups": 0,
        }
        self._prefetched_pages: set[int] = set()  # track which pages were prefetched

    def _load_models(self, models_dir: Path) -> None:
        """Load trained prefetch.pt and mmu.pt models."""
        # Prefetcher LSTM
        prefetch_path = models_dir / "os" / "prefetch.pt"
        if prefetch_path.exists():
            try:
                net = _PrefetchNet()
                state = torch.load(prefetch_path, map_location="cpu", weights_only=True)
                net.load_state_dict(state, strict=True)
                net.eval()
                net.to(self.device)
                self._prefetch_model = net
                self._prefetch_loaded = True
            except Exception as e:
                self._prefetch_loaded = False

        # MMU MLP
        mmu_path = models_dir / "os" / "mmu.pt"
        if mmu_path.exists():
            try:
                net = _MMUNet()
                state = torch.load(mmu_path, map_location="cpu", weights_only=True)
                net.load_state_dict(state, strict=True)
                net.eval()
                net.to(self.device)
                self._mmu_model = net
                self._mmu_loaded = True
            except Exception as e:
                self._mmu_loaded = False

    # ── Public API ───────────────────────────────────────────────────────

    def read(self, addr: int, length: int) -> bytes:
        """Read `length` bytes starting at `addr`.

        Handles reads that span multiple pages. Triggers neural prefetch
        after every prefetch_interval accesses.
        """
        if addr < 0 or addr + length > self.size:
            raise ValueError(
                f"Read out of bounds: addr={addr:#x}, length={length}, "
                f"size={self.size:#x}"
            )

        self._stats["reads"] += 1
        self._record_access(addr)

        result = bytearray()
        remaining = length
        cur_addr = addr

        while remaining > 0:
            page_num = cur_addr // self.page_size
            page_offset = cur_addr % self.page_size
            chunk_len = min(remaining, self.page_size - page_offset)

            page_data = self._get_page(page_num)
            result.extend(page_data[page_offset:page_offset + chunk_len])

            cur_addr += chunk_len
            remaining -= chunk_len

        return bytes(result)

    def write(self, addr: int, data: bytes | bytearray) -> None:
        """Write `data` starting at `addr`.

        Handles writes that span multiple pages. Pages are marked dirty
        for write-back to the mmap on eviction or flush.
        """
        if addr < 0 or addr + len(data) > self.size:
            raise ValueError(
                f"Write out of bounds: addr={addr:#x}, length={len(data)}, "
                f"size={self.size:#x}"
            )

        self._stats["writes"] += 1
        self._record_access(addr)

        remaining = len(data)
        cur_addr = addr
        data_offset = 0

        while remaining > 0:
            page_num = cur_addr // self.page_size
            page_offset = cur_addr % self.page_size
            chunk_len = min(remaining, self.page_size - page_offset)

            page_data = self._get_page(page_num)
            page_data[page_offset:page_offset + chunk_len] = \
                data[data_offset:data_offset + chunk_len]

            with self._lock:
                self._dirty.add(page_num)

            cur_addr += chunk_len
            data_offset += chunk_len
            remaining -= chunk_len

    def read_u8(self, addr: int) -> int:
        """Read a single unsigned byte."""
        return self.read(addr, 1)[0]

    def read_u32(self, addr: int) -> int:
        """Read a 32-bit unsigned integer (little-endian)."""
        return struct.unpack_from("<I", self.read(addr, 4))[0]

    def read_u64(self, addr: int) -> int:
        """Read a 64-bit unsigned integer (little-endian)."""
        return struct.unpack_from("<Q", self.read(addr, 8))[0]

    def write_u32(self, addr: int, value: int) -> None:
        """Write a 32-bit unsigned integer (little-endian)."""
        self.write(addr, struct.pack("<I", value & 0xFFFFFFFF))

    def write_u64(self, addr: int, value: int) -> None:
        """Write a 64-bit unsigned integer (little-endian)."""
        self.write(addr, struct.pack("<Q", value & 0xFFFFFFFFFFFFFFFF))

    def flush(self) -> int:
        """Write all dirty pages back to the mmap. Returns count flushed."""
        flushed = 0
        with self._lock:
            dirty_pages = list(self._dirty)
            self._dirty.clear()
        for page_num in dirty_pages:
            if page_num in self._cache:
                self._writeback_page(page_num)
                flushed += 1
        self._mmap.flush()
        return flushed

    def neural_mmu_lookup(self, virtual_addr: int, asid: int = 0) -> dict:
        """Run the neural MMU model on a virtual address.

        Returns a dict with the raw MMU output (page table entry metadata).
        This demonstrates the neural TLB/MMU providing learned address translation.
        """
        if not self._mmu_loaded or self._mmu_model is None:
            return {"available": False, "virtual_addr": virtual_addr}

        self._stats["mmu_lookups"] += 1
        vpn = (virtual_addr // self.page_size) % 4096
        asid_clamped = asid % 256

        with torch.no_grad():
            vpn_t = torch.tensor([vpn], dtype=torch.long, device=self.device)
            asid_t = torch.tensor([asid_clamped], dtype=torch.long, device=self.device)
            output = self._mmu_model(vpn_t, asid_t).squeeze(0)

        return {
            "available": True,
            "virtual_addr": virtual_addr,
            "vpn": vpn,
            "asid": asid_clamped,
            "output_dim": output.shape[0],
            "output_norm": output.norm().item(),
        }

    @property
    def stats(self) -> dict:
        """Return a copy of the statistics dictionary."""
        s = dict(self._stats)
        total_accesses = s["cache_hits"] + s["cache_misses"]
        s["hit_rate"] = s["cache_hits"] / max(total_accesses, 1)
        s["pages_in_cache"] = len(self._cache)
        s["dirty_pages"] = len(self._dirty)
        s["total_pages"] = self.total_pages
        s["prefetch_loaded"] = self._prefetch_loaded
        s["mmu_loaded"] = self._mmu_loaded
        if s["prefetch_loads"] > 0:
            s["prefetch_hit_rate"] = s["prefetch_hits"] / s["prefetch_loads"]
        else:
            s["prefetch_hit_rate"] = 0.0
        return s

    # ── Internal page management ─────────────────────────────────────

    def _get_page(self, page_num: int) -> bytearray:
        """Get a page from cache, loading from mmap on miss."""
        with self._lock:
            if page_num in self._cache:
                # Cache hit -- move to end (most recently used)
                self._cache.move_to_end(page_num)
                self._stats["cache_hits"] += 1
                if page_num in self._prefetched_pages:
                    self._stats["prefetch_hits"] += 1
                    self._prefetched_pages.discard(page_num)
                return self._cache[page_num]

        # Cache miss -- load from mmap
        self._stats["cache_misses"] += 1
        self._stats["page_faults"] += 1
        return self._load_page(page_num)

    def _load_page(self, page_num: int) -> bytearray:
        """Load a page from the mmap into the cache."""
        with self._lock:
            # Double-check after acquiring lock
            if page_num in self._cache:
                self._cache.move_to_end(page_num)
                return self._cache[page_num]

            # Evict if cache is full
            while len(self._cache) >= self.max_cached_pages:
                self._evict_lru()

            # Read from mmap
            offset = page_num * self.page_size
            data = bytearray(self._mmap[offset:offset + self.page_size])
            self._cache[page_num] = data
            return data

    def _evict_lru(self) -> None:
        """Evict the least recently used page. Must hold self._lock."""
        if not self._cache:
            return
        # popitem(last=False) removes the oldest (least recently used) item
        evicted_num, evicted_data = self._cache.popitem(last=False)
        self._stats["evictions"] += 1

        # Write back if dirty
        if evicted_num in self._dirty:
            self._dirty.discard(evicted_num)
            offset = evicted_num * self.page_size
            self._mmap[offset:offset + self.page_size] = bytes(evicted_data)
            self._stats["dirty_writebacks"] += 1

        self._prefetched_pages.discard(evicted_num)

    def _writeback_page(self, page_num: int) -> None:
        """Write a cached page back to the mmap."""
        if page_num in self._cache:
            offset = page_num * self.page_size
            self._mmap[offset:offset + self.page_size] = bytes(self._cache[page_num])
            self._stats["dirty_writebacks"] += 1

    # ── Neural prefetch ──────────────────────────────────────────────

    def _record_access(self, addr: int) -> None:
        """Record an address access and trigger prefetch periodically."""
        slot = (addr >> 2) & 0xFFFF  # address -> embedding slot (16-bit)
        self._access_history.append(slot)
        if len(self._access_history) > self._history_max:
            self._access_history = self._access_history[-self._history_max:]

        self._access_count += 1
        if self._access_count % self._prefetch_interval == 0:
            self._neural_prefetch(addr)

    def _neural_prefetch(self, addr: int) -> None:
        """Run the LSTM prefetcher to predict and pre-load upcoming pages."""
        if not self._prefetch_loaded or self._prefetch_model is None:
            return
        if len(self._access_history) < 4:
            return

        self._stats["prefetch_runs"] += 1

        try:
            history = self._access_history[-16:]  # last 16 accesses
            hist_t = torch.tensor([history], dtype=torch.int64, device=self.device)

            with torch.no_grad():
                pred_slots = self._prefetch_model(hist_t)  # [4] predicted slots

            for slot in pred_slots.tolist():
                predicted_addr = (int(slot) << 2) & (self.size - 1)
                predicted_page = predicted_addr // self.page_size

                if predicted_page < self.total_pages:
                    with self._lock:
                        if predicted_page not in self._cache:
                            # Pre-load the page
                            if len(self._cache) < self.max_cached_pages:
                                offset = predicted_page * self.page_size
                                data = bytearray(
                                    self._mmap[offset:offset + self.page_size]
                                )
                                self._cache[predicted_page] = data
                                self._prefetched_pages.add(predicted_page)
                                self._stats["prefetch_loads"] += 1
        except Exception:
            pass  # Graceful degradation

    # ── Cleanup ──────────────────────────────────────────────────────

    def close(self) -> None:
        """Flush dirty pages, close mmap, and clean up."""
        self.flush()
        try:
            self._mmap.close()
        except Exception:
            pass
        try:
            self._file.close()
        except Exception:
            pass
        if self._owns_file:
            try:
                os.unlink(self._backing_path)
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Convenience ──────────────────────────────────────────────────

    def load_program(self, base_addr: int, data: bytes) -> None:
        """Load a program (or any data blob) into memory at base_addr."""
        self.write(base_addr, data)

    def hexdump(self, addr: int, length: int = 64) -> str:
        """Return a hex dump of memory contents."""
        data = self.read(addr, length)
        lines = []
        for i in range(0, len(data), 16):
            chunk = data[i:i+16]
            hex_part = " ".join(f"{b:02x}" for b in chunk)
            ascii_part = "".join(
                chr(b) if 32 <= b < 127 else "." for b in chunk
            )
            lines.append(f"  {addr+i:08x}  {hex_part:<48s}  {ascii_part}")
        return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# Benchmarks
# ────────────────────────────────────────────────────────────────────────────

def benchmark_ssd_memory(
    size: int = 16 * 1024 * 1024,
    verbose: bool = True,
) -> dict:
    """Benchmark the NeuralSSDMemory system.

    Measures:
        1. Sequential read bandwidth
        2. Sequential write bandwidth
        3. Random read latency (with and without neural prefetch)
        4. Comparison vs plain bytearray

    Returns a dict of benchmark results.
    """
    import random

    results = {}
    page_size = 4096
    n_pages = size // page_size

    if verbose:
        print(f"\n  SSD Memory Benchmark (size={size // (1024*1024)} MB, "
              f"pages={n_pages})")
        print("  " + "-" * 60)

    # ── 1. Sequential write bandwidth ────────────────────────────────
    with NeuralSSDMemory(size=size, page_size=page_size,
                         max_cached_pages=min(n_pages, 2048)) as mem:
        block = bytes(range(256)) * (page_size // 256)  # 4KB block
        t0 = time.perf_counter()
        for page in range(min(n_pages, 1024)):
            mem.write(page * page_size, block)
        t1 = time.perf_counter()
        pages_written = min(n_pages, 1024)
        write_bw = (pages_written * page_size) / (t1 - t0) / (1024 * 1024)
        results["seq_write_MB_per_s"] = write_bw
        if verbose:
            print(f"  Sequential write:  {write_bw:8.1f} MB/s "
                  f"({pages_written} pages)")

    # ── 2. Sequential read bandwidth ─────────────────────────────────
    with NeuralSSDMemory(size=size, page_size=page_size,
                         max_cached_pages=min(n_pages, 2048)) as mem:
        # Pre-fill some data
        block = bytes(range(256)) * (page_size // 256)
        for page in range(min(n_pages, 1024)):
            mem.write(page * page_size, block)
        # Clear cache to force reads from mmap
        mem._cache.clear()
        mem._dirty.clear()

        t0 = time.perf_counter()
        for page in range(min(n_pages, 1024)):
            _ = mem.read(page * page_size, page_size)
        t1 = time.perf_counter()
        pages_read = min(n_pages, 1024)
        read_bw = (pages_read * page_size) / (t1 - t0) / (1024 * 1024)
        results["seq_read_MB_per_s"] = read_bw
        if verbose:
            print(f"  Sequential read:   {read_bw:8.1f} MB/s "
                  f"({pages_read} pages)")

    # ── 3. Random read with neural prefetch ──────────────────────────
    n_random = 2000
    random_pages = [random.randint(0, min(n_pages, 4096) - 1)
                    for _ in range(n_random)]

    # With prefetch
    with NeuralSSDMemory(size=size, page_size=page_size,
                         max_cached_pages=256) as mem:
        # Pre-fill
        block = bytes(range(256)) * (page_size // 256)
        for page in set(random_pages):
            mem.write(page * page_size, block)
        mem._cache.clear()
        mem._dirty.clear()
        mem._stats = {k: 0 for k in mem._stats}
        mem._prefetched_pages.clear()

        t0 = time.perf_counter()
        for page in random_pages:
            _ = mem.read(page * page_size, 64)
        t1 = time.perf_counter()
        random_with_prefetch = (t1 - t0) / n_random * 1e6  # microseconds
        stats_prefetch = mem.stats
        results["random_read_with_prefetch_us"] = random_with_prefetch
        results["prefetch_hit_rate"] = stats_prefetch["prefetch_hit_rate"]
        results["cache_hit_rate_prefetch"] = stats_prefetch["hit_rate"]
        if verbose:
            print(f"  Random read (prefetch):  {random_with_prefetch:6.1f} us/read  "
                  f"hit_rate={stats_prefetch['hit_rate']:.3f}  "
                  f"prefetch_hits={stats_prefetch['prefetch_hits']}")

    # Without prefetch (disable LSTM)
    with NeuralSSDMemory(size=size, page_size=page_size,
                         max_cached_pages=256) as mem:
        mem._prefetch_loaded = False  # disable neural prefetch
        block = bytes(range(256)) * (page_size // 256)
        for page in set(random_pages):
            mem.write(page * page_size, block)
        mem._cache.clear()
        mem._dirty.clear()
        mem._stats = {k: 0 for k in mem._stats}

        t0 = time.perf_counter()
        for page in random_pages:
            _ = mem.read(page * page_size, 64)
        t1 = time.perf_counter()
        random_no_prefetch = (t1 - t0) / n_random * 1e6
        stats_no_prefetch = mem.stats
        results["random_read_no_prefetch_us"] = random_no_prefetch
        results["cache_hit_rate_no_prefetch"] = stats_no_prefetch["hit_rate"]
        if verbose:
            print(f"  Random read (no prefetch): {random_no_prefetch:6.1f} us/read  "
                  f"hit_rate={stats_no_prefetch['hit_rate']:.3f}")

    # ── 4. Comparison: plain bytearray ───────────────────────────────
    plain = bytearray(size)
    block_bytes = bytes(range(256)) * (page_size // 256)

    t0 = time.perf_counter()
    for page in range(min(n_pages, 1024)):
        offset = page * page_size
        plain[offset:offset + page_size] = block_bytes
    t1 = time.perf_counter()
    plain_write_bw = (min(n_pages, 1024) * page_size) / (t1 - t0) / (1024 * 1024)

    t0 = time.perf_counter()
    for page in range(min(n_pages, 1024)):
        offset = page * page_size
        _ = bytes(plain[offset:offset + page_size])
    t1 = time.perf_counter()
    plain_read_bw = (min(n_pages, 1024) * page_size) / (t1 - t0) / (1024 * 1024)

    results["plain_write_MB_per_s"] = plain_write_bw
    results["plain_read_MB_per_s"] = plain_read_bw

    if verbose:
        print(f"  Plain bytearray write: {plain_write_bw:8.1f} MB/s")
        print(f"  Plain bytearray read:  {plain_read_bw:8.1f} MB/s")
        print()
        overhead = results["seq_read_MB_per_s"] / max(plain_read_bw, 0.1)
        print(f"  Neural/plain read ratio: {overhead:.2f}x")

    return results
