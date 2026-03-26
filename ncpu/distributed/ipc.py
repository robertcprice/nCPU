"""Inter-GPU communication primitives for multi-core nCPU.

Provides shared memory regions and message channels that enable cores to
exchange data during distributed execution. On Apple Silicon unified memory,
shared regions map to the same physical memory. On discrete GPUs, these
abstractions would sit atop CUDA IPC or NCCL collective operations.

Two communication models:
  - SharedMemoryRegion: random-access shared state with soft locking and
    atomic read-modify-write (cache-coherence analogy).
  - MessageChannel: point-to-point ring buffer between two cores, analogous
    to a hardware FIFO or UNIX pipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Message dataclass
# ---------------------------------------------------------------------------

@dataclass
class IPCMessage:
    """A typed message between GPU cores.

    Attributes:
        sender: Source core id.
        receiver: Destination core id.
        data: Tensor payload (arbitrary shape).
        tag: User-defined tag for multiplexing message types on the same
             channel (0 = default / untagged).
    """

    sender: int
    receiver: int
    data: torch.Tensor
    tag: int = 0


# ---------------------------------------------------------------------------
# Shared memory
# ---------------------------------------------------------------------------

class SharedMemoryRegion:
    """A shared memory region accessible by multiple cores.

    On Apple Silicon unified memory, this is literally the same physical
    memory. On discrete GPUs, this would use CUDA IPC or similar.

    The region provides:
      - Indexed read / write of contiguous slices.
      - Atomic add (read-modify-write with soft locking).
      - Compare-and-swap for lock-free synchronisation patterns.
      - Size introspection.
    """

    def __init__(self, size: int = 4096, name: str = "shared") -> None:
        self.name = name
        self.size = size
        self.memory = torch.zeros(size)
        self.locks = torch.zeros(size)  # soft locks for synchronisation

    # -- basic access -------------------------------------------------------

    def read(self, address: int, length: int = 1) -> torch.Tensor:
        """Read *length* contiguous words starting at *address*.

        Returns a **clone** so the caller cannot silently mutate the region.
        """
        if address < 0 or address + length > self.size:
            raise IndexError(
                f"read({address}, {length}) out of bounds for region "
                f"'{self.name}' (size={self.size})"
            )
        return self.memory[address : address + length].clone()

    def write(self, address: int, data: torch.Tensor) -> None:
        """Write *data* into the region starting at *address*."""
        length = data.numel()
        if address < 0 or address + length > self.size:
            raise IndexError(
                f"write({address}, len={length}) out of bounds for region "
                f"'{self.name}' (size={self.size})"
            )
        self.memory[address : address + length] = data.flatten()

    # -- atomic operations --------------------------------------------------

    def atomic_add(self, address: int, value: torch.Tensor) -> torch.Tensor:
        """Atomic add: returns the old value, stores old + value.

        This is a *software* atomic (single-threaded host); it exists so the
        API mirrors what a real multi-device implementation would need.
        """
        if address < 0 or address >= self.size:
            raise IndexError(
                f"atomic_add({address}) out of bounds for region "
                f"'{self.name}' (size={self.size})"
            )
        old = self.memory[address].clone()
        self.memory[address] = old + value
        return old

    def compare_and_swap(
        self, address: int, expected: torch.Tensor, desired: torch.Tensor
    ) -> tuple[bool, torch.Tensor]:
        """Compare-and-swap: if mem[address] == expected, set to desired.

        Returns (success: bool, old_value: Tensor).
        """
        if address < 0 or address >= self.size:
            raise IndexError(
                f"compare_and_swap({address}) out of bounds for region "
                f"'{self.name}' (size={self.size})"
            )
        old = self.memory[address].clone()
        if torch.equal(old, expected):
            self.memory[address] = desired
            return True, old
        return False, old

    # -- bulk operations ----------------------------------------------------

    def clear(self) -> None:
        """Zero the entire region."""
        self.memory.zero_()
        self.locks.zero_()

    def snapshot(self) -> torch.Tensor:
        """Return a full clone of the memory contents."""
        return self.memory.clone()

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"SharedMemoryRegion(name='{self.name}', size={self.size})"


# ---------------------------------------------------------------------------
# Point-to-point message channel (ring buffer)
# ---------------------------------------------------------------------------

class MessageChannel:
    """A communication channel between two cores (like a UNIX pipe).

    Implemented as a fixed-size ring buffer.  The channel is unidirectional:
    one core writes (``send``), the other reads (``recv``).

    Attributes:
        buffer_size: Capacity of the ring buffer in words.
        read_ptr / write_ptr: Current head / tail positions.
    """

    def __init__(self, buffer_size: int = 256) -> None:
        if buffer_size < 2:
            raise ValueError("buffer_size must be >= 2")
        self.buffer = torch.zeros(buffer_size)
        self.read_ptr: int = 0
        self.write_ptr: int = 0
        self.buffer_size: int = buffer_size

    # -- write side ---------------------------------------------------------

    def send(self, data: torch.Tensor) -> bool:
        """Write *data* into the channel.

        Returns True if all values were written, False if the channel is
        full (partial writes never happen -- it is all-or-nothing).
        """
        flat = data.flatten()
        # Pre-check: enough room for the entire payload?
        if flat.numel() > self.available_space():
            return False
        for val in flat:
            self.buffer[self.write_ptr] = val
            self.write_ptr = (self.write_ptr + 1) % self.buffer_size
        return True

    # -- read side ----------------------------------------------------------

    def recv(self, count: int = 1) -> Optional[torch.Tensor]:
        """Read *count* values from the channel.

        Returns None if fewer than *count* values are available.
        """
        if self.available() < count:
            return None
        result = []
        for _ in range(count):
            result.append(self.buffer[self.read_ptr].item())
            self.read_ptr = (self.read_ptr + 1) % self.buffer_size
        return torch.tensor(result)

    # -- status -------------------------------------------------------------

    def available(self) -> int:
        """Number of values available to read."""
        return (self.write_ptr - self.read_ptr) % self.buffer_size

    def available_space(self) -> int:
        """Number of values that can still be written before the buffer is full.

        One slot is always kept empty to distinguish full from empty.
        """
        return (self.buffer_size - 1) - self.available()

    def is_empty(self) -> bool:
        return self.read_ptr == self.write_ptr

    def is_full(self) -> bool:
        return self.available_space() == 0

    def reset(self) -> None:
        """Drain the channel and reset pointers."""
        self.buffer.zero_()
        self.read_ptr = 0
        self.write_ptr = 0

    def __repr__(self) -> str:
        return (
            f"MessageChannel(buffer_size={self.buffer_size}, "
            f"used={self.available()}, free={self.available_space()})"
        )


# ---------------------------------------------------------------------------
# Barrier synchronization
# ---------------------------------------------------------------------------


class Barrier:
    """Distributed barrier for synchronizing GPU cores.

    All cores must reach the barrier before any can proceed.
    Supports BSP (Bulk Synchronous Parallel) execution where:

    1. Cores compute independently (superstep).
    2. All cores synchronize at the barrier.
    3. Communication happens during the barrier phase.
    4. The next superstep begins.

    The barrier tracks a *generation* counter that increments each time
    every participant has arrived.  This enables callers to distinguish
    successive barrier completions and to register callbacks that execute
    at the synchronization point (e.g. flush shared memory, exchange
    messages, log metrics).

    Thread safety: this is a software barrier operating on the single-
    threaded host.  On a real multi-device system the implementation
    would use NCCL all-reduce or similar collective primitives, but the
    API would remain identical.
    """

    def __init__(self, num_participants: int) -> None:
        if num_participants < 1:
            raise ValueError(
                f"num_participants must be >= 1, got {num_participants}"
            )
        self.num_participants = num_participants
        self.arrived: set[int] = set()
        self.generation: int = 0  # increments each time barrier completes
        self.callbacks: list[callable] = []  # run when barrier triggers

    def arrive(self, core_id: int) -> bool:
        """Signal that *core_id* has reached the barrier.

        Returns True when all participants have arrived (the barrier
        fires), False otherwise.  When the barrier fires:
        1. The generation counter increments.
        2. All registered callbacks execute in registration order.
        3. The arrived set is cleared for the next generation.

        It is an error for the same core to arrive twice in the same
        generation; the duplicate is silently ignored (set semantics).
        """
        self.arrived.add(core_id)
        if len(self.arrived) == self.num_participants:
            self.generation += 1
            for cb in self.callbacks:
                cb(self.generation)
            self.arrived.clear()
            return True
        return False

    def on_complete(self, callback: callable) -> None:
        """Register a callback invoked when the barrier fires.

        The callback receives the new generation number as its sole
        argument: ``callback(generation: int) -> None``.
        """
        self.callbacks.append(callback)

    def reset(self) -> None:
        """Reset the barrier to its initial state."""
        self.arrived.clear()
        self.generation = 0
        self.callbacks.clear()

    def __repr__(self) -> str:
        return (
            f"Barrier(participants={self.num_participants}, "
            f"arrived={len(self.arrived)}, generation={self.generation})"
        )
