#!/usr/bin/env python3
"""
Rust Metal Runner — Drop-in replacement for MLXKernelCPUv2 backed by FullARM64CPU.

Uses Rust + Metal with StorageModeShared (zero-copy) for ~290x faster SVC handling.
The Python MLX runner copies 16MB per SVC; this copies 0 bytes.

Usage:
    from kernels.mlx.rust_runner import RustMetalCPU, StopReasonV2, ExecutionResultV2
    cpu = RustMetalCPU(quiet=True)
    cpu.load_program(binary, address=0x10000)
    cpu.set_pc(0x10000)
    result = cpu.execute(max_cycles=100_000)
"""

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
import ncpu_metal

from .instruction_coverage import (
    analyze_instruction_coverage as analyze_instruction_coverage_impl,
    is_known_instruction,
)

# ═══════════════════════════════════════════════════════════════════════════════
# STOP REASONS (matching MLXKernelCPUv2 interface)
# ═══════════════════════════════════════════════════════════════════════════════

class StopReasonV2(IntEnum):
    RUNNING = 0
    HALT = 1
    SYSCALL = 2
    MAX_CYCLES = 3
    BREAKPOINT = 4
    WATCHPOINT = 5

    @property
    def name_str(self) -> str:
        return ["RUNNING", "HALT", "SYSCALL", "MAX_CYCLES", "BREAKPOINT", "WATCHPOINT"][self]


@dataclass
class ExecutionResultV2:
    cycles: int
    elapsed_seconds: float
    stop_reason: StopReasonV2
    final_pc: int

    @property
    def ips(self) -> float:
        if self.elapsed_seconds > 0:
            return self.cycles / self.elapsed_seconds
        return 0.0

    @property
    def stop_reason_name(self) -> str:
        return self.stop_reason.name_str


# Signal → StopReason mapping
_SIGNAL_TO_STOP = {
    0: StopReasonV2.RUNNING,     # SIGNAL_RUNNING → keep going
    1: StopReasonV2.HALT,        # SIGNAL_HALT
    2: StopReasonV2.SYSCALL,     # SIGNAL_SYSCALL
    3: StopReasonV2.MAX_CYCLES,  # SIGNAL_CHECKPOINT → batch exhausted
    4: StopReasonV2.BREAKPOINT,  # SIGNAL_BREAKPOINT → hit breakpoint
    5: StopReasonV2.WATCHPOINT,  # SIGNAL_WATCHPOINT → memory changed
}


# ═══════════════════════════════════════════════════════════════════════════════
# RUST METAL CPU — Drop-in replacement for MLXKernelCPUv2
# ═══════════════════════════════════════════════════════════════════════════════

class RustMetalCPU:
    """
    ARM64 CPU emulator running entirely on Metal GPU via Rust.

    Zero-copy memory access via StorageModeShared — no 16MB copies per SVC.
    Full 139-instruction ARM64 ISA, GPU-side SVC write buffer.

    Drop-in replacement for MLXKernelCPUv2.
    """

    SVC_BUF_BASE = 0x3F0000
    SVC_BUF_HDR = 16
    SVC_BUF_DATA = SVC_BUF_BASE + SVC_BUF_HDR
    HEAP_BASE = 0x60000

    def __init__(self, memory_size: int = 16 * 1024 * 1024, quiet: bool = False,
                 batch_size: int = 10_000_000):
        self._cpu = ncpu_metal.FullARM64CPU(
            memory_size=memory_size,
            cycles_per_batch=batch_size,
        )
        self._memory_size = memory_size
        self._pc = 0
        self.total_instructions = 0
        self.total_syscalls = 0

        if not quiet:
            print(f"[RustMetalCPU] Initialized with {memory_size:,} bytes memory")
            print(f"[RustMetalCPU] Zero-copy StorageModeShared — ~290x faster SVC")

    @property
    def memory_size(self) -> int:
        return self._memory_size

    @property
    def pc(self) -> int:
        return self._cpu.get_pc()

    def set_pc(self, value: int) -> None:
        self._cpu.set_pc(value & 0xFFFFFFFFFFFFFFFF)

    def get_register(self, reg: int) -> int:
        if reg == 31:
            return 0  # XZR always reads as zero
        return self._cpu.get_register(reg)

    def set_register(self, reg: int, value: int) -> None:
        # Convert large unsigned to signed i64 for Rust
        if value >= (1 << 63):
            value -= (1 << 64)
        self._cpu.set_register(reg, value)

    def read_memory(self, address: int, size: int) -> bytes:
        address = address & 0xFFFFFFFF  # Mask to 32-bit address space
        if address + size > self._memory_size:
            # Return zeros for out-of-bounds reads (match Python behavior)
            return b'\x00' * size
        return bytes(self._cpu.read_memory(address, size))

    def read_memory_64(self, address: int) -> int:
        """Read 64-bit value from memory (little-endian)."""
        data = self.read_memory(address, 8)
        return int.from_bytes(data, 'little')

    def write_memory_64(self, address: int, value: int) -> None:
        """Write 64-bit value to memory (little-endian)."""
        self.write_memory(address, value.to_bytes(8, 'little'))

    def write_memory(self, address: int, data: bytes) -> None:
        address = address & 0xFFFFFFFF  # Mask to 32-bit address space
        if address + len(data) > self._memory_size:
            # Silently truncate out-of-bounds writes (match Python behavior)
            if address >= self._memory_size:
                return
            data = data[:self._memory_size - address]
        data_list = list(data) if isinstance(data, (bytes, bytearray)) else list(bytes(data))
        self._cpu.write_memory(address, data_list)

    def load_program(self, binary, address: int = 0x10000) -> None:
        if isinstance(binary, (bytes, bytearray)):
            self._cpu.load_program(list(binary), address)
        elif isinstance(binary, list) and binary and isinstance(binary[0], int) and binary[0] > 255:
            # List of 32-bit instruction words — convert to bytes
            data = b''.join(w.to_bytes(4, 'little') for w in binary)
            self._cpu.load_program(list(data), address)
        else:
            self._cpu.load_program(list(bytes(binary)), address)

    def init_svc_buffer(self) -> None:
        """Initialize the GPU-side SVC write buffer."""
        # Clear the buffer header (pos + count)
        self._cpu.write_memory(self.SVC_BUF_BASE, list(bytes(self.SVC_BUF_HDR)))
        # Set initial brk address
        brk_bytes = struct.pack('<Q', self.HEAP_BASE)
        self._cpu.write_memory(self.SVC_BUF_BASE + 8, list(brk_bytes))

    def drain_svc_buffer(self) -> list[tuple[int, bytes]]:
        """Drain GPU-buffered SYS_WRITE entries. Returns [(fd, data), ...]."""
        return self._cpu.drain_svc_buffer()

    def execute(self, max_cycles: int = 100_000) -> ExecutionResultV2:
        """Execute up to max_cycles instructions on GPU."""
        result = self._cpu.execute(max_cycles=max_cycles)

        stop = _SIGNAL_TO_STOP.get(result.signal, StopReasonV2.RUNNING)

        self.total_instructions += result.total_cycles

        return ExecutionResultV2(
            cycles=result.total_cycles,
            elapsed_seconds=result.elapsed_seconds,
            stop_reason=stop,
            final_pc=result.final_pc,
        )

    def get_registers_numpy(self) -> np.ndarray:
        """Return the raw architectural register file for context switching.

        The Metal kernel stores SP in slot 31. User-facing `get_register(31)`
        still exposes XZR semantics, but scheduler snapshots must preserve the
        raw SP value or every syscall/context switch will corrupt the stack.
        """
        regs = np.empty(32, dtype=np.int64)
        for i in range(32):
            regs[i] = self._cpu.get_register(i)
        return regs

    def set_registers_numpy(self, values: np.ndarray) -> None:
        """Bulk-set all 32 registers from a numpy int64 array."""
        for i in range(32):
            self._cpu.set_register(i, int(values[i]))

    def analyze_instruction_coverage(self, text_base: int, text_size: int) -> list[dict]:
        """Analyze a code region for instruction coverage gaps."""
        return analyze_instruction_coverage_impl(self.read_memory, text_base, text_size)

    @staticmethod
    def _is_known_instruction(inst: int, top: int) -> bool:
        """Check if an instruction is handled by the Metal kernel decode table."""
        return is_known_instruction(inst, top)

    def get_flags(self) -> tuple:
        """Return (N, Z, C, V) as 4 bools."""
        return (
            bool(self._cpu.get_flag(0)),  # N
            bool(self._cpu.get_flag(1)),  # Z
            bool(self._cpu.get_flag(2)),  # C
            bool(self._cpu.get_flag(3)),  # V
        )

    def set_flags(self, n: bool, z: bool, c: bool, v: bool) -> None:
        """Set NZCV flags from 4 bools."""
        self._cpu.set_flag(0, n)
        self._cpu.set_flag(1, z)
        self._cpu.set_flag(2, c)
        self._cpu.set_flag(3, v)

    # ═══════════════════════════════════════════════════════════════════════════
    # GPU-SIDE INSTRUCTION TRACE BUFFER
    # ═══════════════════════════════════════════════════════════════════════════

    def enable_trace(self) -> None:
        """Enable instruction tracing (set trace_count = 0)."""
        self._cpu.enable_trace()

    def disable_trace(self) -> None:
        """Disable instruction tracing (set trace_count = 0xFFFFFFFF)."""
        self._cpu.disable_trace()

    def read_trace(self) -> list[tuple[int, int, int, int, int, int, int, int]]:
        """
        Read trace buffer entries from GPU memory.

        Returns list of (pc, inst, x0, x1, x2, x3, flags, sp) tuples, oldest first.
        flags is packed NZCV in bits [3:0]. sp is the stack pointer.
        """
        return self._cpu.read_trace()

    def clear_trace(self) -> None:
        """Clear the trace buffer (reset pos and count to 0)."""
        self._cpu.clear_trace()

    def set_breakpoint(self, index: int, pc: int) -> None:
        """Set breakpoint at index (0-3) to fire when PC matches."""
        self._cpu.set_breakpoint(index, pc)

    def clear_breakpoints(self) -> None:
        """Remove all breakpoints."""
        self._cpu.clear_breakpoints()

    def set_conditional_breakpoint(self, index: int, pc: int, reg: int, value: int) -> None:
        """Set conditional breakpoint: fires at PC only when regs[reg] == value."""
        self._cpu.set_conditional_breakpoint(index, pc, reg, value)

    def set_watchpoint(self, index: int, addr: int) -> None:
        """Set memory watchpoint at index (0-3). Fires when 8B value at addr changes."""
        self._cpu.set_watchpoint(index, addr)

    def clear_watchpoints(self) -> None:
        """Remove all watchpoints."""
        self._cpu.clear_watchpoints()

    def read_watchpoint_info(self) -> Optional[tuple[int, int, int, int]]:
        """Read watchpoint hit info: (wp_index, address, old_value, new_value) or None."""
        return self._cpu.read_watchpoint_info()

    def clear_all_debug(self) -> None:
        """Clear all debug state: breakpoints, conditions, watchpoints."""
        self._cpu.clear_all_debug()

    def reset(self) -> None:
        self._cpu.reset()
        self.total_instructions = 0
        self.total_syscalls = 0

    def full_reset(self) -> None:
        """Reset all state INCLUDING zeroing main memory."""
        self._cpu.clear_memory(0, self._memory_size)
        self.reset()


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED INSTANCE POOL — Reuse Metal device/pipeline across calls
# ═══════════════════════════════════════════════════════════════════════════════

_shared_instances: dict[tuple[int, int], 'RustMetalCPU'] = {}


def get_shared_cpu(memory_size: int = 16 * 1024 * 1024,
                   batch_size: int = 10_000_000) -> RustMetalCPU:
    """
    Get a shared RustMetalCPU instance, reusing Metal device/pipeline/queue.

    Avoids the overhead of re-compiling the Metal shader and re-allocating
    GPU buffers for every call. The instance is fully reset before returning.
    Useful for tests and benchmarks that create many short-lived CPU instances.
    """
    key = (memory_size, batch_size)
    cpu = _shared_instances.get(key)
    if cpu is not None:
        cpu.full_reset()
        return cpu
    cpu = RustMetalCPU(memory_size=memory_size, quiet=True, batch_size=batch_size)
    _shared_instances[key] = cpu
    return cpu
