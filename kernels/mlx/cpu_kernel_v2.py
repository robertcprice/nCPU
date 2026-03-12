#!/usr/bin/env python3
"""
MLX Metal Kernel V2 - Double-Buffer Memory Architecture.

LEGACY: This module is superseded by kernels.mlx.rust_runner (RustMetalCPU)
which uses Rust + Metal with StorageModeShared for ~500x faster compilation.
Use RustMetalCPU for new code. This module is retained for backwards compatibility.

This version adds FULL MEMORY WRITE SUPPORT via double-buffering:
- memory_in: Read-only input (initial state)
- memory_out: Writable output (copied from input, then modified)
- After kernel returns, memory_out becomes memory_in for next call

ZERO Python interaction during GPU execution - only sync at halt/syscall.

PERFORMANCE:
============
Target: Same ~2M IPS as V1, but with memory write capability.
The 4MB memory copy at kernel start is GPU-to-GPU (~0.4ms),
amortized over millions of cycles = negligible per-instruction cost.

Author: nCPU Project
Date: 2024
"""

import mlx.core as mx
import numpy as np
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Union

from .instruction_coverage import (
    analyze_instruction_coverage as analyze_instruction_coverage_impl,
    is_known_instruction,
)
from .cpu_kernel_v2_source import (
    KERNEL_HEADER_V2,
    KERNEL_SOURCE_V2,
    STOP_RUNNING,
    STOP_HALT,
    STOP_SYSCALL,
    STOP_MAX_CYCLES,
    STOP_BREAKPOINT,
)


# ═══════════════════════════════════════════════════════════════════════════════
# STOP REASON ENUM
# ═══════════════════════════════════════════════════════════════════════════════

class StopReasonV2(IntEnum):
    """Reasons why kernel execution stopped."""
    RUNNING = STOP_RUNNING
    HALT = STOP_HALT
    SYSCALL = STOP_SYSCALL
    MAX_CYCLES = STOP_MAX_CYCLES
    BREAKPOINT = STOP_BREAKPOINT

    @property
    def name_str(self) -> str:
        return ["RUNNING", "HALT", "SYSCALL", "MAX_CYCLES", "BREAKPOINT"][self]


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutionResultV2:
    """Result of kernel execution."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# MLX KERNEL CPU V2 - Double-Buffer Memory
# ═══════════════════════════════════════════════════════════════════════════════

class MLXKernelCPUv2:
    """
    ARM64 CPU Emulator V2 - with full memory read/write support.

    Uses double-buffering: kernel reads from memory_in, writes to memory_out.
    After execution, memory_out becomes the new memory for the next call.
    """

    def __init__(self, memory_size: int = 16 * 1024 * 1024, quiet: bool = False):
        """Initialize CPU with given memory size."""
        self.memory_size = memory_size

        # ── Shadow numpy arrays (Python-side fast path) ──
        # All Python-side reads/writes go through numpy. Only sync to MLX
        # at execute() boundaries. This eliminates per-SVC 16MB round-trips.
        # Lazy sync: GPU→numpy memory transfer is deferred until first read.
        self._memory_np = np.zeros(memory_size, dtype=np.uint8)
        self._registers_np = np.zeros(32, dtype=np.int64)
        self._flags_np = np.zeros(4, dtype=np.float32)
        self._memory_dirty = True    # numpy has changes not yet synced to MLX
        self._memory_np_stale = False # numpy doesn't have GPU's latest memory
        self._registers_dirty = True
        self._flags_dirty = True

        # MLX arrays (GPU-side, synced lazily from numpy at execute() time)
        self.memory = mx.array(self._memory_np, dtype=mx.uint8)
        self.registers = mx.array(self._registers_np, dtype=mx.int64)
        self.flags = mx.array(self._flags_np, dtype=mx.float32)

        # Program Counter
        self._pc = 0

        # SIMD/FP registers V0-V31 (128-bit each, stored as hi:lo int64 pairs)
        self.vreg_lo = mx.zeros(32, dtype=mx.int64)
        self.vreg_hi = mx.zeros(32, dtype=mx.int64)

        # Compile the V2 kernel with memory_out support
        self._kernel = mx.fast.metal_kernel(
            name="arm64_cpu_execute_v2",
            input_names=["memory_in", "registers_in", "pc_in", "flags_in",
                        "max_cycles_in", "memory_size_in",
                        "vreg_lo_in", "vreg_hi_in"],
            output_names=["memory_out", "registers_out", "pc_out", "flags_out",
                         "cycles_out", "stop_reason_out",
                         "vreg_lo_out", "vreg_hi_out"],
            source=KERNEL_SOURCE_V2,
            header=KERNEL_HEADER_V2,
            ensure_row_contiguous=True,
            atomic_outputs=False,
        )

        # Statistics
        self.total_instructions = 0
        self.total_syscalls = 0

        if not quiet:
            print(f"[MLXKernelCPUv2] Initialized with {memory_size:,} bytes memory")
            print(f"[MLXKernelCPUv2] Shadow numpy acceleration enabled")
            print(f"[MLXKernelCPUv2] STR/STRB memory writes SUPPORTED!")

    # ═══════════════════════════════════════════════════════════════════════════
    # PROPERTY ACCESSORS
    # ═══════════════════════════════════════════════════════════════════════════

    @property
    def pc(self) -> int:
        return self._pc

    def set_pc(self, value: int) -> None:
        self._pc = value & 0xFFFFFFFFFFFFFFFF

    def get_register(self, reg: int) -> int:
        if reg == 31:
            return 0
        return int(self._registers_np[reg])

    def set_register(self, reg: int, value: int) -> None:
        if reg == 31:
            return
        self._registers_np[reg] = value
        self._registers_dirty = True

    def get_registers_numpy(self) -> np.ndarray:
        return self._registers_np.copy()

    def set_registers_numpy(self, values: np.ndarray) -> None:
        self._registers_np[:] = values
        self._registers_dirty = True

    def get_flags(self) -> tuple[bool, bool, bool, bool]:
        f = self._flags_np
        return (f[0] > 0.5, f[1] > 0.5, f[2] > 0.5, f[3] > 0.5)

    def set_flags(self, n: bool, z: bool, c: bool, v: bool) -> None:
        self._flags_np[0] = 1.0 if n else 0.0
        self._flags_np[1] = 1.0 if z else 0.0
        self._flags_np[2] = 1.0 if c else 0.0
        self._flags_np[3] = 1.0 if v else 0.0
        self._flags_dirty = True

    # ═══════════════════════════════════════════════════════════════════════════
    # MEMORY OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════

    def _ensure_memory_synced(self) -> None:
        """Lazy sync: pull GPU memory to numpy if stale."""
        if self._memory_np_stale:
            self._memory_np = np.array(self.memory)
            self._memory_np_stale = False

    def read_memory(self, address: int, size: int) -> bytes:
        self._ensure_memory_synced()
        end = min(address + size, self.memory_size)
        return bytes(self._memory_np[address:end])

    def write_memory(self, address: int, data: bytes) -> None:
        self._ensure_memory_synced()
        data_array = np.frombuffer(data, dtype=np.uint8)
        self._memory_np[address:address + len(data)] = data_array
        self._memory_dirty = True

    def load_program(self, program: Union[bytes, list[int]], address: int = 0) -> None:
        if isinstance(program, list):
            data = b''
            for inst in program:
                data += inst.to_bytes(4, 'little')
            program = data
        self.write_memory(address, program)
        print(f"[MLXKernelCPUv2] Loaded {len(program):,} bytes at 0x{address:X}")

    def clear_memory(self) -> None:
        self._memory_np = np.zeros(self.memory_size, dtype=np.uint8)
        self._memory_np_stale = False
        self._memory_dirty = True

    def read_memory_64(self, address: int) -> int:
        """Read 64-bit value from memory (little-endian)."""
        data = self.read_memory(address, 8)
        return int.from_bytes(data, 'little')

    def write_memory_64(self, address: int, value: int) -> None:
        """Write 64-bit value to memory (little-endian)."""
        self.write_memory(address, value.to_bytes(8, 'little'))

    # ═══════════════════════════════════════════════════════════════════════════
    # GPU-SIDE SVC BUFFER
    # ═══════════════════════════════════════════════════════════════════════════

    SVC_BUF_BASE = 0x3F0000
    SVC_BUF_HDR = 16
    SVC_BUF_DATA = SVC_BUF_BASE + SVC_BUF_HDR

    def init_svc_buffer(self) -> None:
        """Initialize the GPU-side SVC write buffer. Call before first execute()."""
        # Zero the header: write_pos=0, entry_count=0, brk_addr=0
        self._memory_np[self.SVC_BUF_BASE:self.SVC_BUF_BASE + self.SVC_BUF_HDR] = 0
        self._memory_dirty = True

    def drain_svc_buffer(self) -> list[tuple[int, bytes]]:
        """
        Drain buffered SYS_WRITE entries from GPU memory.

        Returns list of (fd, data) tuples. Resets buffer for next execute().
        Uses targeted MLX read to avoid full 16MB memory sync when buffer empty.
        """
        # Quick check: read just the 8-byte header from MLX (avoids full sync)
        hdr = np.array(self.memory[self.SVC_BUF_BASE:self.SVC_BUF_BASE + 8])
        entry_count = int(hdr[4]) | (int(hdr[5]) << 8) | (int(hdr[6]) << 16) | (int(hdr[7]) << 24)

        if entry_count == 0:
            return []

        # Has entries — read the buffer data region from MLX
        write_pos = int(hdr[0]) | (int(hdr[1]) << 8) | (int(hdr[2]) << 16) | (int(hdr[3]) << 24)
        buf_data = np.array(self.memory[self.SVC_BUF_DATA:self.SVC_BUF_DATA + write_pos])

        entries = []
        offset = 0
        for _ in range(entry_count):
            fd = int(buf_data[offset])
            length = int(buf_data[offset + 1]) | (int(buf_data[offset + 2]) << 8)
            data = bytes(buf_data[offset + 3:offset + 3 + length])
            entries.append((fd, data))
            offset += 3 + length

        # Reset buffer header in numpy (will be synced at next execute)
        self._ensure_memory_synced()
        self._memory_np[self.SVC_BUF_BASE:self.SVC_BUF_BASE + 8] = 0
        self._memory_dirty = True

        return entries

    # ═══════════════════════════════════════════════════════════════════════════
    # GPU-SIDE INSTRUCTION TRACE BUFFER
    # ═══════════════════════════════════════════════════════════════════════════

    TRACE_BUF_BASE = 0x3B0000
    TRACE_BUF_HDR = 8
    TRACE_ENTRY_SIZE = 56
    TRACE_MAX_ENTRIES = 4096
    DBG_CTRL_BASE = 0x3A0000

    def enable_trace(self) -> None:
        """Enable instruction tracing (set trace_count = 0)."""
        self._ensure_memory_synced()
        self._memory_np[self.TRACE_BUF_BASE:self.TRACE_BUF_BASE + self.TRACE_BUF_HDR] = 0
        self._memory_dirty = True

    def disable_trace(self) -> None:
        """Disable instruction tracing (set trace_count = 0xFFFFFFFF)."""
        self._ensure_memory_synced()
        # Set trace_count = 0xFFFFFFFF to disable
        self._memory_np[self.TRACE_BUF_BASE:self.TRACE_BUF_BASE + 4] = 0  # pos = 0
        self._memory_np[self.TRACE_BUF_BASE + 4:self.TRACE_BUF_BASE + 8] = 0xFF  # count = -1
        self._memory_dirty = True

    def read_trace(self) -> list[tuple[int, int, int, int, int, int, int, int]]:
        """
        Read trace buffer entries from GPU memory.

        Returns list of (pc, inst, x0, x1, x2, x3, flags, sp) tuples.
        flags is a uint32 packed as NZCV in bits [3:0].
        sp is the stack pointer value.
        Tracing must have been enabled before execution.
        """
        # Sync MLX→numpy so we see post-execution GPU writes,
        # and also respect any numpy-side changes (clear_trace, enable_trace)
        self._ensure_memory_synced()
        trace_data = self._memory_np

        hdr = trace_data[self.TRACE_BUF_BASE:self.TRACE_BUF_BASE + 8]
        trace_pos = int(hdr[0]) | (int(hdr[1]) << 8) | (int(hdr[2]) << 16) | (int(hdr[3]) << 24)
        trace_count = int(hdr[4]) | (int(hdr[5]) << 8) | (int(hdr[6]) << 16) | (int(hdr[7]) << 24)

        if trace_count == 0xFFFFFFFF or trace_count == 0:
            return []  # tracing disabled or empty

        # Calculate entries to read
        num_entries = min(trace_count, self.TRACE_MAX_ENTRIES)

        entries = []
        # If buffer wrapped, start from trace_pos; otherwise from 0
        start_idx = trace_pos if trace_count > self.TRACE_MAX_ENTRIES else 0

        for i in range(num_entries):
            idx = (start_idx + i) % self.TRACE_MAX_ENTRIES
            base = self.TRACE_BUF_BASE + self.TRACE_BUF_HDR + idx * self.TRACE_ENTRY_SIZE

            pc = int(trace_data[base]) | (int(trace_data[base+1]) << 8) | \
                 (int(trace_data[base+2]) << 16) | (int(trace_data[base+3]) << 24) | \
                 (int(trace_data[base+4]) << 32) | (int(trace_data[base+5]) << 40) | \
                 (int(trace_data[base+6]) << 48) | (int(trace_data[base+7]) << 56)

            inst = int(trace_data[base+8]) | (int(trace_data[base+9]) << 8) | \
                   (int(trace_data[base+10]) << 16) | (int(trace_data[base+11]) << 24)

            x0 = int.from_bytes(bytes(trace_data[base+12:base+20]), 'little', signed=True)
            x1 = int.from_bytes(bytes(trace_data[base+20:base+28]), 'little', signed=True)
            x2 = int.from_bytes(bytes(trace_data[base+28:base+36]), 'little', signed=True)
            x3 = int.from_bytes(bytes(trace_data[base+36:base+44]), 'little', signed=True)

            flags = int(trace_data[base+44]) | (int(trace_data[base+45]) << 8) | \
                    (int(trace_data[base+46]) << 16) | (int(trace_data[base+47]) << 24)

            sp = int.from_bytes(bytes(trace_data[base+48:base+56]), 'little', signed=True)

            entries.append((pc, inst, x0, x1, x2, x3, flags, sp))

        return entries

    def clear_trace(self) -> None:
        """Clear the trace buffer (reset pos and count to 0)."""
        self._ensure_memory_synced()
        self._memory_np[self.TRACE_BUF_BASE:self.TRACE_BUF_BASE + self.TRACE_BUF_HDR] = 0
        self._memory_dirty = True

    def set_breakpoint(self, index: int, pc: int) -> None:
        """Set breakpoint at index (0-3) to fire at given PC."""
        self._ensure_memory_synced()
        import struct
        bp_offset = self.DBG_CTRL_BASE + 4 + index * 8
        pc_bytes = struct.pack('<Q', pc)
        self._memory_np[bp_offset:bp_offset + 8] = list(pc_bytes)
        # Update count if needed
        current = int.from_bytes(bytes(self._memory_np[self.DBG_CTRL_BASE:self.DBG_CTRL_BASE + 4]), 'little')
        if index + 1 > current:
            count_bytes = struct.pack('<I', index + 1)
            self._memory_np[self.DBG_CTRL_BASE:self.DBG_CTRL_BASE + 4] = list(count_bytes)
        self._memory_dirty = True

    def clear_breakpoints(self) -> None:
        """Remove all breakpoints."""
        self._ensure_memory_synced()
        self._memory_np[self.DBG_CTRL_BASE:self.DBG_CTRL_BASE + 36] = 0
        self._memory_dirty = True

    # ═══════════════════════════════════════════════════════════════════════════
    # EXECUTION
    # ═══════════════════════════════════════════════════════════════════════════

    def execute(self, max_cycles: int = 100000) -> ExecutionResultV2:
        """
        Execute instructions using Metal kernel V2.

        The kernel runs entirely on GPU with full memory read/write support.
        Numpy shadow arrays are synced to MLX only at dispatch boundaries,
        eliminating per-SVC 16MB round-trips.
        """
        start_time = time.perf_counter()

        # ── Sync dirty numpy shadows → MLX (only if Python modified state) ──
        if self._memory_dirty:
            self.memory = mx.array(self._memory_np, dtype=mx.uint8)
            self._memory_dirty = False
        if self._registers_dirty:
            self.registers = mx.array(self._registers_np, dtype=mx.int64)
            self._registers_dirty = False
        if self._flags_dirty:
            self.flags = mx.array(self._flags_np, dtype=mx.float32)
            self._flags_dirty = False

        # Prepare inputs
        pc_array = mx.array([self._pc], dtype=mx.uint64)
        max_cycles_array = mx.array([max_cycles], dtype=mx.uint32)
        memory_size_array = mx.array([self.memory_size], dtype=mx.uint32)

        # Execute kernel - memory_out is a new buffer that kernel writes to
        outputs = self._kernel(
            inputs=[self.memory, self.registers, pc_array, self.flags,
                   max_cycles_array, memory_size_array,
                   self.vreg_lo, self.vreg_hi],
            output_shapes=[
                (self.memory_size,),  # memory_out - FULL MEMORY OUTPUT
                (32,),                 # registers_out
                (1,),                  # pc_out
                (4,),                  # flags_out
                (1,),                  # cycles_out
                (1,),                  # stop_reason_out
                (32,),                 # vreg_lo_out
                (32,),                 # vreg_hi_out
            ],
            output_dtypes=[
                mx.uint8,   # memory_out
                mx.int64,   # registers_out
                mx.uint64,  # pc_out
                mx.float32, # flags_out
                mx.uint32,  # cycles_out
                mx.uint8,   # stop_reason_out
                mx.int64,   # vreg_lo_out
                mx.int64,   # vreg_hi_out
            ],
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
            verbose=False,
        )

        # Force evaluation
        mx.eval(outputs)

        # Unpack outputs
        (memory_out, registers_out, pc_out, flags_out, cycles_out, stop_reason_out,
         vreg_lo_out, vreg_hi_out) = outputs

        # ── Sync MLX outputs → numpy shadows ──
        # Memory: LAZY sync — defer 16MB GPU→numpy until Python reads memory.
        # Registers/flags: sync immediately (small, always needed for SVC args).
        self.memory = memory_out
        self._memory_np_stale = True  # Will sync on first read_memory() call
        self._memory_dirty = False

        self.registers = registers_out
        self._registers_np = np.array(registers_out)
        self._registers_dirty = False

        self.flags = flags_out
        self._flags_np = np.array(flags_out)
        self._flags_dirty = False

        self._pc = int(pc_out[0].item())
        self.vreg_lo = vreg_lo_out
        self.vreg_hi = vreg_hi_out
        cycles = int(cycles_out[0].item())
        stop_reason = StopReasonV2(int(stop_reason_out[0].item()))

        elapsed = time.perf_counter() - start_time

        # Update statistics
        self.total_instructions += cycles
        if stop_reason == StopReasonV2.SYSCALL:
            self.total_syscalls += 1

        return ExecutionResultV2(
            cycles=cycles,
            elapsed_seconds=elapsed,
            stop_reason=stop_reason,
            final_pc=self._pc,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════

    def reset(self) -> None:
        """Reset CPU state."""
        self._registers_np[:] = 0
        self._registers_dirty = True
        self._pc = 0
        self._flags_np[:] = 0.0
        self._flags_dirty = True
        self.total_instructions = 0
        self.total_syscalls = 0

    def dump_registers(self) -> str:
        """Get formatted register dump."""
        regs = self._registers_np
        lines = [f"PC:  0x{self._pc:016X}", ""]
        for i in range(0, 32, 4):
            parts = []
            for j in range(4):
                r = i + j
                name = f"X{r}" if r < 31 else "XZR"
                val = regs[r]
                parts.append(f"{name:3}=0x{val & 0xFFFFFFFFFFFFFFFF:016X}")
            lines.append("  ".join(parts))
        lines.append("")
        n, z, c, v = self.get_flags()
        lines.append(f"Flags: N={int(n)} Z={int(z)} C={int(c)} V={int(v)}")
        return "\n".join(lines)

    def analyze_instruction_coverage(self, text_base: int, text_size: int) -> list[dict]:
        """Analyze a code region for instruction coverage gaps."""
        return analyze_instruction_coverage_impl(self.read_memory, text_base, text_size)

    @staticmethod
    def _is_known_instruction(inst: int, top: int) -> bool:
        """Check if an instruction is handled by the Metal kernel's decode table."""
        return is_known_instruction(inst, top)

    def print_unknown_instructions(self, text_base: int = 0x10000, text_size: int = 0x40000):
        """Analyze and print unknown instructions in the loaded binary."""
        unknowns = self.analyze_instruction_coverage(text_base, text_size)
        if not unknowns:
            print("[diag] No unknown instructions found in code")
            return

        print(f"[diag] {len(unknowns)} unknown instruction class(es):")
        for u in unknowns:
            print(f"  {u['opcode_hex']} at PC=0x{u['pc']:08X} "
                  f"(top={u['top_byte']}, class={u['class_bits']})")


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def create_cpu_v2(memory_size: int = 16 * 1024 * 1024) -> MLXKernelCPUv2:
    """Create a new MLX kernel CPU V2 instance."""
    return MLXKernelCPUv2(memory_size=memory_size)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("MLX KERNEL CPU V2 - MEMORY WRITE TEST")
    print("=" * 70)

    cpu = MLXKernelCPUv2()

    # Test program: Store and Load
    # MOVZ X0, #0x1234        ; X0 = 0x1234
    # MOVZ X1, #0x1000        ; X1 = 0x1000 (address)
    # STR X0, [X1]            ; Store X0 at address 0x1000
    # MOVZ X0, #0             ; Clear X0
    # LDR X2, [X1]            ; Load from 0x1000 into X2
    # HLT

    def encode_movz(rd, imm16, hw=0):
        return 0xD2800000 | (hw << 21) | (imm16 << 5) | rd

    def encode_str_64(rt, rn, imm12=0):
        # STR Xt, [Xn, #imm12*8]
        return 0xF9000000 | (imm12 << 10) | (rn << 5) | rt

    def encode_ldr_64(rt, rn, imm12=0):
        # LDR Xt, [Xn, #imm12*8]
        return 0xF9400000 | (imm12 << 10) | (rn << 5) | rt

    program = [
        encode_movz(0, 0x1234),      # MOVZ X0, #0x1234
        encode_movz(1, 0x1000),      # MOVZ X1, #0x1000
        encode_str_64(0, 1, 0),      # STR X0, [X1]
        encode_movz(0, 0),           # MOVZ X0, #0
        encode_ldr_64(2, 1, 0),      # LDR X2, [X1]
        0xD4400000,                  # HLT
    ]

    cpu.load_program(program, address=0)
    cpu.set_pc(0)

    print("\nExecuting STR/LDR test...")
    result = cpu.execute(max_cycles=100)

    print(f"\nResult: {result.cycles} cycles, {result.stop_reason_name}")
    print(f"X0: 0x{cpu.get_register(0):X} (should be 0)")
    print(f"X1: 0x{cpu.get_register(1):X} (should be 0x1000)")
    print(f"X2: 0x{cpu.get_register(2):X} (should be 0x1234 - loaded from memory!)")

    # Verify memory was actually written
    mem_val = cpu.read_memory_64(0x1000)
    print(f"Memory[0x1000]: 0x{mem_val:X} (should be 0x1234)")

    if cpu.get_register(2) == 0x1234 and mem_val == 0x1234:
        print("\n✅ MEMORY WRITE TEST PASSED!")
    else:
        print("\n❌ MEMORY WRITE TEST FAILED!")
