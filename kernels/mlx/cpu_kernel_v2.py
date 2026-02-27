#!/usr/bin/env python3
"""
MLX Metal Kernel V2 - Double-Buffer Memory Architecture.

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

Author: KVRM Project
Date: 2024
"""

import mlx.core as mx
import numpy as np
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Union

from .cpu_kernel_v2_source import (
    KERNEL_HEADER_V2,
    KERNEL_SOURCE_V2,
    STOP_RUNNING,
    STOP_HALT,
    STOP_SYSCALL,
    STOP_MAX_CYCLES,
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

    @property
    def name_str(self) -> str:
        return ["RUNNING", "HALT", "SYSCALL", "MAX_CYCLES"][self]


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

    def __init__(self, memory_size: int = 4 * 1024 * 1024):
        """Initialize CPU with given memory size."""
        self.memory_size = memory_size

        # Memory (double-buffered)
        self.memory = mx.zeros(memory_size, dtype=mx.uint8)

        # Registers (x0-x30, x31=XZR)
        self.registers = mx.zeros(32, dtype=mx.int64)

        # Program Counter
        self._pc = 0

        # Condition Flags [N, Z, C, V]
        self.flags = mx.zeros(4, dtype=mx.float32)

        # Compile the V2 kernel with memory_out support
        self._kernel = mx.fast.metal_kernel(
            name="arm64_cpu_execute_v2",
            input_names=["memory_in", "registers_in", "pc_in", "flags_in",
                        "max_cycles_in", "memory_size_in"],
            output_names=["memory_out", "registers_out", "pc_out", "flags_out",
                         "cycles_out", "stop_reason_out"],
            source=KERNEL_SOURCE_V2,
            header=KERNEL_HEADER_V2,
            ensure_row_contiguous=True,
            atomic_outputs=False,
        )

        # Statistics
        self.total_instructions = 0
        self.total_syscalls = 0

        print(f"[MLXKernelCPUv2] Initialized with {memory_size:,} bytes memory")
        print(f"[MLXKernelCPUv2] Double-buffer memory architecture enabled")
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
        return int(self.registers[reg].item())

    def set_register(self, reg: int, value: int) -> None:
        if reg == 31:
            return
        self.registers = self.registers.at[reg].add(-self.registers[reg] + value)
        mx.eval(self.registers)

    def get_registers_numpy(self) -> np.ndarray:
        return np.array(self.registers)

    def set_registers_numpy(self, values: np.ndarray) -> None:
        self.registers = mx.array(values, dtype=mx.int64)
        mx.eval(self.registers)

    def get_flags(self) -> tuple[bool, bool, bool, bool]:
        f = np.array(self.flags)
        return (f[0] > 0.5, f[1] > 0.5, f[2] > 0.5, f[3] > 0.5)

    def set_flags(self, n: bool, z: bool, c: bool, v: bool) -> None:
        self.flags = mx.array([
            1.0 if n else 0.0, 1.0 if z else 0.0,
            1.0 if c else 0.0, 1.0 if v else 0.0,
        ], dtype=mx.float32)
        mx.eval(self.flags)

    # ═══════════════════════════════════════════════════════════════════════════
    # MEMORY OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════

    def read_memory(self, address: int, size: int) -> bytes:
        end = min(address + size, self.memory_size)
        return bytes(np.array(self.memory[address:end]))

    def write_memory(self, address: int, data: bytes) -> None:
        data_array = np.frombuffer(data, dtype=np.uint8)
        mem_np = np.array(self.memory)
        mem_np[address:address + len(data)] = data_array
        self.memory = mx.array(mem_np, dtype=mx.uint8)
        mx.eval(self.memory)

    def load_program(self, program: Union[bytes, list[int]], address: int = 0) -> None:
        if isinstance(program, list):
            data = b''
            for inst in program:
                data += inst.to_bytes(4, 'little')
            program = data
        self.write_memory(address, program)
        print(f"[MLXKernelCPUv2] Loaded {len(program):,} bytes at 0x{address:X}")

    def clear_memory(self) -> None:
        self.memory = mx.zeros(self.memory_size, dtype=mx.uint8)
        mx.eval(self.memory)

    def read_memory_64(self, address: int) -> int:
        """Read 64-bit value from memory (little-endian)."""
        data = self.read_memory(address, 8)
        return int.from_bytes(data, 'little')

    def write_memory_64(self, address: int, value: int) -> None:
        """Write 64-bit value to memory (little-endian)."""
        self.write_memory(address, value.to_bytes(8, 'little'))

    # ═══════════════════════════════════════════════════════════════════════════
    # EXECUTION
    # ═══════════════════════════════════════════════════════════════════════════

    def execute(self, max_cycles: int = 100000) -> ExecutionResultV2:
        """
        Execute instructions using Metal kernel V2.

        The kernel runs entirely on GPU with full memory read/write support.
        """
        start_time = time.perf_counter()

        # Prepare inputs
        pc_array = mx.array([self._pc], dtype=mx.uint64)
        max_cycles_array = mx.array([max_cycles], dtype=mx.uint32)
        memory_size_array = mx.array([self.memory_size], dtype=mx.uint32)

        # Execute kernel - memory_out is a new buffer that kernel writes to
        outputs = self._kernel(
            inputs=[self.memory, self.registers, pc_array, self.flags,
                   max_cycles_array, memory_size_array],
            output_shapes=[
                (self.memory_size,),  # memory_out - FULL MEMORY OUTPUT
                (32,),                 # registers_out
                (1,),                  # pc_out
                (4,),                  # flags_out
                (1,),                  # cycles_out
                (1,),                  # stop_reason_out
            ],
            output_dtypes=[
                mx.uint8,   # memory_out
                mx.int64,   # registers_out
                mx.uint64,  # pc_out
                mx.float32, # flags_out
                mx.uint32,  # cycles_out
                mx.uint8,   # stop_reason_out
            ],
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
            verbose=False,
        )

        # Force evaluation
        mx.eval(outputs)

        # Unpack outputs
        memory_out, registers_out, pc_out, flags_out, cycles_out, stop_reason_out = outputs

        # Update state - SWAP memory buffer!
        self.memory = memory_out  # memory_out becomes the new memory
        self.registers = registers_out
        self._pc = int(pc_out[0].item())
        self.flags = flags_out
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
        self.registers = mx.zeros(32, dtype=mx.int64)
        self._pc = 0
        self.flags = mx.zeros(4, dtype=mx.float32)
        self.total_instructions = 0
        self.total_syscalls = 0
        mx.eval(self.registers, self.flags)

    def dump_registers(self) -> str:
        """Get formatted register dump."""
        regs = np.array(self.registers)
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


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def create_cpu_v2(memory_size: int = 4 * 1024 * 1024) -> MLXKernelCPUv2:
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
