#!/usr/bin/env python3
"""
MLX Metal Kernel Wrapper for ARM64 CPU Emulation.

This module provides a Python interface to the custom Metal GPU kernel
that emulates an ARM64 CPU. The kernel runs entirely on the GPU, eliminating
the GPU-CPU synchronization bottleneck that limits PyTorch-based execution.

USAGE:
======

    from mlx_kernel import MLXKernelCPU

    # Create CPU with 4MB memory
    cpu = MLXKernelCPU(memory_size=4*1024*1024)

    # Load program
    cpu.load_program(program_bytes, address=0x10000)
    cpu.set_pc(0x10000)

    # Execute up to 100,000 instructions
    result = cpu.execute(max_cycles=100000)

    print(f"Executed: {result.cycles} instructions")
    print(f"IPS: {result.ips:,.0f}")
    print(f"Stop reason: {result.stop_reason_name}")

PERFORMANCE:
============

Target: 10M-100M+ IPS on Apple Silicon (vs ~120K IPS with PyTorch batched)

Key optimizations:
- Zero GPU-CPU sync during execution loop
- Local register copy in Metal kernel
- Single-threaded for correctness (parallel lanes planned for Phase 2)

INTEGRATION WITH PYTORCH:
=========================

This module can be used alongside PyTorch neural networks:

    import torch
    from mlx_kernel import MLXKernelCPU

    # MLX for fast CPU emulation
    cpu = MLXKernelCPU()

    # PyTorch for neural models
    loop_detector = torch.load('models/loop_detector.pt')
    decoder_model = torch.load('models/decoder.pt')

    # Hybrid execution: detect loops with PyTorch, execute with MLX
    while not halted:
        # Run MLX kernel for N cycles
        result = cpu.execute(max_cycles=10000)

        if result.stop_reason == StopReason.SYSCALL:
            # Handle syscall
            handle_syscall(cpu)

        # Periodically check for loop patterns (optional)
        # ...

Author: KVRM Project
Date: 2024
"""

import mlx.core as mx
import numpy as np
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Union

from .cpu_kernel_source import (
    KERNEL_HEADER,
    KERNEL_SOURCE,
    STOP_RUNNING,
    STOP_HALT,
    STOP_SYSCALL,
    STOP_MAX_CYCLES,
)


# ═══════════════════════════════════════════════════════════════════════════════
# STOP REASON ENUM
# ═══════════════════════════════════════════════════════════════════════════════

class StopReason(IntEnum):
    """Reasons why kernel execution stopped."""
    RUNNING = STOP_RUNNING       # Should not be returned
    HALT = STOP_HALT             # HLT instruction or PC=0 with inst=0
    SYSCALL = STOP_SYSCALL       # SVC instruction (needs kernel handling)
    MAX_CYCLES = STOP_MAX_CYCLES # Reached max_cycles limit

    @property
    def name_str(self) -> str:
        """Human-readable name."""
        return {
            StopReason.RUNNING: "RUNNING",
            StopReason.HALT: "HALT",
            StopReason.SYSCALL: "SYSCALL",
            StopReason.MAX_CYCLES: "MAX_CYCLES",
        }[self]


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutionResult:
    """Result of kernel execution."""
    cycles: int                  # Number of instructions executed
    elapsed_seconds: float       # Wall clock time
    stop_reason: StopReason      # Why execution stopped
    final_pc: int                # Final program counter

    @property
    def ips(self) -> float:
        """Instructions per second."""
        if self.elapsed_seconds > 0:
            return self.cycles / self.elapsed_seconds
        return 0.0

    @property
    def stop_reason_name(self) -> str:
        """Human-readable stop reason."""
        return self.stop_reason.name_str


# ═══════════════════════════════════════════════════════════════════════════════
# MLX KERNEL CPU
# ═══════════════════════════════════════════════════════════════════════════════

class MLXKernelCPU:
    """
    ARM64 CPU Emulator using custom Metal GPU kernel.

    This class manages CPU state (memory, registers, PC, flags) and provides
    an interface to execute instructions using the MLX Metal kernel.

    Attributes:
        memory_size: Size of memory in bytes (default 4MB)
        memory: MLX array representing unified memory
        registers: MLX array representing 32 general-purpose registers
        pc: Program counter (Python int for easy manipulation)
        flags: MLX array representing NZCV condition flags
    """

    def __init__(self, memory_size: int = 4 * 1024 * 1024):
        """
        Initialize the CPU with given memory size.

        Args:
            memory_size: Memory size in bytes (default 4MB)
        """
        self.memory_size = memory_size

        # ═══════════════════════════════════════════════════════════════════
        # STATE INITIALIZATION
        # ═══════════════════════════════════════════════════════════════════

        # Memory (unified code + data)
        self.memory = mx.zeros(memory_size, dtype=mx.uint8)

        # Registers (x0-x30, x31=XZR)
        self.registers = mx.zeros(32, dtype=mx.int64)

        # Program Counter (kept as Python int for easy manipulation)
        self._pc = 0

        # Condition Flags [N, Z, C, V]
        self.flags = mx.zeros(4, dtype=mx.float32)

        # ═══════════════════════════════════════════════════════════════════
        # KERNEL COMPILATION
        # ═══════════════════════════════════════════════════════════════════

        self._kernel = mx.fast.metal_kernel(
            name="arm64_cpu_execute",
            input_names=["memory", "registers_in", "pc_in", "flags_in", "max_cycles_in"],
            output_names=["registers_out", "pc_out", "flags_out", "cycles_out", "stop_reason_out"],
            source=KERNEL_SOURCE,
            header=KERNEL_HEADER,
            ensure_row_contiguous=True,
            atomic_outputs=False,
        )

        # Statistics
        self.total_instructions = 0
        self.total_syscalls = 0

        print(f"[MLXKernelCPU] Initialized with {memory_size:,} bytes memory")
        print(f"[MLXKernelCPU] Metal kernel compiled successfully")

    # ═══════════════════════════════════════════════════════════════════════════
    # PROPERTY ACCESSORS
    # ═══════════════════════════════════════════════════════════════════════════

    @property
    def pc(self) -> int:
        """Get current program counter."""
        return self._pc

    def set_pc(self, value: int) -> None:
        """Set program counter."""
        self._pc = value & 0xFFFFFFFFFFFFFFFF

    def get_register(self, reg: int) -> int:
        """
        Get value of a register.

        Args:
            reg: Register index (0-31, 31=XZR always returns 0)

        Returns:
            Register value as Python int
        """
        if reg == 31:
            return 0
        return int(self.registers[reg].item())

    def set_register(self, reg: int, value: int) -> None:
        """
        Set value of a register.

        Args:
            reg: Register index (0-30, 31=XZR is ignored)
            value: Value to set
        """
        if reg == 31:
            return  # XZR cannot be written
        self.registers = self.registers.at[reg].add(-self.registers[reg] + value)
        mx.eval(self.registers)

    def get_registers_numpy(self) -> np.ndarray:
        """Get all registers as numpy array."""
        return np.array(self.registers)

    def set_registers_numpy(self, values: np.ndarray) -> None:
        """Set all registers from numpy array."""
        self.registers = mx.array(values, dtype=mx.int64)
        mx.eval(self.registers)

    def get_flags(self) -> tuple[bool, bool, bool, bool]:
        """
        Get condition flags as tuple.

        Returns:
            Tuple of (N, Z, C, V) as booleans
        """
        f = np.array(self.flags)
        return (f[0] > 0.5, f[1] > 0.5, f[2] > 0.5, f[3] > 0.5)

    def set_flags(self, n: bool, z: bool, c: bool, v: bool) -> None:
        """Set condition flags."""
        self.flags = mx.array([
            1.0 if n else 0.0,
            1.0 if z else 0.0,
            1.0 if c else 0.0,
            1.0 if v else 0.0,
        ], dtype=mx.float32)
        mx.eval(self.flags)

    # ═══════════════════════════════════════════════════════════════════════════
    # MEMORY OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════

    def read_memory(self, address: int, size: int) -> bytes:
        """
        Read bytes from memory.

        Args:
            address: Starting address
            size: Number of bytes to read

        Returns:
            Bytes read from memory
        """
        end = min(address + size, self.memory_size)
        return bytes(np.array(self.memory[address:end]))

    def write_memory(self, address: int, data: bytes) -> None:
        """
        Write bytes to memory.

        Args:
            address: Starting address
            data: Bytes to write
        """
        data_array = np.frombuffer(data, dtype=np.uint8)
        mem_np = np.array(self.memory)
        mem_np[address:address + len(data)] = data_array
        self.memory = mx.array(mem_np, dtype=mx.uint8)
        mx.eval(self.memory)

    def load_program(self, program: Union[bytes, list[int]], address: int = 0) -> None:
        """
        Load program bytes into memory.

        Args:
            program: Program bytes or list of instruction ints
            address: Load address (default 0)
        """
        if isinstance(program, list):
            # Convert instruction ints to bytes (little-endian)
            data = b''
            for inst in program:
                data += inst.to_bytes(4, 'little')
            program = data

        self.write_memory(address, program)
        print(f"[MLXKernelCPU] Loaded {len(program):,} bytes at 0x{address:X}")

    def clear_memory(self) -> None:
        """Clear all memory to zeros."""
        self.memory = mx.zeros(self.memory_size, dtype=mx.uint8)
        mx.eval(self.memory)

    # ═══════════════════════════════════════════════════════════════════════════
    # EXECUTION
    # ═══════════════════════════════════════════════════════════════════════════

    def execute(self, max_cycles: int = 100000) -> ExecutionResult:
        """
        Execute instructions using Metal kernel.

        This is the main execution entry point. The kernel runs entirely on
        the GPU until it hits a halt, syscall, or max_cycles limit.

        Args:
            max_cycles: Maximum instructions to execute (default 100,000)

        Returns:
            ExecutionResult with cycles, elapsed time, and stop reason
        """
        start_time = time.perf_counter()

        # Prepare inputs as MLX arrays
        pc_array = mx.array([self._pc], dtype=mx.uint64)
        max_cycles_array = mx.array([max_cycles], dtype=mx.uint32)

        # Execute kernel
        outputs = self._kernel(
            inputs=[self.memory, self.registers, pc_array, self.flags, max_cycles_array],
            output_shapes=[(32,), (1,), (4,), (1,), (1,)],
            output_dtypes=[mx.int64, mx.uint64, mx.float32, mx.uint32, mx.uint8],
            grid=(1, 1, 1),       # Single thread for now
            threadgroup=(1, 1, 1),
            verbose=False,
        )

        # Force evaluation and get results
        mx.eval(outputs)

        registers_out, pc_out, flags_out, cycles_out, stop_reason_out = outputs

        # Update state
        self.registers = registers_out
        self._pc = int(pc_out[0].item())
        self.flags = flags_out
        cycles = int(cycles_out[0].item())
        stop_reason = StopReason(int(stop_reason_out[0].item()))

        elapsed = time.perf_counter() - start_time

        # Update statistics
        self.total_instructions += cycles
        if stop_reason == StopReason.SYSCALL:
            self.total_syscalls += 1

        return ExecutionResult(
            cycles=cycles,
            elapsed_seconds=elapsed,
            stop_reason=stop_reason,
            final_pc=self._pc,
        )

    def execute_until_halt(self, max_total_cycles: int = 10_000_000,
                           cycles_per_batch: int = 100_000) -> ExecutionResult:
        """
        Execute until halt, handling syscalls along the way.

        This is a convenience method that loops execute() and handles
        syscalls automatically (by advancing PC). For real syscall
        handling, use execute() directly and handle syscalls in your code.

        Args:
            max_total_cycles: Maximum total instructions
            cycles_per_batch: Instructions per kernel invocation

        Returns:
            Final ExecutionResult
        """
        total_cycles = 0
        start_time = time.perf_counter()

        while total_cycles < max_total_cycles:
            remaining = max_total_cycles - total_cycles
            batch_size = min(cycles_per_batch, remaining)

            result = self.execute(max_cycles=batch_size)
            total_cycles += result.cycles

            if result.stop_reason == StopReason.HALT:
                break
            elif result.stop_reason == StopReason.SYSCALL:
                # Simple handling: advance PC and continue
                self._pc += 4
                # In real use, you would handle the syscall here
            elif result.stop_reason == StopReason.MAX_CYCLES:
                continue  # Keep going

        elapsed = time.perf_counter() - start_time

        return ExecutionResult(
            cycles=total_cycles,
            elapsed_seconds=elapsed,
            stop_reason=result.stop_reason if result.cycles < cycles_per_batch else StopReason.MAX_CYCLES,
            final_pc=self._pc,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════

    def reset(self) -> None:
        """Reset CPU state to initial values."""
        self.registers = mx.zeros(32, dtype=mx.int64)
        self._pc = 0
        self.flags = mx.zeros(4, dtype=mx.float32)
        self.total_instructions = 0
        self.total_syscalls = 0
        mx.eval(self.registers, self.flags)

    def get_state(self) -> dict:
        """
        Get complete CPU state as dictionary.

        Returns:
            Dict with registers, pc, flags, and statistics
        """
        return {
            'registers': np.array(self.registers).tolist(),
            'pc': self._pc,
            'flags': np.array(self.flags).tolist(),
            'total_instructions': self.total_instructions,
            'total_syscalls': self.total_syscalls,
        }

    def set_state(self, state: dict) -> None:
        """
        Set CPU state from dictionary.

        Args:
            state: Dict with registers, pc, flags (as returned by get_state)
        """
        if 'registers' in state:
            self.registers = mx.array(state['registers'], dtype=mx.int64)
        if 'pc' in state:
            self._pc = state['pc']
        if 'flags' in state:
            self.flags = mx.array(state['flags'], dtype=mx.float32)
        if 'total_instructions' in state:
            self.total_instructions = state['total_instructions']
        if 'total_syscalls' in state:
            self.total_syscalls = state['total_syscalls']
        mx.eval(self.registers, self.flags)

    # ═══════════════════════════════════════════════════════════════════════════
    # DEBUGGING
    # ═══════════════════════════════════════════════════════════════════════════

    def dump_registers(self) -> str:
        """
        Get formatted string of all registers.

        Returns:
            Multi-line string showing all register values
        """
        regs = np.array(self.registers)
        lines = []
        lines.append(f"PC:  0x{self._pc:016X}")
        lines.append("")
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

    def disassemble_at(self, address: int, count: int = 10) -> str:
        """
        Simple disassembly of instructions at address.

        Args:
            address: Starting address
            count: Number of instructions

        Returns:
            Multi-line disassembly string
        """
        lines = []
        for i in range(count):
            addr = address + i * 4
            if addr + 4 > self.memory_size:
                break
            mem_bytes = self.read_memory(addr, 4)
            inst = int.from_bytes(mem_bytes, 'little')
            lines.append(f"0x{addr:08X}:  {inst:08X}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_cpu(memory_size: int = 4 * 1024 * 1024) -> MLXKernelCPU:
    """
    Create a new MLX kernel CPU instance.

    Args:
        memory_size: Memory size in bytes (default 4MB)

    Returns:
        MLXKernelCPU instance
    """
    return MLXKernelCPU(memory_size=memory_size)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("MLX KERNEL CPU - BASIC TEST")
    print("=" * 70)

    cpu = MLXKernelCPU()

    # Test program: Simple ADD loop
    # MOVZ X0, #0
    # MOVZ X1, #1000
    # loop: ADD X0, X0, #1
    # SUBS X1, X1, #1
    # CBNZ X1, loop
    # HLT

    program = [
        0xD2800000,  # MOVZ X0, #0
        0xD2807D01,  # MOVZ X1, #1000 (0x3E8)
        0x91000400,  # ADD X0, X0, #1
        0xF1000421,  # SUBS X1, X1, #1
        0xB5FFFFC1,  # CBNZ X1, -8 (back to ADD)
        0xD4400000,  # HLT
    ]

    cpu.load_program(program, address=0)
    cpu.set_pc(0)

    print("\nInitial state:")
    print(cpu.dump_registers())

    print("\nExecuting...")
    result = cpu.execute(max_cycles=100000)

    print(f"\nResult:")
    print(f"  Cycles: {result.cycles:,}")
    print(f"  Elapsed: {result.elapsed_seconds * 1000:.2f} ms")
    print(f"  IPS: {result.ips:,.0f}")
    print(f"  Stop reason: {result.stop_reason_name}")

    print("\nFinal state:")
    print(cpu.dump_registers())

    # Verify X0 should be 1000
    x0 = cpu.get_register(0)
    print(f"\nVerification: X0 = {x0} (expected 1000)")
    assert x0 == 1000, f"X0 should be 1000, got {x0}"
    print("TEST PASSED!")
