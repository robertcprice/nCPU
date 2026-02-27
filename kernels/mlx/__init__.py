"""
MLX Metal Kernel for ARM64 CPU Emulation.

This package provides a custom Metal GPU kernel for high-performance ARM64
CPU emulation on Apple Silicon. It eliminates the GPU-CPU synchronization
bottleneck that limits PyTorch-based execution.

QUICK START:
============

    from mlx_kernel import MLXKernelCPU, StopReason

    # Create CPU
    cpu = MLXKernelCPU(memory_size=4*1024*1024)

    # Load program
    program = [0xD2800000, 0x91000400, 0xD4400000]  # MOVZ, ADD, HLT
    cpu.load_program(program, address=0)
    cpu.set_pc(0)

    # Execute
    result = cpu.execute(max_cycles=100000)
    print(f"Executed {result.cycles} instructions at {result.ips:,.0f} IPS")

PERFORMANCE TARGET:
===================

10M-100M+ IPS on Apple Silicon (vs ~120K IPS with PyTorch batched execution)

PACKAGE CONTENTS:
=================

- MLXKernelCPU: Main CPU emulator class
- StopReason: Enum for execution stop reasons
- ExecutionResult: Dataclass for execution results
- create_cpu(): Convenience function to create CPU instance
- KERNEL_HEADER, KERNEL_SOURCE: Raw Metal shader source code

Author: KVRM Project
Date: 2024
"""

from .cpu_kernel import (
    MLXKernelCPU,
    StopReason,
    ExecutionResult,
    create_cpu,
)

from .cpu_kernel_source import (
    KERNEL_HEADER,
    KERNEL_SOURCE,
    STOP_RUNNING,
    STOP_HALT,
    STOP_SYSCALL,
    STOP_MAX_CYCLES,
    get_kernel_source,
    get_full_kernel_source,
)

__all__ = [
    # Main classes
    'MLXKernelCPU',
    'StopReason',
    'ExecutionResult',
    'create_cpu',
    # Kernel source
    'KERNEL_HEADER',
    'KERNEL_SOURCE',
    'STOP_RUNNING',
    'STOP_HALT',
    'STOP_SYSCALL',
    'STOP_MAX_CYCLES',
    'get_kernel_source',
    'get_full_kernel_source',
]

__version__ = '1.0.0'
