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
        """Analyze a code region for instruction coverage gaps.

        Reads instructions from memory[text_base:text_base+text_size] and
        classifies each one as known/unknown based on the Metal kernel's
        decode table. Returns list of unique unknown instruction classes.

        Args:
            text_base: Start address of .text segment
            text_size: Size of .text segment in bytes

        Returns:
            List of dicts with 'opcode', 'pc', 'opcode_hex', 'top_byte' keys.
        """
        data = self.read_memory(text_base, text_size)
        unknowns = {}

        for off in range(0, len(data) - 3, 4):
            inst = int.from_bytes(data[off:off+4], 'little')
            if inst == 0:
                continue  # Skip zero-fill

            top = (inst >> 24) & 0xFF
            class_key = inst >> 21

            if not self._is_known_instruction(inst, top):
                if class_key not in unknowns:
                    unknowns[class_key] = {
                        'opcode': inst,
                        'pc': text_base + off,
                        'opcode_hex': f'0x{inst:08X}',
                        'top_byte': f'0x{top:02X}',
                        'class_bits': f'0x{class_key:03X}',
                    }

        return list(unknowns.values())

    @staticmethod
    def _is_known_instruction(inst: int, top: int) -> bool:
        """Check if an instruction is handled by the Metal kernel's decode table."""
        # Top-level instruction categories from the Metal kernel
        op0 = (inst >> 25) & 0xF

        # HLT
        if (inst & 0xFFE0001F) == 0xD4400000:
            return True
        # SVC
        if (inst & 0xFFE0001F) == 0xD4000001:
            return True
        # NOP, DMB, DSB, ISB, CLREX
        if inst in (0xD503201F, 0xD50330BF, 0xD5033F9F, 0xD5033FDF, 0xD503305F):
            return True

        # Data processing — immediate
        if top in (0xD2, 0xD3, 0xF2, 0xF3, 0x52, 0x72):  # MOVZ, MOVK
            return True
        if top in (0x92, 0x12):  # MOVN
            return True
        if top in (0x91, 0x11, 0xB1, 0x31):  # ADD/ADDS imm
            return True
        if top in (0xD1, 0x51, 0xF1, 0x71):  # SUB/SUBS imm
            return True
        if top in (0x92, 0x12, 0xB2, 0x32):  # AND/ANDS imm (logical imm)
            return True
        if top in (0xD3, 0x53):  # UBFM/SBFM/BFM
            return True
        if top == 0x93:  # SBFM 64-bit / EXTR
            return True
        if top == 0x33:  # BFM 32-bit
            return True

        # Branches
        if top in (0x14, 0x15, 0x16, 0x17):  # B
            return True
        if top in (0x94, 0x95, 0x96, 0x97):  # BL
            return True
        if top in (0x54,):  # B.cond
            return True
        if (inst & 0xFF000000) in (0xB4000000, 0xB5000000, 0x34000000, 0x35000000):  # CBZ/CBNZ
            return True
        if (inst & 0x7F000000) in (0x36000000, 0x37000000):  # TBZ/TBNZ
            return True
        if (inst & 0xFFFFFC1F) in (0xD61F0000, 0xD63F0000, 0xD65F0000):  # BR/BLR/RET
            return True

        # Load/Store
        if top in (0xF9, 0xB9, 0x39, 0x79):  # LDR/STR unsigned imm
            return True
        if top in (0xF8, 0xB8, 0x38, 0x78):  # LDR/STR reg offset / pre/post index
            return True
        if top in (0xA9, 0xA8, 0x29, 0x28):  # STP/LDP
            return True
        if top in (0x18, 0x1C, 0x58, 0x98, 0x9C, 0xD8):  # LDR literal / PRFM literal
            return True

        # Data processing — register
        if top in (0x8B, 0x0B, 0xAB, 0x2B):  # ADD reg
            return True
        if top in (0xCB, 0x4B, 0xEB, 0x6B):  # SUB/SUBS reg
            return True
        if top in (0x8A, 0x0A, 0xAA, 0x2A):  # AND/ORR reg
            return True
        if top in (0xCA, 0x4A, 0xEA, 0x6A):  # EOR/ANDS reg
            return True
        if top in (0x9A, 0x1A, 0xDA, 0x5A):  # CSEL/ADC/SBC
            return True
        if top in (0x9B, 0x1B):  # MUL/MADD/MSUB/SMULH/UMULH
            return True

        # Division
        if (inst & 0xFF200000) in (0x9AC00000, 0x1AC00000):  # SDIV/UDIV
            return True

        # Shifts
        if (inst & 0xFF200000) in (0x9AC00000, 0x1AC00000):  # shift reg
            return True

        # Bit manipulation
        if (inst & 0xFF200000) == 0xDAC00000:  # CLZ/RBIT/REV
            return True
        if (inst & 0x7F200000) == 0x5AC00000:  # 32-bit bit ops
            return True

        # ADR/ADRP
        if (inst & 0x9F000000) in (0x10000000, 0x90000000):
            return True

        # MRS/MSR
        if (inst & 0xFFF00000) in (0xD5300000, 0xD5100000):
            return True

        # LDXR/STXR
        if (inst & 0xFF200000) in (0xC8400000, 0xC8000000, 0x88400000, 0x88000000):
            return True

        # Extension ops (SXTW, UXTB, etc.) — covered by SBFM/UBFM
        # SIMD load/store pair
        if (inst & 0xFFC00000) in (0xAD000000, 0xAD400000, 0x6D000000, 0x6D400000):
            return True

        return False

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
