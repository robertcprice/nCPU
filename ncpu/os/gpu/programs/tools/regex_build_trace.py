#!/usr/bin/env python3
"""
Trace the EXACT NFA build phase to find where it goes wrong.

We know:
- NFA table at x22=0x8761A0 gets populated between cycles 15000-16000
- Only entries [0]-[3] are populated (out of many more expected)
- No terminal marker (code_min=-1) exists

Strategy: run with batch_size=100, take snapshots, and track PC+registers
when writes happen to the table region.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kernels.mlx.cpu_kernel_v2 import MLXKernelCPUv2, StopReasonV2
from ncpu.os.gpu.elf_loader import load_elf_into_memory, parse_elf, make_busybox_syscall_handler
from ncpu.os.gpu.filesystem import GPUFilesystem

BUSYBOX = str(PROJECT_ROOT / "demos" / "busybox.elf")


def main():
    fs = GPUFilesystem()
    fs.write_file("/etc/passwd",
        "root:x:0:0:root:/root:/bin/sh\n"
        "hello_user:x:1000:1000:hello:/home/hello:/bin/sh\n"
    )

    cpu = MLXKernelCPUv2(quiet=True)
    argv = ["grep", "hello", "/etc/passwd"]
    entry = load_elf_into_memory(cpu, BUSYBOX, argv=argv, quiet=True)
    cpu.set_pc(entry)

    elf_data = Path(BUSYBOX).read_bytes()
    elf_info = parse_elf(elf_data)
    max_seg_end = max(s.vaddr + s.memsz for s in elf_info.segments)
    heap_base = (max_seg_end + 0xFFFF) & ~0xFFFF

    handler = make_busybox_syscall_handler(filesystem=fs, heap_base=heap_base)

    if cpu.memory_size > cpu.SVC_BUF_BASE + 0x10000:
        cpu.init_svc_buffer()

    # Phase 1: Run quickly past the first 14000 cycles (before NFA build)
    total_cycles = 0
    while total_cycles < 14_500:
        result = cpu.execute(max_cycles=1000)
        total_cycles += result.cycles
        for fd, data in cpu.drain_svc_buffer():
            pass
        if result.stop_reason == StopReasonV2.SYSCALL:
            ret = handler(cpu)
            if ret == False:
                return
            elif ret == "exec":
                continue
            cpu.set_pc(cpu.pc + 4)

    print(f"Phase 1 complete: {total_cycles:,} cycles, PC=0x{cpu.pc:X}")

    # Phase 2: Detailed tracing during NFA build
    TRACK_START = 0x876000
    TRACK_END = 0x877000
    TRACK_SIZE = TRACK_END - TRACK_START
    batch_size = 200
    prev_snapshot = cpu.read_memory(TRACK_START, TRACK_SIZE)

    print(f"\nPhase 2: Tracing NFA build (batch_size={batch_size})")

    build_batches = 0
    while total_cycles < 25_000 and build_batches < 200:
        result = cpu.execute(max_cycles=batch_size)
        total_cycles += result.cycles
        build_batches += 1

        for fd, data in cpu.drain_svc_buffer():
            pass

        if result.stop_reason == StopReasonV2.SYSCALL:
            sysno = cpu.get_register(8)
            x0 = cpu.get_register(0)
            x1 = cpu.get_register(1)
            print(f"  cycle {total_cycles:>6}: syscall {sysno} "
                  f"(x0=0x{x0 & 0xFFFFFFFF:X}, x1=0x{x1 & 0xFFFFFFFF:X}) at PC=0x{cpu.pc:X}")
            ret = handler(cpu)
            if ret == False:
                return
            elif ret == "exec":
                continue
            cpu.set_pc(cpu.pc + 4)

        # Snapshot and detect changes
        snapshot = cpu.read_memory(TRACK_START, TRACK_SIZE)
        changes = []
        for i in range(0, TRACK_SIZE, 4):
            old = int.from_bytes(prev_snapshot[i:i+4], 'little')
            new = int.from_bytes(snapshot[i:i+4], 'little')
            if old != new:
                changes.append((TRACK_START + i, old, new))

        if changes:
            print(f"  cycle {total_cycles:>6}, PC=0x{cpu.pc:X}: {len(changes)} word writes")
            for addr, old, new in changes:
                print(f"    0x{addr:X}: 0x{old:08X} -> 0x{new:08X}")

        prev_snapshot = snapshot

        # If we've entered the stuck loop, stop tracing
        if cpu.pc == 0x4261F4 or cpu.pc == 0x426204:
            print(f"\n  *** ENTERED STUCK LOOP at cycle {total_cycles:,} ***")
            break

    # Final analysis: PC trace
    # The key question: what PC range was active during the build?
    # If we had more granularity, we could trace individual instructions.
    # But from the batch boundaries, we can see the flow.

    print(f"\n  Build phase completed. {build_batches} batches.")
    print(f"  Total writes to NFA table region: {sum(1 for c in changes)}")

    # Let's check: did the build function enter the inner loop at 0x425F20-0x425F5C?
    # That function at 0x425F20 writes terminal markers:
    #   425f54: mov w1, #0xFFFFFFFF
    #   425f58: str w1, [x0]    <-- writes -1 to the code_min field!
    # If this function was called correctly, it should have written terminals.

    # Let's check the broader context: what's in the NFA transition arrays?
    # The transitions at 0x876050-0x876150 have data but the TABLE entries
    # at 0x8761A0+ only have partial data.

    print(f"\n  Full NFA region dump (0x876000-0x876300):")
    for addr in range(0x876000, 0x876300, 0x10):
        data = cpu.read_memory(addr, 0x10)
        hex_str = ' '.join(f'{b:02X}' for b in data)
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data)
        nonzero = sum(1 for b in data if b != 0)
        if nonzero > 0:
            print(f"    0x{addr:X}: {hex_str}  {ascii_str}")


if __name__ == "__main__":
    main()
