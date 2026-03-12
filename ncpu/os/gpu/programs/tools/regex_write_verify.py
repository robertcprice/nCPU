#!/usr/bin/env python3
"""
Verify: is the NFA builder writing data that then disappears?

Strategy: Take memory snapshots every 1000 cycles during the regcomp
phase (cycles 5000-15000) and see when/where data appears and disappears.
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

    # Track region 0x876000-0x877000 (the last mmap region where the NFA table lives)
    TRACK_START = 0x876000
    TRACK_END = 0x877000
    TRACK_SIZE = TRACK_END - TRACK_START

    total_cycles = 0
    batch_size = 500
    prev_snapshot = None
    snapshots = []

    print(f"Tracking memory region 0x{TRACK_START:X}-0x{TRACK_END:X}")
    print(f"Taking snapshots every {batch_size} cycles during regcomp phase\n")

    while total_cycles < 20_000:
        result = cpu.execute(max_cycles=batch_size)
        total_cycles += result.cycles

        for fd, data in cpu.drain_svc_buffer():
            pass

        if result.stop_reason == StopReasonV2.SYSCALL:
            sysno = cpu.get_register(8)
            ret = handler(cpu)
            if ret == False:
                return
            elif ret == "exec":
                continue
            cpu.set_pc(cpu.pc + 4)

        if result.stop_reason == StopReasonV2.HALT:
            break

        # Take snapshot
        snapshot = cpu.read_memory(TRACK_START, TRACK_SIZE)
        nonzero = sum(1 for b in snapshot if b != 0)

        if prev_snapshot is not None:
            # Find changes
            changes = 0
            new_nonzero = 0
            cleared = 0
            for i in range(0, len(snapshot), 4):
                old = int.from_bytes(prev_snapshot[i:i+4], 'little')
                new = int.from_bytes(snapshot[i:i+4], 'little')
                if old != new:
                    changes += 1
                    if old == 0 and new != 0:
                        new_nonzero += 1
                    elif old != 0 and new == 0:
                        cleared += 1

            if changes > 0:
                print(f"  cycle {total_cycles:>6}, PC=0x{cpu.pc:X}: "
                      f"{changes} words changed ({new_nonzero} new, {cleared} cleared), "
                      f"total nonzero: {nonzero}")

                # Show details for first few changes
                detail_count = 0
                for i in range(0, len(snapshot), 4):
                    old = int.from_bytes(prev_snapshot[i:i+4], 'little')
                    new = int.from_bytes(snapshot[i:i+4], 'little')
                    if old != new and detail_count < 10:
                        addr = TRACK_START + i
                        # What entry/offset is this?
                        rel = addr - 0x8761A0  # relative to x22
                        entry_num = rel // 0x38 if rel >= 0 else -1
                        offset_in_entry = rel % 0x38 if rel >= 0 else -1
                        print(f"    0x{addr:X} (entry[{entry_num}]+0x{offset_in_entry:X}): "
                              f"0x{old:08X} -> 0x{new:08X}")
                        detail_count += 1
                    if detail_count >= 10 and changes > 10:
                        print(f"    ... {changes - 10} more changes")
                        break

        prev_snapshot = snapshot

    # Final analysis
    print(f"\n  Final state at cycle {total_cycles:,}:")
    final = cpu.read_memory(TRACK_START, TRACK_SIZE)
    nonzero_count = sum(1 for b in final if b != 0)
    print(f"  Total nonzero bytes: {nonzero_count}/{TRACK_SIZE}")

    # Check for 0xFFFFFFFF at stride-aligned positions
    x22 = 0x8761A0
    print(f"\n  First 10 entries at x22=0x{x22:X}:")
    for i in range(10):
        offset = (x22 + i * 0x38) - TRACK_START
        if 0 <= offset < TRACK_SIZE - 4:
            val = int.from_bytes(final[offset:offset+4], 'little', signed=True)
            print(f"    [{i}] code_min = {val}")

    # Check ALL words with value 0xFFFFFFFF in the tracked region
    print(f"\n  All 0xFFFFFFFF words in tracked region:")
    for i in range(0, TRACK_SIZE, 4):
        val = int.from_bytes(final[i:i+4], 'little')
        if val == 0xFFFFFFFF:
            addr = TRACK_START + i
            rel = addr - x22
            entry = rel // 0x38 if rel >= 0 else -1
            off = rel % 0x38 if rel >= 0 else -1
            print(f"    0x{addr:X} (entry[{entry}] + 0x{off:X})")


if __name__ == "__main__":
    main()
