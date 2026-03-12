#!/usr/bin/env python3
"""
Trace when and how the NFA table at x22=0x8761A0 gets populated.

The stuck loop scans from 0x8761A0 with stride 0x38, looking for code_min=-1.
The table has data (code_min=5) but no terminal marker at the right position.

Key question: WHO writes to 0x8761A0-0x876280?
- When does the data appear?
- Is the terminal marker written correctly and then overwritten?
- Or is it never written at all?

Strategy: Take memory snapshots every 100 cycles from cycle 14000 to 16000,
focusing on the region 0x8761A0-0x876300.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kernels.mlx.cpu_kernel_v2 import MLXKernelCPUv2, StopReasonV2
from ncpu.os.gpu.elf_loader import load_elf_into_memory, parse_elf, make_busybox_syscall_handler
from ncpu.os.gpu.filesystem import GPUFilesystem

BUSYBOX = str(PROJECT_ROOT / "demos" / "busybox.elf")

# The exact addresses where we expect terminal markers
TABLE_BASE = 0x8761A0
TABLE_END = 0x876300
TABLE_SIZE = TABLE_END - TABLE_BASE
STRIDE = 0x38


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

    # Phase 1: Run quickly to cycle 14000
    total_cycles = 0
    while total_cycles < 14_000:
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

    print(f"Phase 1: {total_cycles:,} cycles, PC=0x{cpu.pc:X}")

    # Phase 2: Small batches with snapshots
    batch_size = 100
    prev_snapshot = cpu.read_memory(TABLE_BASE, TABLE_SIZE)
    prev_nonzero = sum(1 for b in prev_snapshot if b != 0)

    print(f"\nPhase 2: Tracking memory region 0x{TABLE_BASE:X}-0x{TABLE_END:X}")
    print(f"  Initial nonzero bytes: {prev_nonzero}")

    while total_cycles < 16_500:
        result = cpu.execute(max_cycles=batch_size)
        total_cycles += result.cycles

        for fd, data in cpu.drain_svc_buffer():
            pass

        if result.stop_reason == StopReasonV2.SYSCALL:
            sysno = cpu.get_register(8)
            x0 = cpu.get_register(0) & 0xFFFFFFFFFFFFFFFF
            x1 = cpu.get_register(1) & 0xFFFFFFFFFFFFFFFF
            print(f"  cycle {total_cycles:>6}: syscall {sysno} "
                  f"(x0=0x{x0:X}, x1=0x{x1:X}) at PC=0x{cpu.pc:X}")
            ret = handler(cpu)
            if ret == False:
                return
            elif ret == "exec":
                continue
            cpu.set_pc(cpu.pc + 4)
            continue

        if result.stop_reason == StopReasonV2.HALT:
            print(f"  HALT at cycle {total_cycles:,}")
            return

        # Check for changes
        snapshot = cpu.read_memory(TABLE_BASE, TABLE_SIZE)
        nonzero = sum(1 for b in snapshot if b != 0)

        changes = []
        for i in range(0, TABLE_SIZE, 4):
            old = int.from_bytes(prev_snapshot[i:i+4], 'little')
            new = int.from_bytes(snapshot[i:i+4], 'little')
            if old != new:
                addr = TABLE_BASE + i
                entry_idx = i // STRIDE
                field_off = i % STRIDE
                changes.append((addr, old, new, entry_idx, field_off))

        if changes:
            pc = cpu.pc
            print(f"\n  cycle {total_cycles:>6}, PC=0x{pc:X}: {len(changes)} word changes")
            for addr, old, new, entry_idx, field_off in changes:
                # Identify which field this is
                field_name = "?"
                if field_off == 0x00: field_name = "code_min"
                elif field_off == 0x04: field_name = "code_max"
                elif field_off == 0x08: field_name = "state_lo"
                elif field_off == 0x0C: field_name = "state_hi"
                elif field_off == 0x10: field_name = "state_id"
                elif field_off == 0x20: field_name = "assertions"
                elif field_off == 0x28: field_name = "backref_lo"
                elif field_off == 0x2C: field_name = "backref_hi"
                elif field_off == 0x30: field_name = "neg_cls_lo"
                elif field_off == 0x34: field_name = "neg_cls_hi"

                marker = ""
                if new == 0xFFFFFFFF:
                    if field_off == 0x00:
                        marker = " *** TERMINAL MARKER AT RIGHT PLACE ***"
                    else:
                        marker = f" *** 0xFFFFFFFF at WRONG offset (should be 0x00) ***"

                print(f"    0x{addr:X} entry[{entry_idx}]+0x{field_off:02X} ({field_name}): "
                      f"0x{old:08X} -> 0x{new:08X}{marker}")

        prev_snapshot = snapshot

        # Check if stuck
        if cpu.pc == 0x426204 or cpu.pc == 0x4261F4:
            print(f"\n  *** STUCK LOOP at cycle {total_cycles:,} ***")
            break

    # Final dump of the table
    print(f"\n  === Final NFA table state ===")
    final = cpu.read_memory(TABLE_BASE, TABLE_SIZE)
    for i in range(TABLE_SIZE // STRIDE + 1):
        offset = i * STRIDE
        if offset + STRIDE > TABLE_SIZE:
            break
        data = final[offset:offset+STRIDE]
        nonzero = sum(1 for b in data if b != 0)
        if nonzero > 0:
            code_min = int.from_bytes(data[0:4], 'little', signed=True)
            code_max = int.from_bytes(data[4:8], 'little', signed=True)
            state = int.from_bytes(data[8:16], 'little')
            state_id = int.from_bytes(data[16:20], 'little', signed=True)
            assertions = int.from_bytes(data[32:36], 'little')
            neg_cls = int.from_bytes(data[48:56], 'little')
            addr = TABLE_BASE + offset
            print(f"  entry[{i}] @ 0x{addr:X}:")
            print(f"    code_min={code_min} code_max={code_max} state=0x{state:X} "
                  f"id={state_id} assert=0x{assertions:X} neg=0x{neg_cls:X}")

    # Also check: what's at the x26-based table?
    x26 = cpu.get_register(26) & 0xFFFFFFFFFFFFFFFF
    if x26 > 0 and x26 < cpu.memory_size:
        print(f"\n  === x26 table at 0x{x26:X} ===")
        for i in range(15):
            addr = x26 + i * STRIDE
            if addr + 4 > cpu.memory_size:
                break
            data = cpu.read_memory(addr, 4)
            cm = int.from_bytes(data, 'little', signed=True)
            if cm != 0 or i < 3:
                print(f"    entry[{i}] @ 0x{addr:X}: code_min={cm}")
            if cm == -1:
                print(f"    *** TERMINAL ***")
                break


if __name__ == "__main__":
    main()
