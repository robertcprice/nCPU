#!/usr/bin/env python3
"""
Full syscall trace up to the stuck loop, with detailed mmap tracking
and memory verification at key points.
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
    print("=" * 70)
    print("  FULL SYSCALL TRACE TO STUCK LOOP")
    print("=" * 70)

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

    total_cycles = 0
    max_total = 30_000
    batch_size = 200

    all_mmaps = []

    while total_cycles < max_total:
        result = cpu.execute(max_cycles=batch_size)
        total_cycles += result.cycles

        for fd, data in cpu.drain_svc_buffer():
            pass

        if result.stop_reason == StopReasonV2.HALT:
            print(f"HALT at cycle {total_cycles:,}")
            break
        elif result.stop_reason == StopReasonV2.SYSCALL:
            sysno = cpu.get_register(8)
            x0 = cpu.get_register(0)
            x1 = cpu.get_register(1)
            x2 = cpu.get_register(2)
            x3 = cpu.get_register(3)
            pc = cpu.pc

            ret = handler(cpu)
            ret_val = cpu.get_register(0)

            if sysno == 222:  # mmap
                all_mmaps.append({
                    'addr': ret_val & 0xFFFFFFFFFFFFFFFF,
                    'length': x1 & 0xFFFFFFFF,
                })
                print(f"  cycle {total_cycles:>6}: mmap(addr=0x{x0 & 0xFFFFFFFF:X}, "
                      f"len=0x{x1 & 0xFFFFFFFF:X}) -> 0x{ret_val & 0xFFFFFFFFFFFFFFFF:X}")

            if ret == False:
                break
            elif ret == "exec":
                continue
            cpu.set_pc(cpu.pc + 4)

    # At this point we should be at/near the stuck loop
    print(f"\n  Final PC: 0x{cpu.pc:X}, total cycles: {total_cycles:,}")

    # Check registers
    x2 = cpu.get_register(2) & 0xFFFFFFFFFFFFFFFF
    x3 = cpu.get_register(3) & 0xFFFFFFFFFFFFFFFF
    x22 = cpu.get_register(22) & 0xFFFFFFFFFFFFFFFF
    x23 = cpu.get_register(23) & 0xFFFFFFFFFFFFFFFF

    print(f"  x2  = 0x{x2:X}  (current table pointer)")
    print(f"  x3  = 0x{x3:X}  (count array)")
    print(f"  x22 = 0x{x22:X}  (table base?)")
    print(f"  x23 = 0x{x23:X}  (index pointer)")

    # Find which mmap region x2 falls in
    print(f"\n  Memory map:")
    for m in all_mmaps:
        a = m['addr']
        e = a + m['length']
        contains_x2 = a <= x2 < e
        contains_x3 = a <= x3 < e
        markers = []
        if contains_x2:
            markers.append("x2")
        if contains_x3:
            markers.append("x3")
        marker_str = f" <-- contains {', '.join(markers)}" if markers else ""
        print(f"    0x{a:X} - 0x{e:X} (0x{m['length']:X} bytes){marker_str}")

    # x2 might be beyond all mmap'd regions
    max_mmap_end = max(m['addr'] + m['length'] for m in all_mmaps) if all_mmaps else 0
    if x2 >= max_mmap_end:
        print(f"\n  *** x2 = 0x{x2:X} is BEYOND all mmap'd regions (max end: 0x{max_mmap_end:X}) ***")
        print(f"  Gap: 0x{x2 - max_mmap_end:X} bytes")

    # Now trace what the x22/x16 register (table base) is
    # Look at initial value of x2 in the loop
    # From the disassembly, the outer loop sets x2 = x22 at 0x426220
    # x22 is set up earlier in the function
    print(f"\n  x22 = 0x{x22:X} (appears to be table base, used in 'mov x2, x22')")

    # Check memory at x22
    print(f"\n  Memory at x22 (table start):")
    for i in range(10):
        addr = x22 + (i * 0x38)
        data = cpu.read_memory(addr, 8)
        first_word = int.from_bytes(data[:4], 'little')
        second_word = int.from_bytes(data[4:8], 'little')
        bit31 = (first_word >> 31) & 1
        print(f"    [{i:>2}] 0x{addr:X}: 0x{first_word:08X} 0x{second_word:08X} (bit31={bit31})")

    # Check if the writes actually made it to memory
    # Let's look at memory at x3 (count array)
    print(f"\n  Count array at x3 = 0x{x3:X}:")
    for i in range(8):
        addr = x3 + (i * 4)
        data = cpu.read_memory(addr, 4)
        val = int.from_bytes(data, 'little')
        print(f"    [{i:>2}] 0x{addr:X}: {val}")

    # The STORE instruction at 0x426204 writes to [x3 + x0]
    # x0 = LDRSW[x23] << 2 = 4 << 2 = 0x10
    # So it writes to x3 + 0x10 = 0x5702D0 + 0x10 = 0x5702E0
    target_addr = x3 + 0x10
    data = cpu.read_memory(target_addr, 4)
    val = int.from_bytes(data, 'little')
    print(f"\n  STR target [x3 + x0] = [0x{target_addr:X}] = {val}")
    print(f"  (x1 was incremented to ~0x3347 after 200 iterations)")
    print(f"  If STR worked, this value should be very large")
    if val == 0:
        print(f"  *** VALUE IS ZERO - THE STR INSTRUCTION IS NOT WORKING! ***")
    elif val < 100:
        print(f"  *** VALUE IS SUSPICIOUSLY LOW ***")
    else:
        print(f"  STR appears to be working (value = {val})")

    # Let's also check: is the DOUBLE-BUFFER the issue?
    # The Metal kernel reads from memory_in but writes to memory_out
    # Between dispatches, memory_out is copied back to memory_in
    # But WITHIN a dispatch, reads see memory_in (old) not memory_out (new)
    # For the NFA builder loop, this means:
    #   1. Builder writes transition data to the table (writes go to memory_out)
    #   2. In the SAME dispatch, builder reads back from table (reads from memory_in)
    #   3. The reads see OLD data (zeros), not what was just written!
    print(f"\n  === DOUBLE-BUFFER ANALYSIS ===")
    print(f"  The Metal kernel uses double-buffered memory:")
    print(f"    - Reads come from memory_in (input buffer)")
    print(f"    - Writes go to memory_out (output buffer)")
    print(f"    - After each dispatch, memory_out is synced back to memory_in")
    print(f"  This means writes within a single GPU dispatch are NOT visible")
    print(f"  to subsequent reads in that SAME dispatch.")
    print(f"  ")
    print(f"  For most BusyBox commands (echo, cat, etc.) this doesn't matter")
    print(f"  because they don't read back their own writes within tight loops.")
    print(f"  But the NFA builder and matcher DO read-after-write in tight loops.")
    print(f"  ")
    print(f"  However, the batch size is only {batch_size} cycles per GPU dispatch,")
    print(f"  so the sync happens every {batch_size} cycles. Let's verify if this")
    print(f"  is sufficient by checking if writes persist across dispatches.")

    # Write a test value to a known address, dispatch, then read it back
    test_addr = 0x100  # Safe scratch area
    cpu.write_memory(test_addr, b'\x42\x42\x42\x42')
    data = cpu.read_memory(test_addr, 4)
    print(f"\n  Write test: wrote 0x42424242 to 0x{test_addr:X}")
    print(f"  Read back: 0x{int.from_bytes(data, 'little'):08X}")


if __name__ == "__main__":
    main()
