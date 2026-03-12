#!/usr/bin/env python3
"""
Precise debug: stop execution at the EXACT moment the stuck loop starts
and verify the table pointer, terminal markers, and loop behavior.
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
    print("  PRECISE LOOP ENTRY DIAGNOSTIC")
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

    # Run in small batches until PC enters the stuck loop range
    LOOP_ADDR = 0x4261F4
    total_cycles = 0
    batch_size = 50

    while total_cycles < 50_000:
        result = cpu.execute(max_cycles=batch_size)
        total_cycles += result.cycles

        for fd, data in cpu.drain_svc_buffer():
            pass

        pc = cpu.pc

        if result.stop_reason == StopReasonV2.SYSCALL:
            sysno = cpu.get_register(8)
            ret = handler(cpu)
            if ret == False:
                print(f"EXIT at cycle {total_cycles:,}")
                return
            elif ret == "exec":
                continue
            cpu.set_pc(cpu.pc + 4)
            continue

        if result.stop_reason == StopReasonV2.HALT:
            print(f"HALT at cycle {total_cycles:,}")
            return

        # Check if we've entered the stuck loop
        if 0x4261F0 <= pc <= 0x426220:
            print(f"  Entered stuck loop at cycle {total_cycles:,}, PC=0x{pc:X}")
            break

    # Now dump all relevant state
    print(f"\n  Register state at loop entry:")
    for i in range(0, 30):
        val = cpu.get_register(i) & 0xFFFFFFFFFFFFFFFF
        if val != 0:
            print(f"    X{i:<2} = 0x{val:016X}")

    x2 = cpu.get_register(2) & 0xFFFFFFFFFFFFFFFF
    x3 = cpu.get_register(3) & 0xFFFFFFFFFFFFFFFF
    x22 = cpu.get_register(22) & 0xFFFFFFFFFFFFFFFF
    x23 = cpu.get_register(23) & 0xFFFFFFFFFFFFFFFF

    print(f"\n  Key registers:")
    print(f"    x2  (table cursor) = 0x{x2:X}")
    print(f"    x3  (count array)  = 0x{x3:X}")
    print(f"    x22 (table base)   = 0x{x22:X}")
    print(f"    x23 (index ptr)    = 0x{x23:X}")

    # Read index from [x23]
    idx_data = cpu.read_memory(x23, 4)
    idx = int.from_bytes(idx_data, 'little', signed=True)
    print(f"    [x23] = {idx} (index for count array)")
    print(f"    Effective STR target: x3 + (idx*4) = 0x{x3 + idx*4:X}")

    # Scan table from x22 with stride 0x38, looking for terminal
    print(f"\n  Scanning table from x22=0x{x22:X} (stride 0x38):")
    for i in range(300):
        addr = x22 + (i * 0x38)
        if addr >= cpu.memory_size:
            print(f"    Hit memory boundary at entry {i}")
            break
        data = cpu.read_memory(addr, 0x38)
        code_min = int.from_bytes(data[0:4], 'little', signed=True)
        code_max = int.from_bytes(data[4:8], 'little', signed=True)
        state = int.from_bytes(data[8:12], 'little', signed=True)

        if code_min != 0 or code_max != 0 or state != 0:
            print(f"    [{i:>3}] 0x{addr:X}: code_min={code_min}, code_max={code_max}, state={state}")
        if code_min == -1:
            print(f"    *** TERMINAL FOUND at entry {i} ***")
            # Show the next few entries too
            for j in range(i+1, min(i+3, 300)):
                addr2 = x22 + (j * 0x38)
                d2 = cpu.read_memory(addr2, 12)
                cm2 = int.from_bytes(d2[0:4], 'little', signed=True)
                print(f"    [{j:>3}] 0x{addr2:X}: code_min={cm2}")
            break

    # Also scan from x2 (current cursor)
    print(f"\n  Scanning from x2=0x{x2:X} (current cursor):")
    for i in range(50):
        addr = x2 + (i * 0x38)
        if addr >= cpu.memory_size:
            print(f"    Hit memory boundary at entry {i}")
            break
        data = cpu.read_memory(addr, 4)
        code_min = int.from_bytes(data, 'little', signed=True)
        if code_min != 0:
            print(f"    [{i:>3}] 0x{addr:X}: code_min={code_min}")
        if code_min == -1:
            print(f"    *** TERMINAL at entry {i} ***")
            break

    # Check: is x2 > the terminal entry?
    # If x2 already passed the terminal, the loop will never see it!
    # Find terminal address from x22 scan
    terminal_addr = None
    for i in range(1000):
        addr = x22 + (i * 0x38)
        if addr >= cpu.memory_size:
            break
        data = cpu.read_memory(addr, 4)
        if int.from_bytes(data, 'little', signed=True) == -1:
            terminal_addr = addr
            break

    if terminal_addr:
        print(f"\n  Terminal marker at 0x{terminal_addr:X}")
        print(f"  Current cursor x2 at 0x{x2:X}")
        if x2 > terminal_addr:
            print(f"  *** x2 HAS ALREADY PASSED THE TERMINAL! ***")
            print(f"  The cursor is 0x{x2 - terminal_addr:X} bytes past the terminal")
            print(f"  This is {(x2 - terminal_addr) // 0x38} entries past the terminal")
        elif x2 <= terminal_addr:
            entries_to_go = (terminal_addr - x2) // 0x38
            print(f"  x2 is {entries_to_go} entries before the terminal")
            print(f"  The loop SHOULD find the terminal... unless reads are returning 0")
    else:
        print(f"\n  NO TERMINAL MARKER FOUND IN TABLE!")

    # Now the crucial test: read what the GPU would read at x2
    # and at subsequent addresses to see if the data matches what we expect
    print(f"\n  Verifying table data at x2 (what GPU will read):")
    for i in range(10):
        addr = x2 + (i * 0x38)
        # Read via the CPU's read_memory (which reads from the numpy buffer)
        data = cpu.read_memory(addr, 8)
        word0 = int.from_bytes(data[0:4], 'little', signed=True)
        word1 = int.from_bytes(data[4:8], 'little', signed=True)
        print(f"    0x{addr:X}: word0={word0:>12} (0x{word0 & 0xFFFFFFFF:08X}), "
              f"word1={word1:>12} (0x{word1 & 0xFFFFFFFF:08X})")


if __name__ == "__main__":
    main()
