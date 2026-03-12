#!/usr/bin/env python3
"""
Fast terminal scan: run to the stuck loop with larger batches,
then scan the table from x22 for terminal marker.
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

    # Run with normal batch size until we hit the stuck PC
    total_cycles = 0
    batch_size = 1000  # Moderate size for speed

    while total_cycles < 100_000:
        result = cpu.execute(max_cycles=batch_size)
        total_cycles += result.cycles

        for fd, data in cpu.drain_svc_buffer():
            pass

        if result.stop_reason == StopReasonV2.SYSCALL:
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

        # Check if in stuck loop
        if cpu.pc == 0x426204 or cpu.pc == 0x4261F4:
            print(f"In stuck loop at cycle {total_cycles:,}, PC=0x{cpu.pc:X}")
            break

    x0 = cpu.get_register(0) & 0xFFFFFFFFFFFFFFFF
    x1 = cpu.get_register(1) & 0xFFFFFFFFFFFFFFFF
    x2 = cpu.get_register(2) & 0xFFFFFFFFFFFFFFFF
    x3 = cpu.get_register(3) & 0xFFFFFFFFFFFFFFFF
    x22 = cpu.get_register(22) & 0xFFFFFFFFFFFFFFFF
    x23 = cpu.get_register(23) & 0xFFFFFFFFFFFFFFFF

    print(f"\n  x0={x0:#x} x1={x1:#x} x2={x2:#x} x3={x3:#x}")
    print(f"  x22={x22:#x} x23={x23:#x}")

    # Scan from x22 for terminal
    print(f"\n  Scanning table entries from x22=0x{x22:X}:")
    terminal_idx = None
    nonzero_entries = 0
    for i in range(10000):
        addr = x22 + (i * 0x38)
        if addr >= cpu.memory_size:
            print(f"  Hit memory boundary at entry {i}")
            break
        data = cpu.read_memory(addr, 4)
        code_min = int.from_bytes(data, 'little', signed=True)
        if code_min != 0:
            nonzero_entries += 1
            if nonzero_entries <= 30:
                print(f"    [{i:>4}] 0x{addr:X}: code_min = {code_min} (0x{code_min & 0xFFFFFFFF:08X})")
        if code_min == -1:
            terminal_idx = i
            print(f"    *** TERMINAL at entry {i}, addr 0x{addr:X} ***")
            break

    if terminal_idx is None:
        print(f"\n  NO TERMINAL FOUND! Total nonzero entries: {nonzero_entries}")

        # Check: is x22 even pointing at the right table?
        # Let's look at memory around x22
        print(f"\n  Memory hexdump at x22=0x{x22:X}:")
        for j in range(4):
            addr = x22 + (j * 16)
            data = cpu.read_memory(addr, 16)
            hex_str = ' '.join(f'{b:02X}' for b in data)
            print(f"    0x{addr:X}: {hex_str}")
    else:
        print(f"\n  Terminal found at entry {terminal_idx}")
        print(f"  Current x2 cursor at 0x{x2:X}")
        distance = (x2 - x22) // 0x38
        print(f"  x2 is at entry {distance} (distance from x22)")
        if distance > terminal_idx:
            print(f"  *** x2 has PASSED the terminal by {distance - terminal_idx} entries! ***")
        else:
            print(f"  x2 is {terminal_idx - distance} entries before the terminal")
            # Read entries from x2 to terminal
            print(f"\n  Entries from x2 to terminal:")
            for i in range(min(terminal_idx - distance + 2, 20)):
                addr = x2 + (i * 0x38)
                data = cpu.read_memory(addr, 4)
                cm = int.from_bytes(data, 'little', signed=True)
                print(f"    [{distance + i:>4}] 0x{addr:X}: code_min = {cm} (0x{cm & 0xFFFFFFFF:08X})")

    # Also: check what's at the addresses the loop has been reading
    # x2 advances by 0x38 each iteration. From the loop trace, x2 starts
    # around x22 and goes way past the terminal.
    # The question is: what does the GPU read at these addresses?

    # Let me check: are there 0xFFFFFFFF values at stride-aligned positions?
    print(f"\n  All stride-aligned terminal markers (code_min=-1) in 0x570000-0x600000:")
    for addr in range(0x570000, min(0x600000, cpu.memory_size), 0x38):
        data = cpu.read_memory(addr, 4)
        cm = int.from_bytes(data, 'little', signed=True)
        if cm == -1:
            rel = (addr - x22) // 0x38 if addr >= x22 else -(x22 - addr) // 0x38
            print(f"    0x{addr:X}: -1 (relative entry {rel} from x22)")


if __name__ == "__main__":
    main()
