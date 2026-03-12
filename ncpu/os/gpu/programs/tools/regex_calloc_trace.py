#!/usr/bin/env python3
"""
Trace ALL memory allocation and writes during regcomp (the regex compiler).

regcomp uses calloc/malloc (which calls mmap/brk on musl).
We need to see what happens between the mmap calls and the stuck loop.

Also: verify the hypothesis that 0x8761A0 is the right table address.
The mmap at cycle 14915 returns 0x576000 (or 0x876000 depending on heap_base).
Wait -- 0x876000 vs 0x576000! Let me check the heap_base calculation.
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

    print(f"ELF entry: 0x{entry:X}")
    print(f"Segments:")
    for s in elf_info.segments:
        print(f"  0x{s.vaddr:X} - 0x{s.vaddr + s.memsz:X} (filesz={s.filesz}, memsz={s.memsz})")
    print(f"Max segment end: 0x{max_seg_end:X}")
    print(f"Heap base: 0x{heap_base:X}")
    print(f"Memory size: {cpu.memory_size:,} bytes")

    handler = make_busybox_syscall_handler(filesystem=fs, heap_base=heap_base)

    if cpu.memory_size > cpu.SVC_BUF_BASE + 0x10000:
        cpu.init_svc_buffer()

    # Run and track all mmap returns
    total_cycles = 0
    batch_size = 500

    while total_cycles < 16_000:
        result = cpu.execute(max_cycles=batch_size)
        total_cycles += result.cycles

        for fd, data in cpu.drain_svc_buffer():
            pass

        if result.stop_reason == StopReasonV2.HALT:
            print(f"HALT at cycle {total_cycles:,}")
            return
        elif result.stop_reason == StopReasonV2.SYSCALL:
            sysno = cpu.get_register(8)
            x0 = cpu.get_register(0)
            x1 = cpu.get_register(1)

            ret = handler(cpu)
            ret_val = cpu.get_register(0) & 0xFFFFFFFFFFFFFFFF

            if sysno == 222:  # mmap
                print(f"  cycle {total_cycles:>6}: mmap(addr=0x{x0 & 0xFFFFFFFF:X}, "
                      f"len=0x{x1 & 0xFFFFFFFF:X}) -> 0x{ret_val:X}")
            elif sysno == 214:  # brk
                print(f"  cycle {total_cycles:>6}: brk(0x{x0 & 0xFFFFFFFF:X}) -> 0x{ret_val:X}")

            if ret == False:
                return
            elif ret == "exec":
                continue
            cpu.set_pc(cpu.pc + 4)

    # Check state
    x22 = cpu.get_register(22) & 0xFFFFFFFFFFFFFFFF
    x23 = cpu.get_register(23) & 0xFFFFFFFFFFFFFFFF
    x2 = cpu.get_register(2) & 0xFFFFFFFFFFFFFFFF
    x3 = cpu.get_register(3) & 0xFFFFFFFFFFFFFFFF

    print(f"\n  At cycle {total_cycles:,}, PC=0x{cpu.pc:X}")
    print(f"  x2=0x{x2:X} x3=0x{x3:X} x22=0x{x22:X} x23=0x{x23:X}")

    # Scan the FULL memory for the NFA table structure
    # Look for a pattern: two entries with code_min=5 at stride 0x38
    print(f"\n  Searching for NFA table (two entries with value 5 at stride 0x70):")
    for base in range(0x400000, cpu.memory_size - 0x80, 4):
        d0 = cpu.read_memory(base, 4)
        v0 = int.from_bytes(d0, 'little')
        if v0 == 5:
            # Check if there's another 5 at base + 2*0x38 = base + 0x70
            d1 = cpu.read_memory(base + 0x70, 4)
            v1 = int.from_bytes(d1, 'little')
            if v1 == 5:
                # Check if base + 0x38 is 0
                d_mid = cpu.read_memory(base + 0x38, 4)
                v_mid = int.from_bytes(d_mid, 'little')
                if v_mid == 0:
                    print(f"    Found pattern at 0x{base:X}: "
                          f"[0]=5, [1]=0, [2]=5")
                    # Check what's at offset 0x30 from this base
                    d30 = cpu.read_memory(base + 0x30, 4)
                    v30 = int.from_bytes(d30, 'little', signed=True)
                    print(f"    Offset 0x30: 0x{v30 & 0xFFFFFFFF:08X} (signed: {v30})")
                    # Dump the full first entry
                    full_data = cpu.read_memory(base, 0x38)
                    hex_str = ' '.join(f'{b:02X}' for b in full_data)
                    print(f"    Entry[0]: {hex_str}")

                    # Now look for terminal marker in this table
                    for i in range(200):
                        addr = base + (i * 0x38)
                        d = cpu.read_memory(addr, 4)
                        cm = int.from_bytes(d, 'little', signed=True)
                        if cm == -1:
                            print(f"    Terminal at entry [{i}], addr 0x{addr:X}")
                            break
                    else:
                        print(f"    No terminal in 200 entries")

                    # Only show first match
                    break
        # Skip ahead to avoid scanning every byte (performance)
        if base % 0x100000 == 0 and base > 0x400000:
            break


if __name__ == "__main__":
    main()
