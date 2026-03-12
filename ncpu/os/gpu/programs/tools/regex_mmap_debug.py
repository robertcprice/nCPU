#!/usr/bin/env python3
"""
Deep diagnostic: trace ALL mmap/brk syscalls and memory writes during
regcomp to understand why the NFA table is not populated.

The hypothesis: mmap returns addresses, but the NFA builder writes
to those addresses using instructions that silently fail, so the
table is never populated.

We'll focus on:
1. What addresses mmap returns
2. What gets written to those addresses
3. Whether writes to mmap'd regions actually persist
"""

import sys
import struct
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kernels.mlx.cpu_kernel_v2 import MLXKernelCPUv2, StopReasonV2
from ncpu.os.gpu.elf_loader import load_elf_into_memory, parse_elf, make_busybox_syscall_handler
from ncpu.os.gpu.filesystem import GPUFilesystem

BUSYBOX = str(PROJECT_ROOT / "demos" / "busybox.elf")


def main():
    print("=" * 70)
    print("  MMAP / MEMORY WRITE DIAGNOSTIC")
    print("  Tracing regcomp memory allocation and writes")
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

    # We need to hook into the syscall handler to track mmap calls
    handler = make_busybox_syscall_handler(filesystem=fs, heap_base=heap_base)

    if cpu.memory_size > cpu.SVC_BUF_BASE + 0x10000:
        cpu.init_svc_buffer()

    # Run with custom tracking
    total_cycles = 0
    max_total = 25_000  # Just up to when the loop starts

    mmap_calls = []
    brk_calls = []
    all_syscalls = []
    batch_size = 500

    while total_cycles < max_total:
        result = cpu.execute(max_cycles=batch_size)
        total_cycles += result.cycles

        for fd, data in cpu.drain_svc_buffer():
            text = data.decode('ascii', errors='replace')
            if text.strip():
                print(f"  [output] {text.strip()}")

        if result.stop_reason == StopReasonV2.HALT:
            print(f"HALT at cycle {total_cycles:,}")
            break
        elif result.stop_reason == StopReasonV2.SYSCALL:
            sysno = cpu.get_register(8)
            x0 = cpu.get_register(0)
            x1 = cpu.get_register(1)
            x2 = cpu.get_register(2)
            x3 = cpu.get_register(3)
            x4 = cpu.get_register(4)
            x5 = cpu.get_register(5)
            pc = cpu.pc

            all_syscalls.append({
                'cycle': total_cycles,
                'sysno': sysno,
                'x0': x0, 'x1': x1, 'x2': x2,
                'x3': x3, 'x4': x4, 'x5': x5,
                'pc': pc,
            })

            # Handle the syscall
            ret = handler(cpu)

            # After handler, get return value
            ret_val = cpu.get_register(0)

            if sysno == 222:  # mmap
                mmap_calls.append({
                    'cycle': total_cycles,
                    'addr': x0,
                    'length': x1,
                    'prot': x2,
                    'flags': x3,
                    'fd': x4,
                    'offset': x5,
                    'result': ret_val,
                    'pc': pc,
                })
                print(f"  cycle {total_cycles:>6}: mmap(addr=0x{x0 & 0xFFFFFFFF:X}, "
                      f"len=0x{x1 & 0xFFFFFFFF:X}, prot={x2}, flags=0x{x3 & 0xFFFFFFFF:X}) "
                      f"= 0x{ret_val & 0xFFFFFFFFFFFFFFFF:X}")
            elif sysno == 214:  # brk
                brk_calls.append({
                    'cycle': total_cycles,
                    'addr': x0,
                    'result': ret_val,
                })
                print(f"  cycle {total_cycles:>6}: brk(0x{x0 & 0xFFFFFFFF:X}) = 0x{ret_val & 0xFFFFFFFFFFFFFFFF:X}")
            elif sysno == 56:  # openat
                print(f"  cycle {total_cycles:>6}: openat(dirfd={x0}, path=0x{x1 & 0xFFFFFFFF:X}, flags={x2}) = {ret_val}")
            elif sysno == 63:  # read
                print(f"  cycle {total_cycles:>6}: read(fd={x0}, buf=0x{x1 & 0xFFFFFFFF:X}, len={x2}) = {ret_val}")
            elif sysno == 64:  # write
                pass  # Don't spam
            elif sysno == 96:  # set_tid_address
                print(f"  cycle {total_cycles:>6}: set_tid_address(0x{x0 & 0xFFFFFFFF:X}) = {ret_val}")
            else:
                print(f"  cycle {total_cycles:>6}: syscall {sysno}(x0=0x{x0 & 0xFFFFFFFF:X}) = {ret_val}")

            if ret == False:
                print(f"  EXIT")
                break
            elif ret == "exec":
                continue
            cpu.set_pc(cpu.pc + 4)

    # Summary
    print(f"\n  Total cycles: {total_cycles:,}")
    print(f"  Final PC: 0x{cpu.pc:X}")
    print(f"  mmap calls: {len(mmap_calls)}")
    print(f"  brk calls: {len(brk_calls)}")

    # Check what's in the mmap'd regions
    print(f"\n  Checking mmap'd regions for content:")
    for m in mmap_calls:
        result_addr = m['result'] & 0xFFFFFFFFFFFFFFFF
        length = m['length'] & 0xFFFFFFFF
        # Read first 64 bytes
        data = cpu.read_memory(result_addr, min(64, length))
        nonzero = sum(1 for b in data if b != 0)
        print(f"    mmap @ 0x{result_addr:X} (len=0x{length:X}): "
              f"{nonzero}/{len(data)} non-zero bytes in first {len(data)} bytes")
        if nonzero > 0:
            print(f"      First bytes: {data[:32].hex()}")

    # Check the specific NFA table region
    # From the loop diagnostic, x3 = 0x5702D0 (count array), x2 starts around 0x576030
    print(f"\n  Checking NFA table region (0x570000-0x580000):")
    for addr in range(0x570000, 0x580000, 0x1000):
        data = cpu.read_memory(addr, 256)
        nonzero = sum(1 for b in data if b != 0)
        if nonzero > 0:
            print(f"    0x{addr:X}: {nonzero}/256 non-zero bytes")

    # Double-buffer diagnostic: check if memory_in and memory_out diverge
    print(f"\n  GPU Memory diagnostics:")
    print(f"    Memory size: {cpu.memory_size:,} bytes ({cpu.memory_size / (1024*1024):.1f} MB)")

    # Check if the NFA table addresses fall within GPU memory
    nfa_table_addr = 0x5702D0  # x3 from the trace
    print(f"    NFA table base (x3): 0x{nfa_table_addr:X}")
    if nfa_table_addr < cpu.memory_size:
        print(f"    -> Within GPU memory bounds")
    else:
        print(f"    -> *** OUTSIDE GPU MEMORY BOUNDS! ***")
        print(f"    This means writes to this address are SILENTLY DROPPED!")

    # Check brk/mmap results against memory size
    for m in mmap_calls:
        result_addr = m['result'] & 0xFFFFFFFFFFFFFFFF
        end_addr = result_addr + (m['length'] & 0xFFFFFFFF)
        in_bounds = end_addr < cpu.memory_size
        status = "OK" if in_bounds else "*** OUT OF BOUNDS ***"
        print(f"    mmap 0x{result_addr:X}-0x{end_addr:X}: {status}")


if __name__ == "__main__":
    main()
