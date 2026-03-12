#!/usr/bin/env python3
"""
Verify the NFA table state at the point where the stuck loop begins.

Key findings:
- x22 = 0x8761A0 (used by stuck loop at 0x4261F4)
- x26 = 0x871938 (used by SMADDL for address calc in builder)
- x23 = 0x876130 (points to builder's input AST/tree nodes)

x22 and x26 are the NFA transition table, but the stuck loop walks x22.
The SMADDL in the builder writes entries relative to x26.

Questions:
1. Are x22 and x26 pointing to the same logical table?
2. What does the table look like at x22 when the loop starts?
3. Is there a terminal marker (code_min=-1) that the loop should find?
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

    # Run until stuck loop
    total_cycles = 0
    while total_cycles < 100_000:
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
            continue

        if cpu.pc == 0x4261F4 or cpu.pc == 0x426204:
            print(f"Stuck loop reached at cycle {total_cycles:,}, PC=0x{cpu.pc:X}")
            break

    # Dump all registers
    print("\n  All key registers:")
    for i in range(31):
        val = cpu.get_register(i) & 0xFFFFFFFFFFFFFFFF
        if val != 0:
            print(f"    x{i} = 0x{val:016X}")
    sp = cpu.get_register(31) & 0xFFFFFFFFFFFFFFFF
    print(f"    SP = 0x{sp:016X}")

    x2 = cpu.get_register(2) & 0xFFFFFFFFFFFFFFFF
    x3 = cpu.get_register(3) & 0xFFFFFFFFFFFFFFFF
    x22 = cpu.get_register(22) & 0xFFFFFFFFFFFFFFFF
    x23 = cpu.get_register(23) & 0xFFFFFFFFFFFFFFFF
    x26 = cpu.get_register(26) & 0xFFFFFFFFFFFFFFFF
    x27 = cpu.get_register(27) & 0xFFFFFFFFFFFFFFFF
    x28 = cpu.get_register(28) & 0xFFFFFFFFFFFFFFFF

    print(f"\n  x22 = 0x{x22:X} (stuck loop table base)")
    print(f"  x26 = 0x{x26:X} (SMADDL table base)")
    print(f"  x23 = 0x{x23:X} (index pointer)")
    print(f"  x2  = 0x{x2:X} (loop cursor)")
    print(f"  x3  = 0x{x3:X} (count array)")
    print(f"  x27 = 0x{x27:X}")
    print(f"  x28 = 0x{x28:X}")

    # Verify: what's at x22?
    print(f"\n  === Table at x22=0x{x22:X} (stride 0x38) ===")
    for i in range(20):
        addr = x22 + i * 0x38
        if addr + 0x38 > cpu.memory_size:
            break
        data = cpu.read_memory(addr, 0x38)
        code_min = int.from_bytes(data[0:4], 'little', signed=True)
        code_max = int.from_bytes(data[4:8], 'little', signed=True)
        state_ptr = int.from_bytes(data[8:16], 'little')
        state_id = int.from_bytes(data[16:20], 'little', signed=True)
        field_20 = int.from_bytes(data[32:36], 'little')
        field_24 = int.from_bytes(data[36:40], 'little', signed=True)
        field_28 = int.from_bytes(data[40:44], 'little', signed=True)
        field_30 = int.from_bytes(data[48:56], 'little')

        nonzero = sum(1 for b in data if b != 0)
        if nonzero > 0 or i < 5:
            hex_str = ' '.join(f'{b:02X}' for b in data[:16])
            hex_str2 = ' '.join(f'{b:02X}' for b in data[16:32])
            hex_str3 = ' '.join(f'{b:02X}' for b in data[32:48])
            hex_str4 = ' '.join(f'{b:02X}' for b in data[48:56])
            print(f"    [{i:>3}] 0x{addr:X}: code_min={code_min:>6} code_max={code_max:>6} "
                  f"state=0x{state_ptr:X} id={state_id}")
            print(f"          {hex_str}")
            print(f"          {hex_str2}")
            print(f"          {hex_str3}")
            print(f"          {hex_str4}")
        if code_min == -1:
            print(f"    *** TERMINAL at entry [{i}] ***")
            break
    else:
        print(f"    ... NO TERMINAL FOUND in first 20 entries")

    # Also check the x26-based table
    print(f"\n  === Table at x26=0x{x26:X} (stride 0x38) ===")
    for i in range(20):
        addr = x26 + i * 0x38
        if addr + 4 > cpu.memory_size:
            break
        data = cpu.read_memory(addr, min(0x38, cpu.memory_size - addr))
        code_min = int.from_bytes(data[0:4], 'little', signed=True)
        nonzero = sum(1 for b in data if b != 0)
        if nonzero > 0 or i < 5:
            hex16 = ' '.join(f'{b:02X}' for b in data[:min(16, len(data))])
            print(f"    [{i:>3}] 0x{addr:X}: code_min={code_min:>6}  {hex16}")
        if code_min == -1:
            print(f"    *** TERMINAL at entry [{i}] ***")
            break
    else:
        print(f"    ... NO TERMINAL in first 20 entries")

    # Scan ALL memory for 0xFFFFFFFF at stride-aligned positions from x22
    print(f"\n  === Scanning for terminal markers (0xFFFFFFFF) ===")
    # Check from x22 out to x22 + 200*0x38
    for i in range(200):
        addr = x22 + i * 0x38
        if addr + 4 > cpu.memory_size:
            break
        data = cpu.read_memory(addr, 4)
        val = int.from_bytes(data, 'little', signed=True)
        if val == -1:
            print(f"    entry[{i}] @ 0x{addr:X}: code_min = -1 *** TERMINAL ***")

    # Also scan from x26
    print(f"\n  Scanning from x26=0x{x26:X}:")
    for i in range(200):
        addr = x26 + i * 0x38
        if addr + 4 > cpu.memory_size:
            break
        data = cpu.read_memory(addr, 4)
        val = int.from_bytes(data, 'little', signed=True)
        if val == -1:
            print(f"    entry[{i}] @ 0x{addr:X}: code_min = -1 *** TERMINAL ***")

    # Check the relationship between x22 and x26
    diff = x22 - x26 if x22 >= x26 else -(x26 - x22)
    entries_diff = diff / 0x38
    print(f"\n  x22 - x26 = 0x{diff:X} = {diff} bytes = {entries_diff:.2f} entries")

    # The stuck loop reads [x2] where x2 advances by 0x38.
    # x2 starts at x22 (or wherever it was set). Let's check:
    # At 0x426220: mov x2, x22  ; sets x2 = x22
    # Then at 0x426208: ldr w0, [x2]
    # Then at 0x42620C: add x2, x2, #0x38
    # Then at 0x426210: tbz w0, #31, 0x4261F4  (loop if bit31=0)
    # So x2 walks the table starting from x22.

    # Let's check what x2 points to right now
    print(f"\n  x2 = 0x{x2:X}")
    if x2 >= x22:
        entries_walked = (x2 - x22) // 0x38
        print(f"  Entries walked: {entries_walked}")

    # Dump memory at the current x2 position
    if x2 + 16 < cpu.memory_size:
        data = cpu.read_memory(x2, 16)
        hex_str = ' '.join(f'{b:02X}' for b in data)
        val = int.from_bytes(data[0:4], 'little', signed=True)
        print(f"  [x2] = {hex_str}  (code_min={val})")

    # Check: is there ANY 0xFFFFFFFF in the range x22 to x22 + 10000?
    print(f"\n  Raw scan for 0xFFFFFFFF words in range 0x{x22:X} to 0x{min(x22 + 0x3000, cpu.memory_size):X}:")
    count = 0
    for offset in range(0, min(0x3000, cpu.memory_size - x22), 4):
        addr = x22 + offset
        data = cpu.read_memory(addr, 4)
        val = int.from_bytes(data, 'little')
        if val == 0xFFFFFFFF:
            # What entry/field is this relative to x22?
            entry_idx = offset // 0x38
            field_off = offset % 0x38
            print(f"    0x{addr:X} (entry[{entry_idx}] + 0x{field_off:02X}): 0xFFFFFFFF")
            count += 1
    print(f"  Found {count} instances of 0xFFFFFFFF")


if __name__ == "__main__":
    main()
