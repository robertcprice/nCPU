#!/usr/bin/env python3
"""
Dump the full NFA table structure to understand the layout and find
why terminal markers are not at the right offsets.
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
    print("  NFA TABLE STRUCTURE DUMP")
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

    # Run to just before the stuck loop
    total_cycles = 0
    while total_cycles < 15_000:
        result = cpu.execute(max_cycles=500)
        total_cycles += result.cycles
        for fd, data in cpu.drain_svc_buffer():
            pass
        if result.stop_reason == StopReasonV2.HALT:
            break
        elif result.stop_reason == StopReasonV2.SYSCALL:
            ret = handler(cpu)
            if ret == False:
                break
            elif ret == "exec":
                continue
            cpu.set_pc(cpu.pc + 4)

    # x22 = table base, stride 0x38
    x22 = cpu.get_register(22) & 0xFFFFFFFFFFFFFFFF
    x23 = cpu.get_register(23) & 0xFFFFFFFFFFFFFFFF
    print(f"\n  x22 (table base) = 0x{x22:X}")
    print(f"  x23 (index ptr)  = 0x{x23:X}")

    # Dump full struct for each table entry (0x38 = 56 bytes)
    print(f"\n  Full table entries (struct size 0x38 = 56 bytes):")
    print(f"  {'Entry':>5} {'Addr':>10} | {'W0':>10} {'W1':>10} {'W2':>10} {'W3':>10} "
          f"{'W4':>10} {'W5':>10} {'W6':>10} {'W7':>10} "
          f"{'W8':>10} {'W9':>10} {'WA':>10} {'WB':>10} {'WC':>10} {'WD':>10}")
    print(f"  {'':>5} {'':>10} | {'off=0':>10} {'off=4':>10} {'off=8':>10} {'off=C':>10} "
          f"{'off=10':>10} {'off=14':>10} {'off=18':>10} {'off=1C':>10} "
          f"{'off=20':>10} {'off=24':>10} {'off=28':>10} {'off=2C':>10} {'off=30':>10} {'off=34':>10}")
    print(f"  " + "-" * 180)

    for i in range(20):
        addr = x22 + (i * 0x38)
        if addr + 0x38 > cpu.memory_size:
            break
        data = cpu.read_memory(addr, 0x38)
        words = []
        for j in range(0, 0x38, 4):
            w = int.from_bytes(data[j:j+4], 'little')
            words.append(w)

        # Check if any word is 0xFFFFFFFF
        has_terminal = any(w == 0xFFFFFFFF for w in words)
        marker = " <-- HAS -1" if has_terminal else ""

        word_strs = [f"0x{w:08X}" for w in words]
        print(f"  [{i:>3}] 0x{addr:08X} | {' '.join(word_strs)}{marker}")

    # Also look BEFORE x22 - what's at x22-0x70?
    print(f"\n  Memory before table base (x22-0x70 to x22):")
    for i in range(-2, 0):
        addr = x22 + (i * 0x38)
        data = cpu.read_memory(addr, 0x38)
        words = []
        for j in range(0, 0x38, 4):
            w = int.from_bytes(data[j:j+4], 'little')
            words.append(w)
        has_terminal = any(w == 0xFFFFFFFF for w in words)
        marker = " <-- HAS -1" if has_terminal else ""
        word_strs = [f"0x{w:08X}" for w in words]
        print(f"  [{i:>3}] 0x{addr:08X} | {' '.join(word_strs)}{marker}")

    # Now let's understand the TRE tnfa_transition struct
    # In musl's TRE implementation:
    # typedef struct tnfa_transition {
    #     tre_cint_t code_min;     // int32_t, offset 0
    #     tre_cint_t code_max;     // int32_t, offset 4
    #     int state;               // offset 8
    #     int *tags;               // offset 16 (pointer, 8 bytes on aarch64)
    #     int assertions;          // offset 24
    #     union { ... } u;         // offset 32 (pointer to class or params)
    #     tre_tnfa_t *state_ptr;   // offset 40 (pointer)
    #     ...
    # };
    # sizeof(tnfa_transition) on aarch64 = 0x38 = 56 bytes
    #
    # The terminal marker is code_min = -1 (offset 0)
    # Let's check if our first-word values make sense:

    print(f"\n  TRE tnfa_transition struct interpretation:")
    print(f"  sizeof = 0x38 (56 bytes) on aarch64")
    print(f"  Offset 0x00: code_min (int32) -- terminal when == -1 (0xFFFFFFFF)")
    print(f"  Offset 0x04: code_max (int32)")
    print(f"  Offset 0x08: state (int32)")
    print(f"  Offset 0x0C: padding")
    print(f"  Offset 0x10: *tags (pointer, 8 bytes)")
    print(f"  Offset 0x18: assertions (int32)")
    print(f"  Offset 0x1C: padding")
    print("  Offset 0x20: union { *class, *neg_classes, *params } (pointer, 8 bytes)")
    print(f"  Offset 0x28: *state_ptr (pointer, 8 bytes)")
    print(f"  Offset 0x30: tags_seen? or backref?")

    # Key question: entry[0] at x22 has first word = 5
    # This means code_min = 5 for "hello" regex
    # 5 is a low character code... wait, musl's TRE uses wchar_t codes
    # For regex "hello", the transitions should be:
    #   state 0 -> state 1 on 'h' (code 104)
    #   state 1 -> state 2 on 'e' (code 101)
    #   etc.
    # But code_min = 5 doesn't match any of h,e,l,o characters!
    # Unless it's using some internal encoding...

    # Actually wait, entry[0] code_min = 5, but entries [1],[3],[4]... are all 0
    # Let me check: maybe the table ISN'T at x22
    # Let me re-examine the loop more carefully

    # The stuck loop at 0x4261f4:
    #   ldrsw x0, [x23]       ; x23 = 0x576130, loads index
    #   lsl   x0, x0, #2      ; index * 4
    #   ldr   w1, [x3, x0]    ; x3 = 0x5702D0, loads count[index]
    #   add   w1, w1, #0x1    ; increment
    #   str   w1, [x3, x0]    ; store back
    #   ldr   w0, [x2]        ; x2 = table pointer, loads first word of entry
    #   add   x2, x2, #0x38   ; advance to next entry
    #   tbz   w0, #31, 0x4261f4 ; loop if bit 31 clear

    # So the loop is counting states, iterating over transitions.
    # It reads [x2] which is the code_min field. Terminal is code_min = -1.
    # The table at x22 = 0x5761A0 should have been built by regcomp.

    # Let's look more carefully at the data values
    print(f"\n  Detailed entry [0] at 0x{x22:X}:")
    data = cpu.read_memory(x22, 0x38)
    code_min = int.from_bytes(data[0:4], 'little', signed=True)
    code_max = int.from_bytes(data[4:8], 'little', signed=True)
    state = int.from_bytes(data[8:12], 'little', signed=True)
    tags_ptr = int.from_bytes(data[16:24], 'little')
    assertions = int.from_bytes(data[24:28], 'little', signed=True)

    print(f"    code_min   = {code_min} (0x{data[0:4].hex()})")
    print(f"    code_max   = {code_max} (0x{data[4:8].hex()})")
    print(f"    state      = {state}")
    print(f"    tags       = 0x{tags_ptr:X}")
    print(f"    assertions = {assertions}")

    # Entry [2] also has code_min = 5
    data2 = cpu.read_memory(x22 + 2*0x38, 0x38)
    code_min2 = int.from_bytes(data2[0:4], 'little', signed=True)
    code_max2 = int.from_bytes(data2[4:8], 'little', signed=True)
    state2 = int.from_bytes(data2[8:12], 'little', signed=True)
    print(f"\n  Detailed entry [2] at 0x{x22+2*0x38:X}:")
    print(f"    code_min   = {code_min2}")
    print(f"    code_max   = {code_max2}")
    print(f"    state      = {state2}")

    # What's in the memory just after the last populated entry?
    # The 0xFFFFFFFF values we found earlier were at:
    # 0x576080, 0x576090 (before x22 = 0x5761A0)
    # 0x5761D0, 0x5761E0 (at offset 0x30 and 0x40 from x22)
    # Let's check: 0x5761D0 = x22 + 0x30
    # That's offset 0x30 within entry[0], which would be the tags_seen or
    # backref field, not code_min!

    print(f"\n  0xFFFFFFFF at 0x5761D0 is at x22 + 0x30:")
    print(f"    This is at offset 0x30 within entry[0]")
    print(f"    It's NOT the code_min field (offset 0x00)")
    print(f"    It's a different field in the struct")

    # So the terminal markers ARE there, but at wrong offsets?
    # Let me check: is there a 0xFFFFFFFF at the code_min (first word) position
    # of ANY entry in the first 20?
    print(f"\n  Checking code_min (first word) of each entry for -1:")
    for i in range(20):
        addr = x22 + (i * 0x38)
        data = cpu.read_memory(addr, 4)
        w = int.from_bytes(data, 'little')
        if w != 0:
            sw = int.from_bytes(data, 'little', signed=True)
            print(f"    Entry [{i:>2}] at 0x{addr:X}: code_min = {sw} (0x{w:08X})")

    # The critical test: is there ANY entry with code_min = -1?
    print(f"\n  Scanning 500 entries for code_min = -1:")
    found = False
    for i in range(500):
        addr = x22 + (i * 0x38)
        if addr >= cpu.memory_size:
            break
        data = cpu.read_memory(addr, 4)
        w = int.from_bytes(data, 'little', signed=True)
        if w == -1:
            print(f"    FOUND at entry [{i}], addr 0x{addr:X}")
            found = True
            break
    if not found:
        print(f"    NOT FOUND in 500 entries!")
        print(f"    This confirms the terminal marker was never written at offset 0")
        print(f"    of any transition entry")


if __name__ == "__main__":
    main()
