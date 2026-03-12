#!/usr/bin/env python3
"""Trace regcomp execution to find where wchar_t pattern is created.

Static disassembly corrected the earlier trace target:
- 0x0041A374 is grep's regex compile call site
- 0x0041E4AC is BusyBox's compile-and-report wrapper
- 0x00428C4C is the real musl regcomp entry
- 0x00430ED4 is just a string helper, not regcomp
"""
import sys, struct
sys.path.insert(0, "/Users/bobbyprice/projects/nCPU")

from ncpu.os.gpu.elf_loader import load_elf_into_memory, parse_elf, make_busybox_syscall_handler
from ncpu.os.gpu.filesystem import GPUFilesystem
from kernels.mlx.gpu_cpu import GPUKernelCPU
from kernels.mlx.rust_runner import StopReasonV2
from pathlib import Path

ELF_PATH = "demos/busybox.elf"
GREP_COMPILE_CALL = 0x0041A374
REGCOMP_WRAPPER = 0x0041E4AC
REGCOMP_ENTRY = 0x00428C4C
NFA_CODE_MIN_LOAD = 0x00425FA8


def setup():
    cpu = GPUKernelCPU(quiet=True)
    fs = GPUFilesystem()
    fs.write_file("/etc/passwd", b"root:x:0:0:root:/root:/bin/sh\n")
    entry = load_elf_into_memory(cpu, ELF_PATH, argv=["grep", "root", "/etc/passwd"], quiet=True)
    cpu.set_pc(entry)
    elf_data = Path(ELF_PATH).read_bytes()
    elf_info = parse_elf(elf_data)
    max_seg_end = max(s.vaddr + s.memsz for s in elf_info.segments)
    heap_base = (max_seg_end + 0xFFFF) & ~0xFFFF
    cpu.write_memory(0x3F0008, struct.pack('<Q', heap_base))
    handler = make_busybox_syscall_handler(filesystem=fs, heap_base=heap_base)
    if cpu.memory_size > cpu.SVC_BUF_BASE + 0x10000:
        cpu.init_svc_buffer()
    return cpu, handler


def run_to_bp(cpu, handler, max_cycles=100_000):
    total = 0
    while total < max_cycles:
        result = cpu.execute(max_cycles=1000)
        total += result.cycles
        for fd, data in cpu.drain_svc_buffer():
            pass
        if result.stop_reason == StopReasonV2.BREAKPOINT:
            return total, "BP"
        elif result.stop_reason == StopReasonV2.WATCHPOINT:
            return total, "WP"
        elif result.stop_reason == StopReasonV2.SYSCALL:
            ret = handler(cpu)
            if ret == False:
                return total, "EXIT"
            elif ret != "exec":
                cpu.set_pc(cpu.pc + 4)
        elif result.stop_reason == StopReasonV2.HALT:
            return total, "HALT"
    return total, "MAX"


def read_string(cpu, addr, max_len=64):
    data = cpu.read_memory(addr, max_len)
    null = data.find(0)
    if null >= 0:
        data = data[:null]
    return bytes(data)


if __name__ == "__main__":
    print("=" * 60)
    print("GREP REGCOMP TRACE — Follow the Pattern")
    print("=" * 60)

    # === Step 1: Break at grep's regex compile call site ===
    print(f"\n--- Step 1: Break at grep compile call 0x{GREP_COMPILE_CALL:08X} ---")
    cpu, handler = setup()
    cpu.set_breakpoint(0, GREP_COMPILE_CALL)
    cycles, stop = run_to_bp(cpu, handler)
    print(f"  Stopped: {stop} at cycle {cycles}, PC=0x{cpu.pc:X}")

    if stop == "BP":
        x0 = cpu.get_register(0) & 0xFFFFFFFFFFFFFFFF
        x1 = cpu.get_register(1) & 0xFFFFFFFFFFFFFFFF
        x2 = cpu.get_register(2) & 0xFFFFFFFFFFFFFFFF
        x30 = cpu.get_register(30) & 0xFFFFFFFFFFFFFFFF
        print(f"  x0=0x{x0:X}, x1=0x{x1:X}, x2=0x{x2:X}, LR=0x{x30:X}")

        # x0 = regex_t*, x1 = pattern, w2 = cflags
        try:
            s1 = read_string(cpu, x1)
            print(f"  pattern = {s1}")
        except:
            print(f"  pattern = <unreadable>")

        print(f"  wrapper target = 0x{REGCOMP_WRAPPER:08X}")
        print(f"  regcomp target = 0x{REGCOMP_ENTRY:08X}")

        # Read first few instructions at the real regcomp entry
        print(f"\n  First 8 instructions at 0x{REGCOMP_ENTRY:08X}:")
        for i in range(8):
            addr = REGCOMP_ENTRY + i * 4
            inst = struct.unpack('<I', cpu.read_memory(addr, 4))[0]
            op = (inst >> 24) & 0xFF
            print(f"    0x{addr:08X}: 0x{inst:08X} (op=0x{op:02X})")

    # === Step 2: Break at the actual regcomp entry ===
    print(f"\n--- Step 2: Break at regcomp entry 0x{REGCOMP_ENTRY:08X} ---")
    cpu2, handler2 = setup()
    cpu2.set_breakpoint(0, REGCOMP_ENTRY)
    cycles2, stop2 = run_to_bp(cpu2, handler2)
    print(f"  Stopped: {stop2} at cycle {cycles2}, PC=0x{cpu2.pc:X}")

    if stop2 == "BP":
        x0 = cpu2.get_register(0) & 0xFFFFFFFFFFFFFFFF
        x1 = cpu2.get_register(1) & 0xFFFFFFFFFFFFFFFF
        x2 = cpu2.get_register(2) & 0xFFFFFFFFFFFFFFFF
        print(f"  x0=0x{x0:X}, x1=0x{x1:X}")
        print(f"  x2=0x{x2:X}")
        try:
            print(f"  pattern = {read_string(cpu2, x1)}")
        except:
            print(f"  pattern = <unreadable>")

        for i in range(8):
            addr = REGCOMP_ENTRY + i * 4
            inst = struct.unpack('<I', cpu2.read_memory(addr, 4))[0]
            op = (inst >> 24) & 0xFF
            print(f"    0x{addr:08X}: 0x{inst:08X} (op=0x{op:02X})")

    # === Step 3: Deep trace from regcomp entry to NFA build ===
    print(f"\n--- Step 3: Full trace from regcomp (0x{REGCOMP_ENTRY:08X}) to NFA build ---")
    cpu3, handler3 = setup()

    cpu3.set_breakpoint(0, REGCOMP_ENTRY)
    cycles3, stop3 = run_to_bp(cpu3, handler3)
    print(f"  regcomp entry: {stop3} at cycle {cycles3}")

    if stop3 == "BP":
        x0 = cpu3.get_register(0) & 0xFFFFFFFFFFFFFFFF
        x1 = cpu3.get_register(1) & 0xFFFFFFFFFFFFFFFF
        x2 = cpu3.get_register(2) & 0xFFFFFFFFFFFFFFFF
        print(f"  Args: x0=0x{x0:X} (regex_t*), x1=0x{x1:X} (pattern), x2=0x{x2:X} (cflags)")
        pattern = read_string(cpu3, x1)
        print(f"  Pattern: {pattern}")

        # Now enable trace and continue to the NFA builder
        cpu3.clear_breakpoints()
        cpu3.enable_trace()

        # Run to the NFA code_min load at 0x00425FA8
        cpu3.set_breakpoint(0, NFA_CODE_MIN_LOAD)
        cycles3b, stop3b = run_to_bp(cpu3, handler3, max_cycles=20_000)
        print(f"\n  NFA code_min load: {stop3b} at cycle {cycles3} + {cycles3b}")

        if stop3b == "BP":
            trace = cpu3.read_trace()
            print(f"  Trace: {len(trace)} entries")

            # Find STR W instructions (where wchar_t values would be stored)
            print("\n  ALL STR/STP instructions in trace (first 200):")
            store_count = 0
            for i, (pc, inst, x0t, x1t, x2t, x3t, flags, sp) in enumerate(trace):
                op = (inst >> 24) & 0xFF
                is_store = False
                desc = ""

                # STR W unsigned offset
                if (inst & 0xFFC00000) == 0xB9000000:
                    rd = inst & 0x1F
                    rn = (inst >> 5) & 0x1F
                    imm12 = (inst >> 10) & 0xFFF
                    desc = f"STR W{rd}, [X{rn}, #{imm12*4}]"
                    is_store = True
                # STR X unsigned offset
                elif (inst & 0xFFC00000) == 0xF9000000:
                    rd = inst & 0x1F
                    rn = (inst >> 5) & 0x1F
                    imm12 = (inst >> 10) & 0xFFF
                    desc = f"STR X{rd}, [X{rn}, #{imm12*8}]"
                    is_store = True
                # STP 64-bit
                elif (inst & 0xFFC00000) == 0xA9000000:
                    rd = inst & 0x1F
                    rt2 = (inst >> 10) & 0x1F
                    rn = (inst >> 5) & 0x1F
                    imm7 = (inst >> 15) & 0x7F
                    if imm7 >= 64: imm7 -= 128
                    desc = f"STP X{rd}, X{rt2}, [X{rn}, #{imm7*8}]"
                    is_store = True
                # STP 64-bit pre-index
                elif (inst & 0xFFC00000) == 0xA9800000:
                    rd = inst & 0x1F
                    rt2 = (inst >> 10) & 0x1F
                    rn = (inst >> 5) & 0x1F
                    imm7 = (inst >> 15) & 0x7F
                    if imm7 >= 64: imm7 -= 128
                    desc = f"STP X{rd}, X{rt2}, [X{rn}, #{imm7*8}]!"
                    is_store = True
                # STRB
                elif (inst & 0xFFC00000) == 0x39000000:
                    rd = inst & 0x1F
                    rn = (inst >> 5) & 0x1F
                    imm12 = (inst >> 10) & 0xFFF
                    desc = f"STRB W{rd}, [X{rn}, #{imm12}]"
                    is_store = True
                # 0xB8 complex (incl. post-index, pre-index)
                elif op == 0xB8:
                    opc = (inst >> 22) & 0x3
                    opt = (inst >> 10) & 0x3
                    rd = inst & 0x1F
                    rn = (inst >> 5) & 0x1F
                    if opc == 0:  # store
                        imm9 = (inst >> 12) & 0x1FF
                        if imm9 >= 256: imm9 -= 512
                        modes = {0: "STUR", 1: "post", 2: "reg", 3: "pre"}
                        desc = f"STR W{rd}, [X{rn}] ({modes.get(opt, '?')})"
                        is_store = True

                if is_store:
                    store_count += 1
                    if store_count <= 200:
                        regs = {0: x0t, 1: x1t, 2: x2t, 3: x3t}
                        print(f"    [{i:4d}] PC=0x{pc:08X}: {desc:45s} x0=0x{x0t&0xFFFFFFFFFFFFFFFF:X}")

            print(f"\n  Total stores: {store_count}")

            # Also look for BL calls within the trace
            print("\n  BL calls within trace:")
            for i, (pc, inst, x0t, x1t, x2t, x3t, flags, sp) in enumerate(trace):
                if (inst & 0xFC000000) == 0x94000000:
                    imm26 = inst & 0x3FFFFFF
                    if imm26 >= (1<<25): imm26 -= (1<<26)
                    target = pc + imm26 * 4
                    print(f"    [{i:4d}] PC=0x{pc:08X}: BL 0x{target:08X} x0=0x{x0t&0xFFFFFFFFFFFFFFFF:X} x1=0x{x1t&0xFFFFFFFFFFFFFFFF:X}")

    print("\n--- COMPLETE ---")
