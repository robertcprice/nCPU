#!/usr/bin/env python3
"""Diagnose WHY the AST literal nodes have code_min=code_max=0.

From diag2: The NFA builder correctly copies code_min/code_max from the AST
literal nodes at X23=0x8718A8, but those source values are already 0.
The parser failed to populate them.

Strategy:
1. Set breakpoint at 0x00425FA8 (LDR W0, [X23, #4] — loads code_min from literal)
2. When it fires, dump the literal node at X23 and surrounding memory
3. Also trace the pattern string in memory
4. Find WHERE the parser stores code_min/code_max (or fails to)
"""
import sys, struct
sys.path.insert(0, "/Users/bobbyprice/projects/nCPU")

from ncpu.os.gpu.elf_loader import load_elf_into_memory, parse_elf, make_busybox_syscall_handler
from ncpu.os.gpu.filesystem import GPUFilesystem
from kernels.mlx.gpu_cpu import GPUKernelCPU
from kernels.mlx.rust_runner import StopReasonV2
from pathlib import Path

ELF_PATH = "demos/busybox.elf"


def make_fs():
    fs = GPUFilesystem()
    fs.write_file("/etc/passwd", b"root:x:0:0:root:/root:/bin/sh\n")
    return fs


def setup():
    cpu = GPUKernelCPU(quiet=True)
    fs = make_fs()
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


def run_to_stop(cpu, handler, max_cycles=100_000):
    """Execute until breakpoint/watchpoint/halt/exit."""
    total = 0
    while total < max_cycles:
        result = cpu.execute(max_cycles=1000)
        total += result.cycles
        for fd, data in cpu.drain_svc_buffer():
            pass
        if result.stop_reason == StopReasonV2.BREAKPOINT:
            return total, "BREAKPOINT"
        elif result.stop_reason == StopReasonV2.WATCHPOINT:
            return total, "WATCHPOINT"
        elif result.stop_reason == StopReasonV2.SYSCALL:
            ret = handler(cpu)
            if ret == False:
                return total, "EXIT"
            elif ret != "exec":
                cpu.set_pc(cpu.pc + 4)
        elif result.stop_reason == StopReasonV2.HALT:
            return total, "HALT"
    return total, "MAX_CYCLES"


def hex_dump(cpu, addr, size, label=""):
    data = cpu.read_memory(addr, size)
    if label:
        print(f"  {label}:")
    for i in range(0, min(size, 256), 16):
        hex_bytes = ' '.join(f'{data[j]:02X}' for j in range(i, min(i+16, size)))
        ascii_chars = ''.join(chr(data[j]) if 32 <= data[j] < 127 else '.' for j in range(i, min(i+16, size)))
        print(f"    0x{addr+i:06X}: {hex_bytes:<48s} {ascii_chars}")
    return data


def read32(cpu, addr):
    return struct.unpack('<I', cpu.read_memory(addr, 4))[0]

def read64(cpu, addr):
    return struct.unpack('<Q', cpu.read_memory(addr, 8))[0]


if __name__ == "__main__":
    print("=" * 60)
    print("GREP REGEX DIAGNOSTIC v3 — AST Literal Node Analysis")
    print("=" * 60)

    # === TEST A: Breakpoint at NFA builder (0x00425FA8) ===
    # This is where code_min is loaded from the literal node
    print("\n--- TEST A: Breakpoint at NFA code_min load (0x00425FA8) ---")
    cpu, handler = setup()
    cpu.enable_trace()
    cpu.set_breakpoint(0, 0x00425FA8)

    cycles, stop = run_to_stop(cpu, handler)
    print(f"  Stopped: {stop} at cycle {cycles}, PC=0x{cpu.pc:X}")

    if stop == "BREAKPOINT":
        x23 = cpu.get_register(23) & 0xFFFFFFFFFFFFFFFF
        x19 = cpu.get_register(19) & 0xFFFFFFFFFFFFFFFF
        print(f"  X23 (literal node ptr) = 0x{x23:X}")
        print(f"  X19 (transition ptr)   = 0x{x19:X}")

        # Dump the literal node at X23
        print(f"\n  Literal node at 0x{x23:X}:")
        data = hex_dump(cpu, x23, 48, "tre_literal_t struct")

        # Parse assuming int position (4B), int code_min (4B), int code_max (4B)
        pos_4 = struct.unpack_from('<i', data, 0)[0]
        cm_4 = struct.unpack_from('<i', data, 4)[0]
        cx_8 = struct.unpack_from('<i', data, 8)[0]
        print(f"\n  If position=int(4B): pos={pos_4}, code_min={cm_4} ('{chr(cm_4) if 32<=cm_4<127 else '?'}'), code_max={cx_8} ('{chr(cx_8) if 32<=cx_8<127 else '?'}')")

        # Parse assuming long position (8B), int code_min (4B), int code_max (4B)
        pos_8 = struct.unpack_from('<q', data, 0)[0]
        cm_8 = struct.unpack_from('<i', data, 8)[0]
        cx_12 = struct.unpack_from('<i', data, 12)[0]
        print(f"  If position=long(8B): pos={pos_8}, code_min={cm_8} ('{chr(cm_8) if 32<=cm_8<127 else '?'}'), code_max={cx_12} ('{chr(cx_12) if 32<=cx_12<127 else '?'}')")

        # Also dump all 4 literal nodes (for "root" — 4 chars)
        # The literal nodes are allocated from a memory pool. Check nearby nodes.
        print(f"\n  Scanning for literal nodes near X23:")
        # Check if X22 (0x871AC0) is the transition array base
        x22 = cpu.get_register(22) & 0xFFFFFFFFFFFFFFFF
        x24 = cpu.get_register(24) & 0xFFFFFFFFFFFFFFFF
        print(f"  X22 = 0x{x22:X}, X24 = 0x{x24:X}")

        # Dump a wider region around the literal node
        hex_dump(cpu, x23 - 48, 256, f"Memory around literal node at 0x{x23:X}")

        # Check the pattern string in memory
        # BusyBox passes the pattern via argv[1]. Find it on stack.
        print(f"\n  Looking for pattern 'root' in memory...")
        # The pattern was passed as argv[1] to grep. It's on the stack.
        # Search for it in high memory (stack region)
        sp = cpu.get_register(31) & 0xFFFFFFFFFFFFFFFF
        print(f"  SP = 0x{sp:X}")

        # Also search heap for the pattern
        for addr in range(0x8700, 0x8780):
            try:
                d = cpu.read_memory(addr * 0x100, 256)
                for i in range(len(d) - 4):
                    if d[i:i+4] == b'root':
                        loc = addr * 0x100 + i
                        print(f"  Found 'root' at 0x{loc:X}")
                        hex_dump(cpu, loc - 8, 32, f"Pattern at 0x{loc:X}")
            except:
                pass

        # Now also search for where mbrtowc or btowc might have been called
        # by looking at the trace for calls to conversion functions
        trace = cpu.read_trace()
        print(f"\n  Trace: {len(trace)} entries")

        # Look for stores to X23 range (where literal node is populated)
        print(f"\n  Stores to literal node range 0x{x23:X}-0x{x23+20:X}:")
        for i, (pc, inst, x0, x1, x2, x3, flags, sp_val) in enumerate(trace):
            op = (inst >> 24) & 0xFF
            rn = (inst >> 5) & 0x1F
            rd = inst & 0x1F

            # Check for STR instructions
            is_str = False
            target_addr = None

            # STR W immediate unsigned offset
            if (inst & 0xFFC00000) == 0xB9000000:
                imm12 = (inst >> 10) & 0xFFF
                regs = {0: x0, 1: x1, 2: x2, 3: x3}
                if rn in regs:
                    target_addr = (regs[rn] & 0xFFFFFFFFFFFFFFFF) + imm12 * 4
                    is_str = True

            # STR X immediate unsigned offset
            elif (inst & 0xFFC00000) == 0xF9000000:
                imm12 = (inst >> 10) & 0xFFF
                regs = {0: x0, 1: x1, 2: x2, 3: x3}
                if rn in regs:
                    target_addr = (regs[rn] & 0xFFFFFFFFFFFFFFFF) + imm12 * 8
                    is_str = True

            if is_str and target_addr and x23 <= target_addr < x23 + 20:
                val_regs = {0: x0, 1: x1, 2: x2, 3: x3}
                val = val_regs.get(rd, "?")
                print(f"    [{i:4d}] PC=0x{pc:08X}: inst=0x{inst:08X} → target=0x{target_addr:X}, W{rd}={val}")

    # === TEST B: What's at address 0x8718A8 BEFORE the NFA builder runs? ===
    # Run to a breakpoint BEFORE the NFA builder, then dump the literal
    # The NFA builder function starts around 0x00425F70.
    # Let's break at the function entry.
    print("\n--- TEST B: Break at NFA builder entry (0x00425F70) ---")
    cpu2, handler2 = setup()
    cpu2.enable_trace()
    cpu2.set_breakpoint(0, 0x00425F70)  # tre_ast_to_tnfa entry

    cycles2, stop2 = run_to_stop(cpu2, handler2)
    print(f"  Stopped: {stop2} at cycle {cycles2}, PC=0x{cpu2.pc:X}")

    if stop2 == "BREAKPOINT":
        # X0-X3 should have function arguments
        for r in range(4):
            v = cpu2.get_register(r) & 0xFFFFFFFFFFFFFFFF
            print(f"  x{r} = 0x{v:X}")

        # Dump the literal node address
        hex_dump(cpu2, 0x8718A8, 48, "Literal node at 0x8718A8 (before NFA build)")

        # Also check what's at 0x8718A8 - 48 to 0x8718A8 + 96
        hex_dump(cpu2, 0x871860, 192, "Literal pool area")

    # === TEST C: Trace the pattern conversion ===
    # In musl regcomp, the byte pattern is converted to wchar_t.
    # Let's find the wchar_t pattern in memory.
    print("\n--- TEST C: Find wchar_t pattern after regcomp ---")
    cpu3, handler3 = setup()

    # Run regcomp to completion (about 15000 cycles for NFA build)
    # Break at NFA builder, then continue to completion
    cpu3.set_breakpoint(0, 0x00425FA8)  # code_min load
    cycles3, stop3 = run_to_stop(cpu3, handler3, max_cycles=20_000)
    print(f"  Stopped: {stop3} at cycle {cycles3}")

    if stop3 == "BREAKPOINT":
        x23 = cpu3.get_register(23) & 0xFFFFFFFFFFFFFFFF
        # Read literal node
        data = cpu3.read_memory(x23, 20)
        pos = struct.unpack_from('<i', data, 0)[0]
        cm = struct.unpack_from('<i', data, 4)[0]
        cx = struct.unpack_from('<i', data, 8)[0]
        tags = struct.unpack_from('<i', data, 12)[0]
        assrt = struct.unpack_from('<i', data, 16)[0]
        print(f"  Literal at 0x{x23:X}: position={pos}, code_min={cm} (0x{cm&0xFFFFFFFF:X}), code_max={cx} (0x{cx&0xFFFFFFFF:X}), tags={tags}, assertions={assrt}")

        # Search for wchar_t 'r' (0x72) in heap
        print(f"\n  Searching for wchar_t 'r' (0x00000072) in heap 0x470000-0x877000...")
        found_wchar = []
        for block_start in range(0x470000, 0x877000, 0x1000):
            try:
                block = cpu3.read_memory(block_start, 0x1000)
                for i in range(0, len(block) - 4, 4):
                    val = struct.unpack_from('<I', block, i)[0]
                    if val == 0x72:  # 'r'
                        addr = block_start + i
                        # Check if next values are 'o', 'o', 't'
                        if i + 16 <= len(block):
                            vals = [struct.unpack_from('<I', block, i+j*4)[0] for j in range(4)]
                            if vals == [0x72, 0x6F, 0x6F, 0x74]:
                                print(f"    FOUND wchar_t 'root' at 0x{addr:X}!")
                                hex_dump(cpu3, addr - 8, 32, "wchar pattern")
                                found_wchar.append(addr)
                            elif vals[0] == 0x72:
                                found_wchar.append(addr)
            except:
                pass

        if not found_wchar:
            print("    No wchar_t 'root' found in heap!")
            print("    Searching for byte pattern 'root' in stack...")
            sp = cpu3.get_register(31) & 0xFFFFFFFFFFFFFFFF
            stack_data = cpu3.read_memory(sp, 0x1000)
            for i in range(len(stack_data) - 4):
                if stack_data[i:i+4] == b'root':
                    addr = sp + i
                    print(f"    Found byte 'root' at 0x{addr:X}")
                    hex_dump(cpu3, addr - 8, 32, "Pattern on stack")

    print("\n--- DIAGNOSTIC v3 COMPLETE ---")
