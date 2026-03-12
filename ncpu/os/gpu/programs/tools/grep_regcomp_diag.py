#!/usr/bin/env python3
"""Diagnose why regcomp produces NFA transitions with all-zero code_min/code_max.

Strategy:
1. Set a watchpoint on 0x876068 (NFA state pointer — known to be written correctly)
2. Enable tracing to capture instructions around the write
3. When watchpoint fires, examine trace for surrounding STR W instructions
4. Also set watchpoint on 0x876060 (code_min/code_max — should fire if written non-zero)
"""
import sys, struct
sys.path.insert(0, "/Users/bobbyprice/projects/nCPU")

from ncpu.os.gpu.elf_loader import load_and_run_elf, load_elf_into_memory, parse_elf, make_busybox_syscall_handler
from ncpu.os.gpu.filesystem import GPUFilesystem
from kernels.mlx.gpu_cpu import GPUKernelCPU
from pathlib import Path

ELF_PATH = "demos/busybox.elf"
REGCOMP_ENTRY = 0x00428C4C


def read_mem32(cpu, addr):
    """Read 32-bit value from GPU memory."""
    b = cpu.read_memory(addr, 4)
    return struct.unpack('<I', b)[0]


def read_mem64(cpu, addr):
    """Read 64-bit value from GPU memory."""
    b = cpu.read_memory(addr, 8)
    return struct.unpack('<Q', b)[0]


def decode_instruction(inst):
    """Basic ARM64 instruction decode."""
    op_byte = (inst >> 24) & 0xFF
    rd = inst & 0x1F
    rn = (inst >> 5) & 0x1F
    rt2 = (inst >> 10) & 0x1F
    imm7 = (inst >> 15) & 0x7F
    imm12 = (inst >> 10) & 0xFFF
    rm = (inst >> 16) & 0x1F

    # STP 32-bit signed offset
    if (inst & 0xFFC00000) == 0x29000000:
        imm7_s = imm7 if imm7 < 64 else imm7 - 128
        return f"STP W{rd}, W{rt2}, [X{rn}, #{imm7_s * 4}]"
    # STP 64-bit signed offset
    if (inst & 0xFFC00000) == 0xA9000000:
        imm7_s = imm7 if imm7 < 64 else imm7 - 128
        return f"STP X{rd}, X{rt2}, [X{rn}, #{imm7_s * 8}]"
    # STR 32-bit unsigned offset
    if (inst & 0xFFC00000) == 0xB9000000:
        return f"STR W{rd}, [X{rn}, #{imm12 * 4}]"
    # STR 64-bit unsigned offset
    if (inst & 0xFFC00000) == 0xF9000000:
        return f"STR X{rd}, [X{rn}, #{imm12 * 8}]"
    # STUR 32-bit
    if (inst & 0xFFE00C00) == 0xB8000000:
        imm9 = (inst >> 12) & 0x1FF
        if imm9 >= 256: imm9 -= 512
        return f"STUR W{rd}, [X{rn}, #{imm9}]"
    # STUR 64-bit
    if (inst & 0xFFE00C00) == 0xF8000000:
        imm9 = (inst >> 12) & 0x1FF
        if imm9 >= 256: imm9 -= 512
        return f"STUR X{rd}, [X{rn}, #{imm9}]"
    # STR register offset (32-bit)
    if (inst & 0xFFE00C00) == 0xB8200800:
        return f"STR W{rd}, [X{rn}, X{rm}]"
    # STR register offset (64-bit)
    if (inst & 0xFFE00C00) == 0xF8200800:
        return f"STR X{rd}, [X{rn}, X{rm}]"
    # LDR 32-bit unsigned
    if (inst & 0xFFC00000) == 0xB9400000:
        return f"LDR W{rd}, [X{rn}, #{imm12 * 4}]"
    # LDR 64-bit unsigned
    if (inst & 0xFFC00000) == 0xF9400000:
        return f"LDR X{rd}, [X{rn}, #{imm12 * 8}]"
    # MOV/ADD immediate
    if op_byte == 0x91:
        shift = (inst >> 22) & 1
        val = imm12 << (12 if shift else 0)
        if val == 0:
            return f"MOV X{rd}, X{rn}"
        return f"ADD X{rd}, X{rn}, #{val}"
    # BL
    if op_byte & 0xFC == 0x94:
        imm26 = inst & 0x3FFFFFF
        if imm26 >= (1 << 25): imm26 -= (1 << 26)
        return f"BL #{imm26 * 4:+d}"
    # B
    if op_byte & 0xFC == 0x14:
        imm26 = inst & 0x3FFFFFF
        if imm26 >= (1 << 25): imm26 -= (1 << 26)
        return f"B #{imm26 * 4:+d}"
    # RET
    if inst == 0xD65F03C0:
        return "RET"

    return f"??? (0x{inst:08X}, op=0x{op_byte:02X})"


def make_fs():
    """Create filesystem with /etc/passwd."""
    fs = GPUFilesystem()
    fs.write_file("/etc/passwd", b"root:x:0:0:root:/root:/bin/sh\n")
    return fs


def run_with_watchpoint(wp_addr, label, max_cyc=500_000):
    """Run grep with a watchpoint and tracing, return results."""
    cpu = GPUKernelCPU(quiet=True)

    # Set up filesystem
    fs = make_fs()

    # Load ELF
    entry = load_elf_into_memory(
        cpu, ELF_PATH,
        argv=["grep", "root", "/etc/passwd"],
        quiet=True
    )
    cpu.set_pc(entry)

    # Set up heap
    elf_data = Path(ELF_PATH).read_bytes()
    elf_info = parse_elf(elf_data)
    max_seg_end = max(s.vaddr + s.memsz for s in elf_info.segments)
    heap_base = (max_seg_end + 0xFFFF) & ~0xFFFF
    SVC_BUF_BASE = 0x3F0000
    cpu.write_memory(SVC_BUF_BASE + 8, struct.pack('<Q', heap_base))

    handler = make_busybox_syscall_handler(filesystem=fs, heap_base=heap_base)

    # Enable trace and set watchpoint
    cpu.enable_trace()
    cpu.set_watchpoint(0, wp_addr)

    print(f"\n{'='*60}")
    print(f"Watchpoint on 0x{wp_addr:X} ({label})")
    print(f"{'='*60}")

    # Run in batches to check watchpoint
    from ncpu.os.gpu.runner import run
    total_cycles = 0
    stop = None

    while total_cycles < max_cyc:
        result = run(cpu, handler, max_cycles=50_000, quiet=True)
        total_cycles += result['total_cycles']
        stop = result['stop_reason']

        wp_info = cpu.read_watchpoint_info()
        if wp_info:
            idx, addr, old_val, new_val = wp_info
            print(f"\n*** WATCHPOINT HIT at cycle ~{total_cycles} ***")
            print(f"  Address: 0x{addr:X}")
            print(f"  Old value: 0x{old_val:016X}")
            print(f"  New value: 0x{new_val:016X}")
            print(f"  (as 2x32-bit: lo=0x{new_val & 0xFFFFFFFF:08X}, hi=0x{(new_val >> 32) & 0xFFFFFFFF:08X})")

            # Read trace to find the writing instruction
            trace = cpu.read_trace()
            print(f"\n  Trace has {len(trace)} entries. Last 20:")
            for i, (pc, inst, x0, x1, x2, x3, flags, sp) in enumerate(trace[-20:]):
                decoded = decode_instruction(inst)
                nzcv = f"{'N' if flags & 8 else '.'}{'Z' if flags & 4 else '.'}{'C' if flags & 2 else '.'}{'V' if flags & 1 else '.'}"
                print(f"    [{len(trace)-20+i:4d}] 0x{pc:08X}: {decoded:40s} x0=0x{x0 & 0xFFFFFFFFFFFFFFFF:X} x1=0x{x1 & 0xFFFFFFFFFFFFFFFF:X} x2=0x{x2 & 0xFFFFFFFFFFFFFFFF:X} x3=0x{x3 & 0xFFFFFFFFFFFFFFFF:X} {nzcv}")

            # Continue running to see if there are more hits
            cpu.clear_watchpoints()  # prevent immediate re-trigger
            break

        if stop in ('HALT', 'SYSCALL'):
            print(f"\n  Program exited ({stop}) after {total_cycles} cycles. Watchpoint never fired.")
            break

    return cpu, total_cycles


def scan_nfa_transitions(cpu, heap_start=0x870000, heap_end=0x880000):
    """Scan heap for NFA transition-like structures."""
    print(f"\n{'='*60}")
    print("Scanning heap for NFA transition structs")
    print(f"{'='*60}")

    # Read heap
    heap_data = cpu.read_memory(heap_start, heap_end - heap_start)

    found = []
    for offset in range(0, len(heap_data) - 48, 8):
        # Look for pattern: 8 bytes (code_min+code_max) followed by valid heap pointer
        addr = heap_start + offset
        code_min = struct.unpack_from('<I', heap_data, offset)[0]
        code_max = struct.unpack_from('<I', heap_data, offset + 4)[0]
        state_ptr = struct.unpack_from('<Q', heap_data, offset + 8)[0]

        # Check if state_ptr is in heap range
        if heap_start <= state_ptr < heap_end:
            # Also check if state_ptr's state_ptr is also in range (chain)
            ptr_offset = state_ptr - heap_start
            if ptr_offset + 16 <= len(heap_data):
                next_ptr = struct.unpack_from('<Q', heap_data, ptr_offset + 8)[0]
                if heap_start <= next_ptr < heap_end or next_ptr == 0:
                    found.append((addr, code_min, code_max, state_ptr))

    print(f"Found {len(found)} potential transition structs:")
    for addr, cmin, cmax, state in found[:20]:
        if cmin == 0 and cmax == 0:
            print(f"  0x{addr:06X}: code_min=0, code_max=0, state→0x{state:X}  *** ZEROS ***")
        else:
            cmin_c = chr(cmin) if 32 <= cmin < 127 else f"\\x{cmin:02x}"
            cmax_c = chr(cmax) if 32 <= cmax < 127 else f"\\x{cmax:02x}"
            print(f"  0x{addr:06X}: code_min=0x{cmin:X}('{cmin_c}'), code_max=0x{cmax:X}('{cmax_c}'), state→0x{state:X}")

    return found


def run_no_watchpoint():
    """Run grep to completion and scan NFA data."""
    cpu = GPUKernelCPU(quiet=True)
    fs = make_fs()

    entry = load_elf_into_memory(
        cpu, ELF_PATH,
        argv=["grep", "root", "/etc/passwd"],
        quiet=True
    )
    cpu.set_pc(entry)

    elf_data = Path(ELF_PATH).read_bytes()
    elf_info = parse_elf(elf_data)
    max_seg_end = max(s.vaddr + s.memsz for s in elf_info.segments)
    heap_base = (max_seg_end + 0xFFFF) & ~0xFFFF
    SVC_BUF_BASE = 0x3F0000
    cpu.write_memory(SVC_BUF_BASE + 8, struct.pack('<Q', heap_base))

    handler = make_busybox_syscall_handler(filesystem=fs, heap_base=heap_base)

    from ncpu.os.gpu.runner import run
    result = run(cpu, handler, max_cycles=500_000, quiet=True)
    print(f"Grep completed: {result['total_cycles']} cycles, stop={result['stop_reason']}")

    # Get exit code
    x0 = cpu.get_register(0)
    print(f"Exit code (x0): {x0}")

    return cpu


def find_nfa_chain(cpu, heap_start=0x870000, heap_end=0x880000):
    """Find the actual NFA state chain (5 states for 'root' pattern)."""
    print(f"\n{'='*60}")
    print("Searching for NFA state chain in 0x876000-0x877000")
    print(f"{'='*60}")

    # Read the target range
    data = cpu.read_memory(0x876000, 0x2000)

    # Look for the chain pattern: 5 structs each 48+ bytes with state pointers
    # Each transition: code_min(4) + code_max(4) + state_ptr(8) + ...
    for base_offset in range(0, len(data) - 48*5, 8):
        addr = 0x876000 + base_offset
        cm1 = struct.unpack_from('<I', data, base_offset)[0]
        cx1 = struct.unpack_from('<I', data, base_offset + 4)[0]
        ptr1 = struct.unpack_from('<Q', data, base_offset + 8)[0]

        # Check if ptr1 points to another struct in this range
        if not (0x876000 <= ptr1 < 0x878000):
            continue

        # Check second struct
        off2 = ptr1 - 0x876000
        if off2 + 16 > len(data):
            continue
        ptr2 = struct.unpack_from('<Q', data, off2 + 8)[0]
        if not (0x876000 <= ptr2 < 0x878000):
            continue

        # Check third struct
        off3 = ptr2 - 0x876000
        if off3 + 16 > len(data):
            continue
        ptr3 = struct.unpack_from('<Q', data, off3 + 8)[0]
        if not (0x876000 <= ptr3 < 0x878000):
            continue

        # Found a chain of at least 3! Print it
        print(f"\n  Chain starting at 0x{addr:X}:")
        current = addr
        for step in range(8):  # Follow up to 8 links
            off = current - 0x876000
            if off < 0 or off + 48 > len(data):
                break
            cm = struct.unpack_from('<I', data, off)[0]
            cx = struct.unpack_from('<I', data, off + 4)[0]
            sp = struct.unpack_from('<Q', data, off + 8)[0]
            tags = struct.unpack_from('<Q', data, off + 16)[0]
            assrt = struct.unpack_from('<I', data, off + 24)[0]

            cm_s = f"'{chr(cm)}'" if 32 <= cm < 127 else f"0x{cm:X}"
            cx_s = f"'{chr(cx)}'" if 32 <= cx < 127 else f"0x{cx:X}"
            zero_flag = " *** ZEROS ***" if cm == 0 and cx == 0 else ""
            print(f"    [{step}] 0x{current:X}: code_min={cm_s}, code_max={cx_s}, "
                  f"state→0x{sp:X}, tags=0x{tags:X}, assertions={assrt}{zero_flag}")

            if sp == 0 or sp == current:
                break
            current = sp

        return addr  # Return first chain start

    print("  No NFA chain found in 0x876000-0x878000")
    return None


def dump_memory_region(cpu, addr, size, label):
    """Hex dump a memory region."""
    data = cpu.read_memory(addr, size)
    print(f"\n  {label} at 0x{addr:X} ({size} bytes):")
    for i in range(0, min(size, 256), 16):
        hex_bytes = ' '.join(f'{b:02X}' for b in data[i:i+16])
        ascii_chars = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data[i:i+16])
        print(f"    0x{addr+i:06X}: {hex_bytes:<48s}  {ascii_chars}")


def run_with_early_trace():
    """Break at regcomp, then trace from the real musl entry point."""
    cpu = GPUKernelCPU(quiet=True)
    fs = make_fs()

    entry = load_elf_into_memory(
        cpu, ELF_PATH,
        argv=["grep", "root", "/etc/passwd"],
        quiet=True
    )
    cpu.set_pc(entry)

    elf_data = Path(ELF_PATH).read_bytes()
    elf_info = parse_elf(elf_data)
    max_seg_end = max(s.vaddr + s.memsz for s in elf_info.segments)
    heap_base = (max_seg_end + 0xFFFF) & ~0xFFFF
    SVC_BUF_BASE = 0x3F0000
    cpu.write_memory(SVC_BUF_BASE + 8, struct.pack('<Q', heap_base))

    handler = make_busybox_syscall_handler(filesystem=fs, heap_base=heap_base)

    from kernels.mlx.rust_runner import StopReasonV2

    cpu.set_breakpoint(0, REGCOMP_ENTRY)
    total = 0
    while total < 100_000:
        result = cpu.execute(max_cycles=1000)
        total += result.cycles
        for fd, data in cpu.drain_svc_buffer():
            pass
        if result.stop_reason == StopReasonV2.BREAKPOINT:
            break
        if result.stop_reason == StopReasonV2.SYSCALL:
            ret = handler(cpu)
            if ret == False:
                print(f"\nFailed to reach regcomp entry, exited after {total} cycles")
                return cpu, []
            if ret != "exec":
                cpu.set_pc(cpu.pc + 4)
        elif result.stop_reason == StopReasonV2.HALT:
            print(f"\nFailed to reach regcomp entry, HALT after {total} cycles")
            return cpu, []
    else:
        print(f"\nFailed to reach regcomp entry within {total} cycles")
        return cpu, []

    print(f"\nReached regcomp entry 0x{REGCOMP_ENTRY:08X} at cycle {total}")
    cpu.clear_breakpoints()
    cpu.enable_trace()

    traced = 0
    while traced < 5000:
        result = cpu.execute(max_cycles=1000)
        traced += result.cycles
        for fd, data in cpu.drain_svc_buffer():
            pass
        if result.stop_reason == StopReasonV2.SYSCALL:
            ret = handler(cpu)
            if ret == False:
                break
            if ret != "exec":
                cpu.set_pc(cpu.pc + 4)
        elif result.stop_reason in (StopReasonV2.HALT, StopReasonV2.BREAKPOINT, StopReasonV2.WATCHPOINT):
            break

    print(f"Trace from regcomp entry: {traced} cycles, PC=0x{cpu.pc:X}")

    trace = cpu.read_trace()
    print(f"Trace entries: {len(trace)}")

    # Look for store instructions to heap region (0x870000+)
    heap_stores = []
    for i, (pc, inst, x0, x1, x2, x3, flags, sp) in enumerate(trace):
        decoded = decode_instruction(inst)
        if 'STR' in decoded or 'STP' in decoded or 'STUR' in decoded:
            heap_stores.append((i, pc, inst, decoded, x0, x1, x2, x3))

    print(f"\nStore instructions in trace: {len(heap_stores)}")
    print("First 30 stores:")
    for i, (idx, pc, inst, decoded, x0, x1, x2, x3) in enumerate(heap_stores[:30]):
        print(f"  [{idx:4d}] 0x{pc:08X}: {decoded:40s} x0=0x{x0 & 0xFFFFFFFFFFFFFFFF:X}")

    # Also look for what's in registers when store happens
    # We need to find stores that target the NFA transition addresses
    print("\n\nStores to 0x870xxx-0x877xxx range (NFA data):")
    for idx, pc, inst, decoded, x0, x1, x2, x3 in heap_stores:
        # Try to identify target address from registers
        # For STR/STP, the base register is rn (bits 9-5)
        rn = (inst >> 5) & 0x1F
        # We need the register value, but we only have x0-x3 in trace
        # Check if rn is 0-3
        reg_vals = {0: x0, 1: x1, 2: x2, 3: x3}
        if rn in reg_vals:
            base = reg_vals[rn] & 0xFFFFFFFFFFFFFFFF
            if 0x870000 <= base < 0x878000:
                print(f"  [{idx:4d}] 0x{pc:08X}: {decoded:40s} base(x{rn})=0x{base:X}")

    return cpu, trace


if __name__ == "__main__":
    print("=" * 60)
    print("GREP REGEX DIAGNOSTIC: NFA code_min/code_max Investigation")
    print("=" * 60)

    # Step 1: Run grep to completion and find NFA chain
    print("\n--- STEP 1: Run grep, find NFA chain ---")
    cpu = run_no_watchpoint()
    chain_addr = find_nfa_chain(cpu)

    if chain_addr:
        # Dump raw bytes around the chain
        dump_memory_region(cpu, chain_addr - 16, 320, "NFA transition data")

        # Step 2: Set watchpoint on the FIRST state pointer in the chain
        ptr_addr = chain_addr + 8
        print(f"\n--- STEP 2: Watchpoint on first chain state ptr at 0x{ptr_addr:X} ---")
        cpu2, cycles = run_with_watchpoint(ptr_addr, f"Chain state ptr at 0x{ptr_addr:X}")

        # Step 3: Watchpoint on code_min/code_max of first chain entry
        print(f"\n--- STEP 3: Watchpoint on chain code_min/code_max at 0x{chain_addr:X} ---")
        cpu3, cycles = run_with_watchpoint(chain_addr, f"Chain code_min/max at 0x{chain_addr:X}")

    # Step 4: Capture regcomp with early tracing
    print(f"\n--- STEP 4: Early trace to capture regcomp ---")
    cpu4, trace = run_with_early_trace()

    print("\n--- DIAGNOSTIC COMPLETE ---")
